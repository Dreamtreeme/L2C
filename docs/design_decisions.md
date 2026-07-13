# 기술 및 설계 결정

이 문서는 L2C 프로젝트에서 채택한 주요 구조와 트레이드오프를 기록합니다.

## 1. Classic DOM 수집과 Vision Agent 병행

Classic 경로는 Playwright DOM 기반 수집입니다. 빠르고 정확하지만 사이트별 셀렉터와 예외 처리가 필요합니다.

Vision Agent 경로는 화면 캡처, YOLOv8, PaddleOCR, LLM, 물리 입력 도구를 조합합니다. DOM 구조를 직접 읽지 않으므로 사이트 구조 변경에는 상대적으로 덜 민감하지만, OCR과 LLM 추론 비용이 추가됩니다.

두 경로를 모두 유지하는 이유는 목적이 다르기 때문입니다. Classic은 안정된 사이트의 고속 수집에 적합하고, Vision Agent는 새 사이트 개척과 UI 탐색 가능성 검증에 적합합니다.

## 2. Set-of-Marks 방식 채택

LLM에게 절대 좌표를 직접 예측하게 하지 않고, 로컬 비전 엔진이 화면 요소에 숫자 마커를 붙입니다. LLM은 `marker_id`를 선택하고, 클릭 좌표 계산은 도구가 처리합니다.

이 구조는 DPI, 창 위치, 브라우저 크기 변화에서 발생하는 클릭 오차를 줄입니다. 또한 reasoning prompt에는 이미지와 함께 현재 마커 텍스트 목록을 넣을 수 있어, LLM이 화면을 더 좁은 후보 집합으로 판단할 수 있습니다.

## 3. OCR 엔진 기준

현재 SoM 텍스트 감지는 GPU PaddleOCR을 사용합니다. 입력 이미지는 기본적으로 긴 변 1152px 이하로 줄인 뒤 OCR에 넣고, 감지 좌표는 원본 스크린샷 기준으로 복원합니다. 실제 E2E 스크린샷 재생에서 1280px 입력의 밀집 화면은 15초 이상 걸렸지만, 1152px 입력은 같은 worker 65회 처리에서 p95 0.679초를 유지했습니다.

PaddleOCR은 별도 subprocess를 작업 종료까지 재사용합니다. 요청 수 기준 재시작은 입력 크기에서 발생한 지연을 worker 누적 상태 문제로 오인한 조치였으므로 제거했습니다. 요청 timeout과 1회 재시도는 성능 최적화가 아니라 predictor crash나 실제 hang에서 상위 작업이 무기한 멈추지 않도록 하는 복구 경계입니다.

## 4. Lazy Initialization

`agent.graph.nodes` import 시점에는 브라우저 캡처, YOLOv8, OCR, PyAutoGUI 같은 무거운 의존성을 초기화하지 않습니다. 실제 비전 수집 도구가 호출될 때만 perception/action 객체를 생성합니다.

이렇게 분리하면 SQLite QA 서버나 테스트가 단순 import만으로 GUI/모델 환경에 묶이지 않습니다.

## 5. VLM Caption Bypass

`SKIP_VLM_CAPTION=true` 설정에서는 마커별 설명을 VLM으로 다시 캡셔닝하지 않고, 로컬 OCR 텍스트와 아이콘 타입만 reasoning context에 전달합니다.

이 경로는 perception 비용과 API 호출 수를 줄이는 대신, 텍스트가 없는 아이콘의 의미 추론은 reasoning 단계의 이미지 입력에 더 의존합니다.

## 6. Reflex Recipe 방향

Reflex Recipe는 LLM 판단을 코드로 강제 승격하는 구조가 아니라, 자율 탐색 실행 기록을 후보로 저장하고 Critic 피드백 루프가 재사용 가능성을 판단하는 구조로 둡니다.

고정 행동은 빠르게 재사용하되, 목표나 화면이 달라져 판단이 필요한 구간은 LLM에게 다시 넘기는 것이 현재 방향입니다.

## 7. 로컬 단일 작업자 직렬화

Realtime/Vision은 현재 화면과 물리 입력 장치를 공유하므로 병렬 worker를 실행하지 않습니다. `worker_execution_session()`이 수집, 검토 재시도, 브라우저 정리 전체를 프로세스 전역 잠금으로 직렬화합니다.

Perception, ActionTools, PaddleOCR subprocess와 브라우저 창은 요청 간 재사용합니다. 단, 각 작업은 새 LangGraph state로 시작하고 브라우저는 사이트 공식 시작 주소로 다시 이동합니다. `VISION_CLOSE_BROWSER_AFTER_RUN=1`을 명시한 경우에만 작업 종료 후 창을 닫습니다. 이 구분으로 모델과 브라우저 초기화 비용은 줄이면서 서로 다른 요청의 목표, 마커, 카드 큐가 섞이는 상태 오염을 막습니다.

## 8. 애플리케이션과 그래프 런타임 분리

사용자 질의의 지휘 책임은 `ChatService`, 작업자 실행 생명주기는 `worker_execution_service`, DB 적재는 `job_persistence_service`가 담당합니다. LangGraph의 `nodes.py`는 perception, reasoning, action에 집중합니다.

전환 검증, 상세 OCR 버퍼, 결과 카드 큐, Reflex 재생은 `agent/runtime/`의 독립 모듈로 분리했습니다. 이 정책들은 DOM selector가 아니라 화면 서명, OCR 마커, 좌표비율만 입력으로 받습니다.

## 9. LLM 선택과 물리 도구 계약 분리

LLM은 화면의 의미와 목표 대상을 판단하지만, 선택한 마커가 해당 물리 도구로 실행 가능한지는 실행기가 검증합니다. 예를 들어 `type_in_marker`가 텍스트 없는 작은 아이콘을 가리키면 클릭과 `Ctrl+A`를 보내지 않고 같은 화면에서 다른 마커를 다시 판단하게 합니다.

이 검증은 사이트명, 마커 번호, 화면 문구를 사용하지 않습니다. OCR 텍스트 여부, 내부 텍스트 포함 관계, bbox 형태처럼 화면에서 관찰한 물리적 affordance만 사용합니다. 따라서 사이트 의미를 코드로 복제하지 않으면서 닫기 버튼 같은 명백한 도구 계약 위반을 막습니다.

상세 공고 구조화에서도 증거 책임을 분리합니다. 최종 정제 모델의 1차 입력은 상세 페이지 OCR과 현재 URL입니다. 검색 목록에서 만든 카드 메타데이터는 OCR과 충돌할 수 있으므로 모델 입력에 섞지 않고, 모델이 회사명이나 직무명을 비운 경우에만 후처리 폴백으로 사용합니다.

## 10. 채용 사이트 공식 주소 선택

지원하는 5개 사이트의 공식 시작 주소는 `agent/sites/registry.json`의 `base_url` 한 곳에서 관리합니다. 사용자 요청의 사이트명, slug, 별칭, 도메인은 `get_official_site_url()`이 결정론적으로 해석하고, `open_browser(site=...)`는 해석된 공식 HTTPS 주소만 엽니다.

이 선택은 브라우저 주소를 LLM이 추론하게 하지 않으면서도 사이트별 탐색 로직을 `ActionTools` 안에 중복 하드코딩하지 않기 위한 것입니다. 주소를 연 이후의 검색과 화면 판단은 계속 OCR·화면·물리 입력 경로를 사용합니다.

## 11. 런타임 자원 수명

비싼 실행 자원과 작업 상태의 수명을 분리합니다.

- 프로세스 수명: OmniParser YOLO, PaddleOCR worker, Gemini 기본/구조화 클라이언트
- 브라우저 세션 수명: 전용 Chrome 창, 쿠키, 탭과 사이트 렌더링 상태
- 단일 작업 수명: LangGraph state, 목표, 현재 마커, 결과 카드 큐, 상세 OCR 버퍼

Gemini 클라이언트는 모델명, temperature, 구조화 출력 스키마별로 캐시하지만 실제 API 요청을 미리 보내지는 않습니다. 유료 더미 요청은 서버 측 warm 상태를 보장하지 못하고 비용만 추가하기 때문입니다. 로컬 비전 자원은 백엔드 import 시점이 아니라 첫 수집 요청에서 lazy 초기화하여 DB 질의와 UI 서버 기동이 GPU 환경에 묶이지 않게 합니다.

첫 수집 요청에서는 PaddleOCR worker 준비와 공식 사이트 열기를 병렬로 실행합니다. 이후 요청은 같은 worker와 브라우저를 재사용하되 공식 시작 주소로 다시 이동하고 새 그래프 상태를 생성합니다.
