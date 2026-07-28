---
title: "기술 및 설계 결정"
type: decision
area: architecture
status: active
updated: 2026-07-28
tags:
  - l2c
  - docs/architecture
---

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

현재 SoM 텍스트 감지는 분리된 `.venv-ocr`에서 PaddleOCR 3.7.0과 PaddlePaddle GPU 3.3.1의 `predict()` 계약을 사용합니다. 입력 이미지는 기본적으로 긴 변 1152px 이하로 줄인 뒤 OCR에 넣고, 감지 좌표는 원본 스크린샷 기준으로 복원합니다. 정확한 Python, CUDA, cuDNN 조합과 후보별 검증 결과는 [런타임 호환 기준](runtime_compatibility.md)을 단일 기준으로 사용합니다.

PaddleOCR은 별도 subprocess를 작업 종료까지 재사용합니다. 요청 수 기준 재시작은 구형 런타임에서 발생한 지연을 worker 누적 상태 문제로 오인한 조치였으므로 제거했습니다. 요청 timeout과 1회 재시도는 성능 최적화가 아니라 predictor crash나 실제 hang에서 상위 작업이 무기한 멈추지 않도록 하는 복구 경계입니다. 재시도까지 실패하면 별도의 일회성 OCR 프로세스를 다시 띄우지 않고 호출자에게 실패를 반환합니다. OCR worker와 호환성 검사기는 PaddlePaddle 3.3.1, PaddleOCR 3.7.0과 CUDA 연산 가능 여부를 시작 시 확인합니다.

## 4. Lazy Initialization

작업자 그래프 모듈 import 시점에는 브라우저 캡처, YOLOv8, OCR, PyAutoGUI 같은 무거운 의존성을 초기화하지 않습니다. 실제 비전 수집 도구가 호출될 때만 `VisionWorkerRuntime`이 perception/action 객체를 생성합니다.

이렇게 분리하면 SQLite QA 서버나 테스트가 단순 import만으로 GUI/모델 환경에 묶이지 않습니다.

## 5. 로컬 마커 설명

마커별 설명을 위한 별도 VLM 호출은 사용하지 않습니다. 로컬 OCR 텍스트와 아이콘 타입을 reasoning context에 전달하고, 텍스트가 없는 아이콘의 의미는 reasoning 단계가 현재 화면 이미지에서 판단합니다.

과거에는 `SKIP_VLM_CAPTION` 분기로 기존 VLM 캡셔닝을 우회했지만, 항상 우회하는 경로가 안정화된 뒤 설정과 비활성 구현을 제거했습니다.

## 6. Reflex Recipe 방향

Reflex Recipe는 LLM 판단을 코드로 강제 승격하는 구조가 아니라, 자율 탐색 실행 기록을 후보로 저장하고 Critic 피드백 루프가 재사용 가능성을 판단하는 구조로 둡니다.

고정 행동은 빠르게 재사용하되, 목표나 화면이 달라져 판단이 필요한 구간은 LLM에게 다시 넘기는 것이 현재 방향입니다.

이 구조는 임의의 사이트를 무설정으로 자동화하거나 반복 비용을 0으로 만드는 것을
목표로 하지 않습니다. 사이트별 공식 주소와 화면 단서는 선언형 프로필로
등록하며, 자율탐색 한 번과 Critic 검토에 든 비용도 학습 비용에 포함합니다.
Reflex의 효과는 hit 수가 아니라 동일 품질을 유지하면서 줄어든 추론·토큰과
반복 횟수별 누적 손익분기점으로 판단합니다.

## 7. 로컬 단일 작업자 직렬화

Realtime/Vision은 현재 화면과 물리 입력 장치를 공유하므로 병렬 worker를 실행하지 않습니다. `VisionWorkerRuntime.execution_session()`이 수집, 검토 재시도, 브라우저 정리 전체를 런타임별 잠금으로 직렬화합니다.

Perception, ActionTools, PaddleOCR subprocess, 컴파일된 작업자 그래프와 판단 모델은 요청 간 재사용합니다. 각 작업은 새 LangGraph state와 새 브라우저 창으로 시작하고 기본적으로 작업 종료 시 그 창만 닫습니다. OCR과 화면 모델은 애플리케이션 종료 때까지 유지하므로 서로 다른 요청의 목표·마커·카드 큐는 격리하면서 무거운 초기화 비용은 반복하지 않습니다.

## 8. 애플리케이션과 그래프 런타임 분리

`ApplicationRuntime`이 조사 체크포인터, 컴파일된 조사 그래프, `ChatService`, `VisionWorkerRuntime`, Reflex 승격 작업자를 소유합니다. FastAPI `lifespan`, CLI, E2E 실행기가 이 런타임을 명시적으로 열고 닫습니다. 사용자 질의의 지휘 책임은 `ChatService`, 작업자 실행 순서는 `worker_execution_service`, DB 적재는 `job_persistence_service`가 담당합니다.

전환 검증, 상세 OCR 버퍼, 결과 카드 큐, Reflex 재생은 `agent/runtime/`의 독립 모듈로 분리했습니다. 이 정책들은 DOM selector가 아니라 화면 서명, OCR 마커, 좌표비율만 입력으로 받습니다.

## 9. LLM 선택과 물리 도구 계약 분리

LLM은 화면의 의미와 목표 대상을 판단하지만, 선택한 마커가 해당 물리 도구로 실행 가능한지는 실행기가 검증합니다. 예를 들어 `type_in_marker`가 텍스트 없는 작은 아이콘을 가리키면 클릭과 `Ctrl+A`를 보내지 않고 같은 화면에서 다른 마커를 다시 판단하게 합니다.

LLM이 반환한 LangChain 메시지는 `reasoning_node` 밖으로 전달하지 않습니다. 이 경계에서 `ActionRequest`로 한 번 변환하고, Reflex와 공고 카드 큐도 같은 계약을 직접 생성합니다. `execution_node`는 요청 출처를 분기 기준으로 삼지 않고 검증된 호출만 실행하며, 실행 후에는 별도 `ActionResult`를 남깁니다. 따라서 정책 테스트와 실행기는 특정 채팅 메시지 구현에 의존하지 않습니다.

이 검증은 사이트명, 마커 번호, 화면 문구를 사용하지 않습니다. OCR 텍스트 여부, 내부 텍스트 포함 관계, bbox 형태처럼 화면에서 관찰한 물리적 affordance만 사용합니다. 따라서 사이트 의미를 코드로 복제하지 않으면서 닫기 버튼 같은 명백한 도구 계약 위반을 막습니다.

상세 공고 구조화에서도 증거 책임을 분리합니다. 최종 정제 모델의 1차 입력은 상세 페이지 OCR과 현재 URL입니다. 검색 목록에서 만든 카드 메타데이터는 OCR과 충돌할 수 있으므로 모델 입력에 섞지 않고, 모델이 회사명이나 직무명을 비운 경우에만 후처리 폴백으로 사용합니다.

## 10. 채용 사이트 공식 주소 선택

지원하는 사이트의 주소, 도메인, 화면 역할, 허용 도구, 판단 지침은 각 사이트의 `agent/sites/<slug>/profile.json` 한 파일에서 관리합니다. 사용자 요청의 사이트명, slug, 별칭, 도메인은 `get_official_site_url()`이 결정론적으로 해석하고, `open_browser(site=...)`는 검증된 공식 HTTPS 주소만 엽니다.

이 선택은 브라우저 주소를 LLM이 추론하게 하지 않으면서도 사이트별 탐색 로직을 `ActionTools` 안에 중복 하드코딩하지 않기 위한 것입니다. 주소를 연 이후의 검색과 화면 판단은 계속 OCR·화면·물리 입력 경로를 사용합니다.

## 11. 런타임 자원 수명

비싼 실행 자원과 작업 상태의 수명을 분리합니다.

- 애플리케이션 수명: 조사 체크포인터, 컴파일된 그래프, OmniParser YOLO, PaddleOCR worker, Gemini 기본/구조화 클라이언트, 승격 작업자
- 수집 요청 수명: 전용 Chrome 창, 탭과 사이트 렌더링 상태
- 단일 작업 수명: LangGraph state, 목표, 현재 마커, 결과 카드 큐, 상세 OCR 버퍼

Gemini 클라이언트는 모델명, temperature, 구조화 출력 스키마별로 캐시하지만 실제 API 요청을 미리 보내지는 않습니다. 유료 더미 요청은 서버 측 warm 상태를 보장하지 못하고 비용만 추가하기 때문입니다. 로컬 비전 자원은 `VisionWorkerRuntime`이 첫 수집 요청에서 지연 초기화하여 DB 질의와 UI 서버 기동이 GPU 환경에 묶이지 않게 합니다.

첫 수집 요청에서는 PaddleOCR worker 준비와 공식 사이트 열기를 병렬로 실행합니다. 이후 요청은 같은 OCR worker와 컴파일된 그래프를 재사용하되 새 브라우저 창과 새 그래프 상태로 시작합니다. OCR 하위 프로세스는 요청 횟수로 재시작하지 않고 실제 timeout이나 프로세스 실패가 발생했을 때만 복구합니다.

## 12. 시작 화면 준비와 빈 화면 대기

새 브라우저 창은 `about:blank`를 거치지 않고 사이트 공식 URL을 실행 인자로 직접 전달합니다. 창을 바인딩한 뒤 페이지 본문이 단색 로딩 화면이면 같은 URL을 다시 열지 않고 `PerceptionEngine.capture_usable_screen()` 한 곳에서 시간 한도로 기다립니다.

시작 서비스와 LangGraph는 별도 빈 화면 재시도 횟수를 갖지 않습니다. 대기 중 캡처는 같은 파일을 덮어쓰며, 정상 콘텐츠나 브라우저 오류 화면처럼 판단 가능한 화면이 나타나면 즉시 반환합니다. 제한시간 뒤에도 빈 화면이면 그때만 reasoning이 복구 행동을 판단합니다.

## 13. 지휘자와 경량 모델 분리

복잡한 조사 계획, 화면 행동 판단, worker 결과 검토와 레시피 Critic은 `gemini-3.6-flash`를 사용합니다. 상세 OCR 구조화, 검색 의도 추출, 결과 카드 후보 선택과 요약·정규화는 `gemini-3.5-flash-lite`를 사용합니다. 역할별 기본값은 `agent/application/model_policy.py` 한 곳에서 관리하고 `COMMANDER_MODEL`, `VISION_LIGHTWEIGHT_MODEL`로 교체할 수 있습니다.

선택 근거는 Google이 공개한 2026년 7월 평가에서 3.6 Flash가 OSWorld-Verified 83.0%, CharXiv 85.2%를 기록해 비교 모델보다 컴퓨터 조작과 복잡한 화면 정보 추론에서 우위를 보였고, 3.5 Flash-Lite는 문서 추출·구조화 작업을 주요 용도로 제시하면서 초당 350 출력 토큰의 처리량과 낮은 단가를 제공한 점입니다. 이 수치는 공급사 평가이므로 L2C의 실제 채용 사이트 완료율을 보장하지 않으며, 모델 교체 효과는 동일 E2E의 완료율, 잘못된 도구 선택, 구조화 품질, 실행시간과 토큰 비용으로 별도 검증합니다.

두 신규 모델은 기존 샘플링 인자와 `candidate_count`를 허용하지 않습니다. 현재 LangChain Google 어댑터는 생성자에서 값을 생략해도 Gemini 3 계열에 `temperature=1.0`을 다시 넣으므로, L2C 어댑터가 최종 생성 설정에서 `temperature`, `top_p`, `top_k`, `candidate_count`를 제거합니다. 실행마다 공급자 기본값이 달라지지 않도록 3.6 Flash는 medium thinking, 경량 모델은 minimal thinking을 명시합니다.

- 모델 및 벤치마크: https://deepmind.google/models/gemini/
- 마이그레이션과 API 제약: https://ai.google.dev/gemini-api/docs/latest-model
