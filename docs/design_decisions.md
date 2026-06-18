# 기술 및 설계 결정

이 문서는 L2C 프로젝트에서 채택한 주요 구조와 트레이드오프를 기록합니다.

## 1. Classic DOM 수집과 Vision Agent 병행

Classic 경로는 Playwright DOM 기반 수집입니다. 빠르고 정확하지만 사이트별 셀렉터와 예외 처리가 필요합니다.

Vision Agent 경로는 화면 캡처, YOLOv8, EasyOCR, LLM, 물리 입력 도구를 조합합니다. DOM 구조를 직접 읽지 않으므로 사이트 구조 변경에는 상대적으로 덜 민감하지만, OCR과 LLM 추론 비용이 추가됩니다.

두 경로를 모두 유지하는 이유는 목적이 다르기 때문입니다. Classic은 안정된 사이트의 고속 수집에 적합하고, Vision Agent는 새 사이트 개척과 UI 탐색 가능성 검증에 적합합니다.

## 2. Set-of-Marks 방식 채택

LLM에게 절대 좌표를 직접 예측하게 하지 않고, 로컬 비전 엔진이 화면 요소에 숫자 마커를 붙입니다. LLM은 `marker_id`를 선택하고, 클릭 좌표 계산은 도구가 처리합니다.

이 구조는 DPI, 창 위치, 브라우저 크기 변화에서 발생하는 클릭 오차를 줄입니다. 또한 reasoning prompt에는 이미지와 함께 현재 마커 텍스트 목록을 넣을 수 있어, LLM이 화면을 더 좁은 후보 집합으로 판단할 수 있습니다.

## 3. OCR 엔진 기준

현재 SoM 텍스트 감지는 EasyOCR을 사용합니다. 같은 원티드 검색 결과 스크린샷 기준으로 CPU 환경에서도 1280px OCR 입력이 약 4.5초였고, 카드 제목 마커도 유지되었습니다. 입력 이미지는 기본적으로 긴 변 1280px 이하로 줄인 뒤 OCR에 넣고, 감지 좌표는 원본 스크린샷 기준으로 복원합니다.

## 4. Lazy Initialization

`agent.graph.nodes` import 시점에는 브라우저 캡처, YOLOv8, OCR, PyAutoGUI 같은 무거운 의존성을 초기화하지 않습니다. 실제 비전 수집 도구가 호출될 때만 perception/action 객체를 생성합니다.

이렇게 분리하면 SQLite QA 서버나 테스트가 단순 import만으로 GUI/모델 환경에 묶이지 않습니다.

## 5. VLM Caption Bypass

`SKIP_VLM_CAPTION=true` 설정에서는 마커별 설명을 VLM으로 다시 캡셔닝하지 않고, 로컬 OCR 텍스트와 아이콘 타입만 reasoning context에 전달합니다.

이 경로는 perception 비용과 API 호출 수를 줄이는 대신, 텍스트가 없는 아이콘의 의미 추론은 reasoning 단계의 이미지 입력에 더 의존합니다.

## 6. Reflex Recipe 방향

Reflex Recipe는 LLM 판단을 코드로 강제 승격하는 구조가 아니라, 자율 탐색 실행 기록을 후보로 저장하고 Critic 피드백 루프가 재사용 가능성을 판단하는 구조로 둡니다.

고정 행동은 빠르게 재사용하되, 목표나 화면이 달라져 판단이 필요한 구간은 LLM에게 다시 넘기는 것이 현재 방향입니다.
