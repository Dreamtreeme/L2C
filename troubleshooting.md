# L2C 에이전트 개발 트러블슈팅 히스토리 (Engineering Log)

본 문서는 L2C 비전 에이전트 개발 및 E2E 테스트 과정에서 겪은 문제들과 사용자의 피드백에 따른 가설 검증 및 최적화 내역을 시간 순서대로 기록한 개발 로그입니다.

---

## 1. 원티드 E2E 초기 테스트 및 로그인 루프 발생

### [현상]
바탕화면에서 브라우저를 실행해 원티드 사이트에 접속하고 로그인하는 초기 E2E 시나리오 작동 시, 로그인 버튼 클릭 후 일반 로그인(ID/PW) 입력 화면 주위에서 마우스 액션이 헛돌거나 루프를 도는 현상이 발생함.

### [사용자 피드백 및 요구사항]
* "이미 자동 로그인되어 있을 가능성을 고려하여 로그인 상태를 선제 검증할 것."
* "구글 간편 로그인(Google Sign-In) 버튼을 타격해 기등록된 구글 계정으로 로그인을 시도할 것."

### [원인 분석 및 해결 조치]
* **원인**: `commander.py` 프롬프트에 자동 로그인 상태 분기 처리 지침과 구글 간편 로그인 우선순위 정책이 누락되어 있었음.
* **해결**: [commander.py](file:///c:/Users/psg/Desktop/L2C/agent/prompts/commander.py)의 시스템 프롬프트를 수정하여 에이전트가 로그인 상태를 감지하도록 가이드라인을 보강하고, 구글 로그인 버튼 및 기등록 이메일 요소를 클릭하는 상세 자율 행동 트랙을 구성함.

### [관련 참조 리소스]
* **수정 반영된 지휘자 프롬프트**: [commander.py](file:///c:/Users/psg/Desktop/L2C/agent/prompts/commander.py)
* **전체 실행 내역 및 대화 로그**: [transcript.jsonl](file:///C:/Users/psg/.gemini/antigravity/brain/2176c489-6bf5-40ce-aa85-7bc5c6eb6b97/.system_generated/logs/transcript.jsonl)

---

## 2. Ollama JSON 포맷 응답 붕괴 및 무한 개행 루프

### [현상]
구글 간편 로그인 화면 및 메인화면 분석을 위해 로컬 VLM을 호출했을 때, Ollama API 응답이 유효한 JSON을 반환하지 못하고 `\n\n\n\n...` 또는 `0,0,0...` 처럼 특정 토큰만 무한 반복 출력하며 Python JSON 파서가 크래시되거나 타임아웃(120초)이 발생함.

### [사용자 가설 및 지시]
* "Ollama 호출 시 `format='json'`을 강제하고 프롬프트 제약을 강화해 볼 것."
* "정규식 매칭 파서 대신, 대괄호/중괄호 내의 모든 `{...}`를 추출해 배열로 직접 복원하는 견고한 파서로 보강할 것."
* "동일 동작 무한 반복을 방지하기 위해 Reasoning 노드에 루프 감지 및 차단 로직을 추가할 것."

### [원인 분석 및 해결 조치]
* **원인**: Ollama의 문법 강제 샘플러(Grammar-based Sampler) 수준에서 JSON 포맷을 억제하는 연산 부하가 로컬 3B/7B 경량 VLM의 Attention 연산 용량을 초과함. VLM이 좌표 계산과 JSON 문법 통제를 동시 처리하지 못해 생성 레이어가 무너진 현상(Degeneration)으로 파악됨.
* **해결**:
  1. Ollama 호출 파라미터에서 `"format": "json"` 옵션을 해제하여 모델의 생성 자유도를 보장하되, 출력 텍스트 내 마크다운 코드블록 안에 JSON 데이터를 안전하게 담도록 프롬프트를 조정함.
  2. 단순 정규식 파서 대신 문자열 내 중괄호 `{`와 `}`의 열고 닫힘을 스택(Stack)으로 계측하여 온전한 JSON 객체만 발라내는 **Stack-based Parser**를 `perception.py`에 이식함.
  3. [nodes.py](file:///c:/Users/psg/Desktop/L2C/agent/graph/nodes.py)에 동일 액션 연속 반복 3회 감지 시 에이전트를 안전하게 중단 및 탈출시키는 **Loop Detection & Recursion Limit** 로직을 적용함.

### [관련 참조 리소스]
* **Ollama 포맷 재현 테스트 코드**: [run_ollama_format_test.py](file:///c:/Users/psg/Desktop/L2C/scratch/run_ollama_format_test.py)
* **Ollama 원시 JSON 붕괴 로그**: [raw_ollama_resp.json](file:///C:/Users/psg/.gemini/antigravity/brain/2176c489-6bf5-40ce-aa85-7bc5c6eb6b97/scratch/raw_ollama_resp.json)
* **스택 기반 파서 디버깅 스크립트**: [debug_ollama_output.py](file:///C:/Users/psg/.gemini/antigravity/brain/2176c489-6bf5-40ce-aa85-7bc5c6eb6b97/scratch/debug_ollama_output.py)
* **루프 방지가 탑재된 그래프 노드**: [nodes.py](file:///c:/Users/psg/Desktop/L2C/agent/graph/nodes.py)

---

## 3. 로컬 VLM(3B) 좌표 정합성 붕괴 및 마우스 타격 이탈

### [현상]
GNB(Global Navigation Bar) 바의 '돋보기(검색)' 아이콘이나 '로그인' 버튼 등 작고 촘촘한 요소를 클릭하려 할 때, 마우스 좌표가 엉뚱한 화면 중앙이나 여백을 타격하는 심각한 오차가 관찰됨.

### [사용자 검증 요청]
* "VLM 좌표 인식이 실패하는 원인 분석 및 해결 가설 제안."
* "실제 오차 로그를 추적하여 검증할 것."

### [원인 분석 및 해결 조치]
* **오차 분석**: 돋보기 아이콘의 실제 절대 좌표 범위 `(1440, 227)` 대비 로컬 VLM 3B가 반환한 좌표 기반 복원값은 `(1050, 500)` 영역으로 매핑되어 약 400px 이상의 편차가 발생한 것을 확인.
* **가설 1**: VRAM 제한 완화를 위해 스크린샷 해상도를 `512px`로 지나치게 압축 리사이징함에 따라 텍스트 및 UI 가장자리 정보가 뭉개져 스케일 복원 시 배율 오차가 극대화됨.
* **가설 2**: 3B 경량 모델 고유의 공간적 좌표 임베딩(Spatial Grounding) 한계로 복잡한 웹 UI 요소를 인지하지 못함.
* **해결**: 
  - 웹 브라우저 물리 제어의 신뢰성 보장을 위해 **Perception Node의 메인 분석 엔진을 클라우드 기반 Gemini 3.5 Flash API로 교체**함.
  - 이로 인해 돋보기 좌표 오차가 `(1444, 227)`로 1~3px 오차 내 명중에 성공하며 물리 제어 안전성이 극적으로 향상됨.
  - 로컬 Ollama 구동 시에는 정확도 방어선으로 해상도를 **`1024px`**로 늘리는 Fallback 브랜치 방안을 구축함.

### [관련 참조 리소스]
* **모니터 비율 및 배율 오프셋 보정 디버거**: [debug_coords.py](file:///c:/Users/psg/Desktop/L2C/scratch/debug_coords.py)
* **해상도 및 DPI 스케일 검증 도구**: [inspect_dpi.py](file:///C:/Users/psg/.gemini/antigravity/brain/2176c489-6bf5-40ce-aa85-7bc5c6eb6b97/scratch/inspect_dpi.py)
* **Gemini perception API 개별 검증 스크립트**: [run_gemini_perception_test.py](file:///c:/Users/psg/Desktop/L2C/scratch/run_gemini_perception_test.py)
* **1024px Fallback 및 캡처 스케일링 핵심 파일**: [perception.py](file:///c:/Users/psg/Desktop/L2C/agent/tools/perception.py)

---

## 4. UI 요소 추출 개수 제한(8개)으로 인한 검색 결과 인지 누락

### [현상]
구글 간편 로그인 성공 후 검색어('데이터 분석가') 입력과 검색 결과 목록 로딩까지는 완벽히 도달했으나, 에이전트가 완료(`finish_task`) 상태로 진입하지 못하고 검색 화면에서 동일 동작을 반복하다 recursion_limit에 도달하며 종료됨.

### [로그 분석 및 해결 조치]
* **원인**: 에이전트가 확보한 마지막 UI Context를 분석한 결과, 상단 GNB 바의 메뉴들(채용, 이력서, 이벤트 등)만 검출되고 본문의 검색된 채용 카드 요소들은 전혀 검출 목록에 존재하지 않았음. 이는 VLM 프롬프트에 정의되어 있던 **최대 검출 개수 제한(8개)**으로 인해 GNB에 밀려 본문 카드가 리스트에서 누락된 것이었음.
* **해결**: `perception.py` 내 VLM API 호출 시 최대 검출 요소(Max Elements)를 **`25개`**로 상향하여 본문 채용공고 리스트를 확보함. 지휘자(Gemini)가 검색 결과를 시각적으로 인지할 수 있게 되어 다음 단계에서 즉시 `finish_task`를 선언하고 최종 E2E 성공 처리함.

### [관련 참조 리소스]
* **Gemini UI 다중 요소 추출 테스트**: [test_gemini_full.py](file:///C:/Users/psg/.gemini/antigravity/brain/2176c489-6bf5-40ce-aa85-7bc5c6eb6b97/scratch/test_gemini_full.py)
* **Max Elements 조절 함수가 포함된 인식 도구**: [perception.py](file:///c:/Users/psg/Desktop/L2C/agent/tools/perception.py)

---

## 5. 로컬 VLM (Qwen2.5-VL 7B) 추론 지연 요인 규명 및 가설 검증

### [사용자 요구사항]
* "외부 API 비용(Gemini) 절감을 가정한 0원 로컬 VLM 대체 구동 시뮬레이션."
* "양자화 기술을 고려하여 더 큰 체급인 `qwen2.5vl:7b`로 모델을 변경해 속도 및 품질 테스트를 진행할 것."

### [테스트 및 가설 검증 결과]
* **환경**: RTX 3080 (10GB VRAM) 환경에서 Q4_K_M 양자화 모델(`6.0GB` 가중치) 탑재 후 검증 진행.
* **가설 1 (768px 하향 리사이징)**: 속도 단축을 꾀했으나 **46.64초**로 단축 효과가 전혀 없었으며, 화질 저하로 인해 로그인 버튼 클릭 지점이 타깃 좌측으로 356px 빗나가고 GNB 좌표가 뒤틀리는 등 정확도 저하가 심각해 **기각(Rejected)** 처리함.
* **가설 2 (num_ctx 2048 축소를 통한 GPU 연산 강제)**: VRAM 점유율을 낮춰 CPU 오프레딩을 예방하려 했으나 **54.81초**로 오히려 늘어남.
* **근본 원인 식별**: 현재 윈도우 환경 Ollama(llama.cpp 백엔드) 상에서 Qwen2.5-VL 모델을 돌릴 때 비주얼 엔코더의 Attention 연산 시 **Flash Attention 2 가속이 작동하지 않아** 이미지 토큰 프리필(Prefill) 병목에 고정 40초 이상의 하드웨어 부하가 발생함.

### [관련 참조 리소스]
* **7B 최적화 시뮬레이션 제어 코드**: [simulate_local_limit.py](file:///c:/Users/psg/Desktop/L2C/scratch/simulate_local_limit.py)
* **가설 1 (768px) 실행 결과 로그**: [task-830.log](file:///C:/Users/psg/.gemini/antigravity/brain/2176c489-6bf5-40ce-aa85-7bc5c6eb6b97/.system_generated/tasks/task-830.log)
* **가설 2 (num_ctx 2048) 실행 결과 로그**: [task-850.log](file:///C:/Users/psg/.gemini/antigravity/brain/2176c489-6bf5-40ce-aa85-7bc5c6eb6b97/.system_generated/tasks/task-850.log)

---

## 6. Ollama 한계 극복을 위한 Hugging Face 직접 구동 아키텍처 검토

### [사용자 질문]
* "Ollama 대신 허깅페이스에서 직접 모델을 로드하여 활용 시 연산 가속 여부."

### [결론 및 차기 마일스톤 (Phase 3.5 신설)]
* Ollama 프레임워크를 사용하지 않고 허깅페이스 `transformers`를 통해 가중치를 직접 받아 Python+PyTorch 환경에서 구동하면 **구조적 최적화가 가능**함을 확인함.
  1. Windows 환경 내 `flash-attn` 라이브러리를 CUDA C++로 빌드/설치하여 비주얼 토큰 연산량을 **3~4배 가속**하여 10초 이내 추론 실현 가능.
  2. `max_pixels` 파라미터 제어를 통해 해상도는 1024px로 완벽히 보존하면서 실제 트랜스포머 입력 토큰 개수만 선택적으로 압축 가능.
* 이에 따라, 로컬 VLM 가속화를 차기 핵심 연구 과제인 **`Phase 3.5: 로컬 VLM 최적화 도전 (Ollama -> Hugging Face 직접 구동)`**으로 명명하고 README 로드맵에 등록함.

---

## 7. OmniParser SoM 로컬 파이프라인 실제 구현 및 통합 (OmniParser + PaddleOCR + PIL 인메모리 처리)

### [현상]
* 비주얼 기반의 좌표 인식 및 에이전트 구동의 실효성을 높이기 위해, 기존 Mock 데이터를 걷어내고 실제 로컬 Set-of-Marks (SoM) 파이프라인인 **OmniParser 로컬 엔진**을 완전 통합하려 함.
* 초기 연동 시 Windows 환경의 PIL 이미지 오픈 시점의 **파일 락(File Lock) 및 라이브러리 DLL 충돌**로 멀티프로세싱 안정성이 떨어지는 현상이 식별됨.

### [해결 조치]
1. **인메모리 디코딩 적용**: 디스크 입출력 없이 mss 캡처의 raw BGRA 데이터를 메모리 레벨에서 직접 `BGRX` 디코더를 활용하여 PIL 이미지로 초고속 고정밀 변환 (`wait_stable.py`, `som_engine.py`).
2. **OmniParser YOLO & PaddleOCR 통합**:
   - `som_engine.py`에서 로컬 GPU(CUDA)를 바인딩하여 1.2s 수준으로 탐지 속도 가속화.
   - **IoU 기반 텍스트-아이콘 중복 필터링 (NMS)** 알고리즘을 적용하여 UI 마커 숫자가 겹치거나 지저분해지는 중복 마킹 현상을 말끔히 제거.
3. **듀얼 모니터 좌표 매핑 교정**: 감지된 오프셋 좌표를 브라우저 윈도우 시작점에 매핑하여, 서브 모니터에서도 오차 없이 정밀하게 요소를 타격하는 스케일 보정 코드 적용 완료.

### [관련 참조 리소스]
* **로컬 SoM 엔진**: [som_engine.py](file:///c:/Users/psg/Desktop/L2C/agent/tools/som_engine.py)
* **이미지 메모리 변환 모듈**: [wait_stable.py](file:///c:/Users/psg/Desktop/L2C/agent/utils/wait_stable.py)

---

## 7-1. PaddleOCR GPU 경로 충돌과 복귀

초기에는 PaddleOCR을 CPU 경로로 실행하면서 한 화면 처리 시간이 과도하게 길어져 EasyOCR로 단순화했다. 이후 GPU 환경을 다시 비교한 결과 PaddleOCR이 더 많은 텍스트 좌표를 더 빠르게 반환하는 것이 확인되어, 기본 OCR을 PaddleOCR로 되돌렸다.

Windows에서는 `paddle`을 먼저 import하면 `torch`가 `shm.dll` 로딩 중 실패했고, PaddleOCR GPU inference 시 `cudnn64_8.dll`을 찾지 못하는 문제가 있었다. 해결은 `torch/ultralytics`를 먼저 로드하고, `.venv/Lib/site-packages/nvidia/*/bin` 및 CUDA bin 경로를 프로세스 DLL 검색 경로에 추가한 뒤 PaddleOCR을 초기화하는 방식으로 정리했다.

---

## 8. VLM 캡셔닝 단계 제거 및 통합 최적화 (SKIP_VLM_CAPTION)

### [현상]
* 로컬 탐지를 완료했으나, 매 루프마다 생성한 마킹 이미지를 VLM(Gemini/Ollama)에 전달해 텍스트 설명을 적는 **VLM 캡셔닝 단계(Perception Node의 API 호출)**에서 매번 **6~9초의 큰 지연시간**이 고정 발생하여 전체 시나리오 지연의 주원인으로 나타남.

### [사용자 지시 및 검증 결과]
* **요청**: "이젠 OCR 분석시간보다 다른 병목 시간을 찾아 해결해 봐라. VLM 단계 호출 횟수를 축소해 볼 것."
* **해결 조치**:
  1. Gemini 3.5 Flash는 비전 능력이 탁월하므로 굳이 텍스트 사전 설명이 필요 없음을 간파.
  2. `SKIP_VLM_CAPTION=true` 환경변수 옵션을 추가하여 **VLM 캡셔닝 단계를 완전히 우회(Bypass)** 처리함.
  3. PaddleOCR이 감지한 텍스트 좌표와 OmniParser YOLO의 탐지 좌표를 직접 결합하여 최소한의 텍스트 설명 컨텍스트를 perception 레벨에서 자율 매핑함.
* **결과**: Perception Node 소요 지연 시간이 **7.12초 ➡️ 평균 1.31초로 약 81.7% 급감**함.

### [관련 참조 리소스]
* **캡셔닝 Bypass 분기 구현체**: [perception.py](file:///c:/Users/psg/Desktop/L2C/agent/tools/perception.py)

---

## 9. 대기 정밀도 및 프롬프트 토크나이저 최적화 & 타이머 배제를 통한 초고속 E2E 돌파

### [현상]
* VLM 캡셔닝 단계를 우회했음에도 의사결정(Reasoning) 노드 API 응답 시간이 약 5.8초로 여전히 묵직했고, 클릭 액션 직후 대기 반응속도 및 E2E 테스트 전체 체감 속도가 기대에 못 미침.

### [해결 조치]
1. **프롬프트 토큰 최적화**:
   - 수십 개에 달하는 단순 아이콘 마커 목록(`상호작용 가능한 요소 (icon)`)을 프롬프트 본문에서 제거하고, 하단에 단 한 줄로 축약(`기타 아이콘/버튼 마커 ID 목록: [0, 1, ...]`)하여 전송하도록 `nodes.py` 수정.
   - `COMMANDER_SYSTEM_PROMPT`를 자율 멀티모달 특성에 맞춰 Concise하게 군더더기를 깎아내 전송 토큰 부하를 대폭 경량화함.
   - 결과: **Gemini 의사결정 반응속도가 평균 5.8초 ➡️ 2.6초 수준으로 단축**.
2. **대기 정밀도 및 루프 튜닝**:
   - 클릭 액션 후 화면 안정화를 감지하는 `WaitStable` 체크 주기를 **200ms ➡️ 50ms**로 좁혀 화면 안정 즉시 반응하도록 가속화.
3. **백그라운드 타이머 제거**:
   - 기존에 백그라운드 태스크 진행 상태 확인을 위해 무의식적으로 걸어두던 `schedule` 수동 타이머(12~15초 강제 대기)가 전체 체감 속도를 갉아먹고 있었음을 사용자 피드백으로 인지.
   - 강제 타이머를 배제하고 시스템의 비동기 반응 완료 이벤트(Reactive Wakeup)가 올 때 즉시 리스폰하도록 설계 변경.
* **결과**: 마침내 **전체 E2E 4단계 완주 총 소요 시간 단 `19.3초`** 돌파에 성공함!

### [관련 참조 리소스]
* **대기 정밀도 변경 모듈**: [wait_stable.py](file:///c:/Users/psg/Desktop/L2C/agent/utils/wait_stable.py)
* **프롬프트 토큰 축약 적용 노드**: [nodes.py](file:///c:/Users/psg/Desktop/L2C/agent/graph/nodes.py)
* **지휘자 프롬프트**: [commander.py](file:///c:/Users/psg/Desktop/L2C/agent/prompts/commander.py)

---

## 10. QA 서버 import 시 비전 엔진까지 초기화되는 문제

### [현상]
웹 Q&A 서버(`agent/web_server.py`)와 SQLite 질의 테스트는 화면 인식이나 물리 브라우저 제어가 필요하지 않은데도, `agent.graph.nodes`를 import하는 순간 `PerceptionEngine()`과 `ActionTools()`가 전역 싱글톤으로 생성되었습니다. 이로 인해 서버 시작 경로가 YOLO 모델, mss 캡처 장치, PyAutoGUI, OmniParser 가중치 상태에 종속되었습니다.

### [원인 분석]
Phase 3 이후 성능 최적화를 위해 도구와 LLM 클라이언트를 모듈 전역에서 재사용하도록 만들었지만, QA 지휘자와 비전 실행자가 같은 `nodes.py` 안에 공존하면서 import side effect가 커졌습니다. 특히 Phase 7의 SQLite Q&A 경로는 비전 엔진이 필요하지 않으므로 이 결합은 불필요한 장애면이었습니다.

### [해결 조치]
1. `PerceptionEngine`, `ActionTools`, 브라우저 자동화용 Gemini 도구 바인딩, QA용 Gemini 도구 바인딩을 `_get_*()` 헬퍼로 지연 초기화했습니다.
2. `agent/main.py`와 `realtime_scraping.py`의 초기 `GraphState`에 `step_durations`를 명시하여 상태 스키마 누락 가능성을 줄였습니다.
3. Windows/WSL 혼합 환경에서 줄끝 변경만으로 대형 diff가 생기지 않도록 `.gitattributes`에 LF 정책을 추가했습니다.

### [검증]
* `python.exe -m pytest agent/tests -q` 결과: `6 passed`
* `python.exe -m compileall -q classic agent shared benchmark scratch` 통과
* `git diff --check` 통과
---

## 11. Reflex Recipe 실험: 저수준 클릭 캐시의 한계와 방향 전환

### [목표]
기존 자율 실행형 에이전트는 `OCR/SoM 화면 인식 → LLM 판단 → 도구 호출` 구조로 동작했다. 실제 수집은 가능했지만, 매 화면마다 Gemini reasoning을 호출하므로 시간과 API 비용이 컸다. 이를 줄이기 위해 자율탐색 중 성공 행동을 기록하고, 이후 같은 화면에서는 `OCR/SoM → 캐시된 도구 호출`로 reasoning을 건너뛰는 **Reflex Recipe**를 실험했다.

### [테스트 중 확인한 실패/문제]
1. **처음부터 끝까지 스크립트화하기에는 화면 가변성이 너무 컸다.**
   - 검색 결과 카드, 공고 제목, 회사명, 추천/광고/팝업, 결과 개수는 실행 시점마다 바뀐다.
   - URL과 OCR 텍스트 조합으로 만든 `state_key`는 같은 상세 페이지의 고정 UI에는 잘 맞지만, 검색 결과 목록처럼 데이터가 계속 바뀌는 화면에서는 쉽게 달라진다.

2. **마커만 기록하면 LLM이 실제로 고른 대상의 의미가 사라졌다.**
   - 공고 카드를 눌렀는데 OCR 마커는 카드 제목이 아니라 `상호작용 가능한 요소 (icon)`, 보상/태그 같은 조각으로 저장되는 문제가 있었다.
   - 이 상태로 재생하면 “왜 이 카드를 골랐는지”가 없어져 단순 좌표/텍스트 휴리스틱처럼 변했다.
   - 이후 `target_label`을 추가해 LLM이 고른 카드 제목을 함께 기록하도록 바꿨다.

3. **환경 팝업/승인창이 레시피를 오염시켰다.**
   - 실제 로그에서 `Approve`, `Approve for session` 같은 Codex/브라우저 승인 요소가 화면에 섞였고, 이것도 클릭 대상으로 기록되었다.
   - 이런 요소는 사이트 고유 노하우가 아니므로 다음 실행에서 재사용하면 안 된다.

4. **초기 구현은 Reflex가 동작해도 매번 reasoning으로 되돌아갔다.**
   - `상세 정보 더 보기`, `scroll`, `go_back` 같은 고정 UI 액션은 Reflex로 빠르게 실행 가능했다.
   - 하지만 펼쳐진 상세 본문에서 회사명/직무명/주요업무/자격요건을 구조화하는 작업은 캐시된 클릭만으로 해결되지 않았다.
   - 당시에는 `REFLEX_REASON_AFTER_HIT=1`로 검증 직후 reasoning을 강제했지만, 이 구조는 Reflex의 지연 절감 효과를 줄였다.
   - 현재는 행동별 전환 계약(`common_ready_cues`, 정상 `outcomes`, 선택적 `loading_cues`)을 Critic이 만들고, Reflex는 OCR로 계약만 검사한다. 계약 충족 시 다음 Reflex로 진행하고 미확인·시간 초과 때만 reasoning으로 폴백한다.

5. **방문 카드 순회 정책이 약했다.**
   - 기대 동작은 `검색 → 공고 수 확인 → N개 수집 계획 → 상세 진입 → 수집 → 뒤로가기 → 이전 카드 제외 후 다음 카드 클릭`이었다.
   - 실제로는 “방문한 카드 목록을 구조적으로 관리하고 다음 미방문 카드를 고르는” 전용 정책이 부족해, 이 부분이 LLM의 즉흥 판단에 많이 의존했다.

### [실제 로그 근거]
- `logs/manual_e2e/actual_collect_20260614_target_label_autonomous1.log`
  - 자율탐색으로 1건 적재에는 성공했지만 recursion limit 45에 도달했다.
  - Reasoning은 화면마다 대략 2.5~9초가 걸렸고, 초반에는 `Approve`, `Approve for session` 클릭이 섞여 레시피 오염 가능성이 확인되었다.
  - 이후 공고 카드 클릭에는 `target_label: '[병역특례 현역/보충역] iOS 개발자'`가 기록되어 카드 제목 메타데이터 보강이 유효함을 확인했다.

- `logs/manual_e2e/actual_collect_20260614_target_label_reflex1.log` (전환 계약 도입 전 기록)
  - Reflex hit 자체는 매우 빨랐다. 예: 상세 페이지의 `상세 정보 더 보기` 클릭은 `duration=0.002s`, `go_back`은 `duration=0.001s` 수준으로 기록되었다.
  - 당시에는 Reflex 후 `Routing to reasoning after reflex hit for extraction and next-step verification` 경로로 매번 reasoning에 들어갔다.
  - 결과적으로 1건 적재에는 성공했지만 recursion limit 25 전에 두 번째 공고 추출까지 완료하지 못했다.

- `logs/manual_e2e/actual_collect_20260614_close_browser_smoke1.log`
  - E2E 종료 후 브라우저가 열린 채 남는 문제를 확인하고 `close_browser` cleanup을 추가했다.
  - smoke 실행에서 `Browser cleanup completed ... {'closed': True}` 로그로 종료 정리가 검증되었다.

### [시도한 방향]
1. **성능 최적화**
   - `SKIP_VLM_CAPTION=true`, WaitStable 대기 단축, OCR 입력 리사이즈로 perception 병목을 줄였다.
   - 이후 병목은 주로 reasoning 호출 시간과 반복 판단 비용으로 이동했다.

2. **저수준 Reflex Recipe**
   - `state_key = URL 템플릿 + OCR 앵커 텍스트 해시`로 화면 상태를 만들고, 같은 상태에서 기록된 `click/type/scroll/go_back`을 재생했다.
   - 고정 UI에는 효과가 있었지만, 가변 UI까지 일반화하려 하자 하드코딩과 휴리스틱이 급격히 늘어났다.

3. **의미 메타데이터 보강**
   - 단순 마커 ID 대신 `target_label`, 주변 evidence text, region/ordinal을 기록했다.
   - 카드 제목처럼 LLM이 실제로 고른 의미를 보존하는 데는 도움이 됐지만, 이것만으로 “다음 미방문 카드 선택” 같은 탐색 전략을 대체할 수는 없었다.

4. **가드레일 추가**
   - 같은 상태에서 같은 UI 액션 반복 차단, 화면 전환 액션 뒤 체이닝 제한, 브라우저 재오픈 방지, 종료 후 브라우저 닫기를 추가했다.
   - 안정성은 좋아졌지만, 자율탐색 전략 자체를 해결하는 것은 아니었다.

### [결론]
이번 실험의 핵심 교훈은 **Reflex를 전체 자율탐색의 대체재로 쓰면 범용성이 급격히 떨어진다**는 것이다. Reflex는 `상세 정보 더 보기`, `scroll`, `go_back`, 검색창 열기처럼 사이트 구조상 고정된 UI를 빠르게 재생하는 데 적합하다. 반면 공고 카드 선택, 필터 사용 여부, 검색 범위 축소, 본문 추출, 예외 처리처럼 목표와 데이터에 따라 달라지는 판단은 여전히 LLM 또는 별도 탐색 정책이 필요하다.

따라서 다음 방향은 저수준 클릭 스크립트를 늘리는 것이 아니라 다음처럼 역할을 나누는 것이다.

- **공통 탐색 정책**: 검색 결과에서는 필터/정렬/카드 목록을 먼저 파악하고, 위에서 아래로 미방문 카드를 순회한다.
- **사이트 매뉴얼 JSON**: 자율탐색 성공 후 `화면 종류`, `안정 랜드마크`, `반복 가능한 고정 액션`, `목표에 따라 달라지는 변수`, `무시해야 할 요소`를 요약 저장한다.
- **Reflex 적용 범위 제한**: 검증 가능한 고정 UI 액션만 reasoning 없이 재생한다.
- **LLM 유지 구간**: 카드 선택, 필터 전략, 본문 정보 추출, 팝업/예외 복구는 LLM이 맡는다.

즉, 앞으로의 방향은 “LLM 판단을 통째로 스크립트화”가 아니라 **사람이 웹사이트를 익히듯 성공 탐색에서 사이트별 매뉴얼을 만들고, 그중 안정적인 조작만 Reflex로 단축**하는 쪽이 더 현실적이다.

---

## 12. 스크립트 강제 대신 피드백 루프로 Reflex를 승격하는 방향 전환

### [배경]
저수준 Reflex Recipe 실험 이후, 다음 단계로 사이트별 페이지 구조를 기록하고 각 페이지에서 해야 할 일을 명시하는 방안을 검토했다. 검색 결과 페이지, 상세 페이지, 팝업 같은 `page_role`을 나누고, 공고 카드 컴포넌트의 제목·회사명·태그·보상금 등을 구분하면 Reflex가 더 명확해질 것이라는 가설이었다.

하지만 이 방향을 코드 제약으로 밀어붙이면 다시 하드코딩이 늘어난다는 문제가 있었다. 예를 들어 “검색 결과에서는 첫 번째 공고 카드의 제목을 클릭한다”는 정책은 일반적으로 맞지만, 실제 화면에서는 광고 카드, 추천 카드, 태그, 보상금, 로그인 팝업이 섞일 수 있다. 또한 검색어가 바뀌면 카드 텍스트는 바뀌지만 카드 순회 절차는 유지되고, 상위 5개만 가져오라는 요청이면 반복 횟수만 바뀐다. 즉 모든 행동을 새로 추론할 필요도 없고, 반대로 모든 것을 고정 스크립트로 묶어서도 안 된다.

### [사용자 문제 제기]
- Reflex는 지연시간 없이 빠르게 정해진 행동을 수행해야 한다.
- 검색어가 바뀌어도 검색어 입력만 바뀌고, 검색 결과 카드 순회 절차는 바뀌지 않는다.
- 같은 검색어라도 “상위 5개”처럼 수집 개수가 바뀌면 반복 횟수만 바뀌어야 한다.
- 따라서 추론은 동작 전체를 다시 짜는 것이 아니라, 수정이 필요한 일부 파라미터만 바꿔야 한다.
- 다만 파라미터와 고정 행동을 스크립트가 미리 강제하면 또 다른 하드코딩이 되므로, 피드백 루프에서 성공/실패를 걸러내야 한다.

### [설계 결론]
앞으로의 Reflex는 “정답 스크립트 생성기”가 아니라 **행동 제안 → 실행 → 관찰 → 평가 → 승격** 루프가 되어야 한다.

```text
Explore Actor
→ LLM/VLM이 행동, 대상, 파라미터 후보, 고정 절차 후보를 제안

Action
→ 기존 click_marker/type_in_marker/scroll/go_back 도구로 실제 실행

Observer
→ 화면 변경 action과 다음 OCR·스크린샷을 같은 action seq의 transition_observations로 기록

Critic
→ success / partial / wrong_target / no_effect / loop_risk 로 피드백 라벨링

Recipe Memory
→ 반복 성공 패턴은 confidence 상승, 실패 패턴은 negative example로 저장

Reflex
→ confidence가 충분히 높은 패턴만 LLM 없이 실행
→ 전환 계약이 pending이면 재관찰, ready이면 다음 행동, unknown/timeout이면 Explore로 폴백
```

### [핵심 아이디어]
1. **LLM/VLM은 제안자다.**
   - “이 값은 query 슬롯 후보다”, “이 클릭은 job_card.title 후보를 선택한 것이다”, “이 절차는 검색어가 바뀌어도 유지될 수 있다”를 제안한다.
   - 코드가 처음부터 파라미터/고정 절차를 확정하지 않는다.

2. **Observer가 실행 결과를 사실로 남긴다.**
   - 클릭 후 상세 페이지로 이동했는지, go_back 후 목록으로 돌아왔는지, 수집 데이터가 DB에 적재됐는지 같은 실제 결과를 기록한다.
   - `Approve`, 브라우저 툴바, 광고/팝업처럼 사이트 고유 동작이 아닌 요소가 끼었는지도 오염 신호로 남긴다.

3. **Critic이 승격 여부를 가른다.**
   - 한 번 성공했다고 바로 Reflex로 쓰지 않는다.
   - 같은 사이트·page_role·작업 유형에서 반복 성공해야 confidence가 올라간다.
   - 실패하거나 모호한 패턴은 Reflex 후보에서 제외하거나 negative example로 보존한다.

4. **Reflex는 추론하지 않는다.**
   - Reflex는 이미 승격된 절차에 `query`, `sample_count`, `site` 같은 슬롯만 주입해 빠르게 실행한다.
   - 예: 검색어가 바뀌면 `type_search_query`의 텍스트만 바뀐다.
   - 예: 상위 5개 요청이면 `collect_cards` 루프 횟수만 바뀐다.
   - 공고 카드 순회 정책은 유지하지만, 현재 결과에서 어떤 제목을 누를지는 실행마다 바뀌므로 작업자가 미방문 카드를 판단한다.
   - 검색 열기·검색어 입력·제출·검증 가능한 스크롤·상세 펼치기는 Reflex 후보이고, 현재 카드 선택과 남은 수집 개수에 따른 복귀/종료 판단은 reasoning 구간이다.

### [후속 수정: 탐색 당시 공고 제목이 활성 레시피에 남은 문제]

초기 Critic은 `job_card_title`을 `fixed=false`로 표시했지만 승격 코드는 이 값을 사용하지 않고 후보의 모든 행동을 활성 `recipes`에 저장했다. 그 결과 Android 탐색에서 선택한 `기술연구소 Android App 개발자`가 검색어가 바뀐 뒤에도 Reflex 클릭 대상으로 남았고, 상세 수집 후 목록으로 돌아오면 같은 공고를 다시 여는 루프가 발생했다.

이를 다음처럼 수정했다.

- Critic 단계 출력에 `replay_mode = fixed | parameterized | reasoning`을 추가했다.
- `fixed`는 실행마다 같은 안정 UI, `parameterized`는 `query` 같은 명시적 슬롯만 바뀌는 UI, `reasoning`은 현재 카드 목록·방문 이력·남은 목표 수에 따라 달라지는 판단으로 정의했다.
- 활성 레시피에는 `fixed`와 `parameterized` 단계만 저장하고 `reasoning` 단계는 제외한다.
- 후보를 재승격할 때 해당 후보의 과거 상태 행을 교체하여 이전 특정 공고명 레시피가 남지 않게 했다.
- reasoning 프롬프트에 목표 공고 수, 현재 수집 수, 이미 방문한 공고 제목을 전달하고 목표를 채우면 같은 카드를 다시 열지 않고 종료하도록 했다.

검증은 Android 탐색 후보를 승격한 뒤 검색어를 `ios 개발자`, 목표를 2건으로 바꿔 수행했다. 현재 화면에서 `[Vrew/vFlat] iOS 개발자`와 `[병역특례 현역/보충역] iOS 개발자`를 각각 선택했고, 첫 상세 수집 후 목록 복귀와 두 번째 미방문 카드 선택을 거쳐 `is_finished=true`, 2건 저장으로 종료했다. 탐색 당시 Android 공고명은 재생되지 않았다.

### [수정된 구현 방향]
- README의 Phase 8 목표를 “결정론적 Playwright 스크립트 자동 생성”에서 “피드백 루프 기반 Reflex Recipe 승격”으로 변경한다.
- 초기 탐색은 VLM+OCR을 함께 사용한다. VLM은 페이지 구조와 컴포넌트를 이해하고, OCR/SoM은 실행 가능한 marker_id와 이후 Reflex 재생 기준을 제공한다.
- 레시피 저장 단위는 단순 `marker_id`가 아니라 `proposal`, `before`, `after`, `feedback`, `confidence`를 포함한 실행 에피소드가 된다.
- Reflex는 높은 confidence의 recipe만 사용하고, 화면이 예상과 다르거나 카드 후보를 못 찾으면 추론하지 않고 Explore로 넘긴다.

### [기대 효과]
- 하드코딩된 스크립트를 늘리지 않고도 반복 성공 패턴을 자동으로 승격할 수 있다.
- 검색어/수집 개수처럼 바뀌는 파라미터만 수정하고, 안정적인 행동 절차는 재사용할 수 있다.
- 실패 사례도 negative example로 남아 다음 Reflex 승격을 더 보수적으로 만든다.
- 초기 자율탐색의 고비용은 유지하되, 반복 실행에서는 LLM reasoning 호출을 줄일 수 있다.
