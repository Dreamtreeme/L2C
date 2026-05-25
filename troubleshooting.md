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

## 7. OmniParser SoM 로컬 파이프라인 실제 구현 및 통합 (YOLOv8 + PaddleOCR + PIL 인메모리 처리)

### [현상]
* 비주얼 기반의 좌표 인식 및 에이전트 구동의 실효성을 높이기 위해, 기존 Mock 데이터를 걷어내고 실제 로컬 Set-of-Marks (SoM) 파이프라인인 **OmniParser 로컬 엔진**을 완전 통합하려 함.
* 초기 연동 시 Windows 환경의 PIL 이미지 오픈 시점의 **파일 락(File Lock) 및 라이브러리 DLL 충돌**로 멀티프로세싱 안정성이 떨어지는 현상이 식별됨.

### [해결 조치]
1. **인메모리 디코딩 적용**: 디스크 입출력 없이 mss 캡처의 raw BGRA 데이터를 메모리 레벨에서 직접 `BGRX` 디코더를 활용하여 PIL 이미지로 초고속 고정밀 변환 (`wait_stable.py`, `som_engine.py`).
2. **YOLOv8 & PaddleOCR 통합**:
   - `som_engine.py`에서 로컬 GPU(CUDA)를 바인딩하여 1.2s 수준으로 탐지 속도 가속화.
   - **IoU 기반 텍스트-아이콘 중복 필터링 (NMS)** 알고리즘을 적용하여 UI 마커 숫자가 겹치거나 지저분해지는 중복 마킹 현상을 말끔히 제거.
3. **듀얼 모니터 좌표 매핑 교정**: 감지된 오프셋 좌표를 브라우저 윈도우 시작점에 매핑하여, 서브 모니터에서도 오차 없이 정밀하게 요소를 타격하는 스케일 보정 코드 적용 완료.

### [관련 참조 리소스]
* **로컬 SoM 엔진**: [som_engine.py](file:///c:/Users/psg/Desktop/L2C/agent/tools/som_engine.py)
* **이미지 메모리 변환 모듈**: [wait_stable.py](file:///c:/Users/psg/Desktop/L2C/agent/utils/wait_stable.py)

---

## 8. VLM 캡셔닝 단계 제거 및 통합 최적화 (SKIP_VLM_CAPTION)

### [현상]
* 로컬 탐지를 완료했으나, 매 루프마다 생성한 마킹 이미지를 VLM(Gemini/Ollama)에 전달해 텍스트 설명을 적는 **VLM 캡셔닝 단계(Perception Node의 API 호출)**에서 매번 **6~9초의 큰 지연시간**이 고정 발생하여 전체 시나리오 지연의 주원인으로 나타남.

### [사용자 지시 및 검증 결과]
* **요청**: "이젠 OCR 분석시간보다 다른 병목 시간을 찾아 해결해 봐라. VLM 단계 호출 횟수를 축소해 볼 것."
* **해결 조치**:
  1. Gemini 3.5 Flash는 비전 능력이 탁월하므로 굳이 텍스트 사전 설명이 필요 없음을 간파.
  2. `SKIP_VLM_CAPTION=true` 환경변수 옵션을 추가하여 **VLM 캡셔닝 단계를 완전히 우회(Bypass)** 처리함.
  3. 로컬 PaddleOCR이 감지한 텍스트 데이터와 YOLOv8의 탐지 타입을 직접 결합하여 최소한의 텍스트 설명 컨텍스트를 perception 레벨에서 자율 매핑함.
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
