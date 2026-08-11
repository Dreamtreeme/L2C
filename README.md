# L2C — LLM to Computer

> 자연어 질문을 DB 근거와 실제 웹 수집으로 해결하고, 비전 탐색의 반복 가능한
> 단계만 재사용하는 로컬 채용시장 조사 에이전트입니다.

## 프로젝트 의도

기존 자동화는 두 진영으로 나뉩니다.

**전통 Playwright 자동화**는 빠르고 저렴하지만, 사이트별 셀렉터와 도구 사용 순서를 사람이 직접 분석해야 합니다. UI가 바뀌면 사람이 다시 코드를 고쳐야 합니다.

**범용 웹 에이전트**(Browser Use 등)는 브라우저의 HTML 정보와 선택적 비전
입력을 이용해 사이트별 selector 코드를 직접 작성하지 않고도 탐색할 수 있습니다.
다만 LLM 판단이 반복되는 구간에는 시간과 API 비용이 발생합니다. L2C의
2026-07-28 원티드 `iOS 개발자 공고 1개` 실제 수집에서는 89.72초 동안
63,078토큰과 추정 비용 `$0.1092`를 사용했습니다. 이 실행은 Gemini 3.6 Flash와
Gemini 3.5 Flash Lite를 사용했으며, 비용은 `config/model_pricing.json`에 저장된
당시 가격표로 계산했습니다.

같은 입력 56,256토큰, 출력 6,822토큰과 89.72초 실행시간을
[Browser Use 공식 Pay As You Go 가격표](https://browser-use.com/pricing)에
대입하면 V3 BU Mini는 `$0.0812`, BU Max는 `$0.3373`입니다. 여기에는 작업 시작
`$0.01`과 2분으로 올림한 원격 브라우저 `$0.002`를 포함하고 프록시 비용은
제외했습니다. 이 값은 동일 사용량을 가격표로 환산한 추정치이며, Browser Use가
같은 과제를 수행할 때 실제로 소비하는 토큰과 단계 수는 별도 실행으로 측정해야
합니다.

L2C는 두 방식의 장점을 결합하는 구조를 실험합니다. 지원할 사이트에는 공식
주소와 화면 단서를 담은 작은 선언형 프로필을 등록하고, 검색과 상세 탐색은
사이트별 실행 코드가 아니라 공통 비전 작업자가 수행합니다. 자율 탐색에서 성공한
행동 중 Critic과 화면 증거가 안전하다고 판단한 기록을 검증 가능한 상태 전이
경로로 승격합니다. 검색 진입·입력처럼 고정 가능한 부모 경로는 재생하고, 현재 공고
선택·상세 내용처럼 달라지는 자식 노드는 LLM 판단에 남깁니다.

따라서 이 프로젝트가 검증하려는 것은 **무설정으로 모든 웹사이트를 자동화하거나
반복 비용을 0으로 만드는 것**이 아닙니다. 사이트별 DOM 어댑터 작성량을 선언형
프로필 수준으로 줄이고, 재사용 가능한 단계의 비율과 반복 횟수가 충분할 때
자율 탐색보다 누적 추론 비용이 낮아지는지를 검증하는 것이 핵심입니다.

제품 흐름에서는 사용자의 질문을 먼저 조사 계획으로 바꾸고, SQLite 근거가
충분하면 웹을 열지 않습니다. 부족한 정보만 실제 채용사이트에서 수집한 뒤
구조화된 DB 문서와 `job_id` 출처를 사용해 답변합니다.

첫 번째 실험 도메인은 채용공고 수집입니다. Phase 5의 초기 표본 비교에서는 비전
기반 자율 수집과 본문 추출 가능성을 확인했으며, 이를 전체 사이트 성공률로
해석하지 않습니다. 현재는 Phase 8에서 자율 탐색 기록을 경험 기반 탐색에 재사용해
reasoning 호출을 줄이는 구조를 검증 중입니다. 자세한 결과는
[`troubleshooting.md`](./troubleshooting.md)에서 확인할 수 있습니다.
설계·실험·운영 문서 전체는 [`docs/index.md`](./docs/index.md)에서 찾을 수
있습니다.

## 현재 단계

현재 위치는 **Phase 8: 피드백 루프 기반 Reflex Recipe 승격**입니다.

완성된 것은 전체 범용 replay가 아니라, **고정 가능한 부모 경로를 pHash/OCR 서명으로 재사용하고 마지막 가변 판단은 LLM에게 남기는 경험 기반 탐색**입니다. `ios 개발자 공고 2개` 원티드 경험 기반 탐색에서 기존 자율 탐색과 동일하게 2건을 저장하면서 reasoning 횟수를 24회에서 9회로 줄였습니다.

2026-07-29 현재 원티드·사람인·고용24의 격리 DB에서
`자율 탐색 → Critic 승격 → 경험 기반 탐색` 짝 회귀가 모두 요청한 공고 저장
품질을 통과했습니다. 경험 기반 탐색은 원티드에서 추론 `11회 → 8회`,
사람인에서 `5회 → 3회`, 고용24에서 `5회 → 4회`로 줄었습니다. 실행시간은
원티드 `70.46초 → 63.38초`, 고용24 `38.94초 → 35.89초`로 줄었지만
사람인은 `44.81초 → 48.16초`로 늘어, 추론 감소가 단일 실행의 전체 시간
감소를 항상 보장하지 않음을 다시 확인했습니다.

원티드 `iOS 개발자 2개`에서 승격한 검색 레시피를 `백엔드 개발자 1개`에
재사용한 실행도 39.31초에 1건 저장으로 통과했습니다. 같은 격리 DB에 이미
있는 iOS 공고 2개를 다시 요청한 경우에는 카드 제목·회사 근거로 기존 DB ID
2개를 확인한 뒤 상세 페이지를 열지 않고 20.33초에 종료했습니다.

현재 결론은 다음과 같습니다.

- pHash 기반 Reflex는 원티드의 유사 채용공고 수집 흐름에서 reasoning 감소 효과를 확인했다.
- 동일한 ROI 재생 구조가 사람인·잡코리아·고용24에서도 동작하지만, 승격 범위가 작으면 전체 실행시간은 줄지 않는다.
- 검색 진입, 일부 목록 이동, 반복 가능한 UI 조작처럼 안정적인 상위 경로는 replay 후보로 적합하다.
- 공고 상세 본문, 최초 검색 결과의 카드 의미 선택, 예외 화면처럼 매번 달라지는 판단은 LLM이 담당한다. 한 번 만든 카드 큐 안의 다음 항목은 저장 좌표로 재생한다.
- DB에서 확인된 중복 카드는 신규 수집과 구분하되 사용자 목표를 해결한 카드로 계산해 불필요한 상세 순회를 끝낸다.
- 다음 단계는 사이트별 표본 수를 늘려 경험 기반 탐색 성공률·잘못된 클릭률·공고당 비용을 비교하는 것이다.

## 차별점

기존 자동화:
```python
extract("https://www.wanted.co.kr/wd/123456")  # URL 수동 입력 + 사이트별 코드
```

이 시스템:
```python
agent.run("데이터 분석가 신입 공고 모아줘")  # 지휘자 계획 + 등록 사이트 비전 탐색
```

| 단계 | 전통 Playwright | 범용 웹 에이전트 (Browser Use 등) | L2C |
|------|----------------|-----------------------------|-----|
| 사이트 준비 | selector·실행 순서 구현 | 목표와 접속 정보 제공 | 선언형 프로필 등록 후 공통 작업자 사용 |
| 신규 사이트 추가 | 사이트별 어댑터 코드 작성 | 공통 에이전트로 시작하고 필요한 단계에서 LLM 판단 | DOM 코드 없이 프로필과 첫 비전 탐색으로 시작 |
| 반복 작업 비용 | 낮음 | 동일 사용량 환산 시 BU Mini `$0.0812`, BU Max `$0.3373` | 원티드 iOS 1건 실제 수집 `$0.1092`; 승격된 단계만 추론 생략 |
| UI 변경 대응 | selector 수정 필요 | 현재 화면에서 다시 추론 | 서명 불일치 시 재생 중단 후 비전 탐색으로 폴백 |
| 비용 판단 | 실행당 비용이 낮고 일정 | 사용 모델과 캐시 정책을 포함해 측정 필요 | Critic 비용을 포함한 반복 손익분기점으로 평가 |

L2C의 범용성은 아무 설정 없이 임의의 웹사이트를 처리한다는 뜻이 아닙니다.
등록된 여러 사이트에서 클릭·입력·스크롤·뒤로가기와 같은 공통 물리 도구와
동일한 관찰·복구·학습 구조를 재사용하고, 사이트별 실행 코드를 만들지 않는다는
뜻입니다. 안정된 단일 사이트를 대량 처리할 때는 여전히 Playwright가 더
적합합니다.

## 두 시스템 비교

차이를 정량적으로 측정하기 위해 동일 작업을 두 방식으로 구현했습니다.

**Classic — 전통 Playwright 자동화 (베이스라인)**

Playwright로 DOM 구조를 직접 파싱합니다. 사이트별 마커와 셀렉터를 사전 정의해야 합니다. 빠르고 안정적이지만 사이트별 코드가 필요하고 URL을 수동으로 가져와야 합니다.

원래 의도는 페이지 텍스트를 통째로 LLM에게 던져 알아서 발라내게 하는 방식이었으나, 토큰 비용과 정확도를 끌어올리는 과정에서 본문 셀렉터를 직접 분석해 노이즈를 잘라낸 형태로 최적화되었습니다. 이로 인해 다음 두 가지 한계가 따라옵니다.

1. 토큰을 절약하고 성능을 끌어올리기 위해서는 **사람의 개입이 필수**입니다. "상세 정보 더 보기" 버튼을 클릭하는 로직, 사이트별 본문 셀렉터를 분석해 본문 부분만 파싱하는 코드 등 사람 손이 들어가야 합니다.
2. 그렇지 않고 `document.body.innerText`나 `<main>`·`<article>` 태그 전체를 가져오는 식으로 단순화하면, 원티드가 프론트엔드를 업데이트해 `__b9_L3` 같은 난독화 해시값이 바뀌는 순간 크롤러가 바로 깨집니다. 이 경우에도 **사람의 개입이 필수**입니다.

**Agent — 비전 LLM 에이전트 (자율 탐색 및 시드 수집기)**

화면을 시각으로 이해하고 도구를 사용해 행동합니다. 자연어 명령에서 시작합니다. URL 입력이 불필요한 대신 처리 시간이 길고 LLM 호출 비용이 누적됩니다.

이 시스템에서 비전 에이전트는 **탐색과 피드백 수집기** 역할을 함께 맡습니다.
등록된 신규 사이트의 첫 진입과 UI 변경 시 현재 화면을 분석하고 행동 후보를
제안합니다. 이후에도 전체 작업을 레시피로 고정하지 않고, 승격된 부모 단계만
재생하며 불확실하거나 실패한 구간은 다시 비전 탐색으로 폴백합니다. 현재
경험 기반 탐색에도 가변 카드와 상세 화면의 LLM 호출이 남아 있으므로 Reflex를 최종
운영 전체를 대체한 상태로 보지는 않습니다.

## Agent 아키텍처

```
사용자 자연어 질의
    ↓
[Chat UI → FastAPI]
    ↓
[ChatService — 실행 계측과 API 응답 변환]
    ↓
[Investigation LangGraph — 요청별 최상위 실행 관리자]
  대화 문맥 적재 → 요청 이해 → 중요한 모호성 확인
  → 답변 근거 정의 → SQLite 충분성 검사
  → 부족한 근거의 수집 행동계획
  → collect 노드에서 비전 작업자 실행
  → postprocess 노드에서 OCR 원문을 공고 스키마로 구조화
  → persist 노드에서 공고 UPSERT
  → 작업자 제출물과 레시피 후보를 별도 기록
  → SQLite 근거 재검사 → 최종 답변 및 job_id 인용 검증
    ↓
[WorkerExecutionService — 단일 로컬 작업자 직렬화]
  브라우저 준비·LangGraph 실행·정리를 하나의 잠금 범위로 실행
    ↓
[Vision Worker LangGraph]
  시작 상태 확인 → capture → transition → selection
  ├─ OCR 필요: ocr → transition → selection
  ├─ 결정론적 정책: execution
  ├─ Reflex 후보: reflex → execution 또는 reasoning
  └─ 의미 판단 필요: reasoning → execution
  execution → capture·reasoning·종료
    ↓
[SQLite 저장 → Investigation LangGraph 근거 재검사 → 최종 답변]

[비동기 RecipePromotionWorker]
  레시피 후보 → Critic 가지치기 → 활성 Reflex Recipe 승격
```

지휘자는 `agent/graph/investigation_workflow.py`의 LangGraph로 실행됩니다. 확인 질문이 남아 있으면 DB와 브라우저 도구를 호출하지 않고 `waiting_input`으로 중단하며, 사용자의 선택은 SQLite 조사 상태에 반영되어 다음 질문 또는 근거 검사 단계부터 재개됩니다. 사이트·날짜·개수·분석 목적은 실행 전에 확정된 행동계획에서만 수집 worker로 전달됩니다.

`agent/bootstrap.py`는 체크포인터, DB·대화·수집 포트, 모델 묶음과 비전 런타임을 생성해 그래프에 주입하는 조립 지점입니다. Investigation LangGraph가 대화 문맥, 노드 순서, DB 조회·저장 시점, 분기, 중단·재개와 상태 갱신을 담당합니다. 실제 SQL, 브라우저와 OCR 자원 수명은 주입된 어댑터가 담당합니다. Vision 노드는 전역 런타임 조회 없이 LangGraph `Runtime` 문맥으로 전달된 `VisionWorkerRuntime`을 사용합니다.

작업자 상태는 `request`, `observation`, `decision`, `transition`, `replay`, `collection`, `lifecycle`의 7개 책임 구역으로 나뉩니다. 노드는 자신이 변경한 구역만 `WorkerStateUpdate`로 반환하고 LangGraph reducer가 해당 구역을 병합합니다. 행동 실행은 불변 입력과 가변 결과를 분리하고, 결과 상태에서 최초 입력과 달라진 구역만 반환합니다.

Realtime/Vision 경로는 DOM이나 Playwright selector를 사용하지 않습니다. 전환 검증, 상세 OCR 누적, 카드 큐, Reflex 재생은 `agent/runtime/`에 분리되어 화면 서명·OCR 마커·좌표비율만 사용합니다. 최초 검색 결과에서는 경량 모델이 수집할 카드를 한 번 고르고, 상세에서 돌아온 뒤에는 목록 pHash와 저장 좌표로 큐의 다음 카드를 실행합니다. 큐를 모두 처리한 뒤 추가 탐색이 필요할 때만 일반 화면 추론이 새 큐를 만듭니다. Investigation LangGraph가 요청과 저장 흐름을 지휘하고, `agent/application/`은 상세 정제·DB 작업·비전 실행 어댑터를 제공하며 `agent/bootstrap.py`는 자원 수명주기만 조립합니다.

운영 경로는 지연 초기화(lazy initialization)를 적용합니다. DB 질의와 웹 Q&A 서버는 비전 엔진, YOLO 모델, 물리 GUI 제어 도구를 import 시점에 초기화하지 않고, 실시간 수집이 실제로 필요할 때만 비전 파이프라인을 준비합니다. PaddleOCR subprocess는 작업 동안 계속 재사용하고, 요청 timeout이나 worker 오류가 발생할 때만 재시작합니다. OCR 입력 최대 변은 1152로 제한합니다.

## 진행 상황

- [x] 프로젝트 셋업
- [x] Phase 1: Classic 시스템 베이스라인
  - [x] 원티드 URL 입력 기반 추출
  - [x] 본문 셀렉터 기반 영역 추출 및 상세 정보 더 보기 클릭
  - [x] Gemini 구조화 출력 기반 LLM 정형화 및 SQLite 저장
  - [x] LLM 출력 JSON 모드 및 타입 정규화 (string ↔ list 자동 변환)
  - [x] 사이트별 어댑터 패턴 및 URL 디스패처 (`classic/automation/sites/`)
  - [x] Classic 기준선 3개 사이트 구현 (원티드, 잡코리아, 로켓펀치)

- [x] Phase 2: 비전 및 물리 제어 엔진 기반 에이전트 도구 구축
  - [x] 1. 지표 및 에러 추적 세팅
    - [x] structlog: 소요 시간 등 성능 벤치마크용 JSON 포맷 로깅
  - [x] 2. 백그라운드 엔진 스크립트 (LLM이 직접 호출하지 않는 내부 엔진 계층)
    - [x] Perception: mss 브라우저 캡처, CV 로딩 대기, OCR, pHash를 분리한 화면 인식 파이프라인
    - [x] Loading Wait: 메모리 프레임의 변화·안정성과 회색 저정보 화면을 함께 확인한 뒤 최종 화면만 저장
    - [x] Security: .env 기반 자격증명 관리 시스템
  - [x] 3. 순수 파이썬 좌표 검증 테스트
    - [x] LLM 연동 전 캡처 화면의 마커 좌표와 실제 마우스 클릭 좌표가 어긋나지 않는지 하드코딩으로 1차 확인
  - [x] 4. 에이전트 상태 관리
    - [x] 스크롤이나 클릭 동작의 성공 여부를 비교할 수 있도록 최근 2장의 전후 마커 이미지 보관
    - [x] 파싱된 UI 요소 텍스트 및 Action History 정의
  - [x] 5. LLM이 호출하는 물리 행동 도구 (에이전트의 외부 인터페이스)
    - [x] click_marker: 마커 ID의 절대 좌표 계산 후 PyAutoGUI 물리적 클릭
    - [x] type_in_marker: 한글 씹힘 현상 방지를 위해 pyperclip 모듈을 활용한 클립보드 복사 후 붙여넣기 물리 타이핑
    - [x] scroll: OS 마우스 휠 스크롤
    - [x] press_key: Enter, ESC 등 특수키 입력
    - [x] finish_task: 수집 완료 시 데이터를 반환하며 루프 강제 종료
    - [x] Action Wrapper: 도구 실행 시 성능 로깅 및 안정화 대기 로직 자동 주입

- [x] Phase 3: LangGraph 지휘자 워크플로우 구성 (비전 통합은 Phase 3.5에서 본격 구현)
  - [x] 1. 노드 설계 및 관찰 → 계획 → 행동 루프
    - [x] Perception Node: 시스템이 화면을 캡처하고 로컬 SoM(OmniParser + PaddleOCR)으로 UI 요소와 텍스트를 추출 후 상태 갱신
    - [x] Reasoning Node: 마킹 이미지와 압축된 UI 텍스트 컨텍스트를 Gemini 3.5 Flash에 전달해 다음 행동 도구 선택
    - [x] Action Node: 도구 실행 및 시스템 안정화 후 다시 Perception Node로 회귀
  - [x] 2. 모듈화 기반 서브 그래프 구축
    - [x] 향후 Phase 6의 라우터 확장을 고려하여 구조를 유연하게 분리
  - [x] 3. LangSmith 통합 트래킹
    - [x] 노드 간 궤적, 소요 토큰 수, 프롬프트 입출력 모니터링 적용 (Phase 8 학습 데이터의 원천)
  - [x] 4. 단일 시나리오 E2E 통합 테스트
    - [x] 바탕화면에서 브라우저 조작 및 검색 화면 이동 검증 ("원티드에서 데이터 분석가 신입 공고 검색해줘" 명령)

- [x] Phase 3.5: OmniParser(Set-of-Marks) 로컬 파이프라인 실제 구현 및 순수 비전 본문 JSON 추출
  - [x] 1. 종속성 패키지 설치 (`ultralytics`, `paddleocr`) 및 로컬 OCR 연동 확인
  - [x] 2. OmniParser 공식 YOLOv8 모델 가중치 자동 다운로드 유틸 개발
  - [x] 3. OCR 경계 분리 (`paddle_ocr.py` 문자 검출, `omni_parser.py` 아이콘 검출, `ocr_engine.py` 마커 합성)
  - [x] 4. `perception.py` 리팩토링 및 SoM 연동 (마킹 이미지 주입 및 마커 ID 좌표 매핑 디코딩)
  - [x] 5. VLM 프롬프트 최적화 (멀티모달 SoM 마크 이미지 주입 및 의사결정 프롬프트 팝업/모달 차단 조치 추가)
  - [x] 6. E2E 본문 추출 통합 테스트 검증 및 벤치마크
    - [x] 원티드 데이터 분석가 검색 이동 성공 및 듀얼 모니터 좌표 매핑 속도 검증
    - [x] **[핵심 목적]** 개별 채용공고 카드 클릭 상세 진입 ➡️ "상세 정보 더 보기" 클릭 본문 확장 ➡️ 스크롤을 통한 화면 전체 텍스트 판독 ➡️ 주요업무, 자격요건, 우대사항, 혜택 항목별 구조화된 JSON 본문 데이터 최종 추출 완료 및 파일 저장 검증

- [x] Phase 4: 등록 사이트 간 화면 기반 런타임 재사용 및 본문 비교 검증
  - [x] 1. 로그인 불필요 환경에 대응하는 워크넷 접속 및 검색 추출
  - [x] 2. 잡코리아 등 화면 구조가 다른 사이트에 같은 비전 도구 계약 적용
  - [x] 3. Realtime 경로는 사이트별 DOM 파싱 코드 대신 등록 프로필과 공통 화면 작업자를 사용
  - [x] 4. **[추가]** 검색 결과 채용 공고 카드를 클릭하여 상세 페이지(본문)로 이동한 후, 화면 내 본문 텍스트를 판독·추출하여 파일로 저장
  - [x] 5. **[추가]** 저장된 본문 파일과 실제 사이트의 원문(Ground Truth) 텍스트를 텍스트 유사도 및 차이(Diff) 분석을 통해 정밀 검증하는 프로세스 구축

- [x] Phase 5: Classic 대 Agent 벤치마크 실험 및 본문 정합성 비교 데모
  - [x] 1. 성공률 및 완료 소요 시간에 대한 Structlog 데이터 기반 정량 비교
  - [x] 2. **[추가]** Classic 시스템이 수집한 공고 원문 파일과 Agent가 저장한 본문 텍스트 파일 간의 텍스트 매칭 정확도 및 누락률 정량 비교
  - [x] 3. LangSmith 데이터 기반 에이전트 오류 자가 복구율 분석 및 LangGraph `recursion_limit` 60으로 완화 조정
  - [x] 4. 토큰 사용량 기반 비용 산출 및 로컬 모델 메모리 부족(OOM)으로 발생하던 500 에러를 대비하기 위한 Gemini API 텍스트 추론 경로 추가 (하이브리드 추론 구조로 전환)

- [x] Phase 6: 수집 데이터 전처리 및 DB 적재 신뢰성 강화
  - [x] 1. 고정밀 텍스트 전처리 엔진 구현 (`preprocessor.py`)
    - [x] OCR 텍스트 내 불필요 개행, 특수 기호, 마커 잔영(`[id]`) 제거 및 목록 필드 정규화
    - [x] LLM이 리스트 필드를 단일 문자열 또는 JSON 문자열로 반환해도 안전하게 리스트로 정규화
    - [x] `3년 이상 ~ 7년 이하` 같은 경력 범위 표현 파싱 보강
  - [x] 2. 수집 필드 확장 스키마 설계 및 마이그레이션 (`jd_schema.py`, `database.py`)
    - [x] `source_platform`, `raw_ocr_text`, `content_hash`, `experience_min/max/text` 필드 및 인덱스 동적 추가
  - [x] 3. SQLite DB 이중 중복 방지 적재
    - [x] 동일 URL 또는 동일 content_hash 감지 시 자동 `UPDATE` 처리하는 UPSERT 파이프라인 구축
  - [x] 4. E2E 데이터 파이프라인 정합성 최종 검증 (`test_db_persistence.py`)
    - [x] 실제 수집 데이터 기반 전처리·DB 연동 후 적재 검증 패키지 자동 테스트 통과

- [x] Phase 7: 조사 계획 기반 DB 근거 조회와 지휘자 통합
  - [x] 1. 사용자 질문을 조사 계약으로 변환
    - [x] 직무 범위, 기술 조건, 사이트 검색어, 필요한 근거를 서로 다른 필드로 분리
    - [x] LLM이 임의 SQL이나 동의어 목록을 만들지 않고 검색 의미 사전의 개념 키를 사용
  - [x] 2. 구조화된 SQLite 근거 검사
    - [x] 직무 계층과 필수·우대·언급 기술 조건을 결정론적으로 조회
    - [x] 사전에서 확정할 수 없는 의미 조건만 LLM 검증으로 전달
    - [x] DB 근거가 부족할 때만 비전 수집 단계를 계획
  - [x] 3. 답변 생성 및 출처 검증
    - [x] 검증된 공고만 구조화 문서로 답변 모델에 제공
    - [x] 답변의 `[job_id:N]`이 실제 근거 문서에 포함되는지 검사
  - [x] 4. 정적 검색 의미 사전
      - [x] 버전 관리되는 한국어 업무 영역·직무·기술 별칭 적재
      - [x] 업무 영역부터 직무군까지 DB 공고 수와 사전 직무 수를 구분한 단계형 질문
      - [x] 미등록 직무는 선택 영역의 하위 개념만 LLM에 제공하고 현재 요청에서 확인

- [ ] **Phase 8: 피드백 루프 기반 Reflex Recipe 승격 (현재 단계)**

  > 현재 진행 중인 핵심 단계. 자율탐색은 행동을 선택할 때 `fixed / parameterized / reasoning` 재사용 방식과 입력 슬롯을 함께 기록한다. Critic은 실행 내용을 새로 만들거나 수정하지 않고 실패·불안정 행동만 제거한다. `자율탐색 후보 ∩ Critic 유지 ∩ 화면 증거 통과`에 해당하는 기록을 `이전 상태 + 행동 묶음 + 도착 상태` 전이로 저장한다. 경로는 한 번 선택한 뒤 도착 상태가 확인될 때만 다음 전이로 진행하고, 실패하면 다시 자율탐색으로 폴백한다.

  - [x] 1. 제안 로그 포맷 정규화
    - [x] LLM/VLM이 선택한 행동, 대상 마커, target_label, component, 재사용 방식, 선택 이유(`reason`)를 기록
    - [x] 검색어처럼 실행마다 바뀌는 값은 도구 호출의 `slot_name`과 레시피 `slot_refs`로 보존
  - [x] 2. 전후 관찰(Observer) 파이프라인 구축
    - [x] 화면 변경 행동과 다음 OCR·스크린샷을 같은 `action_events` 항목으로 연결
    - [x] OpenCV 연속 프레임 비교로 화면 변화 시작과 렌더링 안정화를 판단하고, pHash는 저장 상태 확인에 사용
    - [ ] 같은 카드 반복 클릭, 화면 변화 없음, 팝업/승인창 개입 등 오염 신호 탐지
  - [x] 3. Critic 피드백 루프 추가
    - [x] success / partial / no_effect / error 라벨 부여
    - [ ] 공고 카드 클릭 후 상세 페이지 진입, 상세 수집 후 DB 적재, go_back 후 목록 복귀 같은 목표 전이를 기준으로 판단
  - [ ] 4. Recipe Memory 승격 정책
    - [ ] 브라우저 툴바, 시스템 대화상자, 광고/팝업처럼 사이트 고유 동작이 아닌 요소는 승격 금지
    - [x] 자율탐색이 각 단계를 `fixed / parameterized / reasoning`으로 제안하고 Critic은 유지/제거만 판정
    - [x] Critic이 행동·파라미터·슬롯·화면 역할·전환 조건을 덮어쓰지 못하도록 스키마 축소
    - [ ] 승인된 후보를 활성 Recipe Memory에 반영하는 수동/승인 정책 결정
  - [x] 5. 슬롯 기반 Reflex 실행기
    - [x] 검색어가 바뀌면 `query` 슬롯만 교체하고 고정 UI 절차는 유지
    - [x] 행동 후 저장된 도착 ROI 또는 화면 문맥을 확인하고, 불일치나 시간 초과 시 자율탐색으로 폴백
    - [x] 검색 결과의 현재 공고 제목은 동적 대상으로 취급하여 과거 제목을 재생하지 않고, 작업자가 미방문 카드를 선택
    - [x] 상위 N개 요청은 `최초 카드 큐 생성 → 상세 수집 → 목록 화면 확인 → 저장된 다음 카드 좌표 실행` 루프를 반복
  - [x] 6. pHash 기반 경험 탐색 정량 검증
    - [x] 화면 pHash, OCR anchor, target bbox 비율을 결합해 replay 후보를 검증
    - [x] `ios 개발자 공고 2개`에서 기존 자율 탐색과 동일한 2건 저장 확인
    - [x] reasoning 횟수 24회 → 9회, reasoning 시간 141.92초 → 64.94초 감소 확인
    - [x] pHash/OCR 검증 실패 시 무리하게 실행하지 않고 reasoning으로 폴백하는 로그 확인
    - [ ] 검색어/수집 개수 변경 시 파라미터만 바뀌고 고정 절차가 재사용되는지 추가 검증
    - [x] 사람인·잡코리아·고용24에서 같은 ROI 검증·폴백 구조가 유지되는지 검증

## 디렉토리 구조

```
L2C/
├── ARCHITECTURE.md     시스템 아키텍처 문서
├── README.md           프로젝트 메인 리드미
├── troubleshooting.md  트러블슈팅 가이드
├── requirements.txt    종속성 패키지 정의
├── .env.example        환경변수 예시 파일
│
├── classic/            전통 자동화 (베이스라인)
│   ├── automation/       Playwright DOM 파싱 및 사이트별 어댑터
│   └── extractor/        Gemini 구조화 출력 기반 텍스트 정형화
│
├── agent/              비전 LLM 에이전트
│   ├── application/      사용자 지휘·작업자 실행·상세 정제·DB 저장 서비스
│   ├── graph/            LangGraph 순서·분기와 조사·작업자 노드
│   ├── runtime/          상태 계약·비전 의존성·전환·상세 OCR·카드 큐 정책
│   ├── prompts/          조사·상세 정제·신뢰 경계 프롬프트
│   ├── tools/            화면 인식(Perception)·물리 제어·실시간 수집 도구
│   ├── sites/            지휘자용 사이트 프로필과 매뉴얼 JSON
│   ├── recipe/           Reflex Recipe 기록·매칭·재생 보조 모듈
│   ├── utils/            로깅 및 전처리 유틸리티
│   └── tests/            자동화 유닛/통합 테스트 (DB 영속성, 조사 계획 등)
│
├── shared/             공통 모듈
│   ├── db/               SQLite 데이터베이스 관리
│   └── schema/           Pydantic 스키마 정의
│
├── benchmark/          비교 벤치마크 스크립트 및 정합성 리포트
│
├── data/               수집 데이터 및 정합성 검증 비교 리포트
│   ├── screenshots/      에이전트 구동 중 캡처 화면
│   ├── jobs.db           수집 결과 SQLite DB 파일
│   └── *.md/*.json       정합성 비교 리포트 및 추출 결과 캐시
│
└── docs/               추가 설계 관련 문서
    ├── index.md             문서 탐색 진입점
    └── design_decisions.md  기술적 설계 결정
```

## 기술 스택

| 카테고리 | 기술 |
|---------|------|
| 언어·런타임 | Python 3.13.14 |
| Classic 브라우저 자동화 | Playwright DOM/selector |
| Realtime 브라우저 자동화 | 화면/OCR + PyAutoGUI 물리 입력 |
| UI 요소 검출 | OmniParser (Microsoft) |
| OCR | PaddlePaddle 3.3.1 + PaddleOCR 3.7 GPU subprocess 재사용 |
| 에이전트 워크플로우 | LangGraph 1.2 |
| 지휘자 모델 | Gemini 3.6 Flash |
| 비전 판단 모델 | Gemini 3.6 Flash |
| 경량 구조화 모델 | Gemini 3.5 Flash Lite |
| 실행자 텍스트 모델 | Gemini 경량 모델 |
| 검색 방식 | 검색 의미 사전 + 구조화 SQLite 근거 검사 |
| 궤적 트래킹 | LangSmith |
| 자격증명 보안 | .env (python-dotenv) |
| 저장소 | SQLite |

## 보안·법적 고려사항

- **자격증명**: 소스 코드에 직접 노출하지 않고 `.env` 파일에 보관하여 `.gitignore`로 관리합니다.
- **사용 범위**: 본인 계정에 한정하며, 학습 목적으로만 사용합니다.
- **사이트 약관**: 각 사이트의 이용약관을 확인하고 진행합니다.

## 빠른 시작

```powershell
git clone https://github.com/Dreamtreeme/L2C.git
cd L2C

# Windows 원클릭 설치
.\setup.cmd

# 테스트·벤치마크 도구까지 설치
.\setup.cmd -Development

# setup.cmd가 만든 .env에 GEMINI_API_KEY를 설정

# 분할된 비전 작업자 그래프는 VISION_AGENT_RECURSION_LIMIT=180 기본값으로 실행

# 로컬 Chat UI와 백엔드 실행
.\run.cmd

# Classic 방식 — URL 직접 입력
python -m classic.main extract https://www.wanted.co.kr/wd/350432

# 핵심 제품 계약 117개
.\scripts\test.cmd

# 과거 장애와 세부 기능 회귀
.\scripts\test.cmd agent\tests\regression -q

# 벤치마크와 품질 계산 검증
.\scripts\test.cmd agent\tests\evaluation -q

# 전체 232개
.\scripts\test.cmd agent\tests -q

# Realtime E2E: 로그, 구조화 요약, 선택적 LangSmith trace를 함께 생성
python -m benchmark.run_realtime_e2e --site wanted --search-keyword "iOS 개발자" --original-query "ios 개발자 공고 2개" --target-count 2 --count-mode explicit --scenario-id wanted-ios-2 --execution-mode experience_guided --log logs/e2e_wanted_ios2.log

# 구조화 summary의 p50/p95/max, Reflex, OCR 지표 확인
python -m benchmark.profile_reflex_trace logs/e2e_wanted_ios2.summary.json

# 포트폴리오 수집 성공률 30회 실행 계약 확인
python -m benchmark.run_regression_matrix --matrix benchmark/portfolio_collection_matrix.json --dry-run

# 자율 탐색·경험 기반 탐색 18회 짝 비교 계약 확인
python -m benchmark.run_regression_matrix --matrix benchmark/portfolio_reflex_matrix.json --dry-run

# E2E summary와 사람 판정표를 결합한 엄격 성공률 계산
python -m benchmark.manual_evaluation logs/portfolio/manual_evaluation.json

# 설치 용량과 현재 RAM·VRAM 기준값 측정
powershell -File scripts/measure_runtime_resources.ps1
```

`setup.cmd`는 디스크, NVIDIA 드라이버 580 이상과 VRAM 8GB 이상을 다운로드 전에 검사합니다. Python 3.13.14가 없으면 공식 python.org 설치 파일을 받아 SHA-256을 검증한 뒤 설치합니다. 이후 `.venv-app`과 `.venv-ocr`을 만들고 Chromium과 OmniParser·PaddleOCR 모델을 내려받은 뒤 실제 GPU 연산까지 검사합니다. 기본 설치에는 제품 런타임만 포함되며 `-Development`를 지정하면 테스트·벤치마크 의존성도 설치합니다. NVIDIA 드라이버는 하드웨어와 재부팅이 관련되므로 자동 설치하지 않습니다.

OmniParser/PyTorch와 PaddleOCR/PaddlePaddle의 CUDA 런타임은 서로 다른 환경에 둡니다. 따라서 한 프로세스 안에서 DLL과 import 순서를 맞추는 우회 코드가 필요하지 않습니다. 설치 항목을 선택적으로 생략해야 하는 개발 환경에서는 `scripts/setup_runtime.ps1`을 직접 사용할 수 있습니다.

고정 버전의 선택 근거와 GPU 실측 결과는 [`docs/runtime_compatibility.md`](./docs/runtime_compatibility.md)에 정리했습니다.

백엔드와 UI는 `127.0.0.1`에서만 실행됩니다. 설정에 외부 Host나 CORS 출처를 넣거나 원격 클라이언트가 직접 요청하면 시작 또는 요청 단계에서 거부합니다.

웹 화면 오른쪽 위의 활동 아이콘에서 최근 실행 상태를 확인할 수 있습니다.

E2E 요약은 `run_id`, 실행시간, 실패 단계, 단계별 시간, 모델별 토큰, 비용 추정, 수집 품질을 한 파일에 기록합니다. LangSmith를 활성화하면 같은 실행의 trace와 결정론적 feedback도 함께 전송합니다. 설정과 대시보드 기준은 [`docs/e2e_observability.md`](./docs/e2e_observability.md), 고정 행렬과 사람 판정 절차는 [`docs/portfolio_evaluation.md`](./docs/portfolio_evaluation.md)를 참고하세요. 기본 모델 단가는 [`config/model_pricing.json`](./config/model_pricing.json)에서 관리하며, 별도 가격표가 필요할 때만 `LLM_PRICING_FILE`로 덮어씁니다. 가격이 등록되지 않은 모델은 비용을 추정하지 않고 토큰 원시값과 `unpriced_models`에 남깁니다.

Windows Python을 WSL/Git Bash에서 직접 호출해 한글이나 이모지가 깨지는 경우에는 `python -X utf8 -m ...` 형태로 실행하세요.

## 검증 기준

수집 성공은 요청 개수 충족, 서로 다른 상세 URL, 필수 공고 필드, DB 저장과
답변의 `job_id` 인용을 함께 검사합니다. 검색 의도와 실제 업무의 의미 일치,
회사명·직무명·본문 정확도는 고정 판정표로 사람이 확인합니다.

자율 탐색과 경험 기반 탐색은 커밋, 모델, 설정, 사이트, 검색어와 목표 수가 같은
실행만 비교합니다. 두 실행의 수집 품질이 모두 통과한 경우에만 실행시간, 추론,
토큰, 비용과 Critic 승격 비용을 비교합니다.

현재 검증 환경은 Windows와 RTX 3080 10GB 한 대입니다. 사이트별 clean 반복
표본, 잘못된 클릭률, 신규 사이트 적용 공수와 실제 사용자 과업 평가는 아직
완료되지 않았습니다. 가변 공고 선택과 상세 본문 판단은 LLM을 사용하므로
경험 기반 탐색의 적중과 실행시간 개선도 실행마다 달라질 수 있습니다.

## 향후 작업

Phase 8은 현재 진행 중입니다. 원티드의 다른 검색어·수집 개수 일반화와
사람인·고용24의 짝 회귀까지 통과했으므로, 다음 작업은 사이트별 반복 표본을
늘려 성공률·잘못된 클릭률·실행시간 분포·Critic 비용을 함께 비교하는 것입니다.
잡코리아와 로켓펀치는 최근 실패 복구 뒤 새 기준의 짝 회귀가 남아 있습니다.
진행 중인 항목은 GitHub Issues에서 관리합니다.
