# L2C — LLM to Computer

> 자연어 명령으로 채용공고를 수집하는 에이전트 실험 프로젝트입니다.

## 프로젝트 의도

기존 자동화는 두 진영으로 나뉩니다.

**전통 Playwright 자동화**는 빠르고 저렴하지만, 사이트별 셀렉터와 도구 사용 순서를 사람이 직접 분석해야 합니다. UI가 바뀌면 사람이 다시 코드를 고쳐야 합니다.

**비전 LLM 에이전트**(browser-use 등)는 사이트별 사전 분석 없이 자연어 명령만으로 동작합니다. 다만 매 작업마다 LLM 호출 비용이 발생합니다.

이 프로젝트는 두 방식의 장점을 결합하는 구조를 실험합니다. 처음 보는 사이트나 화면은 비전 에이전트가 직접 탐색하고, 그 과정에서 행동 제안, 전후 화면 변화, 성공/실패 피드백을 누적합니다. 반복적으로 성공한 패턴만 Reflex Recipe로 승격하여 이후에는 필요한 파라미터만 바꿔 빠르게 실행합니다. **사이트별 사전 분석 없이 시작 가능하고, 사용량이 누적될수록 추론이 필요한 구간만 남아 운영 비용이 점진적으로 감소**하는 자동화 시스템을 목표로 합니다.

첫 번째 실험 도메인은 채용공고 수집이며, Phase 5의 정량 비교를 통해 비전 기반 자율 수집이 작동함을 확인했습니다. 현재는 Phase 8에서 자율탐색 기록을 반복탐색에 재사용해 reasoning 호출을 줄이는 구조를 검증 중입니다. 자세한 결과는 [`benchmark/jd_comparison_report.md`](./benchmark/jd_comparison_report.md)와 [`troubleshooting.md`](./troubleshooting.md)에서 확인할 수 있습니다. 설계·실험·운영 문서 전체는 [`docs/index.md`](./docs/index.md)에서 찾을 수 있습니다.

## 현재 단계

현재 위치는 **Phase 8: 피드백 루프 기반 Reflex Recipe 승격**입니다.

완성된 것은 전체 범용 replay가 아니라, **고정 가능한 부모 경로를 pHash/OCR 서명으로 재사용하고 마지막 가변 판단은 LLM에게 남기는 하이브리드 반복탐색**입니다. `ios 개발자 공고 2개` 원티드 반복탐색에서 기존 자율탐색과 동일하게 2건을 저장하면서 reasoning 횟수를 24회에서 9회로 줄였습니다.

현재 결론은 다음과 같습니다.

- pHash 기반 Reflex는 원티드의 유사 채용공고 수집 흐름에서 reasoning 감소 효과를 확인했다.
- 검색 진입, 일부 목록 이동, 반복 가능한 UI 조작처럼 안정적인 상위 경로는 replay 후보로 적합하다.
- 공고 상세 본문, 현재 검색 결과의 카드 선택, 예외 화면처럼 매번 달라지는 마지막 판단은 아직 LLM reasoning이 필요하다.
- 다음 단계는 다른 검색어, 다른 수집 개수, 다른 사이트에서 같은 구조가 유지되는지 검증하는 것이다.

## 차별점

기존 자동화:
```python
extract("https://www.wanted.co.kr/wd/123456")  # URL 수동 입력 + 사이트별 코드
```

이 시스템:
```python
agent.run("데이터 분석가 신입 공고 모아줘")  # 사이트 자동 탐색
```

| 단계 | 전통 Playwright | 비전 에이전트 (browser-use 등) | L2C |
|------|----------------|-----------------------------|-----|
| 초기 설정 | 사이트별 셀렉터·시퀀스 사전 분석 | 즉시 동작 | 즉시 동작 (비전 시드) |
| 신규 사이트 추가 | 코드 작성 | 자동 | 자동 |
| 반복 작업 비용 | 거의 무료 | LLM API 매 호출 비용 | 성공 패턴 승격 후 점진적 감소 |
| UI 변경 대응 | 사람이 코드 수정 | 자동 | 자동 (비전 재탐색 후 피드백 재승격) |
| 비용 곡선 | 평탄 (저렴) | 평탄 (비쌈) | 피드백 누적에 따라 우하향 |

비전 에이전트는 키보드 입력·마우스 클릭 같은 범용 행동만으로 모든 브라우저에서 동작합니다. 이 범용성을 유지하면서 비용 문제만 해소하는 것이 핵심 가설입니다.

## 두 시스템 비교

차이를 정량적으로 측정하기 위해 동일 작업을 두 방식으로 구현했습니다.

**Classic — 전통 Playwright 자동화 (베이스라인)**

Playwright로 DOM 구조를 직접 파싱합니다. 사이트별 마커와 셀렉터를 사전 정의해야 합니다. 빠르고 안정적이지만 사이트별 코드가 필요하고 URL을 수동으로 가져와야 합니다.

원래 의도는 페이지 텍스트를 통째로 LLM에게 던져 알아서 발라내게 하는 방식이었으나, 토큰 비용과 정확도를 끌어올리는 과정에서 본문 셀렉터를 직접 분석해 노이즈를 잘라낸 형태로 최적화되었습니다. 이로 인해 다음 두 가지 한계가 따라옵니다.

1. 토큰을 절약하고 성능을 끌어올리기 위해서는 **사람의 개입이 필수**입니다. "상세 정보 더 보기" 버튼을 클릭하는 로직, 사이트별 본문 셀렉터를 분석해 본문 부분만 파싱하는 코드 등 사람 손이 들어가야 합니다.
2. 그렇지 않고 `document.body.innerText`나 `<main>`·`<article>` 태그 전체를 가져오는 식으로 단순화하면, 원티드가 프론트엔드를 업데이트해 `__b9_L3` 같은 난독화 해시값이 바뀌는 순간 크롤러가 바로 깨집니다. 이 경우에도 **사람의 개입이 필수**입니다.

**Agent — 비전 LLM 에이전트 (자동 시드 수집기)**

화면을 시각으로 이해하고 도구를 사용해 행동합니다. 자연어 명령에서 시작합니다. URL 입력이 불필요한 대신 처리 시간이 길고 LLM 호출 비용이 누적됩니다. 정량 비교 결과는 [`benchmark/jd_comparison_report.md`](./benchmark/jd_comparison_report.md)에서 확인할 수 있습니다.

이 시스템에서 비전 에이전트는 **최종 운영 모드가 아니라 탐색과 피드백 수집기**입니다. 신규 사이트의 첫 진입과 UI 변경 시 페이지 구조를 분석하고 행동 후보를 제안합니다. 평상시 운영은 누적된 성공/실패 피드백으로 승격된 Reflex Recipe가 수행하고, 불확실하거나 실패한 경우에만 다시 비전 탐색으로 폴백하도록 설계됩니다 (Phase 8 참고).

## Agent 아키텍처

```
사용자 자연어 질의
    ↓
[Chat UI → FastAPI]
    ↓
[ChatService — 사용자 진입점의 단일 지휘자]
  요청 이해 → 중요한 모호성 확인 → 객관식 사용자 질문
  → 답변에 필요한 근거 정의 → SQLite 충분성 검사
  → 부족한 근거의 수집 행동계획 → 계획된 도구만 실행
  → 의미 조건·게시일 검증 → 최종 답변 및 job_id 인용 검증
    ↓
[CollectionService — 수집 실행 생명주기]
  Worker 실행 → 제출물 검토 → 승인 데이터 저장
    ↓
[WorkerExecutionService — 단일 로컬 작업자 직렬화]
  브라우저 준비·LangGraph 실행·정리를 하나의 잠금 범위로 실행
    ↓
[Vision Worker LangGraph]
  Perception: OmniParser + PaddleOCR + 화면 서명
  ├─ 카드 큐/상세 정책 hit: 결정론적 행동 실행
  ├─ Reflex hit: ROI pHash와 마커 비율 검증 후 행동 실행
  └─ miss: Gemini가 현재 화면의 다음 행동 판단
  Action: PyAutoGUI 기반 클릭·입력·스크롤·뒤로가기
    ↓
[제출물 검토 → 승인 데이터 SQLite 저장 → ChatService 답변]
```

지휘자는 `agent/graph/investigation_workflow.py`의 LangGraph로 실행됩니다. 확인 질문이 남아 있으면 DB와 브라우저 도구를 호출하지 않고 `waiting_input`으로 중단하며, 사용자의 선택은 SQLite 조사 상태에 반영되어 다음 질문 또는 근거 검사 단계부터 재개됩니다. 사이트·날짜·개수·분석 목적은 실행 전에 확정된 행동계획에서만 수집 worker로 전달됩니다.

Realtime/Vision 경로는 DOM이나 Playwright selector를 사용하지 않습니다. 전환 검증, 상세 OCR 누적, 카드 큐, Reflex 재생은 `agent/runtime/`에 분리되어 화면 서명·OCR 마커·좌표비율만 사용합니다. 사용자 지휘, 작업자 실행, 상세 정제, DB 적재는 `agent/application/` 서비스가 담당합니다.

운영 경로는 지연 초기화(lazy initialization)를 적용합니다. DB 질의와 웹 Q&A 서버는 비전 엔진, YOLO 모델, 물리 GUI 제어 도구를 import 시점에 초기화하지 않고, 실시간 수집이 실제로 필요할 때만 비전 파이프라인을 준비합니다. PaddleOCR subprocess는 작업 동안 계속 재사용하고, 요청 timeout이나 worker 오류가 발생할 때만 재시작합니다. OCR 입력 최대 변은 1152로 제한합니다.

## 진행 상황

- [x] 프로젝트 셋업
- [x] Phase 1: Classic 시스템 베이스라인
  - [x] 원티드 URL 입력 기반 추출
  - [x] 본문 셀렉터 기반 영역 추출 및 상세 정보 더 보기 클릭
  - [x] Qwen (Ollama) 기반 LLM 정형화 및 SQLite 저장
  - [x] LLM 출력 JSON 모드 및 타입 정규화 (string ↔ list 자동 변환)
  - [x] 사이트별 어댑터 패턴 및 URL 디스패처 (`classic/automation/sites/`)
  - [x] 5개 주요 사이트 안정화 (원티드, 잡코리아, 사람인, 워크넷, 로켓펀치)

- [x] Phase 2: 비전 및 물리 제어 엔진 기반 에이전트 도구 구축
  - [x] 1. 지표 및 에러 추적 세팅
    - [x] sentry-sdk: 시스템 크래시 및 좌표 이탈 등 치명적 에러 캡처
    - [x] structlog: 소요 시간 등 성능 벤치마크용 JSON 포맷 로깅
  - [x] 2. 백그라운드 엔진 스크립트 (LLM이 직접 호출하지 않는 내부 엔진 계층)
    - [x] Perception: mss 모듈 활용 브라우저 영역 검출 및 OmniParser + PaddleOCR 기반의 로컬 SoM(Set-of-Marks) 파이프라인 마커 합성 구현
    - [x] Wait Stable: 무한 대기 버그를 막기 위해 픽셀 오차율 1퍼센트 이하 조건 및 최대 대기 시간 5초를 적용한 시각적 화면 안정화 대기
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
  - [x] 3. `som_engine.py` 구현 (OmniParser + PaddleOCR을 통한 요소 검출, 중복 제어 NMS, 마크 이미지 합성 및 좌표 매핑)
  - [x] 4. `perception.py` 리팩토링 및 SoM 연동 (마킹 이미지 주입 및 마커 ID 좌표 매핑 디코딩)
  - [x] 5. VLM 프롬프트 최적화 (멀티모달 SoM 마크 이미지 주입 및 의사결정 프롬프트 팝업/모달 차단 조치 추가)
  - [x] 6. E2E 본문 추출 통합 테스트 검증 및 벤치마크
    - [x] 원티드 데이터 분석가 검색 이동 성공 및 듀얼 모니터 좌표 매핑 속도 검증
    - [x] **[핵심 목적]** 개별 채용공고 카드 클릭 상세 진입 ➡️ "상세 정보 더 보기" 클릭 본문 확장 ➡️ 스크롤을 통한 화면 전체 텍스트 판독 ➡️ 주요업무, 자격요건, 우대사항, 혜택 항목별 구조화된 JSON 본문 데이터 최종 추출 완료 및 파일 저장 검증

- [x] Phase 4: 사이트 제로샷 일반화 검증 및 채용공고 본문 비교 검증
  - [x] 1. 로그인 불필요 환경에 대응하는 워크넷 접속 및 검색 추출
  - [x] 2. 잡코리아 등 DOM 구조가 다른 사이트 적용 테스트 (어댑터 구조 수정 및 CSS 셀렉터 최적화 완료)
  - [x] 3. 새로운 사이트 추가 시 Classic 방식처럼 별도의 파싱 코드 작성이 필요 없음을 증명
  - [x] 4. **[추가]** 검색 결과 채용 공고 카드를 클릭하여 상세 페이지(본문)로 이동한 후, 화면 내 본문 텍스트를 판독·추출하여 파일로 저장
  - [x] 5. **[추가]** 저장된 본문 파일과 실제 사이트의 원문(Ground Truth) 텍스트를 텍스트 유사도 및 차이(Diff) 분석을 통해 정밀 검증하는 프로세스 구축

- [x] Phase 5: Classic 대 Agent 벤치마크 실험 및 본문 정합성 비교 데모
  - [x] 1. 성공률 및 완료 소요 시간에 대한 Structlog 데이터 기반 정량 비교
  - [x] 2. **[추가]** Classic 시스템이 수집한 공고 원문 파일과 Agent가 저장한 본문 텍스트 파일 간의 텍스트 매칭 정확도 및 누락률 정량 비교
  - [x] 3. LangSmith 데이터 기반 에이전트 오류 자가 복구율 분석 및 LangGraph `recursion_limit` 60으로 완화 조정
  - [x] 4. 토큰 사용량 기반 비용 산출 및 로컬 모델 메모리 부족(OOM)으로 발생하던 500 에러를 대비하기 위한 Gemini API 텍스트 추론 경로 추가 (하이브리드 추론 구조로 전환)
  - [x] 5. 최종 결과 보고서 작성 및 모니터 위 물리 마우스 자율 조작 벤치마크 리포트 배포 ([`benchmark/jd_comparison_report.md`](./benchmark/jd_comparison_report.md))

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
  - [x] 4. 검색 의미 사전 운영
      - [x] 6개 업무 영역·로컬 직무군·O*NET 세부 직업·소프트웨어 기술 어휘 적재
      - [x] 업무 영역부터 직무군까지 DB 공고 수와 사전 직무 수를 구분한 단계형 질문
      - [x] 선택 영역 하위 후보만 이용한 미등록 직무 의미 확인과 사용자 승인 별칭 승격
      - [x] 미등록 기술을 후보로 모으고 검토 후 별칭 또는 새 개념으로 승인

- [ ] **Phase 8: 피드백 루프 기반 Reflex Recipe 승격 (현재 단계)**

  > 현재 진행 중인 핵심 단계. 비전 에이전트가 처음부터 정답 스크립트를 만들도록 제약하지 않고, 탐색 중 생성한 행동 제안과 실제 실행 결과를 관찰해 성공 패턴만 Reflex Recipe로 승격한다. Reflex는 LLM 추론 없이 빠르게 실행하되, 검색어·수집 개수·사이트 범위처럼 바뀌는 값은 지휘자가 파라미터로 주입한다. 실패하거나 확신이 낮은 상황에서는 다시 비전 탐색으로 폴백한다.

  - [x] 1. 제안 로그 포맷 정규화
    - [x] LLM/VLM이 선택한 행동, 대상 마커, target_label, component 후보, parameter 후보, 선택 이유(`reason`)를 기록
    - [x] 검색어(`query`)와 수집 개수(`target_count`)처럼 실행마다 바뀔 수 있는 후보 값을 별도 필드로 보존
  - [x] 2. 전후 관찰(Observer) 파이프라인 구축
    - [x] 화면 변경 action과 다음 OCR·스크린샷을 `transition_observations`로 연결
    - [x] 로딩 중 관찰은 같은 action seq에 누적하고, 준비 완료 여부는 전환 계약으로 판정
    - [ ] 같은 카드 반복 클릭, 화면 변화 없음, 팝업/승인창 개입 등 오염 신호 탐지
  - [x] 3. Critic 피드백 루프 추가
    - [x] success / partial / wrong_target / no_effect / loop_risk 라벨 부여
    - [ ] 공고 카드 클릭 후 상세 페이지 진입, 상세 수집 후 DB 적재, go_back 후 목록 복귀 같은 목표 전이를 기준으로 판단
  - [ ] 4. Recipe Memory 승격 정책
    - [ ] 같은 사이트·페이지 역할·작업 유형에서 반복 성공한 패턴만 confidence 상승
    - [ ] 실패 패턴은 negative example로 저장하고 Reflex 후보에서 제외
    - [ ] Codex 승인창, 브라우저 툴바, 광고/팝업처럼 사이트 고유 동작이 아닌 요소는 승격 금지
    - [x] Critic이 각 단계를 `fixed / parameterized / reasoning`으로 분류하도록 후보 검토 결과에 기록
    - [x] 검증 전 자동 활성 레시피 쓰기는 비활성화하고 후보 저장/검토만 유지
    - [ ] 승인된 후보를 활성 Recipe Memory에 반영하는 수동/승인 정책 결정
  - [x] 5. 슬롯 기반 Reflex 실행기
    - [x] 검색어가 바뀌면 `query` 슬롯만 교체하고 고정 UI 절차는 유지
    - [x] 행동 후 `common_ready_cues + outcomes`가 충족될 때까지 재관찰하고, 시간 초과 시 Explore로 폴백
    - [x] 검색 결과의 현재 공고 제목은 동적 대상으로 취급하여 과거 제목을 재생하지 않고, 작업자가 미방문 카드를 선택
    - [x] 상위 N개 요청은 `target_count`와 방문 이력을 유지하며 `현재 카드 선택 → 상세 수집 → 필요 시 go_back` 루프를 반복
    - [ ] Reflex는 높은 confidence 패턴만 실행하고, 불확실하면 추론하지 않고 Explore로 폴백
  - [x] 6. pHash 기반 반복탐색 정량 검증
    - [x] 화면 pHash, OCR anchor, target bbox 비율을 결합해 replay 후보를 검증
    - [x] `ios 개발자 공고 2개`에서 기존 자율탐색과 동일한 2건 저장 확인
    - [x] reasoning 횟수 24회 → 9회, reasoning 시간 141.92초 → 64.94초 감소 확인
    - [x] pHash/OCR 검증 실패 시 무리하게 실행하지 않고 reasoning으로 폴백하는 로그 확인
    - [ ] 검색어/수집 개수 변경 시 파라미터만 바뀌고 고정 절차가 재사용되는지 추가 검증
    - [ ] 원티드 외 사이트에서 같은 구조가 유지되는지 검증

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
│   └── extractor/        텍스트 구조화 및 정형화 (Gemini/Ollama)
│
├── agent/              비전 LLM 에이전트
│   ├── application/      사용자 지휘·작업자 실행·상세 정제·DB 저장 서비스
│   ├── graph/            LangGraph 연결, 상태, 도구 스키마, 핵심 노드
│   ├── runtime/          전환·상세 OCR·카드 큐·Reflex 결정론적 런타임
│   ├── prompts/          지휘자 프롬프트 (commander)
│   ├── credentials/      .env 자격증명 관리 매니저
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
├── docs/               추가 설계 관련 문서
│   ├── design_decisions.md  기술적 설계 결정
│   └── lessons_learned.md   트러블슈팅 및 교훈
│
└── scratch/            임시 테스트 및 수동 검증용 샌드박스 스크립트 (GUI 테스트, 개별 워크플로우 시뮬레이션 등)
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
| 실행자 텍스트 모델 | Qwen (Ollama) |
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

# setup.cmd가 만든 .env에 GEMINI_API_KEY를 설정
.\.venv-app\Scripts\Activate.ps1

# 분할된 비전 작업자 그래프는 VISION_AGENT_RECURSION_LIMIT=180 기본값으로 실행

# Classic 방식 — URL 직접 입력
python -m classic.main extract https://www.wanted.co.kr/wd/350432

# Agent 방식 — 자연어 명령 (수집)
python -m agent.main "ai 엔지니어 신입 공고 모아줘"

# Agent 방식 — 자연어 질의 (적재 DB SQLite 조회)
python -m agent.main "수집된 공고 중 신입 가능한 곳 알려줘"

# 웹 Q&A 서버
uvicorn agent.web_server:app --reload

# 기본 회귀 테스트 (외부 API와 물리 브라우저 E2E 제외)
python -m pytest -q

# 실제 외부 API 테스트를 명시적으로 포함
python -m pytest -m external -q

# 두 방식 비교. 실패 시 정상 데이터를 임의로 채우지 않음
python -m benchmark.run_compare_jd

# Realtime E2E: 로그, 구조화 요약, 선택적 LangSmith trace를 함께 생성
python -m benchmark.run_realtime_e2e --site wanted --query "ios 개발자 공고 2개" --target-count 2 --count-mode explicit --scenario-id wanted-ios-2 --run-mode warm --log logs/e2e_wanted_ios2.log

# 구조화 summary의 p50/p95/max, Reflex, OCR 지표 확인
python -m benchmark.profile_reflex_trace logs/e2e_wanted_ios2.summary.json
```

`setup.cmd`는 디스크와 NVIDIA GPU를 먼저 검사하고, Python 3.13.14가 없으면 공식 python.org 설치 파일을 받아 SHA-256을 검증한 뒤 설치합니다. 이후 `.venv-app`과 `.venv-ocr`을 만들고 Chromium과 OmniParser·PaddleOCR 모델을 내려받은 뒤 실제 GPU 연산까지 검사합니다. NVIDIA 드라이버는 하드웨어와 재부팅이 관련되므로 자동 설치하지 않습니다.

OmniParser/PyTorch와 PaddleOCR/PaddlePaddle의 CUDA 런타임은 서로 다른 환경에 둡니다. 따라서 한 프로세스 안에서 DLL과 import 순서를 맞추는 우회 코드가 필요하지 않습니다. 설치 항목을 선택적으로 생략해야 하는 개발 환경에서는 `scripts/setup_runtime.ps1`을 직접 사용할 수 있습니다.

고정 버전의 선택 근거와 GPU 실측 결과는 [`docs/runtime_compatibility.md`](./docs/runtime_compatibility.md)에 정리했습니다.

웹 화면 오른쪽 위의 활동 아이콘에서 최근 실행 상태와 저장 현황을 확인할 수 있습니다. 만료 항목 정리는 먼저 삭제 후보와 예상 용량을 미리 계산하고, 사용자가 확인한 경우에만 오래된 로그·미참조 화면 산출물·감사 이력을 삭제합니다. 현재 공고, 활성 레시피, 공고가 참조하는 화면 파일은 정리 대상에서 제외합니다.

기본 보존 기간은 로그 30일, 화면 산출물과 감사 이력 90일, 공고 변경 이력 180일입니다. 공고별 최신 변경 이력은 기간과 관계없이 5개를 남기며, `RETENTION_*` 환경변수로 기준을 조정할 수 있습니다.

E2E 요약은 `run_id`, 실행시간, 실패 단계, 단계별 시간, 모델별 토큰, 선택적 비용 추정, 수집 품질을 한 파일에 기록합니다. LangSmith를 활성화하면 같은 실행의 trace와 결정론적 feedback도 함께 전송합니다. 설정과 대시보드 기준은 [`docs/e2e_observability.md`](./docs/e2e_observability.md)를 참고하세요. 모델 단가는 `config/model_pricing.example.json` 형식을 참고해 별도 파일로 관리하고 `LLM_PRICING_FILE`에 지정합니다. 가격표가 없으면 부정확한 비용을 만들지 않고 토큰 원시값만 보존합니다.

Windows Python을 WSL/Git Bash에서 직접 호출해 한글이나 이모지가 깨지는 경우에는 `python -X utf8 -m ...` 형태로 실행하세요.

## 향후 작업

Phase 8은 현재 진행 중입니다. 다음 작업은 pHash 기반 반복탐색을 원티드의 다른 검색어·수집 개수로 확장 검증하고, 이후 잡코리아·사람인·워크넷·로켓펀치에서도 같은 부모 경로 replay 구조가 유지되는지 확인하는 것입니다. 진행 중이거나 검토 중인 항목은 [Issues](../../issues)에서 확인할 수 있습니다.

---

지속적으로 확장 중인 실험 프로젝트입니다.
