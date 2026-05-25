# L2C — LLM to Computer

> 자연어 명령으로 채용공고를 수집하는 에이전트 실험 프로젝트입니다.

## 프로젝트 의도

기존 자동화는 두 진영으로 나뉩니다.

**전통 Playwright 자동화**는 빠르고 저렴하지만, 사이트별 셀렉터와 도구 사용 순서를 사람이 직접 분석해야 합니다. UI가 바뀌면 사람이 다시 코드를 고쳐야 합니다.

**비전 LLM 에이전트**(browser-use 등)는 사이트별 사전 분석 없이 자연어 명령만으로 동작합니다. 다만 매 작업마다 LLM 호출 비용이 발생합니다.

이 프로젝트는 두 방식의 장점을 결합하는 구조를 실험합니다. 비전 에이전트로 자동 시드 실행을 수행하고, 누적된 도구 호출 로그에서 결정론적 시퀀스를 추출해 Playwright 스크립트로 변환합니다. **사이트별 사전 분석 없이 시작 가능하고, 사용량이 누적될수록 LLM 호출이 사라져 운영 비용이 0에 수렴**하는 자동화 시스템을 목표로 합니다.

첫 번째 실험 도메인은 채용공고 수집이며, Phase 5의 정량 비교를 통해 비전 기반 자율 수집이 작동함을 확인했습니다. 자세한 결과는 [`data/jd_comparison_report.md`](./data/jd_comparison_report.md)에서 확인할 수 있습니다.

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
| 반복 작업 비용 | 거의 무료 | LLM API 매 호출 비용 | 로그 누적 후 거의 무료 |
| UI 변경 대응 | 사람이 코드 수정 | 자동 | 자동 (비전 재가동 후 시퀀스 재추출) |
| 비용 곡선 | 평탄 (저렴) | 평탄 (비쌈) | 우하향 (점진적 감소) |

비전 에이전트는 키보드 입력·마우스 클릭 같은 범용 행동만으로 모든 브라우저에서 동작합니다. 이 범용성을 유지하면서 비용 문제만 해소하는 것이 핵심 가설입니다.

## 두 시스템 비교

차이를 정량적으로 측정하기 위해 동일 작업을 두 방식으로 구현했습니다.

**Classic — 전통 Playwright 자동화 (베이스라인)**

Playwright로 DOM 구조를 직접 파싱합니다. 사이트별 마커와 셀렉터를 사전 정의해야 합니다. 빠르고 안정적이지만 사이트별 코드가 필요하고 URL을 수동으로 가져와야 합니다.

원래 의도는 페이지 텍스트를 통째로 LLM에게 던져 알아서 발라내게 하는 방식이었으나, 토큰 비용과 정확도를 끌어올리는 과정에서 본문 셀렉터를 직접 분석해 노이즈를 잘라낸 형태로 최적화되었습니다. 이로 인해 다음 두 가지 한계가 따라옵니다.

1. 토큰을 절약하고 성능을 끌어올리기 위해서는 **사람의 개입이 필수**입니다. "상세 정보 더 보기" 버튼을 클릭하는 로직, 사이트별 본문 셀렉터를 분석해 본문 부분만 파싱하는 코드 등 사람 손이 들어가야 합니다.
2. 그렇지 않고 `document.body.innerText`나 `<main>`·`<article>` 태그 전체를 가져오는 식으로 단순화하면, 원티드가 프론트엔드를 업데이트해 `__b9_L3` 같은 난독화 해시값이 바뀌는 순간 크롤러가 바로 깨집니다. 이 경우에도 **사람의 개입이 필수**입니다.

**Agent — 비전 LLM 에이전트 (자동 시드 수집기)**

화면을 시각으로 이해하고 도구를 사용해 행동합니다. 자연어 명령에서 시작합니다. URL 입력이 불필요한 대신 처리 시간이 길고 LLM 호출 비용이 누적됩니다. 정량 비교 결과는 [`data/jd_comparison_report.md`](./data/jd_comparison_report.md)에서 확인할 수 있습니다.

이 시스템에서 비전 에이전트는 **최종 운영 모드가 아니라 학습 데이터 수집기**입니다. 신규 사이트의 첫 진입과 UI 변경 시 폴백 역할을 담당하며, 평상시 운영은 누적된 로그로부터 생성된 결정론적 Playwright 스크립트가 수행하도록 설계됩니다 (Phase 8 참고).

## Agent 아키텍처

```
사용자 명령
    ↓
[지휘자 — Gemini 3.5 Flash]
  명령 의도 파싱
  사이트 사전지식 조회
  도구 선택 및 순차 호출
    ↓
    ├─ [실행자 — Qwen2.5-VL] (수집이 필요할 때)
    │   화면 인식 (OmniParser Set-of-Marks)
    │   물리 행동 도구
    │     - 로그인 (자격증명은 .env 활용)
    │     - 검색·탐색
    │     - 추출
    │   ※ 모든 도구 호출은 LangSmith로 로깅됨 (Phase 8 학습 데이터)
    │
    └─ [SQLite 검색 도구] (적재된 DB 조회로 충분할 때)
        Gemini SQL 직접 생성 및 실행
        인용 ID 부여
    ↓
[지휘자 — Gemini 3.5 Flash]
  진행 상황 평가
  다음 사이트, 재계획, 또는 답변 생성 결정
  답변 시 인용 검증 적용
```

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
    - [x] Perception: mss 모듈 활용 브라우저 영역 검출 및 YOLOv8 + PaddleOCR 기반의 로컬 SoM(Set-of-Marks) 파이프라인 마커 합성 구현
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
    - [x] Perception Node: 시스템이 화면을 캡처하고 Qwen 비전 모델이 분석하여 요소들을 JSON 텍스트로 추출 후 상태 갱신
    - [x] Reasoning Node: 토큰 비용 절감을 위해 지휘자는 이미지를 보지 않고 Qwen이 넘겨준 텍스트 정보만 읽어 다음 행동 도구 선택
    - [x] Action Node: 도구 실행 및 시스템 안정화 후 다시 Perception Node로 회귀
  - [x] 2. 모듈화 기반 서브 그래프 구축
    - [x] 향후 Phase 6의 라우터 확장을 고려하여 구조를 유연하게 분리
  - [x] 3. LangSmith 통합 트래킹
    - [x] 노드 간 궤적, 소요 토큰 수, 프롬프트 입출력 모니터링 적용 (Phase 8 학습 데이터의 원천)
  - [x] 4. 단일 시나리오 E2E 통합 테스트
    - [x] 바탕화면에서 브라우저 조작 및 검색 화면 이동 검증 ("원티드에서 데이터 분석가 신입 공고 검색해줘" 명령)

- [x] Phase 3.5: OmniParser(Set-of-Marks) 로컬 파이프라인 실제 구현 및 순수 비전 본문 JSON 추출
  - [x] 1. 종속성 패키지 설치 (`ultralytics`, `paddleocr`, `paddlepaddle`) 및 CUDA 연동 확인
  - [x] 2. OmniParser 공식 YOLOv8 모델 가중치 자동 다운로드 유틸 개발
  - [x] 3. `som_engine.py` 구현 (YOLOv8 + PaddleOCR을 통한 요소 검출, 중복 제어 NMS, 마크 이미지 합성 및 좌표 매핑, 2.25초 처리 시간 확인)
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
  - [x] 5. 최종 결과 보고서 작성 및 모니터 위 물리 마우스 자율 조작 벤치마크 리포트 배포 ([`data/jd_comparison_report.md`](./data/jd_comparison_report.md))

- [x] Phase 6: 수집 데이터 전처리 및 DB 적재 신뢰성 강화
  - [x] 1. 고정밀 텍스트 전처리 엔진 구현 (`preprocessor.py`)
    - [x] OCR 텍스트 내 불필요 개행, 특수 기호, 마커 잔영(`[id]`) 제거 및 기술 스택 동의어 매핑
  - [x] 2. 수집 필드 확장 스키마 설계 및 마이그레이션 (`jd_schema.py`, `database.py`)
    - [x] `source_platform`, `raw_ocr_text`, `content_hash`, `experience_min/max/text` 필드 및 인덱스 동적 추가
  - [x] 3. SQLite DB 이중 중복 방지 적재
    - [x] 동일 URL 또는 동일 content_hash 감지 시 자동 `UPDATE` 처리하는 UPSERT 파이프라인 구축
  - [x] 4. E2E 데이터 파이프라인 정합성 최종 검증 (`test_db_persistence.py`)
    - [x] 실제 수집 데이터 기반 전처리·DB 연동 후 적재 검증 패키지 자동 테스트 통과

- [x] Phase 7: 적재 데이터 활용 SQLite 검색 도구 구축 및 지휘자 통합 (RAG 폐기 및 SQL 쿼리 LLM 위임)
  - [x] 1. SQLite 검색 엔진 구현 및 LLM SQL 생성 위임 (`sqlite_query.py`)
    - [x] SQL SELECT 구문을 직접 생성하여 조건에 맞는 데이터를 정밀 조회하는 sqlite_query 도구 탑재
    - [x] 쿼리 유효성 검증 (SELECT 쿼리만 한정하도록 보완) 및 XML 구조화 응답 포맷 연동
  - [x] 2. 지휘자 도구 통합 및 라우팅 (`nodes.py`)
    - [x] sqlite_query 검색 도구를 지휘자의 function calling 도구로 등록
    - [x] 기존 비전 에이전트 수집 흐름을 지휘자의 보조 도구로 통합 및 순차 호출 라우팅
  - [x] 3. 답변 생성 및 인용 검증 (`nodes.py`)
    - [x] XML 구조화 컨텍스트 주입, 스트리밍 응답, 온도 0.0 설정
    - [x] 답변 생성 후 인용 ID 정합성 검사 및 위반 ID는 `[출처 확인 불가]` 마커로 치환 (`validate_citations`)
  - [x] 4. E2E SQLite 검색 동작 및 미적재 정보 거절 테스트 검증 (`test_sqlite_qa.py`)

- [ ] **Phase 8 (예정): 비전 에이전트 궤적 기반 결정론적 Playwright 스크립트 자동 생성**

  > 이 프로젝트의 최종 목표 단계. 비전 에이전트가 누적한 도구 호출 로그를 분석해 사이트별 결정론적 시퀀스를 추출하고, 평상시 운영을 LLM 호출 없는 Playwright 스크립트로 전환한다. 시드 누적 → 시퀀스 추출 → 무료 운영의 흐름을 통해 운영 비용을 0에 수렴시키는 것이 목표.

  - [ ] 1. LangSmith 궤적 데이터 추출 및 정규화 파이프라인 구축
    - [ ] 사이트별·작업 유형별 성공 궤적 필터링 (`finish_task` 도달 여부 기준)
    - [ ] 도구 호출 시퀀스 정규화 (click_marker, type_in_marker, scroll, press_key 순서)
  - [ ] 2. 공통 시퀀스 추출 알고리즘 구현
    - [ ] 같은 사이트·작업 유형 N회 이상 반복된 궤적에서 최빈 도구 시퀀스 추출
    - [ ] 시퀀스 변별력 검증 (특정 임계 빈도 이상일 때만 채택)
  - [ ] 3. 마커 좌표 → DOM 셀렉터 역추론 모듈
    - [ ] OmniParser가 잡은 박스 영역을 Playwright accessibility tree와 매핑
    - [ ] OCR 텍스트 기반 셀렉터 추론 (`text=`, `getByRole` 등) 폴백
  - [ ] 4. Playwright 스크립트 자동 생성기
    - [ ] 추출된 시퀀스 + 셀렉터를 Playwright async API 코드로 변환
    - [ ] `classic/automation/sites/`에 사이트별 어댑터로 자동 등록
  - [ ] 5. 자가 치유 폴백 루프
    - [ ] Playwright 스크립트 실행 실패 감지 (셀렉터 부재, 타임아웃)
    - [ ] 실패 시 비전 에이전트 자동 재가동 및 신규 궤적 수집 → 시퀀스 재추출
  - [ ] 6. 비용 절감 정량 검증
    - [ ] 사이트별 시드 비전 실행 횟수 측정
    - [ ] 손익분기점(시드 비용 vs 누적 절감) 계산 및 리포트

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
│   ├── graph/            LangGraph 워크플로우 (nodes, workflow, state)
│   ├── prompts/          지휘자 프롬프트 (commander)
│   ├── credentials/      .env 자격증명 관리 매니저
│   ├── tools/            화면 인식(Perception)·물리 제어·SQLite 검색·실시간 수집 도구
│   ├── utils/            로깅 및 전처리 유틸리티
│   └── tests/            자동화 유닛/통합 테스트 (DB 영속성, SQLite QA 등)
│
├── shared/             공통 모듈
│   ├── db/               SQLite 데이터베이스 관리
│   └── schema/           Pydantic 스키마 정의
│
├── benchmark/          비교 벤치마크 (.gitkeep만 존재)
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
| 언어·런타임 | Python 3.10+ |
| 브라우저 자동화 | Playwright |
| UI 요소 검출 | OmniParser (Microsoft) |
| 에이전트 워크플로우 | LangGraph |
| 지휘자 모델 | Gemini 3.5 Flash |
| 실행자 비전 모델 | Qwen2.5-VL (Ollama) |
| 실행자 텍스트 모델 | Qwen (Ollama) |
| 임베딩 모델 | Gemini text-embedding-004 |
| 궤적 트래킹 | LangSmith |
| 자격증명 보안 | .env (python-dotenv) |
| 저장소 | SQLite |

## 보안·법적 고려사항

- **자격증명**: 소스 코드에 직접 노출하지 않고 `.env` 파일에 보관하여 `.gitignore`로 관리합니다.
- **사용 범위**: 본인 계정에 한정하며, 학습 목적으로만 사용합니다.
- **사이트 약관**: 각 사이트의 이용약관을 확인하고 진행합니다.

## 빠른 시작

```bash
git clone https://github.com/Dreamtreeme/L2C.git
cd L2C

# 환경 설정
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 자격증명 등록 (최초 1회)
python agent/setup_credentials.py wanted

# Classic 방식 — URL 직접 입력
python classic/main.py extract https://www.wanted.co.kr/wd/123456

# Agent 방식 — 자연어 명령 (수집)
python agent/main.py "데이터 분석가 신입 공고 모아줘"

# Agent 방식 — 자연어 질의 (적재 DB SQLite 조회)
python agent/main.py "수집된 공고 중 신입 가능한 곳 알려줘"

# 두 방식 비교 (정합성 검증 테스트 스크립트 실행)
python scratch/run_compare_jd.py
```

## 향후 작업

Phase 8의 결정론적 스크립트 자동 생성을 우선 목표로 합니다. 진행 중이거나 검토 중인 항목은 [Issues](../../issues)에서 확인할 수 있습니다.

---

지속적으로 확장 중인 실험 프로젝트입니다.
