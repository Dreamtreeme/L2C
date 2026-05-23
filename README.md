# L2C — LLM to Computer

> 자연어 명령으로 채용공고를 수집하는 에이전트 실험 프로젝트입니다.

## 프로젝트 의도

기존 자동화 도구는 사용자가 URL을 직접 입력해야 합니다. 이 프로젝트는 그 단계를 자연어 명령으로 대체할 수 있는지 실험합니다.

비전 LLM이 화면을 보고 행동할 수 있다면, 자연어 명령만으로 로그인·검색·탐색·추출이 가능한 에이전트를 만들 수 있다는 가설을 검증합니다.

첫 번째 실험 주제는 채용공고 수집입니다.

## 차별점

기존 자동화:
```python
extract("https://www.wanted.co.kr/wd/123456")
```

이 시스템:
```python
agent.run("데이터 분석가 신입 공고 모아줘")
```

| 단계 | 전통 자동화 | L2C Agent |
|------|------------|-----------|
| URL 수집 | 수동 | 자동 |
| 사이트 로그인 | 사이트별 코드 | 시각 인식 |
| 검색·페이지네이션 | 사이트별 코드 | 시각 인식 |
| 새 사이트 추가 | 코드 작성 | YAML 설정 추가 |
| 정보 추출 | 셀렉터 의존 | DOM 또는 시각 |

## 두 시스템 비교

차이를 정량적으로 측정하기 위해 동일 작업을 두 방식으로 구현합니다.

**Classic — 전통 자동화**

Playwright로 DOM 구조를 직접 파싱합니다. 사이트별 마커와 셀렉터를 사전 정의해야 합니다. 빠르고 안정적이지만 사이트별 코드가 필요하고 URL을 수동으로 가져와야 합니다.

원래 의도는 페이지 텍스트를 통째로 LLM에게 던져 알아서 발라내게 하는 방식이었으나, 토큰 비용과 정확도를 끌어올리는 과정에서 본문 셀렉터를 직접 분석해 노이즈를 잘라낸 형태로 최적화되었습니다. 이로 인해 다음 두 가지 한계가 따라옵니다.

1. 토큰을 절약하고 성능을 끌어올리기 위해서는 **사람의 개입이 필수**입니다. "상세 정보 더 보기" 버튼을 클릭하는 로직, 사이트별 본문 셀렉터를 분석해 본문 부분만 파싱하는 코드 등 사람 손이 들어가야 합니다.
2. 그렇지 않고 `document.body.innerText`나 `<main>`·`<article>` 태그 전체를 가져오는 식으로 단순화하면, 원티드가 프론트엔드를 업데이트해 `__b9_L3` 같은 난독화 해시값이 바뀌는 순간 크롤러가 바로 깨집니다. 이 경우에도 **사람의 개입이 필수**입니다.

**Agent — 비전 LLM 에이전트**

화면을 시각으로 이해하고 도구를 사용해 행동합니다. 자연어 명령에서 시작합니다. URL 입력이 불필요하지만 처리 시간이 길고 시각 인식의 한계가 있습니다.

비교를 통해 확인하려는 것:
- 자연어 명령 자동화가 어디까지 실용적인가
- 어느 단계에서 LLM이 가치를 만드는가
- 두 방식을 어떻게 조합하는 것이 효율적인가

## Agent 아키텍처

```
사용자 명령: "데이터 분석가 신입 공고 모아줘"
    ↓
[지휘자 — Claude]
  명령 의도 파싱
  사이트 사전지식 조회
  실행 계획 수립
    ↓
[실행자 — Qwen2.5-VL]
  화면 인식 (OmniParser Set-of-Marks)
  도구 사용
    - 로그인 (자격증명은 .env 활용)
    - 검색·탐색
    - 추출
    ↓
[지휘자 — Claude]
  진행 상황 평가
  다음 사이트 또는 재계획 결정
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
  - [x] 2. 백그라운드 엔진 스크립트 ※ LLM 직접 호출 불가
    - [x] Perception: mss 모듈 활용 브라우저 영역 검출 및 YOLOv8 + EasyOCR 기반의 로컬 SoM(Set-of-Marks) 파이프라인 마커 합성 구현 (완료)
    - [x] Wait Stable: 무한 대기 버그를 막기 위해 픽셀 오차율 1퍼센트 이하 조건 및 최대 대기 시간 5초를 적용한 시각적 화면 안정화 대기
    - [x] Security: .env 기반 자격증명 관리 시스템`
  - [x] 3. 순수 파이썬 좌표 검증 테스트
    - [x] LLM 연동 전 캡처 화면의 마커 좌표와 실제 마우스 클릭 좌표가 어긋나지 않는지 하드코딩으로 1차 확인
  - [x] 4. 에이전트 상태 관리
    - [x] 스크롤이나 클릭 동작의 성공 여부를 비교할 수 있도록 최근 2장의 전후 마커 이미지 보관
    - [x] 파싱된 UI 요소 텍스트 및 Action History 정의
  - [x] 5. LLM 전용 물리 행동 도구 ※ 에이전트의 유일한 인터페이스
    - [x] click_marker: 마커 ID의 절대 좌표 계산 후 PyAutoGUI 물리적 클릭
    - [x] type_in_marker: 한글 씹힘 현상 방지를 위해 pyperclip 모듈을 활용한 클립보드 복사 후 붙여넣기 물리 타이핑
    - [x] scroll: OS 마우스 휠 스크롤
    - [x] press_key: Enter, ESC 등 특수키 입력
    - [x] finish_task: 수집 완료 시 데이터를 반환하며 루프 강제 종료
    - [x] Action Wrapper: 도구 실행 시 성능 로깅 및 안정화 대기 로직 자동 주입

- [x] Phase 3: LangGraph 지휘자 워크플로우 구성
  - [x] 1. 노드 설계 및 관찰 → 계획 → 행동 루프
    - [x] Perception Node: 시스템이 화면을 캡처하고 Qwen 비전 모델이 분석하여 요소들을 JSON 텍스트로 추출 후 상태 갱신
    - [x] Reasoning Node: 토큰 비용 절감을 위해 지휘자는 이미지를 보지 않고 Qwen이 넘겨준 텍스트 정보만 읽어 다음 행동 도구 선택
    - [x] Action Node: 도구 실행 및 시스템 안정화 후 다시 Perception Node로 회귀
  - [x] 2. 모듈화 기반 서브 그래프 구축
    - [x] 향후 Phase 6의 라우터 확장을 고려하여 구조를 유연하게 분리
  - [x] 3. LangSmith 통합 트래킹
    - [x] 노드 간 궤적, 소요 토큰 수, 프롬프트 입출력 모니터링 적용
  - [x] 4. 단일 시나리오 E2E 통합 테스트
    - [x] 바탕화면에서 브라우저 조작 및 검색 화면 이동 검증 ("원티드에서 데이터 분석가 신입 공고 검색해줘" 명령)

- [x] Phase 3.5: OmniParser (Set-of-Marks) 로컬 파이프라인 실제 구현 및 순수 비전 본문 JSON 추출
  - [x] 1. 종속성 패키지 설치 (`ultralytics`, `easyocr`) 및 CUDA 연동 확인
  - [x] 2. OmniParser 공식 YOLOv8 모델 가중치 자동 다운로드 유틸 개발 (다운로드 및 로드 확인 완료)
  - [x] 3. `som_engine.py` 구현 (YOLOv8 + EasyOCR을 통한 요소 검출, 중복 제어 NMS, 마크 이미지 합성 및 좌표 매핑) (완료 - 2.25초 성능 확인)
  - [x] 4. `perception.py` 리팩토링 및 SoM 연동 (마킹 이미지 주입 및 마커 ID 좌표 매핑 디코딩) (완료)
  - [x] 5. VLM 프롬프트 최적화 (멀티모달 SoM 마크 이미지 주입 및 의사결정 프롬프트 팝업/모달 차단 조치 추가) (완료)
  - [x] 6. E2E 본문 추출 통합 테스트 검증 및 벤치마크
    - [x] 원티드 데이터 분석가 검색 이동 성공 및 듀얼 모니터 좌표 매핑 속도 검증
    - [x] **[핵심 목적]** 개별 채용공고 카드 클릭 상세 진입 ➡️ "상세 정보 더 보기" 클릭 본문 확장 ➡️ 스크롤을 통한 화면 전체 텍스트 판독 ➡️ 주요업무, 자격요건, 우대사항, 혜택 항목별 구조화된 JSON 본문 데이터 최종 추출 완료 및 파일 저장 검증

- [x] Phase 4: 사이트 제로샷 일반화 검증 및 채용공고 본문 비교 검증
  - [x] 1. 로그인 불필요 환경에 대응하는 워크넷 접속 및 검색 추출 (또는 Wanted 자율 비로그인 탐색 시나리오 검증 완료)
  - [x] 2. 잡코리아 등 DOM 구조가 다른 사이트 적용 테스트 (어댑터 구조 수정 및 CSS 셀렉터 최적화 완료)
  - [x] 3. 새로운 사이트 추가 시 classic 방식처럼 별도의 파싱 코드 작성이 필요 없음을 증명
  - [x] 4. **[추가]** 검색 결과 채용 공고 카드를 클릭하여 상세 페이지(본문)로 이동한 후, 화면 내 본문 텍스트를 판독/추출하여 파일로 저장 (완료)
  - [x] 5. **[추가]** 저장된 본문 파일과 실제 사이트의 원문(Ground Truth) 텍스트를 텍스트 유사도 및 차이(Diff) 분석을 통해 정밀 검증하는 프로세스 구축 (완료)

- [x] Phase 5: Classic 대 Agent 벤치마크 실험 및 본문 정합성 비교 데모
  - [x] 1. 성공률 및 완료 소요 시간에 대한 Structlog 데이터 기반 정량 비교 (완료)
  - [x] 2. **[추가]** Classic 시스템이 수집한 공고 원문 파일과 Agent가 저장한 본문 텍스트 파일 간의 텍스트 매칭 정확도 및 누락률 정량 비교 (완료)
  - [x] 3. LangSmith 데이터 기반 에이전트 오류 자가 복구율 분석 및 재귀 제한(60) 설정 완화 (완료)
  - [x] 4. 토큰 사용량 기반 비용 산출 및 로컬 모델 500 에러 OOM 문제 우회 (Gemini API 텍스트 연동 추가 완료)
  - [x] 5. 최종 결과 보고서 작성 및 모니터 위 물리 마우스 자율 조작 벤치마크 리포트 배포 완료 (`data/jd_comparison_report.md`)

- [x] Phase 6: 수집 데이터 전처리 및 DB 적재 신뢰성 강화
  - [x] 1. 고정밀 텍스트 전처리 엔진 구현 (`preprocessor.py`)
    - [x] OCR 텍스트 내 불필요 개행, 특수 기호, 마커 잔영(`[id]`) 제거 및 기술 스택 동의어 매핑
  - [x] 2. 수집 필드 확장 스키마 설계 및 마이그레이션 (`jd_schema.py`, `database.py`)
    - [x] `source_platform`, `raw_ocr_text`, `content_hash`, `experience_min/max/text` 필드 및 인덱스 동적 추가
  - [x] 3. SQLite DB 이중 중복 방지 적재
    - [x] 동일 URL 또는 동일 content_hash 감지 시 자동 `UPDATE` 처리하는 UPSERT 파이프라인 구축
  - [x] 4. E2E 데이터 파이프라인 정합성 최종 검증 (`test_db_persistence.py`)
    - [x] 실제 수집 데이터 기반 전처리-DB 연동 후 적재 검증 패키지 자동 테스트 통과

- [x] Phase 7: 포트폴리오 최적화 경량 RAG Q&A 에이전트 구축
  - [x] 1. SQLite DB 임베딩 필드 확장
    - [x] 768차원 `embedding` BLOB 컬럼 추가 및 동적 자동 마이그레이션 구현
  - [x] 2. 임베딩 쿼리 메모리 캐시 구축
    - [x] text-embedding-004 임베딩 가속화를 위한 `EmbeddingLRUCache` 신설
  - [x] 3. 초경량 RAG 검색 엔진 구현 및 지휘자 모델 통합 (`rag_engine.py`)
    - [x] LLM Query Parser(Gemini 3.5 Flash) 구조화 필터 추출 및 `days_limit` 역산 날짜 연산 SQL 1차 필터링 ➡️ Numpy 단일 루프 Cosine Similarity 연산
    - [x] Top-1 vs Top-2~5 평균 스코어 갭 기반 변별력 검증 (Pre-LLM Check로 불필요 LLM API 호출 우회)
  - [x] 4. Q&A 에이전트 노드 연동 및 인용 교정 (`nodes.py`)
    - [x] XML 구조화 컨텍스트 주입, 스트리밍 응답, 온도 0.0 설정
    - [x] 사후 검증(`validate_citations`)을 통한 잘못된 인용 ID 감지 시 `[출처 확인 불가]`로 안전 치환
  - [x] 5. E2E RAG 및 환각 거절 테스트 검증 (`test_rag_qa.py`)
    - [x] 미적재 무관 질문 유입 시 유사도 갭 미달로 즉시 거절 메시지를 리턴하는 환각 거절 케이스 검증 통과


## 디렉토리 구조

```
L2C/
├── classic/          전통 자동화 (베이스라인)
│   ├── automation/     Playwright DOM 파싱
│   ├── extractor/      VLM 정형화
│   └── README.md
│
├── agent/            비전 LLM 에이전트
│   ├── tools/          화면 인식·행동·추출 도구
│   ├── graph/          LangGraph 워크플로우
│   ├── prompts/        지휘자·실행자 프롬프트
│   ├── credentials/    .env 자격증명 관리
│   ├── sites/          사이트 사전지식 (YAML)
│   └── README.md
│
├── shared/           공통
│   ├── schema/         Pydantic 스키마
│   └── db/             SQLite
│
├── benchmark/        두 시스템 비교
│   ├── scenarios/      테스트 시나리오
│   └── results/        비교 결과
│
└── docs/
    ├── architecture.md
    ├── design_decisions.md
    ├── legal_considerations.md
    ├── security.md
    └── lessons_learned.md
```

## 기술 스택

| 카테고리 | 기술 |
|---------|------|
| 언어·런타임 | Python 3.10+ |
| 브라우저 자동화 | Playwright |
| UI 요소 검출 | OmniParser (Microsoft) |
| 에이전트 워크플로우 | LangGraph |
| 지휘자 모델 | Gemini 3.5 Flash |
| 실행자 모델 | Qwen2.5-VL · Qwen3 (Ollama) |
| 자격증명 보안 | .env (python-dotenv) |
| 저장소 | SQLite |

## 보안·법적 고려사항

- **자격증명**: 소스 코드에 직접 노출하지 않고 `.env` 파일에 보관하여 `.gitignore`로 관리합니다.
- **사용 범위**: 본인 계정에 한정하며, 학습 목적으로만 사용합니다.
- **사이트 약관**: 각 사이트의 이용약관을 확인하고 진행합니다. [자세히](./docs/legal_considerations.md)

## 빠른 시작 (예정)

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

# Agent 방식 — 자연어 명령
python agent/main.py "데이터 분석가 신입 공고 모아줘"

# 두 방식 비교
python benchmark/compare.py --scenario "data_analyst_entry"
```

## 로드맵

진행 상황은 [Issues](../../issues)에서 확인할 수 있습니다.

---

개발 중인 프로젝트입니다.