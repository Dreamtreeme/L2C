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
    - 로그인 (자격증명은 OS 키체인)
    - 검색·탐색
    - 추출
    ↓
[지휘자 — Claude]
  진행 상황 평가
  다음 사이트 또는 재계획 결정
```

## 진행 상황

- [x] 프로젝트 셋업
- [ ] **Phase 1: Classic 시스템 (베이스라인)**
  - [ ] URL 입력 기반 추출
  - [ ] DOM bounding box 영역 식별
  - [ ] 5개 사이트 안정화
- [ ] **Phase 2: Agent Core (원티드)**
  - [ ] 자격증명 시스템 (OS 키체인)
  - [ ] 시각 기반 자동 로그인
  - [ ] 시각 기반 검색·탐색
  - [ ] 자연어 명령으로 5건 수집
- [ ] **Phase 3: LangGraph 지휘자**
  - [ ] 명령 의도 파싱
  - [ ] 사이트 사전지식 베이스
  - [ ] 계획·실행·검증 루프
- [ ] **Phase 4: 사이트 일반화**
  - [ ] 워크넷 (로그인 불필요)
  - [ ] 잡코리아 또는 프로그래머스
  - [ ] 사이트 추가 시 코드 작성 불필요 검증
- [ ] **Phase 5: 비교 실험·데모**
  - [ ] Classic vs Agent 정량 비교
  - [ ] 데모 영상
  - [ ] 결과 보고서

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
│   ├── credentials/    OS 키체인 통합
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
| 지휘자 모델 | Claude |
| 실행자 모델 | Qwen2.5-VL (Ollama) |
| 자격증명 보안 | keyring (OS 키체인) |
| 저장소 | SQLite |

## 보안·법적 고려사항

- **자격증명**: 평문으로 저장하지 않으며, OS 키체인(Windows Credential Manager, macOS Keychain)에만 보관합니다.
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