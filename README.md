# Job Extractor Evolution

> 동일한 채용공고 추출 작업을 두 가지 방식으로 구현하고 비교 분석하는 프로젝트

## 🎯 프로젝트 의도

전통적인 자동화(DOM 크롤링)와 AI 에이전트 방식의 차이를 정량적으로 비교합니다.

**핵심 질문**:
> "사이트별 코드를 작성하지 않고도 AI가 임의의 채용 사이트에서 정보를 추출할 수 있는가?
> 가능하다면, 전통적인 방식 대비 어떤 트레이드오프가 있는가?"

## 📐 두 가지 접근

### 1️⃣ Classic: DOM Bounding Box + VLM
Playwright로 페이지 구조를 파악하고, 비전 모델로 정형화

### 2️⃣ Agent: OmniParser + LangGraph + Tool Use
Set-of-Marks 기반 LLM 에이전트가 사이트 구조를 스스로 파악

## 🚧 현재 진행 상황

- [x] 프로젝트 셋업
- [ ] Classic 시스템 마이그레이션
- [ ] Agent MVP
- [ ] 벤치마크 비교
- [ ] 최종 분석

## 📁 디렉토리 구조

```
classic/        DOM 기반 시스템
agent/          AI 에이전트 시스템
shared/         공통 (스키마, DB)
benchmark/      두 시스템 비교
docs/           설계 문서
scripts/        유틸리티 스크립트
```

## 🛠️ 기술 스택

- Python 3.10+
- Playwright (브라우저 자동화)
- OmniParser (UI 요소 검출)
- LangGraph (에이전트 워크플로우)
- Qwen2.5-VL via Ollama (로컬 LLM)
- SQLite (저장)

## 🚀 빠른 시작 (예정)

```bash
git clone https://github.com/Dreamtreeme/job-extractor-evolution.git
cd job-extractor-evolution

# 환경 설정
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
```

## 📅 로드맵

자세한 진행 상황은 [Issues](../../issues)와 [Projects](../../projects)에서.

---

🚀 **개발 중인 프로젝트입니다. 결과는 정기적으로 업데이트됩니다.**
# L2C
