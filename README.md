# L2C — LLM to Computer

> LLM이 컴퓨터 화면을 어디까지 다룰 수 있는가? 동일한 작업을 전통 자동화와 AI 에이전트로 각각 구현하고 비교 분석하는 실험 모음

## 🎯 프로젝트 의도

LLM이 사람처럼 컴퓨터를 다루는 방향("LLM to Computer")을 정량 실험합니다.
첫 번째 실험 주제는 **채용공고 추출**입니다.

**핵심 질문**:
> "사이트별 코드를 작성하지 않고도 AI가 임의의 채용 사이트에서 정보를 추출할 수 있는가?
> 가능하다면, 전통적인 방식 대비 어떤 트레이드오프가 있는가?"

## 📐 두 가지 접근

### 1️⃣ Classic: DOM Structure Parsing
Playwright로 페이지 구조(DOM)를 직접 파악하고, Bounding Box 기반으로 텍스트 데이터를 구조적으로 추출합니다. (규칙 기반)

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
- Playwright (브라우저 자동화 및 DOM 파싱)
- OmniParser (Agent용 UI 요소 검출)
- LangGraph (Agent용 워크플로우 제어)
- Qwen2.5-VL via Ollama (Agent용 로컬 LLM)
- SQLite (결과 저장)

## 🚀 빠른 시작 (예정)

```bash
git clone https://github.com/Dreamtreeme/L2C.git
cd L2C

# 환경 설정
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
```

## 📅 로드맵

자세한 진행 상황은 [Issues](../../issues)와 [Projects](../../projects)에서.

---

🚀 **개발 중인 프로젝트입니다. 결과는 정기적으로 업데이트됩니다.**
