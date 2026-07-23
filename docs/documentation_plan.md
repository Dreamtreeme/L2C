---
title: "문서 정리 계획"
type: plan
area: documentation
status: active
updated: 2026-07-23
tags:
  - l2c
  - docs/documentation
---

# 문서 정리 계획

## 진행 현황

| 단계 | 작업 | 상태 | 완료 기준 |
|---|---|---|---|
| 0 | Obsidian Markdown·Bases 스킬 설치 | 완료 | Codex 사용자 스킬 경로에서 두 스킬을 읽을 수 있음 |
| 1 | 속성 규칙, 문서 인덱스, Base 대시보드 도입 | 완료 | 핵심 `docs/*.md`가 같은 속성을 사용하고 GitHub 인덱스와 Base에서 조회됨 |
| 2 | 현재 기준과 역사 기록 구분 | 완료 | 오래된 버전·실험 결과가 현재 아키텍처로 오해되지 않음 |
| 3 | 트러블슈팅 사건별 분할 | 대기 | 사건마다 원인·조치·검증이 독립 문서이며 기존 목차 링크가 유지됨 |
| 4 | 설계·실험·운영 폴더 분리 | 대기 | 이동 후 내부 링크 검사 통과, 중복 문서 없음 |
| 5 | 문서 검사 자동화 | 완료 | frontmatter 필수값, 내부 링크, 중복 H1을 테스트에서 검사함 |
| 6 | Obsidian 화면 검증 | 대기 | Base의 전체·계획·역사 보기와 Markdown callout이 정상 렌더링됨 |

## 2단계. 현재성과 역사성 정리

1. `ARCHITECTURE.md`, `README.md`, `docs/design_decisions.md`에서 현재 라이브러리와 실행 흐름이 일치하는지 확인한다.
2. 과거 수치가 현재 기준처럼 쓰인 문서는 `historical`로 바꾸거나 현재 기준 문서로 연결한다.
3. 런타임 버전은 [런타임 호환 기준](runtime_compatibility.md), 현재 구조는 [`ARCHITECTURE.md`](../ARCHITECTURE.md)를 단일 기준으로 사용한다.
4. 실험 결과의 원본 로그가 없으면 성능 기준값으로 승격하지 않는다.

## 3단계. 트러블슈팅 분할

현재 959줄인 `troubleshooting.md`를 한 번에 다시 쓰지 않는다. 각 사건을 다음 형식으로 옮긴다.

```text
docs/operations/troubleshooting/
  001-wanted-login-loop.md
  002-ollama-json-loop.md
  ...
  020-python313-paddleocr3.md
```

각 사건 문서는 증상, 재현 조건, 반증한 가설, 실제 원인, 해결, 검증, 관련 로그를 구분한다. 원문에서 사건 하나를 옮긴 커밋 안에서 `troubleshooting.md`의 해당 본문을 링크 한 줄로 교체해 중복을 남기지 않는다.

## 4단계. 폴더 분리

최종 목표 구조는 다음과 같다.

```text
docs/
  index.md
  architecture/
  experiments/
    reflex/
    ocr/
    model-selection/
  operations/
    troubleshooting/
  reference/
  dashboards/
```

파일 이동은 내용 정리와 같은 커밋에서 하지 않는다. 먼저 링크 검사를 추가한 뒤 `git mv`와 참조 수정만 수행하는 별도 커밋으로 진행한다.

## 5단계. 검사 자동화

검사는 다음 오류만 결정론적으로 판정한다.

- `docs/*.md`의 필수 속성 누락
- 정의되지 않은 `type`, `area`, `status`
- 존재하지 않는 상대 Markdown 링크
- 문서 안의 H1 누락 또는 중복
- `.base` YAML 문법 오류

문서 내용의 품질이나 최신 여부를 문자열 휴리스틱으로 판정하지 않는다.

```powershell
.\.venv-app\Scripts\python.exe scripts\check_docs.py
```

전체 `agent/tests`에도 같은 계약 검사가 포함된다.

## 범위 밖

- 코드에서 생성되는 benchmark 보고서를 사람이 관리하는 문서처럼 강제 변환하지 않는다.
- 사이트별 실행 지침은 검증 가능한 `profile.json`의 `guidance`에 둔다.
- Obsidian 플러그인에 의존하는 내용을 README의 필수 탐색 경로로 만들지 않는다.
