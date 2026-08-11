---
title: "문서 작성 규칙"
type: reference
area: documentation
status: active
updated: 2026-07-23
tags:
  - l2c
  - docs/documentation
---

# 문서 작성 규칙

L2C 문서는 GitHub에서 바로 읽히는 표준 Markdown을 기준으로 작성하고, Obsidian에서는 속성과 Base 대시보드를 추가로 활용한다. 작성 규칙은 `obsidian-markdown`과 `obsidian-bases` 스킬을 참고하되 저장소 공개 문서의 호환성을 우선한다.

## 문서 속성

`docs/` 아래 사람이 직접 관리하는 Markdown 문서는 다음 속성을 사용한다.

```yaml
---
title: "문서 제목"
type: reference
area: runtime
status: active
updated: 2026-07-23
tags:
  - l2c
  - docs/runtime
---
```

필드는 다음 의미로 제한한다.

| 필드 | 값 | 의미 |
|---|---|---|
| `title` | 자유 텍스트 | 문서에서 사용하는 공식 제목 |
| `type` | `hub`, `architecture`, `decision`, `guide`, `plan`, `reference`, `retrospective` | 문서 역할 |
| `area` | `project`, `architecture`, `documentation`, `runtime`, `observability`, `reflex`, `search`, `sites` | 주된 소유 영역 |
| `status` | `active`, `planned`, `historical`, `deprecated` | 현재 유효성 |
| `updated` | `YYYY-MM-DD` | 의미 있는 내용을 마지막으로 검토한 날짜 |
| `tags` | 목록 | `l2c`와 `docs/<area>`만 기본 사용 |

속성값을 테스트를 통과하기 위한 세부 상태로 늘리지 않는다. 문서 하나가 여러 영역을 다루더라도 `area`는 주된 소유 영역 하나만 선택한다.

## 링크

- 저장소 내부 문서는 `[표시 이름](상대경로.md)` 형식을 사용한다.
- `[[위키링크]]`는 GitHub에서 일관되게 동작하지 않으므로 공개 문서의 기본 링크로 사용하지 않는다.
- 코드 파일은 저장소 루트 기준 경로를 백틱으로 표시한다.
- 외부 근거는 검색 결과가 아니라 공식 문서나 원문 URL로 연결한다.
- 문서를 옮길 때는 먼저 모든 참조를 검색하고 같은 변경에서 링크를 수정한다.

## 문서 역할

| 문서 | 역할 |
|---|---|
| `README.md` | 프로젝트 목적, 검증 결과, 실행 방법을 보여주는 포트폴리오 진입점 |
| `ARCHITECTURE.md` | 현재 구현된 시스템 구조의 단일 기준 |
| `docs/index.md` | 모든 사람이 관리하는 문서의 탐색 진입점 |
| `docs/design_decisions.md` | 채택한 결정과 트레이드오프 |
| `troubleshooting.md` | 재현한 장애, 원인과 최종 해결 기록 |

`ARCHITECTURE.md`에는 현재 동작만 기록하고 진행 중인 작업은 GitHub Issues에서 관리한다.

## Callout

중요한 적용 범위나 오래된 자료를 표시할 때만 Obsidian callout을 사용한다. GitHub에서는 일반 인용문으로 읽혀도 의미가 유지되어야 한다.

```markdown
> [!warning] 적용 범위
> 이 내용은 Classic 경로에만 적용됩니다.
```

장식 목적의 callout, 강조색, 태그를 추가하지 않는다.

## 파일과 제목

- 파일명은 소문자 `snake_case.md`를 유지해 현재 저장소 규칙과 맞춘다.
- H1은 파일당 하나만 둔다.
- 제목에 단계 번호를 넣지 않고 계획 표에서 순서를 관리한다.
- 하나의 주제를 두 문서에 복제하지 않고 한쪽을 단일 기준으로 정한 뒤 링크한다.
- 실험 결과에는 조건, 비교 기준, 결과, 해석을 함께 기록한다.

## Obsidian 전용 파일

`.base`와 `.canvas`는 로컬 탐색 보조물이다. 이 파일만 있어야 이해할 수 있는 내용은 두지 않는다. GitHub 사용자는 [문서 인덱스](index.md)에서 같은 정보에 접근할 수 있어야 한다.

## 참고 스킬

- [obsidian-markdown](https://github.com/kepano/obsidian-skills/tree/main/skills/obsidian-markdown)
- [obsidian-bases](https://github.com/kepano/obsidian-skills/tree/main/skills/obsidian-bases)
