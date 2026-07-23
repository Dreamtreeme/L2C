---
title: "L2C 문서 인덱스"
type: hub
area: documentation
status: active
updated: 2026-07-23
tags:
  - l2c
  - docs/documentation
---

# L2C 문서 인덱스

> [!info] 사용 방식
> GitHub에서는 아래 표준 Markdown 링크를 사용합니다. Obsidian에서는 같은 폴더의 [문서 대시보드](documentation.base)로 상태와 영역을 필터링할 수 있습니다.

## 먼저 읽을 문서

| 문서 | 역할 |
|---|---|
| [`README.md`](../README.md) | 프로젝트 목적, 차별점, 검증 결과, 실행 방법 |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | 현재 구현된 시스템 구조 |
| [아키텍처 리팩터링 계획](architecture_refactor_plan.md) | 다음 구조 변경 순서와 삭제 대상 |
| [문서 정리 계획](documentation_plan.md) | 문서 분할과 검증 도입 진행 상태 |

## 아키텍처와 런타임

| 문서 | 상태 | 내용 |
|---|---|---|
| [기술 및 설계 결정](design_decisions.md) | 현재 | 채택한 구조와 트레이드오프 |
| [L2C 네이밍 규칙](naming_conventions.md) | 현재 | 코드와 레시피 단계의 명명 기준 |
| [런타임 호환 기준](runtime_compatibility.md) | 현재 | Python, CUDA, PaddleOCR, PyTorch 검증 조합 |
| [E2E 관측 환경](e2e_observability.md) | 현재 | 실행시간, 토큰, 실패 단계, LangSmith 관측 기준 |

## Reflex와 실험

| 문서 | 상태 | 내용 |
|---|---|---|
| [Reflex Recipe 구현 기준](reflex_recipe_plan.md) | 현재 | 후보 저장, Critic 검토, ROI 재생 흐름 |
| [`troubleshooting.md`](../troubleshooting.md) | 분할 예정 | 실험 과정과 장애 해결 원본 기록 |
| [트러블슈팅 및 교훈](lessons_learned.md) | 역사 기록 | 초기 GUI·VLM 구현에서 얻은 교훈 |
| [`jd_comparison_report.md`](../benchmark/jd_comparison_report.md) | 벤치마크 | Classic과 Vision 수집 비교 |

## 검색과 사이트

| 문서 | 상태 | 내용 |
|---|---|---|
| [검색 의미 사전](search_taxonomy.md) | 현재 | 직무·기술 개념과 별칭 관리 |
| [사이트별 수집 특성 기록](site_characteristics.md) | 역사 기록 | Classic 경로 중심의 초기 사이트 조사 |
| [`agent/sites/README.md`](../agent/sites/README.md) | 현재 | Realtime 사이트 프로필 구조와 공식 주소 |

## 문서 관리

- [문서 작성 규칙](documentation_conventions.md)
- [문서 정리 계획](documentation_plan.md)
- [Obsidian 문서 대시보드](documentation.base)

현재 동작은 [`ARCHITECTURE.md`](../ARCHITECTURE.md), 런타임 버전은 [런타임 호환 기준](runtime_compatibility.md)을 우선합니다. 역사 기록과 계획 문서는 현재 구현의 근거로 단독 사용하지 않습니다.
