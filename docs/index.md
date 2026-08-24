---
title: "L2C 문서 인덱스"
type: hub
area: documentation
status: active
updated: 2026-08-24
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
| [`README.md`](../README.md) | 프로젝트 개요, 핵심 흐름, 확인한 범위와 실행 방법 |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | 현재 구현된 시스템 구조 |
| [제품 데모 및 검증](product_demo.md) | 실제 UI 실행, 재현 시나리오, 검증 수치 |
| [개발 단계 기록](development_history.md) | Phase별 구현 범위와 축소 과정 |

## 아키텍처와 런타임

| 문서 | 상태 | 내용 |
|---|---|---|
| [기술 및 설계 결정](design_decisions.md) | 현재 | 채택한 구조와 트레이드오프 |
| [작업자 상태 계약](worker_state_contract.md) | 현재 | 책임별 WorkerState 구역, 갱신 주체와 불변조건 |
| [L2C 네이밍 규칙](naming_conventions.md) | 현재 | 코드와 레시피 단계의 명명 기준 |
| [런타임 호환 기준](runtime_compatibility.md) | 현재 | Python, CUDA, PaddleOCR, PyTorch 검증 조합 |
| [E2E 관측 환경](e2e_observability.md) | 현재 | 실행시간, 토큰, 실패 단계, LangSmith 관측 기준 |
| [포트폴리오 평가 실행 규격](portfolio_evaluation.md) | 현재 | 동일 조건에서 성공률, 재사용 효과와 자원 사용량을 측정하는 방법 |
| [현재 버전 검증 기록](evidence/l2c_metrics_20260822.md) | 현재 | 5개 사이트 수집, 저장 행동 재사용, DB 근거 답변 결과 |

## Reflex와 실험

| 문서 | 상태 | 내용 |
|---|---|---|
| [Reflex Recipe 구현 기준](reflex_recipe_plan.md) | 현재 | 후보 저장, Critic 검토, ROI 재생 흐름 |
| [`troubleshooting.md`](../troubleshooting.md) | 현재 | 재발 가능한 장애의 원인, 현재 처리와 검증 경로 |
| [조사 및 경험 탐색 리팩터링 완료 기록](next_steps.md) | 보관 | 조사·작업자 그래프 단순화의 단계별 검증 결과 |
| [경험 규칙 리팩터링 완료 기록](experience_rule_refactor_plan.md) | 보관 | 완료한 리팩터링의 계획 변경과 검증 결과 |
| [신규 사이트 적용 공수 검증 회고](site_onboarding_benchmark_plan.md) | 보관 | Codex 보조 Classic·Vision 비교의 계약과 결과 |
| [초기 비전 자동화 실험 기록](history/legacy_vision_experiments.md) | 보관 | 폐기된 로컬 모델, 구버전 OCR과 초기 pHash 실험 |

## 검색과 사이트

| 문서 | 상태 | 내용 |
|---|---|---|
| [검색 의미 사전](search_taxonomy.md) | 현재 | 직무·기술 개념과 별칭 관리 |
| [`agent/sites/README.md`](../agent/sites/README.md) | 현재 | Realtime 사이트 프로필 구조와 공식 주소 |

## 문서 관리

- [문서 작성 규칙](documentation_conventions.md)
- [Obsidian 문서 대시보드](documentation.base)

현재 동작은 [`ARCHITECTURE.md`](../ARCHITECTURE.md), 런타임 버전은 [런타임 호환 기준](runtime_compatibility.md)을 우선합니다.
