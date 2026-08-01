---
title: "L2C 네이밍 규칙"
type: reference
area: architecture
status: active
updated: 2026-07-29
tags:
  - l2c
  - docs/architecture
---

# L2C 네이밍 규칙

이 문서는 코드 분할과 리팩터링 때 사용할 이름 기준이다. 새 코드는 이 규칙을 따르고, 기존 코드는 관련 파일을 수정할 때 함께 정리한다. 대규모 일괄 rename은 피한다.

## 기본 원칙

- 이름은 구현 방식보다 역할을 먼저 드러낸다.
- 같은 개념은 프로젝트 전체에서 같은 이름을 쓴다.
- 후보, 검토, 승격, 실행 중 어느 단계의 데이터인지 이름에 드러낸다.
- LLM이 의미 판단하는 영역과 코드가 결정론적으로 실행하는 영역을 이름으로 분리한다.
- `data`, `info`, `item`, `metadata`, `result`, `recipe`처럼 단독으로는 넓은 이름을 피한다.

## 도메인 용어

| 이름 | 의미 |
|---|---|
| `worker_submission` | 자율 탐색 worker가 끝난 뒤 지휘자/리뷰어에 제출하는 구조화 결과 |
| `recorded_step` | 자율 탐색 중 실제 실행된 원본 행동 기록 |
| `recipe_candidate` | 아직 active가 아닌 예비 레시피 후보 |
| `candidate_review` | Critic이 후보를 검토한 결과 |
| `promotion` | 후보를 active recipe로 승격하는 처리 결과 |
| `active_recipe` | 경험 기반 탐색에서 실제 재생 가능한 활성 레시피 |
| `recipe_action` | 한 화면 관찰을 근거로 실행할 물리 행동 |
| `recipe_checkpoint` | 전이 전후에 다시 확인할 URL·화면 문맥·ROI 앵커 상태 |
| `recipe_transition` | `before + actions + after`로 구성된 검증 가능한 실행 단위 |
| `recipe_path` | 시작 상태, 순서가 보존된 전이 목록과 완료 상태를 담은 활성 레시피 경로 |
| `active_reflex_recipe` | 선택한 경로 키, 현재 전이 번호, 검증 대기 전이 번호와 전체 전이 수를 보관하는 작업 상태 |
| `task_category` | 검색, 로그인, 결제, 사이트 탐색 같은 작업 카테고리 |
| `page_role` | home, search, job_detail, popup처럼 행동 당시 화면을 설명하는 관측 메타데이터 |
| `recipe_key` | active recipe row의 DB 식별자. `site`, `task_category`와 전체 단계의 순서·의미로 계산 |
| `recipe_params` | 반복 실행 시 주입되는 런타임 입력값 |
| `screen_signature` | 현재 전체 화면 관찰 서명. 기본 replay 판단용 이름으로 쓰지 않는다 |
| `roi_signature` | 타깃 주변 crop의 pHash 서명. active replay 판단의 기준 |
| `screen_context_signature` | 좌표 없는 행동 직전 화면의 축약 pHash 서명. 해당 단계의 active replay 판단 기준 |
| `target_snapshot` | 특정 행동 대상의 text, bbox, ratio, label 등 관찰 스냅샷 |
| `job_card_queue` | 채용 검색 결과에서 수집할 공고 카드 작업 큐 |
| `job_results_memory` | 공고 카드 큐를 만든 검색 결과 화면의 복귀 검증용 기억 |
| `job_detail_buffer` | 공고 상세 화면에서 누적한 OCR 본문 |
| `transition_request` | 직전 행동 묶음이 요청한 화면 전환과 저장된 도착 상태 |
| `transition_result` | 화면 전환 검증의 현재 결과 |
| `transition_records` | 한 작업에서 확정된 화면 전환 결과 목록 |
| `pending_action` | 아직 실행하지 않은 검증된 `ActionRequest` |
| `last_action_result` | 직전에 실행한 요청의 `ActionResult`. 다음 행동을 담지 않는다 |
| `tool_call_metadata` | 큐 ID, 전환 출처처럼 물리 도구 인자가 아닌 실행 추적값 |
| `current_capture_id` | 현재 작업 상태로 채택된 물리 화면 캡처의 실행 내 식별자 |
| `decision_capture_id` | 행동을 선택할 때 근거로 사용한 캡처 식별자 |
| `from_capture_id` / `to_capture_id` | 화면 전환의 이전 캡처와 다음 캡처 식별자 |

## 함수 동사 규칙

| 접두어 | 사용 기준 |
|---|---|
| `normalize_` | 값 비교를 위한 결정론적 정규화 |
| `build_` | 입력으로 새 계약 객체, dict 또는 payload를 조립. 외부 부작용 없음 |
| `extract_` | 원문, OCR, LLM 응답에서 구조화 값을 뽑음 |
| `record_` | 런타임 관찰/행동을 메모리 리스트에 추가 |
| `commit_` | DB에 제출물, 후보, 피드백 같은 기록을 저장 |
| `persist_` | 최종 비즈니스 데이터, 예: 공고 DB 적재 |
| `review_` | LLM/Critic 또는 리뷰 정책으로 판정 |
| `promote_` | candidate를 active recipe로 승격 |
| `match_` | 현재 상태가 저장된 조건과 맞는지 결정론적으로 검사 |
| `select_` | 후보 중 하나를 고름 |
| `dispatch_` | 실제 도구/브라우저/상태 변경을 실행 |
| `render_` | 사람이 읽거나 프롬프트에 넣을 문자열 생성 |
| `load_` | 파일/프로필/설정 로드 |
| `get_` | 단일 항목 조회 |
| `list_` | 여러 항목 조회 |
| `ensure_` | 스키마, 기본값, 불변 조건을 보장하는 idempotent 처리 |

## 접미어 규칙

| 접미어 | 의미 |
|---|---|
| `_id` | DB row, marker, queue item 같은 식별자 |
| `_key` | 비교/조회용 계산 키 |
| `_path` | 로컬 파일 경로 |
| `_url` | URL 문자열 |
| `_json` | JSON 문자열. dict가 아니다 |
| `_payload` | 외부 도구, DB, LLM에 넘기는 구조화 dict |
| `_schema` | Pydantic/DB 스키마 |
| `_evidence` | LLM/Critic 판단에 넘기는 관찰 근거 |
| `_trace` | 디버깅/프로파일링용 실행 흔적 |
| `_result` | 함수/도구 실행 반환값 |
| `_status` | 상태 enum/string |
| `_reason` | 사람이 읽는 사유 문자열 |
| `_mode` | 동작 모드 |
| `_bbox` | 픽셀 좌표 `[x1, y1, x2, y2]` |
| `_ratio` | 화면 크기 대비 0~1 비율 |
| `_signature` | 재사용 가능한 화면/ROI 서명 |

## 모듈 이름 규칙

| 파일명 패턴 | 책임 |
|---|---|
| `*_store.py` | DB CRUD와 마이그레이션만 담당 |
| `*_schema.py` | Pydantic 또는 DB 스키마 정의 |
| `*_runtime.py` | 그래프 실행 중 사용하는 재생/큐/버퍼 런타임 로직 |
| `*_snapshot.py` | state/action에서 관찰 스냅샷 생성 |
| `*_matcher.py` | 결정론적 매칭 |
| `*_reviewer.py` | LLM/Critic 검토 |
| `*_promotion.py` | candidate를 active recipe로 승격 |
| `*_prompt.py` | 프롬프트 문자열 조립 |
| `*_report.py` | 사용자/중간 보고서 생성 |

작업자 그래프 모듈은 `worker_<책임>.py`를 사용한다.

| 모듈 | 책임 |
|---|---|
| `worker_observation.py` | 캡처와 OCR 관찰 |
| `worker_transition.py` | 행동 전후 화면 전환 판정 |
| `worker_collection.py` | 관찰 결과와 상세 본문 상태 반영 |
| `worker_selection.py` | 결정론적 행동 선택 |
| `worker_reflex.py` | 활성 레시피 ROI 검증과 재생 진입 |
| `worker_reasoning.py` | LLM 의미 판단과 행동 계약 생성 |
| `worker_execution.py` | 검증된 행동 요청의 실행 진입점 |
| `worker_execution_dispatch.py` | 원자 도구와 상태 행동 실행 |
| `worker_execution_handlers.py` | 실행 전후 정책과 후속 행동 처리 |
| `worker_execution_context.py` | 한 행동 실행 중 변경되는 상태 조립 |
| `worker_recording.py` | 실행 결과와 학습 증거 기록 |

지휘자 조사 그래프 모듈은 `investigation_<책임>.py`를 사용한다.

| 모듈 | 책임 |
|---|---|
| `investigation_context.py` | 조사 상태, 모델 묶음, 공통 요청 문맥 |
| `investigation_request_nodes.py` | 사용자 요청 해석과 확인 질문 |
| `investigation_evidence_policy.py` | 근거 판정과 수집 단계 정규화 순수 정책 |
| `investigation_evidence_nodes.py` | 필요 근거 정의, DB 충분성 검사, 수집 계획 |
| `investigation_collection_nodes.py` | 확정된 단일 수집 단계 실행 |
| `investigation_answer_nodes.py` | 검증된 문서 조회와 최종 답변 |
| `investigation_workflow.py` | 노드 연결, 체크포인트 중단·재개, 실행 진입 |

채용공고 수집 애플리케이션은 `collection_<책임>.py`를 사용한다.

| 모듈 | 책임 |
|---|---|
| `collection_request_builder.py` | 수집 의도, 사이트 프로필과 작업자 목표 생성 |
| `collection_service.py` | 작업자·검토·저장 순서와 재시도 조율 |
| `collection_worker_runner.py` | 단일 비전 작업자 실행과 실행 결과 구성 |
| `collection_submission_service.py` | 제출물 검토, 저장과 레시피 후보 등록 |

## 피해야 할 이름

- `recipe`: 단독 사용 금지. `recipe_candidate`, `active_recipe`, `site_recipe`, `recipe_step` 중 하나를 쓴다.
- `metadata`: 단독 사용 지양. `skill_metadata`, `target_metadata`, `promotion_metadata`처럼 범위를 붙인다.
- `signature`: 단독 사용 지양. `screen_signature`, `roi_signature`로 구분한다.
- `state_key`: 새 active recipe 코드에서 사용하지 않는다. 저장/조회는 `recipe_key`, `site`, `task_category`, URL 범위와 단계별 `roi_signature` 또는 `screen_context_signature`를 기준으로 한다.
- `similar`, `similarity`: active replay 기본 경로에는 쓰지 않는다. ROI replay는 `roi_phash_distance`, `target_ratio_miss`처럼 명확한 실패 사유를 쓴다.
- `data`, `info`, `item`: 지역 범위가 5줄 이상이면 더 구체적인 이름으로 바꾼다.

## Reflex Recipe 단계 이름

실행 방식은 `autonomous`(자율 탐색)와 `experience_guided`(경험 기반 탐색)로 구분한다. 두 실행의 공통 Recipe 단계 이름은 아래와 같다.

```text
recorded_step
-> worker_submission
-> recipe_candidate
-> candidate_review
-> recipe_transition
-> active_recipe
-> reflex_replay
```

## 테스트 이름

테스트 함수는 `test_<대상>_<조건>_<기대결과>` 형태로 쓴다.

예:

```python
def test_reflex_node_roi_signature_missing_falls_back():
    ...

def test_candidate_promotion_skips_non_target_action():
    ...
```

## 기존 코드 정리 우선순위

1. [완료] `target_snapshot` 생성 로직을 `agent/vision/target_snapshot.py`로 공통화한다.
2. [완료] `nodes.py`를 책임별 `worker_*` 모듈로 분리하고 원본 파일을 삭제한다.
3. [완료] `candidate_reviewer.py`에서 promotion 로직을 `candidate_promotion.py`로 분리한다.
4. [완료] `RecipeStore` 조회를 `recipe_key + site + task_category + URL 범위 + 단계별 화면 서명` 기준으로 정리한다.
5. [완료] `investigation_workflow.py`를 조립부와 요청·근거·수집·답변 노드로 분리한다.
6. [완료] 모델 변환과 좌표 계산을 `utils/model_dump.py`, `vision/marker_geometry.py`, `vision/target_snapshot.py`로 통합한다.
7. [완료] 수집 요청, 작업자 실행과 제출물 처리를 애플리케이션 서비스로 분리한다.

입력 검증용 bbox, OCR 줄 병합 bbox, pHash 레거시 비율 복원처럼 의미가 다른 함수는 이름이 비슷하다는 이유만으로 합치지 않는다.
