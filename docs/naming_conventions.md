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
| `worker_submission` | 자율탐색 worker가 끝난 뒤 지휘자/리뷰어에 제출하는 구조화 결과 |
| `recorded_step` | 자율탐색 중 실제 실행된 원본 행동 기록 |
| `recipe_candidate` | 아직 active가 아닌 예비 레시피 후보 |
| `candidate_review` | Critic이 후보를 검토한 결과 |
| `promotion` | 후보를 active recipe로 승격하는 처리 결과 |
| `active_recipe` | 반복탐색에서 실제 재생 가능한 활성 레시피 |
| `recipe_step` | 레시피 안의 단일 행동 단계 |
| `replay_step` | Critic이 fixed/parameterized로 승인한 재생 단계 |
| `task_category` | 검색, 로그인, 결제, 사이트 탐색 같은 작업 카테고리 |
| `recipe_params` | 반복 실행 시 주입되는 런타임 입력값 |
| `screen_signature` | 현재 전체 화면 관찰 서명. 기본 replay 판단용 이름으로 쓰지 않는다 |
| `roi_signature` | 타깃 주변 crop의 pHash 서명. active replay 판단의 기준 |
| `target_snapshot` | 특정 행동 대상의 text, bbox, ratio, label 등 관찰 스냅샷 |
| `result_card_queue` | 검색 결과 목록에서 수집할 카드 작업 큐 |
| `result_page_memory` | 카드 큐를 만든 검색 결과 페이지 복귀 검증용 기억 |
| `detail_ocr_buffer` | 상세페이지 OCR 본문 누적 버퍼 |

## 함수 동사 규칙

| 접두어 | 사용 기준 |
|---|---|
| `normalize_` | 값 비교를 위한 결정론적 정규화 |
| `build_` | 입력으로 새 dict/message/payload를 조립. 외부 부작용 없음 |
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

## 피해야 할 이름

- `recipe`: 단독 사용 금지. `recipe_candidate`, `active_recipe`, `site_recipe`, `recipe_step` 중 하나를 쓴다.
- `metadata`: 단독 사용 지양. `skill_metadata`, `target_metadata`, `promotion_metadata`처럼 범위를 붙인다.
- `signature`: 단독 사용 지양. `screen_signature`, `roi_signature`로 구분한다.
- `state_key`: active recipe 식별자로 단독 사용하지 않는다. 저장/조회에는 `site`, `task_category`와 함께 다룬다.
- `similar`, `similarity`: active replay 기본 경로에는 쓰지 않는다. ROI replay는 `roi_phash_distance`, `target_ratio_miss`처럼 명확한 실패 사유를 쓴다.
- `data`, `info`, `item`: 지역 범위가 5줄 이상이면 더 구체적인 이름으로 바꾼다.

## Reflex Recipe 단계 이름

자율탐색과 반복탐색 경계는 아래 이름을 쓴다.

```text
recorded_step
-> worker_submission
-> recipe_candidate
-> candidate_review
-> replay_step
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

1. `target_snapshot` 생성 로직을 공통화한다.
2. `nodes.py`에서 `result_card_queue`, `detail_ocr_buffer`, `reflex_runtime`, `action_executor`를 분리한다.
3. `candidate_reviewer.py`에서 promotion 로직을 `candidate_promotion.py`로 분리한다.
4. `RecipeStore`를 `site + task_category + state_key` 기준 이름과 스키마로 정리한다.
5. 남은 `_dump_model`, `_bbox`, `_center` 같은 작은 중복 유틸을 공통 모듈로 옮긴다.
