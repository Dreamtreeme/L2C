# L2C 반사 레시피(Reflex Recipe) 구현 계획

> 네이밍 기준은 `docs/naming_conventions.md`를 따른다. 특히 `recorded_step`, `recipe_candidate`, `replay_step`, `active_recipe`, `roi_signature`, `task_category`를 구분해서 쓴다.

## 1. 목표
- **새 사이트**: 비전 ReAct 에이전트가 1회 개척 (현행 그대로).
- **본 적 있는 화면-상태**: 추론 LLM을 건너뛰고, OCR/마커 직후 캐시된 규칙으로 *반사적으로* 마커를 클릭.
- DOM·Playwright·절대좌표 저장 안 함. 전부 **OCR/마커(텍스트) 공간**에 머문다.
- 현재 코드 최대 재사용 — `action_node` 실행 경로는 그대로 두고 "결정(reasoning)"만 캐시로 대체.

핵심 효과: 캐시 히트 시 한 스텝이 `(OCR + LLM추론 + 클릭)` → `(OCR + 클릭)`. 우리가 프로파일한 추론 비중(~50–65%)과 스텝당 API 비용이 사라진다. (perception은 그대로 도므로 "1초"가 아니라 대략 절반.)

## 2. 비목표 (v1 범위 밖)
- DOM 셀렉터 추출 / Classic Playwright 전환 — 비전 전제 위반이라 제외.
- 완전 파라미터화 템플릿, 자가 치유, 아이콘 캡션 — Phase 2~3로 미룸.

## 3. 재사용 맵 (기존 자산 → 계획에서의 역할)
| 기존 코드 | 역할 |
|---|---|
| `perception_node` → `current_markers {id,bbox,text}` | 반사의 입력. 매 스텝 그대로 OCR/마커 생성 |
| `reasoning_node` (`_get_ui_llm_with_tools`, gemini-3.5-flash) | **캐시 미스 때만** 호출 (개척·폴백) |
| `action_node` (`tool_calls` 실행, 내부 `get_bbox`) | 기존 실행 경로를 재사용하고 화면 변경 행동을 `pending_transition`으로 기록 |
| `should_continue` / `add_conditional_edges` | 반사 분기 라우터 패턴 그대로 차용 |
| `Preprocessor.clean_text` | 마커 텍스트 정규화(매칭용) |
| `Database` / `DB_PATH` / UPSERT·마이그레이션 | 레시피 저장소(테이블만 추가) |
| `GraphState` + `operator.add` 리듀서 | 기록 누적 필드 추가 |
| `step_durations` + `benchmark/profile_steps.py`,`profile_run.py` | 효과 측정(LLM 호출수·스텝시간) |
| `shared/schema/jd_schema.py` (JobPosting) | `SiteRecipe` Pydantic을 같은 자리에 추가 |

## 4. 신규 요소 (작게)
1. **`shared/schema/recipe_schema.py`** — Pydantic
   - `RecipeStep`: `{ state_key, target:{text?, bbox_ratio?, center_ratio?}, roi_signature?, action, param?, transition_contract? }`
   - `TransitionContract`: `{ common_ready_cues, outcomes, loading_cues?, timeout_sec }`
   - `SiteRecipe`: `{ site, steps: list[RecipeStep], success_count, updated_at }`
2. **`agent/recipe/state_key.py`** — `compute_state_key(url, markers) -> str`
   - 정규화된 *앵커 마커 텍스트 집합*으로 현재 화면의 레시피 조회 키를 계산한다. 레시피 후보 조회용 키이며, 최종 재생 판정은 ROI 검증이 맡는다.
3. **`agent/recipe/phash_replay.py`** — `match_step_by_screen_signature(step, signature, markers, current_image_path) -> marker_id | result`
   - 저장된 `roi_signature.crop_rect_ratio`로 현재 화면의 같은 영역을 자르고 ROI pHash 거리를 비교한다.
   - ROI가 같으면 `target.center_ratio`에 가까운 현재 OCR marker를 찾아 클릭/입력한다. 실패 시 즉시 reasoning으로 폴백한다.
4. **`agent/recipe/store.py`** — `Database` 위 thin wrapper
   - `get_recipe(state_key)`, `commit_recipe(...)`.
   - DB에 `recipes` 테이블(`state_key`, `steps_json`, `skill_metadata_json`, `success_count`, `updated_at`) 추가 — 기존 동적 ALTER 마이그레이션 패턴 그대로.
5. **`reflex_node` + 라우터** — `workflow.py`
   - `perception → [reflex | reasoning]` 조건부 엣지로 교체.
   - `reflex_node`: 현재 `state_key`와 `task_category`로 active recipe 조회 → ROI pHash/target center ratio 검증 → 성공 시 `click_marker`/`type_in_marker` `tool_calls`를 담은 `AIMessage`를 `last_action_result`에 세팅(= reasoning 출력과 동일 형태). 미스/검증실패 → reasoning.
6. **`GraphState` 신규 필드**: `recorded_steps`, `reflex_state_key`, `reflex_hit`, `pending_transition`, `transition_status`, `transition_observations`.

## 5. 페이즈

### Phase 0 — 기록 전용 (실행 흐름 무변경, 다크 출시)
- `action_node`의 UI 액션 디스패치 직후에 **기록 훅**: `click_marker`/`type_in_marker` 실행 시 `{ state_key=compute_state_key(url, markers), target=클릭한 마커 text/bbox_ratio/center_ratio, roi_signature, action, param=goal 역매핑 }`을 `recorded_steps`에 append.
- 그래프 종료(`finish_task`) 시 accepted worker submission을 `recipe_candidates`에 저장한다. active `recipes` 승격은 후처리 Critic/replay 검토에서만 수행한다.
- 리스크 0(분기·동작 불변). 산출물: 실데이터로 상태키·타깃 텍스트 분포 검증.

### Phase 1 — 반사 재생 (플래그 `REFLEX_ENABLED`)
- `workflow.py`: `perception→reasoning` 무조건 엣지를 조건부로 교체 + `reflex_node` 삽입.
- 캐시 히트 → reasoning(LLM) 스킵 → `action_node`가 동일하게 실행.
- **검증→폴백**: 반사 행동 후 다음 perception이 OCR 전환 계약을 검사한다. `pending`이면 다시 관찰하고, `ready`이면 다음 Reflex를 시도하며, `unknown` 또는 시간 초과면 reasoning으로 강등한다.
- `state_key`는 행동 전 레시피 조회에만 사용하고, 행동 후 성공 여부는 전체 해시 일치가 아니라 부분 OCR 단서와 정상 결과 분기로 판정한다.
- 측정: `profile_run.py`/`profile_steps.py`로 반사 적중률·런당 LLM 호출수·스텝시간 비교(추론 항목 소거 확인).

### Phase 2 — LLM 태깅 의미 + 파라미터 템플릿
- 개척(reasoning) 시 LLM이 텍스트로 함께 산출: ① 행동 의도, ② 어느 입력이 파라미터였는지, ③ 기대 화면 변화.
- 실행 후 `transition_observations`를 Critic이 검토해 OCR로 검사 가능한 공통 완료 단서와 관찰된 정상 결과 분기를 구조화한다. 보지 않은 결과는 추측해서 추가하지 않는다.
- 레시피를 *템플릿화*: "같은 사이트 다른 공고"가 히트하도록 파라미터 슬롯([회사명] 등) 일반화. (LLM이 goal을 아니까 가변/고정 스텝 귀납 가능.)

### Phase 3 — 자가 치유 + 텍스트 없는 타깃
- 레시피 step에 confidence/success_count 부여, 폴백 발생 시 강등·재생성.
- 아이콘만 있는 타깃(돋보기 등)은 해당 마커에 한해 VLM 캡션(현재 `SKIP_VLM_CAPTION=true` 부분 해제) 또는 위치 휴리스틱.

## 6. 리스크 / 완화
- **텍스트 매칭 브리틀** → 항상 검증 + LLM 폴백 안전망. 반사는 reasoning을 *대체*가 아니라 *우회*.
- **상태키 드리프트**(지원자 수 등 동적 텍스트) → 앵커는 안정 substring/구조 마커 우선, `clean_text` 정규화.
- **모호 타깃**("검색"이 여러 개) → region/ordinal 보조키 + 리터럴 vs 파라미터 구분 저장.

## 7. 성공 지표 (전부 기존 계측으로)
- 반사 적중률(%) ↑ · 런당 gemini 호출수 ↓ · 추출 1건당 벽시계·API$ ↓ — `step_durations` + `benchmark/profile_steps.py`로 측정.

## 8. 의존성 순서
`recipe_schema` → `state_key` + `roi_signature` → `store`(+DB 마이그레이션) → Phase 0 기록 훅 → Phase 1 `reflex_node`/라우터 → 후보 저장/후처리 승격 → 측정 → Phase 2 → Phase 3.

## Current implementation update: worker submission review gate

The current implementation no longer promotes a successful autonomous run directly into the active `recipes` table.

The revised flow is:

1. The child vision worker runs the existing `reasoning -> action -> perception/reflex` graph.
2. Each action still records `feedback_episodes` and `recorded_steps` as evidence.
3. At run end, `realtime_scraping` builds a structured `WorkerSubmission`.
4. The submission is shape-validated first. This script layer checks only required structure and observable facts, not semantic recipe quality.
5. `CommanderReview` decides `accept`, `revise`, or `reject`. Semantic LLM review can be enabled with `VISION_WORKER_REVIEW_MODE=llm`; otherwise the shape review is used.
6. If the review says `revise`, the next worker attempt receives `feedback_to_worker` in its goal prompt.
7. Only accepted submissions persist collected job data. Reflex recipe activation is deferred to a later Critic/Memory promotion step and replay test.

This keeps the original principle: code guards structure and safety, while meaning-level decisions such as wrong target, reusable parameter, and recipe candidacy are handled by the feedback loop instead of site-specific hard-coded promotion rules.

### Stored evidence roles

`feedback_episodes` is the event log for action-level evidence. It is optimized for later analysis by run, site, action, and feedback label.

`worker_submissions.payload_json.feedback_episodes` is the immutable submission snapshot that the commander reviewed. It intentionally duplicates the run evidence so a review decision can be audited even if the event log is queried or compacted differently later.

`recipe_candidates` stores accepted submissions that the commander marked as replay candidates. These rows are pending Critic review only; they do not activate Reflex behavior by themselves and do not write to the active `recipes` table.

The candidate promotion gate is LLM-led. Code only packages the candidate row, worker submission, recorded steps, and previous commander review into the `RecipeCandidateReview` prompt shape. The Critic assigns every step one replay mode:

- `fixed`: the same stable UI operation is valid across runs.
- `parameterized`: the operation is stable and only named runtime slots such as `query` change.
- `reasoning`: the choice depends on the current screen, current result set, visited items, or remaining target count.

Only `fixed` and `parameterized` steps are written to active `recipes`. A job title observed during exploration is evidence, not a reusable target, so current search-result card selection remains a reasoning step unless a current runtime title is explicitly supplied. Re-promoting a candidate replaces the state rows owned by that candidate so an older specific-card recipe cannot remain active.

`VISION_RECIPE_LEARNING_MODE` controls only whether the worker stores replay candidates during collection:

- `off`: do not store recipe candidates.
- `record`: store accepted candidates only. This is the default.

Critic review and active recipe promotion are separated into a post-run step. This keeps the worker focused on collection and makes newly recorded evidence available from the next run only after explicit review/promotion.

## Top-level commander graph

`agent.graph.commander_workflow` is the explicit LangGraph orchestration layer above the child vision worker. Its route is:

`plan_sites -> select_site -> run_worker -> review_submission -> (prepare_retry | persist_accepted | mark_failed) -> select_site -> query_db -> summarize`.

`COMMANDER_GRAPH_ENABLED=1` routes the existing QA entry point through this graph. The default path still uses the previous QA tool-calling loop until the graph path is exercised enough to become the default.

### Batch candidate review tool

When `VISION_RECIPE_LEARNING_MODE=record`, accepted worker submissions are stored as `recipe_candidates` with `pending_replay` status only. They are not reviewed or promoted during the collection run.

The commander can later call `review_recipe_candidates`:

- `mode="review"`: load stored candidates, send each one through the Critic LLM gate, and store the Critic decision in `validation_json`. Active `recipes` are not modified even if the Critic returns `accept`.
- `mode="promote"`: run the same Critic gate and allow only `accept` plus `promote_to_active_recipe=true` to write the candidate steps into active `recipes`.

If candidates were already dry-reviewed with `mode="review"`, call the tool with `status="accepted"` when doing a later promotion pass.

This keeps script code limited to candidate selection and result persistence. Reuse quality, wrong-target judgment, parameter/generic-step judgment, and promotion are still feedback-loop decisions made by the Critic.
