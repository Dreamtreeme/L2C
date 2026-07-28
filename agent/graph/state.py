import operator
from typing import TypedDict, List, Dict, Any, Annotated

from agent.graph.action_request import ActionRequest, ActionResult


class TransitionRequest(TypedDict, total=False):
    """화면 변경 행동 뒤 다음 캡처에서 확인할 전환 요청."""

    action_seq: int
    action: str
    from_capture_id: str
    expected_after: str
    source: str
    recipe_key: str
    strategy_key: str
    tool_call_id: str
    step: Dict[str, Any]
    before_url: str
    before_phash: str
    before_screenshot: str
    pending_screen_phash: str
    pending_screenshot: str
    pending_target_phash: str
    pending_target_max_distance: int
    started_at: float
    attempts: int
    contract: Dict[str, Any]
    params: Dict[str, Any]


class TransitionResult(TransitionRequest, total=False):
    """전환 요청과 현재 캡처를 비교한 판정 결과."""

    status: str
    outcome: str
    reason: str
    visual_change_detected: bool
    visual_change_ratio: float | None
    needs_ocr: bool


class GraphState(TypedDict, total=False):
    """
    LangGraph에서 노드 간에 전달되는 상태 스키마입니다.
    """
    # 작업자 실행과 화면 캡처를 연결하는 식별자
    worker_run_id: str
    worker_attempt_index: int
    current_capture_id: str
    capture_sequence: int

    # 사용자의 원래 목표 명령
    goal: str

    # 최근 캡처와 OCR 관찰 결과
    current_screenshot: str
    capture_quality: Dict[str, Any]
    raw_screen_signature: Dict[str, Any]
    analysis_mode: str
    ocr_complete: bool
    previous_screen_observation: Dict[str, Any]

    # 현재 화면에서 추출된 UI 요소 목록 (텍스트)
    ui_context: str

    # 현재 활성 브라우저 URL
    current_url: str

    # 현재 화면의 대략적 역할. Reflex replay 적용 조건으로만 쓴다.
    current_page_role: str

    # 현재 URL 캐시가 브라우저 실제 주소와 달라졌을 가능성
    current_url_stale: bool

    # 로딩/빈 화면이라 OCR과 LLM 판단을 건너뛰어야 하는지 여부
    low_information_screen: bool
    low_information_capture_count: int

    # 원본 마커 데이터 (ID 매핑용)
    current_markers: List[Dict[str, Any]]
    
    # 행동 이력 (최근 수행한 도구 및 결과)
    # Annotated와 operator.add를 사용하여 상태 업데이트 시 리스트가 누적되도록 합니다.
    action_history: Annotated[List[Dict[str, Any]], operator.add]
    
    # 최근 캡처된 이미지 경로들 (디버깅/기록용)
    recent_images: Annotated[List[str], operator.add]
    
    # 최근 마킹된 이미지 경로 (SoM VLM 추론용)
    marked_image: str

    # 현재 화면 pHash/OCR 앵커 서명. Reflex replay에서 같은 화면인지 검증한다.
    screen_signature: Dict[str, Any]
    
    # 에러가 발생한 횟수
    error_count: int
    
    # 수집 완료 여부 플래그
    is_finished: bool
    
    # 수집 완료된 데이터
    collected_data: List[Any]

    # 현재까지 누적 수집된 채용공고 정보 (스크롤 간 정보 보존용)
    extracted_jd: Dict[str, Any]

    # 다음 실행 노드가 처리할 검증된 행동 요청
    pending_action: ActionRequest | None

    # 가장 최근 행동 요청의 실행 결과
    last_action_result: ActionResult | None

    # 실행 노드가 기록 노드에 넘기는 직렬화 가능한 행동 결과 묶음
    execution_records: List[Dict[str, Any]]

    # 비전 실행 중 기록된 UI 행동과 타깃 ROI 단계
    recorded_steps: Annotated[List[Dict[str, Any]], operator.add]

    # 행동 제안, 실행, 관찰을 연결한 피드백 기록
    feedback_episodes: Annotated[List[Dict[str, Any]], operator.add]

    # 직전 Reflex 후보 선택과 실패 원인
    reflex_trace: Dict[str, Any]

    # 직전 문맥 후속 행동의 조회와 선택 결과
    followup_action_trace: Dict[str, Any]

    # Reflex 도구 호출별 행동 후 전환 계약
    reflex_transition_contracts: Dict[str, Any]

    # 한 번 선택한 Reflex 안정 경로의 키와 다음 실행 단계
    active_reflex_recipe: Dict[str, Any]

    # 같은 작업에서 전환 검증에 실패해 다시 쓰지 않을 레시피 키
    reflex_blocked_recipe_keys: List[str]

    # 레시피 재생 때 치환할 입력값
    recipe_params: Dict[str, Any]

    # 상세 판독, 최종 추출과 저장 검증이 함께 쓰는 필수 필드 계약
    job_collection_contract: Dict[str, Any]

    # 행동 뒤 확인할 전환 요청, 현재 판정 결과, 누적 기록
    transition_request: TransitionRequest
    transition_result: TransitionResult
    transition_records: Annotated[List[Dict[str, Any]], operator.add]
    transition_probe_unchanged: bool

    # 한 작업 안에서만 쓰는 공고 카드 큐와 검색 결과 화면 기억
    job_card_queue: List[Dict[str, Any]]
    job_results_memory: Dict[str, Any]
    active_job_card: Dict[str, Any]
    job_card_replay_trace: Dict[str, Any]
    job_card_selection_trace: Dict[str, Any]

    # 화면에서 모델이 판독한 공고 검색 결과 수와 근거
    job_results_availability: Dict[str, Any]

    # 안정적인 공고 화면 흐름에서 LLM 판단을 우회한 정책 기록
    job_page_policy_trace: Dict[str, Any]

    # 공고 상세 OCR을 화면별로 누적하고 마지막에 한 번 정제한다.
    job_detail_buffer: Dict[str, Any]

    # 상세 화면별로 LLM이 확인한 필드 근거와 누락 상태
    job_detail_coverage: Dict[str, Any]

    # 본문이 부족해 원문 이동 또는 추가 공개가 필요한 상태
    job_detail_followup: Dict[str, Any]

    # 상세 수집 뒤 공고 검색 결과 화면으로 복귀해야 하는 상태
    return_to_job_results: Dict[str, Any]

    # 민감하거나 되돌릴 수 없는 행동 전 사용자 승인을 기다리는 상태
    pending_human_approval: bool
    human_approval_request: Dict[str, Any]
