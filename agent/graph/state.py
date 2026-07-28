import operator
from typing import TypedDict, List, Dict, Any, Annotated

from agent.graph.action_request import ActionRequest, ActionResult


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

    # [Reflex Recipe / Phase0] 비전 런 중 기록된 UI 행동/타깃 ROI 스텝. operator.add로 누적.
    recorded_steps: Annotated[List[Dict[str, Any]], operator.add]

    # [Feedback Loop] 행동 제안 -> 실행 -> 관찰 -> 1차 피드백 에피소드. operator.add로 누적.
    feedback_episodes: Annotated[List[Dict[str, Any]], operator.add]

    # [Reflex Recipe] 직전 reflex 후보 선택/매칭/실패 원인 추적 정보
    reflex_trace: Dict[str, Any]

    # [Reflex Recipe] reflex tool_call id별 행동 후 전환 계약
    reflex_transition_contracts: Dict[str, Any]

    # [Reflex Recipe] 같은 worker run 안에서 전환 검증에 실패해 재시도하지 않을 recipe_key 목록
    reflex_blocked_recipe_keys: List[str]

    # [Reflex Recipe] 재생 시 치환할 입력값(recipe_params)
    recipe_params: Dict[str, Any]

    # [Transition Contract] 다음 perception이 검증할 직전 화면 변경 행동
    pending_transition: Dict[str, Any]

    # [Transition Contract] ready / pending / unknown
    transition_status: str
    transition_outcome: str
    transition_source: str
    transition_reason: str
    transition_visual_change_detected: bool
    transition_visual_change_ratio: float | None
    ocr_required: bool
    observed_transition: Dict[str, Any]

    # [Transition Contract] 행동 뒤 관찰된 OCR 화면 기록
    transition_observations: Annotated[List[Dict[str, Any]], operator.add]

    # [Result Card Queue] 한 worker 실행 안에서만 쓰는 검색 결과 카드 큐
    result_card_queue: List[Dict[str, Any]]
    result_page_memory: Dict[str, Any]
    active_result_card: Dict[str, Any]
    queue_replay_trace: Dict[str, Any]
    result_card_selector_trace: Dict[str, Any]

    # [Result Availability] 화면에서 모델이 판독한 검색 결과 총개수와 근거
    result_availability: Dict[str, Any]

    # [Page Policy] 상세 페이지처럼 구조가 안정적인 반복 읽기 흐름에서 LLM 판단을 우회한 액션
    page_policy_trace: Dict[str, Any]

    # [Detail OCR Buffer] 상세 페이지 OCR을 화면별로 누적하고 마지막에 한 번 정제한다.
    detail_ocr_buffer: Dict[str, Any]

    # [Detail Source Follow-up] 본문이 부족해 원문 이동 또는 추가 공개가 필요한 상태
    detail_followup_required: Dict[str, Any]

    # [Detail Return] 상세 수집을 마친 뒤 검색 결과 화면으로 복귀해야 하는 상태
    detail_return_pending: Dict[str, Any]

    # [HITL] Stop autonomous execution before sensitive or irreversible steps
    pending_human_approval: bool
    human_approval_request: Dict[str, Any]
