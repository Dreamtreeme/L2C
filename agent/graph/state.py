import operator
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from pathlib import Path

class GraphState(TypedDict):
    """
    LangGraph에서 노드 간에 전달되는 상태 스키마입니다.
    """
    # 사용자의 원래 목표 명령
    goal: str
    
    # 현재 화면에서 추출된 UI 요소 목록 (텍스트)
    ui_context: str

    # 현재 활성 브라우저 URL
    current_url: str

    # 현재 URL 캐시가 브라우저 실제 주소와 달라졌을 가능성
    current_url_stale: bool

    # 원본 마커 데이터 (ID 매핑용)
    current_markers: List[Dict[str, Any]]
    
    # 행동 이력 (최근 수행한 도구 및 결과)
    # Annotated와 operator.add를 사용하여 상태 업데이트 시 리스트가 누적되도록 합니다.
    action_history: Annotated[List[Dict[str, Any]], operator.add]
    
    # 최근 캡처된 이미지 경로들 (디버깅/기록용)
    recent_images: Annotated[List[Path], operator.add]
    
    # 최근 마킹된 이미지 경로 (SoM VLM 추론용)
    marked_image: str
    
    # 에러가 발생한 횟수
    error_count: int
    
    # 수집 완료 여부 플래그
    is_finished: bool
    
    # 수집 완료된 데이터
    collected_data: List[Any]

    # 현재까지 누적 수집된 채용공고 정보 (스크롤 간 정보 보존용)
    extracted_jd: Dict[str, Any]

    # 가장 최근 LLM의 판단 결과 저장 (AIMessage 객체 등)
    last_action_result: Any

    # 대목표 아래 소목표 계획 목록
    plan: List[str]

    # 현재 실행 중인 계획 단계 인덱스
    current_plan_step: int

    # 각 노드의 실행 시간 기록 (디버깅/성능 측정용)
    # Annotated + operator.add 로 노드마다 append되어 누적됩니다.
    step_durations: Annotated[List[Dict[str, Any]], operator.add]

    # 마지막 action이 화면 전환/렌더링 변화를 유발했는지 여부
    last_action_screen_changed: bool

    # [Reflex Recipe / Phase0] 비전 런 중 기록된 (상태->타깃) 스텝. operator.add로 누적.
    recorded_steps: Annotated[List[Dict[str, Any]], operator.add]

    # [Feedback Loop] 행동 제안 -> 실행 -> 관찰 -> 1차 피드백 에피소드. operator.add로 누적.
    feedback_episodes: Annotated[List[Dict[str, Any]], operator.add]

    # [Reflex Recipe] 현재 perception 결과로 계산한 화면-상태 키
    reflex_state_key: str

    # [Reflex Recipe] 직전 reflex_node가 reasoning을 우회했는지 여부
    reflex_hit: bool

    # [Reflex Recipe] reflex tool_call id별 행동 후 전환 계약
    reflex_transition_contracts: Dict[str, Any]

    # [Reflex Recipe] 재생 시 치환할 입력값(recipe_params)
    recipe_params: Dict[str, Any]

    # [Transition Contract] 다음 perception이 검증할 직전 화면 변경 행동
    pending_transition: Dict[str, Any]

    # [Transition Contract] ready / pending / unknown
    transition_status: str
    transition_outcome: str
    transition_source: str

    # [Transition Contract] 행동 뒤 관찰된 OCR 화면 기록
    transition_observations: Annotated[List[Dict[str, Any]], operator.add]

    # [Page Policy] 상세페이지 자동 스크롤 반복 횟수
    detail_auto_scroll_count: int

    # [Page Policy] OCR observations accumulated inside a detail page before the next reasoning call
    detail_page_observations: List[Dict[str, Any]]

    # [HITL] Stop autonomous execution before sensitive or irreversible steps
    pending_human_approval: bool
    human_approval_request: Dict[str, Any]
