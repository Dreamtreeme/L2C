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
