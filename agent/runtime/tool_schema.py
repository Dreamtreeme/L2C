"""브라우저 작업자가 LLM과 실행기에 공유하는 도구 입력 스키마."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from shared.schema.skill_schema import RecipeInputName


class click_marker(BaseModel):
    """화면의 일반 조작 마커를 클릭합니다. 검색 결과 공고는 set_job_card_queue를 사용합니다."""

    marker_id: int = Field(..., description="클릭할 마커의 ID")
    target_label: Optional[str] = Field(
        None, description="선택한 항목의 보이는 제목(target_label)"
    )
    target_role: Optional[str] = Field(
        None, description="목표 기준 대상 역할(target_role)"
    )
    target_component: Optional[str] = Field(
        None, description="화면 구성요소(target_component)"
    )
    reason: Optional[str] = Field(None, description="이 대상을 선택한 이유(reason)")
    expected_after: Optional[str] = Field(
        None, description="클릭 후 정상이라면 보여야 할 화면 변화(expected_after)"
    )
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(
        None, description="safe_read, safe_navigation, or sensitive."
    )


class type_in_marker(BaseModel):
    """특정 ID의 마커를 클릭한 후 텍스트를 입력합니다."""

    marker_id: int = Field(
        ...,
        description=(
            "텍스트를 입력할 긴 입력 영역 또는 그 안의 placeholder 텍스트 마커 ID. "
            "닫기·검색 같은 작은 아이콘 마커를 선택하지 마십시오."
        ),
    )
    text: str = Field(..., description="입력할 텍스트")
    target_label: Optional[str] = Field(
        None, description="입력 영역의 보이는 라벨(target_label)"
    )
    slot_name: Optional[RecipeInputName] = Field(
        None,
        description="검색어를 재생할 때 사용하는 search_keyword 입력 이름",
    )
    target_role: Optional[str] = Field(
        None, description="목표 기준 대상 역할(target_role)"
    )
    target_component: Optional[str] = Field(
        None, description="화면 구성요소(target_component)"
    )
    reason: Optional[str] = Field(None, description="이 입력을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(
        None, description="입력 후 정상이라면 보여야 할 화면 변화(expected_after)"
    )
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(
        None, description="safe_read, safe_navigation, or sensitive."
    )


class scroll(BaseModel):
    """화면을 스크롤합니다."""

    direction: Literal["down", "up", "left", "right"] = Field(
        "down",
        description="스크롤 방향 ('down', 'up', 'left', 'right')",
    )
    marker_id: Optional[int] = Field(
        None,
        description=(
            "내부 패널·목록처럼 특정 영역을 스크롤할 때 기준으로 삼을 마커 ID. "
            "도구가 이 마커 위로 포인터를 옮긴 뒤 클릭 없이 스크롤합니다. "
            "생략하면 전체 페이지를 스크롤합니다."
        ),
    )
    amount: Literal["small", "page"] = Field(
        "page",
        description=(
            "스크롤 이동량. 연속된 본문을 읽을 때는 인접 화면이 겹치는 'small', "
            "결과 목록을 빠르게 넘길 때는 'page'를 사용합니다."
        ),
    )
    target_label: Optional[str] = Field(
        None, description="스크롤할 영역의 보이는 라벨(target_label)"
    )
    target_role: Optional[str] = Field(
        None, description="스크롤 대상 역할(target_role)"
    )
    target_component: Optional[str] = Field(
        None, description="스크롤 대상 화면 구성요소(target_component)"
    )
    reason: Optional[str] = Field(None, description="스크롤을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(
        None, description="스크롤 후 정상이라면 보여야 할 화면 변화(expected_after)"
    )
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(
        "safe_read",
        description="스크롤은 화면 내용을 더 읽는 읽기 전용 행동입니다.",
    )


class press_key(BaseModel):
    """엔터, ESC 등 특수키를 누릅니다."""

    key: str = Field(..., description="누를 특수키 (예: 'enter', 'esc')")
    reason: Optional[str] = Field(None, description="키 입력을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(
        None, description="키 입력 후 정상이라면 보여야 할 화면 변화(expected_after)"
    )
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(
        None, description="safe_read, safe_navigation, or sensitive."
    )


class open_browser(BaseModel):
    """브라우저를 열고 지정한 URL에 접속합니다."""

    url: str = Field(..., description="접속할 URL (예: https://www.wanted.co.kr)")
    reason: Optional[str] = Field(None, description="이 URL을 여는 이유(reason)")
    expected_after: Optional[str] = Field(
        None,
        description="브라우저 이동 후 정상이라면 보여야 할 화면 변화(expected_after)",
    )
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(
        None, description="safe_read, safe_navigation, or sensitive."
    )


class close_current_tab(BaseModel):
    """현재 활성 브라우저 탭 하나를 닫습니다."""

    reason: Optional[str] = Field(None, description="현재 탭을 닫는 이유(reason)")
    expected_after: Optional[str] = Field(
        None, description="탭을 닫은 뒤 기대 상태(expected_after)"
    )
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(
        None, description="safe_read, safe_navigation, or sensitive."
    )


class switch_tab(BaseModel):
    """현재 브라우저 창에서 인접한 탭으로 전환합니다."""

    direction: Literal["next", "previous"] = Field(
        ...,
        description="전환할 탭 방향 ('next' 또는 'previous')",
    )
    reason: Optional[str] = Field(None, description="탭을 전환하는 이유(reason)")
    expected_after: Optional[str] = Field(
        None, description="탭 전환 뒤 기대 상태(expected_after)"
    )
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(
        None, description="safe_read, safe_navigation, or sensitive."
    )


class review_job_detail(BaseModel):
    """누적한 상세 OCR이 저장 가능한지 검토하도록 요청합니다."""

    reason: Optional[str] = Field(None, description="현재 근거를 검토할 이유(reason)")
    expected_after: Optional[str] = Field(
        None, description="검토 뒤 기대되는 상태(expected_after)"
    )
    page_role: Optional[str] = Field("job_detail", description="Current page role.")
    risk_level: Optional[str] = Field(
        "safe_read", description="safe_read, safe_navigation, or sensitive."
    )


class go_back(BaseModel):
    """브라우저의 뒤로가기 기능을 실행합니다."""

    reason: Optional[str] = Field(None, description="뒤로가기를 수행한 이유(reason)")
    expected_after: Optional[str] = Field(
        None, description="뒤로가기 후 정상이라면 보여야 할 화면 변화(expected_after)"
    )
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(
        None, description="safe_read, safe_navigation, or sensitive."
    )


class VisibleJobCard(BaseModel):
    """현재 화면에서 실제로 보이는 공고 카드 하나."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    marker_id: int = Field(..., ge=0)
    title: str = Field(..., min_length=1)
    company: str = Field(
        "",
        description=(
            "공고 제목과 인접해 별도로 표시된 회사명. "
            "제목 괄호 안의 직무 분야는 회사명으로 사용하지 않는다."
        ),
    )


class set_job_card_queue(BaseModel):
    """검색 결과의 수집 대상 공고를 선택해 런타임 작업 큐에 저장합니다."""

    cards: List[VisibleJobCard] = Field(
        default_factory=list,
        description=(
            "수집할 공고 카드 목록. 각 항목에는 현재 캡처의 marker_id, title, company를 넣으십시오. "
            "현재 화면에 보이는 카드만 넣어야 합니다."
        ),
    )
    available_job_count: Optional[int] = Field(
        None,
        ge=0,
        description="화면에 명시된 전체 검색 결과 개수. 숫자의 의미가 확실할 때만 입력합니다.",
    )
    count_evidence: Optional[str] = Field(
        None, description="전체 결과 개수를 판단한 화면 문구"
    )
    reason: Optional[str] = Field(None, description="이 카드들을 큐에 넣은 이유")


class finish_task(BaseModel):
    """작업을 완료하고 최종 데이터를 반환합니다."""

    result: str = Field(..., description="최종 완료 요약 또는 결과 데이터")


ACTION_TOOL_SCHEMAS = {
    schema.__name__: schema
    for schema in (
        click_marker,
        type_in_marker,
        scroll,
        press_key,
        open_browser,
        close_current_tab,
        review_job_detail,
        go_back,
        set_job_card_queue,
        switch_tab,
        finish_task,
    )
}

def model_action_tool_schema(name: str) -> dict[str, Any]:
    """내부 행동 계약을 모델이 혼동하지 않는 함수 스키마로 변환한다."""

    from langchain_core.utils.function_calling import convert_to_openai_tool

    schema = ACTION_TOOL_SCHEMAS[name]
    tool = deepcopy(convert_to_openai_tool(schema))
    if name == "scroll":
        properties = tool["function"]["parameters"]["properties"]
        distance = properties.pop("amount")
        distance["description"] = (
            "스크롤 이동 거리. 방향을 넣지 말고 small 또는 page 중 하나를 선택합니다."
        )
        properties["scroll_distance"] = distance
    return tool


def normalize_model_action_calls(
    raw_calls: List[dict[str, Any]],
) -> List[dict[str, Any]]:
    """모델 출력을 실행 가능한 내부 행동 묶음으로 정규화한다."""

    calls = deepcopy(raw_calls)
    default_risk_levels = {
        "click_marker": "safe_navigation",
        "type_in_marker": "safe_navigation",
        "press_key": "safe_navigation",
        "open_browser": "safe_navigation",
        "close_current_tab": "safe_navigation",
        "switch_tab": "safe_navigation",
        "go_back": "safe_navigation",
        "scroll": "safe_read",
        "review_job_detail": "safe_read",
    }
    for call in calls:
        name = str(call.get("name") or "")
        args = call.get("args")
        if isinstance(args, dict) and name in default_risk_levels:
            args.setdefault("risk_level", default_risk_levels[name])
        if name != "scroll":
            continue
        if isinstance(args, dict) and "scroll_distance" in args:
            args["amount"] = args.pop("scroll_distance")
    if len(calls) == 1:
        input_call = calls[0]
        input_args = input_call.get("args")
        if (
            input_call.get("name") == "type_in_marker"
            and isinstance(input_args, dict)
            and input_args.get("slot_name") == "search_keyword"
        ):
            calls.append(
                {
                    "name": "press_key",
                    "args": {
                        "key": "enter",
                        "reason": "입력한 검색어를 제출합니다.",
                        "expected_after": "검색 결과 화면이 보입니다.",
                        "page_role": input_args.get("page_role"),
                        "risk_level": "safe_navigation",
                    },
                    "id": f"{input_call.get('id') or 'search_input'}_submit",
                }
            )
    return calls


__all__ = [
    "ACTION_TOOL_SCHEMAS",
    "model_action_tool_schema",
    "normalize_model_action_calls",
    "VisibleJobCard",
    "click_marker",
    "close_current_tab",
    "review_job_detail",
    "finish_task",
    "go_back",
    "open_browser",
    "press_key",
    "scroll",
    "set_job_card_queue",
    "switch_tab",
    "type_in_marker",
]
