"""브라우저 작업자가 LLM과 실행기에 공유하는 도구 입력 스키마."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from shared.schema.jd_schema import JOB_FIELDS, JobField
from shared.schema.skill_schema import RecipeInputName


class _DetailObservation(BaseModel):
    """상세 화면 판단에 이미 사용한 필드 근거를 행동과 함께 보존한다."""

    observed_fields: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "현재 상세 화면에서 실제로 확인한 공고 필드와 짧은 화면 근거. "
            "상세 화면에서만 채우며 추측한 값은 넣지 않는다. 허용 키: "
            + ", ".join(JOB_FIELDS)
        ),
    )

    @field_validator("observed_fields", mode="before")
    @classmethod
    def normalize_observed_field_values(
        cls,
        values: Any,
    ) -> Any:
        """복수 근거 목록을 도구 계약의 짧은 근거 문자열로 합친다."""

        if values is None:
            return {}
        if not isinstance(values, dict):
            return values
        normalized: Dict[str, str] = {}
        for field, value in values.items():
            items = (
                value
                if isinstance(value, (list, tuple, set))
                else [value]
            )
            evidence = "; ".join(
                str(item).strip()
                for item in items
                if str(item).strip()
            )
            if evidence:
                normalized[str(field)] = evidence
        return normalized

    @field_validator("observed_fields")
    @classmethod
    def validate_observed_fields(
        cls,
        values: Dict[str, str],
    ) -> Dict[str, str]:
        unknown = sorted(set(values) - set(JOB_FIELDS))
        if unknown:
            raise ValueError(
                "지원하지 않는 공고 필드입니다: "
                + ", ".join(unknown)
            )
        return values


class click_marker(_DetailObservation):
    """화면의 특정 ID 마커를 클릭합니다."""

    marker_id: int = Field(..., description="클릭할 마커의 ID")
    target_label: Optional[str] = Field(None, description="선택한 항목의 보이는 제목(target_label)")
    target_role: Optional[str] = Field(None, description="목표 기준 대상 역할(target_role)")
    target_component: Optional[str] = Field(None, description="화면 구성요소(target_component)")
    reason: Optional[str] = Field(None, description="이 대상을 선택한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="클릭 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")


class focus_marker(BaseModel):
    """마커를 마우스로 클릭해 키보드·스크롤 포커스를 옮깁니다."""

    marker_id: int = Field(..., description="포커스를 옮길 마커의 ID")
    target_label: Optional[str] = Field(None, description="포커스 대상의 보이는 라벨(target_label)")
    target_role: Optional[str] = Field(None, description="포커스 대상 역할(target_role)")
    target_component: Optional[str] = Field(None, description="포커스를 받을 화면 구성요소(target_component)")
    reason: Optional[str] = Field(None, description="이 구성요소에 포커스를 옮기는 이유(reason)")
    expected_after: Optional[str] = Field(
        None,
        description="포커스 후 이어서 수행할 키보드 또는 스크롤 행동(expected_after)",
    )
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")


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
    target_label: Optional[str] = Field(None, description="입력 영역의 보이는 라벨(target_label)")
    slot_name: Optional[RecipeInputName] = Field(
        None,
        description="검색어를 재생할 때 사용하는 search_keyword 입력 이름",
    )
    target_role: Optional[str] = Field(None, description="목표 기준 대상 역할(target_role)")
    target_component: Optional[str] = Field(None, description="화면 구성요소(target_component)")
    reason: Optional[str] = Field(None, description="이 입력을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="입력 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")


class scroll(_DetailObservation):
    """화면을 스크롤합니다."""

    direction: Literal["down", "up", "left", "right"] = Field(
        "down",
        description="스크롤 방향 ('down', 'up', 'left', 'right')",
    )
    marker_id: Optional[int] = Field(
        None,
        description=(
            "내부 패널·목록처럼 특정 영역을 스크롤할 때 기준으로 삼을 마커 ID. "
            "생략하면 전체 페이지를 스크롤합니다."
        ),
    )
    amount: Literal["small", "page"] = Field(
        "page",
        description="스크롤 이동량 ('small' 또는 'page')",
    )
    target_label: Optional[str] = Field(None, description="스크롤할 영역의 보이는 라벨(target_label)")
    target_role: Optional[str] = Field(None, description="스크롤 대상 역할(target_role)")
    target_component: Optional[str] = Field(None, description="스크롤 대상 화면 구성요소(target_component)")
    reason: Optional[str] = Field(None, description="스크롤을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="스크롤 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(
        "safe_read",
        description="스크롤은 화면 내용을 더 읽는 읽기 전용 행동입니다.",
    )


class press_key(BaseModel):
    """엔터, ESC 등 특수키를 누릅니다."""

    key: str = Field(..., description="누를 특수키 (예: 'enter', 'esc')")
    reason: Optional[str] = Field(None, description="키 입력을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="키 입력 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")


class open_browser(BaseModel):
    """브라우저를 열고 지정한 URL에 접속합니다."""

    url: str = Field(..., description="접속할 URL (예: https://www.wanted.co.kr)")
    reason: Optional[str] = Field(None, description="이 URL을 여는 이유(reason)")
    expected_after: Optional[str] = Field(None, description="브라우저 이동 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")


class close_current_tab(BaseModel):
    """현재 활성 브라우저 탭 하나를 닫습니다."""

    reason: Optional[str] = Field(None, description="현재 탭을 닫는 이유(reason)")
    expected_after: Optional[str] = Field(None, description="탭을 닫은 뒤 기대 상태(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")


class switch_tab(BaseModel):
    """현재 브라우저 창에서 인접한 탭으로 전환합니다."""

    direction: Literal["next", "previous"] = Field(
        ...,
        description="전환할 탭 방향 ('next' 또는 'previous')",
    )
    reason: Optional[str] = Field(None, description="탭을 전환하는 이유(reason)")
    expected_after: Optional[str] = Field(None, description="탭 전환 뒤 기대 상태(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")


class finish_detail_reading(_DetailObservation):
    """누적한 상세 페이지 OCR을 한 번 정제하여 수집 상태에 병합합니다."""

    reason: Optional[str] = Field(None, description="상세 페이지 읽기를 종료하는 이유(reason)")
    unavailable_fields: List[JobField] = Field(
        default_factory=list,
        description=(
            "페이지 전체를 확인했지만 공고가 제공하지 않는 필수 필드. "
            "page_exhausted=true일 때만 완료 필드로 인정한다."
        ),
    )
    page_exhausted: bool = Field(
        False,
        description="더 펼칠 본문이나 아래쪽 공고 내용이 없음을 화면에서 확인했는지 여부",
    )
    expected_after: Optional[str] = Field(None, description="정제 후 정상이라면 다음에 기대되는 상태(expected_after)")
    page_role: Optional[str] = Field("job_detail", description="Current page role.")
    risk_level: Optional[str] = Field("safe_read", description="safe_read, safe_navigation, or sensitive.")


class go_back(BaseModel):
    """브라우저의 뒤로가기 기능을 실행합니다."""

    reason: Optional[str] = Field(None, description="뒤로가기를 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="뒤로가기 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")


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
    """현재 화면의 수집 대상 공고 카드를 런타임 작업 큐에 저장합니다."""

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
    count_evidence: Optional[str] = Field(None, description="전체 결과 개수를 판단한 화면 문구")
    reason: Optional[str] = Field(None, description="이 카드들을 큐에 넣은 이유")


class finish_task(BaseModel):
    """작업을 완료하고 최종 데이터를 반환합니다."""

    result: str = Field(..., description="최종 완료 요약 또는 결과 데이터")


ACTION_TOOL_SCHEMAS = {
    schema.__name__: schema
    for schema in (
        click_marker,
        focus_marker,
        type_in_marker,
        scroll,
        press_key,
        open_browser,
        close_current_tab,
        finish_detail_reading,
        go_back,
        set_job_card_queue,
        switch_tab,
        finish_task,
    )
}

DETAIL_ACTION_TOOL_NAMES = (
    "click_marker",
    "focus_marker",
    "scroll",
    "press_key",
    "finish_detail_reading",
    "go_back",
    "close_current_tab",
    "switch_tab",
    "finish_task",
)
NAVIGATION_ACTION_TOOL_NAMES = (
    "click_marker",
    "focus_marker",
    "type_in_marker",
    "scroll",
    "press_key",
    "go_back",
    "close_current_tab",
    "switch_tab",
    "finish_task",
)
UNKNOWN_ACTION_TOOL_NAMES = (
    *NAVIGATION_ACTION_TOOL_NAMES[:5],
    "open_browser",
    *NAVIGATION_ACTION_TOOL_NAMES[5:],
    "finish_detail_reading",
)


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
    """모델 전용 인자명을 내부 실행 계약의 인자명으로 되돌린다."""

    calls = deepcopy(raw_calls)
    for call in calls:
        if call.get("name") != "scroll":
            continue
        args = call.get("args")
        if isinstance(args, dict) and "scroll_distance" in args:
            args["amount"] = args.pop("scroll_distance")
    return calls


__all__ = [
    "ACTION_TOOL_SCHEMAS",
    "DETAIL_ACTION_TOOL_NAMES",
    "NAVIGATION_ACTION_TOOL_NAMES",
    "UNKNOWN_ACTION_TOOL_NAMES",
    "model_action_tool_schema",
    "normalize_model_action_calls",
    "VisibleJobCard",
    "click_marker",
    "close_current_tab",
    "focus_marker",
    "finish_detail_reading",
    "finish_task",
    "go_back",
    "open_browser",
    "press_key",
    "scroll",
    "set_job_card_queue",
    "switch_tab",
    "type_in_marker",
]
