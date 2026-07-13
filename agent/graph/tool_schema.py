"""브라우저 작업자가 LLM에 노출하는 도구 입력 스키마."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# 클래스 이름은 LLM에 노출되는 도구 이름과 같아야 하므로 소문자를 유지한다.
class click_marker(BaseModel):
    """화면의 특정 ID 마커를 클릭합니다."""

    marker_id: int = Field(..., description="클릭할 마커의 ID")
    target_label: Optional[str] = Field(None, description="선택한 항목의 보이는 제목(target_label)")
    target_role: Optional[str] = Field(None, description="목표 기준 대상 역할(target_role)")
    target_component: Optional[str] = Field(None, description="화면 구성요소(target_component)")
    reason: Optional[str] = Field(None, description="이 대상을 선택한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="클릭 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")


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
    slot_name: Optional[str] = Field(None, description="실행마다 바뀌는 입력 슬롯 이름(slot_name)")
    target_role: Optional[str] = Field(None, description="목표 기준 대상 역할(target_role)")
    target_component: Optional[str] = Field(None, description="화면 구성요소(target_component)")
    reason: Optional[str] = Field(None, description="이 입력을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="입력 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")


class scroll(BaseModel):
    """화면을 스크롤합니다."""

    direction: str = Field("down", description="스크롤 방향 ('down' 또는 'up')")
    reason: Optional[str] = Field(None, description="스크롤을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="스크롤 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")


class press_key(BaseModel):
    """엔터, ESC 등 특수키를 누릅니다."""

    key: str = Field(..., description="누를 특수키 (예: 'enter', 'esc')")
    reason: Optional[str] = Field(None, description="키 입력을 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="키 입력 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")


class open_browser(BaseModel):
    """브라우저를 열고 지정한 URL에 접속합니다."""

    url: str = Field(..., description="접속할 URL (예: https://www.wanted.co.kr)")
    reason: Optional[str] = Field(None, description="이 URL을 여는 이유(reason)")
    expected_after: Optional[str] = Field(None, description="브라우저 이동 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")


class close_browser(BaseModel):
    """열려 있는 브라우저 창을 닫습니다."""

    reason: Optional[str] = Field(None, description="브라우저를 닫는 이유(reason)")
    expected_after: Optional[str] = Field(None, description="브라우저 종료 후 기대 상태(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")


class update_extracted_info(BaseModel):
    """현재 화면에서 식별한 채용 공고 정보를 수집 상태에 병합합니다."""

    data_json: str = Field(..., description="업데이트할 정보 키-값 딕셔너리의 JSON 문자열")
    page_role: Optional[str] = Field(None, description="현재 정보를 읽은 페이지 역할(page_role). 상세 공고면 job_detail.")
    detail_complete: Optional[bool] = Field(None, description="상세 공고 본문 정보가 충분히 수집되었는지 여부(detail_complete).")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")


class finish_detail_reading(BaseModel):
    """누적한 상세 페이지 OCR을 한 번 정제하여 수집 상태에 병합합니다."""

    reason: Optional[str] = Field(None, description="상세 페이지 읽기를 종료하는 이유(reason)")
    detail_complete: Optional[bool] = Field(True, description="상세 공고 본문 정보가 충분히 수집되었는지 여부(detail_complete).")
    expected_after: Optional[str] = Field(None, description="정제 후 정상이라면 다음에 기대되는 상태(expected_after)")
    page_role: Optional[str] = Field("job_detail", description="Current page role.")
    risk_level: Optional[str] = Field("safe_read", description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")


class go_back(BaseModel):
    """브라우저의 뒤로가기 기능을 실행합니다."""

    reason: Optional[str] = Field(None, description="뒤로가기를 수행한 이유(reason)")
    expected_after: Optional[str] = Field(None, description="뒤로가기 후 정상이라면 보여야 할 화면 변화(expected_after)")
    page_role: Optional[str] = Field(None, description="Current page role.")
    risk_level: Optional[str] = Field(None, description="safe_read, safe_navigation, or sensitive.")
    needs_user_confirmation: Optional[bool] = Field(None, description="True before sensitive steps.")


class update_plan_progress(BaseModel):
    """현재 계획 단계를 업데이트하거나 계획을 수정합니다."""

    current_step: int = Field(..., description="수행 중인 계획 단계 인덱스 (0-indexed)")
    plan: Optional[List[str]] = Field(None, description="수정된 계획 단계 목록 (필요한 경우)")


class set_result_card_queue(BaseModel):
    """현재 화면의 수집 대상 공고 카드를 런타임 작업 큐에 저장합니다."""

    cards: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "수집할 공고 카드 목록. 각 항목에는 marker_id와 title/target_label, company를 가능한 만큼 넣으십시오. "
            "현재 화면에 보이는 카드만 넣어야 합니다."
        ),
    )
    titles: Optional[List[str]] = Field(
        None,
        description=(
            "cards 객체를 만들기 어려울 때만 사용합니다. 현재 화면에 보이는 수집 대상 공고 제목 목록입니다. "
            "executor는 OCR 마커 텍스트와 정확히 대응되는 제목만 큐에 저장합니다."
        ),
    )
    companies: Optional[List[str]] = Field(
        None,
        description="titles와 같은 순서로 대응되는 회사명 목록입니다. 모르면 생략하십시오.",
    )
    reason: Optional[str] = Field(None, description="이 카드들을 큐에 넣은 이유")


class finish_task(BaseModel):
    """작업을 완료하고 최종 데이터를 반환합니다."""

    result: str = Field(..., description="최종 완료 요약 또는 결과 데이터")


__all__ = [
    "click_marker",
    "close_browser",
    "finish_detail_reading",
    "finish_task",
    "go_back",
    "open_browser",
    "press_key",
    "scroll",
    "set_result_card_queue",
    "type_in_marker",
    "update_extracted_info",
    "update_plan_progress",
]
