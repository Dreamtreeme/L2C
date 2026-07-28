"""작업자 행동 요청과 실행 결과의 도메인 계약."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from agent.graph.tool_schema import ACTION_TOOL_SCHEMAS


def _message_text(content: Any) -> str:
    """모델 응답의 텍스트 블록을 짧은 행동 설명으로 정규화한다."""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("content") or ""
            else:
                text = str(item)
            if str(text).strip():
                parts.append(str(text).strip())
        return "\n".join(parts)
    return "" if content is None else str(content).strip()


class ToolCallRequest(BaseModel):
    """action executor에 전달할 단일 도구 호출."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "id")
    @classmethod
    def _require_non_empty_identifier(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("도구 이름과 호출 ID는 비어 있을 수 없습니다.")
        return normalized


class ActionRequest(BaseModel):
    """현재 화면 캡처를 근거로 실행할 원자 도구 호출."""

    source: str
    summary: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list, max_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def _require_source(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("행동 요청 출처는 비어 있을 수 없습니다.")
        return normalized

    @model_validator(mode="after")
    def _validate_tool_contracts(self) -> "ActionRequest":
        for call in self.tool_calls:
            schema = ACTION_TOOL_SCHEMAS.get(call.name)
            if schema is None:
                raise ValueError(f"허용되지 않은 작업자 도구입니다: {call.name}")
            unknown = set(call.args) - set(schema.model_fields)
            if unknown:
                raise ValueError(
                    f"{call.name}에 정의되지 않은 인자가 있습니다: {sorted(unknown)}"
                )
            validated = schema.model_validate(call.args)
            normalized_args = validated.model_dump(exclude_none=True)
            for empty_collection_field in (
                "observed_fields",
                "unavailable_fields",
            ):
                if not normalized_args.get(empty_collection_field):
                    normalized_args.pop(empty_collection_field, None)
            call.args = normalized_args
        return self


class ActionResult(BaseModel):
    """하나의 행동 요청을 실행한 최종 결과."""

    source: str
    summary: str = ""
    status: Literal["success", "partial", "error"]
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    screen_changed: bool = False
    is_finished: bool = False


def build_action_request(
    source: str,
    summary: str,
    tool_calls: list[dict[str, Any] | ToolCallRequest],
    *,
    metadata: dict[str, Any] | None = None,
    allowed_tool_names: Sequence[str] | None = None,
) -> ActionRequest:
    """정책 또는 모델의 단일 도구 호출을 검증된 행동 요청으로 만든다."""

    calls: list[ToolCallRequest] = []
    for index, call in enumerate(tool_calls):
        if isinstance(call, ToolCallRequest):
            calls.append(call)
            continue
        payload = dict(call)
        if not str(payload.get("id") or "").strip():
            payload["id"] = f"{source}_{index}"
        calls.append(ToolCallRequest.model_validate(payload))
    if allowed_tool_names is not None:
        allowed = {str(name) for name in allowed_tool_names}
        rejected = sorted({call.name for call in calls if call.name not in allowed})
        if rejected:
            raise ValueError(f"현재 사이트에서 허용되지 않은 도구입니다: {rejected}")
    return ActionRequest(
        source=source,
        summary=summary,
        tool_calls=calls,
        metadata=dict(metadata or {}),
    )


def action_request_from_model_response(
    response: Any,
    *,
    allowed_tool_names: Sequence[str],
) -> ActionRequest:
    """LangChain 모델 응답을 추론 경계에서 한 번만 도메인 계약으로 바꾼다."""

    raw_calls = getattr(response, "tool_calls", None) or []
    return build_action_request(
        "llm",
        _message_text(getattr(response, "content", "")),
        list(raw_calls),
        allowed_tool_names=allowed_tool_names,
    )


__all__ = [
    "ActionRequest",
    "ActionResult",
    "ToolCallRequest",
    "action_request_from_model_response",
    "build_action_request",
]
