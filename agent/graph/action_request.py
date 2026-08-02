"""작업자 행동 요청과 실행 결과의 도메인 계약."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

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
    """현재 화면 캡처를 근거로 실행할 행동 또는 검증된 행동 묶음."""

    source: str
    summary: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
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
        if len(self.tool_calls) > 1:
            from agent.recipe.replay_actions import (
                is_supported_recipe_action_group,
            )

            action_group = [
                {
                    "action": call.name,
                    "param": call.args,
                }
                for call in self.tool_calls
            ]
            if (
                self.source != "reflex"
                or self.metadata.get("execution_unit")
                != "recipe_transition"
                or not is_supported_recipe_action_group(action_group)
            ):
                raise ValueError(
                    "여러 행동은 검증된 경험 기반 탐색 전이에서만 허용됩니다."
                )
        return self


class ActionEvent(TypedDict, total=False):
    """행동 선택부터 화면 전환 검증까지 한 생명주기로 보관하는 기록."""

    seq: int
    result: dict[str, Any]
    recipe_step: dict[str, Any]
    feedback_episode: dict[str, Any]
    transition: dict[str, Any]


def build_action_event(
    seq: int,
    result: dict[str, Any],
    *,
    recipe_step: dict[str, Any] | None = None,
    feedback_episode: dict[str, Any] | None = None,
) -> ActionEvent:
    event: ActionEvent = {
        "seq": int(seq),
        "result": dict(result),
    }
    if recipe_step:
        event["recipe_step"] = dict(recipe_step)
    if feedback_episode:
        event["feedback_episode"] = dict(feedback_episode)
    return event


def action_event_results(events: Sequence[ActionEvent | dict]) -> list[dict[str, Any]]:
    return [
        dict(event.get("result") or {})
        for event in events or []
        if isinstance(event, Mapping) and isinstance(event.get("result"), Mapping)
    ]


def action_event_recipe_steps(events: Sequence[ActionEvent | dict]) -> list[dict[str, Any]]:
    return [
        dict(event.get("recipe_step") or {})
        for event in events or []
        if isinstance(event, Mapping) and isinstance(event.get("recipe_step"), Mapping)
    ]


def action_event_feedback(events: Sequence[ActionEvent | dict]) -> list[dict[str, Any]]:
    return [
        dict(event.get("feedback_episode") or {})
        for event in events or []
        if isinstance(event, Mapping)
        and isinstance(event.get("feedback_episode"), Mapping)
    ]


def action_event_transitions(events: Sequence[ActionEvent | dict]) -> list[dict[str, Any]]:
    return [
        dict(event.get("transition") or {})
        for event in events or []
        if isinstance(event, Mapping) and isinstance(event.get("transition"), Mapping)
    ]


def attach_action_transition(
    events: Sequence[ActionEvent | dict],
    transition: dict[str, Any],
) -> list[ActionEvent]:
    """행동 순번이 같은 이벤트에 화면 전환 검증 결과를 연결한다."""

    try:
        target_seq = int(transition.get("action_seq"))
    except (TypeError, ValueError):
        return [dict(event) for event in events or [] if isinstance(event, Mapping)]

    updated: list[ActionEvent] = []
    for raw_event in events or []:
        if not isinstance(raw_event, Mapping):
            continue
        event: ActionEvent = dict(raw_event)
        try:
            event_seq = int(event.get("seq"))
        except (TypeError, ValueError):
            event_seq = -1
        if event_seq == target_seq:
            event["transition"] = dict(transition)
        updated.append(event)
    return updated


def build_action_request(
    source: str,
    summary: str,
    tool_calls: list[dict[str, Any] | ToolCallRequest],
    *,
    metadata: dict[str, Any] | None = None,
    allowed_tool_names: Sequence[str] | None = None,
) -> ActionRequest:
    """정책 또는 모델 호출을 검증된 행동 요청으로 만든다."""

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
    "ActionEvent",
    "ActionRequest",
    "ToolCallRequest",
    "action_event_feedback",
    "action_event_recipe_steps",
    "action_event_results",
    "action_event_transitions",
    "action_request_from_model_response",
    "attach_action_transition",
    "build_action_event",
    "build_action_request",
]
