"""LLM과 결정론적 정책이 공유하는 다음 행동 요청 형식."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    """action executor에 전달할 단일 도구 호출."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    id: str


class ActionRequest(BaseModel):
    """LLM 여부와 무관하게 실행할 도구 호출 묶음."""

    source: str
    summary: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)

    def to_ai_message(self) -> AIMessage:
        """기존 action_node 계약을 유지하기 위한 임시 어댑터."""

        content = f"[{self.source}]"
        if self.summary:
            content += f" {self.summary}"
        return AIMessage(
            content=content,
            tool_calls=[call.model_dump() for call in self.tool_calls],
        )


def build_action_message(
    source: str,
    summary: str,
    tool_calls: list[dict[str, Any]],
) -> AIMessage:
    """결정론적 정책 결과를 기존 LangChain 메시지 계약으로 변환한다."""

    return ActionRequest(
        source=source,
        summary=summary,
        tool_calls=[ToolCallRequest(**call) for call in tool_calls],
    ).to_ai_message()


__all__ = ["ActionRequest", "ToolCallRequest", "build_action_message"]
