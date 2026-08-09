"""실행 기록을 조사 그래프가 사용하는 대화 문맥으로 변환한다."""

from __future__ import annotations

from agent.observability.run_registry import RunRegistry, get_run_registry
from shared.schema.investigation_schema import ConversationTurn


def load_conversation_context(
    conversation_id: str,
    context_run_id: str = "",
    *,
    registry: RunRegistry | None = None,
    limit: int = 4,
) -> list[ConversationTurn]:
    """최근 완료 대화와 명시적으로 재개한 실행을 구조화해 반환한다."""

    source = registry or get_run_registry()
    items = source.conversation_history(conversation_id, limit=limit)
    known_ids = {str(item.get("run_id") or "") for item in items}
    if context_run_id and context_run_id not in known_ids:
        resumed = source.get(context_run_id)
        if resumed:
            items.append(resumed)
    return [
        ConversationTurn(
            run_id=str(item.get("run_id") or ""),
            user_query=str(item.get("user_query") or item.get("query") or ""),
            assistant_answer=str((item.get("result") or {}).get("text") or ""),
            run_status=str(item.get("status") or ""),
        )
        for item in items[-max(1, int(limit)) :]
    ]


__all__ = ["load_conversation_context"]
