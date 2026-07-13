"""로컬 백엔드가 최근 요청 상태를 조회할 수 있게 보관한다."""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from agent.application.run_contracts import RunEvent, RunPhase, RunStatus


class RunRegistry:
    def __init__(self, limit: int = 100):
        self.limit = max(10, int(limit))
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def start(
        self,
        run_id: str,
        query: str,
        *,
        conversation_id: str = "",
        user_query: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        item = {
            "run_id": run_id,
            "query": query,
            "user_query": str(user_query if user_query is not None else query),
            "conversation_id": str(conversation_id or ""),
            "status": RunStatus.QUEUED.value,
            "phase": "received",
            "message": "요청 대기 중",
            "created_at": now,
            "updated_at": now,
            "result": {},
            "error": "",
            "cancel_requested": False,
        }
        with self._lock:
            self._items[run_id] = item
            self._items.move_to_end(run_id)
            while len(self._items) > self.limit:
                self._items.popitem(last=False)
        return dict(item)

    def apply_event(self, event: RunEvent) -> None:
        with self._lock:
            item = self._items.get(event.run_id)
            if item is None:
                item = self.start(event.run_id, "")
                self._items[event.run_id] = item
            item.update(
                {
                    "status": event.status.value,
                    "phase": event.phase.value,
                    "message": event.message,
                    "updated_at": event.timestamp.isoformat(),
                }
            )

    def complete(self, run_id: str, result: dict[str, Any]) -> None:
        raw_status = str(result.get("run_status") or RunStatus.COMPLETED.value)
        try:
            status = RunStatus(raw_status)
        except ValueError:
            status = RunStatus.COMPLETED
        self._finish(run_id, status, result=result)

    def fail(self, run_id: str, error: str) -> None:
        self._finish(run_id, RunStatus.FAILED, error=error)

    def request_cancel(self, run_id: str) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            item = self._items.get(run_id)
            if item is None:
                return None
            if item.get("status") in {
                RunStatus.COMPLETED.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
            }:
                return dict(item)
            item.update(
                {
                    "cancel_requested": True,
                    "message": "취소 요청을 처리하고 있습니다.",
                    "updated_at": now,
                }
            )
            return dict(item)

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._lock:
            return bool((self._items.get(run_id) or {}).get("cancel_requested"))

    def conversation_history(
        self,
        conversation_id: str,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        if not conversation_id:
            return []
        with self._lock:
            items = [
                dict(item)
                for item in self._items.values()
                if item.get("conversation_id") == conversation_id
                and item.get("status") in {
                    RunStatus.COMPLETED.value,
                    RunStatus.WAITING_INPUT.value,
                }
            ]
        return items[-max(1, int(limit)) :]

    def _finish(
        self,
        run_id: str,
        status: RunStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            item = self._items.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "query": "",
                    "created_at": now,
                },
            )
            if status == RunStatus.COMPLETED:
                phase = RunPhase.COMPLETED.value
            elif status == RunStatus.FAILED:
                phase = RunPhase.FAILED.value
            elif status == RunStatus.CANCELLED:
                phase = RunPhase.CANCELLED.value
            else:
                phase = str(item.get("phase") or RunPhase.PLANNING.value)
            item.update(
                {
                    "status": status.value,
                    "phase": phase,
                    "updated_at": now,
                    "result": result or {},
                    "error": error,
                    "cancel_requested": status == RunStatus.CANCELLED,
                }
            )

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(run_id)
            return dict(item) if item is not None else None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._items.values())[-max(1, int(limit)) :]
            return [dict(item) for item in reversed(items)]


_RUN_REGISTRY = RunRegistry()


def get_run_registry() -> RunRegistry:
    return _RUN_REGISTRY


__all__ = ["RunRegistry", "get_run_registry"]
