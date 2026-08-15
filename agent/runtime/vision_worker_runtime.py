"""비전 작업자의 장기 실행 자원과 물리 입력 직렬화를 소유한다."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

from agent.runtime.worker_data_services import WorkerDataServices
from agent.runtime.worker_contracts import ScreenMarker
from shared.schema.experience_rule_schema import (
    ExperienceRuleStep,
    RuleApplication,
)

from agent.utils.logger import logger


class VisionWorkerRuntime:
    """Perception, 물리 도구, 작업자 그래프와 판단 모델을 지연 생성한다."""

    def __init__(
        self,
        *,
        perception_factory: Callable[[], Any] | None = None,
        action_tools_factory: Callable[[Any], Any] | None = None,
        graph_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._perception_factory = perception_factory
        self._action_tools_factory = action_tools_factory
        self._graph_factory = graph_factory
        self._resource_lock = threading.RLock()
        self._execution_lock = threading.RLock()
        self._perception: Any = None
        self._action_tools: Any = None
        self._graph: Any = None
        self._ui_models: dict[tuple[str, ...], Any] = {}
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("이미 종료된 비전 작업자 런타임입니다.")

    @property
    def is_initialized(self) -> bool:
        return any(
            (
                self._perception is not None,
                self._action_tools is not None,
                self._graph is not None,
                bool(self._ui_models),
            )
        )

    @property
    def ocr_worker_pid(self) -> int | None:
        perception = self._perception
        return perception.ocr_worker_pid if perception is not None else None

    def resource_snapshot(self) -> dict[str, Any]:
        """장기 실행 자원의 재사용 여부를 외부 관측용 값으로 반환한다."""

        with self._resource_lock:
            perception = self._perception
            return {
                "closed": self._closed,
                "initialized": self.is_initialized,
                "ocr_worker_pid": self.ocr_worker_pid,
                "browser_window_bound": bool(
                    perception is not None and perception.browser_window_id
                ),
                "ui_model_variant_count": len(self._ui_models),
                "graph_initialized": self._graph is not None,
            }

    def get_perception(self) -> Any:
        with self._resource_lock:
            self._require_open()
            if self._perception is None:
                if self._perception_factory is None:
                    from agent.tools.perception import PerceptionEngine

                    self._perception = PerceptionEngine()
                else:
                    self._perception = self._perception_factory()
            return self._perception

    def get_action_tools(self) -> Any:
        with self._resource_lock:
            self._require_open()
            if self._action_tools is None:
                perception = self.get_perception()
                if self._action_tools_factory is None:
                    from agent.tools.actions import ActionTools

                    self._action_tools = ActionTools(perception)
                else:
                    self._action_tools = self._action_tools_factory(perception)
            return self._action_tools

    def get_ui_model_with_tools(
        self,
        tool_names: tuple[str, ...],
        tool_schemas: Mapping[str, Any],
        *,
        tier: str = "primary",
    ) -> Any:
        with self._resource_lock:
            self._require_open()
            cache_key = (tier, *tool_names)
            if cache_key not in self._ui_models:
                from agent.llm.clients import get_google_chat_model
                from agent.llm.policy import (
                    lightweight_model_name,
                    worker_reasoning_model_name,
                    worker_reasoning_thinking_level,
                )
                from agent.runtime.tool_schema import model_action_tool_schema

                if tier not in {"lightweight", "primary"}:
                    raise ValueError(f"지원하지 않는 작업자 모델 단계입니다: {tier}")
                lightweight = tier == "lightweight"
                model = get_google_chat_model(
                    (
                        lightweight_model_name()
                        if lightweight
                        else worker_reasoning_model_name()
                    ),
                    temperature=0.1,
                    thinking_level=(
                        "minimal"
                        if lightweight
                        else worker_reasoning_thinking_level()
                    ),
                    execution_role=(
                        "lightweight" if lightweight else "worker_reasoning"
                    ),
                )
                self._ui_models[cache_key] = model.bind_tools(
                    [model_action_tool_schema(name) for name in tool_names]
                )
            return self._ui_models[cache_key]

    def prepare_reasoning_models(self, tool_schemas: Mapping[str, Any]) -> None:
        from agent.runtime.tool_schema import NAVIGATION_ACTION_TOOL_NAMES

        self.get_ui_model_with_tools(
            NAVIGATION_ACTION_TOOL_NAMES,
            tool_schemas,
            tier="lightweight",
        )
        from agent.runtime.job_card_selector import prepare_job_card_selector_model

        prepare_job_card_selector_model()

    def ensure_ocr_worker_ready(self) -> None:
        self.get_perception().ensure_ocr_worker_ready()

    def get_graph(self) -> Any:
        with self._resource_lock:
            self._require_open()
            if self._graph is None:
                if self._graph_factory is None:
                    raise RuntimeError("작업자 그래프 팩토리가 설정되지 않았습니다.")
                self._graph = self._graph_factory()
            return self._graph

    @contextmanager
    def execution_session(self) -> Iterator[None]:
        """작업자 실행을 직렬화하고 종료 시 브라우저만 정리한다."""

        with self._execution_lock:
            self._require_open()
            try:
                yield
            finally:
                try:
                    self._close_browser()
                except Exception as exc:
                    logger.warning("Browser cleanup failed", error=str(exc))

    def _close_browser(self) -> None:
        """브라우저만 닫고 OCR·모델·그래프 자원은 유지한다."""

        action_tools = self._action_tools
        if action_tools is None:
            logger.info(
                "Browser cleanup skipped", reason="action_tools_not_initialized"
            )
            return
        result = action_tools.close_browser()
        if result.get("status") != "success":
            logger.warning(
                "Browser cleanup failed",
                error=result.get("error") or result.get("result"),
            )
            return
        payload = result.get("result")
        if isinstance(payload, dict) and payload.get("closed") is False:
            logger.warning(
                "Browser cleanup incomplete",
                reason=payload.get("reason") or "browser_not_closed",
            )
            return
        logger.info("Browser cleanup completed", result=result)

    def close(self) -> None:
        """진행 중인 물리 입력이 끝난 뒤 브라우저, OCR, 캡처 자원을 닫는다."""

        with self._execution_lock, self._resource_lock:
            if self._closed:
                return
            action_tools = self._action_tools
            perception = self._perception
            if (
                action_tools is not None
                and perception is not None
                and perception.browser_window_id
            ):
                try:
                    self._close_browser()
                except Exception as exc:
                    logger.warning("Browser shutdown failed", error=str(exc))
            if perception is not None:
                perception.close()
            self._ui_models.clear()
            self._graph = None
            self._action_tools = None
            self._perception = None
            self._closed = True


@dataclass(frozen=True, slots=True)
class WorkerDependencies:
    """그래프 실행 동안 노드에 주입하는 불변 작업자 의존성."""

    vision: VisionWorkerRuntime
    data: WorkerDataServices
    resolve_experience_rule: Callable[
        [ExperienceRuleStep, list[ScreenMarker], str],
        RuleApplication,
    ]


__all__ = [
    "VisionWorkerRuntime",
    "WorkerDependencies",
]
