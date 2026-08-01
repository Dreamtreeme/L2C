"""로컬 백엔드 프로세스가 공유하는 애플리케이션 자원을 소유한다."""

from __future__ import annotations

import threading
from pathlib import Path

from agent.application.chat_service import ChatService
from agent.application.collection_service import create_collection_service
from agent.application.recipe_promotion_service import auto_promotion_enabled
from agent.application.recipe_promotion_worker import RecipePromotionWorker
from agent.graph.investigation_workflow import InvestigationWorkflow
from agent.graph.workflow import build_graph
from agent.runtime.investigation_checkpoint import InvestigationCheckpointRuntime
from agent.runtime.vision_worker_runtime import VisionWorkerRuntime
from agent.sites import validate_site_profiles


class ApplicationRuntime:
    """체크포인터, 그래프, 비전 작업자와 후처리 스레드의 단일 소유자."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        checkpoint_runtime: InvestigationCheckpointRuntime | None = None,
        vision_runtime: VisionWorkerRuntime | None = None,
        investigation_workflow: InvestigationWorkflow | None = None,
        chat_service: ChatService | None = None,
        promotion_worker: RecipePromotionWorker | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.checkpoint_runtime = checkpoint_runtime or InvestigationCheckpointRuntime(
            self.db_path
        )
        self.vision_runtime = vision_runtime or VisionWorkerRuntime(
            graph_factory=build_graph
        )
        self.collection_service = create_collection_service(self.vision_runtime)
        self.investigation_workflow = (
            investigation_workflow
            or InvestigationWorkflow(
                db_path=self.db_path,
                checkpoint_runtime=self.checkpoint_runtime,
                collect_jobs=self.collection_service.collect,
            )
        )
        self.chat_service = chat_service or ChatService(
            investigation_workflow=self.investigation_workflow
        )
        self.promotion_worker = promotion_worker or RecipePromotionWorker(self.db_path)
        self._lifecycle_lock = threading.RLock()
        self._started = False
        self._closed = False

    def start(self) -> None:
        """백엔드 수명 동안 필요한 저우선순위 작업자만 시작한다."""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("이미 종료된 애플리케이션 런타임입니다.")
            if self._started:
                return
            validate_site_profiles()
            if auto_promotion_enabled():
                self.promotion_worker.start()
            self._started = True

    def close(self, *, promotion_timeout_sec: float = 1.0) -> None:
        """활성 요청이 끝난 뒤 소유 자원을 역순으로 정리한다."""

        with self._lifecycle_lock:
            if self._closed:
                return
            if self.promotion_worker.is_running:
                self.promotion_worker.stop(timeout_sec=promotion_timeout_sec)
            self.vision_runtime.close()
            self.chat_service.close()
            self.checkpoint_runtime.close()
            self._closed = True
            self._started = False

__all__ = ["ApplicationRuntime"]
