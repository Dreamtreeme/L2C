"""백엔드, 그래프와 장기 실행 자원을 하나의 프로세스로 조립한다."""

from __future__ import annotations

import threading
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

from agent.application.chat_service import ChatService
from agent.application.collection_experience import record_collection_experience
from agent.application.collection_postprocessing import postprocess_collection_batch
from agent.application.collection_storage import store_postprocessed_collection
from agent.application.collection_worker_runner import run_worker_once
from agent.application.conversation_context_service import (
    load_conversation_context as load_default_conversation_context,
)
from agent.application.clarification_service import apply_clarification_answer
from agent.application.evidence_service import (
    inspect_job_evidence,
    load_stored_jobs,
)
from agent.application.occupation_clarification_service import (
    OccupationClarificationService,
)
from agent.application.recipe_promotion_worker import RecipePromotionWorker
from agent.application.search_taxonomy_maintenance import prepare_search_taxonomy
from agent.application.search_taxonomy_service import SearchTaxonomyService
from agent.application.tool_capabilities import build_collection_capabilities
from agent.application.worker_execution_service import WorkerExecutionService
from agent.config import get_settings
from agent.graph.investigation_answer_nodes import InvestigationAnswerNodes
from agent.graph.investigation_collection_nodes import (
    InvestigationCollectionNodes,
)
from agent.graph.investigation_context import InvestigationModels
from agent.graph.investigation_evidence_nodes import InvestigationEvidenceNodes
from agent.graph.investigation_request_nodes import InvestigationRequestNodes
from agent.graph.investigation_workflow import InvestigationWorkflow
from agent.graph.workflow import build_graph
from agent.runtime.investigation_checkpoint import InvestigationCheckpointRuntime
from agent.runtime.vision_worker_runtime import VisionWorkerRuntime
from agent.sites import validate_site_profiles
from shared.schema.collection_intent import CollectionIntent
from shared.schema.collection_run import (
    CollectionBatch,
    CollectionExperienceResult,
    PersistenceReport,
    PostprocessedCollection,
)
from shared.schema.investigation_schema import ToolCapability


def build_investigation_workflow(
    db_path: str | Path,
    *,
    checkpoint_runtime: InvestigationCheckpointRuntime,
    run_collection: Callable[[CollectionIntent], CollectionBatch],
    postprocess_collection: Callable[[CollectionBatch], PostprocessedCollection],
    store_collection: Callable[[PostprocessedCollection], PersistenceReport],
    record_experience: Callable[
        [CollectionBatch, PersistenceReport], CollectionExperienceResult
    ],
    taxonomy_service: SearchTaxonomyService,
    models: InvestigationModels | None = None,
    capabilities: list[ToolCapability] | None = None,
    now: Callable[[], datetime] | None = None,
    conversation_context_loader: Callable | None = None,
) -> InvestigationWorkflow:
    """애플리케이션 서비스와 조사 노드를 한 번 조립한다."""

    resolved_db_path = Path(db_path)
    resolved_models = models or InvestigationModels()
    occupation_clarification = OccupationClarificationService(
        taxonomy_model=resolved_models.taxonomy,
        taxonomy_service=taxonomy_service,
    )

    def current_time() -> datetime:
        return datetime.now().astimezone()

    now_provider = now or current_time
    resolved_capabilities = (
        list(capabilities)
        if capabilities is not None
        else build_collection_capabilities()
    )
    return InvestigationWorkflow(
        checkpointer=checkpoint_runtime.saver,
        request_nodes=InvestigationRequestNodes(
            models=resolved_models,
            occupation_clarification=occupation_clarification,
            apply_clarification=apply_clarification_answer,
            load_conversation_context=(
                conversation_context_loader or load_default_conversation_context
            ),
            now=now_provider,
        ),
        evidence_nodes=InvestigationEvidenceNodes(
            models=resolved_models,
            taxonomy_service=taxonomy_service,
            inspect_evidence=partial(
                inspect_job_evidence,
                resolved_db_path,
                taxonomy_service=taxonomy_service,
            ),
            capabilities=resolved_capabilities,
            now=now_provider,
        ),
        collection_nodes=InvestigationCollectionNodes(
            run_collection,
            postprocess_collection,
            store_collection,
            record_experience,
        ),
        answer_nodes=InvestigationAnswerNodes(
            models=resolved_models,
            load_documents=partial(
                load_stored_jobs,
                resolved_db_path,
            ),
        ),
    )


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
        self.taxonomy_service = prepare_search_taxonomy(self.db_path)
        self.checkpoint_runtime = checkpoint_runtime or InvestigationCheckpointRuntime(
            self.db_path
        )
        self.vision_runtime = vision_runtime or VisionWorkerRuntime(
            graph_factory=build_graph
        )
        self.worker_execution_service = WorkerExecutionService(
            self.vision_runtime,
            run_worker_once,
        )
        self.investigation_workflow = (
            investigation_workflow
            or build_investigation_workflow(
                self.db_path,
                checkpoint_runtime=self.checkpoint_runtime,
                run_collection=self.worker_execution_service.run,
                postprocess_collection=postprocess_collection_batch,
                store_collection=partial(
                    store_postprocessed_collection,
                    db_path=self.db_path,
                ),
                record_experience=record_collection_experience,
                taxonomy_service=self.taxonomy_service,
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
            if get_settings().recipe.auto_promote:
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
            self.checkpoint_runtime.close()
            self._closed = True
            self._started = False


__all__ = ["ApplicationRuntime", "build_investigation_workflow"]
