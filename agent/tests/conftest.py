"""환경 변수 기반 설정 캐시가 테스트 사이에 새지 않게 한다."""

import pytest


@pytest.fixture(autouse=True)
def reset_typed_settings_cache():
    from agent.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def investigation_workflow_factory():
    """앱 조립 경계와 동일하게 조사 그래프와 체크포인터를 함께 소유한다."""

    from agent.application.search_taxonomy_maintenance import (
        prepare_search_taxonomy,
    )
    from agent.bootstrap import build_investigation_workflow
    from agent.runtime.investigation_checkpoint import (
        InvestigationCheckpointRuntime,
    )

    owners = []

    class OwnedWorkflow:
        def __init__(self, workflow, checkpoint_runtime):
            self.workflow = workflow
            self.checkpoint_runtime = checkpoint_runtime
            self.closed = False

        def run(self, *args, **kwargs):
            return self.workflow.run(*args, **kwargs)

        def close(self):
            if not self.closed:
                self.checkpoint_runtime.close()
                self.closed = True

    def create(
        *,
        db_path,
        run_collection,
        persist_collection,
        models=None,
        capabilities=None,
        taxonomy_service=None,
        now=None,
        conversation_context_loader=None,
        run_lookup=None,
    ):
        taxonomy = taxonomy_service or prepare_search_taxonomy(db_path)
        checkpoint = InvestigationCheckpointRuntime(db_path)
        owner = OwnedWorkflow(
            build_investigation_workflow(
                db_path,
                checkpoint_runtime=checkpoint,
                run_collection=run_collection,
                persist_collection=persist_collection,
                taxonomy_service=taxonomy,
                models=models,
                capabilities=capabilities,
                now=now,
                conversation_context_loader=conversation_context_loader,
                run_lookup=run_lookup,
            ),
            checkpoint,
        )
        owners.append(owner)
        return owner

    yield create

    for owner in owners:
        owner.close()
