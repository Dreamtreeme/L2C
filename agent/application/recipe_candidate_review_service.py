"""실행 그래프를 검토하고 남은 원본 행동을 경험 규칙으로 승격한다."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.application.recipe_execution_graph_service import (
    GraphBuilderFn,
    build_candidate_execution_graph,
    build_candidate_graph_payload,
    graph_node_transitions,
)
from agent.config import get_settings
from agent.recipe.store import ExperienceRuleStore
from agent.runtime.worker_actions import is_supported_recipe_action_group
from agent.utils.text import normalize_text
from shared.schema.feedback_schema import (
    CandidateExecutionGraph,
    ExecutionGraphNode,
    RecipeCandidate,
    RecipeCandidateReview,
)


CriticFn = Callable[[dict[str, Any]], dict[str, Any] | RecipeCandidateReview]
ReviewProcessMode = Literal["review", "promote"]


class PrunedExecutionNode(BaseModel):
    """그래프 비평가가 재사용 경로에서 제거한 의미 노드."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    source_event_seqs: list[int] = Field(default_factory=list)
    source_transition_seqs: list[int] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    reason: str = ""


class RulePromotionResult(BaseModel):
    """그래프 검토 결과를 경험 규칙으로 저장한 결과."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    promoted: bool = False
    saved_count: int = 0
    rule_step_count: int = 0
    rule_action_count: int = 0
    pruned_nodes: list[PrunedExecutionNode] = Field(default_factory=list)


def build_candidate_review_payload(
    candidate: RecipeCandidate,
    execution_graph: CandidateExecutionGraph,
) -> dict[str, Any]:
    """완성된 실행 그래프와 원본 전이를 가지치기 모델에 전달한다."""

    source = build_candidate_graph_payload(candidate)
    replay_support = {
        event.transition.seq: is_supported_recipe_action_group(
            event.transition.actions
        )
        for event in candidate.action_events
        if event.transition is not None
    }
    transition_events = {
        int(event["transition"]["seq"]): {
            "before": event.get("before", {}),
            "actions": event.get("actions", []),
            "replay_supported": replay_support.get(
                int(event["transition"]["seq"]),
                False,
            ),
            **dict(event["transition"]),
        }
        for event in source["flat_log"]
        if event.get("transition")
    }
    actionable_nodes = []
    for node in execution_graph.nodes:
        transitions = graph_node_transitions(candidate, node)
        if not transitions:
            continue
        actionable_nodes.append(
            {
                **node.model_dump(mode="json"),
                "transitions": [
                    transition_events[transition.seq] for transition in transitions
                ],
            }
        )
    return {
        "run_id": candidate.run_id,
        "request": source["request"],
        "execution_graph": execution_graph.model_dump(mode="json"),
        "actionable_nodes": actionable_nodes,
        "required_node_verdicts": [
            {
                "node_id": node["node_id"],
                "source_event_seqs": node["source_event_seqs"],
                "source_transition_seqs": [
                    transition["seq"] for transition in node["transitions"]
                ],
            }
            for node in actionable_nodes
        ],
        "worker_execution": source["final_result"],
    }


def _coerce_review(
    raw: dict[str, Any] | RecipeCandidateReview,
) -> RecipeCandidateReview:
    return (
        raw
        if isinstance(raw, RecipeCandidateReview)
        else RecipeCandidateReview.model_validate(raw)
    )


def _fallback_review(reason: str) -> RecipeCandidateReview:
    return RecipeCandidateReview(
        decision="reject",
        reasons=[reason],
        feedback_to_worker="비평가가 경험 경로를 확정하지 못했습니다.",
    )


def _llm_review_graph(payload: dict[str, Any]) -> RecipeCandidateReview:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.llm.clients import get_structured_google_model
    from agent.llm.policy import commander_model_name
    from agent.observability.run_context import invoke_with_metrics
    from agent.prompts.trust_boundary import external_content_contract_en

    model_name = get_settings().models.recipe_critic_model or commander_model_name()
    llm = get_structured_google_model(
        model_name,
        RecipeCandidateReview,
        temperature=0.2,
        execution_role="critic",
    )
    system_text = (
        external_content_contract_en()
        + "\nYou are the Reflex execution-graph pruner. A previous observer has "
        "already represented every source event in purpose nodes and causal edges. "
        "Now decide which semantic nodes form a reusable path for the same request "
        "type. Return exactly one verdict for every required actionable node. Keep or "
        "drop each node as a whole; never split a node into individual transitions. "
        "Use the entire graph before judging an individual node. Keep setup "
        "actions that update hidden UI state when a later commit depends on them, "
        "even when their pixels did not change. Drop mistaken branch actions, their "
        "inspection, and recovery actions when the normal path does not require them. "
        "Drop transient pop-up handling and choices whose target depends on the current "
        "search result. Keep parameterized request input and stable navigation/filter "
        "bundles. If one node mixes a reusable route with a mistaken branch, reject the "
        "candidate because this stage cannot repair the graph boundary. Do not create, "
        "rewrite, reorder, merge, or alter any action. A failed, "
        "sensitive, insufficiently evidenced, or replay_supported=false transition must "
        "not be kept. The kept verdicts must describe one contiguous, causally usable "
        "route. A reusable prefix or subpath is sufficient: later variable choices may "
        "be dropped because runtime reasoning resumes after the retained route. Set "
        "decision=accept when at least one such route remains. Reject only when no "
        "actionable node can be retained."
    )
    return _coerce_review(
        invoke_with_metrics(
            llm,
            [
                SystemMessage(content=system_text),
                HumanMessage(
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            ],
            "recipe_graph_pruner",
        )
    )


def _verdict_contract_errors(
    candidate: RecipeCandidate,
    execution_graph: CandidateExecutionGraph,
    review: RecipeCandidateReview,
) -> list[str]:
    required = {
        node.node_id
        for node in execution_graph.nodes
        if graph_node_transitions(candidate, node)
    }
    counts: dict[str, int] = {}
    for verdict in review.node_verdicts:
        counts[verdict.node_id] = counts.get(verdict.node_id, 0) + 1
    errors = [
        f"node_id={node_id} verdict_count={counts.get(node_id, 0)}"
        for node_id in sorted(required)
        if counts.get(node_id, 0) != 1
    ]
    errors.extend(
        f"unexpected node_id={node_id}" for node_id in sorted(set(counts) - required)
    )
    if review.decision == "accept" and not any(
        verdict.keep for verdict in review.node_verdicts
    ):
        errors.append("accepted review kept no actionable node")
    return errors


def review_candidate(
    candidate: RecipeCandidate,
    execution_graph: CandidateExecutionGraph,
    critic: CriticFn | None = None,
    *,
    raise_on_error: bool = False,
) -> RecipeCandidateReview:
    if not any(
        graph_node_transitions(candidate, node) for node in execution_graph.nodes
    ):
        return _fallback_review("candidate_transition_missing")
    try:
        payload = build_candidate_review_payload(candidate, execution_graph)
        review = _coerce_review(
            critic(payload) if critic is not None else _llm_review_graph(payload)
        )
        errors = _verdict_contract_errors(candidate, execution_graph, review)
        if errors:
            raise ValueError(
                "critic_node_verdict_contract_failed: " + "; ".join(errors[:8])
            )
        return review
    except Exception as exc:
        if raise_on_error:
            raise
        return _fallback_review(f"critic_review_failed: {str(exc)[:200]}")


def _selected_graph_nodes(
    candidate: RecipeCandidate,
    execution_graph: CandidateExecutionGraph,
    review: RecipeCandidateReview,
) -> tuple[list[ExecutionGraphNode], list[PrunedExecutionNode]]:
    """비평가 판정을 그래프 노드 목록에 그대로 적용한다."""

    verdicts = {verdict.node_id: verdict for verdict in review.node_verdicts}
    retained: list[ExecutionGraphNode] = []
    pruned: list[PrunedExecutionNode] = []
    for node in execution_graph.nodes:
        transitions = graph_node_transitions(candidate, node)
        if not transitions:
            continue
        verdict = verdicts[node.node_id]
        if verdict.keep:
            retained.append(node)
        else:
            pruned.append(
                PrunedExecutionNode(
                    node_id=node.node_id,
                    source_event_seqs=list(node.source_event_seqs),
                    source_transition_seqs=[
                        transition.seq for transition in transitions
                    ],
                    actions=[
                        action.action
                        for transition in transitions
                        for action in transition.actions
                    ],
                    reason=normalize_text(verdict.reason) or "critic_pruned",
                )
            )
    return retained, pruned


def review_and_apply_candidate(
    run_id: str,
    db_path=None,
    graph_builder: GraphBuilderFn | None = None,
    critic: CriticFn | None = None,
    mode: str = "review",
    raise_on_critic_error: bool = False,
) -> dict[str, Any]:
    from agent.recipe.candidate_store import RecipeCandidateStore

    candidate_store = RecipeCandidateStore(db_path)
    candidate = candidate_store.get_candidate(run_id)
    if candidate is None:
        return _fallback_review(f"candidate_not_found: {run_id}").model_dump(
            mode="json"
        )

    try:
        execution_graph = build_candidate_execution_graph(
            candidate,
            builder=graph_builder,
        )
        review = review_candidate(
            candidate,
            execution_graph,
            critic=critic,
            raise_on_error=raise_on_critic_error,
        )
    except Exception as exc:
        if raise_on_critic_error:
            raise
        execution_graph = None
        review = _fallback_review(f"execution_graph_failed: {str(exc)[:200]}")

    allow_promotion = _process_mode(mode) == "promote"
    promotion = RulePromotionResult(enabled=allow_promotion)
    if allow_promotion and review.decision == "accept":
        if execution_graph is None:
            raise RuntimeError("accepted review is missing its execution graph")
        retained, pruned = _selected_graph_nodes(
            candidate,
            execution_graph,
            review,
        )
        try:
            if not retained:
                raise ValueError("critic kept no actionable graph node")
            from agent.application.experience_rule_compiler import (
                compile_experience_rule,
            )

            rule = compile_experience_rule(
                candidate,
                execution_graph,
                [node.node_id for node in retained],
            )
            saved_count = ExperienceRuleStore(db_path).save_rule(
                rule,
                source_run_id=candidate.run_id,
            )
            promotion = RulePromotionResult(
                enabled=True,
                promoted=saved_count > 0,
                saved_count=saved_count,
                rule_step_count=len(rule.steps),
                rule_action_count=sum(len(step.actions) for step in rule.steps),
                pruned_nodes=pruned,
            )
        except Exception as exc:
            promotion = RulePromotionResult(
                enabled=True,
                pruned_nodes=pruned,
            )
            review = review.model_copy(
                update={
                    "decision": "reject",
                    "reasons": [
                        *review.reasons,
                        f"experience_rule_compilation_failed: {str(exc)[:200]}",
                    ],
                }
            )

    validation = {
        "execution_graph": (
            execution_graph.model_dump(mode="json") if execution_graph else {}
        ),
        "review": review.model_dump(mode="json"),
        "promotion": promotion.model_dump(mode="json"),
    }
    candidate_store.update_status(
        run_id,
        "accepted" if review.decision == "accept" else "rejected",
        validation=validation,
    )
    result = review.model_dump(mode="json")
    result["run_id"] = run_id
    result["execution_graph"] = validation["execution_graph"]
    result["promotion"] = promotion.model_dump(mode="json")
    return result


def _process_mode(mode: str | None) -> ReviewProcessMode:
    return "promote" if (mode or "review").strip().lower() == "promote" else "review"


__all__ = [
    "CriticFn",
    "GraphBuilderFn",
    "PrunedExecutionNode",
    "RulePromotionResult",
    "build_candidate_review_payload",
    "review_and_apply_candidate",
    "review_candidate",
]
