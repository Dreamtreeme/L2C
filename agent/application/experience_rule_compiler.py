"""비평가가 남긴 원본 전이를 재생 가능한 경험 규칙으로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass

from agent.application.recipe_execution_graph_service import graph_node_transitions
from agent.recipe.task_category import normalize_task_category
from agent.runtime.worker_actions import is_supported_recipe_action_group
from agent.utils.text import normalize_text
from shared.schema.execution_record_schema import ObservedAction, ObservedTransition
from shared.schema.experience_rule_schema import (
    ExpectedEffect,
    ExperienceRule,
    ExperienceRuleNode,
    ExperienceRuleStep,
    RuleAction,
    RuleScreen,
    RuleTarget,
)
from shared.schema.feedback_schema import (
    CandidateExecutionGraph,
    ExecutionGraphNode,
    RecipeCandidate,
)
from shared.schema.skill_schema import RECIPE_INPUT_NAMES, RecipeSkillMetadata


@dataclass(frozen=True)
class RuleSourceStep:
    """하나의 의미 노드에 속한 원본 물리 전이."""

    node: ExecutionGraphNode
    transition: ObservedTransition


def _rule_source_steps(
    candidate: RecipeCandidate,
    execution_graph: CandidateExecutionGraph,
    selected_node_ids: list[str],
) -> list[RuleSourceStep]:
    """선택된 그래프 노드를 원본 화면 전이 순서로 펼친다."""

    if not selected_node_ids or len(selected_node_ids) != len(set(selected_node_ids)):
        raise ValueError("experience rule requires unique selected graph nodes")
    selected = set(selected_node_ids)
    known = {node.node_id for node in execution_graph.nodes}
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown selected graph nodes: {sorted(unknown)}")

    ordered_nodes = [node for node in execution_graph.nodes if node.node_id in selected]
    if [node.node_id for node in ordered_nodes] != selected_node_ids:
        raise ValueError("selected graph nodes must preserve graph order")

    source_steps = [
        RuleSourceStep(node=node, transition=transition)
        for node in ordered_nodes
        for transition in graph_node_transitions(candidate, node)
    ]
    if not source_steps:
        raise ValueError("selected graph nodes contain no physical transition")
    transition_seqs = [item.transition.seq for item in source_steps]
    if len(transition_seqs) != len(set(transition_seqs)):
        raise ValueError("selected graph nodes contain duplicate transitions")
    return source_steps


def _validate_source_transition(transition: ObservedTransition) -> None:
    evidence = transition.evidence
    if evidence is None or evidence.result_status != "success":
        raise ValueError(
            f"source transition {transition.seq} did not execute successfully"
        )
    if any(
        action.risk_level.strip().casefold() == "sensitive"
        for action in transition.actions
    ):
        raise ValueError(
            f"source transition {transition.seq} contains a sensitive action"
        )
    if not is_supported_recipe_action_group(transition.actions):
        raise ValueError(
            f"source transition {transition.seq} is not replayable as recorded"
        )


def _input_slot(action: ObservedAction, candidate: RecipeCandidate) -> str:
    slot = normalize_text(action.parameter_slot())
    if not slot:
        return ""
    if action.action != "type_in_marker" or slot not in RECIPE_INPUT_NAMES:
        raise ValueError(f"invalid input slot for source action {action.source_seq}")
    search_keyword = normalize_text(candidate.collection_intent.search_keyword)
    observed_text = normalize_text(action.param.text)
    if slot == "search_keyword" and not (
        action.param.slot_name == "search_keyword"
        or (search_keyword and observed_text == search_keyword)
    ):
        raise ValueError(
            f"input slot has no source evidence for action {action.source_seq}"
        )
    return slot


def _rule_action(action: ObservedAction, candidate: RecipeCandidate) -> RuleAction:
    input_slot = _input_slot(action, candidate)
    target = None
    if action.target is not None:
        if not action.roi_signature.get("phash"):
            raise ValueError(f"target ROI missing for action {action.source_seq}")
        target = RuleTarget(
            reference=action.target.model_copy(deep=True),
            reference_roi_signature=dict(action.roi_signature),
        )

    param = action.param.model_copy(deep=True)
    if input_slot:
        param = param.model_copy(update={"text": "", "slot_name": input_slot})
    return RuleAction(
        source_seq=action.source_seq,
        action=action.action,
        target=target,
        param=param,
        input_slot=input_slot,
        risk_level=action.risk_level,
    )


def _expected_effect(source_step: RuleSourceStep) -> ExpectedEffect:
    transition = source_step.transition
    before = transition.before
    after = transition.after
    if before.url_template != after.url_template and after.url_template:
        kind = "url_change"
    elif before.page_role != after.page_role and after.page_role:
        kind = "page_change"
    else:
        kind = "screen_change"
    description = normalize_text(
        transition.expected_after
        or source_step.node.intended_result
        or transition.intent
        or source_step.node.purpose
    )
    return ExpectedEffect(
        kind=kind,
        description=description,
        expected_url_template=after.url_template,
        expected_page_role=after.page_role,
        reference_after_signature=dict(after.screen_context_signature),
    )


def compile_experience_rule(
    candidate: RecipeCandidate,
    execution_graph: CandidateExecutionGraph,
    selected_node_ids: list[str],
) -> ExperienceRule:
    """선택된 원본 전이를 의미 수정 없이 경험 규칙으로 옮긴다."""

    source_steps = _rule_source_steps(
        candidate,
        execution_graph,
        selected_node_ids,
    )
    for source_step in source_steps:
        _validate_source_transition(source_step.transition)

    steps: list[ExperienceRuleStep] = []
    input_slots: set[str] = set()
    step_ids_by_node: dict[str, list[str]] = {}
    for index, source_step in enumerate(source_steps, start=1):
        actions = [
            _rule_action(action, candidate)
            for action in source_step.transition.actions
        ]
        input_slots.update(action.input_slot for action in actions if action.input_slot)
        step_id = f"step-{index}"
        step_ids_by_node.setdefault(source_step.node.node_id, []).append(step_id)
        steps.append(
            ExperienceRuleStep(
                step_id=step_id,
                source_transition_seqs=[source_step.transition.seq],
                before=RuleScreen(
                    url_template=source_step.transition.before.url_template,
                    page_role=source_step.transition.before.page_role,
                    reference_signature=dict(
                        source_step.transition.before.screen_context_signature
                    ),
                ),
                actions=actions,
                intent=normalize_text(
                    source_step.transition.intent or source_step.node.purpose
                ),
                expected_effect=_expected_effect(source_step),
                source_node_id=source_step.node.node_id,
            )
        )

    nodes = [
        ExperienceRuleNode(
            node_id=node.node_id,
            purpose=node.purpose,
            source_event_seqs=list(node.source_event_seqs),
            step_ids=step_ids_by_node[node.node_id],
        )
        for node in execution_graph.nodes
        if node.node_id in step_ids_by_node
    ]
    return ExperienceRule(
        site=candidate.site,
        goal=candidate.goal,
        skill_metadata=RecipeSkillMetadata(
            task_category=normalize_task_category(
                candidate.collection_intent.task_category
            ),
            inputs=[{"name": name} for name in sorted(input_slots)],
        ),
        steps=steps,
        nodes=nodes,
    )


__all__ = ["compile_experience_rule"]
