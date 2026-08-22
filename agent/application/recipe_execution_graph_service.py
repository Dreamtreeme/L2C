"""자율탐색 원본 로그 전체를 목적 단위 실행 그래프로 구조화한다."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.config import get_settings
from agent.recipe.task_category import normalize_task_category
from agent.utils.image_utils import image_to_base64_jpeg
from agent.utils.text import normalize_text, url_template
from shared.schema.execution_record_schema import (
    ObservedAction,
    ObservedTransition,
    ScreenCheckpoint,
)
from shared.schema.feedback_schema import (
    CandidateExecutionGraph,
    ExecutionEvent,
    ExecutionGraphNode,
    RecipeCandidate,
)


GraphBuilderFn = Callable[
    [dict[str, Any]],
    dict[str, Any] | CandidateExecutionGraph,
]


def _without_empty_values(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _compact_checkpoint(checkpoint: ScreenCheckpoint | None) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    return _without_empty_values(
        {
            "observation_id": checkpoint.observation_id,
            "url_template": url_template(checkpoint.url_template),
            "page_role": checkpoint.page_role,
        }
    )


def _compact_action(action: ObservedAction) -> dict[str, Any]:
    target = action.target
    return _without_empty_values(
        {
            "source_seq": action.source_seq,
            "action": action.action,
            "target": _without_empty_values(
                {
                    "text": target.text if target else "",
                    "semantic_label": target.semantic_label if target else "",
                    "marker_type": target.marker_type if target else "",
                    "region": target.region if target else "",
                }
            ),
            "parameters": action.param.model_dump(
                mode="json",
                exclude_defaults=True,
                exclude_none=True,
            ),
            "intent": action.intent,
            "target_role": action.target_role,
            "component": action.component,
            "risk_level": action.risk_level,
        }
    )


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = normalize_text(
        result.get("result") or result.get("reason") or result.get("error")
    )
    return _without_empty_values(
        {
            "status": result.get("status"),
            "summary": summary[:500],
            "screen_count": result.get("screen_count"),
            "ocr_chars": result.get("ocr_chars"),
            "queued_count": result.get("queued_count"),
            "queued_titles": result.get("queued_titles"),
        }
    )


def _compact_event(event: ExecutionEvent) -> dict[str, Any]:
    result = dict(event.result or {})
    transition = event.transition
    evidence = transition.evidence if transition else None
    actions = (
        list(transition.actions)
        if transition is not None
        else ([event.candidate_action] if event.candidate_action is not None else [])
    )
    return _without_empty_values(
        {
            "seq": event.seq,
            "source": result.get("action_source")
            or (evidence.source if evidence else ""),
            "before": _compact_checkpoint(event.before_checkpoint),
            "actions": [_compact_action(action) for action in actions],
            "intent": event.intent
            or (event.candidate_action.intent if event.candidate_action else ""),
            "result": _result_summary(result),
            "transition": (
                _without_empty_values(
                    {
                        "seq": transition.seq,
                        "after": _compact_checkpoint(transition.after),
                        "expected_after": transition.expected_after,
                        "intent": transition.intent,
                        "result_status": evidence.result_status if evidence else "",
                        "status": evidence.status if evidence else "",
                        "outcome": evidence.outcome if evidence else "",
                        "reason": evidence.reason if evidence else "",
                        "visual_change_ratio": (
                            evidence.visual_change_ratio if evidence else None
                        ),
                    }
                )
                if transition is not None
                else {}
            ),
        }
    )


def _compact_request(candidate: RecipeCandidate) -> dict[str, Any]:
    intent = candidate.collection_intent
    return _without_empty_values(
        {
            "original_query": intent.original_query,
            "site": candidate.site,
            "task_category": normalize_task_category(intent.task_category),
            "search_keyword": intent.search_keyword,
            "count_mode": intent.count_mode,
            "target_count": intent.target_count,
            "filters": _without_empty_values(intent.filters.model_dump()),
            "purpose": intent.purpose,
            "required_fields": list(intent.required_fields),
        }
    )


def build_candidate_graph_payload(candidate: RecipeCandidate) -> dict[str, Any]:
    """선제 가지치기 없이 모든 실행 이벤트를 시간순으로 반환한다."""

    return {
        "run_id": candidate.run_id,
        "goal": candidate.goal,
        "request": _compact_request(candidate),
        "flat_log": [_compact_event(event) for event in candidate.action_events],
        "final_result": _without_empty_values(
            {
                "run_status": candidate.run_status,
                "collected_count": candidate.collected_count,
                "persisted_count": candidate.persisted_count,
            }
        ),
    }


def graph_node_transitions(
    candidate: RecipeCandidate,
    node: ExecutionGraphNode,
) -> list[ObservedTransition]:
    """그래프 노드의 원본 이벤트에서 실제 물리 전이만 순서대로 찾는다."""

    events = {event.seq: event for event in candidate.action_events}
    return [
        event.transition
        for seq in node.source_event_seqs
        if (event := events.get(seq)) is not None and event.transition is not None
    ]


def candidate_screen_evidence(
    candidate: RecipeCandidate,
) -> list[tuple[str, Path]]:
    """그래프 구성 모델에 원본 실행 순서의 화면 근거를 전달한다."""

    limit = get_settings().recipe.critic_evidence_image_limit
    if limit <= 0:
        return []
    evidence: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()

    def append(label: str, raw_path: Any) -> None:
        if len(evidence) >= limit:
            return
        path_text = str(raw_path or "").strip()
        if not path_text:
            return
        path = Path(path_text)
        if path in seen_paths or not path.is_file():
            return
        seen_paths.add(path)
        evidence.append((label, path))

    for event in candidate.action_events:
        append(
            f"event {event.seq} before",
            event.result.get("before_marked_image")
            or event.result.get("before_screenshot"),
        )
        if event.transition and event.transition.evidence:
            append(
                f"event {event.seq} after",
                event.transition.evidence.marked_image
                or event.transition.evidence.screenshot,
            )
    return evidence


def _coerce_graph(
    raw: dict[str, Any] | CandidateExecutionGraph,
) -> CandidateExecutionGraph:
    return (
        raw
        if isinstance(raw, CandidateExecutionGraph)
        else CandidateExecutionGraph.model_validate(raw)
    )


def _graph_contract_errors(
    candidate: RecipeCandidate,
    graph: CandidateExecutionGraph,
) -> list[str]:
    expected = [event.seq for event in candidate.action_events]
    assigned = [seq for node in graph.nodes for seq in node.source_event_seqs]
    errors: list[str] = []
    if assigned != expected:
        errors.append(f"event coverage mismatch expected={expected} actual={assigned}")
    if graph.unassigned_event_seqs:
        errors.append(f"unassigned events={graph.unassigned_event_seqs}")
    node_ids = [node.node_id for node in graph.nodes]
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate graph node ids")
    known_nodes = set(node_ids)
    for edge in graph.edges:
        if edge.from_node not in known_nodes or edge.to_node not in known_nodes:
            errors.append(f"unknown graph edge={edge.from_node}->{edge.to_node}")
        if edge.from_node == edge.to_node:
            errors.append(f"self graph edge={edge.from_node}")
    return errors


def _llm_build_graph(
    payload: dict[str, Any],
    screen_evidence: list[tuple[str, Path]],
) -> CandidateExecutionGraph:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.llm.clients import get_structured_google_model
    from agent.llm.policy import commander_model_name
    from agent.observability.run_context import invoke_with_metrics
    from agent.prompts.trust_boundary import external_content_contract_en

    model_name = get_settings().models.recipe_critic_model or commander_model_name()
    llm = get_structured_google_model(
        model_name,
        CandidateExecutionGraph,
        temperature=0.4,
        execution_role="critic",
    )
    examples = [
        {
            "flat_log": [
                {"seq": 1, "intent": "직군 필터 열기"},
                {
                    "seq": 2,
                    "intent": "SW 개발 선택",
                    "transition": {"reason": "no_screen_change"},
                },
                {"seq": 3, "intent": "필터 적용"},
            ],
            "nodes": [
                {
                    "purpose": "직군 필터를 SW 개발로 설정하고 적용",
                    "source_event_seqs": [1, 2, 3],
                }
            ],
        },
        {
            "flat_log": [
                {"seq": 4, "intent": "필터 적용", "after": "job_detail"},
                {"seq": 5, "intent": "상세가 요청과 맞는지 확인"},
                {"seq": 6, "intent": "관련 없는 상세에서 목록으로 복귀"},
                {"seq": 7, "intent": "후보 탐색 재개"},
            ],
            "nodes": [
                {"purpose": "의도와 다른 상세 진입", "source_event_seqs": [4]},
                {"purpose": "진입한 상세의 적합성 확인", "source_event_seqs": [5]},
                {"purpose": "검색 목록으로 복구", "source_event_seqs": [6]},
                {"purpose": "후보 탐색 재개", "source_event_seqs": [7]},
            ],
            "relations": ["branch", "recovery", "next"],
        },
    ]
    system_text = (
        external_content_contract_en()
        + "\nYou are a post-hoc execution-graph observer. Convert the complete "
        "flat autonomous action log into a semantic execution graph. This stage "
        "constructs structure only; pruning happens later. Assign every seq "
        "exactly once in chronological order. Never invent, remove, reject, "
        "optimize, or classify actions. Each node is a contiguous action bundle "
        "that completes one independently reportable local purpose. Split when "
        "an intermediate result becomes the input of a different purpose. Group "
        "causally coupled setup and commit actions. Pixel change is not the node "
        "boundary: no_screen_change may represent hidden UI state. Compare intent "
        "with the observed result. Preserve mistaken branches, their inspection, "
        "and recovery as branch/recovery edges. A review followed by more work "
        "toward the same goal creates a feedback edge. Do not mark actions as "
        "reusable or variable. Leave unassigned_event_seqs empty.\nExamples: "
        + json.dumps(examples, ensure_ascii=False, separators=(",", ":"))
    )
    payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if screen_evidence:
        vision = get_settings().vision
        content: list[dict[str, Any]] = [{"type": "text", "text": payload_text}]
        for label, path in screen_evidence:
            encoded = image_to_base64_jpeg(
                path,
                max_dim=vision.reasoning_image_max_dim,
                quality=vision.reasoning_image_quality,
                fast=True,
            )
            content.extend(
                [
                    {"type": "text", "text": f"Screen evidence: {label}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ]
            )
        message = HumanMessage(content=content)
    else:
        message = HumanMessage(content=payload_text)
    return _coerce_graph(
        invoke_with_metrics(
            llm,
            [SystemMessage(content=system_text), message],
            "recipe_execution_graph",
        )
    )


def build_candidate_execution_graph(
    candidate: RecipeCandidate,
    builder: GraphBuilderFn | None = None,
) -> CandidateExecutionGraph:
    """원본 로그 전체를 그래프로 만들고 출처 보존 계약만 검사한다."""

    if not candidate.action_events:
        raise ValueError("candidate action log is empty")
    payload = build_candidate_graph_payload(candidate)
    graph = _coerce_graph(
        builder(payload)
        if builder is not None
        else _llm_build_graph(payload, candidate_screen_evidence(candidate))
    )
    errors = _graph_contract_errors(candidate, graph)
    if errors:
        raise ValueError("execution_graph_contract_failed: " + "; ".join(errors[:8]))
    return graph


__all__ = [
    "GraphBuilderFn",
    "build_candidate_execution_graph",
    "build_candidate_graph_payload",
    "candidate_screen_evidence",
    "graph_node_transitions",
]
