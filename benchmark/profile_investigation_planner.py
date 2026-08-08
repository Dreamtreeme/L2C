"""12개 대표 요청으로 실제 지휘자 모델의 계획 품질을 측정한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm.clients import get_structured_google_model
from agent.application.search_taxonomy_service import SearchTaxonomyService
from agent.prompts.investigation import evidence_plan_prompt, request_analysis_prompt
from benchmark.investigation_quality_eval import evaluate_investigation_analysis
from benchmark.investigation_scenarios import INVESTIGATION_SCENARIOS
from shared.schema.investigation_schema import EvidencePlan, RequestAnalysis


def run_benchmark(
    *,
    model_name: str,
    now: datetime,
    scenario_id: str = "",
    db_path: str | Path = ROOT / "data" / "jobs.db",
    max_concurrency: int = 3,
) -> dict:
    analysis_model = get_structured_google_model(
        model_name,
        RequestAnalysis,
        temperature=0.0,
        execution_role="commander",
    )
    evidence_model = get_structured_google_model(
        model_name,
        EvidencePlan,
        temperature=0.0,
        execution_role="commander",
    )
    taxonomy = SearchTaxonomyService(db_path)
    selected_ids = {
        value.strip()
        for value in scenario_id.split(",")
        if value.strip()
    }
    selected = [
        scenario
        for scenario in INVESTIGATION_SCENARIOS
        if not selected_ids or scenario.scenario_id in selected_ids
    ]

    def evaluate_scenario(scenario):
        try:
            analysis = analysis_model.invoke(
                [
                    SystemMessage(content=request_analysis_prompt(now)),
                    HumanMessage(content=scenario.query),
                ]
            )
            constraints = taxonomy.enrich_constraints(analysis.constraints)
            questions = [
                question
                for question in analysis.clarification_questions
                if not (
                    question.field == "occupation_query"
                    and constraints.occupation_concept_keys
                )
            ]
            analysis = analysis.model_copy(
                update={
                    "constraints": constraints,
                    "clarification_questions": questions,
                }
            )
            evidence_plan = None
            if not analysis.clarification_questions:
                evidence_plan = evidence_model.invoke(
                    [
                        SystemMessage(content=evidence_plan_prompt(now)),
                        HumanMessage(
                            content=json.dumps(
                                {"request": analysis.model_dump(mode="json")},
                                ensure_ascii=False,
                            )
                        ),
                    ]
                )
            evaluation = evaluate_investigation_analysis(
                scenario,
                analysis,
                evidence_plan,
            )
            return {
                **evaluation,
                "query": scenario.query,
                "analysis": analysis.model_dump(mode="json"),
                "evidence_plan": (
                    evidence_plan.model_dump(mode="json") if evidence_plan else None
                ),
            }
        except Exception as exc:
            return {
                "scenario_id": scenario.scenario_id,
                "passed": False,
                "checks": {"execution": False},
                "query": scenario.query,
                "analysis": {},
                "evidence_plan": None,
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as executor:
        results = list(executor.map(evaluate_scenario, selected))
    return {
        "model": model_name,
        "date": now.date().isoformat(),
        "passed": sum(1 for item in results if item["passed"]),
        "total": len(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.getenv("COMMANDER_MODEL", "gemini-3.6-flash"),
    )
    parser.add_argument("--date", default=datetime.now().date().isoformat())
    parser.add_argument("--scenario", default="")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "jobs.db")
    parser.add_argument("--max-concurrency", type=int, default=3)
    args = parser.parse_args()

    result = run_benchmark(
        model_name=args.model,
        now=datetime.fromisoformat(args.date),
        scenario_id=args.scenario,
        db_path=args.db,
        max_concurrency=args.max_concurrency,
    )
    output = result
    if args.summary:
        displayed_results = result["results"]
        if args.failures_only:
            displayed_results = [item for item in displayed_results if not item["passed"]]
        output = {
            "model": result["model"],
            "date": result["date"],
            "passed": result["passed"],
            "total": result["total"],
            "results": [
                {
                    "scenario_id": item["scenario_id"],
                    "passed": item["passed"],
                    "checks": item["checks"],
                    "purpose": item["analysis"].get("purpose", ""),
                    "evidence_policy": item["analysis"].get("evidence_policy", ""),
                    "occupation_query": item["analysis"].get("constraints", {}).get(
                        "occupation_query", ""
                    ),
                    "collection_search_term": item["analysis"].get(
                        "constraints", {}
                    ).get("collection_search_term", ""),
                    "clarification_fields": [
                        question.get("field", "")
                        for question in item["analysis"].get("clarification_questions", [])
                    ],
                    "error": item.get("error", ""),
                }
                for item in displayed_results
            ],
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
