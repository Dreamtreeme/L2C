"""12개 대표 요청으로 실제 지휘자 모델의 계획 품질을 측정한다."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from agent.application.model_clients import get_structured_google_model
from agent.prompts.investigation import evidence_plan_prompt, request_analysis_prompt
from benchmark.investigation_quality_eval import evaluate_investigation_analysis
from benchmark.investigation_scenarios import INVESTIGATION_SCENARIOS
from shared.schema.investigation_schema import EvidencePlan, RequestAnalysis


def run_benchmark(*, model_name: str, now: datetime, scenario_id: str = "") -> dict:
    analysis_model = get_structured_google_model(
        model_name,
        RequestAnalysis,
        temperature=0.0,
    )
    evidence_model = get_structured_google_model(
        model_name,
        EvidencePlan,
        temperature=0.0,
    )
    selected = [
        scenario
        for scenario in INVESTIGATION_SCENARIOS
        if not scenario_id or scenario.scenario_id == scenario_id
    ]
    results = []
    for scenario in selected:
        try:
            analysis = analysis_model.invoke(
                [
                    SystemMessage(content=request_analysis_prompt(now)),
                    HumanMessage(content=scenario.query),
                ]
            )
            evidence_plan = None
            if not analysis.unresolved_fields:
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
            results.append(
                {
                    **evaluation,
                    "query": scenario.query,
                    "analysis": analysis.model_dump(mode="json"),
                    "evidence_plan": (
                        evidence_plan.model_dump(mode="json") if evidence_plan else None
                    ),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "passed": False,
                    "checks": {"execution": False},
                    "query": scenario.query,
                    "analysis": {},
                    "evidence_plan": None,
                    "error": str(exc),
                }
            )
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
        default=os.getenv("COMMANDER_MODEL", "gemini-3.5-flash"),
    )
    parser.add_argument("--date", default=datetime.now().date().isoformat())
    parser.add_argument("--scenario", default="")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    import shared.config  # noqa: F401 - 로컬 환경변수 로드를 보장한다.

    result = run_benchmark(
        model_name=args.model,
        now=datetime.fromisoformat(args.date),
        scenario_id=args.scenario,
    )
    output = result
    if args.summary:
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
                    "unresolved_fields": item["analysis"].get("unresolved_fields", []),
                    "error": item.get("error", ""),
                }
                for item in result["results"]
            ],
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
