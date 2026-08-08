"""실제 토큰 사용량을 외부 모델 가격표로 환산한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.config import get_settings


DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[2] / "config" / "model_pricing.json"


def _non_negative_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def load_model_pricing(path: str | Path | None = None) -> tuple[dict[str, Any], str]:
    """가격표를 읽는다. 파일이 없거나 잘못됐으면 비용을 임의 추정하지 않는다."""

    configured = path or get_settings().paths.llm_pricing_file or DEFAULT_PRICING_PATH
    pricing_path = Path(configured).expanduser()
    if not pricing_path.is_absolute():
        pricing_path = Path.cwd() / pricing_path
    try:
        payload = json.loads(pricing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, ""
    models = payload.get("models") if isinstance(payload, dict) else None
    return (models if isinstance(models, dict) else {}), str(pricing_path.resolve())


def estimate_llm_cost(
    usage_by_model: dict[str, dict[str, Any]],
    *,
    pricing_path: str | Path | None = None,
) -> dict[str, Any]:
    """모델별 사용량을 USD로 계산하고 가격이 없는 모델은 명시적으로 남긴다."""

    prices, source = load_model_pricing(pricing_path)
    items: list[dict[str, Any]] = []
    total = 0.0
    priced_models = 0
    unpriced_models: list[str] = []

    for model, usage in sorted(usage_by_model.items()):
        input_tokens = max(0, int(usage.get("input_tokens") or 0))
        output_tokens = max(0, int(usage.get("output_tokens") or 0))
        input_details = usage.get("input_token_details") or {}
        cached_tokens = min(
            input_tokens,
            max(0, int(input_details.get("cache_read") or input_details.get("cached") or 0)),
        )
        rate = prices.get(model) if isinstance(prices.get(model), dict) else {}
        input_rate = _non_negative_float(rate.get("input_usd_per_million"))
        output_rate = _non_negative_float(rate.get("output_usd_per_million"))
        cached_rate = _non_negative_float(rate.get("cached_input_usd_per_million"))

        if input_rate is None or output_rate is None:
            unpriced_models.append(model)
            items.append(
                {
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_cost_usd": None,
                }
            )
            continue

        uncached_tokens = input_tokens - cached_tokens
        effective_cached_rate = input_rate if cached_rate is None else cached_rate
        cost = (
            uncached_tokens * input_rate
            + cached_tokens * effective_cached_rate
            + output_tokens * output_rate
        ) / 1_000_000
        total += cost
        priced_models += 1
        items.append(
            {
                "model": model,
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": round(cost, 9),
            }
        )

    return {
        "currency": "USD",
        "estimated_total": round(total, 9) if priced_models else None,
        "pricing_source": source,
        "priced_model_count": priced_models,
        "unpriced_models": unpriced_models,
        "items": items,
    }


__all__ = ["estimate_llm_cost", "load_model_pricing"]
