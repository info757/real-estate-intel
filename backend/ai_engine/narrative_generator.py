"""
LLM-powered narrative generator for build recommendations.

Takes structured metrics for a recommendation and produces a concise
natural-language explanation grounded in the supplied values.
Falls back to a templated summary when no LLM is available.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from openai import OpenAI, OpenAIError

from config.settings import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
if settings.openai_api_key:
    try:
        _client = OpenAI(api_key=settings.openai_api_key)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to initialize OpenAI client: %s", exc)
        _client = None


def _build_prompt(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the system/user messages for the LLM call."""
    pretty_metrics = json.dumps(metrics, indent=2, sort_keys=True)

    system_prompt = (
        "You are a senior real-estate analyst preparing talking points for a demo. "
        "Summarise why a proposed build configuration is attractive using ONLY the "
        "numerical facts provided. Always treat sell_probability and expected_dom as the primary, "
        "model-driven metrics. Any seasonality_* fields represent historical baselines; describe them as "
        "'seasonality baseline' and never claim they boost or enhance the primary values—contrast instead. "
        "Never invent data or percentages; restate what is given. Keep it to 2 short sentences, focusing "
        "on demand velocity, pricing position, and margin."
    )

    user_prompt = (
        "Here are the metrics for the lot and recommended configuration:\n"
        f"{pretty_metrics}\n\n"
        "Provide a concise narrative in 2 sentences."
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    }


def _fallback_summary(metrics: Dict[str, Any]) -> str:
    """Deterministic summary when LLM is unavailable."""
    cfg = metrics.get("configuration", {})
    demand = metrics.get("demand", {})
    margin = metrics.get("margin", {})

    beds = cfg.get("beds")
    baths = cfg.get("baths")
    sqft = cfg.get("sqft")
    sell_prob = demand.get("sell_probability")
    sell_prob_source = demand.get("sell_probability_source")
    baseline_prob = demand.get("seasonality_fast_probability")
    dom = demand.get("expected_dom")
    dom_source = demand.get("expected_dom_source")
    baseline_dom = demand.get("seasonality_dom")
    margin_pct = margin.get("gross_margin_pct")
    pricing_block = metrics.get("pricing", {}) or {}
    price_ratio = pricing_block.get("price_to_subdivision_median")
    price_range = pricing_block.get("predicted_price_range")
    price_pred_pretty = pricing_block.get("predicted_sale_price_formatted")
    baseline_price_pretty = pricing_block.get("observed_sale_price_formatted")
    inventory_block = metrics.get("inventory", {}) or {}
    inventory = inventory_block.get("zip_inventory_trend_ratio")

    parts: list[str] = []
    parts.append(
        f"{beds or '?'}BR/{baths or '?'}BA at {sqft or 0:,} sqft scores "
        f"{(sell_prob or 0)*100:.0f}% sell probability ({sell_prob_source or 'triad'}) with "
        f"{dom or 0:.0f} day DOM ({dom_source or 'triad'})."
    )
    if baseline_prob is not None:
        parts.append(
            f"Seasonality baseline shows {(baseline_prob or 0)*100:.0f}% probability and {baseline_dom or 0:.0f} day DOM."
        )
    if price_pred_pretty:
        parts.append(f"Model pricing lands near {price_pred_pretty}.")
    elif baseline_price_pretty:
        parts.append(f"Baseline closed around {baseline_price_pretty}.")

    if margin_pct is not None:
        parts.append(f"Projected gross margin is {margin_pct:.1f}%.")
    if price_ratio is not None:
        parts.append(f"Pricing sits at {price_ratio:.2f}× the subdivision median.")
    if inventory is not None:
        parts.append(f"Inventory trend ratio (30/90d) is {inventory:.2f}, indicating velocity.")

    return " ".join(parts)


def generate_recommendation_narrative(metrics: Dict[str, Any]) -> str:
    """
    Produce a natural language explanation for a recommendation.

    Args:
        metrics: structured data describing the recommendation. Should include
                 configuration, demand, margin, and any contextual stats.

    Returns:
        A concise narrative string suitable for the frontend. Falls back to a
        templated summary if the OpenAI client is unavailable or errors.
    """
    if not metrics:
        return "Insufficient data to generate narrative."

    if not _client:
        return _fallback_summary(metrics)

    payload = _build_prompt(metrics)

    try:
        response = _client.chat.completions.create(
            model=settings.openai_model,
            messages=payload["messages"],
            temperature=0.2,
            max_tokens=180,
        )
        content = response.choices[0].message.content if response.choices else None
        if content:
            return content.strip()
    except OpenAIError as exc:
        logger.warning("OpenAI call failed: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected error generating narrative: %s", exc)

    return _fallback_summary(metrics)

