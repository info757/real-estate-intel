"""Utilities to summarize and recommend micro-market configurations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from backend.ml.target_prep import build_micro_market_key


def _parse_micro_market(micro_market: str) -> Dict[str, Any]:
    parts = (micro_market or "unknown").split("|")
    while len(parts) < 6:
        parts.append("unknown")
    zip_code, property_type, price_bucket, sqft_bucket, beds, baths = parts[:6]
    return {
        "micro_market": micro_market,
        "zip_code": zip_code,
        "property_type": property_type,
        "price_bucket": price_bucket,
        "sqft_bucket": sqft_bucket,
        "beds": beds,
        "baths": baths,
    }


def build_micro_market_summary(
    listings: Iterable[Dict[str, Any]],
    *,
    min_listings: int = 5,
) -> pd.DataFrame:
    """Aggregate historical metrics for each micro-market configuration."""

    rows: List[Dict[str, Any]] = []
    for listing in listings:
        micro_market = listing.get("micro_market")
        if not micro_market:
            micro_market = build_micro_market_key(listing)
        dom = listing.get("dom_to_pending")
        if dom is None or dom < 0:
            continue
        fast_flag = listing.get("fast_seller_label")
        if fast_flag is None:
            continue
        parsed = _parse_micro_market(micro_market)
        rows.append({
            **parsed,
            "dom_to_pending": float(dom),
            "fast_seller": int(fast_flag),
            "dom_label_source": listing.get("dom_label_source", "unknown"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = (
        df.groupby([
            "micro_market",
            "zip_code",
            "property_type",
            "price_bucket",
            "sqft_bucket",
            "beds",
            "baths",
        ], as_index=False)
        .agg(
            n_listings=("dom_to_pending", "size"),
            median_dom_days=("dom_to_pending", "median"),
            p75_dom_days=("dom_to_pending", lambda s: float(s.quantile(0.75))),
            pct_fast_seller=("fast_seller", "mean"),
            observed_ratio=("dom_label_source", lambda s: float((s == "observed").mean() if len(s) else 0.0)),
        )
    )

    summary["pct_fast_seller"] = summary["pct_fast_seller"].mul(100).round(1)
    summary["observed_ratio"] = summary["observed_ratio"].round(3)
    summary = summary[summary["n_listings"] >= max(min_listings, 1)].reset_index(drop=True)
    summary.sort_values(["median_dom_days", "pct_fast_seller"], ascending=[True, False], inplace=True)
    return summary


def rank_micro_market_configurations(
    summary: pd.DataFrame,
    *,
    zip_code: Optional[str] = None,
    property_type: Optional[str] = None,
    min_observed_ratio: float = 0.1,
    top_k: int = 10,
) -> pd.DataFrame:
    """Return top configurations for a given filter."""

    if summary.empty:
        return summary

    subset = summary.copy()
    if zip_code:
        subset = subset[subset["zip_code"].astype(str) == str(zip_code)]
    if property_type:
        subset = subset[subset["property_type"].str.lower() == property_type.lower()]

    subset = subset[subset["observed_ratio"] >= min_observed_ratio]
    subset = subset.sort_values(["median_dom_days", "pct_fast_seller"], ascending=[True, False])
    return subset.head(top_k).reset_index(drop=True)


__all__ = [
    "build_micro_market_summary",
    "rank_micro_market_configurations",
]
