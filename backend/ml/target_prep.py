"""Target preparation utilities for fast seller modeling."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

_DEFAULT_PRICE_BUCKET = 50000
_DEFAULT_SQFT_BUCKET = 500
_MIN_GAP_SAMPLES = 8


@dataclass
class DomInferenceStats:
    observed: int
    inferred: int
    skipped_no_dom_sold: int
    skipped_no_gap: int
    group_coverage: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "observed": self.observed,
            "inferred": self.inferred,
            "skipped_no_dom_sold": self.skipped_no_dom_sold,
            "skipped_no_gap": self.skipped_no_gap,
            "group_coverage": dict(self.group_coverage),
        }


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bucket(value: Optional[float], bucket_size: int) -> str:
    if value is None or bucket_size <= 0:
        return "na"
    bucket_floor = int(value // bucket_size) * bucket_size
    bucket_ceil = bucket_floor + bucket_size
    return f"{bucket_floor}-{bucket_ceil}"


def build_micro_market_key(
    listing: Dict[str, Any],
    *,
    price_bucket: int = _DEFAULT_PRICE_BUCKET,
    sqft_bucket: int = _DEFAULT_SQFT_BUCKET,
) -> str:
    zip_code = str(
        listing.get("zip_code")
        or listing.get("zipCode")
        or listing.get("address_zip")
        or "unknown"
    )

    metadata = listing.get("metadata") or {}
    property_type = (
        metadata.get("property_type_category")
        or metadata.get("property_type_label")
        or listing.get("property_type")
        or listing.get("property_type_label")
        or "unknown"
    )
    property_type = str(property_type).lower()

    price = _safe_float(
        listing.get("sale_price")
        or listing.get("sold_price")
        or listing.get("price")
        or listing.get("list_price")
    )
    sqft = _safe_float(listing.get("sqft") or listing.get("living_sqft"))

    beds = listing.get("beds")
    baths = listing.get("baths") or listing.get("bathstotal")

    price_bucket_label = _bucket(price, price_bucket)
    sqft_bucket_label = _bucket(sqft, sqft_bucket)
    beds_label = str(int(beds)) if beds is not None else "na"
    baths_label = str(int(baths)) if baths is not None else "na"

    return "|".join([
        zip_code,
        property_type,
        price_bucket_label,
        sqft_bucket_label,
        beds_label,
        baths_label,
    ])


def assign_micro_market(
    listings: Iterable[Dict[str, Any]],
    *,
    price_bucket: int = _DEFAULT_PRICE_BUCKET,
    sqft_bucket: int = _DEFAULT_SQFT_BUCKET,
) -> None:
    for listing in listings:
        listing["micro_market"] = build_micro_market_key(
            listing,
            price_bucket=price_bucket,
            sqft_bucket=sqft_bucket,
        )


def _zip_key(listing: Dict[str, Any]) -> str:
    value = listing.get("zip_code") or listing.get("zipCode")
    return str(value) if value else "unknown"


def infer_dom_to_pending(
    listings: List[Dict[str, Any]],
    *,
    group_by: str = "zip",
    min_samples: int = _MIN_GAP_SAMPLES,
) -> DomInferenceStats:
    """Backfill missing DOM-to-pending values using DOM-to-sold gaps."""

    if not listings:
        return DomInferenceStats(0, 0, 0, 0, {})

    if group_by == "zip":
        key_fn: Callable[[Dict[str, Any]], str] = _zip_key
    elif group_by == "micro_market":
        assign_micro_market(listings)
        key_fn = lambda listing: listing.get("micro_market", "unknown")
    else:
        raise ValueError(f"Unsupported group_by='{group_by}'")

    gap_by_group: Dict[str, List[float]] = defaultdict(list)
    observed = 0

    for listing in listings:
        dom_pending = listing.get("dom_to_pending")
        dom_sold = listing.get("dom_to_sold")
        if dom_pending is None or dom_sold is None:
            continue
        if dom_pending < 0 or dom_sold < 0:
            continue
        gap = dom_sold - dom_pending
        if gap < 0:
            continue
        key = key_fn(listing)
        gap_by_group[key].append(float(gap))
        listing.setdefault("dom_label_source", "observed")
        observed += 1

    all_gaps = [gap for gaps in gap_by_group.values() for gap in gaps]
    global_median = float(np.median(all_gaps)) if all_gaps else None

    group_median: Dict[str, float] = {}
    for key, gaps in gap_by_group.items():
        if not gaps:
            continue
        median_value = float(np.median(gaps))
        group_median[key] = median_value

    inferred = 0
    skipped_no_dom_sold = 0
    skipped_no_gap = 0
    coverage_counter: Counter[str] = Counter({k: len(v) for k, v in gap_by_group.items()})

    for listing in listings:
        dom_pending = listing.get("dom_to_pending")
        dom_sold = listing.get("dom_to_sold")
        if dom_pending is not None and dom_pending >= 0:
            listing.setdefault("dom_label_source", "observed")
            continue

        if dom_sold is None or dom_sold < 0:
            listing["dom_label_source"] = "missing_dom_to_sold"
            skipped_no_dom_sold += 1
            continue

        key = key_fn(listing)
        gap = None
        label_source = None

        gaps_for_group = gap_by_group.get(key, [])
        if len(gaps_for_group) >= min_samples:
            gap = group_median.get(key)
            label_source = "modeled_group"
        elif gaps_for_group:
            gap = group_median.get(key)
            label_source = "modeled_group_small"

        if gap is None and global_median is not None:
            gap = global_median
            label_source = "modeled_global"

        if gap is None:
            listing["dom_label_source"] = "missing_gap"
            skipped_no_gap += 1
            continue

        inferred_dom = max(dom_sold - gap, 0)
        listing["dom_to_pending"] = float(inferred_dom)
        listing["pending_to_sold"] = float(gap)
        listing["dom_label_source"] = label_source
        inferred += 1

    return DomInferenceStats(
        observed=observed,
        inferred=inferred,
        skipped_no_dom_sold=skipped_no_dom_sold,
        skipped_no_gap=skipped_no_gap,
        group_coverage=dict(coverage_counter),
    )


__all__ = [
    "DomInferenceStats",
    "assign_micro_market",
    "build_micro_market_key",
    "infer_dom_to_pending",
]
