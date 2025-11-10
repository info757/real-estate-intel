#!/usr/bin/env python3
"""
Generate seasonality-neutral fast-seller probabilities and DOM targets.

This script:
1. Loads cached Triad sold listings (730-day window).
2. Rebuilds the feature matrix using the latest feature engineering pipeline.
3. Neutralises seasonality by setting month/season features to their dataset means.
4. Loads the persisted fast-seller classifier and produces fast-sale probabilities.
5. Attaches DOM heuristics (median DOM per ZIP) for quick build recommendations.
6. Writes a CSV under reports/ with one row per listing.

The output can be used to aggregate by configuration (beds/baths/price buckets)
and drive build recommendations without seasonal bias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from config.settings import settings
from backend.ml.train_fast_seller_model import (
    fetch_sold_listings_with_features,
    calculate_median_thresholds_by_zip,
    prepare_targets,
    engineer_features,
)
from backend.ml.fast_seller_model import FastSellerModel

try:
    from scripts.identify_similar_zips import TRIAD_ZIPS
except ImportError:  # pragma: no cover - fallback if module structure changes
    TRIAD_ZIPS: List[str] = []

# Parameters
DAYS_BACK = 730
REPORT_PATH = Path("reports/seasonality_adjusted_predictions.csv")
SEASONAL_COLUMNS = [
    "sale_month",
    "sale_month_sin",
    "sale_month_cos",
    "sale_quarter",
    "sale_quarter_sin",
    "sale_quarter_cos",
    "is_spring_summer",
    "season",
    "list_month",
    "list_month_sin",
    "list_month_cos",
    "list_day_of_week",
    "list_day_of_week_sin",
    "list_day_of_week_cos",
]
FEATURE_COLUMNS_FOR_REPORT = [
    "price_to_zip_median",
    "price_per_sqft_to_zip",
    "is_above_zip_median_price",
    "is_below_zip_median_price",
    "sqft_to_zip_median",
    "quality_score",
    "condition_score",
    "overall_quality",
    "bath_bed_ratio",
    "beds_per_1000sqft",
    "price_per_sqft",
    "history_event_count",
    "history_listed_count",
    "history_pending_count",
    "history_sold_count",
    "history_price_change_count",
    "history_relisted",
    "days_from_list_to_pending",
    "days_from_list_to_sale",
    "prior_sale_gap_days",
    "price_to_subdivision_median",
    "price_per_sqft_to_subdivision",
    "sqft_to_subdivision_median",
    "zip_price_percentile_rank",
    "zip_sales_count_30d",
    "zip_sales_count_90d",
    "zip_inventory_trend_ratio",
]


def get_target_zip_codes() -> List[str]:
    """Return the list of ZIP codes we should evaluate."""
    if TRIAD_ZIPS:
        return TRIAD_ZIPS

    # Fallback: derive from cached listings directory
    cache_root = Path("cache/listings/realestateapi")
    zips = sorted({path.name.split("_")[1] for path in cache_root.glob("listings_*_730days*.json")})
    return zips


def neutralise_seasonality(features: pd.DataFrame) -> pd.DataFrame:
    """Set seasonal feature columns to their dataset mean values."""
    neutral = features.copy()
    for col in SEASONAL_COLUMNS:
        if col not in neutral.columns:
            continue
        mean_value = neutral[col].mean()
        neutral[col] = mean_value
    return neutral


def extract_listing_attributes(listings: List[Dict[str, Any]], indices: List[int]) -> pd.DataFrame:
    """Collect key attributes from listings for reporting."""
    rows = []
    for idx in indices:
        listing = listings[idx]
        summary = listing.get("summary", {}) or {}
        attr = {
            "property_id": listing.get("property_id") or summary.get("propertyId") or summary.get("id"),
            "zip_code": listing.get("zip_code") or listing.get("zipCode") or summary.get("postalCode"),
            "beds": listing.get("beds") or summary.get("beds"),
            "baths": listing.get("baths") or summary.get("bathstotal"),
            "sqft": listing.get("sqft") or summary.get("universalsize"),
            "sale_price": listing.get("sale_price") or listing.get("price") or summary.get("price"),
            "list_date": listing.get("list_date") or listing.get("listing_date") or summary.get("listDate"),
            "sale_date": listing.get("sale_date") or listing.get("dateSold") or summary.get("soldDate"),
            "property_type": listing.get("property_type") or summary.get("propertyType"),
            "latitude": listing.get("latitude")
            or summary.get("latitude")
            or (listing.get("geocode") or {}).get("lat"),
            "longitude": listing.get("longitude")
            or summary.get("longitude")
            or (listing.get("geocode") or {}).get("lon"),
            "subdivision": listing.get("subdivision")
            or (listing.get("summary") or {}).get("neighborhood", {}).get("name")
            or (listing.get("property_detail_raw") or {}).get("neighborhood", {}).get("name"),
        }
        rows.append(attr)
    return pd.DataFrame(rows)


def main() -> None:
    zip_codes = get_target_zip_codes()
    if not zip_codes:
        raise RuntimeError("No ZIP codes found for analysis.")

    listings = fetch_sold_listings_with_features(
        zip_codes=zip_codes,
        days_back=DAYS_BACK,
        max_per_zip=None,
        use_cache=True,
        parallel=False,
    )

    thresholds_by_zip = calculate_median_thresholds_by_zip(listings)
    y_fast_seller, y_dom, _y_dom_bucket, valid_indices = prepare_targets(listings, thresholds_by_zip)
    features = engineer_features(listings, valid_indices)

    neutral_features = neutralise_seasonality(features)

    model = FastSellerModel(model_dir="models")
    model.load()

    if model.feature_names:
        neutral_features = neutral_features.reindex(columns=model.feature_names, fill_value=0.0)

    fast_probs = model.predict_fast_seller_probability(neutral_features)

    listing_attrs = extract_listing_attributes(listings, valid_indices)

    dom_stats = model.model_metadata.get("dom_stats_by_zip", {})
    dom_global_median = model.model_metadata.get("dom_global_median", float(np.median(y_dom)))

    zip_dom = [
        dom_stats.get(str(z), {}).get("median", dom_global_median)
        for z in listing_attrs["zip_code"]
    ]

    feature_enrichment = []
    for col in FEATURE_COLUMNS_FOR_REPORT:
        if col in features.columns:
            feature_enrichment.append(col)

    enrichment_df = features[feature_enrichment].reset_index(drop=True) if feature_enrichment else pd.DataFrame()

    report_df = listing_attrs.assign(
        fast_seller_probability=np.round(fast_probs, 4),
        dom_zip_median=np.round(zip_dom, 1),
    )

    if not enrichment_df.empty:
        report_df = pd.concat([report_df, enrichment_df], axis=1)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False)

    print(f"Saved seasonality-neutral predictions to {REPORT_PATH.resolve()}")
    print(f"Rows: {len(report_df)} | Columns: {list(report_df.columns)}")


if __name__ == "__main__":
    main()

