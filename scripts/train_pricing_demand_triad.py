#!/usr/bin/env python3
"""
Train pricing and demand models using the cached Triad sold listings.

This script reuses the fast-seller data collection pipeline so the pricing/demand
models stay aligned with the sanitized RealEstateApi ingestion (DOM filters,
deduplication, leakage fixes).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.train_fast_seller_model import (
    fetch_sold_listings_with_features,
    calculate_median_thresholds_by_zip,
    prepare_targets,
)
from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.pricing_model import PricingModel
from backend.ml.demand_model import DemandModel
from scripts.seasonality_adjusted_predictions import get_target_zip_codes

logger = logging.getLogger("train_pricing_demand_triad")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pricing and demand models on Triad sold listings.")
    parser.add_argument("--zip-codes", nargs="+", help="Explicit ZIP codes to train on (defaults to Triad cache).")
    parser.add_argument("--days-back", type=int, default=730, help="History window in days (default: 730).")
    parser.add_argument("--max-per-zip", type=int, default=None, help="Optional cap per ZIP.")
    parser.add_argument("--model-dir", default="models/triad_latest", help="Directory to save trained models.")
    parser.add_argument("--pricing-name", default="triad_pricing_model", help="Filename stem for pricing model.")
    parser.add_argument("--demand-name", default="triad_demand_model", help="Filename stem for demand model.")
    parser.add_argument("--sell-within-days", type=int, default=60, help="Fast-sale threshold for demand classifier.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache and pull fresh listings (slow).")
    parser.add_argument("--tune-pricing", action="store_true", help="Enable pricing model hyperparameter tuning.")
    parser.add_argument("--tune-demand", action="store_true", help="Enable demand model hyperparameter tuning.")
    return parser.parse_args()


def collect_listings(zip_codes: List[str], days_back: int, max_per_zip: int | None, use_cache: bool) -> List[Dict[str, Any]]:
    logger.info("Collecting sold listings: zips=%s days_back=%s max_per_zip=%s cache=%s",
                len(zip_codes), days_back, max_per_zip or "all", use_cache)
    listings = fetch_sold_listings_with_features(
        zip_codes=zip_codes,
        days_back=days_back,
        max_per_zip=max_per_zip,
        use_cache=use_cache,
    )
    if len(listings) < 50:
        raise RuntimeError(f"Only {len(listings)} listings collected. Need at least 50 samples.")
    logger.info("Collected %s listings across %s ZIP codes", len(listings), len({l.get('zip_code') for l in listings}))
    return listings


def build_feature_frames(listings: List[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    engineer = FeatureEngineer()
    feature_df = engineer.engineer_features(listings)

    # Pricing dataset (requires sale_price present)
    pricing_X, pricing_y = engineer.prepare_for_training(feature_df)
    price_mask = (pricing_y > 50_000) & (pricing_y < 3_000_000)
    if price_mask.sum() < len(pricing_y):
        dropped = int(len(pricing_y) - price_mask.sum())
        logger.info("Filtering %s pricing samples outside $50k-$3M range", dropped)
    pricing_X = pricing_X.loc[price_mask].reset_index(drop=True)
    pricing_y = pricing_y.loc[price_mask].reset_index(drop=True)

    # Demand targets (fast-seller classification + DOM regression)
    thresholds = calculate_median_thresholds_by_zip(listings)
    y_fast, y_dom, _dom_bucket, valid_indices = prepare_targets(listings, thresholds)
    if not valid_indices:
        raise RuntimeError("No valid DOM samples for demand model.")

    demand_df = feature_df.iloc[valid_indices].copy()
    available_features = [col for col in engineer.get_feature_names() if col in demand_df.columns]
    if not available_features:
        raise RuntimeError("No overlapping features found for demand model.")

    demand_X = demand_df[available_features].fillna(demand_df[available_features].median())
    y_fast = y_fast.reset_index(drop=True)
    y_dom = y_dom.reset_index(drop=True)
    demand_X = demand_X.reset_index(drop=True)

    return pricing_X, pricing_y, demand_X, (y_fast, y_dom)


def save_training_summary(output_dir: Path, payload: Dict[str, Any]) -> None:
    summary_path = output_dir / "triad_training_summary.json"
    with summary_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote training summary -> %s", summary_path)


def main() -> None:
    args = parse_args()
    zip_codes = args.zip_codes or get_target_zip_codes()
    if not zip_codes:
        raise RuntimeError("No ZIP codes provided or discovered.")

    listings = collect_listings(
        zip_codes=zip_codes,
        days_back=args.days_back,
        max_per_zip=args.max_per_zip,
        use_cache=not args.no_cache,
    )

    pricing_X, pricing_y, demand_X, (y_fast, y_dom) = build_feature_frames(listings)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training pricing model on %s samples / %s features", len(pricing_X), len(pricing_X.columns))
    pricing_model = PricingModel(model_dir=str(model_dir))
    pricing_metrics = pricing_model.train(
        X=pricing_X,
        y=pricing_y,
        hyperparameter_tuning=args.tune_pricing,
    )
    pricing_model_path = pricing_model.save(model_name=args.pricing_name)

    logger.info("Training demand model on %s samples / %s features", len(demand_X), len(demand_X.columns))
    demand_model = DemandModel(model_dir=str(model_dir))
    demand_metrics = demand_model.train(
        X=demand_X,
        y_sold_fast=y_fast,
        y_dom=y_dom,
        sell_within_days=args.sell_within_days,
        hyperparameter_tuning=args.tune_demand,
    )
    demand_model_path = demand_model.save(model_name=args.demand_name)

    summary = {
        "trained_at": datetime.utcnow().isoformat(),
        "zip_codes": zip_codes,
        "listing_count": len(listings),
        "pricing": {
            "model_path": pricing_model_path,
            "metrics": pricing_metrics,
        },
        "demand": {
            "model_path": demand_model_path,
            "metrics": demand_metrics,
            "sell_within_days": args.sell_within_days,
        },
        "feature_overview": {
            "pricing_feature_count": len(pricing_X.columns),
            "demand_feature_count": len(demand_X.columns),
        },
    }
    save_training_summary(model_dir, summary)

    logger.info("✅ Training complete! Pricing MAPE %.2f%% | Demand AUC %.3f | DOM MAPE %.2f%%",
                pricing_metrics["test_mape"],
                demand_metrics["sell_probability"]["test_auc"],
                demand_metrics["dom"]["test_mape"])


if __name__ == "__main__":
    main()

