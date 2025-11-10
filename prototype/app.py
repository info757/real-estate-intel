"""
Streamlit prototype for Real Estate Intelligence Platform.
Rapid prototype for client demos with dashboard, analysis, and AI chat.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import sys
import os
from typing import Dict, Any, List, Optional, Set, Callable
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data_collectors.market_data import MarketAnalysisCollector
from backend.data_collectors.land_scraper import LandScraperOrchestrator
from backend.data_collectors.sales_data import ProductOptimizationAnalyzer
from backend.analyzers.submarket_ranker import SubmarketRanker
from backend.analyzers.land_analyzer import LandOpportunityAnalyzer
from backend.analyzers.financial_optimizer import FinancialOptimizer
from backend.ai_engine.rag_system import QdrantRAGSystem
from backend.analyzers.feature_analyzer import feature_analyzer
from backend.analyzers.demand_predictor import demand_predictor
from backend.data_collectors.safe_listings_scraper import safe_listings_scraper
from backend.analyzers.popularity_analyzer import popularity_analyzer
from backend.ml.guardrails import guardrails
from backend.ml.backtesting import backtester
from backend.ai_engine.narrative_generator import generate_recommendation_narrative
from backend.ml.pricing_model import PricingModel
from backend.ml.demand_model import DemandModel
from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.train_fast_seller_model import fetch_sold_listings_with_features
from backend.ml.micro_market_recommender import rank_micro_market_configurations
from backend.ml.recommendation_engine import RecommendationEngine
from backend.ml.feature_tags import tag_listing_features, FEATURE_LIBRARY
from config.settings import settings

try:  # Optional dependency for feature impact regression
    from sklearn.linear_model import RidgeCV
except Exception:  # pragma: no cover - degrade gracefully if sklearn is unavailable
    RidgeCV = None

PROPERTY_TYPE_SELECT_TO_NORM: Dict[str, Optional[str]] = {
    "Any": None,
    "Single Family": "single_family",
    "Townhome": "townhome",
    "Condo": "condo",
}

PROPERTY_TYPE_SELECT_TO_ENGINE: Dict[str, str] = {
    "Any": "Single Family Home",
    "Single Family": "Single Family Home",
    "Townhome": "Townhome",
    "Condo": "Condo",
}

PROPERTY_TYPE_ALIASES: Dict[str, str] = {
    "sfr": "single_family",
    "sf": "single_family",
    "single family": "single_family",
    "single-family": "single_family",
    "singlefamily": "single_family",
    "single family residence": "single_family",
    "single_family": "single_family",
    "sfh": "single_family",
    "detached": "single_family",
    "residential": "single_family",
    "other": "single_family",
    "sfd": "single_family",
    "townhome": "townhome",
    "town home": "townhome",
    "town house": "townhome",
    "townhouse": "townhome",
    "attached": "townhome",
    "th": "townhome",
    "condo": "condo",
    "condominium": "condo",
    "apartment": "condo",
    "multi-family": "multi_family",
    "multi family": "multi_family",
    "multifamily": "multi_family",
}

FEATURE_IMPACT_MIN_SAMPLES = 5
MIN_PRICE_LIFT_DOLLARS = 2500.0
MIN_DOM_REDUCTION_DAYS = 0.25


def normalize_property_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    label = str(value).strip().lower()
    if not label:
        return None
    return PROPERTY_TYPE_ALIASES.get(label, label)


def parse_bucket_range(bucket: Any) -> tuple[Optional[float], Optional[float]]:
    if bucket is None:
        return (None, None)
    label = str(bucket).strip()
    if not label or label.lower() == "na":
        return (None, None)
    if "-" not in label:
        try:
            value = float(label)
            return (value, value)
        except ValueError:
            return (None, None)
    start_str, end_str = label.split("-", 1)
    try:
        return (float(start_str), float(end_str))
    except ValueError:
        return (None, None)


def _make_feature_record(key: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    label = data.get("label", key.replace("_", " ").title())
    label_lower = label.lower()
    key_lower = str(key).lower()
    if "pool" in label_lower or "pool" in key_lower:
        return None

    count = data.get("count")
    price_lift = safe_float(data.get("price_lift"))
    price_lift_pct = safe_float(data.get("price_lift_pct"))
    dom_delta = safe_float(data.get("dom_delta"))

    price_band_raw = data.get("price_lift_bands") or {}
    dom_band_raw = data.get("dom_delta_bands") or {}

    price_band = {
        str(band): safe_float(val)
        for band, val in price_band_raw.items()
        if safe_float(val) is not None
    }
    dom_band = {
        str(band): safe_float(val)
        for band, val in dom_band_raw.items()
        if safe_float(val) is not None
    }

    record = {
        "key": key,
        "label": label,
        "count": count,
        "price_lift": price_lift,
        "price_lift_pct": price_lift_pct,
        "dom_delta": dom_delta,
        "price_method": data.get("price_method"),
        "dom_method": data.get("dom_method"),
        "price_lift_bands": price_band if price_band else None,
        "dom_delta_bands": dom_band if dom_band else None,
    }
    best_price_band = _best_positive_price_band(record)
    best_dom_band = _best_negative_dom_band(record)

    price_signal: Optional[float] = None
    if price_lift is not None and price_lift > 0:
        price_signal = price_lift
    elif best_price_band:
        price_signal = best_price_band[1]

    dom_signal: Optional[float] = None
    if dom_delta is not None and dom_delta < 0:
        dom_signal = dom_delta
    elif best_dom_band:
        dom_signal = best_dom_band[1]

    record["best_price_band"] = best_price_band
    record["best_dom_band"] = best_dom_band
    record["price_signal"] = price_signal
    record["dom_signal"] = dom_signal
    return record


def _best_positive_price_band(record: Dict[str, Any]) -> Optional[tuple[str, float]]:
    bands = record.get("price_lift_bands") or {}
    best_band = None
    best_val: Optional[float] = None
    for band, value in bands.items():
        if value is None:
            continue
        if value > 0 and (best_val is None or value > best_val):
            best_val = value
            best_band = band
    if best_band is None or best_val is None:
        return None
    return best_band, best_val


def _best_negative_dom_band(record: Dict[str, Any]) -> Optional[tuple[str, float]]:
    bands = record.get("dom_delta_bands") or {}
    best_band = None
    best_val: Optional[float] = None
    for band, value in bands.items():
        if value is None:
            continue
        if value < 0 and (best_val is None or value < best_val):
            best_val = value
            best_band = band
    if best_band is None or best_val is None:
        return None
    return best_band, best_val


def _price_sort_score(record: Dict[str, Any]) -> tuple[float, float, float]:
    signal = record.get("price_signal") or 0.0
    pct = record.get("price_lift_pct") or 0.0
    count = record.get("count") or 0.0
    return (signal, pct, count)


def _dom_sort_score(record: Dict[str, Any]) -> tuple[float, float]:
    signal = record.get("dom_signal") or 0.0
    count = -(record.get("count") or 0.0)
    return (signal, count)


def extract_top_feature_moves(
    impacts: Optional[Dict[str, Dict[str, Any]]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return price-boosting and DOM-reducing features sorted by impact."""
    if not impacts:
        return [], []

    price_levers: List[Dict[str, Any]] = []
    dom_levers: List[Dict[str, Any]] = []

    for key, data in impacts.items():
        record = _make_feature_record(key, data)
        if not record:
            continue

        price_signal = record.get("price_signal")
        if price_signal is not None and price_signal >= MIN_PRICE_LIFT_DOLLARS:
            price_levers.append(record.copy())

        dom_signal = record.get("dom_signal")
        if dom_signal is not None and abs(dom_signal) >= MIN_DOM_REDUCTION_DAYS:
            dom_levers.append(record.copy())

    price_levers.sort(key=_price_sort_score, reverse=True)
    dom_levers.sort(key=_dom_sort_score)
    return price_levers, dom_levers

GLOBAL_CSS = """
<style>
:root {
    --bg: #0f172a;
    --card-bg: rgba(15, 23, 42, 0.72);
    --accent: #38bdf8;
    --accent-soft: rgba(56, 189, 248, 0.12);
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5f5;
    --radius: 18px;
}

.block-container {
    padding-top: 2.4rem;
    padding-bottom: 2.6rem;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.1rem 0;
    flex-wrap: wrap;
}

.metric-card {
    flex: 1 1 240px;
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 1.35rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.35);
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 25px 55px rgba(15, 23, 42, 0.45);
}

.metric-title {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
}

.metric-value {
    font-size: 1.95rem;
    font-weight: 700;
    color: var(--text-primary);
}

.metric-footnote {
    margin-top: 0.6rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.25rem;
    margin-top: 1.25rem;
}

.alt-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 1rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
}

.alt-card-title {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.35rem;
}

.alt-card-metric {
    display: flex;
    justify-content: space-between;
    color: var(--text-secondary);
    font-size: 0.9rem;
    padding: 0.2rem 0;
}

.inline-highlight {
    color: var(--accent);
    font-weight: 600;
}
</style>
"""


def render_metric_card(title: str, value: str, footnote: Optional[str] = None) -> None:
    foot_html = f'<div class="metric-footnote">{footnote}</div>' if footnote else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {foot_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_seasonality_report() -> pd.DataFrame:
    report_path = Path("reports/seasonality_adjusted_predictions.csv")
    if not report_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(report_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["property_type_norm"] = df["property_type"].apply(normalize_property_type)
    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    df["baths"] = pd.to_numeric(df["baths"], errors="coerce")
    df["sqft"] = pd.to_numeric(df["sqft"], errors="coerce")
    return df


@st.cache_data
def load_micro_market_summary() -> pd.DataFrame:
    summary_path = Path("models/micro_market_summary.csv")
    if not summary_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(summary_path)
    df["property_type"] = df["property_type"].astype(str).str.lower()
    df["property_type_norm"] = df["property_type"].apply(normalize_property_type)
    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    df["baths"] = pd.to_numeric(df["baths"], errors="coerce")
    df["median_dom_days"] = pd.to_numeric(df["median_dom_days"], errors="coerce")
    df["pct_fast_seller"] = pd.to_numeric(df["pct_fast_seller"], errors="coerce")
    return df


def _find_listing_for_configuration(
    seasonality_df: pd.DataFrame,
    *,
    zip_code: str,
    beds: Optional[float],
    baths: Optional[float],
    sqft_bucket: Optional[str],
    property_type_norm: Optional[str],
    subdivision: Optional[str],
    target_lat: Optional[float] = None,
    target_lon: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    subset = seasonality_df[
        seasonality_df["zip_code"].astype(str) == str(zip_code)
    ].copy()
    if subset.empty:
        return None
    if property_type_norm:
        pt_subset = subset[subset["property_type_norm"] == property_type_norm]
        if pt_subset.empty:
            # Allow fallback when feed uses broader labels (e.g. 'other' for SFR)
            alias_matches = subset[
                subset["property_type_norm"].apply(
                    lambda val: normalize_property_type(val) == property_type_norm
                    if val is not None
                    else False
                )
            ]
            if not alias_matches.empty:
                pt_subset = alias_matches
        if not pt_subset.empty:
            subset = pt_subset
    if subdivision:
        sub_subset = subset[
            subset["subdivision"].fillna("").str.lower() == subdivision.lower()
        ]
        if not sub_subset.empty:
            subset = sub_subset
    if beds is not None and np.isfinite(beds):
        subset = subset[np.isclose(subset["beds"], beds, atol=0.25)]
    if baths is not None and np.isfinite(baths):
        subset = subset[np.isclose(subset["baths"], baths, atol=0.25)]
    low, high = parse_bucket_range(sqft_bucket)
    if low is not None and high is not None:
        subset = subset[(subset["sqft"] >= low) & (subset["sqft"] <= high)]
    subset = subset.dropna(subset=["sale_price"])
    if subset.empty:
        return None

    if target_lat is not None and target_lon is not None:
        subset["distance"] = np.sqrt(
            (subset["latitude"] - target_lat) ** 2 + (subset["longitude"] - target_lon) ** 2
        )
    else:
        subset["distance"] = 0.0

    subset = subset.sort_values(
        ["distance", "sale_price", "fast_seller_probability"],
        ascending=[True, False, False],
    )
    return subset.iloc[0].to_dict()


def generate_candidate_recommendations(
    *,
    zip_code: str,
    property_type_label: str,
    subdivision: Optional[str],
    triad_cache: Dict[str, Any],
    target_lat: Optional[float],
    target_lon: Optional[float],
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    engine = get_recommendation_engine()
    property_type = PROPERTY_TYPE_SELECT_TO_ENGINE.get(property_type_label, "Single Family Home")
    lot_features = {
        "zip_code": zip_code,
        "latitude": target_lat,
        "longitude": target_lon,
        "subdivision": subdivision,
        "lot_size_acres": 0.25,
        "lot_condition": "flat",
        "utilities_status": "all_utilities",
    }

    engine_recommendations: List[Dict[str, Any]] = []
    engine_error: Optional[str] = None
    try:
        historical_data = load_listings_for_zip(zip_code)
        if historical_data and subdivision:
            subdivision_key = subdivision.strip().lower()

            def _listing_subdivision_name(listing: Dict[str, Any]) -> Optional[str]:
                summary = listing.get("summary") or {}
                property_detail = listing.get("property_detail_raw") or {}
                candidates = [
                    listing.get("subdivision"),
                    (summary.get("neighborhood") or {}).get("name"),
                    summary.get("subdivision"),
                    (property_detail.get("neighborhood") or {}).get("name"),
                ]
                for candidate in candidates:
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip().lower()
                return None

            filtered = [
                listing
                for listing in historical_data
                if _listing_subdivision_name(listing) == subdivision_key
            ]
            if filtered:
                historical_data = filtered
        engine_payload = engine.generate_recommendations(
            lot_features=lot_features,
            property_type=property_type,
            historical_data=historical_data,
            top_n=top_k,
            use_market_insights=False,
        )
        engine_context = {
            "total_evaluated": engine_payload.get("total_evaluated"),
            "total_passing": engine_payload.get("total_passing_constraints"),
            "candidate_source": engine_payload.get("candidate_source"),
            "constraints": engine_payload.get("constraints"),
            "lot_features": engine_payload.get("lot_features"),
            "property_type": engine_payload.get("property_type"),
            "evaluation_errors": engine_payload.get("evaluation_errors"),
        }
        for idx, rec in enumerate(engine_payload.get("recommendations", []), start=1):
            config = rec.get("configuration", {})
            demand = rec.get("demand", {})
            margin = rec.get("margin", {})

            listing = {
                "property_id": f"engine-{zip_code}-{idx}",
                "zip_code": zip_code,
                "subdivision": subdivision,
                "property_type": property_type_label,
                "beds": config.get("beds"),
                "baths": config.get("baths"),
                "sqft": config.get("sqft"),
                "finish_level": config.get("finish_level"),
                "sale_price": rec.get("predicted_price"),
                "fast_seller_probability": demand.get("sell_probability"),
                "dom_zip_median": demand.get("expected_dom"),
                "engine_margin_pct": margin.get("gross_margin_pct"),
                "engine_margin": margin.get("gross_margin"),
                "engine_price": rec.get("predicted_price"),
            }

            triad_predictions = {
                "predicted_price": rec.get("predicted_price"),
                "sell_probability": demand.get("sell_probability"),
                "expected_dom": demand.get("expected_dom"),
                "gross_margin_pct": margin.get("gross_margin_pct"),
            }

            engine_recommendations.append(
                {
                    "listing": listing,
                    "triad_predictions": triad_predictions,
                    "engine_recommendation": rec,
                    "engine_context": engine_context,
                }
            )
    except Exception as exc:  # pragma: no cover - defensive
        engine_error = str(exc)
        st.warning(f"Builder engine unavailable: {exc}")

    if engine_recommendations:
        return engine_recommendations

    # Fallback to micro-market analogue approach if engine yields nothing
    seasonality_df = load_seasonality_report()
    micro_summary = load_micro_market_summary()
    if seasonality_df.empty or micro_summary.empty:
        return []

    property_type_norm = PROPERTY_TYPE_SELECT_TO_NORM.get(property_type_label)
    ranked = rank_micro_market_configurations(
        micro_summary,
        zip_code=zip_code,
        property_type=property_type_norm,
        top_k=top_k,
        min_observed_ratio=0.0,
    )
    if ranked.empty:
        return []

    recommendations: List[Dict[str, Any]] = []
    for _, row in ranked.iterrows():
        beds = row.get("beds")
        baths = row.get("baths")
        sqft_bucket = row.get("sqft_bucket")
        listing = _find_listing_for_configuration(
            seasonality_df,
            zip_code=zip_code,
            beds=beds,
            baths=baths,
            sqft_bucket=sqft_bucket,
            property_type_norm=property_type_norm,
            subdivision=subdivision,
            target_lat=target_lat,
            target_lon=target_lon,
        )
        if listing is None:
            continue

        cache_key = "|".join(
            [
                str(zip_code),
                str(listing.get("property_id") or listing.get("propertyId") or listing.get("id") or ""),
                str(listing.get("latitude") or ""),
                str(listing.get("longitude") or ""),
            ]
        )
        if cache_key not in triad_cache:
            triad_cache[cache_key] = compute_tri_model_predictions(zip_code, listing)
        triad_preds = triad_cache.get(cache_key)

        recommendations.append(
            {
                "micro_market": row.to_dict(),
                "listing": listing,
                "triad_predictions": triad_preds,
                "engine_recommendation": None,
                "engine_error": engine_error,
                "engine_context": {
                    "candidate_source": "micro_market",
                    "engine_error": engine_error,
                },
            }
        )

    return recommendations


def generate_engine_recommendations(
    *,
    zip_code: str,
    latitude: Optional[float],
    longitude: Optional[float],
    subdivision: Optional[str],
    property_type_label: str,
    historical_data: Optional[List[Dict[str, Any]]] = None,
    top_n: int = 3,
    use_market_insights: bool = False,
) -> List[Dict[str, Any]]:
    engine = get_recommendation_engine()
    default_lot_acres = 0.25
    lot_features: Dict[str, Any] = {
        "zip_code": zip_code,
        "latitude": latitude,
        "longitude": longitude,
        "subdivision": subdivision,
        "lot_size_acres": default_lot_acres,
        "lot_size_sqft": default_lot_acres * 43560,
        "lot_condition": "flat",
        "utilities_status": "all_utilities",
    }

    property_type = PROPERTY_TYPE_SELECT_TO_ENGINE.get(property_type_label, "Single Family Home")

    try:
        result = engine.generate_recommendations(
            lot_features=lot_features,
            property_type=property_type,
            candidate_configs=None,
            historical_data=historical_data,
            current_listings_context=None,
            top_n=top_n,
            use_market_insights=use_market_insights,
        )
        return result.get("recommendations", [])
    except Exception as exc:
        st.warning(f"Builder recommendation engine unavailable: {exc}")
        return []


def get_high_end_comparables(
    *,
    zip_code: str,
    property_type_label: str,
    subdivision: Optional[str],
    triad_cache: Dict[str, Any],
    target_lat: Optional[float],
    target_lon: Optional[float],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    df = load_seasonality_report()
    if df.empty:
        return []

    subset = df[df["zip_code"].astype(str) == str(zip_code)].copy()
    if subset.empty:
        return []

    prop_norm = PROPERTY_TYPE_SELECT_TO_NORM.get(property_type_label)
    if prop_norm and "property_type_norm" in subset.columns:
        prop_subset = subset[subset["property_type_norm"] == prop_norm]
        if not prop_subset.empty:
            subset = prop_subset

    subset = subset.dropna(subset=["sale_price"])
    if subset.empty:
        return []

    if subdivision:
        sub_subset = subset[
            subset["subdivision"].fillna("").str.lower() == subdivision.lower()
        ]
        if not sub_subset.empty:
            subset = sub_subset

    if target_lat is not None and target_lon is not None:
        subset["distance"] = np.sqrt(
            (subset["latitude"] - target_lat) ** 2 + (subset["longitude"] - target_lon) ** 2
        )
        subset = subset.sort_values(["distance", "sale_price"], ascending=[True, False])
    else:
        subset["distance"] = np.nan
        subset = subset.sort_values("sale_price", ascending=False)

    subset = subset.head(top_n)
    if subset.empty:
        return []

    recommendations: List[Dict[str, Any]] = []
    for _, row in subset.iterrows():
        listing = row.to_dict()
        cache_key = "|".join(
            [
                str(zip_code),
                str(listing.get("property_id") or listing.get("propertyId") or listing.get("id") or ""),
                str(listing.get("latitude") or ""),
                str(listing.get("longitude") or ""),
            ]
        )
        if cache_key not in triad_cache:
            triad_cache[cache_key] = compute_tri_model_predictions(zip_code, listing)
        triad_preds = triad_cache.get(cache_key)
        recommendations.append(
            {
                "listing": listing,
                "triad_predictions": triad_preds,
                "distance": listing.get("distance"),
            }
        )

    return recommendations


def select_optimal_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Select the candidate that maximises predicted price while controlling DOM."""
    best_entry: Optional[Dict[str, Any]] = None
    best_score: float = float("-inf")

    for entry in candidates:
        listing = entry.get("listing") or {}
        triad = entry.get("triad_predictions") or {}

        price = safe_float(triad.get("predicted_price")) or safe_float(listing.get("sale_price"))
        dom = safe_float(triad.get("expected_dom")) or safe_float(
            listing.get("dom_zip_median") or listing.get("days_from_list_to_pending")
        )
        probability = safe_float(triad.get("sell_probability")) or safe_float(
            listing.get("fast_seller_probability")
        )

        if price is None or dom is None or dom <= 0:
            continue

        score = price / max(dom, 1.0)
        if probability is not None:
            score *= (0.5 + probability / 2.0)

        if score > best_score:
            best_score = score
            best_entry = entry

    return best_entry


@st.cache_data
def get_market_features(
    lat: float,
    lon: float,
    zip_code: Optional[str] = None,
    subdivision: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    df = load_seasonality_report()
    if df.empty or lat is None or lon is None:
        return None

    candidate_df = df.copy()
    if subdivision:
        candidates = candidate_df[
            candidate_df["subdivision"].fillna("").str.lower() == subdivision.lower()
        ]
        if not candidates.empty:
            candidate_df = candidates
    if zip_code:
        candidates = candidate_df[
            candidate_df["zip_code"].astype(str) == str(zip_code)
        ]
        if not candidates.empty:
            candidate_df = candidates

    if candidate_df.empty:
        return None

    distances = np.sqrt(
        (candidate_df["latitude"] - lat) ** 2 +
        (candidate_df["longitude"] - lon) ** 2
    )
    if distances.empty:
        return None
    nearest_idx = distances.idxmin()
    nearest_distance = distances.loc[nearest_idx]
    if not np.isfinite(nearest_distance) or nearest_distance > 0.02:
        return None
    return candidate_df.loc[nearest_idx].to_dict()


@st.cache_resource(show_spinner=False)
def load_tri_models() -> tuple[PricingModel, DemandModel]:
    pricing = PricingModel(model_dir="models/triad_latest")
    pricing.load(model_name="triad_pricing_model")
    demand = DemandModel(model_dir="models/triad_latest")
    demand.load(model_name="triad_demand_model")
    return pricing, demand


@st.cache_resource(show_spinner=False)
def get_feature_engineer() -> FeatureEngineer:
    return FeatureEngineer()


@st.cache_resource(show_spinner=False)
def get_recommendation_engine() -> RecommendationEngine:
    return RecommendationEngine(
        min_sell_probability=0.45,
        max_dom=120,
        min_margin_pct=0.0,
        sga_allocation=0.10,
    )


@st.cache_data(show_spinner=False, ttl=600)
def load_listings_for_zip(zip_code: str):
    return fetch_sold_listings_with_features(
        zip_codes=[str(zip_code)],
        days_back=730,
        max_per_zip=None,
        use_cache=True,
    )


@st.cache_data(show_spinner=False, ttl=600)
def get_listing_lookup(zip_code: str) -> Dict[str, Dict[str, Any]]:
    listings = load_listings_for_zip(str(zip_code))
    lookup: Dict[str, Dict[str, Any]] = {}
    for listing in listings or []:
        pid = (
            listing.get("property_id")
            or (listing.get("summary") or {}).get("propertyId")
            or (listing.get("summary") or {}).get("id")
        )
        if pid is not None:
            lookup[str(pid)] = listing
    return lookup


def _extract_sale_price(listing: Dict[str, Any]) -> Optional[float]:
    """Pull the best-available sold price for a listing."""
    summary = listing.get("summary") or {}
    candidates = [
        listing.get("sale_price"),
        listing.get("price"),
        summary.get("mlsSoldPrice"),
        summary.get("lastSaleAmount"),
        summary.get("mlsListingPrice"),
        summary.get("listingAmount"),
        summary.get("price"),
    ]
    for candidate in candidates:
        value = safe_float(candidate)
        if value is not None and value > 0:
            return value

    for history in listing.get("priceHistory") or []:
        event = (history or {}).get("event")
        price = safe_float((history or {}).get("price"))
        if event and str(event).lower() == "sold" and price is not None and price > 0:
            return price

    return None


@st.cache_data(show_spinner=False, ttl=600)
def get_feature_dataset(zip_code: str) -> pd.DataFrame:
    listings = load_listings_for_zip(str(zip_code))
    rows: list[Dict[str, Any]] = []
    for listing in listings or []:
        flags = tag_listing_features(listing)
        summary = listing.get("summary") or {}
        property_detail = listing.get("property_detail_raw") or {}

        sale_price = _extract_sale_price(listing)
        dom_pending = listing.get("dom_to_pending")
        if dom_pending is None:
            dom_pending = (listing.get("timeline") or {}).get("dom_to_pending")

        beds = safe_float(
            listing.get("beds")
            or summary.get("beds")
            or summary.get("bedrooms")
            or property_detail.get("beds")
            or property_detail.get("bedrooms")
        )
        baths = safe_float(
            listing.get("baths")
            or listing.get("bathrooms")
            or summary.get("bathstotal")
            or summary.get("bathsTotal")
            or property_detail.get("bathstotal")
        )
        sqft = safe_float(
            listing.get("sqft")
            or listing.get("square_feet")
            or summary.get("universalsize")
            or summary.get("squareFeet")
            or property_detail.get("universalsize")
            or property_detail.get("squareFeet")
        )
        year_built = safe_float(
            listing.get("year_built")
            or listing.get("yearBuilt")
            or summary.get("yearBuilt")
            or property_detail.get("yearBuilt")
        )
        lot_sqft = safe_float(
            listing.get("lot_sqft")
            or listing.get("lotSize")
            or listing.get("lotSizeSquareFeet")
            or summary.get("lotSize")
            or summary.get("lotSizeSquareFeet")
            or property_detail.get("lotSizeSquareFeet")
        )

        row: Dict[str, Any] = {
            "sale_price": sale_price,
            "dom_to_pending": safe_float(dom_pending),
            "beds": beds,
            "baths": baths,
            "sqft": sqft,
            "year_built": year_built,
            "lot_sqft": lot_sqft,
        }
        if sale_price is not None and sqft:
            row["price_per_sqft"] = sale_price / sqft if sqft > 0 else None
        row.update(flags)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


@st.cache_data(show_spinner=False, ttl=600)
def get_feature_impacts(zip_code: str) -> Dict[str, Dict[str, Any]]:
    cache_path = Path("data/cache/feature_impacts")
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"{zip_code}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            cache_file.unlink(missing_ok=True)

    df = get_feature_dataset(str(zip_code))
    if df.empty:
        _update_feature_coverage(cache_path, str(zip_code), 0, {})
        return {}

    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
    df = df[df["sale_price"] > 0].copy()
    if df.empty:
        _update_feature_coverage(cache_path, str(zip_code), 0, {})
        cache_file.write_text(json.dumps({}, indent=2, sort_keys=True))
        return {}

    df["dom_to_pending"] = pd.to_numeric(df["dom_to_pending"], errors="coerce")
    # Price band segmentation
    if df["sale_price"].nunique(dropna=True) >= 3:
        try:
            df["price_band"] = pd.qcut(
                df["sale_price"],
                q=3,
                labels=["low", "mid", "high"],
                duplicates="drop",
            )
        except Exception:
            try:
                df["price_band"] = pd.cut(
                    df["sale_price"],
                    bins=3,
                    labels=["low", "mid", "high"],
                )
            except Exception:
                df["price_band"] = None
    else:
        df["price_band"] = None

    impacts: Dict[str, Dict[str, Any]] = {}
    feature_cols = [key for key in FEATURE_LIBRARY if key in df.columns]
    control_candidates = ["sqft", "beds", "baths", "year_built", "lot_sqft"]
    control_cols = [
        col
        for col in control_candidates
        if col in df.columns
        and pd.to_numeric(df[col], errors="coerce").notna().sum() >= FEATURE_IMPACT_MIN_SAMPLES
    ]

    def _ridge_uplift(
        data: pd.DataFrame,
        target_col: str,
        *,
        transform: Optional[Callable[[pd.Series], pd.Series]] = None,
        inverse: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Optional[Dict[str, Any]]:
        if data.empty:
            return None
        cols = [target_col] + control_cols + feature_cols
        if any(col not in data.columns for col in cols):
            return None

        model_df = data[cols].copy()
        model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
        model_df = model_df.dropna(subset=[target_col])
        if model_df.empty:
            return None

        for col in control_cols:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
            median_val = model_df[col].median(skipna=True)
            model_df[col] = model_df[col].fillna(median_val if pd.notna(median_val) else 0.0)
        for col in feature_cols:
            model_df[col] = model_df[col].fillna(0.0)

        X = model_df[control_cols + feature_cols].astype(float)
        y = model_df[target_col].astype(float)

        if transform is not None:
            try:
                y = transform(y)
            except Exception:
                return None

        min_rows = max(FEATURE_IMPACT_MIN_SAMPLES * 2, len(control_cols) + 2)
        if len(model_df) < min_rows:
            return None

        try:
            model = RidgeCV(alphas=np.logspace(-3, 3, 25))
            model.fit(X, y)
        except Exception:
            return None

        lifts: Dict[str, float] = {}
        pct_lifts: Dict[str, float] = {}

        for feature in feature_cols:
            mask = X[feature] > 0.5
            sample_count = int(mask.sum())
            if sample_count < FEATURE_IMPACT_MIN_SAMPLES:
                continue

            X_on = X.loc[mask]
            if X_on.empty:
                continue
            X_off = X_on.copy()
            X_off[feature] = 0.0

            preds_on = model.predict(X_on)
            preds_off = model.predict(X_off)

            if inverse is not None:
                try:
                    preds_on = inverse(preds_on)
                    preds_off = inverse(preds_off)
                except Exception:
                    continue

            diff = preds_on - preds_off
            if not len(diff):
                continue

            lifts[feature] = float(np.median(diff))

            if inverse is not None and np.all(preds_off > 0):
                pct_lifts[feature] = float(np.median(diff / preds_off) * 100.0)

        if not lifts:
            return None

        return {"lifts": lifts, "pct_lifts": pct_lifts}

    use_model = (
        RidgeCV is not None
        and bool(feature_cols)
        and bool(control_cols)
        and len(df) >= max(FEATURE_IMPACT_MIN_SAMPLES * 2, len(control_cols) + 2)
    )

    if use_model:
        df_model = df[["sale_price", "dom_to_pending"] + control_cols + feature_cols].copy()
        df_model["price_band"] = df.get("price_band")

        price_results = _ridge_uplift(
            df_model,
            "sale_price",
            transform=np.log,
            inverse=np.exp,
        )
        dom_results = None
        dom_subset = df_model.dropna(subset=["dom_to_pending"])
        if not dom_subset.empty:
            dom_results = _ridge_uplift(dom_subset, "dom_to_pending")

        band_price_results: Dict[str, Dict[str, Any]] = {}
        band_dom_results: Dict[str, Dict[str, Any]] = {}
        if "price_band" in df_model.columns:
            for band in sorted(df_model["price_band"].dropna().unique()):
                band_mask = df_model["price_band"] == band
                if band_mask.sum() < max(FEATURE_IMPACT_MIN_SAMPLES * 2, len(control_cols) + 2):
                    continue
                band_data = df_model.loc[band_mask]
                price_band_result = _ridge_uplift(
                    band_data,
                    "sale_price",
                    transform=np.log,
                    inverse=np.exp,
                )
                if price_band_result:
                    band_price_results[str(band)] = price_band_result

                dom_band_subset = band_data.dropna(subset=["dom_to_pending"])
                if not dom_band_subset.empty:
                    dom_band_result = _ridge_uplift(dom_band_subset, "dom_to_pending")
                    if dom_band_result:
                        band_dom_results[str(band)] = dom_band_result
    else:
        price_results = dom_results = None
        band_price_results = {}
        band_dom_results = {}

    base_price = float(df["sale_price"].median(skipna=True)) if not df["sale_price"].dropna().empty else None
    base_dom = float(df["dom_to_pending"].median(skipna=True)) if not df["dom_to_pending"].dropna().empty else None

    for feature_key, meta in FEATURE_LIBRARY.items():
        if feature_key not in df.columns:
            continue
        feature_df = df[df[feature_key] == 1]
        sample_count = int(feature_df.shape[0])
        if sample_count < FEATURE_IMPACT_MIN_SAMPLES:
            continue

        fallback_price_lift = None
        fallback_dom_delta = None
        feature_price = feature_df["sale_price"].median(skipna=True)
        feature_dom = feature_df["dom_to_pending"].median(skipna=True)
        if base_price is not None and pd.notna(feature_price):
            fallback_price_lift = float(feature_price - base_price)
        if base_dom is not None and pd.notna(feature_dom):
            fallback_dom_delta = float(feature_dom - base_dom)

        price_lift = fallback_price_lift
        price_lift_pct = None
        dom_delta = fallback_dom_delta
        price_method = "median_delta"
        dom_method = "median_delta"

        if price_results:
            lifts = price_results.get("lifts", {})
            pct_lifts = price_results.get("pct_lifts", {})
            if lifts and feature_key in lifts:
                price_lift = lifts[feature_key]
                price_lift_pct = pct_lifts.get(feature_key)
                price_method = "ridge_regression"

        if dom_results:
            dom_lifts = dom_results.get("lifts", {})
            if dom_lifts and feature_key in dom_lifts:
                dom_delta = dom_lifts[feature_key]
                dom_method = "ridge_regression"

        entry: Dict[str, Any] = {
            "label": meta.get("label", feature_key.replace("_", " ").title()),
            "count": sample_count,
            "price_lift": price_lift,
            "price_lift_pct": price_lift_pct,
            "dom_delta": dom_delta,
            "price_method": price_method,
            "dom_method": dom_method,
        }

        if band_price_results:
            band_lifts = {}
            for band_key, result in band_price_results.items():
                lifts = result.get("lifts") if result else None
                if lifts and feature_key in lifts:
                    band_lifts[band_key] = lifts[feature_key]
            if band_lifts:
                entry["price_lift_bands"] = band_lifts

        if band_dom_results:
            band_dom = {}
            for band_key, result in band_dom_results.items():
                lifts = result.get("lifts") if result else None
                if lifts and feature_key in lifts:
                    band_dom[band_key] = lifts[feature_key]
            if band_dom:
                entry["dom_delta_bands"] = band_dom

        impacts[feature_key] = entry

    if impacts:
        cache_file.write_text(json.dumps(impacts, indent=2, sort_keys=True))
    else:
        cache_file.write_text(json.dumps({}, indent=2, sort_keys=True))

    _update_feature_coverage(cache_path, str(zip_code), int(df.shape[0]), impacts)
    return impacts


def _extract_listing_coords(listing: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    lat = listing.get("latitude") or (listing.get("summary") or {}).get("latitude")
    lon = listing.get("longitude") or (listing.get("summary") or {}).get("longitude")
    if lat is None:
        lat = (listing.get("geocode") or {}).get("lat")
    if lon is None:
        geocode = listing.get("geocode") or {}
        lon = geocode.get("lon") or geocode.get("lng")
    return safe_float(lat), safe_float(lon)


def compute_tri_model_predictions(zip_code: str, primary_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not zip_code:
        return None

    try:
        listings = load_listings_for_zip(str(zip_code))
    except Exception as exc:
        st.warning(f"Triad model lookup failed for ZIP {zip_code}: {exc}")
        return None

    if not listings:
        return None

    target = None
    str_pid = None
    if primary_row is not None:
        pid_value = (
            primary_row.get("property_id")
            or primary_row.get("propertyId")
            or primary_row.get("id")
        )
        if pid_value is not None:
            str_pid = str(pid_value)

    if str_pid:
        for listing in listings:
            candidate_id = (
                listing.get("property_id")
                or (listing.get("summary") or {}).get("propertyId")
                or (listing.get("summary") or {}).get("id")
            )
            if candidate_id is not None and str(candidate_id) == str_pid:
                target = listing
                break

    if target is None and primary_row is not None:
        primary_lat = safe_float(primary_row.get("latitude"))
        primary_lon = safe_float(primary_row.get("longitude"))
        if primary_lat is not None and primary_lon is not None:
            best_listing = None
            best_distance = None
            for listing in listings:
                lat, lon = _extract_listing_coords(listing)
                if lat is None or lon is None:
                    continue
                distance = float(np.hypot(lat - primary_lat, lon - primary_lon))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_listing = listing
            if best_listing is not None:
                target = best_listing

    if target is None:
        target = listings[0]

    pricing_model, demand_model = load_tri_models()
    engineer = get_feature_engineer()

    try:
        features_df = engineer.engineer_features([target])
    except Exception as exc:
        st.warning(f"Feature engineering failed for ZIP {zip_code}: {exc}")
        return None

    pricing_features = features_df.reindex(columns=pricing_model.feature_names, fill_value=0.0)
    pricing_features = pricing_features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    demand_features = features_df.reindex(columns=demand_model.feature_names, fill_value=0.0)
    demand_features = demand_features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # ensure DOM-related columns stay meaningful
    for dom_col in ["dom_to_pending", "dom_to_sold"]:
        if dom_col in features_df.columns:
            demand_features[dom_col] = features_df[dom_col]

    try:
        price_pred, price_lower, price_upper = pricing_model.predict(
            pricing_features,
            return_intervals=True,
        )
        demand_preds = demand_model.predict(demand_features)
    except Exception as exc:
        st.warning(f"Triad model prediction failed: {exc}")
        return None

    return {
        "predicted_price": float(price_pred[0]),
        "price_lower": float(price_lower[0]),
        "price_upper": float(price_upper[0]),
        "sell_probability": float(np.clip(demand_preds["sell_probability"][0], 0.0, 1.0)),
        "expected_dom": float(max(demand_preds["expected_dom"][0], 0.0)),
        "feature_snapshot": features_df.to_dict(orient="records")[0],
    }


DEMO_LOCATIONS = [
    {
        "label": "New Irving Park • Greensboro 27408 (High-end SFR)",
        "zip_code": "27408",
        "latitude": 36.1010667665,
        "longitude": -79.8401642095,
        "subdivision": "New Irving Park",
        "notes": "4BR/3BA executive homes in New Irving Park. Fast-sale prob ≈94%, DOM ≈40 days.",
    },
    {
        "label": "Wyngate Village • Winston-Salem 27103 (Move-up townhome)",
        "zip_code": "27103",
        "latitude": 36.0642284927,
        "longitude": -80.3442158592,
        "subdivision": "Wyngate Village",
        "notes": "3BR townhomes around Hanes Mall. Fast-sale prob ≈86%, DOM ≈40 days.",
    },
    {
        "label": "Dogwood Acres • Asheboro 27205 (Entry-level SFR)",
        "zip_code": "27205",
        "latitude": 35.7200937986,
        "longitude": -79.8331894687,
        "subdivision": "Dogwood Acres",
        "notes": "3BR starter homes. Fast-sale prob ≈88%, DOM ≈47 days, minimal relists.",
    },
]

# Page config
st.set_page_config(
    page_title="Real Estate Intelligence Platform",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .score-high {
        color: #28a745;
        font-weight: bold;
    }
    .score-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .score-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'submarkets' not in st.session_state:
    st.session_state.submarkets = None
if 'land_listings' not in st.session_state:
    st.session_state.land_listings = None
if 'products' not in st.session_state:
    st.session_state.products = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def get_score_color_class(score):
    """Get CSS class for score coloring."""
    if score >= 0.7:
        return "score-high"
    elif score >= 0.5:
        return "score-medium"
    else:
        return "score-low"


def format_currency(value):
    """Format value as currency."""
    return f"${value:,.0f}"


def safe_float(value) -> Optional[float]:
    """Convert to float when possible, returning None for invalid numbers."""
    if value is None:
        return None
    try:
        result = float(value)
        if np.isnan(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def main():
    """Main application."""
    
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown('<div class="main-header">🏗️ BuildOptima</div>', unsafe_allow_html=True)
    st.markdown("*What to build, where to build it, and how fast it will move*")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏘️ RE Intel")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📊 Market Analysis", "🎯 Micro-Market Analysis",
             "🏞️ Land Opportunities", "🏗️ Product Intelligence", 
             "💰 Financial Modeling", "🧠 ML Recommendations", "🔥 Listing Popularity", "🤖 AI Assistant"]
        )
        
        st.markdown("---")
        st.markdown("### Settings")
        st.markdown(f"**Target State:** {settings.target_state}")
        st.markdown(f"**Counties:** {', '.join(settings.get_target_counties_list())}")
        
        st.markdown("---")
        st.markdown("### Analysis Weights")
        st.progress(settings.school_weight, text=f"Schools: {settings.school_weight*100:.0f}%")
        st.progress(settings.crime_weight, text=f"Crime: {settings.crime_weight*100:.0f}%")
        st.progress(settings.growth_weight, text=f"Growth: {settings.growth_weight*100:.0f}%")
        st.progress(settings.price_weight, text=f"Pricing: {settings.price_weight*100:.0f}%")
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Market Analysis":
        show_market_analysis()
    elif page == "🎯 Micro-Market Analysis":
        show_micro_market_analysis()
    elif page == "🏞️ Land Opportunities":
        show_land_opportunities()
    elif page == "🏗️ Product Intelligence":
        show_product_intelligence()
    elif page == "💰 Financial Modeling":
        show_financial_modeling()
    elif page == "🧠 ML Recommendations":
        show_ml_recommendations()
    elif page == "🔥 Listing Popularity":
        show_listing_popularity()
    elif page == "🤖 AI Assistant":
        show_ai_assistant()


def show_dashboard():
    """Show main dashboard."""
    st.header("Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Submarkets Analyzed", "15", "↑ 3 this week")
    with col2:
        st.metric("Land Opportunities", "47", "↑ 8 new")
    with col3:
        st.metric("Avg. Opportunity Score", "0.72", "↑ 0.05")
    with col4:
        st.metric("Projected Avg. IRR", "18.5%", "↑ 2.3%")
    
    st.markdown("---")
    
    # Quick insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 5 Submarkets")
        
        # Mock data for demo
        top_markets = pd.DataFrame({
            "City": ["Cary", "Apex", "Holly Springs", "Morrisville", "Wake Forest"],
            "County": ["Wake", "Wake", "Wake", "Wake", "Wake"],
            "Score": [0.85, 0.82, 0.79, 0.76, 0.74],
            "Price/SqFt": [185, 178, 172, 180, 165]
        })
        
        fig = px.bar(top_markets, x="Score", y="City", orientation='h',
                    color="Score", color_continuous_scale="blues",
                    title="Composite Scores")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Land Price Distribution")
        
        # Mock data
        price_data = pd.DataFrame({
            "Price Range": ["$50k-$75k", "$75k-$100k", "$100k-$150k", "$150k+"],
            "Count": [12, 18, 14, 3]
        })
        
        fig = px.pie(price_data, values="Count", names="Price Range",
                    title="Available Land by Price")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    activity_data = pd.DataFrame({
        "Date": ["2025-10-31", "2025-10-30", "2025-10-29", "2025-10-28"],
        "Activity": [
            "New land listing in Apex - $82k",
            "Submarket analysis updated: Cary",
            "8 new listings scraped from Zillow",
            "Product optimization completed: Holly Springs"
        ],
        "Type": ["Land", "Analysis", "Data", "Analysis"]
    })
    
    st.dataframe(activity_data, use_container_width=True, hide_index=True)


def show_market_analysis():
    """Show market analysis page."""
    st.header("📊 Submarket Analysis")
    
    st.markdown("Analyze submarkets based on schools, crime, growth, and pricing.")
    
    # Input section
    with st.expander("🔍 Analyze New Submarkets", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cities_input = st.text_area(
                "Cities (one per line)",
                "Cary\nApex\nHolly Springs\nMorrisville\nWake Forest",
                height=150
            )
        
        with col2:
            counties_input = st.text_area(
                "Counties (one per line)",
                "Wake\nWake\nWake\nWake\nWake",
                height=150
            )
        
        with col3:
            zip_codes_input = st.text_area(
                "Zip Codes (optional, one per line)",
                "27519\n27502\n27540\n27560\n27587",
                height=150
            )
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Analyzing submarkets..."):
                # Parse inputs
                cities = [c.strip() for c in cities_input.split('\n') if c.strip()]
                counties = [c.strip() for c in counties_input.split('\n') if c.strip()]
                zip_codes = [z.strip() for z in zip_codes_input.split('\n') if z.strip()]
                
                # Pad zip codes if needed
                while len(zip_codes) < len(cities):
                    zip_codes.append(None)
                
                # Build locations list
                locations = []
                for i, city in enumerate(cities):
                    locations.append({
                        "city": city,
                        "county": counties[i] if i < len(counties) else "Wake",
                        "zip_code": zip_codes[i] if i < len(zip_codes) else None,
                        "state": "NC"
                    })
                
                # Run analysis
                ranker = SubmarketRanker()
                st.session_state.submarkets = ranker.rank_submarkets(locations)
                
                st.success(f"✅ Analyzed {len(st.session_state.submarkets)} submarkets!")
    
    # Display results
    if st.session_state.submarkets:
        st.markdown("---")
        st.subheader("📈 Results")
        
        # Summary table
        summary_data = []
        for sm in st.session_state.submarkets:
            summary_data.append({
                "Rank": len(summary_data) + 1,
                "City": sm.city,
                "County": sm.county,
                "Composite Score": f"{sm.composite_score:.3f}",
                "Schools": f"{sm.school_score:.3f}",
                "Crime": f"{sm.crime_score:.3f}",
                "Growth": f"{sm.growth_score:.3f}",
                "Price": f"{sm.price_score:.3f}"
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Detailed view
        st.markdown("---")
        st.subheader("🔍 Detailed Analysis")
        
        selected_city = st.selectbox(
            "Select city for details:",
            [sm.city for sm in st.session_state.submarkets]
        )
        
        selected = next((sm for sm in st.session_state.submarkets if sm.city == selected_city), None)
        
        if selected:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Composite Score", f"{selected.composite_score:.3f}")
            with col2:
                st.metric("School Score", f"{selected.school_score:.3f}")
            with col3:
                st.metric("Crime Score", f"{selected.crime_score:.3f}")
            with col4:
                st.metric("Growth Score", f"{selected.growth_score:.3f}")
            
            # Score breakdown chart
            scores = {
                "Schools": selected.school_score,
                "Crime": selected.crime_score,
                "Growth": selected.growth_score,
                "Price": selected.price_score
            }
            
            fig = go.Figure(data=[
                go.Scatterpolar(
                    r=list(scores.values()),
                    theta=list(scores.keys()),
                    fill='toself'
                )
            ])
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1])),
                title=f"Score Breakdown: {selected.city}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional details
            if selected.schools:
                st.markdown("**🏫 Schools:**")
                for school in selected.schools[:3]:
                    st.markdown(f"- {school.name}: Rating {school.rating}/10")
            
            if selected.pricing_data:
                st.markdown(f"**💰 Median Price/SqFt:** ${selected.pricing_data.median_price_per_sqft:.2f}")


def show_land_opportunities():
    """Show land opportunities page."""
    st.header("🏞️ Land Acquisition")
    
    st.markdown("Discover and analyze land opportunities across target markets.")
    
    # Scraping controls
    with st.expander("🔍 Search for Land", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            search_cities = st.multiselect(
                "Cities",
                ["Cary", "Apex", "Holly Springs", "Morrisville", "Wake Forest", "Durham", "Chapel Hill"],
                default=["Cary", "Apex"]
            )
        
        with col2:
            max_price = st.number_input("Max Price", min_value=0, value=0.000, step=10000)
        
        if st.button("🔎 Search Land Listings", type="primary"):
            with st.spinner("Searching land listings..."):
                scraper = LandScraperOrchestrator()
                
                locations = [{"city": city, "state": "NC", "max_price": max_price} for city in search_cities]
                st.session_state.land_listings = scraper.scrape_multiple_locations(locations)
                
                st.success(f"✅ Found {len(st.session_state.land_listings)} listings!")
    
    # Display listings
    if st.session_state.land_listings:
        st.markdown("---")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_city = st.multiselect(
                "Filter by City",
                list(set([l.city for l in st.session_state.land_listings]))
            )
        
        with col2:
            filter_zoning = st.multiselect(
                "Filter by Zoning",
                ["residential", "commercial", "agricultural", "mixed_use"]
            )
        
        with col3:
            sort_by = st.selectbox("Sort By", ["Price (Low to High)", "Price (High to Low)", "Acreage"])
        
        # Apply filters
        filtered = st.session_state.land_listings
        if filter_city:
            filtered = [l for l in filtered if l.city in filter_city]
        if filter_zoning:
            filtered = [l for l in filtered if l.zoning.value in filter_zoning]
        
        # Sort
        if sort_by == "Price (Low to High)":
            filtered.sort(key=lambda x: x.price)
        elif sort_by == "Price (High to Low)":
            filtered.sort(key=lambda x: x.price, reverse=True)
        elif sort_by == "Acreage":
            filtered.sort(key=lambda x: x.acreage or 0, reverse=True)
        
        # Display
        st.subheader(f"📋 {len(filtered)} Listings")
        
        for listing in filtered[:10]:  # Show top 10
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    st.markdown(f"**{listing.address or 'Address Not Listed'}**")
                    st.caption(f"{listing.city}, {listing.county} | {listing.zoning.value}")
                
                with col2:
                    st.metric("Price", format_currency(listing.price))
                
                with col3:
                    st.metric("Acreage", f"{listing.acreage or 'N/A'}")
                
                with col4:
                    if st.button("View Details", key=listing.listing_id):
                        st.write(f"Source: {listing.source}")
                        st.write(f"URL: {listing.url}")
                
                st.markdown("---")


def show_product_intelligence():
    """Show product optimization page."""
    st.header("🏗️ Product Intelligence")
    
    st.markdown("Determine optimal house configurations based on recent sales data.")
    
    # Input
    with st.expander("🔍 Analyze Product Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            product_city = st.selectbox("City", ["Cary", "Apex", "Holly Springs", "Morrisville"])
        
        with col2:
            product_county = st.text_input("County", "Wake")
        
        if st.button("📊 Analyze", type="primary"):
            with st.spinner("Analyzing recent sales..."):
                analyzer = ProductOptimizationAnalyzer()
                product = analyzer.analyze_submarket(product_city, product_county)
                
                if 'products' not in st.session_state or st.session_state.products is None:
                    st.session_state.products = []
                st.session_state.products.append(product)
                
                st.success("✅ Analysis complete!")
    
    # Display
    if st.session_state.products:
        product = st.session_state.products[-1]  # Latest
        
        st.markdown("---")
        st.subheader(f"📈 Optimal Configuration: {product.city}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Square Feet", f"{product.optimal_sqft_min}-{product.optimal_sqft_max}")
        with col2:
            st.metric("Bedrooms", product.optimal_bedrooms)
        with col3:
            st.metric("Bathrooms", product.optimal_bathrooms)
        with col4:
            st.metric("Avg Days on Market", f"{product.avg_days_on_market:.0f}")
        
        # Features
        if product.recommended_features:
            st.markdown("### 🌟 Recommended Features")
            
            feature_df = pd.DataFrame(product.recommended_features)
            feature_df['Frequency %'] = (feature_df['frequency'] * 100).round(1)
            
            fig = px.bar(feature_df[:8], x='Frequency %', y='feature', orientation='h',
                        title="Feature Popularity")
            st.plotly_chart(fig, use_container_width=True)
        
        # Incentives
        if product.effective_incentives:
            st.markdown("### 💡 Effective Incentives")
            
            for incentive in product.effective_incentives:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{incentive['incentive']}**")
                with col2:
                    st.metric("DOM Reduction", f"{incentive['days_on_market_reduction']} days")


def show_micro_market_analysis():
    """Show micro-market analysis with subdivision filtering."""
    st.header("🎯 Micro-Market Analysis")
    
    st.markdown("""
    **Hyper-local intelligence:** Analyze specific subdivisions and neighborhoods within a ZIP code.  
    Same ZIP code, different recommendations!
    """)
    
    # Tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["📍 Subdivision Analysis", "🔄 Compare Subdivisions", "📏 Radius Search"])
    
    # TAB 1: Subdivision Analysis
    with tab1:
        st.subheader("Analyze Specific Subdivision")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            zip_code = st.text_input("ZIP Code", value="27410", key="sub_zip")
        
        with col2:
            months_back = st.selectbox("Historical Period", [6, 12, 24, 36], index=1, key="sub_months")
        
        with col3:
            property_type = st.selectbox(
                "Property Type",
                ["ALL", "Single Family Home", "Townhome", "Condo"],
                index=0,
                key="sub_property_type"
            )
        
        # Get available subdivisions button
        if st.button("🔍 Show Available Subdivisions", key="get_subdivs"):
            with st.spinner("Fetching subdivisions..."):
                try:
                    subdivisions = feature_analyzer.get_subdivisions(zip_code)
                    
                    if subdivisions:
                        st.session_state.subdivisions = subdivisions
                        st.success(f"✅ Found {len(subdivisions)} subdivisions in ZIP {zip_code}")
                    else:
                        st.warning("No subdivisions found in this ZIP code")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Show subdivision selector if available
        if hasattr(st.session_state, 'subdivisions') and st.session_state.subdivisions:
            st.markdown("---")
            
            # Show top subdivisions
            top_subdivs = st.session_state.subdivisions[:20]
            subdiv_df = pd.DataFrame(top_subdivs)
            
            st.markdown("**Top 20 Subdivisions by Property Count:**")
            st.dataframe(subdiv_df, use_container_width=True, hide_index=True)
            
            # Subdivision selector
            subdiv_name = st.selectbox(
                "Select Subdivision to Analyze",
                ["All (No Filter)"] + [s['name'] for s in top_subdivs],
                key="selected_subdiv"
            )
            
            if st.button("🚀 Analyze", type="primary", key="analyze_subdiv"):
                with st.spinner(f"Analyzing {subdiv_name}..."):
                    try:
                        # Determine if filtering by subdivision
                        subdivision_filter = None if subdiv_name == "All (No Filter)" else subdiv_name
                        
                        # Run feature analysis
                        feature_analysis = feature_analyzer.analyze_feature_impact(
                            zip_code,
                            months_back=months_back,
                            min_samples=3,  # Lower threshold to allow more configs
                            subdivision=subdivision_filter,
                            property_type=property_type
                        )
                        
                        # Run demand prediction
                        demand_analysis = demand_predictor.predict_optimal_config(
                            zip_code,
                            months_back=months_back,
                            min_samples=3,  # Lower threshold to allow more configs
                            subdivision=subdivision_filter,
                            property_type=property_type
                        )
                        
                        # Check for errors
                        if 'error' in feature_analysis or 'error' in demand_analysis:
                            st.error(f"Insufficient data for {subdiv_name}. Try selecting a different subdivision or increase the historical period.")
                        elif 'optimal_config' not in demand_analysis or not demand_analysis.get('optimal_config'):
                            # This shouldn't happen, but let's handle it
                            st.warning(f"⚠️ Found {feature_analysis.get('property_count', 0)} properties but couldn't generate recommendation.")
                            with st.expander("Debug Info"):
                                st.json({"demand_keys": list(demand_analysis.keys()), "has_optimal": 'optimal_config' in demand_analysis})
                        else:
                            # Display results
                            st.success("✅ Analysis complete!")
                            
                            # Debug: Show what we got
                            with st.expander("🔍 Debug Info", expanded=False):
                                st.json({"optimal_config_type": type(demand_analysis.get('optimal_config')).__name__, 
                                        "optimal_config_keys": list(demand_analysis.get('optimal_config', {}).keys()) if isinstance(demand_analysis.get('optimal_config'), dict) else "Not a dict"})
                            
                            # Market Overview
                            st.markdown("---")
                            st.subheader("📊 Market Overview")
                            
                            market_stats = feature_analysis['market_stats']
                            optimal = demand_analysis['optimal_config']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Properties Analyzed", feature_analysis['property_count'])
                            with col2:
                                st.metric("Median Price", format_currency(market_stats.get('median_sale_price', 0)))
                            with col3:
                                st.metric("Median Size", f"{market_stats.get('median_size', 0):,.0f} sqft")
                            with col4:
                                st.metric("Price/SqFt", f"${market_stats.get('median_price_per_sqft', 0):.2f}")
                            
                            # Optimal Configuration
                            st.markdown("---")
                            st.subheader("🎯 Optimal Build Recommendation")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Bedrooms", optimal.get('bedrooms', 'N/A'))
                            with col2:
                                st.metric("Bathrooms", optimal.get('bathrooms', 'N/A'))
                            with col3:
                                st.metric("Square Feet", f"{optimal.get('sqft', 0):,}")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Expected Price", format_currency(optimal.get('median_sale_price', 0)))
                            with col2:
                                st.metric("Sales Velocity", f"{optimal.get('sales_velocity', 0):.1f} units/mo")
                            with col3:
                                confidence_pct = optimal.get('confidence', 0) * 100
                                st.metric("Confidence", f"{confidence_pct:.0f}%")
                            
                            st.info(f"**Why:** {optimal.get('rationale', 'Analysis complete')}")
                            
                            # Configuration Performance
                            st.markdown("---")
                            st.subheader("📈 Configuration Performance")
                            
                            config_df = pd.DataFrame(demand_analysis['all_configurations'][:5])
                            
                            if not config_df.empty:
                                fig = px.bar(config_df, 
                                           x='configuration', 
                                           y='sales_velocity',
                                           color='demand_score',
                                           title="Sales Velocity by Configuration",
                                           labels={'sales_velocity': 'Sales per Month', 'configuration': 'Configuration'})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Top Features
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("✨ Top Interior Features")
                                interior = feature_analysis.get('interior_features', [])[:5]
                                if interior:
                                    for feat in interior:
                                        with st.expander(f"{feat['feature']} ({feat['value']})"):
                                            st.write(f"**Priority:** {feat['priority'].title()}")
                                            st.write(feat['rationale'])
                                else:
                                    st.info("No interior features analyzed")
                            
                            with col2:
                                st.subheader("🏠 Top Exterior Features")
                                exterior = feature_analysis.get('exterior_features', [])[:5]
                                if exterior:
                                    for feat in exterior:
                                        with st.expander(f"{feat['feature']} ({feat['value']})"):
                                            st.write(f"**Priority:** {feat['priority'].title()}")
                                            st.write(feat['rationale'])
                                else:
                                    st.info("No exterior features analyzed")
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        st.info("💡 **Tips:** Try a different subdivision or property type, or increase the historical period.")
                        with st.expander("🔍 Technical Details"):
                            import traceback
                            st.code(traceback.format_exc())
    
    # TAB 2: Compare Subdivisions
    with tab2:
        st.subheader("Compare Multiple Subdivisions")
        
        zip_code_compare = st.text_input("ZIP Code", value="27410", key="compare_zip")
        
        if st.button("🔍 Load Subdivisions", key="get_subdivs_compare"):
            with st.spinner("Fetching subdivisions..."):
                try:
                    subdivisions = feature_analyzer.get_subdivisions(zip_code_compare)
                    if subdivisions:
                        st.session_state.subdivisions_compare = subdivisions
                        st.success(f"✅ Found {len(subdivisions)} subdivisions")
                    else:
                        st.warning("No subdivisions found")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if hasattr(st.session_state, 'subdivisions_compare') and st.session_state.subdivisions_compare:
            top_subdivs = st.session_state.subdivisions_compare[:15]
            
            col1, col2 = st.columns(2)
            
            with col1:
                subdiv1 = st.selectbox("First Subdivision", [s['name'] for s in top_subdivs], key="compare1")
            
            with col2:
                subdiv2 = st.selectbox("Second Subdivision", [s['name'] for s in top_subdivs], index=1 if len(top_subdivs) > 1 else 0, key="compare2")
            
            if st.button("⚖️ Compare", type="primary", key="run_compare"):
                with st.spinner("Running comparison..."):
                    try:
                        results = []
                        
                        for subdiv in [subdiv1, subdiv2]:
                            # Analyze each subdivision
                            feature_analysis = feature_analyzer.analyze_feature_impact(
                                zip_code_compare,
                                months_back=24,
                                min_samples=3,
                                subdivision=subdiv,
                                property_type="ALL"  # All property types for comparison
                            )
                            
                            demand_analysis = demand_predictor.predict_optimal_config(
                                zip_code_compare,
                                months_back=24,
                                min_samples=3,
                                subdivision=subdiv,
                                property_type="ALL"  # All property types for comparison
                            )
                            
                            if 'error' not in feature_analysis and 'error' not in demand_analysis:
                                results.append({
                                    'name': subdiv,
                                    'feature': feature_analysis,
                                    'demand': demand_analysis
                                })
                        
                        if len(results) == 2:
                            st.success("✅ Comparison complete!")
                            
                            # Side-by-side comparison
                            col1, col2 = st.columns(2)
                            
                            for i, result in enumerate(results):
                                with (col1 if i == 0 else col2):
                                    st.markdown(f"### {result['name']}")
                                    
                                    stats = result['feature']['market_stats']
                                    optimal = result['demand']['optimal_config']
                                    
                                    st.metric("Properties", result['feature']['property_count'])
                                    st.metric("Median Price", format_currency(stats.get('median_sale_price', 0)))
                                    st.metric("Median Size", f"{stats.get('median_size', 0):,.0f} sqft")
                                    st.metric("Price/SqFt", f"${stats.get('median_price_per_sqft', 0):.2f}")
                                    
                                    st.markdown("**Optimal Build:**")
                                    st.write(f"**{optimal['configuration']}** @ {optimal['sqft']:,} sqft")
                                    st.write(f"Expected: {format_currency(optimal['median_sale_price'])}")
                            
                            # Comparison insights
                            st.markdown("---")
                            st.subheader("💡 Key Differences")
                            
                            price_diff = results[0]['feature']['market_stats'].get('median_sale_price', 0) - results[1]['feature']['market_stats'].get('median_sale_price', 0)
                            size_diff = results[0]['feature']['market_stats'].get('median_size', 0) - results[1]['feature']['market_stats'].get('median_size', 0)
                            
                            st.write(f"**Price Difference:** {format_currency(abs(price_diff))} ({results[0]['name'] if price_diff > 0 else results[1]['name']} is higher)")
                            st.write(f"**Size Difference:** {abs(size_diff):,.0f} sqft ({results[0]['name'] if size_diff > 0 else results[1]['name']} is larger)")
                            
                            st.info("💡 **Insight:** Same ZIP code, but build different products for each subdivision!")
                        
                        else:
                            st.warning("Could not analyze one or both subdivisions. Try different selections.")
                    
                    except Exception as e:
                        st.error(f"Error during comparison: {str(e)}")
    
    # TAB 3: Radius Search
    with tab3:
        st.subheader("Analyze Properties Within Radius")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            zip_code_radius = st.text_input("ZIP Code", value="27410", key="radius_zip")
        
        with col2:
            center_lat = st.number_input("Latitude", value=36.089, format="%.6f", key="radius_lat")
        
        with col3:
            center_lon = st.number_input("Longitude", value=-79.908, format="%.6f", key="radius_lon")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            radius_miles = st.slider("Radius (miles)", 0.25, 2.0, 0.5, 0.25, key="radius_miles")
        
        with col2:
            property_type_radius = st.selectbox(
                "Property Type",
                ["ALL", "Single Family Home", "Townhome", "Condo"],
                index=0,
                key="radius_property_type"
            )
        
        st.info("💡 **Tip:** Use Google Maps to find the latitude/longitude of your lot. Right-click on the location and select 'What's here?'")
        
        if st.button("🎯 Analyze Radius", type="primary", key="analyze_radius"):
            with st.spinner(f"Analyzing properties within {radius_miles} miles..."):
                try:
                    # Run analysis with radius filter (lower threshold for small areas)
                    feature_analysis = feature_analyzer.analyze_feature_impact(
                        zip_code_radius,
                        months_back=24,
                        min_samples=1,  # Lower threshold for radius search
                        radius_miles=radius_miles,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        property_type=property_type_radius
                    )
                    
                    demand_analysis = demand_predictor.predict_optimal_config(
                        zip_code_radius,
                        months_back=24,
                        min_samples=1,  # Lower threshold for radius search
                        radius_miles=radius_miles,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        property_type=property_type_radius
                    )
                    
                    if 'error' in feature_analysis or 'error' in demand_analysis:
                        error_msg = feature_analysis.get('error') or demand_analysis.get('error')
                        st.warning(f"⚠️ Insufficient data within {radius_miles} miles.")
                        st.info(f"**Found:** {feature_analysis.get('property_count', 0)} properties\n\n**Issue:** {error_msg}\n\n💡 Try increasing the radius or choosing a location with more recent sales.")
                    elif 'optimal_config' not in demand_analysis or not demand_analysis['optimal_config']:
                        st.warning(f"⚠️ Found {feature_analysis.get('property_count', 0)} properties but couldn't generate recommendation.")
                        st.info("💡 The properties may not have complete data (beds/baths/size). Try increasing the radius to 1.5 or 2 miles.")
                    else:
                        st.success(f"✅ Found {feature_analysis['property_count']} properties within {radius_miles} miles!")
                        
                        # Display results (similar to Tab 1)
                        market_stats = feature_analysis.get('market_stats', {})
                        optimal = demand_analysis['optimal_config']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Properties", feature_analysis['property_count'])
                        with col2:
                            st.metric("Median Price", format_currency(market_stats.get('median_sale_price', 0)))
                        with col3:
                            st.metric("Median Size", f"{market_stats.get('median_size', 0):,.0f} sqft")
                        with col4:
                            st.metric("Price/SqFt", f"${market_stats.get('median_price_per_sqft', 0):.2f}")
                        
                        st.markdown("---")
                        st.subheader("🎯 Optimal Build for This Location")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Bedrooms", optimal.get('bedrooms', 'N/A'))
                            st.metric("Bathrooms", optimal.get('bathrooms', 'N/A'))
                        with col2:
                            st.metric("Square Feet", f"{optimal.get('sqft', 0):,}")
                            st.metric("Expected Price", format_currency(optimal.get('median_sale_price', 0)))
                        with col3:
                            st.metric("Sales Velocity", f"{optimal.get('sales_velocity', 0):.1f} units/mo")
                            st.metric("Confidence", f"{optimal.get('confidence', 0)*100:.0f}%")
                        
                        st.info(f"**Why:** {optimal.get('rationale', 'Analysis complete')}")
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("💡 **Tips:** Try increasing the radius, or use a different ZIP code with more properties.")


def show_financial_modeling():
    """Show financial modeling page."""
    st.header("💰 Financial Modeling")
    
    st.markdown("Calculate IRR, ROI, and perform sensitivity analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Project Inputs")
        
        land_cost = st.number_input("Land Cost", value=80000, step=5000)
        house_sqft = st.number_input("House Square Feet", value=2000, step=100)
        construction_cost_per_sqft = st.number_input("Construction Cost per SqFt", value=150, step=10)
        timeline_months = st.number_input("Timeline (months)", value=8, step=1)
        sale_price_per_sqft = st.number_input("Sale Price per SqFt", value=180, step=5)
    
    with col2:
        st.subheader("📈 Advanced Settings")
        
        carrying_cost_monthly = st.number_input("Carrying Cost (monthly)", value=500, step=50)
        soft_cost_percentage = st.number_input("Soft Costs %", value=10, step=1) / 100
        discount_rate = st.number_input("Discount Rate %", value=12, step=1) / 100
    
    if st.button("💡 Calculate", type="primary"):
        optimizer = FinancialOptimizer()
        
        construction_cost = house_sqft * construction_cost_per_sqft
        carrying_costs = carrying_cost_monthly * timeline_months
        soft_costs = construction_cost * soft_cost_percentage
        projected_sale = house_sqft * sale_price_per_sqft
        
        financials = optimizer.analyze_project(
            land_cost=land_cost,
            construction_cost=construction_cost,
            carrying_costs=carrying_costs,
            other_costs=soft_costs,
            projected_sale_price=projected_sale,
            timeline_months=timeline_months,
            discount_rate=discount_rate
        )
        
        st.markdown("---")
        st.subheader("📊 Financial Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Investment", format_currency(financials.total_investment))
        with col2:
            st.metric("Projected Sale", format_currency(financials.projected_sale_price))
        with col3:
            st.metric("Gross Profit", format_currency(financials.gross_profit))
        with col4:
            st.metric("Gross Margin", f"{financials.gross_margin:.1f}%")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("IRR (Annual)", f"{financials.irr*100:.2f}%" if financials.irr else "N/A")
        with col2:
            st.metric("ROI", f"{financials.roi:.1f}%")
        with col3:
            st.metric("NPV", format_currency(financials.npv))
        
        # Sensitivity analysis
        st.markdown("---")
        st.subheader("📉 Sensitivity Analysis")
        
        sensitivity = optimizer.sensitivity_analysis(financials)
        
        sens_df = pd.DataFrame(sensitivity)
        fig = px.line(sens_df, x='price_variation', y=['roi', 'irr'],
                     title="ROI & IRR Sensitivity to Sale Price",
                     labels={'value': 'Percentage', 'variable': 'Metric'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(sens_df, use_container_width=True, hide_index=True)


def show_ai_assistant():
    """Show AI chat assistant."""
    st.header("🤖 AI Assistant")
    
    st.markdown("Ask questions about markets, land opportunities, and development strategy in natural language.")
    
    # Initialize RAG if not already done
    if st.session_state.rag_system is None:
        with st.spinner("Initializing AI Assistant..."):
            try:
                st.session_state.rag_system = QdrantRAGSystem()
                st.success("✅ AI Assistant ready!")
            except Exception as e:
                st.error(f"❌ Could not initialize AI Assistant: {e}")
                st.info("Make sure Qdrant is running locally: `docker run -p 6333:6333 qdrant/qdrant`")
                return
    
    # Index data button
    if st.session_state.submarkets or st.session_state.land_listings or st.session_state.products:
        if st.button("📚 Index Data for AI"):
            with st.spinner("Indexing data..."):
                st.session_state.rag_system.bulk_index_data(
                    submarkets=st.session_state.submarkets,
                    land_listings=st.session_state.land_listings,
                    products=st.session_state.products
                )
                st.success("✅ Data indexed!")
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.chat_message("user").write(msg['content'])
        else:
            st.chat_message("assistant").write(msg['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask anything about your real estate data..."):
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        st.chat_message("user").write(prompt)
        
        # Get response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_system.query(prompt)
                answer = response['answer']
                
                # Add assistant message
                st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
                st.chat_message("assistant").write(answer)
                
                # Show sources
                if response.get('sources'):
                    with st.expander("📚 Sources"):
                        for source in response['sources']:
                            st.caption(source)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.chat_history.append({'role': 'assistant', 'content': error_msg})
                st.chat_message("assistant").error(error_msg)


def show_ml_recommendations():
    """Show ML-based build recommendations using seasonality report."""
    st.header("🧠 BuildOptima Recommendation Studio")
    st.caption("Blend current Triad model outputs with historical seasonality context to guide your next spec.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Builder Inputs")

    demo_labels = ["Custom entry"] + [loc["label"] for loc in DEMO_LOCATIONS]
    selected_demo = st.sidebar.selectbox("Hero Locations", demo_labels, index=0, key="preview_demo")

    defaults = {
        "zip": "27410",
        "lat": 36.089,
        "lon": -79.908,
        "subdivision": "",
        "notes": "",
    }
    if selected_demo != "Custom entry":
        chosen = next((loc for loc in DEMO_LOCATIONS if loc["label"] == selected_demo), None)
        if chosen:
            defaults.update(
                {
                    "zip": chosen["zip_code"],
                    "lat": chosen["latitude"],
                    "lon": chosen["longitude"],
                    "subdivision": chosen.get("subdivision") or "",
                    "notes": chosen.get("notes") or "",
                }
            )

    previous_demo = st.session_state.get("_last_selected_demo")
    if selected_demo != "Custom entry" and previous_demo != selected_demo:
        st.session_state["preview_zip"] = defaults["zip"]
        st.session_state["preview_lat"] = defaults["lat"]
        st.session_state["preview_lon"] = defaults["lon"]
        st.session_state["preview_subdivision"] = defaults["subdivision"]
    st.session_state.setdefault("preview_zip", defaults["zip"])
    st.session_state.setdefault("preview_lat", defaults["lat"])
    st.session_state.setdefault("preview_lon", defaults["lon"])
    st.session_state.setdefault("preview_subdivision", defaults["subdivision"])
    st.session_state.setdefault("preview_property_type", "Any")
    st.session_state["_last_selected_demo"] = selected_demo

    with st.sidebar.form("builder_form"):
        zip_code = st.text_input("ZIP Code", key="preview_zip")
        latitude = st.number_input("Latitude", format="%.6f", key="preview_lat")
        longitude = st.number_input("Longitude", format="%.6f", key="preview_lon")
        subdivision = st.text_input("Subdivision (optional)", key="preview_subdivision")
        property_type = st.selectbox(
            "Property Type",
            ["Any", "Single Family", "Townhome", "Condo"],
            key="preview_property_type",
        )
        generate_clicked = st.form_submit_button("Generate builder recommendation", type="primary")

    if defaults.get("notes"):
        st.sidebar.caption(defaults["notes"])

    def match_rows(zip_code_val: str, lat_val: float, lon_val: float, prop_type_val: str, subdivision_val: str) -> pd.DataFrame:
        df = load_seasonality_report()
        if df.empty:
            return df

        candidates = df[df["zip_code"].astype(str) == str(zip_code_val)].copy()
        prop_norm = PROPERTY_TYPE_SELECT_TO_NORM.get(prop_type_val)
        if prop_norm and "property_type_norm" in candidates.columns:
            candidates = candidates[candidates["property_type_norm"] == prop_norm]
        if subdivision_val:
            subset = candidates[candidates["subdivision"].fillna("").str.lower() == subdivision_val.lower()]
            if not subset.empty:
                candidates = subset

        if candidates.empty:
            return candidates

        candidates["distance"] = np.sqrt(
            (candidates["latitude"] - lat_val) ** 2 + (candidates["longitude"] - lon_val) ** 2
        )
        candidates["fast_seller_probability"] = candidates.get("fast_seller_probability", 0.0).fillna(0.0)
        candidates.sort_values(["distance", "fast_seller_probability"], ascending=[True, False], inplace=True)
        return candidates.head(3)

    if generate_clicked:
        matches = match_rows(zip_code, latitude, longitude, property_type, subdivision)
        if matches.empty:
            st.warning("No comparable listings found in the seasonality report. Try another ZIP or adjust coordinates.")
            st.session_state.pop("preview_matches", None)
        else:
            st.session_state.preview_matches = matches.to_dict(orient="records")
            st.session_state.triad_model_cache = {}
            st.toast(f"Loaded {len(matches)} nearby baseline comps.")

    zip_code_clean = str(zip_code).strip()
    triad_cache = st.session_state.setdefault("triad_model_cache", {})

    recommended_specs = generate_candidate_recommendations(
        zip_code=zip_code_clean,
        property_type_label=property_type,
        subdivision=subdivision or None,
        triad_cache=triad_cache,
        target_lat=safe_float(latitude),
        target_lon=safe_float(longitude),
        top_k=4,
    )

    st.markdown(f'<span class="subheader-pill">🎯 {selected_demo}</span>', unsafe_allow_html=True)

    engine_picks = [entry for entry in recommended_specs if entry.get("engine_recommendation")]
    if not engine_picks:
        st.error(
            "The builder engine couldn't generate a Triad-backed spec for this selection yet. "
            "Try a nearby ZIP or refresh after the latest training run completes."
        )
        return

    primary_context = engine_picks[0]
    alt_contexts = engine_picks[1:]

    engine_payload = primary_context.get("engine_recommendation") or {}
    triad_predictions = primary_context.get("triad_predictions") or {}
    listing_stub = primary_context.get("listing") or {}
    engine_context = primary_context.get("engine_context") or {}

    def _to_int(value: Any) -> Optional[int]:
        val = safe_float(value)
        if val is None:
            return None
        return int(round(val))

    def _format_baths(value: Any) -> str:
        if value is None:
            return "?"
        if isinstance(value, (int, float)):
            if float(value).is_integer():
                return str(int(value))
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return str(value)

    config = engine_payload.get("configuration") or {}
    config_beds = config.get("beds", listing_stub.get("beds"))
    config_baths = config.get("baths", listing_stub.get("baths"))
    config_sqft = config.get("sqft", listing_stub.get("sqft"))
    config_finish = config.get("finish_level") or listing_stub.get("finish_level") or "standard"

    bed_int = _to_int(config_beds)
    bath_label = _format_baths(config_baths)
    sqft_int = _to_int(config_sqft)
    sqft_label = f"{sqft_int:,}" if sqft_int is not None else "?"
    bed_label = str(bed_int) if bed_int is not None else "?"

    spec_title = f"{bed_label} BR / {bath_label} BA · {sqft_label} sqft"

    demand_block = engine_payload.get("demand") or {}
    margin_block = engine_payload.get("margin") or {}
    pricing_details = engine_payload.get("pricing_details") or {}

    predicted_price = safe_float(engine_payload.get("predicted_price"))
    triad_prob = safe_float(demand_block.get("sell_probability"))
    triad_dom = safe_float(demand_block.get("expected_dom"))
    price_lower = safe_float(triad_predictions.get("price_lower"))
    price_upper = safe_float(triad_predictions.get("price_upper"))

    if predicted_price is None:
        predicted_price = safe_float(triad_predictions.get("predicted_price"))
    if triad_prob is None:
        triad_prob = safe_float(triad_predictions.get("sell_probability"))
    if triad_dom is None:
        triad_dom = safe_float(triad_predictions.get("expected_dom"))

    price_interval = None
    if price_lower is not None and price_upper is not None:
        price_interval = (price_lower, price_upper)

    margin_pct = safe_float(margin_block.get("gross_margin_pct"))
    margin_dollars = safe_float(margin_block.get("gross_margin"))
    risk_score = safe_float(engine_payload.get("risk_score"))
    model_price_raw = safe_float(pricing_details.get("model_price"))
    psf_anchor_raw = safe_float(pricing_details.get("psf_anchor"))
    anchored_price_raw = safe_float(pricing_details.get("anchored_price"))

    feature_impacts = get_feature_impacts(zip_code_clean)

    def _format_pct(value: Any) -> Optional[str]:
        val = safe_float(value)
        if val is None:
            return None
        return f"{val * 100:.0f}%"

    source_label_map = {
        "historical": "Triad sold listings",
        "generated": "Triad sold listings",
        "fallback_lattice": "static fallback lattice",
        "micro_market": "micro-market fallback",
        "provided": "externally supplied configs",
    }

    summary_bits: List[str] = []
    candidate_source = engine_context.get("candidate_source")
    source_label = source_label_map.get(candidate_source)
    if not source_label:
        source_label = "Triad sold listings" if not engine_context.get("engine_error") else "unknown"
    summary_bits.append(f"Source: {source_label}")

    total_eval = engine_context.get("total_evaluated")
    total_pass = engine_context.get("total_passing")
    if isinstance(total_eval, (int, float)):
        summary_bits.append(f"Configs evaluated: {int(total_eval)}")
        if isinstance(total_pass, (int, float)):
            summary_bits.append(f"Met guardrails: {int(total_pass)}")

    constraint_bits: List[str] = []
    constraints = engine_context.get("constraints") or {}
    min_prob_label = _format_pct(constraints.get("min_sell_probability"))
    max_dom_val = safe_float(constraints.get("max_dom"))
    if min_prob_label:
        constraint_bits.append(f"min sell prob {min_prob_label}")
    if max_dom_val is not None:
        constraint_bits.append(f"max DOM {int(max_dom_val)} days")
    if constraint_bits:
        summary_bits.append("Guardrails: " + ", ".join(constraint_bits))

    eval_errors = engine_context.get("evaluation_errors") or []
    if eval_errors:
        summary_bits.append(f"Issues: {eval_errors[0]}")
        if len(eval_errors) > 1:
            summary_bits.append(f"+{len(eval_errors) - 1} more")

    if summary_bits:
        st.caption(" · ".join(summary_bits))
    if model_price_raw or psf_anchor_raw:
        price_caption_parts: list[str] = []
        if model_price_raw:
            price_caption_parts.append(f"Model price: {format_currency(model_price_raw)}")
        if psf_anchor_raw:
            price_caption_parts.append(f"High-end anchor: {psf_anchor_raw:,.0f} $/sqft")
        if anchored_price_raw and anchored_price_raw != model_price_raw:
            price_caption_parts.append(f"Anchor price: {format_currency(anchored_price_raw)}")
        if price_caption_parts:
            st.caption(" | ".join(price_caption_parts))

    if engine_payload and engine_payload.get("passes_constraints") is False:
        guardrail_reasons: List[str] = []
        min_prob = safe_float(constraints.get("min_sell_probability"))
        if triad_prob is not None and min_prob is not None and triad_prob < min_prob:
            guardrail_reasons.append(
                f"sell probability { _format_pct(triad_prob) } below { _format_pct(min_prob) }"
            )
        max_dom_constraint = safe_float(constraints.get("max_dom"))
        if triad_dom is not None and max_dom_constraint is not None and triad_dom > max_dom_constraint:
            guardrail_reasons.append(
                f"expected DOM {int(round(triad_dom))} days above {int(max_dom_constraint)} day cap"
            )
        if not guardrail_reasons:
            guardrail_reasons.append("Review demand guardrails.")
        st.warning("Guardrail miss: " + " | ".join(guardrail_reasons))

    price_levers, dom_levers = extract_top_feature_moves(feature_impacts)
    top_feature_recs = price_levers[:3]

    predicted_price_clean = _to_int(predicted_price)
    price_interval_clean = (
        ( _to_int(price_interval[0]), _to_int(price_interval[1]) )
        if price_interval
        else None
    )

    narrative_metrics = {
        "lot": {
            "zip_code": listing_stub.get("zip_code") or zip_code_clean,
            "latitude": safe_float(listing_stub.get("latitude")),
            "longitude": safe_float(listing_stub.get("longitude")),
            "subdivision": listing_stub.get("subdivision") or (subdivision or None),
        },
        "configuration": {
            "beds": bed_int,
            "baths": bath_label,
            "sqft": sqft_int,
            "finish_level": config_finish,
            "stories": config.get("stories", listing_stub.get("stories", 2)),
            "garage_spaces": config.get("garage_spaces", listing_stub.get("garage_spaces", 2)),
        },
        "demand": {
            "sell_probability": triad_prob,
            "sell_probability_source": "triad_model",
            "expected_dom": triad_dom,
            "expected_dom_source": "triad_model",
            "seasonality_fast_probability": None,
            "seasonality_dom": None,
        },
        "margin": {
            "gross_margin": margin_dollars,
            "gross_margin_pct": margin_pct,
            "roi": None,
        },
        "pricing": {
            "predicted_sale_price": predicted_price_clean,
            "predicted_sale_price_formatted": format_currency(predicted_price) if predicted_price is not None else None,
        },
        "inventory": {},
    }
    if price_interval_clean and all(price_interval_clean):
        narrative_metrics["pricing"]["predicted_price_low"] = price_interval_clean[0]
        narrative_metrics["pricing"]["predicted_price_high"] = price_interval_clean[1]
        narrative_metrics["pricing"]["predicted_price_low_formatted"] = format_currency(price_interval_clean[0])
        narrative_metrics["pricing"]["predicted_price_high_formatted"] = format_currency(price_interval_clean[1])
        narrative_metrics["pricing"]["predicted_price_range"] = (
            f"{format_currency(price_interval_clean[0])} to {format_currency(price_interval_clean[1])}"
        )
    narrative_metrics["features"] = top_feature_recs

    tabs = st.tabs(["Recommendation", "Alternatives", "Feature Boosters", "Raw Data"])
    recommendation_tab, alternatives_tab, features_tab, raw_tab = tabs

    with recommendation_tab:
        recommendation_tab.markdown(f"### {spec_title}")
        recommendation_tab.caption(f"Finish level: **{config_finish.replace('_', ' ').title()}**")

        price_help = (
            f"Range {format_currency(price_interval[0])} – {format_currency(price_interval[1])}"
            if price_interval
            else None
        )
        margin_help = (
            f"≈ {format_currency(margin_dollars)} gross margin"
            if margin_dollars is not None
            else None
        )

        col1, col2, col3, col4 = recommendation_tab.columns(4)
        col1.metric(
            "Predicted price",
            format_currency(predicted_price) if predicted_price is not None else "—",
            help=price_help,
        )
        col2.metric(
            "Sell probability",
            f"{triad_prob * 100:.0f}%" if triad_prob is not None else "—",
        )
        col3.metric(
            "Expected DOM",
            f"{triad_dom:.0f} days" if triad_dom is not None else "—",
        )
        col4.metric(
            "Gross margin %",
            f"{margin_pct:.1f}%" if margin_pct is not None else "—",
            help=margin_help,
        )
        if risk_score is not None:
            recommendation_tab.caption(f"Risk score (lower is better): **{risk_score:.2f}**")

        if price_levers:
            recommendation_tab.markdown("#### Top price levers")
            for item in price_levers[:3]:
                lift = item.get("price_lift")
                pct = item.get("price_lift_pct")
                parts = []
                if lift is not None:
                    parts.append(f"+{format_currency(lift)}")
                if pct is not None:
                    parts.append(f"({pct:+.1f}%)")
                summary = " ".join(parts) if parts else "—"
                band_hint = ""
                bands = item.get("price_lift_bands") or {}
                if bands:
                    best_band = max(bands.items(), key=lambda kv: safe_float(kv[1]) or float("-inf"))
                    if safe_float(best_band[1]) is not None:
                        band_hint = f" · best in {best_band[0]}: {format_currency(best_band[1])}"
                method = item.get("price_method") or "unknown"
                sample_text = f" (n={item['count']})" if item.get("count") else ""
                recommendation_tab.markdown(
                    f"- **{item['label']}** · {summary}{band_hint} · method: {method}{sample_text}"
                )
        elif feature_impacts:
            recommendation_tab.info("Feature impact stats are loading; rerun shortly for prioritized options.")

        if dom_levers:
            recommendation_tab.markdown("#### Top DOM levers")
            for item in dom_levers[:3]:
                dom_delta = item.get("dom_delta")
                dom_text = f"{dom_delta:+.1f} days" if dom_delta is not None else "—"
                band_hint = ""
                bands = item.get("dom_delta_bands") or {}
                if bands:
                    best_band = min(bands.items(), key=lambda kv: safe_float(kv[1]) if safe_float(kv[1]) is not None else float("inf"))
                    if safe_float(best_band[1]) is not None:
                        band_hint = f" · strongest in {best_band[0]}: {best_band[1]:+.1f} days"
                method = item.get("dom_method") or "unknown"
                sample_text = f" (n={item['count']})" if item.get("count") else ""
                recommendation_tab.markdown(
                    f"- **{item['label']}** · {dom_text}{band_hint} · method: {method}{sample_text}"
                )
        elif feature_impacts:
            recommendation_tab.info("No DOM reductions surfaced yet; expand the dataset or widen the look-back window.")
        else:
            recommendation_tab.warning("No feature impact data available yet for this ZIP.")

        recommendation_tab.markdown("#### Narrative")
        recommendation_tab.info(generate_recommendation_narrative(narrative_metrics))

    with alternatives_tab:
        alt_rows: List[Dict[str, Any]] = []
        for rank, context in enumerate(alt_contexts, start=2):
            alt_payload = context.get("engine_recommendation") or {}
            alt_config = alt_payload.get("configuration") or {}
            alt_demand = alt_payload.get("demand") or {}
            alt_margin = alt_payload.get("margin") or {}
            alt_rows.append(
                {
                    "Rank": rank,
                    "Beds": _to_int(alt_config.get("beds")),
                    "Baths": _format_baths(alt_config.get("baths")),
                    "Sqft": _to_int(alt_config.get("sqft")),
                    "Predicted price": format_currency(alt_payload.get("predicted_price"))
                    if alt_payload.get("predicted_price") is not None
                    else "—",
                    "Sell probability": f"{safe_float(alt_demand.get('sell_probability')) * 100:.0f}%"
                    if alt_demand.get("sell_probability") is not None
                    else "—",
                    "Expected DOM": f"{safe_float(alt_demand.get('expected_dom')):.0f}"
                    if alt_demand.get("expected_dom") is not None
                    else "—",
                    "Gross margin %": f"{safe_float(alt_margin.get('gross_margin_pct')):.1f}%"
                    if alt_margin.get("gross_margin_pct") is not None
                    else "—",
                }
            )

        if alt_rows:
            alt_df = pd.DataFrame(alt_rows)
            alternatives_tab.dataframe(alt_df, use_container_width=True, hide_index=True)
        else:
            alternatives_tab.info("No additional builder-engine specs surfaced. Triad models converged on the primary plan.")

    with features_tab:
        if feature_impacts:
            features_tab.markdown("##### Price levers (builder-ready)")
            price_rows: List[Dict[str, Any]] = []
            price_candidates: List[Dict[str, Any]] = []
            for key, payload in feature_impacts.items():
                record = _make_feature_record(key, payload)
                if not record:
                    continue
                price_signal = record.get("price_signal")
                if price_signal is None or price_signal < MIN_PRICE_LIFT_DOLLARS:
                    continue
                record = record.copy()
                if record.get("price_lift") is not None and record["price_lift"] <= 0:
                    record["price_lift"] = None
                if record.get("price_lift_pct") is not None and record["price_lift_pct"] <= 0:
                    record["price_lift_pct"] = None
                price_candidates.append(record)

            # Deduplicate by key to keep the strongest version
            dedup_price: Dict[str, Dict[str, Any]] = {}
            for candidate in price_candidates:
                key = candidate["key"]
                existing = dedup_price.get(key)
                if not existing:
                    dedup_price[key] = candidate
                    continue
                existing_score = (
                    existing.get("price_signal") or 0.0,
                    existing.get("price_lift_pct") or 0.0,
                    existing.get("count") or 0,
                )
                candidate_score = (
                    candidate.get("price_signal") or 0.0,
                    candidate.get("price_lift_pct") or 0.0,
                    candidate.get("count") or 0,
                )
                if candidate_score > existing_score:
                    dedup_price[key] = candidate

            price_candidates = sorted(dedup_price.values(), key=_price_sort_score, reverse=True)

            price_display: List[Dict[str, Any]] = []
            existing_keys: Set[str] = set()
            for candidate in price_candidates:
                if len(price_display) >= 6:
                    break
                key = candidate["key"]
                if key in existing_keys:
                    continue
                price_display.append(candidate)
                existing_keys.add(key)
            if len(price_display) < 6:
                for item in price_levers:
                    if len(price_display) >= 6:
                        break
                    if item["key"] in existing_keys:
                        continue
                    price_display.append(item)
                    existing_keys.add(item["key"])

            if price_display:
                for item in price_display:
                    bands = item.get("price_lift_bands") or {}
                    best_band = ""
                    if bands:
                        band_tuple = item.get("best_price_band") or _best_positive_price_band(item)
                        if band_tuple:
                            best_band, best_val = band_tuple
                        else:
                            best_band, best_val = max(
                                bands.items(),
                                key=lambda kv: safe_float(kv[1]) if safe_float(kv[1]) is not None else float("-inf"),
                            )
                        best_val_clean = safe_float(best_val)
                        if best_val_clean is not None and best_val_clean > 0:
                            best_band = f"{best_band}: {format_currency(best_val_clean)}"
                        else:
                            best_band = ""
                    price_rows.append(
                        {
                            "Feature": item["label"],
                            "Median lift $": format_currency(item["price_lift"])
                            if item.get("price_lift") is not None
                            else "—",
                            "Lift %": f"{item['price_lift_pct']:+.1f}%"
                            if item.get("price_lift_pct") is not None
                            else "—",
                            "Sample size": item.get("count") if item.get("count") is not None else "—",
                            "Best price band": best_band or "—",
                            "Method": item.get("price_method") or "—",
                        }
                    )
                price_df = pd.DataFrame(price_rows)
                features_tab.dataframe(price_df, use_container_width=True, hide_index=True)
            else:
                features_tab.info("No price levers surfaced; expand the feature dataset to unlock more signals.")

            features_tab.markdown("##### DOM levers (speed to sell)")
            dom_rows: List[Dict[str, Any]] = []
            dom_candidates: List[Dict[str, Any]] = []
            for key, payload in feature_impacts.items():
                record = _make_feature_record(key, payload)
                if not record:
                    continue
                dom_signal = record.get("dom_signal")
                if dom_signal is None or abs(dom_signal) < MIN_DOM_REDUCTION_DAYS:
                    continue
                record = record.copy()
                if record.get("dom_delta") is not None and record["dom_delta"] >= 0:
                    record["dom_delta"] = None
                dom_candidates.append(record)

            dom_candidates = sorted(dom_candidates, key=_dom_sort_score)

            dom_display: List[Dict[str, Any]] = []
            existing_dom_keys: Set[str] = set()
            for candidate in dom_candidates:
                if len(dom_display) >= 6:
                    break
                key = candidate["key"]
                if key in existing_dom_keys:
                    continue
                dom_display.append(candidate)
                existing_dom_keys.add(key)
            if len(dom_display) < 6:
                for item in dom_levers:
                    if len(dom_display) >= 6:
                        break
                    if item["key"] in existing_dom_keys:
                        continue
                    dom_display.append(item)
                    existing_dom_keys.add(item["key"])

            if dom_display:
                for item in dom_display:
                    bands = item.get("dom_delta_bands") or {}
                    best_band = ""
                    if bands:
                        band_tuple = item.get("best_dom_band") or _best_negative_dom_band(item)
                        if band_tuple:
                            best_band, best_val = band_tuple
                        else:
                            best_band, best_val = min(
                                bands.items(),
                                key=lambda kv: safe_float(kv[1]) if safe_float(kv[1]) is not None else float("inf"),
                            )
                        best_val_clean = safe_float(best_val)
                        if best_val_clean is not None:
                            best_band = f"{best_band}: {best_val_clean:+.1f}d"
                    dom_rows.append(
                        {
                            "Feature": item["label"],
                            "Median DOM delta": f"{item['dom_delta']:+.1f}d"
                            if item.get("dom_delta") is not None
                            else "—",
                            "Sample size": item.get("count") if item.get("count") is not None else "—",
                            "Best DOM band": best_band or "—",
                            "Method": item.get("dom_method") or "—",
                        }
                    )
                dom_df = pd.DataFrame(dom_rows)
                features_tab.dataframe(dom_df, use_container_width=True, hide_index=True)
            else:
                features_tab.info("No DOM reductions surfaced yet; widen the lookback window for more comps.")

            def _format_price_bands(bands: Optional[Dict[str, Any]]) -> str:
                if not bands:
                    return ""
                parts = []
                for band, val in sorted(bands.items()):
                    if val is None:
                        continue
                    parts.append(f"{band}: {format_currency(val)}")
                return "; ".join(parts)

            def _format_dom_bands(bands: Optional[Dict[str, Any]]) -> str:
                if not bands:
                    return ""
                parts = []
                for band, val in sorted(bands.items()):
                    if val is None:
                        continue
                    parts.append(f"{band}: {val:+.1f}d")
                return "; ".join(parts)

            rows = []
            for key, payload in feature_impacts.items():
                label = payload.get("label", key.replace("_", " ").title())
                if "pool" in str(label).lower() or "pool" in str(key).lower():
                    continue
                rows.append(
                    {
                        "Feature": label,
                        "Sample count": payload.get("count"),
                        "Price lift": payload.get("price_lift"),
                        "Price lift %": payload.get("price_lift_pct"),
                        "Price method": payload.get("price_method"),
                        "DOM delta": payload.get("dom_delta"),
                        "DOM method": payload.get("dom_method"),
                        "Price bands": _format_price_bands(payload.get("price_lift_bands")),
                        "DOM bands": _format_dom_bands(payload.get("dom_delta_bands")),
                    }
                )

            feature_df = pd.DataFrame(rows)
            if not feature_df.empty:
                feature_df.sort_values(
                    by=["Price lift"],
                    ascending=False,
                    inplace=True,
                    na_position="last",
                )
                feature_df["Price lift"] = feature_df["Price lift"].map(
                    lambda v: format_currency(v) if v is not None else "—"
                )
                feature_df["Price lift %"] = feature_df["Price lift %"].map(
                    lambda v: f"{v:+.1f}%" if v is not None else "—"
                )
                feature_df["DOM delta"] = feature_df["DOM delta"].map(
                    lambda v: f"{v:+.1f}d" if v is not None else "—"
                )
                with features_tab.expander("See full feature stats"):
                    st.dataframe(feature_df, use_container_width=True, hide_index=True)
            else:
                features_tab.info("No qualifying features met the sample threshold for this ZIP.")
        else:
            features_tab.warning("Feature impact cache is empty for this ZIP. Run the feature impact script or widen the dataset.")

    with raw_tab:
        raw_tab.json(
            {
                "primary": primary_context,
                "alternatives": alt_contexts,
                "feature_impacts": feature_impacts,
            }
        )


def show_listing_popularity():
    """Show listing popularity analysis page."""
    st.header("🔥 Listing Popularity Analysis")
    
    st.markdown("""
    **Analyze active listings from Zillow to identify what makes properties popular:**
    - Which listings get the most attention (views, saves)
    - What features drive popularity
    - DOM to pending analysis (fast-selling properties)
    - Competitive insights for your builds
    """)
    
    st.markdown("---")
    
    # Input section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        zip_code = st.text_input("ZIP Code", value="27410", key="listing_zip")
    
    with col2:
        status = st.selectbox(
            "Listing Status",
            ["active", "pending", "sold"],
            index=0,
            key="listing_status"
        )
    
    with col3:
        max_results = st.number_input(
            "Max Results",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="listing_max_results"
        )
    
    # Source selection
    source = st.radio(
        "Data Source",
        ["auto", "rapidapi_zillow", "attom"],
        index=0,
        help="Auto tries RapidAPI first, falls back to Attom if needed",
        horizontal=True
    )
    
    st.markdown("---")
    
    # Fetch button
    if st.button("🚀 Fetch & Analyze Listings", type="primary", key="fetch_listings"):
        with st.spinner(f"Fetching {status} listings from ZIP {zip_code}..."):
            try:
                # Fetch listings
                listings = safe_listings_scraper.fetch_listings(
                    zip_code=zip_code,
                    status=status,
                    max_results=max_results,
                    source=source
                )
                
                if not listings:
                    st.error("❌ No listings found. Please check:")
                    st.info("💡 Make sure your RapidAPI key is set in .env (or use Attom with --source attom)")
                    return
                
                st.success(f"✅ Found {len(listings)} listings")
                
                # Analyze popularity
                st.markdown("### 📊 Popularity Analysis")
                
                # Determine metric based on available data
                has_views_saves = any(l.get('views') or l.get('saves') for l in listings)
                popularity_metric = 'composite' if has_views_saves else 'fast_dom'
                
                results = popularity_analyzer.analyze_popular_listings(
                    listings=listings,
                    top_n=20,
                    popularity_metric=popularity_metric
                )
                
                # Display top listings
                st.subheader("🏆 Top Popular Listings")
                
                for i, listing in enumerate(results['top_listings'][:10], 1):
                    with st.expander(f"#{i}: {listing.get('address', 'Unknown')[:60]}...", expanded=(i <= 3)):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Price", f"${listing.get('price', 0):,}")
                        with col2:
                            st.metric("Beds/Baths", f"{listing.get('beds', '?')}/{listing.get('baths', '?')}")
                        with col3:
                            st.metric("SqFt", f"{listing.get('sqft', 0):,}")
                        with col4:
                            st.metric("Popularity Score", f"{listing['popularity_score']:.2f}")
                        
                        # Additional metrics
                        col5, col6, col7 = st.columns(3)
                        with col5:
                            dom = listing.get('days_on_zillow', 'N/A')
                            st.metric("Days on Zillow", dom if isinstance(dom, int) else 'N/A')
                        with col6:
                            views = listing.get('views', 'N/A')
                            st.metric("Views", views if views else 'N/A')
                        with col7:
                            saves = listing.get('saves', 'N/A')
                            st.metric("Saves", saves if saves else 'N/A')
                        
                        # Features
                        features = listing.get('features', [])
                        if features:
                            st.write("**Features:**", ", ".join(features[:10]))
                        
                        # Detail URL
                        if listing.get('detail_url'):
                            st.markdown(f"[View on Zillow →]({listing['detail_url']})")
                
                # Feature analysis
                if results['feature_analysis'].get('drivers'):
                    st.subheader("🎯 Popularity Drivers")
                    st.markdown("Features that appear more frequently in popular listings:")
                    
                    drivers = results['feature_analysis']['drivers'][:10]
                    feature_impact = results['feature_analysis'].get('feature_impact', {})
                    
                    for feature in drivers:
                        impact = feature_impact[feature]
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{feature.title()}**")
                            st.caption(
                                f"Appears in {impact['pct_in_top']:.1f}% of top listings "
                                f"(vs {impact['pct_in_all']:.1f}% overall) - "
                                f"{impact['impact_ratio']:.1f}x more common"
                            )
                        with col2:
                            if impact['impact_ratio'] >= 2.0:
                                st.success("🔥 Strong Driver")
                            elif impact['impact_ratio'] >= 1.5:
                                st.info("📈 Driver")
                            else:
                                st.write("✓")
                        st.markdown("---")
                
                # Price analysis
                if results['price_analysis']:
                    st.subheader("💰 Price Analysis")
                    price_data = results['price_analysis']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Median Price", f"${price_data.get('median', 0):,}")
                    with col2:
                        st.metric("Average Price", f"${price_data.get('mean', 0):,}")
                    with col3:
                        st.metric("Price Range", f"${price_data.get('min', 0):,} - ${price_data.get('max', 0):,}")
                    with col4:
                        st.metric("Q25-Q75", f"${price_data.get('q25', 0):,} - ${price_data.get('q75', 0):,}")
                
                # Configuration analysis
                if results['config_analysis']:
                    st.subheader("🏠 Configuration Analysis")
                    config_data = results['config_analysis']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Most Common Configurations:**")
                        for config, count in list(config_data.get('most_common_configs', {}).items())[:5]:
                            st.write(f"- {config}: {count} listings")
                    
                    with col2:
                        if config_data.get('avg_sqft'):
                            st.metric("Avg SqFt", f"{config_data['avg_sqft']:,}")
                        if config_data.get('avg_price_per_sqft'):
                            st.metric("Avg $/SqFt", f"${config_data['avg_price_per_sqft']:,}")
                
                # Pending analysis (if we have both active and pending)
                if status == 'pending' and len(listings) > 0:
                    st.subheader("⏱️ DOM to Pending Analysis")
                    st.info("💡 For DOM to pending analysis, fetch active listings first, then pending to compare.")
                    
                    # Try to analyze if we can
                    try:
                        dom_values = [l.get('days_on_zillow') for l in listings if l.get('days_on_zillow')]
                        if dom_values:
                            st.write(f"**Median DOM to Pending:** {int(np.median(dom_values))} days")
                            st.write(f"**Average DOM to Pending:** {np.mean(dom_values):.1f} days")
                            st.write(f"**Range:** {min(dom_values)} - {max(dom_values)} days")
                    except:
                        pass
                
                # Store in session state for download
                st.session_state['listing_analysis_results'] = results
                st.session_state['listing_data'] = listings
                
                # Download button
                st.markdown("---")
                if st.button("📥 Download Analysis Results (JSON)", key="download_listing_analysis"):
                    import json
                    download_data = {
                        'analysis': results,
                        'listings': listings,
                        'zip_code': zip_code,
                        'status': status,
                        'analyzed_at': datetime.now().isoformat()
                    }
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(download_data, indent=2, default=str),
                        file_name=f"listing_analysis_{zip_code}_{status}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # Show cached results if available
    if 'listing_analysis_results' in st.session_state:
        st.markdown("---")
        st.info("💡 Previous analysis results are cached. Fetch new listings to update.")


def _update_feature_coverage(
    cache_path: Path, zip_code: str, total_rows: int, impacts: Dict[str, Dict[str, Any]]
) -> None:
    """Persist per-ZIP coverage plus aggregate stats for feature impacts."""
    coverage_file = cache_path / "coverage.json"
    default_payload: Dict[str, Any] = {"by_zip": {}, "summary": {}}

    try:
        coverage = json.loads(coverage_file.read_text())
        if not isinstance(coverage, dict):
            coverage = default_payload.copy()
    except Exception:
        coverage = default_payload.copy()

    by_zip = coverage.setdefault("by_zip", {})
    if total_rows <= 0 and not impacts:
        by_zip.pop(zip_code, None)
    else:
        by_zip[zip_code] = {
            "updated_at": datetime.utcnow().isoformat(),
            "listings": int(total_rows),
            "features": impacts,
        }

    summary: Dict[str, Dict[str, Any]] = {}
    for info in by_zip.values():
        features = info.get("features") or {}
        seen: Set[str] = set()
        for feat_key, payload in features.items():
            entry = summary.setdefault(
                feat_key,
                {
                    "label": payload.get("label")
                    or FEATURE_LIBRARY.get(feat_key, {}).get("label", feat_key.replace("_", " ").title()),
                    "zip_count": 0,
                    "total_count": 0,
                    "total_listings": 0,
                },
            )
            if feat_key not in seen:
                entry["zip_count"] += 1
                seen.add(feat_key)
            entry["total_count"] += int(payload.get("count") or 0)
            entry["total_listings"] += int(info.get("listings") or 0)

    coverage["summary"] = summary
    coverage_file.write_text(json.dumps(coverage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()






