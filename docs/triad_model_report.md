# Triad Fast-Seller Pipeline Report

## 1. End-to-End Flow Overview
- **Objective**: Predict the probability that a new-build configuration sells quickly (fast-seller probability) and its expected days-on-market (DOM).
- **Pipeline stages**: ingest sold listings → sanitize & cache → engineer enriched features → train pricing/demand models → serve predictions in Streamlit.
- **Key modules**: `backend/ml/train_fast_seller_model.py`, `backend/ml/feature_engineering.py`, `scripts/train_pricing_demand_triad.py`, `prototype/app.py`.

## 2. Data Ingestion & Sanitization
- **Source**: RealEstateApi sold listings for the entire Triad (730-day lookback).
- **Loader**: `RealEstateApiListingLoader.fetch_sold_with_details` handles pagination, respects max-per-ZIP caps, and collects MLS detail payloads.
- **Caching**: listings saved under `cache/listings/realestateapi/` with suffixes (`maxall` or `max{n}`) via `get_cache_path` / `save_cached_listings`.
- **Sanitization** (`_sanitize_fetched_listings`):
  - Deduplicates on `property_id` / `mls_id`.
  - Drops records without a resolvable sold date (using `_extract_listing_sale_datetime`).
  - Applies a recency cutoff based on `days_back`.
  - Provides stats (raw, kept, duplicates, stale, missing_sale) logged per ZIP.
- **DOM extraction**: `SoldListingsAnalyzer.extract_timeline` normalizes listing/pending/sold dates, handles timezone conversions, fills missing DOM from history, and infers DOM-to-pending when only DOM-to-sold is available.
- **Safeguards**: if sanitized ZIP count exceeds `settings.realestateapi_max_results_per_zip` (default 6000), training aborts with a descriptive error to prevent runaway jobs.

## 3. Feature Engineering Highlights
- **Structural**: beds, baths, sqft, lot size, year built, stories.
- **Pricing ratios**: price-per-sqft, price-to-ZIP median, price-to-subdivision median.
- **Seasonality encodings**: `sale_month_sin/cos`, `sale_quarter_sin/cos`, `is_spring_summer`, days since sale.
- **Location signals**: subdivision frequency, lat/lon, property-type dummies (SFR/townhome/condo).
- **Quality & condition**: normalized scores built from property detail and LLM-extracted descriptions (via `FeatureExtractor` + `_create_llm_extracted_features`).
- **Listing history**: counts of MLS events (listed/pending/sold/price change), relist flags, `days_from_list_to_pending`, `days_from_list_to_sale`, `prior_sale_gap_days`.
- **Inventory proxies**: rolling ZIP sale counts (30/90 days) and `zip_inventory_trend_ratio` (30d / 90d).
- **Seasonality report** (`scripts/seasonality_adjusted_predictions.py`): rebuilds feature matrix, neutralises seasonal columns, and outputs baseline fast-seller metrics used for historical comparison in the UI.

## 4. Model Training (`scripts/train_pricing_demand_triad.py`)
1. **Collect listings** via `fetch_sold_listings_with_features` (reuses the sanitized ingestion path; optionally bypass cache with `--no-cache`).
2. **Engineer features** (`FeatureEngineer.engineer_features`) once for the combined listing set.
3. **Pricing model prep**: filter training targets to $50k–$3M, then feed into `PricingModel.train` (XGBoost regressor). Latest metrics: Test MAPE ≈ **3.84%**, Test R² ≈ **0.95**.
4. **Demand model prep**:
   - Compute ZIP-level DOM medians through `calculate_median_thresholds_by_zip`.
   - Build classification/regression targets via `prepare_targets` (fast-seller mask, DOM quantile clipping, bucket labels).
   - Align feature matrix to `FeatureEngineer.get_feature_names` and fill medians.
   - Train `DemandModel.train` (XGBoost classifier for sell probability + regressor for DOM). Latest metrics: Sell probability Test AUC ≈ **0.566**, DOM Test MAPE ≈ **82.5%** (high due to noisy DOM labels and metro/rural variety).
5. **Artifacts**: Both models saved under `models/triad_latest/` (`triad_pricing_model*.pkl`, `triad_demand_model_*.pkl`) with metadata summarised in `triad_training_summary.json`.

## 5. Streamlit Inference (`prototype/app.py` > `show_ml_recommendations`)
- **Hero selection**: user chooses preloaded coordinates or custom entry.
- **Seasonality baseline**: filtered rows from `seasonality_adjusted_predictions.csv` provide historical fast-seller probability/DOM for comparison.
- **Triad scoring** (`compute_tri_model_predictions`):
  - Pull matching cached listings (by property ID or nearest lat/lon) and rerun feature engineering for that record.
  - Load `triad_pricing_model` & `triad_demand_model` and compute:
    - `sell_probability` (current model view).
    - `expected_dom` (current model view).
    - `predicted_price` + 80% interval (pricing model).
  - Return a feature snapshot for UI inspection.
- **UI presentation**:
  - KPI cards show Triad metrics; caption compares them to the seasonality baseline (e.g., “triad model 45% / 48 days vs. seasonality 77% / 35 days”).
  - Inventory trend ratio shown when available.
  - Narrative generation (`backend/ai_engine/narrative_generator.py`) instructs the LLM to emphasise TRIAD metrics as primary and label seasonality numbers as baseline context.

## 6. Why Triad vs. Seasonality Differ
- **Seasonality baseline**: Historical two-year average (mostly pre-2024) often reflects hotter absorption periods.
- **Triad model**: Trained on the full 730-day window but makes predictions using **current feature values** (price positioning, inventory ratios, relist history, etc.).
  - Example: If `price_to_zip_median` > 1 and `zip_inventory_trend_ratio` > 1 (supply increasing), the classifier lowers sell probability and increases expected DOM.
  - This yields statements like “Triad model 45% / 48 days; seasonality baseline 77% / 35 days” — signalling market softening in today’s conditions.

## 7. Next Steps: Populating `current_listings_context`
- **Current state**: Demand model only ingests closed sales; inventory proxy relies on trailing sold counts (30d/90d).
- **Why we need live actives/pending**:
  - Identify real-time competition, pricing pressure, and absorption speed.
  - Improve fast-seller probabilities by understanding today’s pipeline rather than trailing sales alone.
- **Implementation ideas**:
  1. **Active/Pending scraper**: Extend `safe_listings_scraper` (or a new collector) to pull active & pending listings with consistent fields (price, beds, DOM-so-far, status timestamps).
  2. **Feature engineering for actives**: Mirror structure of sold features (price ratios, inventory counters, time-on-market so far).
  3. **Streamlit integration**: Populate `current_listings_context` when computing recommendations, allowing the recommendation engine to incorporate live competition in scorecards.
  4. **Model refresh**: Optionally retrain demand model with a hybrid dataset (sold + active snapshots) to better capture late-2024/2025 dynamics.

Integrating real-time listings will align the model’s view with today’s market conditions, closing the gap between trailing-seasonality baselines and actionable predictions.
