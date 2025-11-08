"""
Training script for fast seller prediction model.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.settings import settings
from backend.data_collectors.safe_listings_scraper import SafeListingsScraper
from backend.data_collectors.realestateapi_loader import RealEstateApiListingLoader
from backend.analyzers.sold_listings_analyzer import SoldListingsAnalyzer
from backend.ml.feature_extractor import FeatureExtractor
from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.fast_seller_model import FastSellerModel
from backend.ml.target_prep import (
    assign_micro_market,
    infer_dom_to_pending,
)
from backend.ml.micro_market_recommender import build_micro_market_summary
from scripts.identify_similar_zips import (
    get_training_zips, SIMILAR_ZIPS, ALL_GREENSBORO_ZIPS,
    get_triad_zips,
    get_zips_for_counties,
    TRIAD_COUNTIES,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)




def _cache_filename_suffix(max_per_zip: Optional[int]) -> str:
    if max_per_zip is None or max_per_zip <= 0:
        return "maxall"
    return f"max{max_per_zip}"


def _parse_sale_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except (OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith(" UTC"):
            text = text[:-4] + "+00:00"
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt)
                return dt
            except ValueError:
                continue
    return None


def _extract_listing_sale_datetime(listing: Dict[str, Any]) -> Optional[datetime]:
    for key in ("dateSold", "sold_date", "soldDate", "mlsLastSaleDate", "lastSaleDate"):
        dt = _parse_sale_datetime(listing.get(key))
        if dt:
            return dt
    detail = listing.get("property_detail_raw") or {}
    if isinstance(detail, dict):
        dt = _parse_sale_datetime(detail.get("mlsLastSaleDate") or detail.get("lastSaleDate"))
        if dt:
            return dt
        sale_history = detail.get("saleHistory")
        if isinstance(sale_history, list):
            for entry in sale_history:
                if not isinstance(entry, dict):
                    continue
                dt = _parse_sale_datetime(entry.get("saleDate") or entry.get("recordingDate"))
                if dt:
                    return dt
    return None


def _sanitize_fetched_listings(
    listings: List[Dict[str, Any]],
    *,
    days_back: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    raw_count = len(listings)
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    unique: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    duplicates = 0
    stale = 0
    missing_sale = 0

    for listing in listings:
        key = (
            listing.get("property_id")
            or listing.get("mls_id")
            or (listing.get("summary") or {}).get("propertyId")
            or (listing.get("summary") or {}).get("id")
        )
        if key and key in seen_keys:
            duplicates += 1
            continue
        if key:
            seen_keys.add(key)

        sale_dt = _extract_listing_sale_datetime(listing)
        if sale_dt is None:
            missing_sale += 1
            continue
        if sale_dt < cutoff:
            stale += 1
            continue

        unique.append(listing)

    stats = {
        "raw_count": raw_count,
        "unique_count": len(unique),
        "duplicates": duplicates,
        "stale": stale,
        "missing_sale": missing_sale,
    }
    return unique, stats


def get_cache_path(zip_code: str, days_back: int, source: str, max_per_zip: Optional[int]) -> Path:
    """Get cache file path for a ZIP code and source."""
    cache_dir = Path("cache/listings") / source
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = _cache_filename_suffix(max_per_zip)
    return cache_dir / f"listings_{zip_code}_{days_back}days_{suffix}.json"


def load_cached_listings(zip_code: str, days_back: int, source: str, max_per_zip: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    """Load cached listings if available."""
    cache_path = get_cache_path(zip_code, days_back, source, max_per_zip)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache for {zip_code}: {e}")
            return None

    # Backwards compatibility: fall back to legacy cache filename when a specific cap is requested
    if max_per_zip and max_per_zip > 0:
        legacy_path = Path("cache/listings") / source / f"listings_{zip_code}_{days_back}days.json"
        if legacy_path.exists():
            try:
                with open(legacy_path, 'r') as f:
                    logger.info("Using legacy cache for %s (days_back=%s, max_per_zip=%s)", zip_code, days_back, max_per_zip)
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load legacy cache for {zip_code}: {e}")
    return None


def save_cached_listings(zip_code: str, days_back: int, listings: List[Dict[str, Any]], source: str, max_per_zip: Optional[int]):
    """Save listings to cache."""
    cache_path = get_cache_path(zip_code, days_back, source, max_per_zip)
    try:
        with open(cache_path, 'w') as f:
            json.dump(listings, f, indent=2, default=str)
        logger.info(f"Cached {len(listings)} listings for {zip_code} [{source}]")
    except Exception as e:
        logger.warning(f"Failed to save cache for {zip_code}: {e}")


def fetch_sold_listings_with_features(
    zip_codes: list,
    days_back: int = 365,
    max_per_zip: Optional[int] = None,
    use_cache: bool = True,
    parallel: bool = False  # Sequential to avoid rate limits
) -> list:
    """Fetch sold listings, extract features, and calculate DOM metrics."""
    listings_source = getattr(settings, 'listings_source', 'rapidapi').lower()
    use_realestateapi = listings_source == 'realestateapi'

    scraper = None
    loader = None

    if use_realestateapi:
        loader = RealEstateApiListingLoader()
    else:
        scraper = SafeListingsScraper()

    analyzer = SoldListingsAnalyzer()
    extractor = FeatureExtractor()

    all_listings = []
    total_zips = len(zip_codes)
    max_label = "all" if not max_per_zip or max_per_zip <= 0 else max_per_zip
    logger.info(
        "Starting data collection: %s ZIPs | days_back=%s | max_per_zip=%s | use_cache=%s | source=%s",
        total_zips,
        days_back,
        max_label,
        use_cache,
        listings_source,
    )
    collection_start = time.perf_counter()

    def process_zip(zip_code: str, index: int) -> List[Dict[str, Any]]:
        """Process a single ZIP code."""
        source_tag = listings_source
        zip_start = time.perf_counter()
        logger.info("[%s/%s] ⏳ Processing ZIP %s", index, total_zips, zip_code)
        listings: Optional[List[Dict[str, Any]]] = None
        loaded_from_cache = False

        # Check cache first
        if use_cache:
            cached = load_cached_listings(zip_code, days_back, source_tag, max_per_zip)
            if cached:
                duration = time.perf_counter() - zip_start
                logger.info(
                    "[%s/%s] ✅ ZIP %s loaded from cache (%s listings) in %.1fs",
                    index,
                    total_zips,
                    zip_code,
                    len(cached),
                    duration,
                )
                listings = cached
                loaded_from_cache = True

        if listings is None:
            logger.info(
                "[%s/%s] 🚀 Fetching sold listings for ZIP %s via %s",
                index,
                total_zips,
                zip_code,
                source_tag,
            )
            max_results_arg = max_per_zip if max_per_zip and max_per_zip > 0 else None

            if use_realestateapi:
                listings = loader.fetch_sold_with_details(
                    zip_code=zip_code,
                    days_back=days_back,
                    max_results=max_results_arg,
                    include_mls_detail=True,
                )
            else:
                scraper_max = max_results_arg if max_results_arg else 10000
                listings = scraper.fetch_sold_with_details(
                    zip_code=zip_code,
                    days_back=days_back,
                    max_results=scraper_max,
                    fetch_details=True
                )

        listings, sanitize_stats = _sanitize_fetched_listings(
            listings,
            days_back=days_back,
        )
        logger.info(
            "[%s/%s] 🧼 ZIP %s sanitized -> raw=%s | kept=%s | duplicates=%s | stale=%s | missing_sale=%s",
            index,
            total_zips,
            zip_code,
            sanitize_stats["raw_count"],
            sanitize_stats["unique_count"],
            sanitize_stats["duplicates"],
            sanitize_stats["stale"],
            sanitize_stats["missing_sale"],
        )

        if use_realestateapi:
            max_cap = settings.realestateapi_max_results_per_zip or 0
            if max_cap and sanitize_stats["unique_count"] >= max_cap:
                raise RuntimeError(
                    f"ZIP {zip_code} returned {sanitize_stats['unique_count']} usable listings, "
                    f"which meets/exceeds the configured cap ({max_cap}). "
                    "Refine the query (smaller window, county subset) or raise the cap explicitly."
                )

        if not listings:
            logger.warning(
                "[%s/%s] ⚠️ ZIP %s produced no usable sold listings after sanitization",
                index,
                total_zips,
                zip_code,
            )
            return []

        # Calculate DOM metrics
        listings = analyzer.calculate_dom_metrics(listings)

        dom_cutoff = getattr(settings, "dom_regression_dom_cutoff", None)
        dom_min = getattr(settings, "dom_regression_min_dom", 0) or 0
        if dom_cutoff and dom_cutoff > 0:
            filtered_listings: List[Dict[str, Any]] = []
            dom_filtered = 0
            dom_low_filtered = 0
            dom_high_filtered = 0
            for item in listings:
                dom_value = item.get("dom_to_pending")
                if dom_value is None:
                    dom_filtered += 1
                    continue
                if dom_value < dom_min:
                    dom_filtered += 1
                    dom_low_filtered += 1
                    continue
                if dom_cutoff and dom_value > dom_cutoff:
                    dom_filtered += 1
                    dom_high_filtered += 1
                    continue
                filtered_listings.append(item)
            logger.info(
                "[%s/%s] 🧮 DOM filter -> kept=%s removed=%s (min=%s days, cutoff=%s days, under_min=%s, over_max=%s)",
                index,
                total_zips,
                len(filtered_listings),
                dom_filtered,
                dom_min,
                dom_cutoff,
                dom_low_filtered,
                dom_high_filtered,
            )
            listings = filtered_listings
            if not listings:
                logger.warning(
                    "[%s/%s] ⚠️ DOM filter removed all listings for ZIP %s; retaining heuristic-only view",
                    index,
                    total_zips,
                    zip_code,
                )
                return []

        # Extract features from descriptions (with batching)
        listings = extractor.batch_extract_features(listings, batch_size=15)

        # Cache results
        if use_cache and not loaded_from_cache:
            save_cached_listings(zip_code, days_back, listings, source_tag, max_per_zip)
        elif use_cache and loaded_from_cache:
            # Overwrite cache with refreshed DOM/features if they were missing previously
            save_cached_listings(zip_code, days_back, listings, source_tag, max_per_zip)

        duration = time.perf_counter() - zip_start
        logger.info(
            "[%s/%s] ✅ ZIP %s collected %s listings via %s in %.1fs",
            index,
            total_zips,
            zip_code,
            len(listings),
            source_tag,
            duration,
        )
        return listings

    # Process ZIPs sequentially to avoid rate limits / request caps
    for idx, zip_code in enumerate(zip_codes, 1):
        try:
            listings = process_zip(zip_code, idx)
        except Exception:
            logger.exception("[%s/%s] ❌ Failed while processing ZIP %s", idx, total_zips, zip_code)
            raise
        all_listings.extend(listings)

    collection_duration = time.perf_counter() - collection_start
    logger.info(
        "Completed data collection in %.1fs | Total listings: %s | Average per ZIP: %.1f",
        collection_duration,
        len(all_listings),
        (len(all_listings) / total_zips) if total_zips else 0,
    )

    unique_zips = sorted({
        str(listing.get('zip_code') or listing.get('zipCode') or '')
        for listing in all_listings
        if listing.get('zip_code') or listing.get('zipCode')
    })
    if unique_zips:
        logger.info("Unique ZIPs discovered in fetched data: %s", ', '.join(unique_zips))
    else:
        logger.warning("No ZIP codes discovered in fetched listings")
    return all_listings


def calculate_median_thresholds_by_zip(listings: list) -> dict:
    """
    Calculate median DOM per ZIP code to use as market-relative thresholds.
    Fast sellers = properties with DOM < median DOM for that ZIP (fastest 50%).
    
    Args:
        listings: List of listings with dom_to_pending and zip_code
        
    Returns:
        Dictionary mapping zip_code -> median_dom threshold
    """
    from collections import defaultdict
    import statistics
    
    # Group DOM values by ZIP code
    dom_by_zip = defaultdict(list)
    
    for listing in listings:
        dom = listing.get('dom_to_pending')
        # Try multiple ways to get ZIP code
        zip_code = listing.get('zip_code') or listing.get('zipCode')
        if not zip_code and listing.get('address'):
            # Try to extract from address (last 5-digit sequence)
            import re
            zip_match = re.search(r'\b(\d{5})\b', listing.get('address', ''))
            zip_code = zip_match.group(1) if zip_match else None
        
        if dom is not None and dom >= 0 and zip_code:
            dom_by_zip[str(zip_code)].append(dom)
    
    # Calculate median per ZIP
    thresholds_by_zip = {}
    for zip_code, doms in dom_by_zip.items():
        if len(doms) >= 3:  # Need at least 3 samples for meaningful median
            median_dom = statistics.median(doms)
            thresholds_by_zip[zip_code] = median_dom
            logger.info(f"ZIP {zip_code}: median DOM = {median_dom:.1f} days (threshold for fastest 50%)")
        else:
            # If insufficient data, use overall median as fallback
            all_doms = [d for doms_list in dom_by_zip.values() for d in doms_list]
            if all_doms:
                thresholds_by_zip[zip_code] = statistics.median(all_doms)
                logger.warning(f"ZIP {zip_code}: insufficient data ({len(doms)} samples), using overall median {thresholds_by_zip[zip_code]:.1f} days")
    
    if not thresholds_by_zip:
        # Fallback: use overall median if no ZIP-specific data
        all_doms = [l.get('dom_to_pending') for l in listings if l.get('dom_to_pending') is not None and l.get('dom_to_pending') >= 0]
        if all_doms:
            overall_median = statistics.median(all_doms)
            logger.warning(f"No ZIP-specific thresholds calculated, using overall median: {overall_median:.1f} days")
            # Apply to all ZIPs found
            for zip_code in set(l.get('zip_code') or 'unknown' for l in listings):
                thresholds_by_zip[zip_code] = overall_median
    
    return thresholds_by_zip


def prepare_targets(listings: list, thresholds_by_zip: dict) -> tuple:
    """
    Prepare target variables for training using market-relative thresholds.
    
    Args:
        listings: List of listings with dom_to_pending and zip_code
        thresholds_by_zip: Dictionary mapping zip_code -> median_dom threshold
        
    Returns:
        Tuple of (y_fast_seller, y_dom, y_dom_bucket, valid_indices)
    """
    y_fast_seller = []
    y_dom = []
    y_dom_bucket = []
    valid_indices = []
    
    # Get overall median as fallback for listings without ZIP match
    import statistics
    from collections import defaultdict
    all_doms = [l.get('dom_to_pending') for l in listings if l.get('dom_to_pending') is not None and l.get('dom_to_pending') >= 0]
    overall_median = statistics.median(all_doms) if all_doms else 30.0
    
    # Prepare DOM distributions per ZIP for quantile-based smoothing
    doms_by_zip: Dict[str, List[float]] = defaultdict(list)
    for listing in listings:
        dom = listing.get('dom_to_pending')
        if dom is None or dom < 0:
            continue
        zip_code = listing.get('zip_code') or listing.get('zipCode')
        if not zip_code and listing.get('address'):
            import re
            zip_match = re.search(r'\b(\d{5})\b', listing.get('address', ''))
            zip_code = zip_match.group(1) if zip_match else None
        if zip_code:
            doms_by_zip[str(zip_code)].append(dom)
    
    import numpy as np
    if all_doms:
        global_lower = float(np.percentile(all_doms, 5))
        global_upper = float(np.percentile(all_doms, 95))
    else:
        global_lower, global_upper = 0.0, 90.0
    
    dom_quantiles_by_zip: Dict[str, tuple[float, float]] = {}
    for zip_code, dom_list in doms_by_zip.items():
        if len(dom_list) >= 8:
            dom_quantiles_by_zip[zip_code] = (
                float(np.percentile(dom_list, 5)),
                float(np.percentile(dom_list, 95))
            )
        else:
            dom_quantiles_by_zip[zip_code] = (global_lower, global_upper)
    
    bucket_edges = [7, 14, 30, 60]
    bucket_labels = ["0-7", "8-14", "15-30", "31-60", "60+"]

    for i, listing in enumerate(listings):
        dom = listing.get('dom_to_pending')
        # Try multiple ways to get ZIP code
        zip_code = listing.get('zip_code') or listing.get('zipCode')
        if not zip_code and listing.get('address'):
            # Try to extract from address (last 5-digit sequence)
            import re
            zip_match = re.search(r'\b(\d{5})\b', listing.get('address', ''))
            zip_code = zip_match.group(1) if zip_match else None
        
        if dom is not None and dom >= 0:
            # Get ZIP-specific threshold, fallback to overall median
            threshold = thresholds_by_zip.get(str(zip_code), overall_median) if zip_code else overall_median
            low_q, high_q = dom_quantiles_by_zip.get(str(zip_code), (global_lower, global_upper))
            dom_clipped = float(np.clip(dom, low_q, high_q))
            
            # Fast seller = DOM < median for that ZIP (fastest 50%)
            y_fast_seller.append(1 if dom < threshold else 0)
            y_dom.append(dom_clipped)
            bucket_index = 0
            if dom_clipped is not None:
                for idx, edge in enumerate(bucket_edges):
                    if dom_clipped <= edge:
                        bucket_index = idx
                        break
                else:
                    bucket_index = len(bucket_edges)
            y_dom_bucket.append(bucket_labels[bucket_index])
            valid_indices.append(i)
            listing['dom_bucket'] = bucket_labels[bucket_index]
            listing['fast_seller_label'] = y_fast_seller[-1]
 
    logger.info(f"Valid samples: {len(valid_indices)}/{len(listings)}")
    fast_count = sum(y_fast_seller)
    logger.info(f"Fast sellers: {fast_count} ({fast_count/len(y_fast_seller)*100:.1f}% - should be ~50% with median threshold)")
    
    return pd.Series(y_fast_seller), pd.Series(y_dom), pd.Series(y_dom_bucket, dtype="category"), valid_indices


def engineer_features(listings: list, valid_indices: list) -> pd.DataFrame:
    """Engineer features for ML model."""
    engineer = FeatureEngineer()
    
    # Filter to valid listings
    valid_listings = [listings[i] for i in valid_indices]
    
    # Gather LLM extracted features if present
    extracted_features_list = [
        listing.get('extracted_features', {}) for listing in valid_listings
    ]
    
    # Engineer features using the fast-seller specific pipeline
    df = engineer.engineer_features_for_fast_seller(
        valid_listings,
        market_context=None,
        extracted_features_list=extracted_features_list
    )
    
    # Drop non-numeric columns that XGBoost cannot handle
    # _original_property is stored for extraction but not needed for training
    columns_to_drop = ['_original_property']
    
    # Also drop any other object-type columns (except target variables)
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['y_fast_seller', 'y_dom']:
            columns_to_drop.append(col)
    
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    logger.info(f"Engineered {len(df.columns)} features (dropped {len(columns_to_drop)} non-numeric columns)")
    return df


def train_fast_seller_model(
    zip_codes: list,
    days_back: int = 180,
    max_per_zip: Optional[int] = None,
    hyperparameter_tuning: bool = True,
    save_models: bool = True
) -> dict:
    """
    Main training function.
    Uses market-relative thresholds (median DOM per ZIP) to identify fast sellers.
    """
    import statistics

    logger.info("="*80)
    logger.info("FAST SELLER MODEL TRAINING")
    logger.info("="*80)
    logger.info("Using market-relative thresholds: median DOM per ZIP (fastest 50%)")
    logger.info("Requested ZIPs: %s", ', '.join(zip_codes))

    # Fetch data
    data_start = time.perf_counter()
    listings = fetch_sold_listings_with_features(zip_codes, days_back, max_per_zip)
    logger.info(
        "Data fetch complete in %.1fs (listings=%s)",
        time.perf_counter() - data_start,
        len(listings),
    )

    if len(listings) < 50:
        raise ValueError(f"Insufficient data: {len(listings)} listings. Need at least 50.")

    # Calculate market-relative thresholds (median DOM per ZIP)
    logger.info("Calculating market-relative thresholds per ZIP...")
    thresholds_start = time.perf_counter()
    thresholds_by_zip = calculate_median_thresholds_by_zip(listings)
    logger.info(
        "Calculated thresholds for %s ZIP codes in %.1fs",
        len(thresholds_by_zip),
        time.perf_counter() - thresholds_start,
    )

    # Prepare targets using ZIP-specific thresholds
    logger.info("Preparing target variables (fast seller label + DOM)...")
    targets_start = time.perf_counter()
    y_fast_seller, y_dom, y_dom_bucket, valid_indices = prepare_targets(listings, thresholds_by_zip)
    bucket_distribution = {str(k): int(v) for k, v in y_dom_bucket.value_counts(dropna=False).items()}
    logger.info("DOM bucket distribution: %s", bucket_distribution)
    logger.info(
        "Targets prepared in %.1fs | valid samples=%s | fast seller positives=%s",
        time.perf_counter() - targets_start,
        len(valid_indices),
        int(y_fast_seller.sum()),
    )
    metro_zips_setting = getattr(settings, "dom_regression_metro_zips", [])
    metro_zips = {str(z).strip() for z in metro_zips_setting if str(z).strip()}
    dom_regression_mask_values: List[bool] = []
    for idx in valid_indices:
        listing = listings[idx]
        zip_code = listing.get('zip_code') or listing.get('zipCode')
        if not zip_code and listing.get('address'):
            import re
            zip_match = re.search(r'\b(\d{5})\b', listing.get('address', ''))
            zip_code = zip_match.group(1) if zip_match else None
        dom_regression_mask_values.append(str(zip_code) in metro_zips if zip_code else False)
    dom_regression_mask = pd.Series(dom_regression_mask_values, index=y_dom.index, dtype=bool)
    metro_sample_count = int(dom_regression_mask.sum())
    metro_ratio = float(metro_sample_count / len(dom_regression_mask)) if len(dom_regression_mask) else 0.0
    logger.info(
        "DOM regression metro coverage: %s samples (%.1f%% of %s)",
        metro_sample_count,
        metro_ratio * 100,
        len(dom_regression_mask),
    )
    from collections import defaultdict
    dom_values_by_zip = defaultdict(list)
    for idx in valid_indices:
        listing = listings[idx]
        dom_value = listing.get('dom_to_pending')
        if dom_value is None or dom_value < 0:
            continue
        zip_code = listing.get('zip_code') or listing.get('zipCode')
        if not zip_code and listing.get('address'):
            import re
            zip_match = re.search(r'\b(\d{5})\b', listing.get('address', ''))
            zip_code = zip_match.group(1) if zip_match else None
        if zip_code:
            dom_values_by_zip[str(zip_code)].append(dom_value)
    dom_stats_by_zip = {}
    for zip_code, dom_list in dom_values_by_zip.items():
        arr = np.array(dom_list)
        dom_stats_by_zip[zip_code] = {
            'count': int(len(arr)),
            'median': float(np.median(arr)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90))
        }

    if len(valid_indices) < 50:
        raise ValueError(f"Insufficient valid samples: {len(valid_indices)}. Need at least 50.")

    # Engineer features
    feature_start = time.perf_counter()
    logger.info("Engineering feature matrix for %s samples...", len(valid_indices))
    X = engineer_features(listings, valid_indices)
    X = X.reset_index(drop=True)
    y_fast_seller = y_fast_seller.reset_index(drop=True)
    y_dom = y_dom.reset_index(drop=True)
    dom_regression_mask = dom_regression_mask.reset_index(drop=True)
    feature_duration = time.perf_counter() - feature_start
    logger.info(
        "Feature engineering complete in %.1fs | shape=(%s, %s)",
        feature_duration,
        len(X),
        len(X.columns),
    )

    if len(X) != len(y_fast_seller):
        raise ValueError(f"Feature count mismatch: X={len(X)}, y={len(y_fast_seller)}")

    # Train model
    model = FastSellerModel(model_dir="models")

    # Store market-relative thresholds in model metadata
    model.model_metadata['thresholds_by_zip'] = thresholds_by_zip
    model.model_metadata['threshold_method'] = 'median_dom_per_zip'
    model.model_metadata['threshold_description'] = 'Fast sellers = DOM < median DOM for that ZIP (fastest 50%)'

    model.model_metadata['dom_stats_by_zip'] = dom_stats_by_zip
    model.model_metadata['dom_global_median'] = float(np.median(y_dom))
    model.model_metadata['dom_global_p75'] = float(np.percentile(y_dom, 75))
    model.model_metadata['dom_global_p90'] = float(np.percentile(y_dom, 90))
    model.model_metadata['dom_strategy'] = 'zip_median_heuristic'
    model.model_metadata['dom_bucket_counts'] = bucket_distribution
    model.model_metadata['dom_regression_scope'] = 'metro_only' if metro_zips else 'all_zips'
    model.model_metadata['dom_regression_zip_whitelist'] = sorted(metro_zips)
    model.model_metadata['dom_regression_mask_positive_count'] = metro_sample_count
    model.model_metadata['dom_regression_mask_positive_ratio'] = metro_ratio
    model.model_metadata['dom_regression_min_dom'] = getattr(settings, 'dom_regression_min_dom', None)

    logger.info(
        "Starting model training (hyperparameter_tuning=%s, save_models=%s)...",
        hyperparameter_tuning,
        save_models,
    )
    dom_regression_enabled = getattr(settings, 'dom_regression_enabled', True)
    dom_regression_min_samples = getattr(settings, 'dom_regression_min_samples', 10000)
    dom_regression_max_mae = getattr(settings, 'dom_regression_max_mae', None)
    dom_cutoff = getattr(settings, 'dom_regression_dom_cutoff', None)
    model.model_metadata['dom_regression_enabled'] = dom_regression_enabled
    model.model_metadata['dom_regression_cutoff'] = dom_cutoff
    model.model_metadata['dom_regression_total_labeled'] = int(len(y_dom))
    model.model_metadata['dom_regression_max_mae'] = dom_regression_max_mae
    training_start = time.perf_counter()
    metrics = model.train(
        X=X,
        y_fast_seller=y_fast_seller,
        y_dom=y_dom,
        hyperparameter_tuning=hyperparameter_tuning,
        enable_dom_regression=dom_regression_enabled,
        dom_regression_min_samples=dom_regression_min_samples,
        dom_regression_max_mae=dom_regression_max_mae,
        dom_regression_mask=dom_regression_mask,
    )
    training_duration = time.perf_counter() - training_start
    logger.info("Model training finished in %.1fs", training_duration)

    if save_models:
        model.save()
        logger.info("Models persisted to %s", model.model_dir)

    # Build micro-market summary tables
    micro_market_summary = build_micro_market_summary(listings)
    summary_rows = len(micro_market_summary)
    model.model_metadata['micro_market_summary_rows'] = summary_rows
    if summary_rows:
        summary_path = model.model_dir / "micro_market_summary.csv"
        micro_market_summary.to_csv(summary_path, index=False)
        model.model_metadata['micro_market_summary_path'] = str(summary_path)
        logger.info("Saved micro-market summary to %s (%s rows)", summary_path, summary_rows)
        logger.info("Top micro-market configurations:\n%s", micro_market_summary.head(5).to_string(index=False))
    else:
        logger.warning("Micro-market summary is empty; insufficient labeled data")

    # Feature importance
    feature_importance = model.get_feature_importance('classifier')
    logger.info("\nTop 10 Features (Classifier):")
    for i, (feat, imp) in enumerate(list(feature_importance.items())[:10], 1):
        logger.info(f"  {i}. {feat}: {imp:.4f}")

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)

    return {
        'metrics': metrics,
        'feature_importance': feature_importance,
        'training_samples': len(X)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fast seller prediction model')
    parser.add_argument('--zip-codes', nargs='+', default=None,
                        help='ZIP codes to train on')
    parser.add_argument('--days-back', type=int, default=365,  # 12 months
                        help='Days of historical data to fetch')
    parser.add_argument('--max-per-zip', type=int, default=0,
                        help='Maximum listings per ZIP code (0 = no limit)')
    # Note: Threshold is now calculated automatically as median DOM per ZIP (market-relative)
    parser.add_argument('--use-similar-zips', action='store_true', default=True,
                        help='Use similar ZIPs to 27410 (Option A)')
    parser.add_argument('--triad', action='store_true',
                        help='Use full Triad market ZIPs (Greensboro, Winston-Salem, High Point)')
    parser.add_argument('--triad-counties', action='store_true',
                        help='Discover ZIPs dynamically across Triad counties')
    parser.add_argument('--counties', nargs='+',
                        help='Explicit list of counties to discover ZIPs for (defaults to state NC)')
    parser.add_argument('--state', default='NC',
                        help='State abbreviation for county discovery (default: NC)')
    parser.add_argument('--all-zips', action='store_true',
                        help='Use all Greensboro ZIPs (Option B)')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                        help='Enable hyperparameter tuning')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save models')
    
    args = parser.parse_args()
    
    # Determine ZIP codes to use
    if args.zip_codes:
        zip_codes = args.zip_codes
    elif args.counties:
        county_list = [c.strip() for c in args.counties if c.strip()]
        if not county_list:
            raise ValueError("--counties provided but no valid county names parsed")
        zip_codes = get_zips_for_counties(county_list, state=args.state)
        if not zip_codes:
            raise ValueError(f"Failed to discover ZIPs for counties: {', '.join(county_list)}")
        logger.info(
            "Using counties (%s) -> %s ZIPs",
            ', '.join(county_list),
            len(zip_codes)
        )
        logger.info("Discovered ZIPs: %s", ', '.join(zip_codes))
    elif args.triad_counties:
        zip_codes = get_zips_for_counties(TRIAD_COUNTIES, state=args.state)
        if not zip_codes:
            raise ValueError("Failed to discover ZIPs for Triad counties")
        logger.info(
            "Using Triad counties (%s) -> %s ZIPs",
            ', '.join(TRIAD_COUNTIES),
            len(zip_codes)
        )
        logger.info("Discovered ZIPs: %s", ', '.join(zip_codes))
    elif args.triad:
        zip_codes = get_triad_zips()
        logger.info(f"Using full Triad ZIP list ({len(zip_codes)} ZIPs)")
        logger.info("Triad ZIPs: %s", ', '.join(zip_codes))
    elif args.all_zips:
        zip_codes = ALL_GREENSBORO_ZIPS
        logger.info(f"Using Option B: All Greensboro ZIPs ({len(zip_codes)} ZIPs)")
    else:
        zip_codes = get_training_zips(use_similar_only=True)
        logger.info(f"Using Option A: Similar ZIPs to 27410 ({len(zip_codes)} ZIPs)")
        logger.info(f"ZIP codes: {', '.join(zip_codes)}")
    
    try:
        results = train_fast_seller_model(
            zip_codes=zip_codes,
            days_back=args.days_back,
            max_per_zip=None if args.max_per_zip == 0 else args.max_per_zip,
            hyperparameter_tuning=args.hyperparameter_tuning,
            save_models=not args.no_save
        )
        
        clf_auc = results['metrics']['classifier'].get('test_auc')
        if clf_auc is not None:
            logger.info(f"\nClassifier AUC: {clf_auc:.3f}")
        reg_metrics = results['metrics']['regressor']
        if reg_metrics.get('test_mape') is not None:
            logger.info(f"Regressor MAPE: {reg_metrics['test_mape']:.2f}%")
        else:
            logger.info("DOM predictions use median-by-ZIP heuristic (no MAPE computed)")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
