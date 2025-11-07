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
from typing import List, Dict, Any, Optional
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.data_collectors.safe_listings_scraper import SafeListingsScraper
from backend.analyzers.sold_listings_analyzer import SoldListingsAnalyzer
from backend.ml.feature_extractor import FeatureExtractor
from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.fast_seller_model import FastSellerModel
from scripts.identify_similar_zips import get_training_zips, SIMILAR_ZIPS, ALL_GREENSBORO_ZIPS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def get_cache_path(zip_code: str, days_back: int) -> Path:
    """Get cache file path for a ZIP code."""
    cache_dir = Path("cache/listings")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"listings_{zip_code}_{days_back}days.json"

def load_cached_listings(zip_code: str, days_back: int) -> Optional[List[Dict[str, Any]]]:
    """Load cached listings if available."""
    cache_path = get_cache_path(zip_code, days_back)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache for {zip_code}: {e}")
    return None

def save_cached_listings(zip_code: str, days_back: int, listings: List[Dict[str, Any]]):
    """Save listings to cache."""
    cache_path = get_cache_path(zip_code, days_back)
    try:
        with open(cache_path, 'w') as f:
            json.dump(listings, f, indent=2, default=str)
        logger.info(f"Cached {len(listings)} listings for {zip_code}")
    except Exception as e:
        logger.warning(f"Failed to save cache for {zip_code}: {e}")


def fetch_sold_listings_with_features(
    zip_codes: list,
    days_back: int = 365,
    max_per_zip: int = 200,
    use_cache: bool = True,
    parallel: bool = False  # Sequential to avoid rate limits
) -> list:
    """Fetch sold listings, extract features, and calculate DOM metrics."""
    scraper = SafeListingsScraper()
    analyzer = SoldListingsAnalyzer()
    extractor = FeatureExtractor()
    
    all_listings = []
    
    def process_zip(zip_code: str) -> List[Dict[str, Any]]:
        """Process a single ZIP code."""
        # Check cache first
        if use_cache:
            cached = load_cached_listings(zip_code, days_back)
            if cached:
                logger.info(f"✅ Loaded {len(cached)} cached listings for ZIP {zip_code}")
                return cached
        
        logger.info(f"Fetching sold listings for ZIP {zip_code}...")
        listings = scraper.fetch_sold_with_details(
            zip_code=zip_code,
            days_back=days_back,
            max_results=max_per_zip,
            fetch_details=True
        )
        
        # Calculate DOM metrics
        listings = analyzer.calculate_dom_metrics(listings)
        
        # Extract features from descriptions (with batching)
        listings = extractor.batch_extract_features(listings, batch_size=15)
        
        # Cache results
        if use_cache:
            save_cached_listings(zip_code, days_back, listings)
        
        logger.info(f"✅ Collected {len(listings)} listings from ZIP {zip_code}")
        return listings
    
    # Process ZIPs sequentially to avoid rate limits
    if False and parallel and len(zip_codes) > 1:  # Disabled parallel to avoid rate limits
        logger.info(f"Processing {len(zip_codes)} ZIPs in parallel...")
        with ThreadPoolExecutor(max_workers=3) as executor:  # Limit to 3 to respect rate limits
            futures = {executor.submit(process_zip, zip_code): zip_code for zip_code in zip_codes}
            for future in as_completed(futures):
                zip_code = futures[future]
                try:
                    listings = future.result()
                    all_listings.extend(listings)
                except Exception as e:
                    logger.error(f"Error processing ZIP {zip_code}: {e}")
    else:
        for zip_code in zip_codes:
            listings = process_zip(zip_code)
            all_listings.extend(listings)
    
    logger.info(f"✅ Total listings collected: {len(all_listings)}")
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
        Tuple of (y_fast_seller, y_dom, valid_indices)
    """
    y_fast_seller = []
    y_dom = []
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
            valid_indices.append(i)
    
    logger.info(f"Valid samples: {len(valid_indices)}/{len(listings)}")
    fast_count = sum(y_fast_seller)
    logger.info(f"Fast sellers: {fast_count} ({fast_count/len(y_fast_seller)*100:.1f}% - should be ~50% with median threshold)")
    
    return pd.Series(y_fast_seller), pd.Series(y_dom), valid_indices


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
    max_per_zip: int = 100,
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
    
    # Fetch data
    listings = fetch_sold_listings_with_features(zip_codes, days_back, max_per_zip)
    
    if len(listings) < 50:
        raise ValueError(f"Insufficient data: {len(listings)} listings. Need at least 50.")
    
    # Calculate market-relative thresholds (median DOM per ZIP)
    logger.info("Calculating market-relative thresholds per ZIP...")
    thresholds_by_zip = calculate_median_thresholds_by_zip(listings)
    logger.info(f"Calculated thresholds for {len(thresholds_by_zip)} ZIP codes")
    
    # Prepare targets using ZIP-specific thresholds
    y_fast_seller, y_dom, valid_indices = prepare_targets(listings, thresholds_by_zip)
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
    X = engineer_features(listings, valid_indices)
    
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

    metrics = model.train(
        X=X,
        y_fast_seller=y_fast_seller,
        y_dom=y_dom,
        hyperparameter_tuning=hyperparameter_tuning
    )
    
    if save_models:
        model.save()
    
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
    parser.add_argument('--max-per-zip', type=int, default=200,
                        help='Maximum listings per ZIP code')
    # Note: Threshold is now calculated automatically as median DOM per ZIP (market-relative)
    parser.add_argument('--use-similar-zips', action='store_true', default=True,
                        help='Use similar ZIPs to 27410 (Option A)')
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
            max_per_zip=args.max_per_zip,
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
