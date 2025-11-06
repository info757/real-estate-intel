"""
Merge cached data from all ZIP codes (original + new) and retrain the model.
This should be run after both the original training and new ZIP collection complete.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path
import json
from backend.ml.train_fast_seller_model import (
    load_cached_listings, train_fast_seller_model
)
from scripts.identify_similar_zips import ALL_GREENSBORO_ZIPS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_all_cached_data(days_back: int = 365) -> list:
    """
    Load and merge all cached listing data from all ZIP codes.
    
    Args:
        days_back: Days back used for caching (must match cache keys)
        
    Returns:
        Merged list of all listings
    """
    all_listings = []
    cached_zips = []
    missing_zips = []
    
    logger.info("="*80)
    logger.info("MERGING CACHED DATA FROM ALL ZIP CODES")
    logger.info("="*80)
    
    for zip_code in sorted(ALL_GREENSBORO_ZIPS):
        cached = load_cached_listings(zip_code, days_back)
        if cached:
            all_listings.extend(cached)
            cached_zips.append(zip_code)
            logger.info(f"✅ ZIP {zip_code}: {len(cached)} listings")
        else:
            missing_zips.append(zip_code)
            logger.warning(f"⚠️  ZIP {zip_code}: No cached data")
    
    logger.info("="*80)
    logger.info(f"TOTAL: {len(all_listings)} listings from {len(cached_zips)} ZIP codes")
    if missing_zips:
        logger.warning(f"Missing data for {len(missing_zips)} ZIP codes: {', '.join(missing_zips)}")
    logger.info("="*80)
    
    return all_listings


def retrain_with_merged_data(
    days_back: int = 365,
    hyperparameter_tuning: bool = True
):
    """
    Retrain the fast-seller model with merged data from all ZIP codes.
    
    Args:
        days_back: Days back used for caching
        hyperparameter_tuning: Whether to tune hyperparameters
    """
    logger.info("="*80)
    logger.info("RETRAINING MODEL WITH MERGED DATA")
    logger.info("="*80)
    
    # Merge all cached data
    all_listings = merge_all_cached_data(days_back)
    
    if len(all_listings) < 50:
        raise ValueError(f"Insufficient data: {len(all_listings)} listings. Need at least 50.")
    
    # Get all ZIP codes that have data
    zip_codes_with_data = sorted(set(l.get('zip_code') or 'unknown' for l in all_listings))
    logger.info(f"ZIP codes with data: {', '.join(zip_codes_with_data)}")
    
    # Retrain model
    # Note: train_fast_seller_model will fetch data, but we've already merged it
    # We need to use the merged data directly. Let me check the function signature...
    
    # Actually, train_fast_seller_model calls fetch_sold_listings_with_features
    # which will use cached data. So we can just call it with all ZIP codes
    # and it will use the cached data we just verified exists.
    
    results = train_fast_seller_model(
        zip_codes=ALL_GREENSBORO_ZIPS,  # Use all ZIP codes, cached data will be loaded
        days_back=days_back,
        max_per_zip=200,
        hyperparameter_tuning=hyperparameter_tuning,
        save_models=True
    )
    
    logger.info("="*80)
    logger.info("RETRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Classifier AUC: {results['metrics']['classifier']['test_auc']:.3f}")
    logger.info(f"Regressor MAPE: {results['metrics']['regressor']['test_mape']:.2f}%")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge cached data and retrain model')
    parser.add_argument('--days-back', type=int, default=365,
                        help='Days back used for caching')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                        help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    try:
        results = retrain_with_merged_data(
            days_back=args.days_back,
            hyperparameter_tuning=args.hyperparameter_tuning
        )
        logger.info("\n✅ Retraining completed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

