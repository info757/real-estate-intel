"""
Training script for fast seller prediction model.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.data_collectors.safe_listings_scraper import SafeListingsScraper
from backend.analyzers.sold_listings_analyzer import SoldListingsAnalyzer
from backend.ml.feature_extractor import FeatureExtractor
from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.fast_seller_model import FastSellerModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_sold_listings_with_features(
    zip_codes: list,
    days_back: int = 180,
    max_per_zip: int = 100
) -> list:
    """Fetch sold listings, extract features, and calculate DOM metrics."""
    scraper = SafeListingsScraper()
    analyzer = SoldListingsAnalyzer()
    extractor = FeatureExtractor()
    
    all_listings = []
    
    for zip_code in zip_codes:
        logger.info(f"Fetching sold listings for ZIP {zip_code}...")
        listings = scraper.fetch_sold_with_details(
            zip_code=zip_code,
            days_back=days_back,
            max_results=max_per_zip,
            fetch_details=True
        )
        
        # Calculate DOM metrics
        listings = analyzer.calculate_dom_metrics(listings)
        
        # Extract features from descriptions
        listings = extractor.batch_extract_features(listings)
        
        all_listings.extend(listings)
        logger.info(f"Collected {len(listings)} listings from ZIP {zip_code}")
    
    logger.info(f"Total listings collected: {len(all_listings)}")
    return all_listings


def prepare_targets(listings: list, fast_threshold: int = 14) -> tuple:
    """Prepare target variables for training."""
    y_fast_seller = []
    y_dom = []
    valid_indices = []
    
    for i, listing in enumerate(listings):
        dom = listing.get('dom_to_pending')
        
        if dom is not None and dom >= 0:
            y_fast_seller.append(1 if dom <= fast_threshold else 0)
            y_dom.append(dom)
            valid_indices.append(i)
    
    logger.info(f"Valid samples: {len(valid_indices)}/{len(listings)}")
    logger.info(f"Fast sellers: {sum(y_fast_seller)} ({sum(y_fast_seller)/len(y_fast_seller)*100:.1f}%)")
    
    return pd.Series(y_fast_seller), pd.Series(y_dom), valid_indices


def engineer_features(listings: list, valid_indices: list) -> pd.DataFrame:
    """Engineer features for ML model."""
    engineer = FeatureEngineer()
    
    # Filter to valid listings
    valid_listings = [listings[i] for i in valid_indices]
    
    # Engineer features
    df = engineer.engineer_features(valid_listings)
    
    # Add LLM-extracted features as binary flags
    # Get all unique features across listings
    all_features = set()
    for listing in valid_listings:
        features = listing.get('extracted_features', {})
        all_features.update(features.get('interior', []))
        all_features.update(features.get('exterior', []))
        all_features.update(features.get('upgrades', []))
    
    # Create binary feature columns
    feature_columns = {}
    for feature in all_features:
        # Clean feature name for column name
        col_name = f"has_{feature.lower().replace(' ', '_').replace('-', '_')[:30]}"
        feature_columns[col_name] = []
    
    for listing in valid_listings:
        features = listing.get('extracted_features', {})
        listing_features = set()
        listing_features.update(features.get('interior', []))
        listing_features.update(features.get('exterior', []))
        listing_features.update(features.get('upgrades', []))
        
        for feature in all_features:
            col_name = f"has_{feature.lower().replace(' ', '_').replace('-', '_')[:30]}"
            feature_columns[col_name].append(1 if feature in listing_features else 0)
    
    # Add feature columns to DataFrame
    for col_name, values in feature_columns.items():
        df[col_name] = values
    
    logger.info(f"Engineered {len(df.columns)} features")
    return df


def train_fast_seller_model(
    zip_codes: list,
    days_back: int = 180,
    max_per_zip: int = 100,
    fast_threshold: int = 14,
    hyperparameter_tuning: bool = True,
    save_models: bool = True
) -> dict:
    """Main training function."""
    logger.info("="*80)
    logger.info("FAST SELLER MODEL TRAINING")
    logger.info("="*80)
    
    # Fetch data
    listings = fetch_sold_listings_with_features(zip_codes, days_back, max_per_zip)
    
    if len(listings) < 50:
        raise ValueError(f"Insufficient data: {len(listings)} listings. Need at least 50.")
    
    # Prepare targets
    y_fast_seller, y_dom, valid_indices = prepare_targets(listings, fast_threshold)
    
    if len(valid_indices) < 50:
        raise ValueError(f"Insufficient valid samples: {len(valid_indices)}. Need at least 50.")
    
    # Engineer features
    X = engineer_features(listings, valid_indices)
    
    if len(X) != len(y_fast_seller):
        raise ValueError(f"Feature count mismatch: X={len(X)}, y={len(y_fast_seller)}")
    
    # Train model
    model = FastSellerModel(model_dir="models")
    metrics = model.train(
        X=X,
        y_fast_seller=y_fast_seller,
        y_dom=y_dom,
        fast_seller_threshold=fast_threshold,
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
    parser.add_argument('--zip-codes', nargs='+', default=['27410'],
                        help='ZIP codes to train on')
    parser.add_argument('--days-back', type=int, default=180,
                        help='Days of historical data to fetch')
    parser.add_argument('--max-per-zip', type=int, default=100,
                        help='Maximum listings per ZIP code')
    parser.add_argument('--fast-threshold', type=int, default=14,
                        help='DOM threshold for fast sellers')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                        help='Enable hyperparameter tuning')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save models')
    
    args = parser.parse_args()
    
    try:
        results = train_fast_seller_model(
            zip_codes=args.zip_codes,
            days_back=args.days_back,
            max_per_zip=args.max_per_zip,
            fast_threshold=args.fast_threshold,
            hyperparameter_tuning=args.hyperparameter_tuning,
            save_models=not args.no_save
        )
        
        logger.info(f"\nClassifier AUC: {results['metrics']['classifier']['test_auc']:.3f}")
        logger.info(f"Regressor MAPE: {results['metrics']['regressor']['test_mape']:.2f}%")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
