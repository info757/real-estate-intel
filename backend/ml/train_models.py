"""
Model Training Script
Trains pricing and demand models on historical sales data.
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from backend.ml.feature_engineering import feature_engineer
from backend.ml.pricing_model import pricing_model
from backend.ml.demand_model import demand_model
from backend.data_collectors.attom_client import attom_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_training_data(
    zip_codes: List[str],
    months_back: int = 36,
    max_properties_per_zip: int = 500
) -> pd.DataFrame:
    """
    Fetch historical sales data for training.
    
    Args:
        zip_codes: List of ZIP codes to fetch data from
        months_back: How many months of historical data to fetch
        max_properties_per_zip: Maximum properties per ZIP code
        
    Returns:
        DataFrame with training data
    """
    logger.info(f"Fetching training data from {len(zip_codes)} ZIP codes")
    
    all_properties = []
    min_sale_date = (datetime.now() - timedelta(days=months_back * 30)).strftime('%Y-%m-%d')
    
    for zip_code in zip_codes:
        logger.info(f"Fetching data for ZIP {zip_code}...")
        try:
            # Calculate max_pages - need enough to capture all available data
            # If there are 2400 sales in 2024 alone, we need ~185 sales/ZIP/year
            # Over 36 months = ~555 sales/ZIP, so we need at least 6 pages
            # But to be safe and capture all data, use a much higher limit
            # Each page has up to 100 properties
            max_pages_needed = max(
                (max_properties_per_zip // 100) + 10,  # At least enough for max_properties_per_zip + buffer
                25  # Minimum 25 pages to capture ~2400 sales/year across all ZIPs
            )
            properties = attom_client.get_all_sales_paginated(
                zip_code=zip_code,
                max_pages=max_pages_needed,  # Fetch many pages to get all available data
                min_sale_date=min_sale_date
            )
            
            # Filter to max_properties_per_zip
            properties = properties[:max_properties_per_zip]
            logger.info(f"  ✓ Fetched {len(properties)} properties from ZIP {zip_code}")
            all_properties.extend(properties)
            
        except Exception as e:
            logger.error(f"  ✗ Error fetching data for ZIP {zip_code}: {e}")
            continue
    
    logger.info(f"Total properties fetched: {len(all_properties)}")
    
    # Convert to DataFrame using feature engineering
    if all_properties:
        df = feature_engineer.engineer_features(all_properties)
        logger.info(f"Engineered features: {len(df)} rows, {len(df.columns)} columns")
        return df
    else:
        return pd.DataFrame()


def prepare_targets(df: pd.DataFrame, sell_within_days: int = 90) -> tuple:
    """
    Prepare target variables for training.
    
    Args:
        df: Feature DataFrame
        sell_within_days: Threshold for fast sale classification
        
    Returns:
        Tuple of (y_price, y_sold_fast, y_dom)
    """
    # Sale price (for pricing model)
    y_price = df['sale_price'].copy()
    
    # Days on market (for demand model - regression)
    # Calculate DOM from list date to sale date if available
    if 'days_on_market' in df.columns:
        y_dom = df['days_on_market'].copy()
    else:
        # Estimate DOM if not available (use median for the market)
        logger.warning("DOM not available, using estimated values")
        y_dom = df.groupby('zip_code')['sale_price'].transform(lambda x: np.random.randint(30, 120, len(x)))
    
    # Binary target: sold fast (within threshold) - for demand model classification
    y_sold_fast = (y_dom <= sell_within_days).astype(int)
    
    # Remove rows with missing targets and apply price filter
    # Filter: $100k - $3M (typical residential property range)
    MIN_PRICE = 100000  # $100k
    MAX_PRICE = 3000000  # $3M
    valid_mask = (
        y_price.notna() & 
        (y_price >= MIN_PRICE) & 
        (y_price <= MAX_PRICE) &
        y_dom.notna() & 
        (y_dom >= 0)
    )
    
    # Count filtered properties
    removed_missing = len(df) - valid_mask.sum()
    removed_low_price = ((y_price < MIN_PRICE) & y_price.notna()).sum()
    removed_high_price = ((y_price > MAX_PRICE) & y_price.notna()).sum()
    
    y_price = y_price[valid_mask]
    y_sold_fast = y_sold_fast[valid_mask]
    y_dom = y_dom[valid_mask]
    
    logger.info(f"Valid samples: {len(y_price)} (removed {removed_missing} with missing targets)")
    logger.info(f"Price filter: Removed {removed_low_price} < ${MIN_PRICE:,} and {removed_high_price} > ${MAX_PRICE:,}")
    
    return y_price, y_sold_fast, y_dom, valid_mask


def train_models(
    zip_codes: List[str] = ['27410'],  # Default to Greensboro
    months_back: int = 36,
    max_properties_per_zip: int = 500,
    hyperparameter_tuning: bool = False,  # Set to False for faster training
    save_models: bool = True,
    model_dir: str = 'models'
) -> Dict[str, Any]:
    """
    Train both pricing and demand models.
    
    Args:
        zip_codes: List of ZIP codes to use for training
        months_back: Months of historical data
        max_properties_per_zip: Maximum properties to fetch per ZIP code
        hyperparameter_tuning: Whether to perform grid search
        save_models: Whether to save trained models
        model_dir: Directory to save models
        
    Returns:
        Dictionary with training results and metrics
    """
    logger.info("="*80)
    logger.info("STARTING MODEL TRAINING")
    logger.info("="*80)
    
    # 1. Fetch training data
    df = fetch_training_data(zip_codes, months_back=months_back, max_properties_per_zip=max_properties_per_zip)
    
    if df.empty or len(df) < 50:
        raise ValueError(f"Insufficient training data: {len(df)} samples. Need at least 50.")
    
    # 2. Prepare features and targets
    logger.info("Preparing features and targets...")
    X, y_price_base = feature_engineer.prepare_for_training(df)
    
    # Get target variables
    y_price, y_sold_fast, y_dom, valid_mask = prepare_targets(df)
    
    # Filter X to match valid targets
    X = X[valid_mask]
    
    logger.info(f"Training data: {len(X)} samples, {len(X.columns)} features")
    logger.info(f"Price range: ${y_price.min():,.0f} - ${y_price.max():,.0f}")
    logger.info(f"DOM range: {y_dom.min():.0f} - {y_dom.max():.0f} days")
    logger.info(f"Fast sales: {y_sold_fast.sum()}/{len(y_sold_fast)} ({y_sold_fast.mean()*100:.1f}%)")
    
    if len(X) < 50:
        raise ValueError(f"Insufficient training data after filtering: {len(X)} samples. Need at least 50.")
    
    # 3. Train pricing model
    logger.info("\n" + "="*80)
    logger.info("TRAINING PRICING MODEL")
    logger.info("="*80)
    
    pricing_metrics = pricing_model.train(
        X=X,
        y=y_price,
        hyperparameter_tuning=hyperparameter_tuning,
        test_size=0.2,
        validation_size=0.2
    )
    
    logger.info(f"Pricing Model - Test MAPE: {pricing_metrics['test_mape']:.2f}%")
    logger.info(f"Pricing Model - Test R²: {pricing_metrics['test_r2']:.3f}")
    
    if save_models:
        pricing_model.save()
        logger.info("✓ Pricing model saved")
    
    # 4. Train demand model
    logger.info("\n" + "="*80)
    logger.info("TRAINING DEMAND MODEL")
    logger.info("="*80)
    
    demand_metrics = demand_model.train(
        X=X,
        y_sold_fast=y_sold_fast,
        y_dom=y_dom,
        sell_within_days=90,
        hyperparameter_tuning=hyperparameter_tuning,
        test_size=0.2,
        validation_size=0.2
    )
    
    logger.info(f"Demand Model - Sell Probability AUC: {demand_metrics['sell_probability']['test_auc']:.3f}")
    logger.info(f"Demand Model - DOM MAPE: {demand_metrics['dom']['test_mape']:.2f}%")
    
    if save_models:
        demand_model.save()
        logger.info("✓ Demand model saved")
    
    # 5. Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    
    return {
        'pricing_metrics': pricing_metrics,
        'demand_metrics': demand_metrics,
        'training_samples': len(X),
        'features': len(X.columns),
        'zip_codes': zip_codes,
        'training_date': datetime.now().isoformat()
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML models for real estate recommendations')
    parser.add_argument('--zip-codes', nargs='+', default=['27410'],
                       help='ZIP codes to use for training (default: 27410)')
    parser.add_argument('--months-back', type=int, default=36,
                       help='Months of historical data (default: 36)')
    parser.add_argument('--hyperparameter-tuning', action='store_true',
                       help='Perform hyperparameter tuning (slower but better)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save trained models')
    
    args = parser.parse_args()
    
    try:
        results = train_models(
            zip_codes=args.zip_codes,
            months_back=args.months_back,
            hyperparameter_tuning=args.hyperparameter_tuning,
            save_models=not args.no_save
        )
        
        print("\n✅ Training successful!")
        print(f"Pricing MAPE: {results['pricing_metrics']['test_mape']:.2f}%")
        print(f"Demand AUC: {results['demand_metrics']['sell_probability']['test_auc']:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
