#!/usr/bin/env python3
"""
Train ML models on all Greensboro, NC ZIP codes.
Optimized for maximum accuracy (not speed).
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.ml.train_models import train_models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All Greensboro, NC ZIP codes
GREENSBORO_ZIP_CODES = [
    '27401',  # Downtown/Central
    '27402',  # Southeast
    '27403',  # Northeast
    '27404',  # Northwest
    '27405',  # Southwest
    '27406',  # North
    '27407',  # West
    '27408',  # East
    '27409',  # Northeast
    '27410',  # North (Hamilton Lakes area)
    '27411',  # Unique
    '27412',  # Unique
    '27455',  # P.O. Box
    # Additional ZIPs that may be partially in Greensboro
    '27214',  # May be partially in Greensboro
    '27215',  # May be partially in Greensboro
    '27495',  # May be partially in Greensboro
    '27497',  # May be partially in Greensboro
    '27498',  # May be partially in Greensboro
    '27499',  # May be partially in Greensboro
]

# Core Greensboro ZIPs (most reliable)
CORE_GREENSBORO_ZIPS = [
    '27401', '27402', '27403', '27404', '27405', 
    '27406', '27407', '27408', '27409', '27410',
    '27411', '27412', '27455'
]


def main():
    """Train models on all Greensboro ZIP codes."""
    
    logger.info("="*80)
    logger.info("GREENSBORO, NC MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Training on {len(CORE_GREENSBORO_ZIPS)} core Greensboro ZIP codes")
    logger.info(f"ZIP codes: {', '.join(CORE_GREENSBORO_ZIPS)}")
    logger.info("")
    logger.info("Settings (optimized for accuracy):")
    logger.info("  - Historical data: 36 months")
    logger.info("  - Max properties per ZIP: 2500 (to capture all available sales)")
    logger.info("  - Hyperparameter tuning: ENABLED")
    logger.info("  - This will take approximately 20-30 minutes (more data)")
    logger.info("="*80)
    logger.info("")
    
    try:
        results = train_models(
            zip_codes=CORE_GREENSBORO_ZIPS,
            months_back=36,  # 3 years of historical data for better accuracy
            max_properties_per_zip=2500,  # High limit to capture all available sales data
            hyperparameter_tuning=True,  # Enable for best accuracy (slower but better)
            save_models=True
        )
        
        logger.info("")
        logger.info("="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Training samples: {results['training_samples']:,}")
        logger.info(f"Features: {results['features']}")
        logger.info(f"ZIP codes used: {len(results['zip_codes'])}")
        logger.info("")
        logger.info("Pricing Model Performance:")
        logger.info(f"  Test MAPE: {results['pricing_metrics']['test_mape']:.2f}%")
        logger.info(f"  Test R²: {results['pricing_metrics']['test_r2']:.3f}")
        logger.info("")
        logger.info("Demand Model Performance:")
        logger.info(f"  Sell Probability AUC: {results['demand_metrics']['sell_probability']['test_auc']:.3f}")
        logger.info(f"  DOM MAPE: {results['demand_metrics']['dom']['test_mape']:.2f}%")
        logger.info("")
        logger.info("Models saved to: models/")
        logger.info("="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

