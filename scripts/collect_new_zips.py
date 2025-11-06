"""
Collect data from new ZIP codes and merge with existing cached data.
This script can run independently of the main training process.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from backend.data_collectors.safe_listings_scraper import SafeListingsScraper
from backend.analyzers.sold_listings_analyzer import SoldListingsAnalyzer
from backend.ml.feature_extractor import FeatureExtractor
from backend.ml.train_fast_seller_model import (
    get_cache_path, load_cached_listings, save_cached_listings
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# New ZIP codes to collect
NEW_ZIP_CODES = ['27429', '27435', '27438', '27495', '27497', '27498', '27499']


def collect_new_zips(
    zip_codes: list,
    days_back: int = 365,
    max_per_zip: int = 200
) -> int:
    """
    Collect data from new ZIP codes.
    
    Args:
        zip_codes: List of ZIP codes to collect
        days_back: Days of historical data to fetch
        max_per_zip: Maximum listings per ZIP code
        
    Returns:
        Total number of listings collected
    """
    scraper = SafeListingsScraper()
    analyzer = SoldListingsAnalyzer()
    extractor = FeatureExtractor()
    
    total_collected = 0
    
    for zip_code in zip_codes:
        logger.info("="*80)
        logger.info(f"Processing ZIP {zip_code}")
        logger.info("="*80)
        
        # Check if already cached
        cached = load_cached_listings(zip_code, days_back)
        if cached:
            logger.info(f"✅ ZIP {zip_code} already cached: {len(cached)} listings")
            total_collected += len(cached)
            continue
        
        try:
            # Fetch sold listings
            logger.info(f"Fetching sold listings for ZIP {zip_code}...")
            listings = scraper.fetch_sold_with_details(
                zip_code=zip_code,
                days_back=days_back,
                max_results=max_per_zip,
                fetch_details=True
            )
            
            if not listings:
                logger.warning(f"⚠️  No listings found for ZIP {zip_code}")
                continue
            
            # Calculate DOM metrics
            logger.info(f"Calculating DOM metrics for {len(listings)} listings...")
            listings = analyzer.calculate_dom_metrics(listings)
            
            # Extract features from descriptions
            logger.info(f"Extracting features from descriptions...")
            listings = extractor.batch_extract_features(listings, batch_size=15)
            
            # Cache results
            save_cached_listings(zip_code, days_back, listings)
            
            logger.info(f"✅ Collected and cached {len(listings)} listings from ZIP {zip_code}")
            total_collected += len(listings)
            
        except Exception as e:
            logger.error(f"❌ Error processing ZIP {zip_code}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("="*80)
    logger.info(f"COLLECTION COMPLETE: {total_collected} total listings from {len(zip_codes)} ZIP codes")
    logger.info("="*80)
    
    return total_collected


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect data from new ZIP codes')
    parser.add_argument('--zip-codes', nargs='+', default=NEW_ZIP_CODES,
                        help='ZIP codes to collect (default: new ZIPs)')
    parser.add_argument('--days-back', type=int, default=365,
                        help='Days of historical data to fetch')
    parser.add_argument('--max-per-zip', type=int, default=200,
                        help='Maximum listings per ZIP code')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("COLLECTING DATA FROM NEW ZIP CODES")
    logger.info("="*80)
    logger.info(f"ZIP codes: {', '.join(args.zip_codes)}")
    logger.info(f"Days back: {args.days_back}")
    logger.info(f"Max per ZIP: {args.max_per_zip}")
    logger.info("")
    
    try:
        total = collect_new_zips(
            zip_codes=args.zip_codes,
            days_back=args.days_back,
            max_per_zip=args.max_per_zip
        )
        logger.info(f"\n✅ Successfully collected {total} listings")
    except Exception as e:
        logger.error(f"\n❌ Collection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

