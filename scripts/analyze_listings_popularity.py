#!/usr/bin/env python3
"""
Safe script to analyze listing popularity using official APIs.
Usage: python scripts/analyze_listings_popularity.py --zip 27410
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data_collectors.safe_listings_scraper import safe_listings_scraper
from backend.analyzers.popularity_analyzer import popularity_analyzer

def main():
    parser = argparse.ArgumentParser(
        description='Analyze listing popularity using safe API methods'
    )
    parser.add_argument('--zip', type=str, default='27410', help='ZIP code to analyze')
    parser.add_argument('--max-results', type=int, default=100, help='Maximum listings to fetch')
    parser.add_argument('--status', type=str, default='active', 
                       choices=['active', 'pending', 'sold'],
                       help='Listing status to analyze')
    parser.add_argument('--source', type=str, default='auto',
                       choices=['auto', 'rapidapi_realtor', 'rapidapi_zillow', 'attom'],
                       help='Data source (auto tries all available)')
    parser.add_argument('--output-dir', type=str, default='data/listings_analysis',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"LISTING POPULARITY ANALYSIS - ZIP {args.zip}")
    print("="*80)
    print(f"Source: {args.source} | Status: {args.status}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch listings using safe methods
    print(f"📥 Fetching {args.status} listings (using official APIs)...")
    print()
    
    try:
        listings = safe_listings_scraper.fetch_listings(
            zip_code=args.zip,
            status=args.status,
            max_results=args.max_results,
            source=args.source
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tips: Set RAPIDAPI_KEY env var or use --source attom")
        return 1
    
    if not listings:
        print("❌ No listings found. Check API keys and ZIP code.")
        return 1
    
    print(f"✅ Found {len(listings)} listings")
    
    # Save and analyze
    listings_file = output_dir / f"listings_{args.zip}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(listings_file, 'w') as f:
        json.dump(listings, f, indent=2, default=str)
    print(f"💾 Saved to {listings_file}")
    print()
    
    # Analyze if we have data
    has_data = any(l.get('views') or l.get('saves') or l.get('days_on_zillow') for l in listings)
    if has_data:
        print("📊 Analyzing popularity...")
        results = popularity_analyzer.analyze_popular_listings(listings, top_n=20)
        
        print("\nTOP 10 LISTINGS:")
        for i, listing in enumerate(results['top_listings'][:10], 1):
            print(f"{i}. {listing.get('address', 'Unknown')[:50]}")
            print(f"   ${listing.get('price', 0):,} | Score: {listing['popularity_score']:.2f}")
    
    print("\n✅ Complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())

