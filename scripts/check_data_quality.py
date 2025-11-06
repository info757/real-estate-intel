"""
Check data quality for fast-seller model training.
Identifies outliers, missing data, and data quality issues.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def check_data_quality(listings: list) -> dict:
    """Analyze data quality and identify issues."""
    print("="*80)
    print("DATA QUALITY ANALYSIS")
    print("="*80)
    print()
    
    if not listings:
        print("❌ No listings provided")
        return {}
    
    print(f"Total listings: {len(listings)}")
    print()
    
    # Convert to DataFrame for analysis
    df_data = []
    for listing in listings:
        row = {
            'price': listing.get('price'),
            'beds': listing.get('beds'),
            'baths': listing.get('baths'),
            'sqft': listing.get('sqft'),
            'dom_to_pending': listing.get('dom_to_pending'),
            'has_description': bool(listing.get('description')),
            'has_features': bool(listing.get('extracted_features')),
            'has_dom': listing.get('dom_to_pending') is not None,
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    issues = []
    
    # Check missing DOM
    missing_dom = df['has_dom'].sum()
    missing_dom_pct = (missing_dom / len(df) * 100) if len(df) > 0 else 0
    if missing_dom > 0:
        issues.append(f"Missing DOM: {missing_dom} ({missing_dom_pct:.1f}%)")
    
    # Check missing descriptions
    missing_desc = (~df['has_description']).sum()
    missing_desc_pct = (missing_desc / len(df) * 100) if len(df) > 0 else 0
    if missing_desc > 0:
        issues.append(f"Missing descriptions: {missing_desc} ({missing_desc_pct:.1f}%)")
    
    # Check price outliers
    prices = df['price'].dropna()
    if len(prices) > 0:
        Q1 = prices.quantile(0.25)
        Q3 = prices.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
        if len(outliers) > 0:
            issues.append(f"Price outliers: {len(outliers)} (outside ${lower_bound:,.0f} - ${upper_bound:,.0f})")
    
    # Check DOM outliers
    doms = df['dom_to_pending'].dropna()
    if len(doms) > 0:
        Q1 = doms.quantile(0.25)
        Q3 = doms.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR
        outliers = doms[doms > upper_bound]
        if len(outliers) > 0:
            issues.append(f"DOM outliers: {len(outliers)} (>{upper_bound:.0f} days)")
    
    # Check fast seller distribution
    if len(doms) > 0:
        fast_sellers = (doms <= 14).sum()
        fast_pct = (fast_sellers / len(doms) * 100) if len(doms) > 0 else 0
        print(f"Fast sellers (≤14 days): {fast_sellers} ({fast_pct:.1f}%)")
        if fast_pct < 10 or fast_pct > 50:
            issues.append(f"Unbalanced fast seller distribution: {fast_pct:.1f}%")
    
    # Display statistics
    print("\nStatistics:")
    print("-" * 80)
    if len(prices) > 0:
        print(f"Price: ${prices.median():,.0f} median (${prices.min():,.0f} - ${prices.max():,.0f})")
    if len(doms) > 0:
        print(f"DOM to pending: {doms.median():.0f} median ({doms.min():.0f} - {doms.max():.0f} days)")
    if df['beds'].notna().sum() > 0:
        print(f"Beds: {df['beds'].median():.0f} median ({df['beds'].min():.0f} - {df['beds'].max():.0f})")
    if df['sqft'].notna().sum() > 0:
        print(f"Sqft: {df['sqft'].median():,.0f} median ({df['sqft'].min():,.0f} - {df['sqft'].max():,.0f})")
    
    # Display issues
    print()
    if issues:
        print("⚠️  Issues Found:")
        print("-" * 80)
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No major data quality issues detected")
    
    print()
    print("="*80)
    
    return {
        'total_listings': len(listings),
        'issues': issues,
        'stats': {
            'missing_dom': missing_dom,
            'missing_desc': missing_desc,
            'fast_seller_pct': fast_pct if len(doms) > 0 else 0
        }
    }

if __name__ == '__main__':
    # Load from cache or provide path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', default='cache/listings', help='Cache directory')
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        sys.exit(1)
    
    # Load all cached listings
    all_listings = []
    for cache_file in cache_dir.glob("listings_*.json"):
        try:
            with open(cache_file, 'r') as f:
                listings = json.load(f)
                all_listings.extend(listings)
        except Exception as e:
            print(f"⚠️  Error loading {cache_file}: {e}")
    
    if not all_listings:
        print("❌ No listings found in cache")
        sys.exit(1)
    
    check_data_quality(all_listings)
