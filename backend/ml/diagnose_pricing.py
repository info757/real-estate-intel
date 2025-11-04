"""
Diagnostic script to analyze pricing model performance and identify issues.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.ml.train_models import fetch_training_data, prepare_targets
from datetime import datetime, timedelta

def analyze_price_distribution():
    """Analyze price distribution and identify outliers."""
    
    print("="*80)
    print("PRICING MODEL DIAGNOSTIC ANALYSIS")
    print("="*80)
    print()
    
    # Use the core Greensboro ZIP codes we trained on
    print("Fetching training data...")
    CORE_GREENSBORO_ZIPS = [
        '27401', '27402', '27403', '27404', '27405', '27406', '27407', 
        '27408', '27409', '27410', '27411', '27412', '27455'
    ]
    months_back = 36
    max_properties_per_zip = 2500
    
    df = fetch_training_data(CORE_GREENSBORO_ZIPS, months_back=months_back, max_properties_per_zip=max_properties_per_zip)
    
    if df.empty:
        print("No data fetched!")
        return
    
    # Extract prices
    y_price, _, _, valid_mask = prepare_targets(df)
    prices = y_price[valid_mask]
    
    print(f"\nTotal samples: {len(prices)}")
    print(f"Price statistics:")
    print(f"  Min: ${prices.min():,.0f}")
    print(f"  25th percentile: ${prices.quantile(0.25):,.0f}")
    print(f"  Median: ${prices.median():,.0f}")
    print(f"  75th percentile: ${prices.quantile(0.75):,.0f}")
    print(f"  95th percentile: ${prices.quantile(0.95):,.0f}")
    print(f"  99th percentile: ${prices.quantile(0.99):,.0f}")
    print(f"  Max: ${prices.max():,.0f}")
    print(f"  Mean: ${prices.mean():,.0f}")
    print(f"  Std: ${prices.std():,.0f}")
    
    # Identify outliers using IQR method
    Q1 = prices.quantile(0.25)
    Q3 = prices.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # Using 3x IQR for more lenient outlier detection
    upper_bound = Q3 + 3 * IQR
    
    outliers_low = prices[prices < lower_bound]
    outliers_high = prices[prices > upper_bound]
    
    print(f"\nOutlier Detection (using 3x IQR method):")
    print(f"  Lower bound: ${lower_bound:,.0f}")
    print(f"  Upper bound: ${upper_bound:,.0f}")
    print(f"  Low outliers (< ${lower_bound:,.0f}): {len(outliers_low)} properties")
    print(f"  High outliers (> ${upper_bound:,.0f}): {len(outliers_high)} properties")
    
    if len(outliers_low) > 0:
        print(f"\n  Low outlier prices:")
        for price in outliers_low.head(10):
            print(f"    ${price:,.0f}")
    
    if len(outliers_high) > 0:
        print(f"\n  High outlier prices:")
        for price in outliers_high.head(10):
            print(f"    ${price:,.0f}")
    
    # Check for suspiciously low prices (likely data errors)
    suspicious_low = prices[prices < 10000]  # Less than $10k
    print(f"\n  Suspiciously low prices (< $10k): {len(suspicious_low)} properties")
    if len(suspicious_low) > 0:
        for price in suspicious_low.head(10):
            print(f"    ${price:,.0f}")
    
    # Check for suspiciously high prices (likely data errors)
    suspicious_high = prices[prices > 5000000]  # More than $5M
    print(f"\n  Suspiciously high prices (> $5M): {len(suspicious_high)} properties")
    if len(suspicious_high) > 0:
        for price in suspicious_high.head(10):
            print(f"    ${price:,.0f}")
    
    # Calculate what MAPE would be with outliers removed
    prices_clean = prices[(prices >= lower_bound) & (prices <= upper_bound)]
    print(f"\nWith outliers removed:")
    print(f"  Remaining samples: {len(prices_clean)} ({len(prices_clean)/len(prices)*100:.1f}%)")
    print(f"  Price range: ${prices_clean.min():,.0f} - ${prices_clean.max():,.0f}")
    
    # Analyze price distribution
    print(f"\nPrice distribution analysis:")
    print(f"  Skewness: {prices.skew():.2f} (normal = 0, right-skewed > 0)")
    print(f"  Log skewness: {np.log(prices).skew():.2f}")
    
    # Check for common price patterns (likely data errors)
    print(f"\nCommon suspicious price patterns:")
    round_prices = prices[prices % 1000 == 0]
    very_round = prices[prices % 10000 == 0]
    print(f"  Prices ending in $000: {len(round_prices)} ({len(round_prices)/len(prices)*100:.1f}%)")
    print(f"  Prices ending in $0000: {len(very_round)} ({len(very_round)/len(prices)*100:.1f}%)")
    
    # Analyze user's suggested filter ($100k - $3M)
    user_filter_low = 100000
    user_filter_high = 3000000
    prices_user_filter = prices[(prices >= user_filter_low) & (prices <= user_filter_high)]
    removed_low = len(prices[prices < user_filter_low])
    removed_high = len(prices[prices > user_filter_high])
    
    print(f"\n" + "="*80)
    print("FILTER ANALYSIS: $100k - $3M")
    print("="*80)
    print(f"Properties with prices < $100k: {removed_low} ({removed_low/len(prices)*100:.1f}%)")
    print(f"Properties with prices > $3M: {removed_high} ({removed_high/len(prices)*100:.1f}%)")
    print(f"Properties kept: {len(prices_user_filter)}/{len(prices)} ({len(prices_user_filter)/len(prices)*100:.1f}%)")
    print(f"\nFiltered price range: ${prices_user_filter.min():,.0f} - ${prices_user_filter.max():,.0f}")
    print(f"Filtered price statistics:")
    print(f"  Median: ${prices_user_filter.median():,.0f}")
    print(f"  Mean: ${prices_user_filter.mean():,.0f}")
    print(f"  Std: ${prices_user_filter.std():,.0f}")
    print(f"  Skewness: {prices_user_filter.skew():.2f}")
    
    # Recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    print("1. Apply price filter: $100k - $3M")
    print(f"   - Remove {removed_low} properties < $100k ({removed_low/len(prices)*100:.1f}%)")
    print(f"   - Remove {removed_high} properties > $3M ({removed_high/len(prices)*100:.1f}%)")
    print(f"   - Keep {len(prices_user_filter)}/{len(prices)} samples ({len(prices_user_filter)/len(prices)*100:.1f}%)")
    
    if prices_user_filter.skew() > 1:
        print("\n2. Consider log transformation:")
        print(f"   - Filtered prices still right-skewed (skewness = {prices_user_filter.skew():.2f})")
        print(f"   - Log transformation can help normalize the distribution")
        print(f"   - Train on log(prices) and convert back for predictions")
    
    print("\n3. Consider using alternative metrics:")
    print("   - Median Absolute Percentage Error (MdAPE) - less sensitive to outliers")
    print("   - Mean Absolute Error (MAE) - absolute dollar error")
    print("   - Root Mean Squared Log Error (RMSLE) - better for skewed data")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    analyze_price_distribution()

