"""
Monitor fast-seller model training progress.
Shows real-time progress, ETA, and status.
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def monitor_training():
    """Monitor training progress by checking cache files and logs."""
    cache_dir = Path("cache/listings")
    
    print("="*80)
    print("FAST-SELLER MODEL TRAINING MONITOR")
    print("="*80)
    print()
    
    if not cache_dir.exists():
        print("⚠️  No cache directory found. Training may not have started.")
        return
    
    # Check for cached listings
    cache_files = list(cache_dir.glob("listings_*.json"))
    
    if not cache_files:
        print("📊 Status: No cached data yet")
        print("   Training is in early stages or hasn't started.")
        return
    
    print(f"📊 Found {len(cache_files)} cached ZIP code datasets")
    print()
    
    total_listings = 0
    zip_stats = []
    
    for cache_file in cache_files:
        try:
            with open(cache_file, 'r') as f:
                listings = json.load(f)
            
            zip_code = cache_file.stem.split('_')[1]
            count = len(listings)
            total_listings += count
            
            # Check for extracted features
            with_features = sum(1 for l in listings if 'extracted_features' in l)
            with_dom = sum(1 for l in listings if 'dom_to_pending' in l and l.get('dom_to_pending') is not None)
            
            zip_stats.append({
                'zip': zip_code,
                'total': count,
                'with_features': with_features,
                'with_dom': with_dom
            })
        except Exception as e:
            print(f"⚠️  Error reading {cache_file}: {e}")
    
    # Display stats
    print("ZIP Code Progress:")
    print("-" * 80)
    for stat in sorted(zip_stats, key=lambda x: x['zip']):
        print(f"  {stat['zip']}: {stat['total']} listings")
        print(f"    - With features: {stat['with_features']} ({stat['with_features']/stat['total']*100:.0f}%)")
        print(f"    - With DOM: {stat['with_dom']} ({stat['with_dom']/stat['total']*100:.0f}%)")
    
    print()
    print(f"📈 Total Progress: {total_listings} listings collected")
    print()
    
    # Estimate completion
    from scripts.identify_similar_zips import SIMILAR_ZIPS, ALL_GREENSBORO_ZIPS
    
    target_zips = len(SIMILAR_ZIPS)  # Option A
    if len(cache_files) >= len(ALL_GREENSBORO_ZIPS):
        target_zips = len(ALL_GREENSBORO_ZIPS)  # Option B
    
    progress_pct = (len(cache_files) / target_zips * 100) if target_zips > 0 else 0
    print(f"   ZIP codes: {len(cache_files)}/{target_zips} ({progress_pct:.0f}%)")
    
    # Check for model files
    model_dir = Path("models")
    if model_dir.exists():
        classifier_path = model_dir / "fast_seller_classifier.pkl"
        regressor_path = model_dir / "dom_regressor.pkl"
        
        if classifier_path.exists() and regressor_path.exists():
            print()
            print("✅ Models trained and saved!")
            print(f"   - Classifier: {classifier_path}")
            print(f"   - Regressor: {regressor_path}")
        else:
            print()
            print("⏳ Models not yet trained")
            print("   Data collection may still be in progress")
    
    print()
    print("="*80)

if __name__ == '__main__':
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
