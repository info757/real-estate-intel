"""Quick script to count dom_to_pending vs dom_to_sold listings."""
import json
from pathlib import Path

cache_dir = Path("cache/listings")

if not cache_dir.exists():
    print("No cache directory - training hasn't started yet")
    exit(0)

cache_files = list(cache_dir.glob("listings_*.json"))
if not cache_files:
    print("No cached data yet - data collection in progress")
    exit(0)

total = 0
with_pending = 0
with_sold_only = 0
with_both = 0

for cache_file in cache_files:
    try:
        with open(cache_file, 'r') as f:
            listings = json.load(f)
        for listing in listings:
            total += 1
            dom_pending = listing.get('dom_to_pending')
            dom_sold = listing.get('dom_to_sold')
            
            if dom_pending is not None:
                with_pending += 1
            if dom_sold is not None and dom_pending is None:
                with_sold_only += 1
            if dom_pending is not None and dom_sold is not None:
                with_both += 1
    except Exception as e:
        print(f"Error reading {cache_file}: {e}")

print("="*80)
print("DOM METRICS COUNT")
print("="*80)
print(f"Total listings: {total}")
print(f"With dom_to_pending: {with_pending} ({with_pending/total*100:.1f}%)" if total > 0 else "With dom_to_pending: 0")
print(f"With dom_to_sold only: {with_sold_only} ({with_sold_only/total*100:.1f}%)" if total > 0 else "With dom_to_sold only: 0")
print(f"With both: {with_both} ({with_both/total*100:.1f}%)" if total > 0 else "With both: 0")
print()

if total > 0:
    if with_pending < 1000:
        print(f"⚠️  WARNING: Only {with_pending} listings with dom_to_pending")
        print("   Recommendation: Implement Option 1C normalization to use dom_to_sold data")
    else:
        print(f"✅ Good: {with_pending} listings with dom_to_pending (sufficient for training)")
