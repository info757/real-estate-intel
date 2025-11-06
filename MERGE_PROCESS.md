# Collecting New ZIP Codes and Merging Data

## Overview
This process collects data from new ZIP codes and merges with existing cached data.

## New ZIP Codes to Collect
- 27429, 27435, 27438, 27495, 27497, 27498, 27499

## Process Flow

### Step 1: Wait for Current Training to Complete
The current training is collecting data from the original 13 ZIP codes.
Monitor with: `tail -f training.log`

### Step 2: Collect New ZIP Codes (Run This Next)
```bash
cd /Users/williamholt/real-estate-intel
python3 scripts/collect_new_zips.py --days-back 365 --max-per-zip 200
```

**What this does:**
- Fetches sold listings from the 7 new ZIP codes
- Calculates DOM metrics (listing date → pending date)
- Extracts features using LLM (from property descriptions)
- Caches the data (same format as original training)
- Uses same rate limiting (0.4s delays) as main training

**Monitor progress:**
```bash
# Run in foreground to see real-time progress
python3 scripts/collect_new_zips.py

# Or run in background and monitor
nohup python3 scripts/collect_new_zips.py > collection_new_zips.log 2>&1 &
tail -f collection_new_zips.log
```

### Step 3: Merge and Retrain (After Collection Completes)
```bash
python3 scripts/merge_and_retrain.py --days-back 365 --hyperparameter-tuning
```

**What this does:**
- Loads all cached data from all 20 ZIP codes (original 13 + new 7)
- Verifies data availability
- Retrains the fast-seller model with the merged dataset
- Uses median-based market-relative thresholds (per ZIP)
- Saves the updated models

## Notes
- Data is cached per ZIP, so if collection fails, you can resume
- The merge script will only use ZIP codes that have cached data
- Missing ZIP codes will be logged but won't block the process
- The retrained model will use all available data for better accuracy
