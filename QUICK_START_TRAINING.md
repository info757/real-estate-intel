# Quick Start: Fast-Seller Model Training

## What's Ready

✅ **Training script updated** with:
- Option A/B strategy (similar ZIPs → all Greensboro if needed)
- Batch LLM feature extraction (5-10x faster)
- Caching support (resume if interrupted)
- Parallel ZIP fetching (2-3x faster)
- Progress logging

✅ **Monitoring script** created: `scripts/monitor_training.py`
✅ **Data quality script** created: `scripts/check_data_quality.py`

## Start Training

### Option 1: Use Similar ZIPs (Recommended First)
```bash
python backend/ml/train_fast_seller_model.py --use-similar-zips --days-back 365
```

### Option 2: Use All Greensboro ZIPs
```bash
python backend/ml/train_fast_seller_model.py --all-zips --days-back 365
```

### Option 3: Custom ZIPs
```bash
python backend/ml/train_fast_seller_model.py --zip-codes 27408 27410 27411 27412 --days-back 365
```

## Monitor Progress

In another terminal:
```bash
python scripts/monitor_training.py
```

## Check Data Quality

After data collection:
```bash
python scripts/check_data_quality.py
```

## Training Options

- `--days-back 365` - 12 months of data (default)
- `--max-per-zip 200` - Max listings per ZIP (default: 200)
- `--fast-threshold 14` - DOM threshold for fast sellers (default: 14 days)
- `--hyperparameter-tuning` - Enable tuning (slower but better)
- `--no-save` - Don't save models (for testing)

## Expected Timeline

- **Data Collection:** 2-3 hours (with caching, can resume)
- **Model Training:** 1-2 hours
- **Total:** 3-5 hours (can run in background)

## Troubleshooting

If training fails:
1. Check cache: `ls cache/listings/`
2. Check data quality: `python scripts/check_data_quality.py`
3. Review logs for errors
4. Try with fewer ZIPs first

## Next Steps After Training

1. Test model: `python scripts/test_fast_seller_integration.py`
2. Test in UI: Start Streamlit and check ML Recommendations page
3. Verify fast-seller predictions appear
