# RealEstateApi Sold Listing Ingestion

## Quick Start

- The loader now enforces an upper bound on raw records per ZIP (`settings.realestateapi_max_results_per_zip`, default `6000`) and keeps only sales from the past 730 days (configurable via the `days_back` argument).
- Duplicate `property_id`/`mls_id` entries and listings without a resolvable sold date are discarded before caching.
- The training pipeline re-validates the sanitised data and aborts early if the usable count for any ZIP hits the configured cap, surfacing an actionable error instead of running indefinitely.

## Sample Verification Run (2025‑11‑08)

```
python scripts <<NOTE: executed via RealEstateApiListingLoader>>
ZIP 27410 -> 780 usable listings (after cap applied to 20,548 raw)
ZIP 27215 -> 2,158 usable listings (after cap applied to 19,261 raw)
```

Cache re-use keeps subsequent training runs fast; the targeted call to

```
PYTHONPATH=venv/lib/python3.11/site-packages \
python backend/ml/train_fast_seller_model.py \
  --zip-codes 27410 27215 \
  --days-back 730 \
  --no-save
```

loads the same cached data instantly. The current dataset lacks DOM history for those ZIPs, so the run stops with `Valid samples: 0/2938` (no `dom_to_pending` values). This is expected behaviour: nothing else is processed and the failure happens within seconds instead of hours. Once RealEstateApi exposes consistent DOM timestamps the run will complete normally with the same safeguards in place.

## Operational Notes

- Increase the cap only if a market truly requires more than 6 000 in-range sales; otherwise the loader will warn and stop at the limit.
- If a ZIP aborts during training with “meets/exceeds the configured cap”, rerun with a shorter `days_back` window or adjust `REALESTATEAPI_MAX_RESULTS_PER_ZIP`.
- Logs from `backend/ml/train_fast_seller_model` now show sanitisation stats per ZIP (`raw`, `kept`, `duplicates`, `stale`, `missing_sale`) to simplify monitoring.

