from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml.train_fast_seller_model import fetch_sold_listings_with_features
from scripts.identify_similar_zips import get_triad_zips

LOGGER = logging.getLogger("build_new_build_spec_cache")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("backend.data_collectors.realestateapi_loader").setLevel(logging.ERROR)

CACHE_PATH = Path("data/cache/new_build_specs.json")
DEFAULT_DAYS_BACK = 1095  # three years


def _is_new_build(listing: Dict[str, Any], *, cutoff_year: int) -> bool:
    candidates = [
        listing.get("year_built"),
        listing.get("yearBuilt"),
        (listing.get("summary") or {}).get("yearBuilt"),
        (listing.get("property_detail_raw") or {}).get("yearBuilt"),
        (listing.get("metadata") or {}).get("year_built"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            year_numeric = int(float(candidate))
            if year_numeric >= cutoff_year:
                return True
        except Exception:
            continue
    return False


def _extract_property_type(listing: Dict[str, Any]) -> str:
    metadata = listing.get("metadata") or {}
    label = metadata.get("property_type_label") or listing.get("property_type")
    if isinstance(label, str) and label.strip():
        label = label.strip().lower()
        if "town" in label:
            return "Townhome"
        if "condo" in label or "apartment" in label:
            return "Condo"
    return "Single Family Home"


def _extract_subdivision_key(listing: Dict[str, Any]) -> Optional[str]:
    summary = listing.get("summary") or {}
    property_detail = listing.get("property_detail_raw") or {}
    candidates = [
        listing.get("subdivision"),
        (summary.get("neighborhood") or {}).get("name"),
        summary.get("subdivision"),
        (property_detail.get("neighborhood") or {}).get("name"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    return None


def build_cache(
    zip_codes: List[str], *, days_back: int, max_per_zip: Optional[int], new_build_years: int
) -> Dict[str, Any]:
    cutoff_year = datetime.datetime.now().year - new_build_years
    entries: Dict[str, List[Dict[str, Any]]] = {}

    listings = fetch_sold_listings_with_features(
        zip_codes=zip_codes,
        days_back=days_back,
        max_per_zip=max_per_zip,
        use_cache=True,
    )
    LOGGER.info("Collected %s listings across %s zips", len(listings), len(zip_codes))

    for listing in listings:
        if not _is_new_build(listing, cutoff_year=cutoff_year):
            continue

        property_type = _extract_property_type(listing)
        beds = listing.get("beds") or (listing.get("summary") or {}).get("beds")
        baths = listing.get("baths") or (listing.get("summary") or {}).get("baths")
        sqft = listing.get("sqft") or (listing.get("summary") or {}).get("universalsize")
        try:
            beds_val = int(float(beds)) if beds is not None else None
            baths_val = float(baths) if baths is not None else None
            sqft_val = int(float(sqft)) if sqft is not None else None
        except Exception:
            continue

        if any(val is None for val in (beds_val, baths_val, sqft_val)):
            continue
        if sqft_val <= 0 or beds_val <= 0 or baths_val <= 0:
            continue

        entry = {
            "beds": beds_val,
            "baths": round(baths_val, 1),
            "sqft": sqft_val,
            "subdivision_key": _extract_subdivision_key(listing),
        }
        entries.setdefault(property_type, []).append(entry)

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cache of new-build spec combos.")
    parser.add_argument("--zip", dest="zip_codes", nargs="+", help="ZIP codes to include")
    parser.add_argument("--triad", action="store_true", help="Use all Triad ZIPs")
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK)
    parser.add_argument("--output", default=str(CACHE_PATH))
    parser.add_argument("--max-per-zip", type=int, default=1000)
    parser.add_argument("--new-build-years", type=int, default=5)
    args = parser.parse_args()

    zip_codes = args.zip_codes or []
    if args.triad:
        zip_codes.extend(get_triad_zips())
    if not zip_codes:
        raise SystemExit("Please supply at least one ZIP via --zip or --triad")

    cache = build_cache(
        list(dict.fromkeys(zip_codes)),
        days_back=args.days_back,
        max_per_zip=args.max_per_zip,
        new_build_years=args.new_build_years,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(cache, fh, indent=2)
    LOGGER.info("Wrote new-build spec cache -> %s", output_path)


if __name__ == "__main__":
    main()
