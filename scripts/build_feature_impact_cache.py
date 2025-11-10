"""
Utility script to pre-compute feature impact statistics for a list of ZIP codes.

This populates `data/cache/feature_impacts/<zip>.json` and refreshes the
aggregate `coverage.json`, then emits a CSV summary under `reports/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

# Ensure application modules are importable
import sys
import os
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.identify_similar_zips import get_triad_zips  # noqa: E402
from prototype.app import (  # noqa: E402
    get_feature_impacts,
    get_feature_dataset,
)


def iter_zip_codes(include: Iterable[str] | None, triad_only: bool) -> Iterable[str]:
    """Yield ZIP codes we should process."""
    if include:
        for zip_code in include:
            yield str(zip_code)
        return

    if triad_only:
        for zip_code in get_triad_zips():
            yield zip_code
        return

    raise SystemExit("No ZIP codes provided. Use --triad or --zip.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache feature impact statistics by ZIP code.")
    parser.add_argument(
        "--zip",
        dest="zip_codes",
        action="append",
        help="Specific ZIP code to process (can be passed multiple times).",
    )
    parser.add_argument(
        "--triad",
        dest="triad",
        action="store_true",
        help="Process the predefined Triad ZIP list.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of ZIP codes to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute feature impacts even when a cache file already exists.",
    )
    args = parser.parse_args()

    cache_dir = Path("data/cache/feature_impacts")
    cache_dir.mkdir(parents=True, exist_ok=True)

    processed = []

    for index, zip_code in enumerate(iter_zip_codes(args.zip_codes, args.triad), start=1):
        if args.limit and index > args.limit:
            break
        print(f"[{index}] Processing ZIP {zip_code}…", flush=True)
        cache_file = cache_dir / f"{zip_code}.json"
        if args.force and cache_file.exists():
            cache_file.unlink()
        try:
            impacts = get_feature_impacts(zip_code)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ⚠️  Failed to compute impacts for {zip_code}: {exc}")
            continue

        dataset = get_feature_dataset(zip_code)
        processed.append(
            {
                "zip_code": zip_code,
                "listings": len(dataset),
                "feature_count": len(impacts),
            }
        )
        print(f"  ✓ cached {len(impacts)} features (listings: {len(dataset)})")

    if not processed:
        print("No ZIP codes processed; exiting.")
        return

    # Save per-ZIP manifest for quick inspection
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(processed, indent=2, sort_keys=True))

    # Refresh coverage summary CSV
    coverage_path = cache_dir / "coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
    else:
        coverage = {}

    rows: list[dict[str, object]] = []
    for zip_code, payload in (coverage.get("by_zip") or {}).items():
        features = payload.get("features") or {}
        listings = payload.get("listings")
        for feature_key, stats in features.items():
            rows.append(
                {
                    "zip_code": zip_code,
                    "feature": feature_key,
                    "label": stats.get("label"),
                    "count": stats.get("count"),
                    "price_lift": stats.get("price_lift"),
                    "dom_delta": stats.get("dom_delta"),
                    "listings": listings,
                    "updated_at": payload.get("updated_at"),
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        df.sort_values(["feature", "zip_code"], inplace=True)
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / "feature_impact_coverage.csv"
        df.to_csv(output_path, index=False)
        print(f"Wrote coverage CSV to {output_path} ({len(df)} rows)")
    else:
        print("No coverage data found after processing.")


if __name__ == "__main__":  # pragma: no cover
    main()


