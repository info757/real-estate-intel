"""Quick probe to verify RealEstateApi PropertySearch connectivity."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_collectors.realestateapi_client import (
    RealEstateAPIClient,
    PROPERTY_SEARCH_ENDPOINT,
)


def probe_realestateapi(city: str = "Greensboro", state: str = "NC", status: str = "sold") -> dict:
    client = RealEstateAPIClient()

    today = datetime.utcnow().date()
    one_year_ago = today - timedelta(days=365)

    response = client.search_properties(
        filters={
            "city": city,
            "state": state,
            "and": [
                {
                    "status": {"equals": status.upper()}
                },
                {
                    "sale_date": {
                        "between": [
                            one_year_ago.isoformat(),
                            today.isoformat()
                        ]
                    }
                }
            ]
        },
        page_size=100,
        include_count=True,
    )

    records = response.get("results") or response.get("listings") or []
    pagination = response.get("pagination") or response.get("meta") or {}

    zips = set()
    for rec in records:
        address = rec.get("address") or {}
        zip_code = address.get("zip") or address.get("postalCode")
        if not zip_code:
            property_info = rec.get("propertyInfo") or rec.get("property_info") or {}
            pi_address = property_info.get("address") or {}
            zip_code = pi_address.get("zip")
        if zip_code:
            zips.add(str(zip_code))

    summary = {
        "records": len(records),
        "unique_zips": sorted(zips),
        "pagination": pagination,
        "sample_record_keys": list(records[0].keys()) if records else [],
    }

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    probe_realestateapi()
