"""
Identify ZIP codes similar to 27410 and discover full Greensboro coverage.
"""

import sys
import os
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data_collectors.realestateapi_client import RealEstateAPIClient

# Similar ZIPs to 27410 (mid-to-upper market areas in Greensboro)
SIMILAR_ZIPS = ['27408', '27410', '27411', '27412']

# All Greensboro ZIPs (fallback/default)
ALL_GREENSBORO_ZIPS = [
    '27401', '27402', '27403', '27404', '27405', '27406', '27407',
    '27408', '27409', '27410', '27411', '27412', '27429', '27435',
    '27438', '27455', '27495', '27497', '27498', '27499'
]


def get_training_zips(use_similar_only: bool = True) -> List[str]:
    """
    Get ZIP codes for training.

    Args:
        use_similar_only: If True, return similar ZIPs. If False, return all.

    Returns:
        List of ZIP codes
    """
    if use_similar_only:
        return SIMILAR_ZIPS
    else:
        return ALL_GREENSBORO_ZIPS


def discover_greensboro_zips(max_pages: int = 10) -> List[str]:
    """Query RealEstateApi to discover every ZIP code within Greensboro."""
    client = RealEstateAPIClient()
    discovered: set[str] = set()
    search_endpoints = [
        "/v2/PropertySearch",
    ]

    for endpoint in search_endpoints:
        discovered.clear()
        for page in range(1, max_pages + 1):
            try:
                response = client.search_mls_listings(
                    status="sold",
                    geography={"city": "Greensboro", "state": "NC"},
                    include_count=True,
                    page_size=250,
                    page=page,
                    endpoint=endpoint,
                )
            except Exception:
                discovered.clear()
                break

            records = response.get("results") or response.get("listings") or response.get("data") or []
            if not isinstance(records, list) or not records:
                break

            for rec in records:
                zip_candidates = []
                address_block = rec.get("address") or {}
                zip_candidates.append(address_block.get("zip") or address_block.get("postalCode"))
                property_info = rec.get("propertyInfo") or rec.get("property_info") or {}
                pi_address = property_info.get("address") or {}
                zip_candidates.append(pi_address.get("zip") or pi_address.get("postalCode"))
                zip_candidates.append(rec.get("zip") or rec.get("zipCode"))
                zip_candidates.append(rec.get("postal_code"))

                for candidate in zip_candidates:
                    if candidate and isinstance(candidate, str) and candidate.isdigit() and len(candidate) == 5:
                        discovered.add(candidate)

            pagination = response.get("pagination") or response.get("meta") or {}
            total_pages = pagination.get("total_pages") or pagination.get("totalPages")
            if total_pages and page >= int(total_pages):
                break

        if discovered:
            break

    return sorted(discovered)


if __name__ == '__main__':
    print("Similar ZIPs to 27410:", ', '.join(SIMILAR_ZIPS))
    print("All Greensboro ZIPs (default list):", ', '.join(ALL_GREENSBORO_ZIPS))
    try:
        discovered = discover_greensboro_zips()
        if discovered:
            print("Discovered Greensboro ZIPs via PropertySearch:", ', '.join(discovered))
        else:
            print("No ZIPs discovered via PropertySearch")
    except Exception as exc:
        print(f"Warning: Could not discover ZIPs automatically: {exc}")
