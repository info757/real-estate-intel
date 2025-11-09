"""
Identify ZIP codes similar to 27410 and discover full Greensboro coverage.
"""

import sys
import os
from pathlib import Path
from typing import List, Iterable, Set

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

# Core Piedmont Triad counties (Greensboro / Winston-Salem / High Point region)
TRIAD_COUNTIES = [
    "Alamance County",
    "Caswell County",
    "Davidson County",
    "Davie County",
    "Forsyth County",
    "Guilford County",
    "Montgomery County",
    "Randolph County",
    "Rockingham County",
    "Stokes County",
    "Surry County",
    "Yadkin County",
]


def _zips_from_cache(pattern: str = "listings_*_730days*.json") -> List[str]:
    """Load ZIPs from cached RealEstateApi listing files if available."""
    cache_root = Path("cache/listings/realestateapi")
    if not cache_root.exists():
        return []
    zips = {
        path.name.split("_")[1]
        for path in cache_root.glob(pattern)
        if path.name.startswith("listings_")
    }
    return sorted(zips)


def _collect_zips_from_records(records: Iterable[dict]) -> Set[str]:
    """Collect ZIP codes from API records."""
    discovered: Set[str] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        candidates = []
        address_block = rec.get("address") or {}
        candidates.extend([
            address_block.get("zip"),
            address_block.get("postalCode"),
            address_block.get("postal_code"),
        ])
        summary = rec.get("summary") or {}
        candidates.extend([
            summary.get("postalCode"),
            summary.get("zip"),
        ])
        property_info = rec.get("propertyInfo") or rec.get("property_info") or {}
        pi_address = property_info.get("address") or {}
        candidates.extend([
            pi_address.get("zip"),
            pi_address.get("postalCode"),
        ])
        candidates.extend([
            rec.get("zip"),
            rec.get("zipCode"),
            rec.get("postal_code"),
        ])
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.isdigit() and len(candidate) == 5:
                discovered.add(candidate)
    return discovered


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


def get_zips_for_counties(counties: List[str], state: str = "NC", max_pages: int = 5) -> List[str]:
    """
    Discover ZIP codes for the requested counties. Uses cache first, then falls back to API.
    """
    cached = _zips_from_cache()
    if cached:
        return cached

    client = RealEstateAPIClient()
    discovered: Set[str] = set()

    for county in counties:
        for page in range(1, max_pages + 1):
            try:
                response = client.search_mls_listings(
                    status="sold",
                    geography={"county": county, "state": state},
                    include_count=True,
                    page_size=250,
                    page=page,
                )
            except Exception:
                break

            records = response.get("results") or response.get("listings") or response.get("data") or []
            discovered.update(_collect_zips_from_records(records))

            meta = response.get("pagination") or response.get("meta") or {}
            total_pages = meta.get("total_pages") or meta.get("totalPages")
            if total_pages and page >= int(total_pages):
                break

    combined = discovered or set(ALL_GREENSBORO_ZIPS)
    return sorted(combined)


def get_triad_zips(use_cache_first: bool = True) -> List[str]:
    """
    Return ZIP codes that cover the Piedmont Triad counties.
    """
    if use_cache_first:
        cached = _zips_from_cache()
        if cached:
            return cached
    county_zips = get_zips_for_counties(TRIAD_COUNTIES, state="NC")
    if county_zips:
        return county_zips
    return sorted(set(SIMILAR_ZIPS) | set(ALL_GREENSBORO_ZIPS))


# Backwards compatibility export
TRIAD_ZIPS = get_triad_zips()


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
