"""High-level helpers for fetching listings via RealEstateApi.

This module provides a drop-in replacement for the RapidAPI-based
`SafeListingsScraper` so the ML pipelines can source MLS-grade data via the
RealEstateApi MCP/REST endpoints.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from config.settings import settings
from backend.data_collectors.realestateapi_client import (
    RealEstateAPIClient,
    PropertyUseMapper,
)

logger = logging.getLogger(__name__)


def _first(items: Iterable[Any], predicate) -> Optional[Any]:
    for item in items:
        if predicate(item):
            return item
    return None


class RealEstateApiListingLoader:
    """Convenience wrapper around :class:`RealEstateAPIClient`."""

    def __init__(self, client: Optional[RealEstateAPIClient] = None):
        self.client = client or RealEstateAPIClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_sold_with_details(
        self,
        *,
        zip_code: str,
        days_back: int = 365,
        max_results: int = 250,
        include_mls_detail: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fetch recently sold listings and enrich them with detailed metadata.

        Parameters
        ----------
        zip_code:
            ZIP code boundary for the query.
        days_back:
            How far back (days) to fetch MLS history. Determines the minimum
            listing/sale/contract dates used in the search predicates.
        max_results:
            Maximum number of listings to return.
        include_mls_detail:
            Whether to fetch the MLS detail endpoint for richer descriptions
            and media metadata.
        """

        search_start = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        listings: List[Dict[str, Any]] = []
        page = 1
        page_size = min(max_results, 250)

        while len(listings) < max_results:
            response = self.client.search_mls_listings(
                status="sold",
                geography={"zip": zip_code},
                list_date_min=search_start,
                status_date_min=search_start,
                page=page,
                page_size=page_size,
            )

            records = self._extract_records(response)
            if not records:
                break

            logger.info("RealEstateApi returned %d sold records on page %d", len(records), page)

            for record in records:
                normalized = self._normalize_search_record(record)
                if not normalized:
                    continue

                detailed = self._enrich_listing(normalized, include_mls_detail=include_mls_detail)
                listings.append(detailed)
                if len(listings) >= max_results:
                    break

            if not self._has_next_page(response, page):
                break
            page += 1

        logger.info("Collected %d sold listings for %s", len(listings), zip_code)
        return listings

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    def _extract_records(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = (
            response.get("results")
            or response.get("data")
            or response.get("properties")
            or response.get("listings")
            or []
        )
        if isinstance(candidates, dict):
            candidates = list(candidates.values())
        if not isinstance(candidates, list):
            logger.warning("Unexpected RealEstateApi response format: %s", type(candidates))
            return []
        return candidates

    def _has_next_page(self, response: Dict[str, Any], current_page: int) -> bool:
        pagination = response.get("pagination") or response.get("meta") or {}
        if isinstance(pagination, dict):
            next_page = pagination.get("next_page") or pagination.get("nextPage")
            total_pages = pagination.get("total_pages") or pagination.get("totalPages")
            if next_page is not None:
                return bool(next_page)
            if total_pages is not None:
                return current_page < int(total_pages)
        return False

    # ------------------------------------------------------------------
    def _normalize_search_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        property_id = (
            record.get("propertyId")
            or record.get("property_id")
            or record.get("id")
            or record.get("_id")
        )
        if not property_id:
            logger.debug("Skipping record without property_id: %s", record.keys())
            return None

        address_block = (
            record.get("address")
            or record.get("propertyInfo", {}).get("address", {})
            or {}
        )
        zip_code = (
            address_block.get("zip")
            or address_block.get("postalCode")
            or address_block.get("zipCode")
            or record.get("zip")
        )

        price = (
            record.get("mls_listing_price")
            or record.get("currentPrice")
            or record.get("price")
            or record.get("listPrice")
        )

        summary = {
            "property_id": property_id,
            "mls_id": record.get("mls_id") or record.get("listingId") or record.get("mlsId"),
            "address": address_block.get("label")
            or address_block.get("formatted")
            or record.get("fullAddress")
            or record.get("address"),
            "city": address_block.get("city") or record.get("city"),
            "state": address_block.get("state") or record.get("state"),
            "zip_code": zip_code,
            "price": _safe_numeric(price),
            "beds": _safe_numeric(record.get("beds") or record.get("bedrooms")),
            "baths": _safe_numeric(record.get("baths") or record.get("bathrooms")),
            "sqft": _safe_numeric(record.get("sqft") or record.get("squareFeet")),
            "lot_sqft": _safe_numeric(record.get("lotSqft") or record.get("lotSquareFeet")),
            "geocode": {
                "lat": record.get("latitude") or record.get("lat"),
                "lon": record.get("longitude") or record.get("lng"),
            },
            "summary": record,
        }

        return summary

    def _enrich_listing(self, listing: Dict[str, Any], *, include_mls_detail: bool) -> Dict[str, Any]:
        property_id = listing["property_id"]
        detail_payload: Dict[str, Any] = {}
        mls_payload: Dict[str, Any] = {}

        try:
            detail_payload = self.client.get_property_detail(property_id=property_id, use_cache=True)
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("Failed property_detail for %s: %s", property_id, exc)

        mls_id = listing.get("mls_id")
        if include_mls_detail and mls_id:
            try:
                mls_payload = self.client.get_mls_detail(mls_id=mls_id, use_cache=True)
            except Exception as exc:  # pragma: no cover
                logger.debug("MLS detail fetch failed for %s: %s", mls_id, exc)

        enriched = dict(listing)
        enriched["property_detail"] = detail_payload
        enriched["mls_detail"] = mls_payload

        property_info = (
            detail_payload.get("propertyInfo")
            or detail_payload.get("property_info")
            or {}
        )
        prop_type = PropertyUseMapper.normalize(property_info)
        enriched.setdefault("metadata", {})
        enriched["metadata"].update(
            {
                "property_type_label": prop_type.label,
                "property_type_category": prop_type.category,
                "raw_property_use_code": property_info.get("propertyUseCode"),
            }
        )

        enriched["description"] = self._extract_description(detail_payload, mls_payload)
        enriched.update(self._extract_dates(detail_payload, mls_payload))
        enriched["priceHistory"] = self._extract_price_history(detail_payload, mls_payload)

        enriched["beds"] = enriched.get("beds") or _safe_numeric(property_info.get("bedrooms"))
        enriched["baths"] = enriched.get("baths") or _safe_numeric(
            property_info.get("bathrooms") or property_info.get("bathTotal")
        )
        enriched["sqft"] = enriched.get("sqft") or _safe_numeric(
            property_info.get("livingSquareFeet") or property_info.get("buildingSquareFeet")
        )
        enriched["lot_sqft"] = enriched.get("lot_sqft") or _safe_numeric(property_info.get("lotSquareFeet"))

        return enriched

    def _extract_description(self, detail: Dict[str, Any], mls_detail: Dict[str, Any]) -> Optional[str]:
        candidates = [
            (mls_detail or {}).get("publicRemarks"),
            (detail.get("mlsHistory", [{}])[0] if detail else {}).get("publicRemarks"),
            detail.get("publicRemarks"),
            (detail.get("propertyInfo", {}) or {}).get("description"),
        ]
        for item in candidates:
            if item:
                return item
        return None

    def _extract_dates(self, detail: Dict[str, Any], mls_detail: Dict[str, Any]) -> Dict[str, Optional[str]]:
        list_date = None
        pending_date = None
        sold_date = None

        mls_history = detail.get("mlsHistory") if isinstance(detail, dict) else None
        if isinstance(mls_history, list):
            sorted_events = sorted(
                (evt for evt in mls_history if isinstance(evt, dict)),
                key=lambda evt: evt.get("statusDate") or evt.get("status_date") or "",
            )
            active_event = _first(sorted_events, lambda evt: str(evt.get("status", "")).lower() in {"active", "listed"})
            pending_event = _first(sorted_events, lambda evt: str(evt.get("status", "")).lower() in {"pending", "contingent"})
            sold_event = _first(sorted_events, lambda evt: str(evt.get("status", "")).lower() in {"sold", "closed", "offmarket"})

            list_date = _extract_date(active_event)
            pending_date = _extract_date(pending_event)
            sold_date = _extract_date(sold_event)

        sale_history = detail.get("saleHistory") if isinstance(detail, dict) else None
        if not sold_date and isinstance(sale_history, list) and sale_history:
            sold_date = sale_history[0].get("saleDate")

        if not list_date:
            if isinstance(mls_detail, dict):
                list_date = mls_detail.get("listDate") or mls_detail.get("listingDate")
            if not list_date and isinstance(detail, dict):
                list_date = detail.get("listDate")

        return {
            "datePosted": list_date,
            "pendingDate": pending_date,
            "dateSold": sold_date,
        }

    def _extract_price_history(self, detail: Dict[str, Any], mls_detail: Dict[str, Any]) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        mls_history = detail.get("mlsHistory") if isinstance(detail, dict) else None
        if isinstance(mls_history, list):
            for event in mls_history:
                if not isinstance(event, dict):
                    continue
                history.append(
                    {
                        "event": _format_status(event.get("status")),
                        "date": event.get("statusDate") or event.get("status_date"),
                        "price": event.get("price") or event.get("listingPrice"),
                    }
                )

        if not history and isinstance(mls_detail, dict):
            for event in mls_detail.get("history", []) or []:
                if not isinstance(event, dict):
                    continue
                history.append(
                    {
                        "event": event.get("event") or _format_status(event.get("status")),
                        "date": event.get("date"),
                        "price": event.get("price"),
                    }
                )

        sale_history = detail.get("saleHistory") if isinstance(detail, dict) else None
        if isinstance(sale_history, list):
            for sale in sale_history:
                if not isinstance(sale, dict):
                    continue
                history.append(
                    {
                        "event": _format_status(sale.get("transactionType") or "sold"),
                        "date": sale.get("saleDate") or sale.get("recordingDate"),
                        "price": sale.get("saleAmount"),
                    }
                )

        return history


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------


def _safe_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_status(status: Optional[str]) -> str:
    if not status:
        return "Unknown"
    label = str(status).strip().lower()
    if not label:
        return "Unknown"
    mapping = {
        "active": "Listed",
        "listed": "Listed",
        "comingsoon": "Listed",
        "coming soon": "Listed",
        "pending": "Pending sale",
        "contingent": "Pending sale",
        "undercontract": "Pending sale",
        "under contract": "Pending sale",
        "sold": "Sold",
        "closed": "Sold",
        "offmarket": "Sold",
        "off market": "Sold",
        "withdrawn": "Withdrawn",
        "expired": "Expired",
        "price drop": "Price change",
    }
    return mapping.get(label, label.title())


def _extract_date(event: Optional[Dict[str, Any]]) -> Optional[str]:
    if not event:
        return None
    return event.get("statusDate") or event.get("status_date") or event.get("date")
