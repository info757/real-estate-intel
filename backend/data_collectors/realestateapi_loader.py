"""High-level helpers for fetching listings via RealEstateApi.

This module provides a drop-in replacement for the RapidAPI-based
`SafeListingsScraper` so the ML pipelines can source MLS-grade data via the
RealEstateApi MCP/REST endpoints.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path
import json

from config.settings import settings
from backend.data_collectors.realestateapi_client import (
    RealEstateAPIClient,
    PropertyUseMapper,
)

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache/listings/realestateapi")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _first(items: Iterable[Any], predicate) -> Optional[Any]:
    for item in items:
        if predicate(item):
            return item
    return None


def _parse_datetime_value(value: Any) -> Optional[datetime]:
    """Parse a date/time string into a :class:`datetime` instance."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # Treat as Unix timestamp (seconds)
        try:
            return datetime.fromtimestamp(value)
        except (OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith(" UTC"):
            text = text[:-4] + "+00:00"
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


def _normalize_to_naive(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _extract_sale_datetime_from_history(history: Any) -> Optional[datetime]:
    if not isinstance(history, list):
        return None
    for entry in history:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "")).lower()
        if status in {"sold", "closed", "offmarket"}:
            dt = _normalize_to_naive(_parse_datetime_value(entry.get("statusDate") or entry.get("status_date")))
            if dt:
                return dt
        dt = _normalize_to_naive(_parse_datetime_value(entry.get("saleDate") or entry.get("recordingDate")))
        if dt:
            return dt
    return None


def _extract_sale_datetime_from_search_record(record: Dict[str, Any]) -> Optional[datetime]:
    if not isinstance(record, dict):
        return None
    for key in (
        "mlsLastSaleDate",
        "lastSaleDate",
        "saleDate",
        "soldDate",
        "recordingDate",
    ):
        dt = _normalize_to_naive(_parse_datetime_value(record.get(key)))
        if dt:
            return dt
    sale_history = record.get("saleHistory")
    dt = _extract_sale_datetime_from_history(sale_history)
    if dt:
        return dt
    mls_history = record.get("mlsHistory")
    dt = _extract_sale_datetime_from_history(mls_history)
    if dt:
        return dt
    return None


def _extract_sale_datetime_from_listing(listing: Dict[str, Any]) -> Optional[datetime]:
    if not isinstance(listing, dict):
        return None
    for key in (
        "dateSold",
        "sold_date",
        "soldDate",
        "mlsLastSaleDate",
        "lastSaleDate",
    ):
        dt = _normalize_to_naive(_parse_datetime_value(listing.get(key)))
        if dt:
            return dt
    price_history = listing.get("priceHistory")
    dt = _extract_sale_datetime_from_history(price_history)
    if dt:
        return dt

    detail = listing.get("property_detail_raw")
    dt = _extract_sale_datetime_from_search_record(detail or {})
    if dt:
        return dt

    mls_detail = listing.get("mls_detail_raw")
    dt = _extract_sale_datetime_from_search_record(mls_detail or {})
    if dt:
        return dt
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
        max_results: Optional[int] = 250,
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

        search_cutoff = _normalize_to_naive(datetime.utcnow() - timedelta(days=days_back))
        listings: List[Dict[str, Any]] = []
        page_size = 250 if max_results is None or max_results <= 0 else min(max_results, 250)
        result_index = 0
        max_server_results = settings.realestateapi_max_results_per_zip or 0
        effective_cap: Optional[int] = max_server_results if max_server_results > 0 else None
        if max_results and max_results > 0:
            effective_cap = min(effective_cap, max_results) if effective_cap else max_results

        seen_property_ids: set[str] = set()
        raw_seen = 0
        prefilter_skipped = 0
        stale_skipped = 0
        dedupe_skipped = 0
        total_available: Optional[int] = None
        cap_triggered = False

        while True:
            current_page_size = page_size
            if effective_cap:
                remaining_cap = effective_cap - result_index
                if remaining_cap <= 0:
                    cap_triggered = True
                    break
                current_page_size = max(1, min(page_size, remaining_cap))

            response = self.client.search_properties(
                filters={
                    "zip": zip_code,
                    "resultIndex": result_index,
                },
                page_size=current_page_size,
                include_count=False,
            )

            records = self._extract_records(response)
            if total_available is None:
                total_available = response.get("resultCount")
                if total_available is not None:
                    logger.info(
                        "RealEstateApi reports %s raw records for %s (cutoff=%s days)",
                        total_available,
                        zip_code,
                        days_back,
                    )
                if effective_cap and total_available and total_available > effective_cap:
                    logger.warning(
                        "Applying raw fetch cap of %s records for %s (available=%s)",
                        effective_cap,
                        zip_code,
                        total_available,
                    )

            if not records:
                break

            logger.info(
                "RealEstateApi returned %d records starting at index %d",
                len(records),
                result_index,
            )

            for record in records:
                raw_seen += 1

                property_id = (
                    record.get("propertyId")
                    or record.get("property_id")
                    or record.get("id")
                    or record.get("_id")
                )
                if property_id and property_id in seen_property_ids:
                    dedupe_skipped += 1
                    continue

                if not self._record_is_recent_enough(record, search_cutoff):
                    prefilter_skipped += 1
                    continue

                normalized = self._normalize_search_record(record)
                if not normalized:
                    continue

                if property_id:
                    seen_property_ids.add(property_id)

                detailed = self._enrich_listing(normalized, include_mls_detail=include_mls_detail)
                if not self._listing_is_recent_enough(detailed, search_cutoff):
                    stale_skipped += 1
                    continue

                listings.append(detailed)
                if max_results and max_results > 0 and len(listings) >= max_results:
                    break

            next_index = response.get("resultIndex")
            total_count = response.get("resultCount")
            if max_results and max_results > 0 and len(listings) >= max_results:
                break
            if next_index is None or total_count is None or next_index >= total_count:
                break
            if next_index == result_index:
                result_index += len(records)
            else:
                result_index = next_index

        if cap_triggered and effective_cap:
            logger.warning(
                "Stopped fetching %s after reaching configured cap of %s raw records",
                zip_code,
                effective_cap,
            )

        logger.info(
            "Collected %d sold listings for %s (raw=%d, deduped=%d, prefilter_skip=%d, stale_skip=%d)",
            len(listings),
            zip_code,
            raw_seen,
            dedupe_skipped,
            prefilter_skipped,
            stale_skipped,
        )
        self._persist_cache(zip_code, days_back, listings, max_results)
        return listings

    def _persist_cache(self, zip_code: str, days_back: int, listings: List[Dict[str, Any]], max_results: Optional[int]):
        suffix = "maxall" if max_results is None or max_results <= 0 else f"max{max_results}"
        cache_file = CACHE_ROOT / f"listings_{zip_code}_{days_back}days_{suffix}.json"
        try:
            with cache_file.open("w", encoding="utf-8") as fh:
                json.dump(listings, fh, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to write cache for %s: %s", zip_code, exc)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    def _record_is_recent_enough(self, record: Dict[str, Any], cutoff: datetime) -> bool:
        sale_dt = _extract_sale_datetime_from_search_record(record)
        if sale_dt is None:
            return True
        return sale_dt >= cutoff

    def _listing_is_recent_enough(self, listing: Dict[str, Any], cutoff: datetime) -> bool:
        sale_dt = _extract_sale_datetime_from_listing(listing)
        if sale_dt is None:
            return False
        return sale_dt >= cutoff
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
            detail_response = self.client.get_property_detail(property_id=property_id, use_cache=True)
            detail_payload = detail_response.get("data") if isinstance(detail_response, dict) else detail_response
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("Failed property_detail for %s: %s", property_id, exc)
            detail_payload = {}

        mls_id = listing.get("mls_id")
        if include_mls_detail and mls_id:
            try:
                mls_response = self.client.get_mls_detail(mls_id=mls_id, use_cache=True)
                mls_payload = mls_response.get("data") if isinstance(mls_response, dict) else mls_response
            except Exception as exc:  # pragma: no cover
                logger.debug("MLS detail fetch failed for %s: %s", mls_id, exc)
                mls_payload = {}
        else:
            mls_payload = {}

        enriched = dict(listing)
        enriched["property_detail_raw"] = detail_payload
        enriched["mls_detail_raw"] = mls_payload

        property_info = (
            detail_payload.get("propertyInfo")
            if isinstance(detail_payload, dict)
            else {}
        ) or {}
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
        mls_history_list = []
        if isinstance(detail, dict):
            history = detail.get("mlsHistory")
            if isinstance(history, list):
                mls_history_list = history

        candidates = [
            (mls_detail or {}).get("publicRemarks"),
            (mls_history_list[0] if mls_history_list else {}).get("publicRemarks") if mls_history_list else None,
            detail.get("publicRemarks") if isinstance(detail, dict) else None,
            ((detail.get("propertyInfo", {}) or {}).get("description") if isinstance(detail, dict) else None),
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
                list_date = detail.get("mlsListingDate") or detail.get("listDate")

        if not pending_date and isinstance(detail, dict):
            pending_date = detail.get("mlsPendingDate") or detail.get("pendingDate")

        if not sold_date and isinstance(detail, dict):
            sold_date = detail.get("mlsLastSaleDate") or detail.get("lastSaleDate")

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
