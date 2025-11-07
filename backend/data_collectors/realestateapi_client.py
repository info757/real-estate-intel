"""RealEstateApi client utilities.

This module centralizes access to the RealEstateApi platform so higher-level
pipelines can swap from RapidAPI/Zillow to the MCP-backed dataset without
rewriting downstream logic.

Key capabilities:
- Authenticated REST calls with retries, timeouts, and lightweight caching
- Wrappers for property search, property detail, and MLS detail endpoints
- Helpers for mapping property use codes/types to BuildOptima property filters

Notes
-----
* The RealEstateApi team exposes the same datasets via a Production MCP server.
  The `use_realestateapi_mcp` flag is wired in but currently falls back to the
  REST HTTP endpoints until the streaming MPC protocol is implemented.
* Endpoint paths default to the documented `v2` routes; override them via
  keyword arguments if your account is provisioned differently.

Documentation References:
- MCP connection guide: https://github.com/RealEstateApi/docs/blob/main/reference/mcp-server.md
- Property detail schema: https://developer.realestateapi.com/reference/property-detail-response-object
- MLS detail schema: https://developer.realestateapi.com/reference/mls-detail-api
- Property use codes: https://developer.realestateapi.com/reference/property-use-codes-reference
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import settings

logger = logging.getLogger(__name__)


DEFAULT_CACHE_DIR = Path("cache/realestateapi")
PROPERTY_DETAIL_ENDPOINT = "/v2/property/detail"
PROPERTY_SEARCH_ENDPOINT = "/v2/property/search"
MLS_DETAIL_ENDPOINT = "/v2/mls/detail"
DEFAULT_TIMEOUT_SECONDS = 30


class RealEstateAPIError(RuntimeError):
    """Domain-specific error for RealEstateApi failures."""


class RealEstateAPICache:
    """Lightweight file cache to minimize duplicate RealEstateApi calls."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, endpoint: str, payload: Dict[str, Any]) -> str:
        raw = f"{endpoint}:{json.dumps(payload, sort_keys=True)}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, endpoint: str, payload: Dict[str, Any], ttl: timedelta) -> Optional[Dict[str, Any]]:
        cache_key = self._key(endpoint, payload)
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age > ttl:
            return None

        try:
            with cache_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            logger.warning("RealEstateApi cache decode error for %s", cache_file)
            return None

    def set(self, endpoint: str, payload: Dict[str, Any], data: Dict[str, Any]) -> None:
        cache_key = self._key(endpoint, payload)
        cache_file = self.cache_dir / f"{cache_key}.json"
        with cache_file.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)


@dataclass(frozen=True)
class PropertyType:
    """Simple structure describing a BuildOptima property type."""

    label: str
    category: str


# Mapping of RealEstateApi property use codes to BuildOptima property type labels.
_PROPERTY_USE_CODE_MAP: Dict[int, PropertyType] = {
    357: PropertyType("Multi-Family", "multi_family"),
    358: PropertyType("Multi-Family", "multi_family"),
    359: PropertyType("Multi-Family", "multi_family"),
    360: PropertyType("Multi-Family", "multi_family"),
    361: PropertyType("Multi-Family", "multi_family"),
    362: PropertyType("Multi-Family", "multi_family"),
    363: PropertyType("Single Family Home", "single_family"),
    364: PropertyType("Single Family Home", "single_family"),
    365: PropertyType("Single Family Home", "single_family"),
    366: PropertyType("Condo", "condo"),
    367: PropertyType("Condo", "condo"),
    368: PropertyType("Multi-Family", "multi_family"),
    369: PropertyType("Multi-Family", "multi_family"),
    370: PropertyType("Multi-Family", "multi_family"),
    371: PropertyType("Single Family Home", "single_family"),
    372: PropertyType("Multi-Family", "multi_family"),
    373: PropertyType("Mobile / Manufactured", "mobile"),
    374: PropertyType("Multi-Family", "multi_family"),
    375: PropertyType("Single Family Home", "single_family"),
    376: PropertyType("Single Family Home", "single_family"),
    377: PropertyType("Single Family Home", "single_family"),
    378: PropertyType("Multi-Family", "multi_family"),
    379: PropertyType("Condo", "condo"),
    380: PropertyType("Single Family Home", "single_family"),
    381: PropertyType("Multi-Family", "multi_family"),
    382: PropertyType("Row House", "single_family"),
    383: PropertyType("Single Family Home", "single_family"),
    384: PropertyType("Single Family Home", "single_family"),
    385: PropertyType("Single Family Home", "single_family"),
    386: PropertyType("Townhome", "townhome"),
    387: PropertyType("Timeshare", "other"),
    388: PropertyType("Multi-Family", "multi_family"),
    389: PropertyType("Residential Land", "land"),
    390: PropertyType("Single Family Home", "single_family"),
    391: PropertyType("Other", "other"),
    392: PropertyType("Residential Land", "land"),
    393: PropertyType("Commercial Land", "other"),
    397: PropertyType("Miscellaneous Land", "land"),
    400: PropertyType("Recreational Land", "land"),
    401: PropertyType("Residential Land", "land"),
    402: PropertyType("Under Construction", "single_family"),
    447: PropertyType("Tiny House", "single_family"),
    452: PropertyType("Garden Home", "single_family"),
    1023: PropertyType("Accessory Dwelling Unit", "adu"),
}

# Fallback string-based mapping (case insensitive) when the API returns descriptive text.
_PROPERTY_USE_STRING_MAP: Dict[str, PropertyType] = {
    "single family residence": PropertyType("Single Family Home", "single_family"),
    "single family": PropertyType("Single Family Home", "single_family"),
    "townhouse": PropertyType("Townhome", "townhome"),
    "row house": PropertyType("Row House", "single_family"),
    "condominium": PropertyType("Condo", "condo"),
    "condo": PropertyType("Condo", "condo"),
    "multi-family": PropertyType("Multi-Family", "multi_family"),
    "multifamily": PropertyType("Multi-Family", "multi_family"),
    "apartment": PropertyType("Multi-Family", "multi_family"),
    "duplex": PropertyType("Multi-Family", "multi_family"),
    "triplex": PropertyType("Multi-Family", "multi_family"),
    "quadplex": PropertyType("Multi-Family", "multi_family"),
    "mobile home": PropertyType("Mobile / Manufactured", "mobile"),
    "manufactured": PropertyType("Mobile / Manufactured", "mobile"),
    "vacant land": PropertyType("Residential Land", "land"),
    "residential vacant land": PropertyType("Residential Land", "land"),
    "land": PropertyType("Residential Land", "land"),
    "garden home": PropertyType("Single Family Home", "single_family"),
    "accessory dwelling unit": PropertyType("Accessory Dwelling Unit", "adu"),
}

_DEFAULT_PROPERTY_TYPE = PropertyType("Single Family Home", "single_family")


class PropertyUseMapper:
    """Utility helpers for working with RealEstateApi property use metadata."""

    @staticmethod
    def from_code(code: Optional[Union[int, str]]) -> PropertyType:
        if code is None:
            return _DEFAULT_PROPERTY_TYPE
        try:
            value = int(code)
        except (TypeError, ValueError):
            return PropertyUseMapper.from_string(str(code))
        return _PROPERTY_USE_CODE_MAP.get(value, _DEFAULT_PROPERTY_TYPE)

    @staticmethod
    def from_string(label: Optional[str]) -> PropertyType:
        if not label:
            return _DEFAULT_PROPERTY_TYPE
        return _PROPERTY_USE_STRING_MAP.get(label.strip().lower(), _DEFAULT_PROPERTY_TYPE)

    @staticmethod
    def normalize(property_info: Dict[str, Any]) -> PropertyType:
        """Derive a BuildOptima property type from a RealEstateApi record."""
        if not property_info:
            return _DEFAULT_PROPERTY_TYPE
        code = property_info.get("propertyUseCode") or property_info.get("property_use_code")
        if code:
            return PropertyUseMapper.from_code(code)

        use_text = (
            property_info.get("propertyUse")
            or property_info.get("property_use")
            or property_info.get("type")
            or property_info.get("propertyType")
        )
        return PropertyUseMapper.from_string(use_text)

    @staticmethod
    def summarize(record: Dict[str, Any]) -> Dict[str, Any]:
        """Return a summary block for UI/model consumption."""
        property_info = record.get("propertyInfo") or record.get("property_info") or {}
        prop_type = PropertyUseMapper.normalize(property_info)
        return {
            "property_type_label": prop_type.label,
            "property_type_category": prop_type.category,
            "raw_property_use_code": property_info.get("propertyUseCode"),
            "raw_property_use": property_info.get("propertyUse"),
        }


class RealEstateAPIClient:
    """HTTP client for RealEstateApi endpoints with optional caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mcp_url: Optional[str] = None,
        use_mcp: Optional[bool] = None,
        cache: Optional[RealEstateAPICache] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key or settings.realestateapi_api_key
        self.base_url = (base_url or settings.realestateapi_base_url).rstrip("/")
        self.mcp_url = (mcp_url or settings.realestateapi_mcp_url).rstrip("/")
        self.use_mcp = use_mcp if use_mcp is not None else settings.use_realestateapi_mcp
        self.timeout = timeout
        self.cache = cache or RealEstateAPICache()
        self.session = self._create_session()

        if not self.api_key:
            logger.warning("RealEstateApi API key is not configured; requests will fail.")
        if self.use_mcp:
            logger.info(
                "RealEstateApi MCP mode requested. Streaming support will forward to REST until MCP transport is wired."
            )

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        })
        return session

    # ------------------------------------------------------------------
    # Low-level request helpers
    # ------------------------------------------------------------------
    def _resolve_url(self, endpoint: str) -> str:
        if self.use_mcp:
            # TODO: Wire up MCP SSE bridge. For now, log and fall back to REST endpoint.
            logger.debug("Using REST fallback for endpoint %s while MCP integration is pending", endpoint)
        return f"{self.base_url}{endpoint}"

    def _handle_response(self, response: Response, endpoint: str) -> Dict[str, Any]:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise RealEstateAPIError(f"RealEstateApi request failed [{endpoint}]: {detail}") from exc

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise RealEstateAPIError(f"Failed to parse RealEstateApi response for {endpoint}") from exc

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        cache_ttl: timedelta = timedelta(hours=6),
    ) -> Dict[str, Any]:
        payload = payload or {}
        if use_cache:
            cached = self.cache.get(endpoint, payload, cache_ttl)
            if cached is not None:
                logger.debug("RealEstateApi cache hit for %s", endpoint)
                return cached

        url = self._resolve_url(endpoint)
        logger.debug("RealEstateApi %s %s payload=%s", method.upper(), url, payload)

        response = self.session.request(
            method=method.upper(),
            url=url,
            json=payload if method.upper() in {"POST", "PUT"} else None,
            params=payload if method.upper() == "GET" else None,
            timeout=self.timeout,
        )

        data = self._handle_response(response, endpoint)
        if use_cache:
            self.cache.set(endpoint, payload, data)
        return data

    # ------------------------------------------------------------------
    # High-level helpers for core endpoints
    # ------------------------------------------------------------------
    def search_properties(
        self,
        filters: Dict[str, Any],
        *,
        page_size: int = 250,
        page: int = 1,
        include_count: bool = False,
        endpoint: str = PROPERTY_SEARCH_ENDPOINT,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            **filters,
            "size": page_size,
            "page": page,
            "count": include_count,
        }
        return self._request("POST", endpoint, payload, use_cache=use_cache)

    def search_mls_listings(
        self,
        *,
        status: str = "active",
        geography: Optional[Dict[str, Any]] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        endpoint: str = PROPERTY_SEARCH_ENDPOINT,
        **filters: Any,
    ) -> Dict[str, Any]:
        status_flag = {
            "active": {"mls_active": True},
            "pending": {"mls_pending": True},
            "cancelled": {"mls_cancelled": True},
            "sold": {"mls_sold": True},
        }.get(status.lower())
        if status_flag is None:
            raise ValueError(f"Unsupported MLS status '{status}'.")

        if "page_size" in filters:
            filters["size"] = filters.pop("page_size")
        filters.setdefault("size", 250)
        filters.setdefault("page", 1)
        filters.setdefault("count", False)

        payload: Dict[str, Any] = {**status_flag, **filters}
        if geography:
            payload.update(geography)
        if price_min is not None:
            payload["mls_listing_min"] = price_min
        if price_max is not None:
            payload["mls_listing_max"] = price_max

        return self._request("POST", endpoint, payload)

    def get_property_detail(
        self,
        *,
        property_id: Optional[str] = None,
        address: Optional[str] = None,
        endpoint: str = PROPERTY_DETAIL_ENDPOINT,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        if not property_id and not address:
            raise ValueError("Either property_id or address must be provided.")

        payload: Dict[str, Any] = {}
        if property_id:
            payload["property_id"] = property_id
        if address:
            payload["address"] = address

        return self._request("POST", endpoint, payload, use_cache=use_cache, cache_ttl=timedelta(days=1))

    def get_mls_detail(
        self,
        *,
        mls_id: Optional[str] = None,
        listing_id: Optional[str] = None,
        endpoint: str = MLS_DETAIL_ENDPOINT,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        if not mls_id and not listing_id:
            raise ValueError("Either mls_id or listing_id must be provided.")

        payload: Dict[str, Any] = {}
        if mls_id:
            payload["mls_id"] = mls_id
        if listing_id:
            payload["listing_id"] = listing_id

        return self._request("POST", endpoint, payload, use_cache=use_cache, cache_ttl=timedelta(hours=6))

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
    @staticmethod
    def extract_dom_metrics(mls_history: Iterable[Dict[str, Any]]) -> Dict[str, Optional[int]]:
        """Compute DOM to pending/sold from MLS history events."""
        pending_date = None
        sold_date = None
        list_date = None

        for event in sorted(mls_history, key=lambda e: e.get("statusDate") or e.get("status_date")):
            status = (event.get("status") or "").lower()
            status_date = event.get("statusDate") or event.get("status_date")
            if not status_date:
                continue
            if status in {"active", "listed"} and not list_date:
                list_date = status_date
            if status in {"pending", "contingent"} and not pending_date:
                pending_date = status_date
            if status in {"sold", "closed"} and not sold_date:
                sold_date = status_date

        return {
            "list_date": list_date,
            "pending_date": pending_date,
            "sold_date": sold_date,
        }

    def annotate_with_property_type(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Return record with added BuildOptima property type metadata."""
        summary = PropertyUseMapper.summarize(record)
        annotated = dict(record)
        annotated.setdefault("metadata", {})
        annotated["metadata"].update(summary)
        return annotated
