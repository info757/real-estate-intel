"""
Attom Data API Client
Handles authentication, rate limiting, caching, and API requests.
"""

import time
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttomRateLimiter:
    """Rate limiter for Attom API to stay within daily limits."""
    
    def __init__(self, max_requests_per_day: int = 500):
        self.max_requests = max_requests_per_day
        self.requests_made = 0
        self.reset_time = datetime.now() + timedelta(days=1)
        self.request_history = []
    
    def can_make_request(self) -> bool:
        """Check if we can make another API request."""
        self._reset_if_needed()
        return self.requests_made < self.max_requests
    
    def record_request(self):
        """Record that a request was made."""
        self._reset_if_needed()
        self.requests_made += 1
        self.request_history.append(datetime.now())
        logger.info(f"Attom API requests: {self.requests_made}/{self.max_requests}")
    
    def _reset_if_needed(self):
        """Reset counter if 24 hours have passed."""
        if datetime.now() >= self.reset_time:
            self.requests_made = 0
            self.reset_time = datetime.now() + timedelta(days=1)
            self.request_history = []
            logger.info("Attom API rate limit counter reset")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        self._reset_if_needed()
        return {
            "requests_made": self.requests_made,
            "max_requests": self.max_requests,
            "remaining": self.max_requests - self.requests_made,
            "reset_time": self.reset_time.isoformat()
        }


class AttomCache:
    """Simple file-based cache for Attom API responses."""
    
    def __init__(self, cache_dir: str = "cache/attom"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key from endpoint and parameters."""
        param_str = json.dumps(params, sort_keys=True)
        key_data = f"{endpoint}:{param_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, endpoint: str, params: Dict[str, Any], ttl_days: int) -> Optional[Dict[str, Any]]:
        """Get cached response if still valid."""
        cache_key = self._get_cache_key(endpoint, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            # Check if cache is still valid
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age < timedelta(days=ttl_days):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        logger.debug(f"Cache hit for {endpoint}")
                        return data
                except Exception as e:
                    logger.error(f"Error reading cache: {e}")
        
        return None
    
    def set(self, endpoint: str, params: Dict[str, Any], data: Dict[str, Any]):
        """Store response in cache."""
        cache_key = self._get_cache_key(endpoint, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logger.debug(f"Cached response for {endpoint}")
        except Exception as e:
            logger.error(f"Error writing cache: {e}")


class AttomDataClient:
    """
    Comprehensive client for Attom Data API.
    Handles authentication, rate limiting, caching, and all key endpoints.
    """
    
    def __init__(self):
        self.api_key = settings.attom_api_key
        self.base_url = settings.attom_api_base_url
        self.rate_limiter = AttomRateLimiter()
        self.cache = AttomCache()
        self.session = self._create_session()
        
        if not self.api_key:
            logger.warning("Attom API key not configured")
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl_days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """
        Make API request with rate limiting and caching.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            cache_ttl_days: Cache time-to-live in days
        
        Returns:
            API response as dictionary or None if error
        """
        if not self.api_key:
            logger.error("Attom API key not configured")
            return None
        
        params = params or {}
        
        # Check cache first
        cached_data = self.cache.get(endpoint, params, cache_ttl_days)
        if cached_data:
            return cached_data
        
        # Check rate limit
        if not self.rate_limiter.can_make_request():
            logger.warning("Attom API rate limit reached, using cached data only")
            return None
        
        # Make request
        url = f"{self.base_url}{endpoint}"
        headers = {
            "apikey": self.api_key,
            "Accept": "application/json"
        }
        
        try:
            logger.info(f"Attom API request: {endpoint}")
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            self.rate_limiter.record_request()
            
            response.raise_for_status()
            data = response.json()
            
            # Cache successful response
            self.cache.set(endpoint, params, data)
            
            return data
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Attom API HTTP error: {e}")
            if e.response.status_code == 401:
                logger.error("Invalid API key")
            elif e.response.status_code == 404:
                logger.warning(f"No data found for: {params}")
            return None
        except Exception as e:
            logger.error(f"Attom API request failed: {e}")
            return None
    
    # ===== PROPERTY DATA ENDPOINTS =====
    
    def get_property_details(
        self,
        address: str,
        zip_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive property details.
        
        Args:
            address: Street address
            zip_code: ZIP code
        
        Returns:
            Property details including characteristics, tax info, etc.
        """
        endpoint = "/propertyapi/v1.0.0/property/detail"
        params = {
            "address1": address,
            "address2": zip_code
        }
        return self._make_request(endpoint, params, cache_ttl_days=30)
    
    def get_properties_by_zip(
        self,
        zip_code: str,
        page: int = 1,
        page_size: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        Get properties in a ZIP code.
        
        Args:
            zip_code: ZIP code
            page: Page number
            page_size: Results per page
        
        Returns:
            List of properties
        """
        endpoint = "/propertyapi/v1.0.0/property/address"
        params = {
            "postalcode": zip_code,
            "page": page,
            "pagesize": min(page_size, 100)
        }
        return self._make_request(endpoint, params, cache_ttl_days=7)
    
    # ===== SALES HISTORY ENDPOINTS =====
    
    def get_sales_history(
        self,
        zip_code: str,
        page: int = 1,
        page_size: int = 50,
        min_sale_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get sales history for a ZIP code.
        
        Args:
            zip_code: ZIP code
            page: Page number
            page_size: Results per page
            min_sale_date: Minimum sale date (YYYY-MM-DD) - Note: may not be supported by API
        
        Returns:
            Sales transactions with property details
        """
        endpoint = "/propertyapi/v1.0.0/sale/detail"
        params = {
            "postalcode": zip_code,
            "page": page,
            "pagesize": min(page_size, 100)
        }
        
        # Note: minsaledate parameter may not be supported
        # Filter will be done client-side if needed
        
        return self._make_request(endpoint, params, cache_ttl_days=7)
    
    def get_property_sales_history(
        self,
        address: str,
        zip_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get sales history for a specific property.
        
        Args:
            address: Street address
            zip_code: ZIP code
        
        Returns:
            Property sales history
        """
        endpoint = "/propertyapi/v1.0.0/sale/detail"
        params = {
            "address1": address,
            "address2": zip_code
        }
        return self._make_request(endpoint, params, cache_ttl_days=30)
    
    # ===== MARKET TRENDS ENDPOINTS =====
    
    def get_market_trends(
        self,
        zip_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get market trends for a ZIP code by analyzing sales data.
        
        Args:
            zip_code: ZIP code
        
        Returns:
            Recent sales data for trend analysis
        """
        # Note: Free tier may not have dedicated trend endpoints
        # Use sales data instead
        endpoint = "/propertyapi/v1.0.0/sale/detail"
        params = {
            "postalcode": zip_code,
            "pagesize": 100
        }
        return self._make_request(endpoint, params, cache_ttl_days=1)
    
    # ===== ADDITIONAL HELPER METHODS =====
    
    def get_all_sales_paginated(
        self,
        zip_code: str,
        max_pages: int = 10,
        min_sale_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all sales for a ZIP code across multiple pages.
        
        Args:
            zip_code: ZIP code
            max_pages: Maximum number of pages to fetch
            min_sale_date: Minimum sale date (YYYY-MM-DD) for client-side filtering
        
        Returns:
            List of all properties with sales data
        """
        all_properties = []
        
        for page in range(1, max_pages + 1):
            result = self.get_sales_history(zip_code, page=page, page_size=100)
            
            if not result or 'property' not in result:
                logger.info(f"Stopping at page {page}: No result or no property key")
                break
            
            raw_properties = result['property']
            
            # If API returns no properties (before filtering), we've reached the end
            if len(raw_properties) == 0:
                logger.info(f"Stopping at page {page}: API returned empty page")
                break
            
            # Client-side date filtering if needed
            if min_sale_date:
                properties = [
                    p for p in raw_properties
                    if p.get('sale', {}).get('saleTransDate', '') >= min_sale_date
                ]
            else:
                properties = raw_properties
            
            all_properties.extend(properties)
            
            # Check if we've reached the last page based on API's total count
            status = result.get('status', {})
            total = status.get('total', 0)
            raw_count = len(raw_properties)
            current_count = len(all_properties)
            
            # Stop if we've fetched all available properties from API (before filtering)
            # Don't stop just because filtered results are empty - more pages might have matches
            if total > 0 and (page * 100) >= total:
                logger.info(f"Reached end: page {page} * 100 >= total {total}")
                break
            
            logger.info(f"Fetched page {page}: {raw_count} raw properties, {len(properties)} after date filter, {current_count} total collected")
        
        return all_properties
    
    # ===== UTILITY METHODS =====
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit usage statistics."""
        return self.rate_limiter.get_usage_stats()
    
    def test_connection(self) -> bool:
        """
        Test API connection and authentication.
        
        Returns:
            True if connection successful, False otherwise
        """
        # Try a simple request
        result = self.get_properties_by_zip("27513", page_size=1)  # Cary, NC
        return result is not None


# Global client instance
attom_client = AttomDataClient()

