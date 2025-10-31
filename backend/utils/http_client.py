"""
HTTP client utilities with rate limiting and retry logic.
"""

import time
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, Any, Optional
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitedClient:
    """HTTP client with rate limiting and retry logic."""
    
    def __init__(self, delay: float = None, max_retries: int = None, user_agent: str = None):
        self.delay = delay or settings.rate_limit_delay
        self.max_retries = max_retries or settings.max_retries
        self.user_agent = user_agent or settings.user_agent
        self.last_request_time = 0
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent
        })
    
    def _wait_if_needed(self):
        """Wait if necessary to respect rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> requests.Response:
        """Make a GET request with rate limiting and retry logic."""
        self._wait_if_needed()
        
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)
        
        logger.info(f"GET {url}")
        response = self.session.get(url, params=params, headers=request_headers, timeout=30)
        response.raise_for_status()
        return response
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def post(self, url: str, data: Optional[Dict[str, Any]] = None, json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> requests.Response:
        """Make a POST request with rate limiting and retry logic."""
        self._wait_if_needed()
        
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)
        
        logger.info(f"POST {url}")
        response = self.session.post(url, data=data, json=json, headers=request_headers, timeout=30)
        response.raise_for_status()
        return response
    
    def close(self):
        """Close the session."""
        self.session.close()


# Global HTTP client instance
http_client = RateLimitedClient()

