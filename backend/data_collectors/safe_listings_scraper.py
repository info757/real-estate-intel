"""
Safe Listings Scraper
Uses official APIs (RapidAPI, Realtor.com) to avoid ToS violations.
Falls back to Attom data if API keys not available.
"""

import requests
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafeListingsScraper:
    """
    Safely fetches listings using official APIs or Attom data.
    Avoids direct scraping to prevent ToS violations.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_attom_fallback: bool = True):
        """
        Initialize safe listings scraper.
        
        Args:
            api_key: RapidAPI key (optional, will check env var)
            use_attom_fallback: If True, fall back to Attom data when APIs unavailable
        """
        self.rapidapi_key = api_key or self._get_rapidapi_key()
        self.use_attom_fallback = use_attom_fallback
        self.attom_client = None
        
        if use_attom_fallback:
            try:
                from backend.data_collectors.attom_client import AttomDataClient
                self.attom_client = AttomDataClient()
            except Exception as e:
                logger.warning(f"Could not initialize Attom client: {e}")
    
    def _get_rapidapi_key(self) -> Optional[str]:
        """Get RapidAPI key from settings or environment."""
        try:
            from config.settings import settings
            if settings.rapidapi_key:
                return settings.rapidapi_key
        except:
            pass
        import os
        return os.getenv('RAPIDAPI_KEY', None)
    
    def fetch_listings(
        self,
        zip_code: str,
        status: str = 'active',
        max_results: int = 100,
        source: str = 'auto'  # 'auto', 'rapidapi_zillow', 'rapidapi_realtor', 'attom'
    ) -> List[Dict[str, Any]]:
        """
        Fetch listings using the safest available method.
        
        Args:
            zip_code: ZIP code to search
            status: 'active', 'pending', or 'sold'
            max_results: Maximum results to return
            source: Which source to use ('auto' tries all available)
            
        Returns:
            List of listing dictionaries with popularity metrics
        """
        if source == 'auto':
            # Try RapidAPI Realtor first (most permissive)
            if self.rapidapi_key:
                try:
                    logger.info("Attempting to fetch via RapidAPI Realtor.com...")
                    listings = self._fetch_realtor_rapidapi(zip_code, status, max_results)
                    if listings:
                        return listings
                except Exception as e:
                    logger.warning(f"RapidAPI Realtor failed: {e}")
                
                # Try RapidAPI Zillow wrapper
                try:
                    logger.info("Attempting to fetch via RapidAPI Zillow wrapper...")
                    listings = self._fetch_zillow_rapidapi(zip_code, status, max_results)
                    if listings:
                        return listings
                except Exception as e:
                    logger.warning(f"RapidAPI Zillow failed: {e}")
            
            # Fall back to Attom data
            if self.use_attom_fallback and self.attom_client:
                try:
                    logger.info("Falling back to Attom Data API...")
                    listings = self._fetch_from_attom(zip_code, status, max_results)
                    if listings:
                        return listings
                except Exception as e:
                    logger.warning(f"Attom fallback failed: {e}")
            
            logger.error("No data sources available. Please set RAPIDAPI_KEY or ensure Attom API is configured.")
            return []
        
        elif source == 'rapidapi_realtor':
            return self._fetch_realtor_rapidapi(zip_code, status, max_results)
        elif source == 'rapidapi_zillow':
            return self._fetch_zillow_rapidapi(zip_code, status, max_results)
        elif source == 'attom':
            return self._fetch_from_attom(zip_code, status, max_results)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _fetch_realtor_rapidapi(
        self,
        zip_code: str,
        status: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Fetch listings via RapidAPI Realtor.com wrapper."""
        if not self.rapidapi_key:
            raise ValueError("RapidAPI key not available")
        
        url = "https://realtor.p.rapidapi.com/properties/v3/list"
        
        # Map status
        status_map = {
            'active': 'for_sale',
            'pending': 'for_sale',  # Realtor API may not separate pending
            'sold': 'sold'
        }
        
        headers = {
            'X-RapidAPI-Key': self.rapidapi_key,
            'X-RapidAPI-Host': 'realtor.p.rapidapi.com'
        }
        
        params = {
            'postal_code': zip_code,
            'offset': 0,
            'limit': min(max_results, 200),
            'status': status_map.get(status, 'for_sale'),
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse Realtor.com response format
        listings = []
        properties = data.get('data', {}).get('home_search', {}).get('results', [])
        
        for prop in properties[:max_results]:
            listing = self._parse_realtor_listing(prop)
            if listing:
                listings.append(listing)
        
        logger.info(f"Fetched {len(listings)} listings from Realtor.com API")
        return listings
    
    def _fetch_zillow_rapidapi(
        self,
        zip_code: str,
        status: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Fetch listings via RapidAPI Zillow wrapper with pagination support."""
        if not self.rapidapi_key:
            raise ValueError("RapidAPI key not available")
        
        url = "https://zillow-com1.p.rapidapi.com/propertyExtendedSearch"
        
        headers = {
            'X-RapidAPI-Key': self.rapidapi_key,
            'X-RapidAPI-Host': 'zillow-com1.p.rapidapi.com'
        }
        
        listings = []
        page = 1
        results_per_page = 50  # API seems to return max 41-50 per page
        
        while len(listings) < max_results:
            params = {
                'location': zip_code,
                'home_type': 'Houses',  # Can be expanded to include condos, etc.
                'limit': results_per_page,
                'page': page
            }
            
            # Add status filter if supported
            if status == 'sold':
                params['status_type'] = 'RecentlySold'
            
            try:
                response = requests.get(url, headers=headers, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                
                # Check pagination metadata
                total_results = data.get('totalResultCount', 0)
                total_pages = data.get('totalPages', 1)
                current_page = data.get('currentPage', page)
                
                # Get properties from this page
                properties = data.get('props', [])
                
                if not properties:
                    logger.info(f"No more properties on page {page}")
                    break
                
                # Parse and add listings
                for prop in properties:
                    listing = self._parse_zillow_rapidapi_listing(prop)
                    if listing:
                        listings.append(listing)
                    
                    # Stop if we've reached max_results
                    if len(listings) >= max_results:
                        break
                
                logger.info(f"Fetched page {page}/{total_pages}: {len(properties)} properties, {len(listings)} total collected")
                
                # Check if we've reached the last page
                if current_page >= total_pages:
                    logger.info(f"Reached last page ({total_pages})")
                    break
                
                # Check if we've fetched all available results
                if total_results > 0 and len(listings) >= total_results:
                    logger.info(f"Fetched all {total_results} available results")
                    break
                
                page += 1
                
                # Be respectful with rate limiting
                time.sleep(1)  # Small delay between pages
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching page {page}: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error on page {page}: {e}")
                break
        
        logger.info(f"Fetched {len(listings)} listings from Zillow API (via RapidAPI)")
        return listings[:max_results]
    def fetch_sold_with_details(
        self,
        zip_code: str,
        days_back: int = 180,
        max_results: int = 100,
        fetch_details: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch recently sold listings with full detail pages."""
        if not self.rapidapi_key:
            raise ValueError("RapidAPI key not available")
        
        logger.info(f"Fetching sold listings for ZIP {zip_code}...")
        sold_listings = self._fetch_zillow_rapidapi(
            zip_code=zip_code, status='sold', max_results=max_results
        )
        
        if not fetch_details:
            return sold_listings
        
        logger.info(f"Fetching detail pages for {len(sold_listings)} listings...")
        detailed_listings = []
        
        for i, listing in enumerate(sold_listings):
            zpid = listing.get('zpid')
            detail_url = listing.get('detail_url')
            
            if not zpid and not detail_url:
                continue
            
            try:
                detail_data = self._fetch_listing_detail(zpid=zpid, detail_url=detail_url)
                if detail_data:
                    detailed_listings.append({**listing, **detail_data})
                else:
                    detailed_listings.append(listing)
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Error fetching detail: {e}")
                detailed_listings.append(listing)
        
        return detailed_listings
    
    def _fetch_listing_detail(
        self,
        zpid: Optional[str] = None,
        detail_url: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch detailed information for a single listing."""
        if not self.rapidapi_key:
            return None
        
        if zpid:
            url = "https://zillow-com1.p.rapidapi.com/property"
            params = {'zpid': zpid}
        elif detail_url:
            import re
            zpid_match = re.search(r'/(\d+)_zpid', detail_url)
            if zpid_match:
                url = "https://zillow-com1.p.rapidapi.com/property"
                params = {'zpid': zpid_match.group(1)}
            else:
                url = "https://zillow-com1.p.rapidapi.com/propertyByUrl"
                params = {'property_url': detail_url}
        else:
            return None
        
        headers = {
            'X-RapidAPI-Key': self.rapidapi_key,
            'X-RapidAPI-Host': 'zillow-com1.p.rapidapi.com'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            return {
                'description': data.get('description', ''),
                'datePosted': data.get('datePosted'),
                'dateSold': data.get('dateSold'),
                'priceHistory': data.get('priceHistory', []),
                'yearBuilt': data.get('yearBuilt'),
                'lotSize': data.get('lotSize'),
            }
        except Exception as e:
            logger.warning(f"Error fetching detail: {e}")
            return None
    

    def _fetch_from_attom(
        self,
        zip_code: str,
        status: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent sales from Attom (closest to 'active' listings we can get).
        Attom doesn't have active listings, but we can use recent sales as proxy.
        """
        if not self.attom_client:
            raise ValueError("Attom client not available")
        
        # Get recent sales (last 90 days) - closest proxy to active listings
        listings = self.attom_client.get_all_sales_paginated(
            zip_code=zip_code,
            min_sale_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
            max_pages=min((max_results // 100) + 1, 10)
        )
        
        # Convert Attom property format to listing format
        converted_listings = []
        for prop in listings[:max_results]:
            listing = self._convert_attom_to_listing(prop)
            if listing:
                converted_listings.append(listing)
        
        logger.info(f"Converted {len(converted_listings)} Attom sales to listing format")
        return converted_listings
    
    def _parse_realtor_listing(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Realtor.com API response to listing format."""
        try:
            # Realtor.com API structure (may need adjustment based on actual response)
            property_data = prop.get('property', {})
            address = property_data.get('address', {})
            
            listing = {
                'zpid': prop.get('property_id'),  # Realtor uses property_id
                'address': f"{address.get('line', '')}, {address.get('city', '')}, {address.get('state_code', '')} {address.get('postal_code', '')}".strip(),
                'zip_code': address.get('postal_code'),
                'price': property_data.get('price'),
                'beds': property_data.get('beds'),
                'baths': property_data.get('baths'),
                'sqft': property_data.get('building_size', {}).get('size'),
                'list_date': property_data.get('list_date'),
                'status': property_data.get('status'),
                'days_on_zillow': property_data.get('days_on_market'),  # May be different field
                'views': None,  # Realtor API may not provide this
                'saves': None,
                'features': self._extract_realtor_features(property_data),
                'detail_url': property_data.get('rdc_web_url'),
                'scraped_at': datetime.now().isoformat(),
            }
            
            return listing
        except Exception as e:
            logger.warning(f"Error parsing Realtor listing: {e}")
            return None
    
    def _parse_zillow_rapidapi_listing(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Zillow RapidAPI response to listing format."""
        try:
            # Extract address - can be string or dict
            address = prop.get('address', '')
            if isinstance(address, dict):
                address_str = address.get('streetAddress', {}).get('full', '')
                zip_code = address.get('zipcode', '')
            else:
                address_str = str(address) if address else ''
                # Try to extract ZIP from address string
                import re
                zip_match = re.search(r'\d{5}', address_str)
                zip_code = zip_match.group(0) if zip_match else ''
            
            # Extract detail URL
            detail_url = prop.get('detailUrl', '')
            if detail_url and not detail_url.startswith('http'):
                detail_url = f"https://www.zillow.com{detail_url}"
            
            listing = {
                'zpid': str(prop.get('zpid', '')) if prop.get('zpid') else None,
                'address': address_str,
                'zip_code': zip_code or prop.get('zipcode', ''),
                'price': prop.get('price'),
                'beds': prop.get('bedrooms'),
                'baths': prop.get('bathrooms'),
                'sqft': prop.get('livingArea'),
                'list_date': prop.get('listDate'),
                'status': prop.get('listingStatus', prop.get('homeStatus')),
                'days_on_zillow': prop.get('daysOnZillow'),
                'views': None,  # May need additional API call
                'saves': None,
                'features': [],  # May need additional API call for details
                'detail_url': detail_url,
                'scraped_at': datetime.now().isoformat(),
            }
            
            return listing
        except Exception as e:
            logger.warning(f"Error parsing Zillow listing: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _convert_attom_to_listing(self, prop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Attom property format to listing format."""
        try:
            address = prop.get('address', {})
            property_details = prop.get('property', {})
            sale = prop.get('sale', {})
            
            listing = {
                'zpid': None,  # Attom doesn't have ZPID
                'address': f"{address.get('oneLine', '')}".strip(),
                'zip_code': address.get('postal1'),
                'price': sale.get('amount', {}).get('saleamt'),
                'beds': property_details.get('structure', {}).get('bedrooms', {}).get('beds1'),
                'baths': property_details.get('structure', {}).get('bathrooms', {}).get('bathstotalcalc'),
                'sqft': property_details.get('structure', {}).get('size', {}).get('livingsize'),
                'list_date': None,  # Attom may not have list date
                'status': 'sold',  # Attom is sales data
                'days_on_zillow': None,
                'views': None,
                'saves': None,
                'features': [],  # Would need to parse from property details
                'detail_url': None,
                'scraped_at': datetime.now().isoformat(),
            }
            
            return listing
        except Exception as e:
            logger.warning(f"Error converting Attom property: {e}")
            return None
    
    def _extract_realtor_features(self, property_data: Dict[str, Any]) -> List[str]:
        """Extract features from Realtor.com property data."""
        features = []
        
        # Check various property fields for features
        description = property_data.get('description', '').lower()
        
        feature_keywords = [
            'granite', 'quartz', 'stainless steel', 'hardwood', 'carpet', 'tile',
            'fireplace', 'garage', 'fenced yard', 'deck', 'patio', 'pool',
            'updated kitchen', 'renovated', 'new roof', 'central air', 'heat pump'
        ]
        
        for keyword in feature_keywords:
            if keyword in description:
                features.append(keyword)
        
        return features


# Singleton instance
safe_listings_scraper = SafeListingsScraper()

