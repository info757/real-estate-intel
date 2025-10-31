"""
Land listing scraper for Zillow, Realtor.com, LandWatch, and other sources.
Includes deduplication and price tracking.
"""

import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import logging
from backend.models.schemas import LandListing, ListingStatus, ZoningType
from backend.utils.http_client import http_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ListingDeduplicator:
    """Handle deduplication of listings across sources."""
    
    def __init__(self, storage_path: str = "data/processed/listings.json"):
        self.storage_path = storage_path
        self.listings_cache: Dict[str, LandListing] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load existing listings from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    listing = LandListing(**item)
                    self.listings_cache[listing.listing_id] = listing
            logger.info(f"Loaded {len(self.listings_cache)} cached listings")
        except FileNotFoundError:
            logger.info("No existing cache found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    def _save_cache(self):
        """Save listings to storage."""
        try:
            import os
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            data = [listing.dict() for listing in self.listings_cache.values()]
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved {len(self.listings_cache)} listings to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def generate_listing_id(self, address: str, city: str, price: float, source: str) -> str:
        """Generate unique ID for a listing."""
        key = f"{source}_{address}_{city}_{price}".lower().replace(" ", "")
        return hashlib.md5(key.encode()).hexdigest()
    
    def is_duplicate(self, listing_id: str) -> bool:
        """Check if listing already exists."""
        return listing_id in self.listings_cache
    
    def add_or_update(self, listing: LandListing) -> bool:
        """Add new listing or update existing. Returns True if it's a new listing."""
        is_new = listing.listing_id not in self.listings_cache
        
        if not is_new:
            # Update existing listing
            existing = self.listings_cache[listing.listing_id]
            
            # Track price changes
            if existing.price != listing.price:
                price_change = {
                    "date": datetime.now().isoformat(),
                    "old_price": existing.price,
                    "new_price": listing.price
                }
                listing.price_history = existing.price_history + [price_change]
            
            # Update days on market
            if existing.listing_date:
                listing.days_on_market = (datetime.now() - existing.listing_date).days
        
        self.listings_cache[listing.listing_id] = listing
        return is_new
    
    def get_all_listings(self) -> List[LandListing]:
        """Get all cached listings."""
        return list(self.listings_cache.values())
    
    def save(self):
        """Save current state."""
        self._save_cache()


class ZillowScraper:
    """Scraper for Zillow land listings."""
    
    def __init__(self):
        self.base_url = "https://www.zillow.com"
    
    def search_land(self, city: str, state: str = "NC", max_price: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for land listings on Zillow."""
        logger.info(f"Searching Zillow for land in {city}, {state}")
        
        # In a real implementation, this would scrape Zillow
        # For now, return mock data
        return self._get_mock_zillow_data(city, state)
    
    def _get_mock_zillow_data(self, city: str, state: str) -> List[Dict[str, Any]]:
        """Mock Zillow data for development."""
        return [
            {
                "address": f"123 Land Rd, {city}",
                "city": city,
                "state": state,
                "price": 75000,
                "acreage": 1.5,
                "zoning": "residential",
                "url": f"https://www.zillow.com/homedetails/{city}-land-1"
            },
            {
                "address": f"456 Plot Dr, {city}",
                "city": city,
                "state": state,
                "price": 95000,
                "acreage": 2.0,
                "zoning": "residential",
                "url": f"https://www.zillow.com/homedetails/{city}-land-2"
            }
        ]


class RealtorScraper:
    """Scraper for Realtor.com land listings."""
    
    def __init__(self):
        self.base_url = "https://www.realtor.com"
    
    def search_land(self, city: str, state: str = "NC", max_price: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for land listings on Realtor.com."""
        logger.info(f"Searching Realtor.com for land in {city}, {state}")
        
        # In a real implementation, this would scrape Realtor.com
        return self._get_mock_realtor_data(city, state)
    
    def _get_mock_realtor_data(self, city: str, state: str) -> List[Dict[str, Any]]:
        """Mock Realtor.com data for development."""
        return [
            {
                "address": f"789 Parcel Ln, {city}",
                "city": city,
                "state": state,
                "price": 82000,
                "acreage": 1.8,
                "zoning": "residential",
                "url": f"https://www.realtor.com/realestateandhomes-detail/{city}-land-1"
            }
        ]


class LandWatchScraper:
    """Scraper for LandWatch.com listings."""
    
    def __init__(self):
        self.base_url = "https://www.landwatch.com"
    
    def search_land(self, county: str, state: str = "NC", max_price: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for land listings on LandWatch."""
        logger.info(f"Searching LandWatch for land in {county} County, {state}")
        
        # In a real implementation, this would scrape LandWatch
        return self._get_mock_landwatch_data(county, state)
    
    def _get_mock_landwatch_data(self, county: str, state: str) -> List[Dict[str, Any]]:
        """Mock LandWatch data for development."""
        return [
            {
                "address": f"Rural Lot, {county} County",
                "city": county,
                "county": county,
                "state": state,
                "price": 68000,
                "acreage": 3.5,
                "zoning": "agricultural",
                "url": f"https://www.landwatch.com/{state}/{county}-county/land-1"
            }
        ]


class LandScraperOrchestrator:
    """Orchestrates land scraping from multiple sources with deduplication."""
    
    def __init__(self):
        self.zillow = ZillowScraper()
        self.realtor = RealtorScraper()
        self.landwatch = LandWatchScraper()
        self.deduplicator = ListingDeduplicator()
    
    def _parse_zoning(self, zoning_str: str) -> ZoningType:
        """Parse zoning string to enum."""
        zoning_lower = zoning_str.lower() if zoning_str else ""
        
        if "residential" in zoning_lower:
            return ZoningType.RESIDENTIAL
        elif "commercial" in zoning_lower:
            return ZoningType.COMMERCIAL
        elif "agricultural" in zoning_lower or "farm" in zoning_lower:
            return ZoningType.AGRICULTURAL
        elif "industrial" in zoning_lower:
            return ZoningType.INDUSTRIAL
        elif "mixed" in zoning_lower:
            return ZoningType.MIXED_USE
        else:
            return ZoningType.UNKNOWN
    
    def _raw_to_listing(self, raw_data: Dict[str, Any], source: str) -> LandListing:
        """Convert raw scraper data to LandListing model."""
        listing_id = self.deduplicator.generate_listing_id(
            address=raw_data.get("address", ""),
            city=raw_data.get("city", ""),
            price=raw_data.get("price", 0),
            source=source
        )
        
        return LandListing(
            listing_id=listing_id,
            source=source,
            url=raw_data.get("url", ""),
            address=raw_data.get("address"),
            city=raw_data.get("city", ""),
            county=raw_data.get("county", ""),
            state=raw_data.get("state", "NC"),
            zip_code=raw_data.get("zip_code"),
            latitude=raw_data.get("latitude"),
            longitude=raw_data.get("longitude"),
            price=raw_data.get("price", 0),
            acreage=raw_data.get("acreage"),
            zoning=self._parse_zoning(raw_data.get("zoning", "")),
            utilities_available=raw_data.get("utilities", []),
            status=ListingStatus.ACTIVE,
            listing_date=raw_data.get("listing_date"),
            days_on_market=raw_data.get("days_on_market"),
            seller_type=raw_data.get("seller_type"),
            description=raw_data.get("description")
        )
    
    def scrape_city(self, city: str, state: str = "NC", max_price: Optional[float] = None) -> List[LandListing]:
        """Scrape all sources for a given city."""
        logger.info(f"Starting land scrape for {city}, {state}")
        
        all_listings = []
        new_count = 0
        updated_count = 0
        
        # Scrape Zillow
        try:
            zillow_data = self.zillow.search_land(city, state, max_price)
            for raw in zillow_data:
                listing = self._raw_to_listing(raw, "zillow")
                is_new = self.deduplicator.add_or_update(listing)
                all_listings.append(listing)
                if is_new:
                    new_count += 1
                else:
                    updated_count += 1
        except Exception as e:
            logger.error(f"Error scraping Zillow: {e}")
        
        # Scrape Realtor.com
        try:
            realtor_data = self.realtor.search_land(city, state, max_price)
            for raw in realtor_data:
                listing = self._raw_to_listing(raw, "realtor")
                is_new = self.deduplicator.add_or_update(listing)
                all_listings.append(listing)
                if is_new:
                    new_count += 1
                else:
                    updated_count += 1
        except Exception as e:
            logger.error(f"Error scraping Realtor.com: {e}")
        
        logger.info(f"Scrape complete: {new_count} new, {updated_count} updated")
        self.deduplicator.save()
        
        return all_listings
    
    def scrape_county(self, county: str, state: str = "NC", max_price: Optional[float] = None) -> List[LandListing]:
        """Scrape all sources for a given county."""
        logger.info(f"Starting land scrape for {county} County, {state}")
        
        all_listings = []
        new_count = 0
        
        # Scrape LandWatch (county-based)
        try:
            landwatch_data = self.landwatch.search_land(county, state, max_price)
            for raw in landwatch_data:
                listing = self._raw_to_listing(raw, "landwatch")
                is_new = self.deduplicator.add_or_update(listing)
                all_listings.append(listing)
                if is_new:
                    new_count += 1
        except Exception as e:
            logger.error(f"Error scraping LandWatch: {e}")
        
        logger.info(f"County scrape complete: {new_count} new listings")
        self.deduplicator.save()
        
        return all_listings
    
    def scrape_multiple_locations(self, locations: List[Dict[str, str]]) -> List[LandListing]:
        """Scrape multiple locations."""
        all_listings = []
        
        for loc in locations:
            if "city" in loc:
                listings = self.scrape_city(
                    city=loc["city"],
                    state=loc.get("state", "NC"),
                    max_price=loc.get("max_price")
                )
                all_listings.extend(listings)
            elif "county" in loc:
                listings = self.scrape_county(
                    county=loc["county"],
                    state=loc.get("state", "NC"),
                    max_price=loc.get("max_price")
                )
                all_listings.extend(listings)
        
        return all_listings
    
    def get_all_listings(self) -> List[LandListing]:
        """Get all cached listings."""
        return self.deduplicator.get_all_listings()
    
    def filter_listings(self,
                       max_price: Optional[float] = None,
                       min_acreage: Optional[float] = None,
                       max_acreage: Optional[float] = None,
                       zoning: Optional[ZoningType] = None,
                       city: Optional[str] = None,
                       county: Optional[str] = None) -> List[LandListing]:
        """Filter listings by criteria."""
        listings = self.get_all_listings()
        
        filtered = []
        for listing in listings:
            # Apply filters
            if max_price and listing.price > max_price:
                continue
            if min_acreage and (not listing.acreage or listing.acreage < min_acreage):
                continue
            if max_acreage and (not listing.acreage or listing.acreage > max_acreage):
                continue
            if zoning and listing.zoning != zoning:
                continue
            if city and listing.city.lower() != city.lower():
                continue
            if county and listing.county.lower() != county.lower():
                continue
            
            filtered.append(listing)
        
        return filtered

