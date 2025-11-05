"""
Sold Listings Analyzer
Analyzes recently sold listings to extract timeline data and calculate DOM metrics.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoldListingsAnalyzer:
    """Analyzes sold listings to extract timeline and DOM metrics."""
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def extract_timeline(
        self,
        listing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse priceHistory to find listing→pending→sold dates.
        
        Args:
            listing: Listing dictionary with priceHistory array
            
        Returns:
            Dictionary with listing_date, pending_date, sold_date, and DOM metrics
        """
        timeline = {
            'listing_date': None,
            'pending_date': None,
            'sold_date': None,
            'dom_to_pending': None,
            'dom_to_sold': None,
            'pending_to_sold': None,
        }
        
        # Get datePosted as listing date
        date_posted = listing.get('datePosted')
        if date_posted:
            timeline['listing_date'] = self._parse_date(date_posted)
        
        # Get dateSold as sold date
        date_sold = listing.get('dateSold')
        if date_sold:
            timeline['sold_date'] = self._parse_date(date_sold)
        
        # Parse priceHistory to find pending date
        price_history = listing.get('priceHistory', [])
        if isinstance(price_history, list):
            for event in price_history:
                if isinstance(event, dict):
                    event_type = event.get('event', '').lower()
                    event_date_str = event.get('date')
                    
                    if event_type == 'pending sale' or 'pending' in event_type:
                        timeline['pending_date'] = self._parse_date(event_date_str)
                        break
        
        # Calculate DOM metrics
        if timeline['listing_date']:
            if timeline['pending_date']:
                timeline['dom_to_pending'] = (timeline['pending_date'] - timeline['listing_date']).days
            if timeline['sold_date']:
                timeline['dom_to_sold'] = (timeline['sold_date'] - timeline['listing_date']).days
        
        if timeline['pending_date'] and timeline['sold_date']:
            timeline['pending_to_sold'] = (timeline['sold_date'] - timeline['pending_date']).days
        
        return timeline
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        # Try common date formats
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%m/%d/%Y',
            '%d/%m/%Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str[:10], fmt[:10])  # Take first 10 chars for date
            except (ValueError, IndexError):
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def calculate_dom_metrics(
        self,
        listings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate DOM metrics for a list of listings.
        
        Args:
            listings: List of listing dictionaries
            
        Returns:
            List of listings with added DOM metrics
        """
        enriched_listings = []
        
        for listing in listings:
            timeline = self.extract_timeline(listing)
            
            # Add timeline data to listing
            enriched_listing = {**listing, **timeline}
            enriched_listings.append(enriched_listing)
        
        return enriched_listings
    
    def filter_fast_sellers(
        self,
        listings: List[Dict[str, Any]],
        dom_threshold: int = 14
    ) -> List[Dict[str, Any]]:
        """
        Identify properties with DOM to pending < threshold.
        
        Args:
            listings: List of listings with DOM metrics
            dom_threshold: Maximum DOM to pending for "fast seller"
            
        Returns:
            List of fast-selling properties
        """
        fast_sellers = []
        
        for listing in listings:
            dom_to_pending = listing.get('dom_to_pending')
            
            if dom_to_pending is not None and dom_to_pending <= dom_threshold:
                fast_sellers.append(listing)
        
        return fast_sellers
    
    def analyze_fast_seller_characteristics(
        self,
        listings: List[Dict[str, Any]],
        dom_threshold: int = 14
    ) -> Dict[str, Any]:
        """
        Statistical analysis of fast sellers vs slow sellers.
        
        Args:
            listings: List of listings with DOM metrics
            dom_threshold: DOM threshold for fast vs slow
            
        Returns:
            Dictionary with comparison statistics
        """
        fast_sellers = self.filter_fast_sellers(listings, dom_threshold)
        slow_sellers = [
            l for l in listings
            if l.get('dom_to_pending') is not None and l.get('dom_to_pending', 999) > dom_threshold
        ]
        
        analysis = {
            'total_listings': len(listings),
            'fast_sellers': len(fast_sellers),
            'slow_sellers': len(slow_sellers),
            'fast_seller_pct': (len(fast_sellers) / len(listings) * 100) if listings else 0,
        }
        
        # Calculate average metrics
        if fast_sellers:
            analysis['fast_seller_avg_price'] = self._avg([l.get('price') for l in fast_sellers if l.get('price')])
            analysis['fast_seller_avg_beds'] = self._avg([l.get('beds') for l in fast_sellers if l.get('beds')])
            analysis['fast_seller_avg_baths'] = self._avg([l.get('baths') for l in fast_sellers if l.get('baths')])
            analysis['fast_seller_avg_sqft'] = self._avg([l.get('sqft') for l in fast_sellers if l.get('sqft')])
            analysis['fast_seller_avg_dom_to_pending'] = self._avg([l.get('dom_to_pending') for l in fast_sellers if l.get('dom_to_pending') is not None])
        
        if slow_sellers:
            analysis['slow_seller_avg_price'] = self._avg([l.get('price') for l in slow_sellers if l.get('price')])
            analysis['slow_seller_avg_beds'] = self._avg([l.get('beds') for l in slow_sellers if l.get('beds')])
            analysis['slow_seller_avg_baths'] = self._avg([l.get('baths') for l in slow_sellers if l.get('baths')])
            analysis['slow_seller_avg_sqft'] = self._avg([l.get('sqft') for l in slow_sellers if l.get('sqft')])
            analysis['slow_seller_avg_dom_to_pending'] = self._avg([l.get('dom_to_pending') for l in slow_sellers if l.get('dom_to_pending') is not None])
        
        return analysis
    
    def _avg(self, values: List[float]) -> float:
        """Calculate average of numeric values."""
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return 0.0
        return sum(valid_values) / len(valid_values)


# Singleton instance
sold_listings_analyzer = SoldListingsAnalyzer()
