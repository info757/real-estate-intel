"""
Current Listings Data Pipeline
Collects real-time listings data to enhance ML predictions with competitive context and demand signals.
"""

import requests
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ListingsScraper:
    """
    Scrapes and analyzes current real estate listings from Zillow/Realtor APIs.
    Provides competitive context and demand signals for ML models.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_source: str = 'zillow'):
        """
        Initialize listings scraper.
        
        Args:
            api_key: RapidAPI key for Zillow/Realtor API (optional, will use env var if not provided)
            api_source: 'zillow' or 'realtor' (default: zillow)
        """
        self.api_key = api_key or self._get_api_key_from_env()
        self.api_source = api_source
        self.base_url = self._get_api_url()
        self.cache = {}  # Simple in-memory cache
        
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
        import os
        return os.getenv('RAPIDAPI_KEY', None)
    
    def _get_api_url(self) -> str:
        """Get API base URL based on source."""
        if self.api_source == 'zillow':
            return "https://zillow-com1.p.rapidapi.com"
        elif self.api_source == 'realtor':
            return "https://realtor.p.rapidapi.com"
        else:
            raise ValueError(f"Unknown API source: {self.api_source}")
    
    def scrape_zillow_listings(
        self,
        zip_code: str,
        radius_miles: float = 5.0,
        status: str = 'active',
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch active listings from Zillow API.
        
        Args:
            zip_code: ZIP code to search
            radius_miles: Search radius in miles
            status: 'active', 'pending', or 'sold'
            max_results: Maximum number of results to return
            
        Returns:
            List of listing dictionaries
        """
        # For MVP: Return mock data structure
        # In production: Replace with actual API call
        
        logger.info(f"Fetching listings for ZIP {zip_code} (radius: {radius_miles} miles)")
        
        # TODO: Implement actual API call when RapidAPI key is available
        # For now, return structure that matches what we'll get from API
        
        mock_listings = self._generate_mock_listings(zip_code, max_results)
        
        logger.info(f"Found {len(mock_listings)} listings")
        return mock_listings
    
    def _generate_mock_listings(self, zip_code: str, count: int = 20) -> List[Dict[str, Any]]:
        """
        Generate mock listings for testing (replace with real API call).
        """
        import random
        
        subdivisions = ['Hamilton Lakes', 'Brooks', 'Sterling Park', 'Oak Ridge', 'Willow Creek']
        feature_options = [
            ['granite', 'hardwood', 'stainless'],
            ['granite', 'carpet', 'gas range'],
            ['quartz', 'LVP', 'stainless', 'smart home'],
            ['granite', 'hardwood', 'stainless', 'fireplace'],
            ['laminate', 'carpet'],
        ]
        
        listings = []
        base_date = datetime.now()
        
        for i in range(count):
            beds = random.choice([2, 3, 3, 4, 4, 5])
            baths = beds - random.choice([0, 0.5, 1])
            sqft = beds * random.randint(450, 650)
            
            listing = {
                'address': f'{100 + i} Example St',
                'zip': zip_code,
                'subdivision': random.choice(subdivisions),
                'lat': 36.089 + random.uniform(-0.1, 0.1),
                'lon': -79.908 + random.uniform(-0.1, 0.1),
                'list_price': sqft * random.randint(180, 250),
                'beds': beds,
                'baths': baths,
                'sqft': sqft,
                'lot_size_acres': round(random.uniform(0.15, 0.5), 2),
                'dom': random.randint(1, 90),
                'status': 'active',
                'property_type': random.choice(['SFR', 'SFR', 'SFR', 'Townhome', 'Condo']),
                'views': random.randint(50, 2000),
                'saves': random.randint(0, 50),
                'hot_home': random.random() < 0.2,  # 20% are "hot"
                'price_reduced': random.random() < 0.15,  # 15% reduced
                'features': random.choice(feature_options),
                'photo_count': random.randint(10, 40),
                'list_date': (base_date - timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d'),
                'scraped_date': datetime.now().strftime('%Y-%m-%d'),
            }
            
            listings.append(listing)
        
        return listings
    
    def extract_features_from_text(self, description_text: str) -> List[str]:
        """
        Extract features from listing description using NLP/text mining.
        
        Args:
            description_text: Listing description text
            
        Returns:
            List of identified features
        """
        if not description_text:
            return []
        
        # Define feature keywords to search for
        feature_keywords = {
            'granite': ['granite', 'granite counter'],
            'quartz': ['quartz', 'quartz counter'],
            'hardwood': ['hardwood', 'hardwood floor', 'wood floor'],
            'lvp': ['lvp', 'luxury vinyl', 'vinyl plank'],
            'carpet': ['carpet', 'carpeted'],
            'stainless': ['stainless', 'stainless steel', 'stainless appliance'],
            'smart_home': ['smart home', 'nest', 'alexa', 'smart thermostat'],
            'fireplace': ['fireplace', 'gas fireplace'],
            'deck': ['deck', 'wooden deck'],
            'patio': ['patio', 'covered patio'],
            'pool': ['pool', 'swimming pool'],
            'garage': ['garage', 'car garage', '2 car', '3 car'],
        }
        
        description_lower = description_text.lower()
        found_features = []
        
        for feature, keywords in feature_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    found_features.append(feature)
                    break  # Found this feature, move to next
        
        return found_features
    
    def calculate_competitive_context(
        self,
        zip_code: str,
        proposed_config: Dict[str, Any],
        radius_miles: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate competitive context metrics for a proposed house configuration.
        
        Args:
            zip_code: ZIP code of the lot
            proposed_config: Dict with 'beds', 'baths', 'sqft', 'list_price'
            radius_miles: Search radius in miles
            
        Returns:
            Competitive context metrics dict
        """
        # Get current listings
        listings = self.scrape_zillow_listings(zip_code, radius_miles, status='active')
        
        if not listings:
            return {
                'error': 'No listings found',
                'active_listings_similar': 0,
                'total_active_listings': 0,
            }
        
        # Filter to similar listings (±20% price, same beds/baths)
        price_min = proposed_config.get('list_price', 0) * 0.8
        price_max = proposed_config.get('list_price', 999999999) * 1.2
        beds = proposed_config.get('beds', 3)
        baths = proposed_config.get('baths', 2)
        
        similar_listings = [
            l for l in listings
            if (price_min <= l['list_price'] <= price_max and
                l['beds'] == beds and
                abs(l['baths'] - baths) <= 0.5)
        ]
        
        # Calculate metrics
        total_active = len(listings)
        similar_count = len(similar_listings)
        
        # Inventory level classification
        if total_active < 20:
            inventory_level = 'low'
        elif total_active < 50:
            inventory_level = 'medium'
        else:
            inventory_level = 'high'
        
        # Price positioning (percentile)
        all_prices = sorted([l['list_price'] for l in listings])
        if all_prices:
            proposed_price = proposed_config.get('list_price', all_prices[len(all_prices) // 2])
            rank = sum(1 for p in all_prices if p < proposed_price)
            price_percentile = rank / len(all_prices) if all_prices else 0.5
        else:
            price_percentile = 0.5
        
        # Average DOM
        avg_dom = sum(l['dom'] for l in listings) / len(listings) if listings else 0
        
        # Price reduction rate
        reduced_count = sum(1 for l in listings if l.get('price_reduced', False))
        price_reduction_rate = reduced_count / len(listings) if listings else 0
        
        # Hot homes percentage
        hot_count = sum(1 for l in listings if l.get('hot_home', False))
        hot_home_percentage = hot_count / len(listings) if listings else 0
        
        return {
            'active_listings_similar': similar_count,
            'total_active_listings': total_active,
            'inventory_level': inventory_level,
            'price_percentile': round(price_percentile, 2),
            'avg_dom_active': round(avg_dom, 1),
            'price_reduction_rate': round(price_reduction_rate, 2),
            'hot_home_percentage': round(hot_home_percentage, 2),
            'recommendation': self._generate_competitive_recommendation(
                inventory_level, similar_count, price_percentile
            )
        }
    
    def _generate_competitive_recommendation(
        self,
        inventory_level: str,
        similar_count: int,
        price_percentile: float
    ) -> str:
        """Generate human-readable competitive recommendation."""
        if inventory_level == 'low' and similar_count < 5:
            return "Excellent timing - low competition, strong seller's market"
        elif inventory_level == 'high' and similar_count > 10:
            return "High competition - consider differentiating features or pricing strategy"
        elif price_percentile > 0.8:
            return "Premium pricing - ensure features justify top-tier position"
        elif price_percentile < 0.3:
            return "Value pricing - may sell quickly but check margin targets"
        else:
            return "Moderate competition - solid market conditions"
    
    def track_listing_outcomes(
        self,
        listings: List[Dict[str, Any]],
        closed_sales: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Match listings to closed sales and calculate list-to-sale metrics.
        
        Args:
            listings: Historical listings data
            closed_sales: Closed sales data (from Attom)
            
        Returns:
            List of outcomes with list-to-sale ratios
        """
        outcomes = []
        
        # Match by address
        sales_by_address = {s['address']: s for s in closed_sales if 'address' in s}
        
        for listing in listings:
            address = listing.get('address', '')
            if address in sales_by_address:
                sale = sales_by_address[address]
                
                list_price = listing.get('list_price', 0)
                sale_price = sale.get('sale_price', 0)
                
                if list_price > 0 and sale_price > 0:
                    outcome = {
                        'address': address,
                        'list_price': list_price,
                        'sale_price': sale_price,
                        'list_to_sale_ratio': sale_price / list_price,
                        'list_date': listing.get('list_date'),
                        'sale_date': sale.get('sale_date'),
                        'dom_to_sold': (
                            (datetime.strptime(sale.get('sale_date', ''), '%Y-%m-%d') -
                             datetime.strptime(listing.get('list_date', ''), '%Y-%m-%d')).days
                            if sale.get('sale_date') and listing.get('list_date') else None
                        ),
                    }
                    outcomes.append(outcome)
        
        logger.info(f"Matched {len(outcomes)} listings to sales")
        return outcomes
    
    def analyze_feature_trends(
        self,
        listings: List[Dict[str, Any]],
        days_back: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze feature popularity trends from recent listings.
        
        Args:
            listings: List of listings with 'features' field
            days_back: Number of days to look back for trend comparison
            
        Returns:
            Feature trends dict with counts and change percentages
        """
        # Split listings into current and previous period
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        current_listings = [
            l for l in listings
            if datetime.strptime(l.get('list_date', '1900-01-01'), '%Y-%m-%d') >= cutoff_date
        ]
        
        previous_cutoff = cutoff_date - timedelta(days=days_back)
        previous_listings = [
            l for l in listings
            if previous_cutoff <= datetime.strptime(l.get('list_date', '1900-01-01'), '%Y-%m-%d') < cutoff_date
        ]
        
        # Count features in each period
        current_features = Counter()
        previous_features = Counter()
        
        for listing in current_listings:
            for feature in listing.get('features', []):
                current_features[feature] += 1
        
        for listing in previous_listings:
            for feature in listing.get('features', []):
                previous_features[feature] += 1
        
        # Calculate trends
        trends = {}
        all_features = set(current_features.keys()) | set(previous_features.keys())
        
        for feature in all_features:
            current_count = current_features.get(feature, 0)
            previous_count = previous_features.get(feature, 0)
            
            if previous_count > 0:
                change_pct = ((current_count - previous_count) / previous_count) * 100
            else:
                change_pct = 100 if current_count > 0 else 0
            
            # Classify trend
            if change_pct > 15:
                trend_status = 'hot'
            elif change_pct > 5:
                trend_status = 'up'
            elif change_pct < -15:
                trend_status = 'down'
            elif change_pct < -5:
                trend_status = 'cooling'
            else:
                trend_status = 'stable'
            
            trends[feature] = {
                'count': current_count,
                'previous_count': previous_count,
                'change_pct': round(change_pct, 1),
                'trending': trend_status
            }
        
        # Sort by current count (most popular first)
        trends = dict(sorted(trends.items(), key=lambda x: x[1]['count'], reverse=True))
        
        logger.info(f"Analyzed trends for {len(trends)} features")
        return trends


# Singleton instance
listings_scraper = ListingsScraper()

