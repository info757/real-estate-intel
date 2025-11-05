"""
Popularity Analyzer
Analyzes Zillow listings to identify what features make listings popular and fast-selling.
"""

import logging
from typing import Dict, Any, List, Optional
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PopularityAnalyzer:
    """
    Analyzes listing popularity and identifies features that drive demand.
    """
    
    def __init__(self):
        """Initialize popularity analyzer."""
        pass
    
    def analyze_popular_listings(
        self,
        listings: List[Dict[str, Any]],
        top_n: int = 20,
        popularity_metric: str = 'composite'  # 'views', 'saves', 'composite', 'fast_dom'
    ) -> Dict[str, Any]:
        """
        Analyze the most popular listings and identify common features.
        
        Args:
            listings: List of listing dictionaries
            top_n: Number of top listings to analyze
            popularity_metric: Metric to rank by ('views', 'saves', 'composite', 'fast_dom')
            
        Returns:
            Analysis results with top listings and feature insights
        """
        logger.info(f"Analyzing popularity for {len(listings)} listings")
        
        # Score listings by popularity
        scored_listings = self._score_listings(listings, popularity_metric)
        
        # Get top N
        top_listings = sorted(scored_listings, key=lambda x: x['popularity_score'], reverse=True)[:top_n]
        
        # Analyze features of popular listings
        feature_analysis = self._analyze_popular_features(top_listings, all_listings=listings)
        
        # Analyze price points
        price_analysis = self._analyze_price_points(top_listings)
        
        # Analyze configuration (beds/baths/sqft)
        config_analysis = self._analyze_configurations(top_listings)
        
        return {
            'top_listings': top_listings,
            'feature_analysis': feature_analysis,
            'price_analysis': price_analysis,
            'config_analysis': config_analysis,
            'total_analyzed': len(listings),
            'popularity_metric': popularity_metric,
        }
    
    def _score_listings(
        self,
        listings: List[Dict[str, Any]],
        metric: str = 'composite'
    ) -> List[Dict[str, Any]]:
        """Score listings by popularity metric."""
        scored = []
        
        for listing in listings:
            score = 0.0
            
            if metric == 'views':
                score = float(listing.get('views', 0) or 0)
            elif metric == 'saves':
                score = float(listing.get('saves', 0) or 0) * 10  # Weight saves more
            elif metric == 'fast_dom':
                # Lower DOM = higher score
                dom = listing.get('days_on_zillow') or listing.get('dom_to_pending') or 999
                score = max(0, 1000 - (dom * 10))  # 0 days = 1000, 100 days = 0
            else:  # composite
                # Weighted combination
                views = float(listing.get('views', 0) or 0)
                saves = float(listing.get('saves', 0) or 0)
                dom = listing.get('days_on_zillow') or listing.get('dom_to_pending') or 999
                
                # Normalize
                views_norm = min(views / 100.0, 10.0)
                saves_norm = min(saves * 2.0, 10.0)
                dom_norm = max(0, 10.0 - (dom / 10.0))
                
                score = (views_norm * 0.3) + (saves_norm * 0.4) + (dom_norm * 0.3)
            
            listing_copy = listing.copy()
            listing_copy['popularity_score'] = round(score, 2)
            scored.append(listing_copy)
        
        return scored
    
    def _analyze_popular_features(
        self,
        top_listings: List[Dict[str, Any]],
        all_listings: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Analyze which features appear most in popular listings."""
        # Count features in top listings
        top_features = Counter()
        top_listing_count = len(top_listings)
        
        for listing in top_listings:
            features = listing.get('features', [])
            for feature in features:
                top_features[feature] += 1
        
        # Compare to all listings if provided
        all_features = Counter()
        if all_listings:
            for listing in all_listings:
                features = listing.get('features', [])
                for feature in features:
                    all_features[feature] += 1
        
        # Calculate feature impact (appears in X% of top listings vs. all listings)
        feature_impact = {}
        for feature, count in top_features.items():
            top_pct = (count / top_listing_count * 100) if top_listing_count > 0 else 0
            
            all_count = all_features.get(feature, 0)
            all_pct = (all_count / len(all_listings) * 100) if all_listings and len(all_listings) > 0 else 0
            
            # Impact ratio (how much more common in top listings)
            impact_ratio = top_pct / all_pct if all_pct > 0 else 2.0
            
            feature_impact[feature] = {
                'count_in_top': count,
                'pct_in_top': round(top_pct, 1),
                'count_in_all': all_count,
                'pct_in_all': round(all_pct, 1),
                'impact_ratio': round(impact_ratio, 2),
                'is_driver': impact_ratio >= 1.5,  # 50% more common = driver
            }
        
        # Sort by impact ratio (most impactful first)
        feature_impact = dict(sorted(feature_impact.items(), key=lambda x: x[1]['impact_ratio'], reverse=True))
        
        return {
            'top_features': dict(top_features.most_common(20)),
            'feature_impact': feature_impact,
            'drivers': [f for f, data in feature_impact.items() if data['is_driver']],
        }
    
    def _analyze_price_points(
        self,
        listings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze price points of popular listings."""
        prices = [l['price'] for l in listings if l.get('price')]
        
        if not prices:
            return {}
        
        return {
            'median': int(np.median(prices)),
            'mean': int(np.mean(prices)),
            'min': min(prices),
            'max': max(prices),
            'q25': int(np.percentile(prices, 25)),
            'q75': int(np.percentile(prices, 75)),
        }
    
    def _analyze_configurations(
        self,
        listings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze bed/bath/sqft configurations of popular listings."""
        configs = []
        
        for listing in listings:
            configs.append({
                'beds': listing.get('beds'),
                'baths': listing.get('baths'),
                'sqft': listing.get('sqft'),
                'price_per_sqft': listing['price'] / listing.get('sqft', 1) if listing.get('price') and listing.get('sqft') else None,
            })
        
        # Most common configurations
        config_counts = Counter()
        for c in configs:
            if c['beds'] and c['baths']:
                key = f"{c['beds']}BR/{c['baths']}BA"
                config_counts[key] += 1
        
        # Average sizes
        sqft_values = [c['sqft'] for c in configs if c.get('sqft')]
        price_per_sqft_values = [c['price_per_sqft'] for c in configs if c.get('price_per_sqft')]
        
        return {
            'most_common_configs': dict(config_counts.most_common(10)),
            'avg_sqft': int(np.mean(sqft_values)) if sqft_values else None,
            'median_sqft': int(np.median(sqft_values)) if sqft_values else None,
            'avg_price_per_sqft': int(np.mean(price_per_sqft_values)) if price_per_sqft_values else None,
            'median_price_per_sqft': int(np.median(price_per_sqft_values)) if price_per_sqft_values else None,
        }
    
    def analyze_dom_to_pending(
        self,
        pending_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze DOM to pending to understand what makes listings sell fast.
        
        Args:
            pending_analyses: List of analyses from ZillowScraper.analyze_pending_listings
            
        Returns:
            Analysis of fast-selling listings
        """
        logger.info(f"Analyzing DOM to pending for {len(pending_analyses)} listings")
        
        # Filter to valid DOM values
        valid_analyses = [a for a in pending_analyses if a.get('dom_to_pending') is not None]
        
        if not valid_analyses:
            return {'error': 'No valid DOM data available'}
        
        # Sort by DOM (fastest first)
        fast_sellers = sorted(valid_analyses, key=lambda x: x['dom_to_pending'])[:20]
        
        # Analyze features of fast sellers
        fast_seller_features = self._analyze_popular_features(fast_sellers, all_listings=pending_analyses)
        
        # DOM statistics
        dom_values = [a['dom_to_pending'] for a in valid_analyses]
        
        # Price analysis for fast sellers
        fast_seller_prices = [a['price'] for a in fast_sellers if a.get('price')]
        
        return {
            'fastest_sellers': fast_sellers[:10],  # Top 10 fastest
            'dom_stats': {
                'median': int(np.median(dom_values)),
                'mean': round(np.mean(dom_values), 1),
                'min': min(dom_values),
                'max': max(dom_values),
                'q25': int(np.percentile(dom_values, 25)),
                'q75': int(np.percentile(dom_values, 75)),
            },
            'fast_seller_features': fast_seller_features,
            'fast_seller_price_analysis': {
                'median': int(np.median(fast_seller_prices)) if fast_seller_prices else None,
                'mean': int(np.mean(fast_seller_prices)) if fast_seller_prices else None,
            },
            'total_analyzed': len(valid_analyses),
        }


# Singleton instance
popularity_analyzer = PopularityAnalyzer()

