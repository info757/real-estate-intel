"""
Pattern Discovery
Discovers non-obvious patterns in fast-selling properties using association rules and statistical analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternDiscovery:
    """Discovers patterns in fast-selling properties."""
    
    def __init__(self):
        """Initialize pattern discovery."""
        pass
    
    def discover_feature_combinations(
        self,
        listings: List[Dict[str, Any]],
        fast_threshold: int = 14,
        min_lift: float = 1.5,
        min_confidence: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find feature combinations that correlate with fast sales using association rules.
        
        Args:
            listings: List of listings with extracted_features and dom_to_pending
            fast_threshold: DOM threshold for fast sellers
            min_lift: Minimum lift to consider pattern significant
            min_confidence: Minimum confidence for association rule
            
        Returns:
            List of discovered patterns with lift and confidence
        """
        fast_sellers = [l for l in listings if l.get('dom_to_pending', 999) <= fast_threshold]
        all_listings = listings
        
        # Extract feature sets for each listing
        fast_feature_sets = [self._get_feature_set(l) for l in fast_sellers]
        all_feature_sets = [self._get_feature_set(l) for l in all_listings]
        
        # Find frequent feature pairs and triplets
        patterns = []
        
        # Analyze pairs
        patterns.extend(self._analyze_feature_combinations(
            fast_feature_sets, all_feature_sets, size=2, min_lift=min_lift, min_confidence=min_confidence
        ))
        
        # Analyze triplets
        patterns.extend(self._analyze_feature_combinations(
            fast_feature_sets, all_feature_sets, size=3, min_lift=min_lift, min_confidence=min_confidence
        ))
        
        # Sort by lift (descending)
        patterns.sort(key=lambda x: x['lift'], reverse=True)
        
        return patterns
    
    def _get_feature_set(self, listing: Dict[str, Any]) -> set:
        """Extract all features as a set."""
        features = listing.get('extracted_features', {})
        feature_set = set()
        feature_set.update(features.get('interior', []))
        feature_set.update(features.get('exterior', []))
        feature_set.update(features.get('upgrades', []))
        return feature_set
    
    def _analyze_feature_combinations(
        self,
        fast_sets: List[set],
        all_sets: List[set],
        size: int,
        min_lift: float,
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """Analyze feature combinations of a given size."""
        from itertools import combinations
        
        # Count occurrences
        fast_combos = Counter()
        all_combos = Counter()
        
        for feature_set in fast_sets:
            for combo in combinations(sorted(feature_set), size):
                fast_combos[combo] += 1
        
        for feature_set in all_sets:
            for combo in combinations(sorted(feature_set), size):
                all_combos[combo] += 1
        
        patterns = []
        total_fast = len(fast_sets)
        total_all = len(all_sets)
        
        for combo, fast_count in fast_combos.items():
            all_count = all_combos.get(combo, 0)
            
            if all_count < 3:  # Need at least 3 occurrences
                continue
            
            # Calculate support and confidence
            support_fast = fast_count / total_fast if total_fast > 0 else 0
            support_all = all_count / total_all if total_all > 0 else 0
            
            if support_all == 0:
                continue
            
            confidence = support_fast / support_all if support_all > 0 else 0
            lift = confidence / (fast_count / total_fast) if total_fast > 0 else 0
            
            if lift >= min_lift and confidence >= min_confidence:
                patterns.append({
                    'features': list(combo),
                    'support_fast': support_fast,
                    'support_all': support_all,
                    'confidence': confidence,
                    'lift': lift,
                    'fast_count': fast_count,
                    'all_count': all_count
                })
        
        return patterns
    
    def discover_price_patterns(
        self,
        listings: List[Dict[str, Any]],
        fast_threshold: int = 14
    ) -> Dict[str, Any]:
        """
        Identify optimal pricing strategies.
        
        Args:
            listings: List of listings with price and dom_to_pending
            fast_threshold: DOM threshold for fast sellers
            
        Returns:
            Dictionary with pricing pattern insights
        """
        fast_sellers = [l for l in listings if l.get('dom_to_pending', 999) <= fast_threshold]
        slow_sellers = [l for l in listings if l.get('dom_to_pending', 999) > fast_threshold]
        
        # Analyze price endings
        fast_endings = Counter()
        slow_endings = Counter()
        
        for listing in fast_sellers:
            price = listing.get('price')
            if price:
                ending = price % 1000
                fast_endings[ending] += 1
        
        for listing in slow_sellers:
            price = listing.get('price')
            if price:
                ending = price % 1000
                slow_endings[ending] += 1
        
        # Calculate price positioning (vs neighborhood median)
        fast_prices = [l.get('price') for l in fast_sellers if l.get('price')]
        slow_prices = [l.get('price') for l in slow_sellers if l.get('price')]
        
        fast_median = statistics.median(fast_prices) if fast_prices else None
        slow_median = statistics.median(slow_prices) if slow_prices else None
        
        all_prices = fast_prices + slow_prices
        market_median = statistics.median(all_prices) if all_prices else None
        
        fast_relative = (fast_median / market_median - 1) * 100 if (fast_median and market_median) else None
        slow_relative = (slow_median / market_median - 1) * 100 if (slow_median and market_median) else None
        
        return {
            'fast_median_price': fast_median,
            'slow_median_price': slow_median,
            'market_median_price': market_median,
            'fast_relative_to_market_pct': fast_relative,
            'slow_relative_to_market_pct': slow_relative,
            'price_ending_patterns': {
                'fast': dict(fast_endings.most_common(5)),
                'slow': dict(slow_endings.most_common(5))
            }
        }
    
    def discover_timing_patterns(
        self,
        listings: List[Dict[str, Any]],
        fast_threshold: int = 14
    ) -> Dict[str, Any]:
        """
        Identify optimal listing timing patterns.
        
        Args:
            listings: List of listings with listing_date and dom_to_pending
            fast_threshold: DOM threshold for fast sellers
            
        Returns:
            Dictionary with timing pattern insights
        """
        from datetime import datetime
        
        fast_by_day = Counter()
        fast_by_month = Counter()
        slow_by_day = Counter()
        slow_by_month = Counter()
        
        for listing in listings:
            listing_date = listing.get('listing_date')
            dom = listing.get('dom_to_pending')
            
            if not listing_date or dom is None:
                continue
            
            if isinstance(listing_date, str):
                try:
                    listing_date = datetime.fromisoformat(listing_date.replace('Z', '+00:00'))
                except:
                    continue
            
            day_of_week = listing_date.strftime('%A')
            month = listing_date.strftime('%B')
            
            if dom <= fast_threshold:
                fast_by_day[day_of_week] += 1
                fast_by_month[month] += 1
            else:
                slow_by_day[day_of_week] += 1
                slow_by_month[month] += 1
        
        return {
            'day_of_week': {
                'fast': dict(fast_by_day.most_common()),
                'slow': dict(slow_by_day.most_common())
            },
            'month': {
                'fast': dict(fast_by_month.most_common()),
                'slow': dict(slow_by_month.most_common())
            }
        }
    
    def generate_insights_report(
        self,
        listings: List[Dict[str, Any]],
        fast_threshold: int = 14
    ) -> Dict[str, Any]:
        """
        Generate human-readable insights report.
        
        Args:
            listings: List of listings with features and DOM
            fast_threshold: DOM threshold for fast sellers
            
        Returns:
            Dictionary with formatted insights
        """
        feature_combos = self.discover_feature_combinations(listings, fast_threshold)
        price_patterns = self.discover_price_patterns(listings, fast_threshold)
        timing_patterns = self.discover_timing_patterns(listings, fast_threshold)
        
        # Generate human-readable insights
        insights = []
        
        # Feature combination insights
        for pattern in feature_combos[:10]:  # Top 10
            features_str = " + ".join(pattern['features'])
            insights.append({
                'type': 'feature_combination',
                'insight': f"Properties with {features_str} sell {pattern['lift']:.1f}x faster",
                'details': pattern
            })
        
        # Price insights
        if price_patterns.get('fast_relative_to_market_pct'):
            rel_pct = price_patterns['fast_relative_to_market_pct']
            direction = "below" if rel_pct < 0 else "above"
            insights.append({
                'type': 'pricing',
                'insight': f"Fast-selling properties are priced {abs(rel_pct):.1f}% {direction} market median",
                'details': price_patterns
            })
        
        return {
            'feature_combinations': feature_combos[:10],
            'price_patterns': price_patterns,
            'timing_patterns': timing_patterns,
            'human_readable_insights': insights
        }


# Singleton instance
pattern_discovery = PatternDiscovery()
