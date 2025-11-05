"""
Feature Impact Analyzer
Analyzes the impact of property features on DOM to pending.
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureImpactAnalyzer:
    """Analyzes feature impact on selling speed."""
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def analyze_feature_impact_on_dom(
        self,
        listings: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Correlate features with DOM to pending.
        
        Args:
            listings: List of listings with extracted_features and dom_to_pending
            
        Returns:
            Dictionary mapping feature to impact statistics
        """
        feature_doms = defaultdict(list)
        feature_counts = defaultdict(int)
        
        # Collect DOM values for each feature
        for listing in listings:
            dom = listing.get('dom_to_pending')
            if dom is None:
                continue
            
            features = listing.get('extracted_features', {})
            
            # Check all feature categories
            for feature in features.get('interior', []):
                feature_doms[feature].append(dom)
                feature_counts[feature] += 1
            
            for feature in features.get('exterior', []):
                feature_doms[feature].append(dom)
                feature_counts[feature] += 1
            
            for feature in features.get('upgrades', []):
                feature_doms[feature].append(dom)
                feature_counts[feature] += 1
        
        # Calculate statistics for each feature
        feature_stats = {}
        
        # Calculate overall median DOM for comparison
        all_doms = [l.get('dom_to_pending') for l in listings if l.get('dom_to_pending') is not None]
        overall_median_dom = statistics.median(all_doms) if all_doms else 30
        
        for feature, doms in feature_doms.items():
            if len(doms) >= 3:  # Minimum samples
                median_dom = statistics.median(doms)
                avg_dom = statistics.mean(doms)
                dom_reduction = overall_median_dom - median_dom
                dom_reduction_pct = (dom_reduction / overall_median_dom * 100) if overall_median_dom > 0 else 0
                
                feature_stats[feature] = {
                    'count': len(doms),
                    'median_dom': median_dom,
                    'avg_dom': avg_dom,
                    'dom_reduction': dom_reduction,
                    'dom_reduction_pct': dom_reduction_pct,
                    'overall_median': overall_median_dom
                }
        
        return feature_stats
    
    def rank_features_by_impact(
        self,
        feature_stats: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank features by their impact on selling speed.
        
        Args:
            feature_stats: Feature statistics from analyze_feature_impact_on_dom
            
        Returns:
            Sorted list of features by DOM reduction (best first)
        """
        ranked = []
        
        for feature, stats in feature_stats.items():
            ranked.append({
                'feature': feature,
                **stats
            })
        
        # Sort by DOM reduction (descending - most reduction first)
        ranked.sort(key=lambda x: x['dom_reduction'], reverse=True)
        
        return ranked
    
    def generate_feature_recommendations(
        self,
        listings: List[Dict[str, Any]],
        top_n: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Suggest high-impact features for new builds.
        
        Args:
            listings: List of listings with features and DOM
            top_n: Number of top features to return
            
        Returns:
            List of recommended features with impact data
        """
        feature_stats = self.analyze_feature_impact_on_dom(listings)
        ranked = self.rank_features_by_impact(feature_stats)
        
        # Filter to features with meaningful impact
        meaningful = [
            f for f in ranked
            if f['dom_reduction'] > 0 and f['count'] >= 5  # At least 5 samples and positive impact
        ]
        
        return meaningful[:top_n]
    
    def compare_fast_vs_slow_sellers(
        self,
        listings: List[Dict[str, Any]],
        fast_threshold: int = 14
    ) -> Dict[str, Any]:
        """
        Compare feature prevalence in fast vs slow sellers.
        
        Args:
            listings: List of listings with features and DOM
            fast_threshold: DOM threshold for fast sellers
            
        Returns:
            Dictionary with feature prevalence comparison
        """
        fast_sellers = [l for l in listings if l.get('dom_to_pending', 999) <= fast_threshold]
        slow_sellers = [l for l in listings if l.get('dom_to_pending', 999) > fast_threshold]
        
        # Count feature appearances
        fast_features = defaultdict(int)
        slow_features = defaultdict(int)
        
        for listing in fast_sellers:
            features = listing.get('extracted_features', {})
            for feature in features.get('interior', []) + features.get('exterior', []) + features.get('upgrades', []):
                fast_features[feature] += 1
        
        for listing in slow_sellers:
            features = listing.get('extracted_features', {})
            for feature in features.get('interior', []) + features.get('exterior', []) + features.get('upgrades', []):
                slow_features[feature] += 1
        
        # Calculate prevalence rates
        total_fast = len(fast_sellers)
        total_slow = len(slow_sellers)
        
        comparison = {}
        
        all_features = set(list(fast_features.keys()) + list(slow_features.keys()))
        
        for feature in all_features:
            fast_count = fast_features[feature]
            slow_count = slow_features[feature]
            
            fast_rate = (fast_count / total_fast * 100) if total_fast > 0 else 0
            slow_rate = (slow_count / total_slow * 100) if total_slow > 0 else 0
            
            if slow_rate > 0:
                lift = fast_rate / slow_rate
            else:
                lift = float('inf') if fast_rate > 0 else 1.0
            
            comparison[feature] = {
                'fast_count': fast_count,
                'slow_count': slow_count,
                'fast_rate': fast_rate,
                'slow_rate': slow_rate,
                'lift': lift  # How much more common in fast sellers
            }
        
        # Sort by lift (highest first)
        comparison_list = [
            {'feature': k, **v}
            for k, v in comparison.items()
        ]
        comparison_list.sort(key=lambda x: x['lift'], reverse=True)
        
        return {
            'total_fast': total_fast,
            'total_slow': total_slow,
            'features': comparison_list
        }


# Singleton instance
feature_impact_analyzer = FeatureImpactAnalyzer()
