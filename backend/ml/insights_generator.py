"""
Insights Generator
Combines model predictions with pattern discovery to generate human-readable insights.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightsGenerator:
    """Generates human-readable insights from model predictions and patterns."""
    
    def __init__(self):
        """Initialize insights generator."""
        pass
    
    def generate_recommendation_insights(
        self,
        configuration: Dict[str, Any],
        fast_seller_prob: float,
        predicted_dom: float,
        margin_pct: float,
        feature_importance: Optional[Dict[str, float]] = None,
        discovered_patterns: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Combine model predictions with pattern discovery to generate insights.
        
        Args:
            configuration: Build configuration (beds, baths, sqft, etc.)
            fast_seller_prob: Probability of selling fast (0-1)
            predicted_dom: Predicted DOM to pending
            margin_pct: Gross margin percentage
            feature_importance: Feature importance scores from model
            discovered_patterns: Patterns from pattern discovery
            
        Returns:
            Dictionary with formatted insights
        """
        insights = {
            'summary': self._generate_summary(fast_seller_prob, predicted_dom, margin_pct),
            'market_alignment': self._assess_market_alignment(fast_seller_prob, predicted_dom),
            'key_drivers': self._identify_key_drivers(feature_importance),
            'pattern_insights': self._extract_pattern_insights(configuration, discovered_patterns),
            'recommendations': self._generate_actionable_recommendations(
                configuration, fast_seller_prob, predicted_dom, feature_importance
            )
        }
        
        return insights
    
    def _generate_summary(
        self,
        fast_seller_prob: float,
        predicted_dom: float,
        margin_pct: float
    ) -> str:
        """Generate overall summary insight."""
        # Assess overall attractiveness
        if fast_seller_prob > 0.7 and predicted_dom < 14 and margin_pct > 15:
            return "Excellent opportunity: High probability of fast sale with strong margins"
        elif fast_seller_prob > 0.5 and predicted_dom < 21 and margin_pct > 10:
            return "Strong opportunity: Good balance of speed and profitability"
        elif fast_seller_prob > 0.3:
            return "Moderate opportunity: Acceptable sell probability, monitor market conditions"
        else:
            return "Lower confidence: Consider adjusting configuration or pricing strategy"
    
    def _assess_market_alignment(
        self,
        fast_seller_prob: float,
        predicted_dom: float
    ) -> Dict[str, Any]:
        """Assess how well the configuration aligns with fast-selling market trends."""
        alignment_score = (fast_seller_prob * 0.7) + ((30 - min(predicted_dom, 30)) / 30 * 0.3)
        
        if alignment_score > 0.7:
            level = "Excellent"
            description = "Configuration strongly aligns with fast-selling market preferences"
        elif alignment_score > 0.5:
            level = "Good"
            description = "Configuration aligns well with market trends"
        elif alignment_score > 0.3:
            level = "Moderate"
            description = "Configuration has moderate market appeal"
        else:
            level = "Low"
            description = "Configuration may need adjustment to match market preferences"
        
        return {
            'score': alignment_score,
            'level': level,
            'description': description,
            'fast_seller_probability': fast_seller_prob,
            'predicted_dom': predicted_dom
        }
    
    def _identify_key_drivers(
        self,
        feature_importance: Optional[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Identify top features driving the fast-seller prediction."""
        if not feature_importance:
            return []
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5
        
        drivers = []
        for feature, importance in sorted_features:
            # Human-readable feature name
            readable_name = self._make_readable(feature)
            
            drivers.append({
                'feature': feature,
                'readable_name': readable_name,
                'importance': float(importance),
                'impact': self._describe_impact(importance)
            })
        
        return drivers
    
    def _make_readable(self, feature_name: str) -> str:
        """Convert feature name to human-readable format."""
        # Remove prefixes
        if feature_name.startswith('has_'):
            feature_name = feature_name[4:]
        if feature_name.startswith('price_'):
            feature_name = feature_name[6:]
        
        # Replace underscores with spaces and title case
        readable = feature_name.replace('_', ' ').title()
        
        # Common feature name mappings
        mappings = {
            'Beds': 'Number of Bedrooms',
            'Baths': 'Number of Bathrooms',
            'Sqft': 'Square Footage',
            'Price Per Sqft': 'Price per Square Foot',
            'Lot Size Acres': 'Lot Size (Acres)',
        }
        
        return mappings.get(readable, readable)
    
    def _describe_impact(self, importance: float) -> str:
        """Describe the impact level of a feature."""
        if importance > 0.15:
            return "High impact - strongly influences fast-seller probability"
        elif importance > 0.08:
            return "Moderate impact - meaningful contributor to sell speed"
        elif importance > 0.04:
            return "Low-moderate impact - slight influence on market appeal"
        else:
            return "Minimal impact - negligible effect"
    
    def _extract_pattern_insights(
        self,
        configuration: Dict[str, Any],
        discovered_patterns: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        """Extract relevant insights from discovered patterns."""
        if not discovered_patterns:
            return []
        
        insights = []
        config_features = set([
            f"{configuration.get('beds', '')}BR",
            f"{configuration.get('baths', '')}BA",
        ])
        
        # Check if configuration matches any discovered patterns
        for pattern in discovered_patterns[:3]:  # Top 3 patterns
            pattern_features = set(pattern.get('features', []))
            
            # Check for overlap
            overlap = config_features.intersection(pattern_features)
            if overlap:
                lift = pattern.get('lift', 1.0)
                insights.append(
                    f"Configuration matches market pattern: Properties with similar features "
                    f"sell {lift:.1f}x faster than average"
                )
        
        return insights
    
    def _generate_actionable_recommendations(
        self,
        configuration: Dict[str, Any],
        fast_seller_prob: float,
        predicted_dom: float,
        feature_importance: Optional[Dict[str, float]]
    ) -> List[str]:
        """Generate actionable recommendations for improving the configuration."""
        recommendations = []
        
        # Speed recommendations
        if predicted_dom > 21:
            recommendations.append(
                "Consider adding high-impact features to reduce DOM by 5-10 days"
            )
        
        if fast_seller_prob < 0.5:
            recommendations.append(
                "Configuration may benefit from trending features to increase sell probability"
            )
        
        # Feature recommendations based on importance
        if feature_importance:
            top_missing = self._identify_missing_high_importance_features(
                configuration, feature_importance
            )
            if top_missing:
                recommendations.append(
                    f"Consider adding: {', '.join(top_missing[:2])} "
                    f"to improve market appeal"
                )
        
        return recommendations
    
    def _identify_missing_high_importance_features(
        self,
        configuration: Dict[str, Any],
        feature_importance: Dict[str, float]
    ) -> List[str]:
        """Identify high-importance features that are missing from configuration."""
        # Get top features
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        missing = []
        for feature, importance in top_features:
            if importance > 0.05:  # Only significant features
                # Check if feature is present (simplified check)
                if feature.startswith('has_'):
                    feature_name = feature[4:].replace('_', ' ').title()
                    missing.append(feature_name)
        
        return missing[:3]  # Top 3 missing features
    
    def explain_fast_seller_prediction(
        self,
        fast_seller_prob: float,
        predicted_dom: float,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> str:
        """Generate human-readable explanation of fast-seller prediction."""
        explanation_parts = []
        
        # Overall assessment
        if fast_seller_prob > 0.7:
            explanation_parts.append(
                f"This configuration has a {fast_seller_prob*100:.0f}% probability "
                f"of selling within 14 days, indicating strong market appeal."
            )
        elif fast_seller_prob > 0.5:
            explanation_parts.append(
                f"This configuration has a {fast_seller_prob*100:.0f}% probability "
                f"of selling quickly, showing good market alignment."
            )
        else:
            explanation_parts.append(
                f"This configuration has a {fast_seller_prob*100:.0f}% probability "
                f"of fast sale, which may require additional market optimization."
            )
        
        # DOM prediction
        if predicted_dom < 14:
            explanation_parts.append(
                f"Expected time to pending is {predicted_dom:.0f} days, indicating "
                f"high buyer interest."
            )
        elif predicted_dom < 21:
            explanation_parts.append(
                f"Expected time to pending is {predicted_dom:.0f} days, within typical "
                f"market range."
            )
        else:
            explanation_parts.append(
                f"Expected time to pending is {predicted_dom:.0f} days, suggesting "
                f"longer market exposure may be needed."
            )
        
        # Top drivers
        if feature_importance:
            top_driver = max(feature_importance.items(), key=lambda x: x[1])
            driver_name = self._make_readable(top_driver[0])
            explanation_parts.append(
                f"The primary driver is {driver_name}, which strongly influences "
                f"buyer appeal."
            )
        
        return " ".join(explanation_parts)
    
    def suggest_feature_additions(
        self,
        current_features: Dict[str, Any],
        discovered_patterns: List[Dict[str, Any]],
        feature_impact: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Recommend features to add based on patterns and impact analysis.
        
        Args:
            current_features: Current configuration features
            discovered_patterns: Patterns from pattern discovery
            feature_impact: Feature impact scores (DOM reduction)
            
        Returns:
            List of recommended features with expected impact
        """
        recommendations = []
        
        # Get top impactful features from patterns
        for pattern in discovered_patterns[:5]:
            features = pattern.get('features', [])
            lift = pattern.get('lift', 1.0)
            
            for feature in features:
                if feature not in current_features:
                    dom_reduction = feature_impact.get(feature, {}).get('dom_reduction', 0)
                    
                    recommendations.append({
                        'feature': feature,
                        'expected_lift': lift,
                        'estimated_dom_reduction': dom_reduction,
                        'rationale': f"Properties with this feature sell {lift:.1f}x faster"
                    })
        
        # Sort by impact
        recommendations.sort(key=lambda x: x.get('expected_lift', 0), reverse=True)
        return recommendations[:5]  # Top 5
    
    def estimate_dom_impact(
        self,
        feature_name: str,
        current_dom: float,
        feature_impact_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate how adding a feature would impact DOM.
        
        Args:
            feature_name: Name of feature to add
            current_dom: Current predicted DOM
            feature_impact_data: Feature impact analysis data
            
        Returns:
            Dictionary with impact estimate
        """
        impact = feature_impact_data.get(feature_name, {})
        dom_reduction = impact.get('dom_reduction', 0)
        new_dom = max(0, current_dom - dom_reduction)
        
        return {
            'feature': feature_name,
            'current_dom': current_dom,
            'new_dom': new_dom,
            'reduction': dom_reduction,
            'reduction_pct': (dom_reduction / current_dom * 100) if current_dom > 0 else 0,
            'estimated_impact': f"Adding {feature_name} would reduce DOM by approximately {dom_reduction:.0f} days"
        }


# Singleton instance
insights_generator = InsightsGenerator()
