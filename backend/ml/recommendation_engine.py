"""
Recommendation Engine
Combines pricing, demand, and cost models to generate optimal build recommendations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from itertools import product

from backend.ml.pricing_model import pricing_model
from backend.ml.demand_model import demand_model
from backend.ml.cost_estimator import cost_estimator, CostBreakdown
from backend.ml.feature_engineering import feature_engineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Main recommendation engine that generates optimal build recommendations.
    
    Process:
    1. Generate candidate house configurations
    2. For each configuration:
       - Predict sale price
       - Predict demand (sell probability + DOM)
       - Estimate construction cost
       - Calculate margin
    3. Rank by margin, subject to demand constraints
    4. Return top recommendations with explanations
    """
    
    def __init__(
        self,
        min_sell_probability: float = 0.70,  # Must have ≥70% chance to sell within threshold
        max_dom: int = 90,  # Maximum acceptable days on market
        min_margin_pct: float = 15.0,  # Minimum 15% gross margin
        sga_allocation: float = 0.10  # 10% of price for SG&A
    ):
        """
        Initialize recommendation engine.
        
        Args:
            min_sell_probability: Minimum probability of selling within threshold (0-1)
            max_dom: Maximum acceptable days on market
            min_margin_pct: Minimum gross margin percentage
            sga_allocation: SG&A allocation as fraction of sale price
        """
        self.min_sell_probability = min_sell_probability
        self.max_dom = max_dom
        self.min_margin_pct = min_margin_pct
        self.sga_allocation = sga_allocation
        
    def generate_recommendations(
        self,
        lot_features: Dict[str, Any],
        property_type: str = 'Single Family Home',
        candidate_configs: Optional[List[Dict[str, Any]]] = None,
        historical_data: Optional[List[Dict[str, Any]]] = None,
        current_listings_context: Optional[Dict[str, Any]] = None,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Generate optimal build recommendations for a specific lot.
        
        Args:
            lot_features: Dictionary with lot information:
                - zip_code: str
                - latitude: float
                - longitude: float
                - lot_size_acres: float (optional)
                - lot_condition: str (optional, default: 'flat')
                - utilities_status: str (optional, default: 'all_utilities')
                - subdivision: str (optional)
            property_type: 'Single Family Home', 'Townhome', or 'Condo'
            candidate_configs: Optional list of specific configurations to evaluate.
                             If None, generates candidates automatically.
            historical_data: Optional historical sales data for feature engineering context
            current_listings_context: Optional competitive context from listings scraper
            top_n: Number of top recommendations to return
            
        Returns:
            Dictionary with recommendations and metadata
        """
        logger.info(f"Generating recommendations for lot in {lot_features.get('zip_code', 'unknown ZIP')}")
        
        # Generate candidate configurations if not provided
        if candidate_configs is None:
            candidate_configs = self._generate_candidate_configs(property_type, lot_features)
        
        logger.info(f"Evaluating {len(candidate_configs)} candidate configurations")
        
        # Evaluate each configuration
        results = []
        for config in candidate_configs:
            try:
                evaluation = self._evaluate_configuration(
                    config=config,
                    lot_features=lot_features,
                    property_type=property_type,
                    historical_data=historical_data,
                    current_listings_context=current_listings_context
                )
                
                if evaluation:
                    results.append(evaluation)
            except Exception as e:
                logger.warning(f"Error evaluating config {config}: {e}")
                continue
        
        if not results:
            return {
                'error': 'No valid configurations evaluated',
                'recommendations': [],
                'total_evaluated': 0
            }
        
        # Filter by constraints
        filtered_results = self._apply_constraints(results)
        
        # Sort by margin (descending)
        filtered_results.sort(key=lambda x: x['margin']['gross_margin'], reverse=True)
        
        # Get top N
        top_recommendations = filtered_results[:top_n]
        
        # Generate explanations for top recommendations
        for rec in top_recommendations:
            rec['explanation'] = self._generate_explanation(rec, filtered_results)
        
        return {
            'recommendations': top_recommendations,
            'total_evaluated': len(results),
            'total_passing_constraints': len(filtered_results),
            'constraints': {
                'min_sell_probability': self.min_sell_probability,
                'max_dom': self.max_dom,
                'min_margin_pct': self.min_margin_pct
            },
            'lot_features': lot_features,
            'property_type': property_type
        }
    
    def _generate_candidate_configs(
        self,
        property_type: str,
        lot_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate house configurations to evaluate.
        
        Args:
            property_type: Property type
            lot_features: Lot features (may inform size constraints)
            
        Returns:
            List of configuration dictionaries
        """
        configs = []
        
        # Define ranges based on property type
        if property_type == 'Single Family Home':
            bed_range = [3, 4, 5]
            bath_range = [2.0, 2.5, 3.0, 3.5, 4.0]
            sqft_ranges = [
                (1500, 1800),   # Small
                (1800, 2200),   # Medium-small
                (2200, 2600),   # Medium
                (2600, 3000),   # Medium-large
                (3000, 3500),   # Large
            ]
            finish_levels = ['standard', 'premium']
        elif property_type == 'Townhome':
            bed_range = [2, 3, 4]
            bath_range = [2.0, 2.5, 3.0]
            sqft_ranges = [
                (1200, 1600),
                (1600, 2000),
                (2000, 2400),
            ]
            finish_levels = ['standard', 'premium']
        else:  # Condo
            bed_range = [1, 2, 3]
            bath_range = [1.0, 1.5, 2.0, 2.5]
            sqft_ranges = [
                (800, 1200),
                (1200, 1600),
                (1600, 2000),
            ]
            finish_levels = ['standard', 'premium']
        
        # Generate combinations
        for beds, baths, (sqft_min, sqft_max), finish in product(
            bed_range, bath_range, sqft_ranges, finish_levels
        ):
            # Skip unrealistic combinations (e.g., 1 bed / 3 baths)
            if baths > beds + 1:
                continue
            
            # Use midpoint of sqft range (or could sample multiple)
            sqft = (sqft_min + sqft_max) // 2
            
            configs.append({
                'beds': beds,
                'baths': baths,
                'sqft': sqft,
                'finish_level': finish,
                'property_type': property_type,
                'stories': 1 if sqft < 2000 else 2,  # Smaller homes typically single-story
                'garage_spaces': 1 if property_type != 'Single Family Home' else 2
            })
        
        logger.info(f"Generated {len(configs)} candidate configurations")
        return configs
    
    def _evaluate_configuration(
        self,
        config: Dict[str, Any],
        lot_features: Dict[str, Any],
        property_type: str,
        historical_data: Optional[List[Dict[str, Any]]] = None,
        current_listings_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single configuration.
        
        Returns:
            Evaluation dictionary or None if evaluation fails
        """
        try:
            # 1. Create feature vector for ML models
            feature_vector = self._create_feature_vector(
                config=config,
                lot_features=lot_features,
                historical_data=historical_data,
                current_listings_context=current_listings_context
            )
            
            if feature_vector is None or feature_vector.empty:
                return None
            
            # 2. Predict price
            predicted_price = pricing_model.predict(feature_vector)[0]
            
            # 3. Predict demand
            demand_pred = demand_model.predict(feature_vector)
            sell_probability = demand_pred['sell_probability'][0]
            expected_dom = demand_pred['expected_dom'][0]
            
            # 4. Estimate cost
            cost_breakdown = cost_estimator.estimate_cost_for_config(
                config=config,
                lot_features=lot_features
            )
            
            # 5. Calculate margin
            margin = cost_estimator.estimate_margin(
                predicted_price=predicted_price,
                cost_breakdown=cost_breakdown,
                sga_allocation=self.sga_allocation
            )
            
            # 6. Calculate risk score (lower is better)
            risk_score = self._calculate_risk_score(
                sell_probability=sell_probability,
                expected_dom=expected_dom,
                margin_pct=margin['gross_margin_pct']
            )
            
            return {
                'configuration': config,
                'predicted_price': float(predicted_price),
                'demand': {
                    'sell_probability': float(sell_probability),
                    'expected_dom': float(expected_dom),
                    'meets_demand_threshold': sell_probability >= self.min_sell_probability
                },
                'cost': {
                    'construction_cost': float(cost_breakdown.total_cost),
                    'cost_per_sqft': float(cost_breakdown.cost_per_sqft),
                    'cost_breakdown': {
                        'base_cost': float(cost_breakdown.base_cost),
                        'lot_adjustment': float(cost_breakdown.lot_adjustment),
                        'size_adjustment': float(cost_breakdown.size_adjustment),
                        'location_adjustment': float(cost_breakdown.location_adjustment),
                    }
                },
                'margin': margin,
                'risk_score': risk_score,
                'passes_constraints': (
                    sell_probability >= self.min_sell_probability and
                    expected_dom <= self.max_dom and
                    margin['gross_margin_pct'] >= self.min_margin_pct
                )
            }
            
        except Exception as e:
            logger.error(f"Error evaluating configuration {config}: {e}")
            return None
    
    def _create_feature_vector(
        self,
        config: Dict[str, Any],
        lot_features: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None,
        current_listings_context: Optional[Dict[str, Any]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Create feature vector for ML models.
        
        This is a simplified version - in production, this would use
        the full feature_engineering module with proper data transformation.
        """
        try:
            # Create a minimal feature vector from config and lot features
            # In production, we'd use feature_engineer.create_features() with proper data
            
            # Basic features
            features = {
                'beds': config.get('beds', config.get('bedrooms', 3)),
                'baths': config.get('baths', config.get('bathrooms', 2.5)),
                'sqft': config.get('sqft', config.get('square_feet', 2000)),
                'lot_size_acres': lot_features.get('lot_size_acres', 0.25),
                'zip_code': lot_features.get('zip_code', '27410'),
                'latitude': lot_features.get('latitude', 36.089),
                'longitude': lot_features.get('longitude', -79.908),
                'year_built': 2024,  # New construction
                'age': 0,
                'subdivision': lot_features.get('subdivision', ''),
                'proptype': lot_features.get('property_type', config.get('property_type', 'SFR')),
            }
            
            # Add finish level as numeric score
            finish_scores = {'starter': 5, 'standard': 6, 'premium': 7, 'luxury': 9}
            features['quality_score'] = finish_scores.get(config.get('finish_level', 'standard'), 6)
            
            # Add competitive context if available
            if current_listings_context:
                features.update({
                    'num_similar_active_listings': current_listings_context.get('num_similar_active_listings', 0),
                    'total_active_listings': current_listings_context.get('total_active_listings', 0),
                    'inventory_level': current_listings_context.get('inventory_level', 'medium'),
                    'proposed_price_percentile': current_listings_context.get('proposed_price_percentile', 0.5),
                    'avg_dom_active': current_listings_context.get('avg_dom_active', 0),
                })
            
            # Convert to DataFrame (single row)
            df = pd.DataFrame([features])
            
            # Add derived features
            df['price_per_sqft'] = df['sqft'] / df['sqft']  # Placeholder - would use market data
            df['lot_to_house_ratio'] = (df['lot_size_acres'] * 43560) / df['sqft']
            df['beds_baths_ratio'] = df['beds'] / df['baths']
            
            # Ensure all required features are present (add defaults for missing ones)
            # This is simplified - production would use full feature engineering
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            return None
    
    def _calculate_risk_score(
        self,
        sell_probability: float,
        expected_dom: float,
        margin_pct: float
    ) -> float:
        """
        Calculate risk score (lower is better).
        
        Factors:
        - Lower sell probability = higher risk
        - Higher DOM = higher risk
        - Lower margin = higher risk
        """
        # Normalize components (0-1 scale, higher = worse)
        prob_risk = 1.0 - sell_probability  # Low prob = high risk
        dom_risk = min(expected_dom / 180.0, 1.0)  # Max DOM risk at 180 days
        margin_risk = max((15.0 - margin_pct) / 15.0, 0.0)  # Negative margin = max risk
        
        # Weighted combination
        risk_score = (0.4 * prob_risk) + (0.3 * dom_risk) + (0.3 * margin_risk)
        
        return float(risk_score)
    
    def _apply_constraints(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by constraints."""
        filtered = []
        for result in results:
            if result.get('passes_constraints', False):
                filtered.append(result)
        return filtered
    
    def _generate_explanation(
        self,
        recommendation: Dict[str, Any],
        all_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable explanation for a recommendation.
        
        In MVP, this is rule-based. In full version, LLM would be used.
        """
        config = recommendation['configuration']
        margin = recommendation['margin']
        demand = recommendation['demand']
        
        explanation_parts = []
        
        # Why this configuration
        explanation_parts.append(
            f"Recommended: {config['beds']}BR/{config['baths']}BA, "
            f"{config['sqft']:,} sqft, {config['finish_level']} finishes"
        )
        
        # Financial justification
        explanation_parts.append(
            f"Predicted sale price: ${margin['predicted_price']:,.0f} | "
            f"Construction cost: ${margin['construction_cost']:,.0f} | "
            f"Gross margin: ${margin['gross_margin']:,.0f} ({margin['gross_margin_pct']:.1f}%)"
        )
        
        # Demand justification
        explanation_parts.append(
            f"Demand: {demand['sell_probability']*100:.0f}% probability of selling within 90 days, "
            f"expected {demand['expected_dom']:.0f} days on market"
        )
        
        # Ranking context
        rank = all_results.index(recommendation) + 1
        if rank == 1:
            explanation_parts.append("Top recommendation based on highest margin while meeting demand constraints.")
        else:
            explanation_parts.append(f"Ranked #{rank} out of {len(all_results)} configurations.")
        
        return " | ".join(explanation_parts)


# Singleton instance
recommendation_engine = RecommendationEngine()
