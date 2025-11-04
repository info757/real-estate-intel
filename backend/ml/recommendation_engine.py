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
        print(f"[DEBUG] Starting recommendation generation for ZIP {lot_features.get('zip_code', 'unknown')}")  # Visible in Streamlit
        
        # Check if model files exist - use absolute path relative to project root
        import os
        from pathlib import Path
        
        # Get the project root (assuming this file is in backend/ml/)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # Go up from backend/ml/ to project root
        model_dir = project_root / "models"
        pricing_model_path = model_dir / "pricing_model.pkl"
        demand_model_path = model_dir / "demand_model_probability.pkl"
        
        print(f"[DEBUG] Project root: {project_root}")
        print(f"[DEBUG] Model directory: {model_dir}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        
        pricing_exists = pricing_model_path.exists()
        demand_exists = demand_model_path.exists()
        
        print(f"[DEBUG] Pricing model file exists: {pricing_exists} at {pricing_model_path}")
        print(f"[DEBUG] Demand model file exists: {demand_exists} at {demand_model_path}")
        
        if not pricing_exists or not demand_exists:
            missing = []
            if not pricing_exists:
                missing.append(f"pricing model ({pricing_model_path})")
            if not demand_exists:
                missing.append(f"demand model ({demand_model_path})")
            error_msg = f'Model files not found: {", ".join(missing)}. Please train models first using: python backend/ml/train_models.py'
            print(f"[DEBUG] ERROR: {error_msg}")
            logger.error(error_msg)
            return {
                'error': error_msg,
                'recommendations': [],
                'total_evaluated': 0
            }
        
        # Update model_dir on singleton instances to use Path object (not string)
        pricing_model.model_dir = model_dir
        demand_model.model_dir = model_dir
        
        # Try to load models if they exist but aren't loaded
        try:
            if pricing_model.model is None:
                try:
                    print("[DEBUG] Loading pricing model...")
                    pricing_model.load()
                    logger.info("Loaded pricing model from disk")
                    print("[DEBUG] Pricing model loaded successfully")
                except Exception as e:
                    error_msg = f"Could not load pricing model: {e}"
                    logger.error(error_msg)
                    print(f"[DEBUG] ERROR: {error_msg}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return {
                        'error': f'Failed to load pricing model: {str(e)}. Please check model file.',
                        'recommendations': [],
                        'total_evaluated': 0
                    }
            
            if demand_model.sell_probability_model is None or demand_model.dom_model is None:
                try:
                    print("[DEBUG] Loading demand model...")
                    demand_model.load()
                    logger.info("Loaded demand model from disk")
                    print("[DEBUG] Demand model loaded successfully")
                except Exception as e:
                    error_msg = f"Could not load demand model: {e}"
                    logger.error(error_msg)
                    print(f"[DEBUG] ERROR: {error_msg}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return {
                        'error': f'Failed to load demand model: {str(e)}. Please check model file.',
                        'recommendations': [],
                        'total_evaluated': 0
                    }
        except Exception as e:
            error_msg = f"Error loading models: {e}"
            logger.error(error_msg)
            print(f"[DEBUG] ERROR: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'error': f'Error loading models: {str(e)}',
                'recommendations': [],
                'total_evaluated': 0
            }
        
        # Check if models are loaded
        if pricing_model.model is None:
            error_msg = 'Pricing model not loaded. Model file exists but failed to load.'
            print(f"[DEBUG] ERROR: {error_msg}")
            logger.error(error_msg)
            return {
                'error': error_msg,
                'recommendations': [],
                'total_evaluated': 0
            }
        
        if demand_model.sell_probability_model is None or demand_model.dom_model is None:
            error_msg = 'Demand model not loaded. Model file exists but failed to load.'
            print(f"[DEBUG] ERROR: {error_msg}")
            logger.error(error_msg)
            return {
                'error': error_msg,
                'recommendations': [],
                'total_evaluated': 0
            }
        
        print("[DEBUG] Models loaded successfully, generating candidate configurations...")
        
        # Generate candidate configurations if not provided
        if candidate_configs is None:
            candidate_configs = self._generate_candidate_configs(property_type, lot_features)
        
        logger.info(f"Evaluating {len(candidate_configs)} candidate configurations")
        
        # Evaluate each configuration
        results = []
        evaluation_errors = []
        for i, config in enumerate(candidate_configs):
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
                else:
                    # Log first few failures for debugging
                    if len(evaluation_errors) < 3:
                        evaluation_errors.append(f"Config {i+1}: {config.get('beds', '?')}BR/{config.get('baths', '?')}BA - evaluation returned None")
            except Exception as e:
                error_msg = f"Config {i+1}: {config.get('beds', '?')}BR/{config.get('baths', '?')}BA - {str(e)}"
                logger.warning(f"Error evaluating {error_msg}")
                if len(evaluation_errors) < 3:
                    evaluation_errors.append(error_msg)
                continue
        
        logger.info(f"Successfully evaluated {len(results)} configurations out of {len(candidate_configs)} candidates")
        
        if not results:
            error_msg = f'No valid configurations evaluated after trying {len(candidate_configs)} candidates.'
            if evaluation_errors:
                error_msg += f' Sample errors: {"; ".join(evaluation_errors)}'
            else:
                error_msg += ' All evaluations returned None - check logs for details.'
            return {
                'error': error_msg,
                'recommendations': [],
                'total_evaluated': len(candidate_configs),
                'successful_evaluations': 0
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
            # Ensure models are loaded before prediction
            if pricing_model.model is None:
                try:
                    pricing_model.load()
                    logger.debug("Loaded pricing model in _evaluate_configuration")
                except Exception as e:
                    logger.error(f"Failed to load pricing model: {e}")
                    return None
            
            if demand_model.sell_probability_model is None or demand_model.dom_model is None:
                try:
                    demand_model.load()
                    logger.debug("Loaded demand model in _evaluate_configuration")
                except Exception as e:
                    logger.error(f"Failed to load demand model: {e}")
                    return None
            
            # 1. Create feature vector for ML models
            # Ensure property_type is in lot_features for feature vector creation
            lot_features_with_type = lot_features.copy()
            lot_features_with_type['property_type'] = property_type
            
            feature_vector = self._create_feature_vector(
                config=config,
                lot_features=lot_features_with_type,
                historical_data=historical_data,
                current_listings_context=current_listings_context
            )
            
            if feature_vector is None or feature_vector.empty:
                logger.warning(f"Feature vector is None or empty for config {config}")
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
        Create feature vector for ML models with all required features.
        """
        try:
            import pandas as pd
            import numpy as np
            from datetime import datetime
            
            # Basic features from config
            beds = config.get('beds', config.get('bedrooms', 3))
            baths = config.get('baths', config.get('bathrooms', 2.5))
            sqft = config.get('sqft', config.get('square_feet', 2000))
            lot_size_acres = lot_features.get('lot_size_acres', 0.25)
            lot_size_sqft = lot_size_acres * 43560  # Convert acres to sqft
            year_built = 2024  # New construction
            stories = config.get('stories', 1 if sqft < 2000 else 2)
            latitude = lot_features.get('latitude', 36.089)
            longitude = lot_features.get('longitude', -79.908)
            subdivision = lot_features.get('subdivision', '')
            property_type = lot_features.get('property_type', config.get('property_type', 'Single Family Home'))
            
            # Calculate market-based features from historical data if available
            # These are needed for price_per_sqft, price_per_bedroom, etc.
            market_price_per_sqft = 150.0  # Default fallback
            if historical_data:
                try:
                    from backend.ml.feature_engineering import feature_engineer
                    # Create a temporary DataFrame from historical data to calculate market stats
                    hist_df = feature_engineer.engineer_features(historical_data)
                    if not hist_df.empty and 'price_per_sqft' in hist_df.columns:
                        market_price_per_sqft = hist_df['price_per_sqft'].median()
                        if pd.isna(market_price_per_sqft) or market_price_per_sqft <= 0:
                            market_price_per_sqft = 150.0
                except Exception as e:
                    logger.warning(f"Could not calculate market stats from historical data: {e}")
            
            # Calculate subdivision size (number of properties in subdivision)
            subdivision_size = 50  # Default
            if historical_data and subdivision:
                try:
                    subdivision_size = len([p for p in historical_data 
                                          if self._safe_get(p, ['area', 'subdname']) == subdivision])
                    subdivision_size = max(subdivision_size, 10)  # Minimum 10
                except:
                    subdivision_size = 50
            
            # Finish level to quality score mapping
            finish_scores = {'starter': 4, 'standard': 5, 'premium': 7, 'luxury': 9}
            quality_score = finish_scores.get(config.get('finish_level', 'standard'), 5)
            
            # Property type indicators
            is_sfr = 1 if 'single' in property_type.lower() or property_type == 'SFR' else 0
            is_townhome = 1 if 'town' in property_type.lower() or 'townhouse' in property_type.lower() else 0
            is_condo = 1 if 'condo' in property_type.lower() else 0
            
            # Temporal features (using current date for new construction)
            now = datetime.now()
            age_years = 0  # New construction
            sale_year = now.year
            sale_month = now.month
            sale_quarter = (sale_month - 1) // 3 + 1
            is_spring_summer = 1 if sale_month in [3, 4, 5, 6, 7, 8] else 0
            days_since_sale = 0  # New construction, not sold yet
            
            # Calculate derived features
            price_per_sqft = market_price_per_sqft  # Will be adjusted by model, but need base value
            price_per_bedroom = price_per_sqft * sqft / beds if beds > 0 else 0
            price_per_bathroom = price_per_sqft * sqft / baths if baths > 0 else 0
            lot_coverage_ratio = sqft / lot_size_sqft if lot_size_sqft > 0 else 0
            beds_per_1000sqft = (beds / sqft) * 1000 if sqft > 0 else 0
            bath_bed_ratio = baths / beds if beds > 0 else 0
            condition_score = 5  # New construction = good condition
            overall_quality = quality_score  # Use quality_score as overall_quality
            
            # Create feature dictionary with ALL required features
            features = {
                'beds': beds,
                'baths': baths,
                'sqft': sqft,
                'lot_size_acres': lot_size_acres,
                'lot_size_sqft': lot_size_sqft,
                'year_built': year_built,
                'stories': stories,
                'latitude': latitude,
                'longitude': longitude,
                'price_per_sqft': price_per_sqft,
                'price_per_bedroom': price_per_bedroom,
                'price_per_bathroom': price_per_bathroom,
                'lot_coverage_ratio': lot_coverage_ratio,
                'beds_per_1000sqft': beds_per_1000sqft,
                'bath_bed_ratio': bath_bed_ratio,
                'age_years': age_years,
                'sale_year': sale_year,
                'sale_month': sale_month,
                'sale_quarter': sale_quarter,
                'is_spring_summer': is_spring_summer,
                'days_since_sale': days_since_sale,
                'is_sfr': is_sfr,
                'is_townhome': is_townhome,
                'is_condo': is_condo,
                'subdivision_size': subdivision_size,
                'quality_score': quality_score,
                'condition_score': condition_score,
                'overall_quality': overall_quality,
            }
            
            # Convert to DataFrame (single row)
            df = pd.DataFrame([features])
            
            # Ensure all numeric columns are numeric
            numeric_cols = [col for col in df.columns if col not in ['subdivision', 'proptype']]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill any NaN values with defaults
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _safe_get(self, obj: Any, keys: List[str]) -> Any:
        """Safely get nested dictionary value."""
        for key in keys:
            if isinstance(obj, dict):
                obj = obj.get(key, None)
            else:
                return None
            if obj is None:
                return None
        return obj
    
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
