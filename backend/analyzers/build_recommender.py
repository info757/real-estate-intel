"""
Build Recommendation Engine
Generates specific build recommendations for land parcels by combining:
- Feature analysis (what features sell)
- Demand prediction (what configurations sell)
- Financial modeling (what makes money)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from backend.analyzers.feature_analyzer import feature_analyzer
from backend.analyzers.demand_predictor import demand_predictor
from backend.analyzers.financial_optimizer import FinancialOptimizer
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildRecommender:
    """
    Generates comprehensive build recommendations for specific lots.
    """
    
    def __init__(self):
        self.feature_analyzer = feature_analyzer
        self.demand_predictor = demand_predictor
        self.financial_optimizer = FinancialOptimizer()
    
    def recommend_for_lot(
        self,
        lot_address: str,
        lot_zip_code: str,
        lot_price: float,
        lot_acreage: float,
        zoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive build recommendation for a specific lot.
        
        Args:
            lot_address: Lot address
            lot_zip_code: ZIP code
            lot_price: Purchase price for the lot
            lot_acreage: Lot size in acres
            zoning: Zoning designation
        
        Returns:
            Comprehensive build recommendation
        """
        logger.info(f"Generating recommendation for lot: {lot_address}")
        
        try:
            # Step 1: Predict optimal configuration based on market demand
            logger.info("Analyzing market demand...")
            demand_analysis = self.demand_predictor.predict_optimal_config(
                lot_zip_code,
                lot_size=lot_acreage,
                months_back=12,
                min_samples=5
            )
            
            if 'error' in demand_analysis:
                return {
                    "error": "Insufficient market data",
                    "details": demand_analysis
                }
            
            optimal_config = demand_analysis['optimal_config']
            
            # Step 2: Analyze features that drive sales
            logger.info("Analyzing feature impact...")
            feature_analysis = self.feature_analyzer.analyze_feature_impact(
                lot_zip_code,
                months_back=12,
                min_samples=5
            )
            
            if 'error' in feature_analysis:
                feature_analysis = {"interior_features": [], "exterior_features": [], "utilities": []}
            
            # Step 3: Select must-have features
            interior_features = self._select_features(
                feature_analysis.get('interior_features', []),
                top_n=5
            )
            
            exterior_features = self._select_features(
                feature_analysis.get('exterior_features', []),
                top_n=5
            )
            
            utilities = self._select_features(
                feature_analysis.get('utilities', []),
                top_n=3
            )
            
            # Step 4: Calculate financial projections
            logger.info("Calculating financial projections...")
            financials = self._calculate_financials(
                lot_price=lot_price,
                sqft=optimal_config['sqft'],
                projected_sale_price=optimal_config['median_sale_price'],
                estimated_features_cost=self._estimate_features_cost(
                    interior_features,
                    exterior_features
                )
            )
            
            # Step 5: Get comparable sales
            comps = self._get_comparable_sales(demand_analysis)
            
            # Step 6: Generate market rationale
            rationale = self._generate_rationale(
                optimal_config,
                demand_analysis,
                feature_analysis
            )
            
            # Step 7: Calculate confidence score
            confidence = self._calculate_overall_confidence(
                optimal_config,
                feature_analysis,
                demand_analysis
            )
            
            return {
                "lot": {
                    "address": lot_address,
                    "zip_code": lot_zip_code,
                    "price": lot_price,
                    "acreage": lot_acreage,
                    "zoning": zoning
                },
                "recommendation": {
                    "bedrooms": optimal_config['bedrooms'],
                    "bathrooms": optimal_config['bathrooms'],
                    "sqft": optimal_config['sqft'],
                    "configuration": optimal_config['configuration'],
                    "stories": self._estimate_stories(optimal_config['sqft']),
                    "style": self._suggest_style(feature_analysis)
                },
                "interior_features": interior_features,
                "exterior_features": exterior_features,
                "utilities": utilities,
                "market_rationale": rationale,
                "financial_projection": financials,
                "comparable_sales": comps,
                "confidence": confidence,
                "analysis_date": datetime.now().isoformat(),
                "data_source": "Attom Data API"
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return {
                "error": str(e),
                "lot_address": lot_address
            }
    
    def _select_features(
        self,
        features: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Select top features for recommendation."""
        
        result = []
        
        for feature in features[:top_n]:
            result.append({
                "feature": feature['feature'],
                "value": feature['value'],
                "priority": feature['priority'],
                "rationale": feature['rationale'],
                "frequency": feature.get('frequency_pct', 0),
                "price_impact": feature.get('price_premium', 0)
            })
        
        return result
    
    def _calculate_financials(
        self,
        lot_price: float,
        sqft: int,
        projected_sale_price: float,
        estimated_features_cost: float
    ) -> Dict[str, Any]:
        """Calculate financial projections."""
        
        # Construction costs
        base_construction_cost = sqft * settings.default_construction_cost_per_sqft
        total_construction_cost = base_construction_cost + estimated_features_cost
        
        # Carrying costs
        carrying_costs = (
            settings.default_carrying_cost_monthly *
            (settings.default_build_time_months + settings.default_sale_time_months)
        )
        
        # Total investment
        total_investment = lot_price + total_construction_cost + carrying_costs
        
        # Profit calculations
        gross_profit = projected_sale_price - total_investment
        net_profit = gross_profit * 0.9  # Account for closing costs, etc.
        
        # ROI
        roi = (net_profit / total_investment * 100) if total_investment > 0 else 0
        
        # IRR (simplified monthly calculation)
        timeline_months = settings.default_build_time_months + settings.default_sale_time_months
        monthly_return = (projected_sale_price / total_investment) ** (1 / timeline_months) - 1
        irr = (((1 + monthly_return) ** 12) - 1) * 100
        
        # Determine confidence level
        if roi > 15 and irr > 25:
            confidence_level = "high"
        elif roi > 10 and irr > 18:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return {
            "lot_cost": lot_price,
            "construction_cost": total_construction_cost,
            "base_construction_cost": base_construction_cost,
            "features_cost": estimated_features_cost,
            "carrying_costs": carrying_costs,
            "total_investment": total_investment,
            "projected_sale_price": projected_sale_price,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "roi": round(roi, 1),
            "irr": round(irr, 1),
            "timeline_months": timeline_months,
            "confidence_level": confidence_level
        }
    
    def _estimate_features_cost(
        self,
        interior_features: List[Dict[str, Any]],
        exterior_features: List[Dict[str, Any]]
    ) -> float:
        """Estimate cost of recommended features."""
        
        # Simplified cost estimation
        # In production, you'd have a detailed feature cost database
        
        total_cost = 0
        
        # Interior features cost estimation
        interior_cost_map = {
            "granite": 5000,
            "quartz": 6000,
            "hardwood": 8000,
            "tile": 3000,
            "fireplace": 4000,
            "smart_home": 2000
        }
        
        for feature in interior_features:
            feature_name_lower = feature['feature'].lower()
            for key, cost in interior_cost_map.items():
                if key in feature_name_lower:
                    total_cost += cost
                    break
        
        # Exterior features cost estimation
        exterior_cost_map = {
            "deck": 5000,
            "patio": 4000,
            "fence": 3000,
            "sprinkler": 2500,
            "garage": 15000
        }
        
        for feature in exterior_features:
            feature_name_lower = feature['feature'].lower()
            for key, cost in exterior_cost_map.items():
                if key in feature_name_lower:
                    total_cost += cost
                    break
        
        return total_cost
    
    def _get_comparable_sales(
        self,
        demand_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract comparable sales from demand analysis."""
        
        # Get the optimal configuration
        optimal_config = demand_analysis['optimal_config']
        config_name = optimal_config['configuration']
        
        # Find this configuration in all configurations
        all_configs = demand_analysis.get('all_configurations', [])
        
        for config in all_configs:
            if config['configuration'] == config_name:
                return [{
                    "configuration": config['configuration'],
                    "sales_count": config['sales_count'],
                    "median_price": config['median_price'],
                    "median_size": config['median_size'],
                    "avg_price_per_sqft": config['avg_price_per_sqft']
                }]
        
        return []
    
    def _generate_rationale(
        self,
        optimal_config: Dict[str, Any],
        demand_analysis: Dict[str, Any],
        feature_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate market rationale for the recommendation."""
        
        # Configuration rationale
        config_rationale = optimal_config.get('rationale', '')
        
        # Market context
        market_context = demand_analysis.get('market_context', {})
        
        # Feature insights
        interior_count = len(feature_analysis.get('interior_features', []))
        exterior_count = len(feature_analysis.get('exterior_features', []))
        
        return {
            "configuration": config_rationale,
            "market_temperature": self._assess_market_temperature(
                optimal_config.get('sales_velocity', 0)
            ),
            "features": f"Analyzed {interior_count} interior and {exterior_count} exterior features",
            "competition": self._assess_competition(demand_analysis)
        }
    
    def _assess_market_temperature(self, sales_velocity: float) -> str:
        """Assess market temperature based on sales velocity."""
        
        if sales_velocity > 3:
            return "hot - very high demand"
        elif sales_velocity > 1.5:
            return "warm - good demand"
        elif sales_velocity > 0.5:
            return "moderate - average demand"
        else:
            return "cool - lower demand"
    
    def _assess_competition(self, demand_analysis: Dict[str, Any]) -> str:
        """Assess competition level."""
        
        all_configs = demand_analysis.get('all_configurations', [])
        
        if not all_configs:
            return "Unknown competition level"
        
        total_sales = sum(c['sales_count'] for c in all_configs)
        optimal_sales = all_configs[0]['sales_count']
        
        market_share = (optimal_sales / total_sales * 100) if total_sales > 0 else 0
        
        if market_share > 40:
            return f"dominant configuration ({market_share:.0f}% of market)"
        elif market_share > 25:
            return f"popular configuration ({market_share:.0f}% of market)"
        else:
            return f"moderate competition ({market_share:.0f}% of market)"
    
    def _calculate_overall_confidence(
        self,
        optimal_config: Dict[str, Any],
        feature_analysis: Dict[str, Any],
        demand_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall confidence in the recommendation."""
        
        # Demand confidence
        demand_confidence = optimal_config.get('confidence', 0)
        
        # Data quality (sample size)
        sample_size = demand_analysis.get('property_count', 0)
        data_quality = min(sample_size / 50, 1.0)  # Max at 50 properties
        
        # Feature coverage
        feature_count = (
            len(feature_analysis.get('interior_features', [])) +
            len(feature_analysis.get('exterior_features', []))
        )
        feature_coverage = min(feature_count / 10, 1.0)  # Max at 10 features
        
        # Overall confidence (weighted average)
        overall = (
            demand_confidence * 0.5 +
            data_quality * 0.3 +
            feature_coverage * 0.2
        )
        
        if overall > 0.7:
            level = "high"
        elif overall > 0.5:
            level = "medium"
        else:
            level = "low"
        
        return {
            "score": round(overall, 2),
            "level": level,
            "factors": {
                "demand_confidence": round(demand_confidence, 2),
                "data_quality": round(data_quality, 2),
                "feature_coverage": round(feature_coverage, 2)
            },
            "sample_size": sample_size
        }
    
    def _estimate_stories(self, sqft: int) -> int:
        """Estimate number of stories based on square footage."""
        if sqft < 1800:
            return 1
        elif sqft < 3000:
            return 2
        else:
            return 2  # Could be 2 or 3, default to 2
    
    def _suggest_style(self, feature_analysis: Dict[str, Any]) -> str:
        """Suggest architectural style based on market features."""
        
        # Simplified style suggestion
        # In production, you'd analyze actual architectural styles from sales data
        
        styles = ["craftsman", "traditional", "modern", "colonial", "ranch"]
        
        # Default to traditional
        return "traditional"


# Global instance
build_recommender = BuildRecommender()

