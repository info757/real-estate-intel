"""
Land opportunity analyzer with ROI estimation.
"""

from typing import List, Optional
import logging
from backend.models.schemas import LandListing, SubmarketScore
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LandOpportunityAnalyzer:
    """Analyzes land listings for development opportunity."""
    
    def score_land_listing(self, listing: LandListing, submarket: Optional[SubmarketScore] = None) -> float:
        """Score a land listing (0-1 scale)."""
        score = 0.0
        
        # Price per acre analysis (30% weight)
        if listing.acreage and listing.acreage > 0:
            price_per_acre = listing.price / listing.acreage
            
            # Ideal range is $30k-$60k per acre in NC
            if 30000 <= price_per_acre <= 60000:
                price_score = 1.0
            elif 20000 <= price_per_acre < 30000:
                price_score = 0.8
            elif 60000 < price_per_acre <= 80000:
                price_score = 0.7
            else:
                price_score = 0.4
            
            score += price_score * 0.3
        else:
            score += 0.15  # Neutral if no acreage data
        
        # Zoning suitability (20% weight)
        if listing.zoning.value == "residential":
            score += 0.20
        elif listing.zoning.value == "mixed_use":
            score += 0.15
        else:
            score += 0.05
        
        # Utilities available (15% weight)
        utility_score = len(listing.utilities_available) / 4.0  # 4 major utilities
        score += utility_score * 0.15
        
        # Submarket quality (35% weight)
        if submarket:
            score += submarket.composite_score * 0.35
        else:
            score += 0.175  # Neutral
        
        return min(score, 1.0)
    
    def estimate_roi(self, listing: LandListing, house_sqft: int, submarket: Optional[SubmarketScore] = None) -> dict:
        """Estimate ROI for developing a house on this land."""
        # Costs
        land_cost = listing.price
        construction_cost = house_sqft * settings.default_construction_cost_per_sqft
        
        # Estimate carrying costs (property taxes, insurance, etc.)
        build_months = settings.default_build_time_months
        sale_months = settings.default_sale_time_months
        total_months = build_months + sale_months
        carrying_costs = settings.default_carrying_cost_monthly * total_months
        
        # Soft costs (permits, design, marketing, etc.) - typically 10% of construction
        soft_costs = construction_cost * 0.10
        
        total_investment = land_cost + construction_cost + carrying_costs + soft_costs
        
        # Projected sale price
        if submarket and submarket.pricing_data:
            price_per_sqft = submarket.pricing_data.median_price_per_sqft
        else:
            price_per_sqft = 175.0  # Default for NC
        
        projected_sale_price = house_sqft * price_per_sqft
        
        # Profit and returns
        gross_profit = projected_sale_price - total_investment
        roi_percentage = (gross_profit / total_investment) * 100 if total_investment > 0 else 0
        gross_margin = (gross_profit / projected_sale_price) * 100 if projected_sale_price > 0 else 0
        
        return {
            "land_cost": land_cost,
            "construction_cost": construction_cost,
            "carrying_costs": carrying_costs,
            "soft_costs": soft_costs,
            "total_investment": total_investment,
            "projected_sale_price": projected_sale_price,
            "gross_profit": gross_profit,
            "roi_percentage": roi_percentage,
            "gross_margin": gross_margin,
            "timeline_months": total_months
        }
    
    def rank_land_opportunities(self, listings: List[LandListing], submarkets: List[SubmarketScore] = None) -> List[dict]:
        """Rank land listings by opportunity score."""
        logger.info(f"Ranking {len(listings)} land opportunities")
        
        # Create submarket lookup
        submarket_map = {}
        if submarkets:
            for sm in submarkets:
                submarket_map[sm.city.lower()] = sm
        
        ranked = []
        for listing in listings:
            # Find matching submarket
            submarket = submarket_map.get(listing.city.lower())
            
            # Score the listing
            opportunity_score = self.score_land_listing(listing, submarket)
            
            # Estimate ROI for a typical house (2000 sqft)
            roi_est = self.estimate_roi(listing, 2000, submarket)
            
            ranked.append({
                "listing": listing,
                "opportunity_score": opportunity_score,
                "roi_estimate": roi_est,
                "submarket_score": submarket.composite_score if submarket else None
            })
        
        # Sort by opportunity score
        ranked.sort(key=lambda x: x["opportunity_score"], reverse=True)
        
        logger.info(f"Top opportunity score: {ranked[0]['opportunity_score']:.3f}" if ranked else "No opportunities found")
        
        return ranked

