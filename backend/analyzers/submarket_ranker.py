"""
Submarket ranking algorithm with configurable weights.
"""

from typing import List
import logging
from backend.models.schemas import SubmarketScore
from backend.data_collectors.market_data import MarketAnalysisCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubmarketRanker:
    """Ranks submarkets based on composite scores."""
    
    def __init__(self):
        self.market_collector = MarketAnalysisCollector()
    
    def rank_submarkets(self, locations: List[dict]) -> List[SubmarketScore]:
        """Analyze and rank multiple submarkets."""
        logger.info(f"Ranking {len(locations)} submarkets")
        
        submarkets = self.market_collector.analyze_multiple_submarkets(locations)
        
        # Already sorted by composite score in analyze_multiple_submarkets
        logger.info(f"Ranking complete. Top submarket: {submarkets[0].city if submarkets else 'None'}")
        
        return submarkets
    
    def get_top_submarkets(self, locations: List[dict], top_n: int = 10) -> List[SubmarketScore]:
        """Get top N submarkets."""
        all_submarkets = self.rank_submarkets(locations)
        return all_submarkets[:top_n]
    
    def compare_submarkets(self, submarket1: SubmarketScore, submarket2: SubmarketScore) -> dict:
        """Compare two submarkets in detail."""
        comparison = {
            "submarket1": {
                "name": f"{submarket1.city}, {submarket1.county}",
                "composite_score": submarket1.composite_score,
                "school_score": submarket1.school_score,
                "crime_score": submarket1.crime_score,
                "growth_score": submarket1.growth_score,
                "price_score": submarket1.price_score
            },
            "submarket2": {
                "name": f"{submarket2.city}, {submarket2.county}",
                "composite_score": submarket2.composite_score,
                "school_score": submarket2.school_score,
                "crime_score": submarket2.crime_score,
                "growth_score": submarket2.growth_score,
                "price_score": submarket2.price_score
            },
            "winner": submarket1.city if submarket1.composite_score > submarket2.composite_score else submarket2.city,
            "score_difference": abs(submarket1.composite_score - submarket2.composite_score)
        }
        
        return comparison

