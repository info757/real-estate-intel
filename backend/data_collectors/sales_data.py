"""
Sales and product data collector.
Extracts features and incentives from recent home sales.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from bs4 import BeautifulSoup
from backend.models.schemas import (
    SaleRecord, HouseFeatures, ProductOptimization
)
from backend.utils.http_client import http_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from listing descriptions using NLP."""
    
    # Common features to look for
    FLOORING_KEYWORDS = ["hardwood", "carpet", "tile", "laminate", "vinyl", "bamboo"]
    COUNTERTOP_KEYWORDS = ["granite", "quartz", "marble", "butcher block", "concrete"]
    APPLIANCE_KEYWORDS = ["stainless steel", "professional grade", "smart appliances"]
    FEATURE_KEYWORDS = [
        "smart home", "energy efficient", "solar panels", "tankless water heater",
        "crown molding", "fireplace", "walk-in closet", "cathedral ceilings",
        "open floor plan", "gourmet kitchen", "master suite", "finished basement"
    ]
    INCENTIVE_KEYWORDS = [
        "closing cost assistance", "rate buydown", "free upgrades",
        "price reduction", "seller concessions", "incentives available"
    ]
    
    def extract_features(self, description: str) -> HouseFeatures:
        """Extract features from description text."""
        if not description:
            return HouseFeatures()
        
        desc_lower = description.lower()
        
        # Extract flooring types
        flooring = [f for f in self.FLOORING_KEYWORDS if f in desc_lower]
        
        # Extract countertops
        countertops = next((c for c in self.COUNTERTOP_KEYWORDS if c in desc_lower), None)
        
        # Extract appliances
        appliances = next((a for a in self.APPLIANCE_KEYWORDS if a in desc_lower), None)
        
        # Extract boolean features
        smart_home = "smart home" in desc_lower or "smart thermostat" in desc_lower
        energy_efficient = "energy efficient" in desc_lower or "energy star" in desc_lower
        
        # Extract additional features
        additional = [f for f in self.FEATURE_KEYWORDS if f in desc_lower]
        
        return HouseFeatures(
            flooring_types=flooring,
            countertops=countertops,
            appliances=appliances,
            smart_home=smart_home,
            energy_efficient=energy_efficient,
            additional_features=additional
        )
    
    def extract_incentives(self, description: str) -> List[str]:
        """Extract mentioned incentives from description."""
        if not description:
            return []
        
        desc_lower = description.lower()
        return [i for i in self.INCENTIVE_KEYWORDS if i in desc_lower]


class SalesDataCollector:
    """Collect recent home sales data."""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
    
    def get_recent_sales(self, city: str, state: str = "NC", months_back: int = 6) -> List[SaleRecord]:
        """Get recent home sales for a city."""
        logger.info(f"Fetching recent sales for {city}, {state} (last {months_back} months)")
        
        # In a real implementation, this would scrape MLS or use Zillow/Realtor APIs
        # For now, return mock data
        return self._get_mock_sales_data(city, state, months_back)
    
    def _get_mock_sales_data(self, city: str, state: str, months_back: int) -> List[SaleRecord]:
        """Generate mock sales data for development."""
        sales = []
        base_date = datetime.now()
        
        # Generate mock sales
        configurations = [
            {"beds": 3, "baths": 2.0, "sqft": 1800, "price": 325000},
            {"beds": 4, "baths": 2.5, "sqft": 2200, "price": 395000},
            {"beds": 3, "baths": 2.0, "sqft": 1650, "price": 310000},
            {"beds": 4, "baths": 3.0, "sqft": 2500, "price": 450000},
            {"beds": 3, "baths": 2.5, "sqft": 2000, "price": 365000},
        ]
        
        descriptions = [
            "Beautiful home with hardwood floors, granite countertops, and stainless steel appliances. Smart home features throughout.",
            "Spacious open floor plan with tile flooring, quartz countertops. Energy efficient with tankless water heater.",
            "Charming property with crown molding, fireplace, and walk-in closets. Recent price reduction!",
            "Modern home with smart home system, solar panels, and gourmet kitchen. Closing cost assistance available.",
            "Elegant design with cathedral ceilings, master suite, and finished basement. Rate buydown incentive."
        ]
        
        for i, config in enumerate(configurations):
            days_ago = (i * 30) % (months_back * 30)
            sale_date = base_date - timedelta(days=days_ago)
            
            description = descriptions[i % len(descriptions)]
            features = self.feature_extractor.extract_features(description)
            features.bedrooms = config["beds"]
            features.bathrooms = config["baths"]
            features.sqft = config["sqft"]
            features.garage = "2-car" if config["beds"] >= 4 else "1-car"
            features.lot_size = 0.25
            
            incentives = self.feature_extractor.extract_incentives(description)
            
            sale = SaleRecord(
                listing_id=f"sale_{city}_{i}",
                source="mls",
                address=f"{100 + i} Main St",
                city=city,
                county="",
                state=state,
                zip_code="27601",
                sale_price=config["price"],
                list_price=config["price"] + 5000,
                sale_date=sale_date,
                days_on_market=30 + (i * 5),
                price_per_sqft=config["price"] / config["sqft"],
                features=features,
                incentives=incentives,
                collected_date=datetime.now()
            )
            sales.append(sale)
        
        logger.info(f"Retrieved {len(sales)} recent sales")
        return sales
    
    def analyze_sales_for_product_optimization(self, sales: List[SaleRecord]) -> Dict[str, Any]:
        """Analyze sales data to determine optimal product configuration."""
        if not sales:
            return {}
        
        # Analyze square footage
        sqfts = [s.features.sqft for s in sales if s.features.sqft]
        bedrooms = [s.features.bedrooms for s in sales if s.features.bedrooms]
        bathrooms = [s.features.bathrooms for s in sales if s.features.bathrooms]
        days_on_market = [s.days_on_market for s in sales if s.days_on_market]
        
        # Feature frequency analysis
        feature_counts = {}
        for sale in sales:
            for feature in sale.features.additional_features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Incentive effectiveness
        incentive_dom = {}  # Average days on market by incentive
        for sale in sales:
            for incentive in sale.incentives:
                if incentive not in incentive_dom:
                    incentive_dom[incentive] = []
                if sale.days_on_market:
                    incentive_dom[incentive].append(sale.days_on_market)
        
        return {
            "sqft_range": (min(sqfts) if sqfts else 0, max(sqfts) if sqfts else 0),
            "avg_sqft": sum(sqfts) / len(sqfts) if sqfts else 0,
            "common_bedrooms": max(set(bedrooms), key=bedrooms.count) if bedrooms else 0,
            "common_bathrooms": max(set(bathrooms), key=bathrooms.count) if bathrooms else 0,
            "avg_days_on_market": sum(days_on_market) / len(days_on_market) if days_on_market else 0,
            "feature_frequency": feature_counts,
            "incentive_avg_dom": {k: sum(v) / len(v) for k, v in incentive_dom.items() if v}
        }


class ProductOptimizationAnalyzer:
    """Analyze sales data to recommend optimal product configurations."""
    
    def __init__(self):
        self.sales_collector = SalesDataCollector()
    
    def analyze_submarket(self, city: str, county: str, state: str = "NC") -> ProductOptimization:
        """Analyze a submarket to determine optimal product."""
        logger.info(f"Analyzing product optimization for {city}, {county}")
        
        # Get recent sales
        sales = self.sales_collector.get_recent_sales(city, state)
        analysis = self.sales_collector.analyze_sales_for_product_optimization(sales)
        
        if not analysis:
            logger.warning("No sales data available")
            return ProductOptimization(
                city=city,
                county=county,
                optimal_sqft_min=1800,
                optimal_sqft_max=2200,
                optimal_bedrooms=3,
                optimal_bathrooms=2.0,
                sample_size=0
            )
        
        # Calculate optimal specs
        avg_sqft = analysis["avg_sqft"]
        optimal_sqft_min = int(avg_sqft * 0.9)
        optimal_sqft_max = int(avg_sqft * 1.1)
        
        # Feature recommendations
        recommended_features = []
        total_sales = len(sales)
        for feature, count in analysis["feature_frequency"].items():
            frequency = count / total_sales
            if frequency >= 0.3:  # Feature appears in 30%+ of sales
                recommended_features.append({
                    "feature": feature,
                    "frequency": frequency,
                    "usage_recommended": frequency >= 0.5
                })
        
        # Sort by frequency
        recommended_features.sort(key=lambda x: x["frequency"], reverse=True)
        
        # Incentive effectiveness
        avg_dom = analysis["avg_days_on_market"]
        effective_incentives = []
        for incentive, dom in analysis["incentive_avg_dom"].items():
            reduction = avg_dom - dom
            effective_incentives.append({
                "incentive": incentive,
                "days_on_market_reduction": int(reduction),
                "usage_recommended": reduction > 5
            })
        
        # Sort by effectiveness
        effective_incentives.sort(key=lambda x: x["days_on_market_reduction"], reverse=True)
        
        # Calculate optimal price range (based on price per sqft)
        if sales:
            price_per_sqft = [s.price_per_sqft for s in sales if s.price_per_sqft]
            if price_per_sqft:
                avg_price_per_sqft = sum(price_per_sqft) / len(price_per_sqft)
                optimal_price_min = int(optimal_sqft_min * avg_price_per_sqft)
                optimal_price_max = int(optimal_sqft_max * avg_price_per_sqft)
            else:
                optimal_price_min = None
                optimal_price_max = None
        else:
            optimal_price_min = None
            optimal_price_max = None
        
        return ProductOptimization(
            city=city,
            county=county,
            optimal_sqft_min=optimal_sqft_min,
            optimal_sqft_max=optimal_sqft_max,
            optimal_bedrooms=int(analysis["common_bedrooms"]),
            optimal_bathrooms=analysis["common_bathrooms"],
            recommended_features=recommended_features,
            effective_incentives=effective_incentives,
            avg_days_on_market=avg_dom,
            optimal_price_range_min=optimal_price_min,
            optimal_price_range_max=optimal_price_max,
            sample_size=total_sales,
            last_updated=datetime.now()
        )

