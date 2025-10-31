"""
Market analysis data collector.
Collects data from schools, crime statistics, growth metrics, and pricing sources.
"""

import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from config.settings import settings
from backend.models.schemas import (
    SchoolData, CrimeStats, GrowthMetrics, PricingData, SubmarketScore
)
from backend.utils.http_client import http_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchoolDataCollector:
    """Collect school data for submarkets."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.greatschools_api_key
        self.base_url = "https://api.greatschools.org/schools"
    
    def get_schools_by_zip(self, zip_code: str, state: str = "NC") -> List[SchoolData]:
        """Get schools for a given zip code."""
        if not self.api_key:
            logger.warning("No GreatSchools API key configured, returning mock data")
            return self._get_mock_school_data(zip_code)
        
        try:
            # GreatSchools API endpoint (example structure)
            params = {
                "key": self.api_key,
                "state": state,
                "zip": zip_code,
                "limit": 20
            }
            
            response = http_client.get(f"{self.base_url}/nearby", params=params)
            data = response.json()
            
            schools = []
            for school_data in data.get("schools", []):
                school = SchoolData(
                    name=school_data.get("name", ""),
                    rating=school_data.get("rating"),
                    test_scores=school_data.get("testScores"),
                    student_teacher_ratio=school_data.get("studentTeacherRatio"),
                    grade_levels=school_data.get("gradeLevels"),
                    distance_miles=school_data.get("distance"),
                    school_type=school_data.get("type")
                )
                schools.append(school)
            
            logger.info(f"Retrieved {len(schools)} schools for zip {zip_code}")
            return schools
            
        except Exception as e:
            logger.error(f"Error fetching school data for {zip_code}: {e}")
            return self._get_mock_school_data(zip_code)
    
    def _get_mock_school_data(self, zip_code: str) -> List[SchoolData]:
        """Return mock school data for development/testing."""
        return [
            SchoolData(
                name=f"Elementary School - {zip_code}",
                rating=7.5,
                test_scores=75.0,
                student_teacher_ratio=15.0,
                grade_levels="K-5",
                distance_miles=1.2,
                school_type="elementary"
            ),
            SchoolData(
                name=f"Middle School - {zip_code}",
                rating=7.0,
                test_scores=72.0,
                student_teacher_ratio=16.0,
                grade_levels="6-8",
                distance_miles=2.1,
                school_type="middle"
            ),
            SchoolData(
                name=f"High School - {zip_code}",
                rating=8.0,
                test_scores=80.0,
                student_teacher_ratio=17.0,
                grade_levels="9-12",
                distance_miles=3.5,
                school_type="high"
            )
        ]
    
    def calculate_school_score(self, schools: List[SchoolData]) -> float:
        """Calculate aggregate school score (0-1 scale)."""
        if not schools:
            return 0.0
        
        # Weight high schools more, then middle, then elementary
        weights = {"high": 0.4, "middle": 0.35, "elementary": 0.25}
        
        weighted_sum = 0
        total_weight = 0
        
        for school in schools:
            if school.rating:
                weight = weights.get(school.school_type, 0.33)
                weighted_sum += (school.rating / 10.0) * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class CrimeDataCollector:
    """Collect crime statistics from FBI and local sources."""
    
    def __init__(self):
        self.fbi_api_base = "https://api.usa.gov/crime/fbi/sapi"
    
    def get_crime_stats(self, city: str, state: str = "NC") -> Optional[CrimeStats]:
        """Get crime statistics for a city."""
        try:
            # FBI Crime Data API (example structure)
            # Note: Actual API structure may vary
            logger.info(f"Fetching crime data for {city}, {state}")
            
            # For now, return mock data as FBI API requires specific setup
            return self._get_mock_crime_data(city)
            
        except Exception as e:
            logger.error(f"Error fetching crime data for {city}: {e}")
            return self._get_mock_crime_data(city)
    
    def _get_mock_crime_data(self, city: str) -> CrimeStats:
        """Return mock crime data for development/testing."""
        # Mock data with reasonable ranges
        base_rate = 25.0  # Base rate per 1000 residents
        
        return CrimeStats(
            violent_crime_rate=base_rate * 0.3,
            property_crime_rate=base_rate * 0.7,
            total_crime_rate=base_rate,
            year=2023,
            trend="stable"
        )
    
    def calculate_crime_score(self, crime_stats: Optional[CrimeStats]) -> float:
        """Calculate crime score (0-1 scale, lower crime = higher score)."""
        if not crime_stats or not crime_stats.total_crime_rate:
            return 0.5  # Neutral if no data
        
        # National average is approximately 30 crimes per 1000 residents
        # Lower is better, so invert the scale
        national_avg = 30.0
        rate = crime_stats.total_crime_rate
        
        # Score inversely proportional to crime rate
        if rate <= 10:
            score = 1.0
        elif rate <= 20:
            score = 0.9
        elif rate <= national_avg:
            score = 0.7
        elif rate <= 40:
            score = 0.5
        elif rate <= 50:
            score = 0.3
        else:
            score = 0.1
        
        return score


class GrowthMetricsCollector:
    """Collect economic growth metrics from Census and BLS."""
    
    def __init__(self, api_key: str = None):
        self.census_api_key = api_key or settings.census_api_key
        self.census_base = "https://api.census.gov/data"
        self.bls_base = "https://api.bls.gov/publicAPI/v2"
    
    def get_growth_metrics(self, city: str, county: str, state: str = "NC") -> Optional[GrowthMetrics]:
        """Get growth metrics for an area."""
        try:
            logger.info(f"Fetching growth metrics for {city}, {county}, {state}")
            
            # For now, return mock data as Census API requires specific setup
            return self._get_mock_growth_data(city)
            
        except Exception as e:
            logger.error(f"Error fetching growth metrics for {city}: {e}")
            return self._get_mock_growth_data(city)
    
    def _get_mock_growth_data(self, city: str) -> GrowthMetrics:
        """Return mock growth data for development/testing."""
        return GrowthMetrics(
            population_growth_1yr=2.5,  # percentage
            population_growth_3yr=7.2,
            population_growth_5yr=11.5,
            employment_growth=3.1,
            median_income=65000.0,
            income_growth=2.8
        )
    
    def calculate_growth_score(self, metrics: Optional[GrowthMetrics]) -> float:
        """Calculate growth score (0-1 scale)."""
        if not metrics:
            return 0.5  # Neutral if no data
        
        # Weight recent growth more heavily
        growth_score = 0.0
        
        if metrics.population_growth_1yr:
            # 0-5% growth is good, >5% is excellent
            growth_score += min(metrics.population_growth_1yr / 5.0, 1.0) * 0.4
        
        if metrics.population_growth_3yr:
            growth_score += min((metrics.population_growth_3yr / 3) / 5.0, 1.0) * 0.3
        
        if metrics.employment_growth:
            growth_score += min(metrics.employment_growth / 4.0, 1.0) * 0.3
        
        return min(growth_score, 1.0)


class PricingDataCollector:
    """Collect pricing data from various sources."""
    
    def get_pricing_data(self, city: str, zip_code: str, state: str = "NC") -> Optional[PricingData]:
        """Get pricing data for an area."""
        try:
            logger.info(f"Fetching pricing data for {city}, {zip_code}, {state}")
            
            # This would normally scrape Zillow/Realtor or use their APIs
            # For now, return mock data
            return self._get_mock_pricing_data(city)
            
        except Exception as e:
            logger.error(f"Error fetching pricing data for {city}: {e}")
            return self._get_mock_pricing_data(city)
    
    def _get_mock_pricing_data(self, city: str) -> PricingData:
        """Return mock pricing data for development/testing."""
        median = 175.0  # dollars per sqft
        
        return PricingData(
            median_price_per_sqft=median,
            mean_price_per_sqft=median + 10,
            percentile_25=median - 25,
            percentile_75=median + 30,
            sample_size=50,
            last_updated=datetime.now()
        )
    
    def calculate_price_score(self, pricing: Optional[PricingData]) -> float:
        """Calculate price score (0-1 scale, considers affordability and premium potential)."""
        if not pricing:
            return 0.5
        
        # Sweet spot is around $150-200/sqft in NC
        # Too low might indicate poor market, too high reduces margins
        price = pricing.median_price_per_sqft
        
        if 150 <= price <= 200:
            score = 1.0
        elif 130 <= price < 150:
            score = 0.8
        elif 200 < price <= 220:
            score = 0.8
        elif 110 <= price < 130:
            score = 0.6
        elif 220 < price <= 250:
            score = 0.6
        else:
            score = 0.4
        
        return score


class MarketAnalysisCollector:
    """Main collector that orchestrates all market data collection."""
    
    def __init__(self):
        self.school_collector = SchoolDataCollector()
        self.crime_collector = CrimeDataCollector()
        self.growth_collector = GrowthMetricsCollector()
        self.pricing_collector = PricingDataCollector()
    
    def analyze_submarket(self, city: str, county: str, zip_code: Optional[str] = None, state: str = "NC") -> SubmarketScore:
        """Perform comprehensive submarket analysis."""
        logger.info(f"Analyzing submarket: {city}, {county}, {state}")
        
        # Collect all data
        schools = self.school_collector.get_schools_by_zip(zip_code) if zip_code else []
        crime_stats = self.crime_collector.get_crime_stats(city, state)
        growth_metrics = self.growth_collector.get_growth_metrics(city, county, state)
        pricing_data = self.pricing_collector.get_pricing_data(city, zip_code, state) if zip_code else None
        
        # Calculate scores
        school_score = self.school_collector.calculate_school_score(schools)
        crime_score = self.crime_collector.calculate_crime_score(crime_stats)
        growth_score = self.growth_collector.calculate_growth_score(growth_metrics)
        price_score = self.pricing_collector.calculate_price_score(pricing_data)
        
        # Calculate composite score using configurable weights
        composite_score = (
            settings.school_weight * school_score +
            settings.crime_weight * crime_score +
            settings.growth_weight * growth_score +
            settings.price_weight * price_score
        )
        
        # Create submarket score object
        submarket = SubmarketScore(
            zip_code=zip_code,
            city=city,
            county=county,
            state=state,
            school_score=school_score,
            crime_score=crime_score,
            growth_score=growth_score,
            price_score=price_score,
            composite_score=composite_score,
            schools=schools,
            crime_stats=crime_stats,
            growth_metrics=growth_metrics,
            pricing_data=pricing_data,
            last_updated=datetime.now()
        )
        
        logger.info(f"Submarket analysis complete. Composite score: {composite_score:.3f}")
        return submarket
    
    def analyze_multiple_submarkets(self, locations: List[Dict[str, str]]) -> List[SubmarketScore]:
        """Analyze multiple submarkets and return sorted by composite score."""
        results = []
        
        for loc in locations:
            try:
                submarket = self.analyze_submarket(
                    city=loc.get("city"),
                    county=loc.get("county"),
                    zip_code=loc.get("zip_code"),
                    state=loc.get("state", "NC")
                )
                results.append(submarket)
            except Exception as e:
                logger.error(f"Error analyzing {loc}: {e}")
                continue
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        return results

