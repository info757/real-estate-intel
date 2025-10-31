"""
Data models and schemas for the Real Estate Intelligence Platform.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ZoningType(str, Enum):
    """Common zoning types."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    MIXED_USE = "mixed_use"
    AGRICULTURAL = "agricultural"
    INDUSTRIAL = "industrial"
    UNKNOWN = "unknown"


class ListingStatus(str, Enum):
    """Status of a listing."""
    ACTIVE = "active"
    PENDING = "pending"
    SOLD = "sold"
    OFF_MARKET = "off_market"
    EXPIRED = "expired"


# Market Analysis Models

class SchoolData(BaseModel):
    """School information for a location."""
    name: str
    rating: Optional[float] = None  # 1-10 scale
    test_scores: Optional[float] = None
    student_teacher_ratio: Optional[float] = None
    grade_levels: Optional[str] = None
    distance_miles: Optional[float] = None
    school_type: Optional[str] = None  # elementary, middle, high


class CrimeStats(BaseModel):
    """Crime statistics for an area."""
    violent_crime_rate: Optional[float] = None  # per 1000 residents
    property_crime_rate: Optional[float] = None
    total_crime_rate: Optional[float] = None
    year: Optional[int] = None
    trend: Optional[str] = None  # increasing, decreasing, stable


class GrowthMetrics(BaseModel):
    """Economic growth metrics for an area."""
    population_growth_1yr: Optional[float] = None
    population_growth_3yr: Optional[float] = None
    population_growth_5yr: Optional[float] = None
    employment_growth: Optional[float] = None
    median_income: Optional[float] = None
    income_growth: Optional[float] = None


class PricingData(BaseModel):
    """Pricing information for a submarket."""
    median_price_per_sqft: float
    mean_price_per_sqft: float
    percentile_25: Optional[float] = None
    percentile_75: Optional[float] = None
    sample_size: int = 0
    last_updated: datetime = Field(default_factory=datetime.now)


class SubmarketScore(BaseModel):
    """Comprehensive submarket analysis."""
    zip_code: Optional[str] = None
    city: str
    county: str
    state: str = "NC"
    
    # Scores (0-1 scale)
    school_score: float = 0.0
    crime_score: float = 0.0  # Lower is better
    growth_score: float = 0.0
    price_score: float = 0.0
    composite_score: float = 0.0
    
    # Raw data
    schools: List[SchoolData] = []
    crime_stats: Optional[CrimeStats] = None
    growth_metrics: Optional[GrowthMetrics] = None
    pricing_data: Optional[PricingData] = None
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)
    data_completeness: float = 1.0  # 0-1 scale


# Land Acquisition Models

class LandListing(BaseModel):
    """Land/lot listing information."""
    listing_id: str
    source: str  # zillow, realtor, landwatch, etc.
    url: str
    
    # Location
    address: Optional[str] = None
    city: str
    county: str
    state: str = "NC"
    zip_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Property details
    price: float
    acreage: Optional[float] = None
    zoning: ZoningType = ZoningType.UNKNOWN
    utilities_available: List[str] = []  # water, sewer, electric, gas
    
    # Listing details
    status: ListingStatus = ListingStatus.ACTIVE
    listing_date: Optional[datetime] = None
    days_on_market: Optional[int] = None
    seller_type: Optional[str] = None  # agent, owner, bank, etc.
    description: Optional[str] = None
    
    # Analysis
    opportunity_score: Optional[float] = None  # 0-1 scale
    estimated_roi: Optional[float] = None
    notes: Optional[str] = None
    
    # Tracking
    first_seen: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    price_history: List[Dict[str, Any]] = []


# Sales and Product Models

class HouseFeatures(BaseModel):
    """Features of a sold home."""
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    sqft: Optional[int] = None
    lot_size: Optional[float] = None  # acres
    year_built: Optional[int] = None
    stories: Optional[int] = None
    garage: Optional[str] = None  # "1-car", "2-car", "3-car", None
    basement: Optional[bool] = None
    pool: Optional[bool] = None
    
    # Finishes
    flooring_types: List[str] = []  # hardwood, carpet, tile, etc.
    countertops: Optional[str] = None  # granite, quartz, marble, etc.
    appliances: Optional[str] = None  # stainless steel, etc.
    smart_home: bool = False
    energy_efficient: bool = False
    
    # Extracted from description
    additional_features: List[str] = []


class SaleRecord(BaseModel):
    """Record of a home sale."""
    listing_id: str
    source: str
    
    # Location
    address: Optional[str] = None
    city: str
    county: str
    state: str = "NC"
    zip_code: Optional[str] = None
    
    # Sale details
    sale_price: float
    list_price: Optional[float] = None
    sale_date: datetime
    days_on_market: Optional[int] = None
    price_per_sqft: Optional[float] = None
    
    # House features
    features: HouseFeatures
    
    # Incentives mentioned
    incentives: List[str] = []  # closing cost assistance, rate buydown, etc.
    
    # Metadata
    collected_date: datetime = Field(default_factory=datetime.now)


class ProductOptimization(BaseModel):
    """Optimal product configuration for a submarket."""
    city: str
    county: str
    
    # Optimal specs
    optimal_sqft_min: int
    optimal_sqft_max: int
    optimal_bedrooms: int
    optimal_bathrooms: float
    
    # Feature recommendations
    recommended_features: List[Dict[str, Any]] = []
    # Each dict: {feature: str, frequency: float, avg_price_premium: float, roi_score: float}
    
    # Incentive effectiveness
    effective_incentives: List[Dict[str, Any]] = []
    # Each dict: {incentive: str, frequency: float, days_on_market_reduction: int, usage_recommended: bool}
    
    # Market insights
    avg_days_on_market: Optional[float] = None
    optimal_price_range_min: Optional[float] = None
    optimal_price_range_max: Optional[float] = None
    
    sample_size: int = 0
    last_updated: datetime = Field(default_factory=datetime.now)


# Financial Models

class ProjectFinancials(BaseModel):
    """Financial analysis for a development project."""
    project_name: Optional[str] = None
    
    # Inputs
    land_cost: float
    construction_cost: float
    carrying_costs: float
    other_costs: float = 0.0
    total_investment: Optional[float] = None
    
    # Timeline (months)
    acquisition_time: int = 1
    build_time: int = 6
    sale_time: int = 2
    total_timeline: Optional[int] = None
    
    # Returns
    projected_sale_price: float
    gross_profit: Optional[float] = None
    gross_margin: Optional[float] = None  # percentage
    
    # Metrics
    irr: Optional[float] = None  # Internal Rate of Return
    roi: Optional[float] = None  # Return on Investment
    discount_rate: float = 0.12
    npv: Optional[float] = None  # Net Present Value
    
    # Sensitivity
    break_even_price: Optional[float] = None
    margin_of_safety: Optional[float] = None  # percentage
    
    created_date: datetime = Field(default_factory=datetime.now)


class SGAMetrics(BaseModel):
    """SG&A and operational efficiency metrics."""
    period_start: datetime
    period_end: datetime
    
    # Costs
    land_acquisition_costs: float = 0.0
    staff_costs: float = 0.0
    technology_costs: float = 0.0
    other_sga: float = 0.0
    total_sga: Optional[float] = None
    
    # Efficiency
    projects_completed: int = 0
    sga_per_project: Optional[float] = None
    automation_savings: Optional[float] = None
    
    # Scaling metrics
    projects_per_staff: Optional[float] = None
    automation_percentage: float = 0.0  # 0-1 scale


# Chat/Query Models

class ChatMessage(BaseModel):
    """Chat message for AI query interface."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class QueryResult(BaseModel):
    """Result of an AI query."""
    query: str
    answer: str
    sources: List[str] = []
    confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)

