"""
PostgreSQL database models with PostGIS for geospatial data.
SQLAlchemy ORM models for production system.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from geoalchemy2 import Geometry
from datetime import datetime
import enum

from config.settings import settings

Base = declarative_base()


# Enums
class ZoningTypeEnum(str, enum.Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    MIXED_USE = "mixed_use"
    AGRICULTURAL = "agricultural"
    INDUSTRIAL = "industrial"
    UNKNOWN = "unknown"


class ListingStatusEnum(str, enum.Enum):
    ACTIVE = "active"
    PENDING = "pending"
    SOLD = "sold"
    OFF_MARKET = "off_market"
    EXPIRED = "expired"


# Models

class Submarket(Base):
    """Submarket analysis data."""
    __tablename__ = "submarkets"
    
    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, nullable=False, index=True)
    county = Column(String, nullable=False, index=True)
    state = Column(String(2), nullable=False, default="NC")
    zip_code = Column(String(10))
    
    # Scores
    composite_score = Column(Float, nullable=False)
    school_score = Column(Float)
    crime_score = Column(Float)
    growth_score = Column(Float)
    price_score = Column(Float)
    
    # Metadata
    data_completeness = Column(Float, default=1.0)
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    schools = relationship("School", back_populates="submarket")
    land_listings = relationship("LandListing", back_populates="submarket")
    
    def __repr__(self):
        return f"<Submarket {self.city}, {self.county}: {self.composite_score:.3f}>"


class School(Base):
    """School data."""
    __tablename__ = "schools"
    
    id = Column(Integer, primary_key=True, index=True)
    submarket_id = Column(Integer, ForeignKey("submarkets.id"))
    
    name = Column(String, nullable=False)
    rating = Column(Float)  # 1-10 scale
    test_scores = Column(Float)
    student_teacher_ratio = Column(Float)
    grade_levels = Column(String)
    school_type = Column(String)  # elementary, middle, high
    distance_miles = Column(Float)
    
    # Geospatial
    location = Column(Geometry('POINT', srid=4326))
    
    submarket = relationship("Submarket", back_populates="schools")


class CrimeStat(Base):
    """Crime statistics."""
    __tablename__ = "crime_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, nullable=False, index=True)
    county = Column(String, nullable=False)
    state = Column(String(2), nullable=False)
    
    violent_crime_rate = Column(Float)  # per 1000 residents
    property_crime_rate = Column(Float)
    total_crime_rate = Column(Float)
    year = Column(Integer)
    trend = Column(String)  # increasing, decreasing, stable
    
    updated_at = Column(DateTime, default=datetime.utcnow)


class EconomicData(Base):
    """Economic and growth metrics."""
    __tablename__ = "economic_data"
    
    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, nullable=False, index=True)
    county = Column(String, nullable=False)
    state = Column(String(2), nullable=False)
    
    population_growth_1yr = Column(Float)
    population_growth_3yr = Column(Float)
    population_growth_5yr = Column(Float)
    employment_growth = Column(Float)
    median_income = Column(Float)
    income_growth = Column(Float)
    
    year = Column(Integer)
    updated_at = Column(DateTime, default=datetime.utcnow)


class LandListing(Base):
    """Land/lot listings."""
    __tablename__ = "land_listings"
    
    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(String, unique=True, nullable=False, index=True)
    source = Column(String, nullable=False)  # zillow, realtor, etc.
    url = Column(Text)
    
    # Location
    address = Column(String)
    city = Column(String, nullable=False, index=True)
    county = Column(String, nullable=False)
    state = Column(String(2), nullable=False)
    zip_code = Column(String(10))
    
    # Geospatial
    location = Column(Geometry('POINT', srid=4326))
    
    # Property details
    price = Column(Float, nullable=False)
    acreage = Column(Float)
    zoning = Column(SQLEnum(ZoningTypeEnum), default=ZoningTypeEnum.UNKNOWN)
    utilities_available = Column(JSON)  # List of utilities
    
    # Listing details
    status = Column(SQLEnum(ListingStatusEnum), default=ListingStatusEnum.ACTIVE)
    listing_date = Column(DateTime)
    days_on_market = Column(Integer)
    seller_type = Column(String)
    description = Column(Text)
    
    # Analysis
    opportunity_score = Column(Float)
    estimated_roi = Column(Float)
    notes = Column(Text)
    
    # Tracking
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    price_history = Column(JSON)  # List of price changes
    
    # Relationships
    submarket_id = Column(Integer, ForeignKey("submarkets.id"))
    submarket = relationship("Submarket", back_populates="land_listings")


class SaleRecord(Base):
    """Home sales records."""
    __tablename__ = "sales_records"
    
    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)
    
    # Location
    address = Column(String)
    city = Column(String, nullable=False, index=True)
    county = Column(String, nullable=False)
    state = Column(String(2), nullable=False)
    zip_code = Column(String(10))
    
    # Sale details
    sale_price = Column(Float, nullable=False)
    list_price = Column(Float)
    sale_date = Column(DateTime, nullable=False, index=True)
    days_on_market = Column(Integer)
    price_per_sqft = Column(Float)
    
    # House features (stored as JSON for flexibility)
    features = Column(JSON)
    incentives = Column(JSON)  # List of incentives
    
    # Metadata
    collected_date = Column(DateTime, default=datetime.utcnow)


class Project(Base):
    """Development projects."""
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    
    # Location
    city = Column(String, nullable=False)
    county = Column(String, nullable=False)
    state = Column(String(2), default="NC")
    
    # Financials
    land_cost = Column(Float)
    construction_cost = Column(Float)
    carrying_costs = Column(Float)
    other_costs = Column(Float)
    total_investment = Column(Float)
    projected_sale_price = Column(Float)
    
    # Metrics
    irr = Column(Float)  # Internal Rate of Return
    roi = Column(Float)  # Return on Investment
    npv = Column(Float)  # Net Present Value
    gross_margin = Column(Float)
    
    # Timeline
    timeline_months = Column(Integer)
    start_date = Column(DateTime)
    projected_completion = Column(DateTime)
    
    # Status
    status = Column(String, default="planning")  # planning, in_progress, completed
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class QueryLog(Base):
    """AI query logs."""
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    answer = Column(Text)
    sources = Column(JSON)
    context_used = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)


# Database connection
def get_engine():
    """Get database engine."""
    return create_engine(
        settings.database_url,
        echo=settings.debug,
        pool_pre_ping=True
    )


def get_session():
    """Get database session."""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def init_db():
    """Initialize database (create all tables)."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")


if __name__ == "__main__":
    # Run this to initialize the database
    init_db()

