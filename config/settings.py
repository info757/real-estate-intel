"""
Configuration management for Real Estate Intelligence Platform.
Loads settings from environment variables with defaults.
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI API
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    
    # Qdrant Vector Database
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: str = Field(default="", env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="real_estate_data", env="QDRANT_COLLECTION_NAME")
    qdrant_use_https: bool = Field(default=False, env="QDRANT_USE_HTTPS")
    
    # Data Source APIs
    attom_api_key: str = Field(default="", env="ATTOM_API_KEY")
    attom_api_base_url: str = Field(default="https://api.gateway.attomdata.com", env="ATTOM_API_BASE_URL")
    greatschools_api_key: str = Field(default="", env="GREATSCHOOLS_API_KEY")
    census_api_key: str = Field(default="", env="CENSUS_API_KEY")
    zillow_api_key: str = Field(default="", env="ZILLOW_API_KEY")
    realtor_api_key: str = Field(default="", env="REALTOR_API_KEY")
    
    # Database (Production)
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/real_estate_intel",
        env="DATABASE_URL"
    )
    postgres_user: str = Field(default="real_estate_user", env="POSTGRES_USER")
    postgres_password: str = Field(default="password", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="real_estate_intel", env="POSTGRES_DB")
    
    # Redis (Production)
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Application Settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Scraping Settings
    user_agent: str = Field(
        default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        env="USER_AGENT"
    )
    rate_limit_delay: float = Field(default=2.0, env="RATE_LIMIT_DELAY")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    
    # Analysis Weights (Configurable)
    school_weight: float = Field(default=0.30, env="SCHOOL_WEIGHT")
    crime_weight: float = Field(default=0.25, env="CRIME_WEIGHT")
    growth_weight: float = Field(default=0.25, env="GROWTH_WEIGHT")
    price_weight: float = Field(default=0.20, env="PRICE_WEIGHT")
    
    # Target Markets
    target_state: str = Field(default="NC", env="TARGET_STATE")
    target_counties: str = Field(
        default="Wake,Durham,Mecklenburg,Forsyth,Guilford",
        env="TARGET_COUNTIES"
    )
    
    # Financial Parameters
    default_construction_cost_per_sqft: float = Field(default=150.0, env="DEFAULT_CONSTRUCTION_COST_PER_SQFT")
    default_carrying_cost_monthly: float = Field(default=500.0, env="DEFAULT_CARRYING_COST_MONTHLY")
    default_build_time_months: int = Field(default=6, env="DEFAULT_BUILD_TIME_MONTHS")
    default_sale_time_months: int = Field(default=2, env="DEFAULT_SALE_TIME_MONTHS")
    default_discount_rate: float = Field(default=0.12, env="DEFAULT_DISCOUNT_RATE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_target_counties_list(self) -> List[str]:
        """Parse target counties from comma-separated string."""
        return [c.strip() for c in self.target_counties.split(",")]
    
    def validate_weights(self) -> bool:
        """Ensure analysis weights sum to approximately 1.0"""
        total = self.school_weight + self.crime_weight + self.growth_weight + self.price_weight
        return abs(total - 1.0) < 0.01
    
    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = self.school_weight + self.crime_weight + self.growth_weight + self.price_weight
        if total > 0:
            self.school_weight /= total
            self.crime_weight /= total
            self.growth_weight /= total
            self.price_weight /= total


# Global settings instance
settings = Settings()

# Normalize weights on initialization
if not settings.validate_weights():
    settings.normalize_weights()
