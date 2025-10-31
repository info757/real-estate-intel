"""
FastAPI backend for Real Estate Intelligence Platform (Phase 2 - Production).
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
import logging

from backend.models.schemas import (
    SubmarketScore, LandListing, SaleRecord, ProductOptimization,
    ProjectFinancials, QueryResult
)
from backend.analyzers.submarket_ranker import SubmarketRanker
from backend.data_collectors.land_scraper import LandScraperOrchestrator
from backend.data_collectors.sales_data import ProductOptimizationAnalyzer
from backend.analyzers.land_analyzer import LandOpportunityAnalyzer
from backend.analyzers.financial_optimizer import FinancialOptimizer
from backend.ai_engine.rag_system import QdrantRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Real Estate Intelligence Platform API",
    description="AI-powered market analysis and development optimization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
submarket_ranker = SubmarketRanker()
land_scraper = LandScraperOrchestrator()
product_analyzer = ProductOptimizationAnalyzer()
land_analyzer = LandOpportunityAnalyzer()
financial_optimizer = FinancialOptimizer()
rag_system = QdrantRAGSystem()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Real Estate Intelligence Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# Market Analysis Endpoints

@app.post("/api/markets/analyze", response_model=List[SubmarketScore])
async def analyze_markets(locations: List[dict]):
    """Analyze multiple submarkets."""
    try:
        results = submarket_ranker.rank_submarkets(locations)
        return results
    except Exception as e:
        logger.error(f"Error analyzing markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/markets/top", response_model=List[SubmarketScore])
async def get_top_markets(limit: int = Query(10, ge=1, le=50)):
    """Get top N submarkets."""
    try:
        # This would query from database in production
        # For now, return from submarket_ranker
        return []
    except Exception as e:
        logger.error(f"Error getting top markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Land Listing Endpoints

@app.post("/api/land/search", response_model=List[LandListing])
async def search_land(
    cities: List[str],
    max_price: Optional[float] = None,
    min_acreage: Optional[float] = None,
    max_acreage: Optional[float] = None
):
    """Search for land listings."""
    try:
        locations = [{"city": city, "state": "NC", "max_price": max_price} for city in cities]
        listings = land_scraper.scrape_multiple_locations(locations)
        
        # Apply filters
        if min_acreage:
            listings = [l for l in listings if l.acreage and l.acreage >= min_acreage]
        if max_acreage:
            listings = [l for l in listings if l.acreage and l.acreage <= max_acreage]
        
        return listings
    except Exception as e:
        logger.error(f"Error searching land: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/land/{listing_id}", response_model=LandListing)
async def get_land_listing(listing_id: str):
    """Get specific land listing."""
    try:
        # In production, query from database
        listings = land_scraper.get_all_listings()
        listing = next((l for l in listings if l.listing_id == listing_id), None)
        
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        return listing
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting listing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/land/opportunities")
async def analyze_land_opportunities(listings: List[LandListing], submarkets: Optional[List[SubmarketScore]] = None):
    """Analyze land opportunities with scoring."""
    try:
        opportunities = land_analyzer.rank_land_opportunities(listings, submarkets)
        return opportunities
    except Exception as e:
        logger.error(f"Error analyzing opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Product Intelligence Endpoints

@app.post("/api/products/optimize", response_model=ProductOptimization)
async def optimize_product(city: str, county: str, state: str = "NC"):
    """Get optimal product configuration for a submarket."""
    try:
        result = product_analyzer.analyze_submarket(city, county, state)
        return result
    except Exception as e:
        logger.error(f"Error optimizing product: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Financial Modeling Endpoints

@app.post("/api/financial/analyze", response_model=ProjectFinancials)
async def analyze_project_financials(
    land_cost: float,
    construction_cost: float,
    carrying_costs: float,
    other_costs: float,
    projected_sale_price: float,
    timeline_months: int,
    discount_rate: float = 0.12
):
    """Analyze project financials."""
    try:
        result = financial_optimizer.analyze_project(
            land_cost=land_cost,
            construction_cost=construction_cost,
            carrying_costs=carrying_costs,
            other_costs=other_costs,
            projected_sale_price=projected_sale_price,
            timeline_months=timeline_months,
            discount_rate=discount_rate
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing financials: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/financial/sensitivity")
async def sensitivity_analysis(financials: ProjectFinancials):
    """Perform sensitivity analysis."""
    try:
        result = financial_optimizer.sensitivity_analysis(financials)
        return result
    except Exception as e:
        logger.error(f"Error in sensitivity analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# AI Query Endpoints

@app.post("/api/chat/query")
async def query_ai(question: str):
    """Query the AI assistant."""
    try:
        result = rag_system.query(question)
        return result
    except Exception as e:
        logger.error(f"Error querying AI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/index")
async def index_data(
    submarkets: Optional[List[SubmarketScore]] = None,
    land_listings: Optional[List[LandListing]] = None,
    products: Optional[List[ProductOptimization]] = None
):
    """Index data into vector database."""
    try:
        rag_system.bulk_index_data(submarkets, land_listings, products)
        return {"status": "success", "message": "Data indexed successfully"}
    except Exception as e:
        logger.error(f"Error indexing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

