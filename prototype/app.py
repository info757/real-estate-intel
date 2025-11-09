"""
Streamlit prototype for Real Estate Intelligence Platform.
Rapid prototype for client demos with dashboard, analysis, and AI chat.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data_collectors.market_data import MarketAnalysisCollector
from backend.data_collectors.land_scraper import LandScraperOrchestrator
from backend.data_collectors.sales_data import ProductOptimizationAnalyzer
from backend.analyzers.submarket_ranker import SubmarketRanker
from backend.analyzers.land_analyzer import LandOpportunityAnalyzer
from backend.analyzers.financial_optimizer import FinancialOptimizer
from backend.ai_engine.rag_system import QdrantRAGSystem
from backend.analyzers.feature_analyzer import feature_analyzer
from backend.analyzers.demand_predictor import demand_predictor
from backend.data_collectors.safe_listings_scraper import safe_listings_scraper
from backend.analyzers.popularity_analyzer import popularity_analyzer
from backend.ml.guardrails import guardrails
from backend.ml.backtesting import backtester
from backend.ai_engine.narrative_generator import generate_recommendation_narrative
from backend.ml.pricing_model import PricingModel
from backend.ml.demand_model import DemandModel
from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.train_fast_seller_model import fetch_sold_listings_with_features
from config.settings import settings
from pathlib import Path

GLOBAL_CSS = """
<style>
:root {
    --bg: #0f172a;
    --card-bg: rgba(15, 23, 42, 0.72);
    --accent: #38bdf8;
    --accent-soft: rgba(56, 189, 248, 0.12);
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5f5;
    --radius: 18px;
}

.block-container {
    padding-top: 2.4rem;
    padding-bottom: 2.6rem;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.1rem 0;
    flex-wrap: wrap;
}

.metric-card {
    flex: 1 1 240px;
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 1.35rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.35);
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 25px 55px rgba(15, 23, 42, 0.45);
}

.metric-title {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
}

.metric-value {
    font-size: 1.95rem;
    font-weight: 700;
    color: var(--text-primary);
}

.metric-footnote {
    margin-top: 0.6rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.25rem;
    margin-top: 1.25rem;
}

.alt-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 1rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
}

.alt-card-title {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.35rem;
}

.alt-card-metric {
    display: flex;
    justify-content: space-between;
    color: var(--text-secondary);
    font-size: 0.9rem;
    padding: 0.2rem 0;
}

.inline-highlight {
    color: var(--accent);
    font-weight: 600;
}
</style>
"""


def render_metric_card(title: str, value: str, footnote: Optional[str] = None) -> None:
    foot_html = f'<div class="metric-footnote">{footnote}</div>' if footnote else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {foot_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_seasonality_report() -> pd.DataFrame:
    report_path = Path("reports/seasonality_adjusted_predictions.csv")
    if not report_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(report_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


@st.cache_data
def get_market_features(
    lat: float,
    lon: float,
    zip_code: Optional[str] = None,
    subdivision: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    df = load_seasonality_report()
    if df.empty or lat is None or lon is None:
        return None

    candidate_df = df.copy()
    if subdivision:
        candidates = candidate_df[
            candidate_df["subdivision"].fillna("").str.lower() == subdivision.lower()
        ]
        if not candidates.empty:
            candidate_df = candidates
    if zip_code:
        candidates = candidate_df[
            candidate_df["zip_code"].astype(str) == str(zip_code)
        ]
        if not candidates.empty:
            candidate_df = candidates

    if candidate_df.empty:
        return None

    distances = np.sqrt(
        (candidate_df["latitude"] - lat) ** 2 +
        (candidate_df["longitude"] - lon) ** 2
    )
    if distances.empty:
        return None
    nearest_idx = distances.idxmin()
    nearest_distance = distances.loc[nearest_idx]
    if not np.isfinite(nearest_distance) or nearest_distance > 0.02:
        return None
    return candidate_df.loc[nearest_idx].to_dict()


@st.cache_resource(show_spinner=False)
def load_tri_models() -> tuple[PricingModel, DemandModel]:
    pricing = PricingModel(model_dir="models/triad_latest")
    pricing.load(model_name="triad_pricing_model")
    demand = DemandModel(model_dir="models/triad_latest")
    demand.load(model_name="triad_demand_model")
    return pricing, demand


@st.cache_resource(show_spinner=False)
def get_feature_engineer() -> FeatureEngineer:
    return FeatureEngineer()


@st.cache_data(show_spinner=False, ttl=600)
def load_listings_for_zip(zip_code: str):
    return fetch_sold_listings_with_features(
        zip_codes=[str(zip_code)],
        days_back=730,
        max_per_zip=None,
        use_cache=True,
    )


def _extract_listing_coords(listing: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    lat = listing.get("latitude") or (listing.get("summary") or {}).get("latitude")
    lon = listing.get("longitude") or (listing.get("summary") or {}).get("longitude")
    if lat is None:
        lat = (listing.get("geocode") or {}).get("lat")
    if lon is None:
        geocode = listing.get("geocode") or {}
        lon = geocode.get("lon") or geocode.get("lng")
    return safe_float(lat), safe_float(lon)


def compute_tri_model_predictions(zip_code: str, primary_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not zip_code:
        return None

    try:
        listings = load_listings_for_zip(str(zip_code))
    except Exception as exc:
        st.warning(f"Triad model lookup failed for ZIP {zip_code}: {exc}")
        return None

    if not listings:
        return None

    target = None
    str_pid = None
    if primary_row is not None:
        pid_value = (
            primary_row.get("property_id")
            or primary_row.get("propertyId")
            or primary_row.get("id")
        )
        if pid_value is not None:
            str_pid = str(pid_value)

    if str_pid:
        for listing in listings:
            candidate_id = (
                listing.get("property_id")
                or (listing.get("summary") or {}).get("propertyId")
                or (listing.get("summary") or {}).get("id")
            )
            if candidate_id is not None and str(candidate_id) == str_pid:
                target = listing
                break

    if target is None and primary_row is not None:
        primary_lat = safe_float(primary_row.get("latitude"))
        primary_lon = safe_float(primary_row.get("longitude"))
        if primary_lat is not None and primary_lon is not None:
            best_listing = None
            best_distance = None
            for listing in listings:
                lat, lon = _extract_listing_coords(listing)
                if lat is None or lon is None:
                    continue
                distance = float(np.hypot(lat - primary_lat, lon - primary_lon))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_listing = listing
            if best_listing is not None:
                target = best_listing

    if target is None:
        target = listings[0]

    pricing_model, demand_model = load_tri_models()
    engineer = get_feature_engineer()

    try:
        features_df = engineer.engineer_features([target])
    except Exception as exc:
        st.warning(f"Feature engineering failed for ZIP {zip_code}: {exc}")
        return None

    pricing_features = features_df.reindex(columns=pricing_model.feature_names, fill_value=0.0)
    pricing_features = pricing_features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    demand_features = features_df.reindex(columns=demand_model.feature_names, fill_value=0.0)
    demand_features = demand_features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # ensure DOM-related columns stay meaningful
    for dom_col in ["dom_to_pending", "dom_to_sold"]:
        if dom_col in features_df.columns:
            demand_features[dom_col] = features_df[dom_col]

    try:
        price_pred, price_lower, price_upper = pricing_model.predict(
            pricing_features,
            return_intervals=True,
        )
        demand_preds = demand_model.predict(demand_features)
    except Exception as exc:
        st.warning(f"Triad model prediction failed: {exc}")
        return None

    return {
        "predicted_price": float(price_pred[0]),
        "price_lower": float(price_lower[0]),
        "price_upper": float(price_upper[0]),
        "sell_probability": float(np.clip(demand_preds["sell_probability"][0], 0.0, 1.0)),
        "expected_dom": float(max(demand_preds["expected_dom"][0], 0.0)),
        "feature_snapshot": features_df.to_dict(orient="records")[0],
    }


DEMO_LOCATIONS = [
    {
        "label": "New Irving Park • Greensboro 27408 (High-end SFR)",
        "zip_code": "27408",
        "latitude": 36.1010667665,
        "longitude": -79.8401642095,
        "subdivision": "New Irving Park",
        "notes": "4BR/3BA executive homes in New Irving Park. Fast-sale prob ≈94%, DOM ≈40 days.",
    },
    {
        "label": "Wyngate Village • Winston-Salem 27103 (Move-up townhome)",
        "zip_code": "27103",
        "latitude": 36.0642284927,
        "longitude": -80.3442158592,
        "subdivision": "Wyngate Village",
        "notes": "3BR townhomes around Hanes Mall. Fast-sale prob ≈86%, DOM ≈40 days.",
    },
    {
        "label": "Dogwood Acres • Asheboro 27205 (Entry-level SFR)",
        "zip_code": "27205",
        "latitude": 35.7200937986,
        "longitude": -79.8331894687,
        "subdivision": "Dogwood Acres",
        "notes": "3BR starter homes. Fast-sale prob ≈88%, DOM ≈47 days, minimal relists.",
    },
]

# Page config
st.set_page_config(
    page_title="Real Estate Intelligence Platform",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .score-high {
        color: #28a745;
        font-weight: bold;
    }
    .score-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .score-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'submarkets' not in st.session_state:
    st.session_state.submarkets = None
if 'land_listings' not in st.session_state:
    st.session_state.land_listings = None
if 'products' not in st.session_state:
    st.session_state.products = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def get_score_color_class(score):
    """Get CSS class for score coloring."""
    if score >= 0.7:
        return "score-high"
    elif score >= 0.5:
        return "score-medium"
    else:
        return "score-low"


def format_currency(value):
    """Format value as currency."""
    return f"${value:,.0f}"


def safe_float(value) -> Optional[float]:
    """Convert to float when possible, returning None for invalid numbers."""
    if value is None:
        return None
    try:
        result = float(value)
        if np.isnan(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def main():
    """Main application."""
    
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown('<div class="main-header">🏗️ BuildOptima</div>', unsafe_allow_html=True)
    st.markdown("*What to build, where to build it, and how fast it will move*")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏘️ RE Intel")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📊 Market Analysis", "🎯 Micro-Market Analysis",
             "🏞️ Land Opportunities", "🏗️ Product Intelligence", 
             "💰 Financial Modeling", "🧠 ML Recommendations", "🔥 Listing Popularity", "🤖 AI Assistant"]
        )
        
        st.markdown("---")
        st.markdown("### Settings")
        st.markdown(f"**Target State:** {settings.target_state}")
        st.markdown(f"**Counties:** {', '.join(settings.get_target_counties_list())}")
        
        st.markdown("---")
        st.markdown("### Analysis Weights")
        st.progress(settings.school_weight, text=f"Schools: {settings.school_weight*100:.0f}%")
        st.progress(settings.crime_weight, text=f"Crime: {settings.crime_weight*100:.0f}%")
        st.progress(settings.growth_weight, text=f"Growth: {settings.growth_weight*100:.0f}%")
        st.progress(settings.price_weight, text=f"Pricing: {settings.price_weight*100:.0f}%")
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Market Analysis":
        show_market_analysis()
    elif page == "🎯 Micro-Market Analysis":
        show_micro_market_analysis()
    elif page == "🏞️ Land Opportunities":
        show_land_opportunities()
    elif page == "🏗️ Product Intelligence":
        show_product_intelligence()
    elif page == "💰 Financial Modeling":
        show_financial_modeling()
    elif page == "🧠 ML Recommendations":
        show_ml_recommendations()
    elif page == "🔥 Listing Popularity":
        show_listing_popularity()
    elif page == "🤖 AI Assistant":
        show_ai_assistant()


def show_dashboard():
    """Show main dashboard."""
    st.header("Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Submarkets Analyzed", "15", "↑ 3 this week")
    with col2:
        st.metric("Land Opportunities", "47", "↑ 8 new")
    with col3:
        st.metric("Avg. Opportunity Score", "0.72", "↑ 0.05")
    with col4:
        st.metric("Projected Avg. IRR", "18.5%", "↑ 2.3%")
    
    st.markdown("---")
    
    # Quick insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 5 Submarkets")
        
        # Mock data for demo
        top_markets = pd.DataFrame({
            "City": ["Cary", "Apex", "Holly Springs", "Morrisville", "Wake Forest"],
            "County": ["Wake", "Wake", "Wake", "Wake", "Wake"],
            "Score": [0.85, 0.82, 0.79, 0.76, 0.74],
            "Price/SqFt": [185, 178, 172, 180, 165]
        })
        
        fig = px.bar(top_markets, x="Score", y="City", orientation='h',
                    color="Score", color_continuous_scale="blues",
                    title="Composite Scores")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Land Price Distribution")
        
        # Mock data
        price_data = pd.DataFrame({
            "Price Range": ["$50k-$75k", "$75k-$100k", "$100k-$150k", "$150k+"],
            "Count": [12, 18, 14, 3]
        })
        
        fig = px.pie(price_data, values="Count", names="Price Range",
                    title="Available Land by Price")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    activity_data = pd.DataFrame({
        "Date": ["2025-10-31", "2025-10-30", "2025-10-29", "2025-10-28"],
        "Activity": [
            "New land listing in Apex - $82k",
            "Submarket analysis updated: Cary",
            "8 new listings scraped from Zillow",
            "Product optimization completed: Holly Springs"
        ],
        "Type": ["Land", "Analysis", "Data", "Analysis"]
    })
    
    st.dataframe(activity_data, use_container_width=True, hide_index=True)


def show_market_analysis():
    """Show market analysis page."""
    st.header("📊 Submarket Analysis")
    
    st.markdown("Analyze submarkets based on schools, crime, growth, and pricing.")
    
    # Input section
    with st.expander("🔍 Analyze New Submarkets", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cities_input = st.text_area(
                "Cities (one per line)",
                "Cary\nApex\nHolly Springs\nMorrisville\nWake Forest",
                height=150
            )
        
        with col2:
            counties_input = st.text_area(
                "Counties (one per line)",
                "Wake\nWake\nWake\nWake\nWake",
                height=150
            )
        
        with col3:
            zip_codes_input = st.text_area(
                "Zip Codes (optional, one per line)",
                "27519\n27502\n27540\n27560\n27587",
                height=150
            )
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Analyzing submarkets..."):
                # Parse inputs
                cities = [c.strip() for c in cities_input.split('\n') if c.strip()]
                counties = [c.strip() for c in counties_input.split('\n') if c.strip()]
                zip_codes = [z.strip() for z in zip_codes_input.split('\n') if z.strip()]
                
                # Pad zip codes if needed
                while len(zip_codes) < len(cities):
                    zip_codes.append(None)
                
                # Build locations list
                locations = []
                for i, city in enumerate(cities):
                    locations.append({
                        "city": city,
                        "county": counties[i] if i < len(counties) else "Wake",
                        "zip_code": zip_codes[i] if i < len(zip_codes) else None,
                        "state": "NC"
                    })
                
                # Run analysis
                ranker = SubmarketRanker()
                st.session_state.submarkets = ranker.rank_submarkets(locations)
                
                st.success(f"✅ Analyzed {len(st.session_state.submarkets)} submarkets!")
    
    # Display results
    if st.session_state.submarkets:
        st.markdown("---")
        st.subheader("📈 Results")
        
        # Summary table
        summary_data = []
        for sm in st.session_state.submarkets:
            summary_data.append({
                "Rank": len(summary_data) + 1,
                "City": sm.city,
                "County": sm.county,
                "Composite Score": f"{sm.composite_score:.3f}",
                "Schools": f"{sm.school_score:.3f}",
                "Crime": f"{sm.crime_score:.3f}",
                "Growth": f"{sm.growth_score:.3f}",
                "Price": f"{sm.price_score:.3f}"
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Detailed view
        st.markdown("---")
        st.subheader("🔍 Detailed Analysis")
        
        selected_city = st.selectbox(
            "Select city for details:",
            [sm.city for sm in st.session_state.submarkets]
        )
        
        selected = next((sm for sm in st.session_state.submarkets if sm.city == selected_city), None)
        
        if selected:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Composite Score", f"{selected.composite_score:.3f}")
            with col2:
                st.metric("School Score", f"{selected.school_score:.3f}")
            with col3:
                st.metric("Crime Score", f"{selected.crime_score:.3f}")
            with col4:
                st.metric("Growth Score", f"{selected.growth_score:.3f}")
            
            # Score breakdown chart
            scores = {
                "Schools": selected.school_score,
                "Crime": selected.crime_score,
                "Growth": selected.growth_score,
                "Price": selected.price_score
            }
            
            fig = go.Figure(data=[
                go.Scatterpolar(
                    r=list(scores.values()),
                    theta=list(scores.keys()),
                    fill='toself'
                )
            ])
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1])),
                title=f"Score Breakdown: {selected.city}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional details
            if selected.schools:
                st.markdown("**🏫 Schools:**")
                for school in selected.schools[:3]:
                    st.markdown(f"- {school.name}: Rating {school.rating}/10")
            
            if selected.pricing_data:
                st.markdown(f"**💰 Median Price/SqFt:** ${selected.pricing_data.median_price_per_sqft:.2f}")


def show_land_opportunities():
    """Show land opportunities page."""
    st.header("🏞️ Land Acquisition")
    
    st.markdown("Discover and analyze land opportunities across target markets.")
    
    # Scraping controls
    with st.expander("🔍 Search for Land", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            search_cities = st.multiselect(
                "Cities",
                ["Cary", "Apex", "Holly Springs", "Morrisville", "Wake Forest", "Durham", "Chapel Hill"],
                default=["Cary", "Apex"]
            )
        
        with col2:
            max_price = st.number_input("Max Price", min_value=0, value=0.000, step=10000)
        
        if st.button("🔎 Search Land Listings", type="primary"):
            with st.spinner("Searching land listings..."):
                scraper = LandScraperOrchestrator()
                
                locations = [{"city": city, "state": "NC", "max_price": max_price} for city in search_cities]
                st.session_state.land_listings = scraper.scrape_multiple_locations(locations)
                
                st.success(f"✅ Found {len(st.session_state.land_listings)} listings!")
    
    # Display listings
    if st.session_state.land_listings:
        st.markdown("---")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_city = st.multiselect(
                "Filter by City",
                list(set([l.city for l in st.session_state.land_listings]))
            )
        
        with col2:
            filter_zoning = st.multiselect(
                "Filter by Zoning",
                ["residential", "commercial", "agricultural", "mixed_use"]
            )
        
        with col3:
            sort_by = st.selectbox("Sort By", ["Price (Low to High)", "Price (High to Low)", "Acreage"])
        
        # Apply filters
        filtered = st.session_state.land_listings
        if filter_city:
            filtered = [l for l in filtered if l.city in filter_city]
        if filter_zoning:
            filtered = [l for l in filtered if l.zoning.value in filter_zoning]
        
        # Sort
        if sort_by == "Price (Low to High)":
            filtered.sort(key=lambda x: x.price)
        elif sort_by == "Price (High to Low)":
            filtered.sort(key=lambda x: x.price, reverse=True)
        elif sort_by == "Acreage":
            filtered.sort(key=lambda x: x.acreage or 0, reverse=True)
        
        # Display
        st.subheader(f"📋 {len(filtered)} Listings")
        
        for listing in filtered[:10]:  # Show top 10
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    st.markdown(f"**{listing.address or 'Address Not Listed'}**")
                    st.caption(f"{listing.city}, {listing.county} | {listing.zoning.value}")
                
                with col2:
                    st.metric("Price", format_currency(listing.price))
                
                with col3:
                    st.metric("Acreage", f"{listing.acreage or 'N/A'}")
                
                with col4:
                    if st.button("View Details", key=listing.listing_id):
                        st.write(f"Source: {listing.source}")
                        st.write(f"URL: {listing.url}")
                
                st.markdown("---")


def show_product_intelligence():
    """Show product optimization page."""
    st.header("🏗️ Product Intelligence")
    
    st.markdown("Determine optimal house configurations based on recent sales data.")
    
    # Input
    with st.expander("🔍 Analyze Product Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            product_city = st.selectbox("City", ["Cary", "Apex", "Holly Springs", "Morrisville"])
        
        with col2:
            product_county = st.text_input("County", "Wake")
        
        if st.button("📊 Analyze", type="primary"):
            with st.spinner("Analyzing recent sales..."):
                analyzer = ProductOptimizationAnalyzer()
                product = analyzer.analyze_submarket(product_city, product_county)
                
                if 'products' not in st.session_state or st.session_state.products is None:
                    st.session_state.products = []
                st.session_state.products.append(product)
                
                st.success("✅ Analysis complete!")
    
    # Display
    if st.session_state.products:
        product = st.session_state.products[-1]  # Latest
        
        st.markdown("---")
        st.subheader(f"📈 Optimal Configuration: {product.city}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Square Feet", f"{product.optimal_sqft_min}-{product.optimal_sqft_max}")
        with col2:
            st.metric("Bedrooms", product.optimal_bedrooms)
        with col3:
            st.metric("Bathrooms", product.optimal_bathrooms)
        with col4:
            st.metric("Avg Days on Market", f"{product.avg_days_on_market:.0f}")
        
        # Features
        if product.recommended_features:
            st.markdown("### 🌟 Recommended Features")
            
            feature_df = pd.DataFrame(product.recommended_features)
            feature_df['Frequency %'] = (feature_df['frequency'] * 100).round(1)
            
            fig = px.bar(feature_df[:8], x='Frequency %', y='feature', orientation='h',
                        title="Feature Popularity")
            st.plotly_chart(fig, use_container_width=True)
        
        # Incentives
        if product.effective_incentives:
            st.markdown("### 💡 Effective Incentives")
            
            for incentive in product.effective_incentives:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{incentive['incentive']}**")
                with col2:
                    st.metric("DOM Reduction", f"{incentive['days_on_market_reduction']} days")


def show_micro_market_analysis():
    """Show micro-market analysis with subdivision filtering."""
    st.header("🎯 Micro-Market Analysis")
    
    st.markdown("""
    **Hyper-local intelligence:** Analyze specific subdivisions and neighborhoods within a ZIP code.  
    Same ZIP code, different recommendations!
    """)
    
    # Tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["📍 Subdivision Analysis", "🔄 Compare Subdivisions", "📏 Radius Search"])
    
    # TAB 1: Subdivision Analysis
    with tab1:
        st.subheader("Analyze Specific Subdivision")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            zip_code = st.text_input("ZIP Code", value="27410", key="sub_zip")
        
        with col2:
            months_back = st.selectbox("Historical Period", [6, 12, 24, 36], index=1, key="sub_months")
        
        with col3:
            property_type = st.selectbox(
                "Property Type",
                ["ALL", "Single Family Home", "Townhome", "Condo"],
                index=0,
                key="sub_property_type"
            )
        
        # Get available subdivisions button
        if st.button("🔍 Show Available Subdivisions", key="get_subdivs"):
            with st.spinner("Fetching subdivisions..."):
                try:
                    subdivisions = feature_analyzer.get_subdivisions(zip_code)
                    
                    if subdivisions:
                        st.session_state.subdivisions = subdivisions
                        st.success(f"✅ Found {len(subdivisions)} subdivisions in ZIP {zip_code}")
                    else:
                        st.warning("No subdivisions found in this ZIP code")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Show subdivision selector if available
        if hasattr(st.session_state, 'subdivisions') and st.session_state.subdivisions:
            st.markdown("---")
            
            # Show top subdivisions
            top_subdivs = st.session_state.subdivisions[:20]
            subdiv_df = pd.DataFrame(top_subdivs)
            
            st.markdown("**Top 20 Subdivisions by Property Count:**")
            st.dataframe(subdiv_df, use_container_width=True, hide_index=True)
            
            # Subdivision selector
            subdiv_name = st.selectbox(
                "Select Subdivision to Analyze",
                ["All (No Filter)"] + [s['name'] for s in top_subdivs],
                key="selected_subdiv"
            )
            
            if st.button("🚀 Analyze", type="primary", key="analyze_subdiv"):
                with st.spinner(f"Analyzing {subdiv_name}..."):
                    try:
                        # Determine if filtering by subdivision
                        subdivision_filter = None if subdiv_name == "All (No Filter)" else subdiv_name
                        
                        # Run feature analysis
                        feature_analysis = feature_analyzer.analyze_feature_impact(
                            zip_code,
                            months_back=months_back,
                            min_samples=3,  # Lower threshold to allow more configs
                            subdivision=subdivision_filter,
                            property_type=property_type
                        )
                        
                        # Run demand prediction
                        demand_analysis = demand_predictor.predict_optimal_config(
                            zip_code,
                            months_back=months_back,
                            min_samples=3,  # Lower threshold to allow more configs
                            subdivision=subdivision_filter,
                            property_type=property_type
                        )
                        
                        # Check for errors
                        if 'error' in feature_analysis or 'error' in demand_analysis:
                            st.error(f"Insufficient data for {subdiv_name}. Try selecting a different subdivision or increase the historical period.")
                        elif 'optimal_config' not in demand_analysis or not demand_analysis.get('optimal_config'):
                            # This shouldn't happen, but let's handle it
                            st.warning(f"⚠️ Found {feature_analysis.get('property_count', 0)} properties but couldn't generate recommendation.")
                            with st.expander("Debug Info"):
                                st.json({"demand_keys": list(demand_analysis.keys()), "has_optimal": 'optimal_config' in demand_analysis})
                        else:
                            # Display results
                            st.success("✅ Analysis complete!")
                            
                            # Debug: Show what we got
                            with st.expander("🔍 Debug Info", expanded=False):
                                st.json({"optimal_config_type": type(demand_analysis.get('optimal_config')).__name__, 
                                        "optimal_config_keys": list(demand_analysis.get('optimal_config', {}).keys()) if isinstance(demand_analysis.get('optimal_config'), dict) else "Not a dict"})
                            
                            # Market Overview
                            st.markdown("---")
                            st.subheader("📊 Market Overview")
                            
                            market_stats = feature_analysis['market_stats']
                            optimal = demand_analysis['optimal_config']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Properties Analyzed", feature_analysis['property_count'])
                            with col2:
                                st.metric("Median Price", format_currency(market_stats.get('median_sale_price', 0)))
                            with col3:
                                st.metric("Median Size", f"{market_stats.get('median_size', 0):,.0f} sqft")
                            with col4:
                                st.metric("Price/SqFt", f"${market_stats.get('median_price_per_sqft', 0):.2f}")
                            
                            # Optimal Configuration
                            st.markdown("---")
                            st.subheader("🎯 Optimal Build Recommendation")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Bedrooms", optimal.get('bedrooms', 'N/A'))
                            with col2:
                                st.metric("Bathrooms", optimal.get('bathrooms', 'N/A'))
                            with col3:
                                st.metric("Square Feet", f"{optimal.get('sqft', 0):,}")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Expected Price", format_currency(optimal.get('median_sale_price', 0)))
                            with col2:
                                st.metric("Sales Velocity", f"{optimal.get('sales_velocity', 0):.1f} units/mo")
                            with col3:
                                confidence_pct = optimal.get('confidence', 0) * 100
                                st.metric("Confidence", f"{confidence_pct:.0f}%")
                            
                            st.info(f"**Why:** {optimal.get('rationale', 'Analysis complete')}")
                            
                            # Configuration Performance
                            st.markdown("---")
                            st.subheader("📈 Configuration Performance")
                            
                            config_df = pd.DataFrame(demand_analysis['all_configurations'][:5])
                            
                            if not config_df.empty:
                                fig = px.bar(config_df, 
                                           x='configuration', 
                                           y='sales_velocity',
                                           color='demand_score',
                                           title="Sales Velocity by Configuration",
                                           labels={'sales_velocity': 'Sales per Month', 'configuration': 'Configuration'})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Top Features
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("✨ Top Interior Features")
                                interior = feature_analysis.get('interior_features', [])[:5]
                                if interior:
                                    for feat in interior:
                                        with st.expander(f"{feat['feature']} ({feat['value']})"):
                                            st.write(f"**Priority:** {feat['priority'].title()}")
                                            st.write(feat['rationale'])
                                else:
                                    st.info("No interior features analyzed")
                            
                            with col2:
                                st.subheader("🏠 Top Exterior Features")
                                exterior = feature_analysis.get('exterior_features', [])[:5]
                                if exterior:
                                    for feat in exterior:
                                        with st.expander(f"{feat['feature']} ({feat['value']})"):
                                            st.write(f"**Priority:** {feat['priority'].title()}")
                                            st.write(feat['rationale'])
                                else:
                                    st.info("No exterior features analyzed")
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        st.info("💡 **Tips:** Try a different subdivision or property type, or increase the historical period.")
                        with st.expander("🔍 Technical Details"):
                            import traceback
                            st.code(traceback.format_exc())
    
    # TAB 2: Compare Subdivisions
    with tab2:
        st.subheader("Compare Multiple Subdivisions")
        
        zip_code_compare = st.text_input("ZIP Code", value="27410", key="compare_zip")
        
        if st.button("🔍 Load Subdivisions", key="get_subdivs_compare"):
            with st.spinner("Fetching subdivisions..."):
                try:
                    subdivisions = feature_analyzer.get_subdivisions(zip_code_compare)
                    if subdivisions:
                        st.session_state.subdivisions_compare = subdivisions
                        st.success(f"✅ Found {len(subdivisions)} subdivisions")
                    else:
                        st.warning("No subdivisions found")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if hasattr(st.session_state, 'subdivisions_compare') and st.session_state.subdivisions_compare:
            top_subdivs = st.session_state.subdivisions_compare[:15]
            
            col1, col2 = st.columns(2)
            
            with col1:
                subdiv1 = st.selectbox("First Subdivision", [s['name'] for s in top_subdivs], key="compare1")
            
            with col2:
                subdiv2 = st.selectbox("Second Subdivision", [s['name'] for s in top_subdivs], index=1 if len(top_subdivs) > 1 else 0, key="compare2")
            
            if st.button("⚖️ Compare", type="primary", key="run_compare"):
                with st.spinner("Running comparison..."):
                    try:
                        results = []
                        
                        for subdiv in [subdiv1, subdiv2]:
                            # Analyze each subdivision
                            feature_analysis = feature_analyzer.analyze_feature_impact(
                                zip_code_compare,
                                months_back=24,
                                min_samples=3,
                                subdivision=subdiv,
                                property_type="ALL"  # All property types for comparison
                            )
                            
                            demand_analysis = demand_predictor.predict_optimal_config(
                                zip_code_compare,
                                months_back=24,
                                min_samples=3,
                                subdivision=subdiv,
                                property_type="ALL"  # All property types for comparison
                            )
                            
                            if 'error' not in feature_analysis and 'error' not in demand_analysis:
                                results.append({
                                    'name': subdiv,
                                    'feature': feature_analysis,
                                    'demand': demand_analysis
                                })
                        
                        if len(results) == 2:
                            st.success("✅ Comparison complete!")
                            
                            # Side-by-side comparison
                            col1, col2 = st.columns(2)
                            
                            for i, result in enumerate(results):
                                with (col1 if i == 0 else col2):
                                    st.markdown(f"### {result['name']}")
                                    
                                    stats = result['feature']['market_stats']
                                    optimal = result['demand']['optimal_config']
                                    
                                    st.metric("Properties", result['feature']['property_count'])
                                    st.metric("Median Price", format_currency(stats.get('median_sale_price', 0)))
                                    st.metric("Median Size", f"{stats.get('median_size', 0):,.0f} sqft")
                                    st.metric("Price/SqFt", f"${stats.get('median_price_per_sqft', 0):.2f}")
                                    
                                    st.markdown("**Optimal Build:**")
                                    st.write(f"**{optimal['configuration']}** @ {optimal['sqft']:,} sqft")
                                    st.write(f"Expected: {format_currency(optimal['median_sale_price'])}")
                            
                            # Comparison insights
                            st.markdown("---")
                            st.subheader("💡 Key Differences")
                            
                            price_diff = results[0]['feature']['market_stats'].get('median_sale_price', 0) - results[1]['feature']['market_stats'].get('median_sale_price', 0)
                            size_diff = results[0]['feature']['market_stats'].get('median_size', 0) - results[1]['feature']['market_stats'].get('median_size', 0)
                            
                            st.write(f"**Price Difference:** {format_currency(abs(price_diff))} ({results[0]['name'] if price_diff > 0 else results[1]['name']} is higher)")
                            st.write(f"**Size Difference:** {abs(size_diff):,.0f} sqft ({results[0]['name'] if size_diff > 0 else results[1]['name']} is larger)")
                            
                            st.info("💡 **Insight:** Same ZIP code, but build different products for each subdivision!")
                        
                        else:
                            st.warning("Could not analyze one or both subdivisions. Try different selections.")
                    
                    except Exception as e:
                        st.error(f"Error during comparison: {str(e)}")
    
    # TAB 3: Radius Search
    with tab3:
        st.subheader("Analyze Properties Within Radius")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            zip_code_radius = st.text_input("ZIP Code", value="27410", key="radius_zip")
        
        with col2:
            center_lat = st.number_input("Latitude", value=36.089, format="%.6f", key="radius_lat")
        
        with col3:
            center_lon = st.number_input("Longitude", value=-79.908, format="%.6f", key="radius_lon")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            radius_miles = st.slider("Radius (miles)", 0.25, 2.0, 0.5, 0.25, key="radius_miles")
        
        with col2:
            property_type_radius = st.selectbox(
                "Property Type",
                ["ALL", "Single Family Home", "Townhome", "Condo"],
                index=0,
                key="radius_property_type"
            )
        
        st.info("💡 **Tip:** Use Google Maps to find the latitude/longitude of your lot. Right-click on the location and select 'What's here?'")
        
        if st.button("🎯 Analyze Radius", type="primary", key="analyze_radius"):
            with st.spinner(f"Analyzing properties within {radius_miles} miles..."):
                try:
                    # Run analysis with radius filter (lower threshold for small areas)
                    feature_analysis = feature_analyzer.analyze_feature_impact(
                        zip_code_radius,
                        months_back=24,
                        min_samples=1,  # Lower threshold for radius search
                        radius_miles=radius_miles,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        property_type=property_type_radius
                    )
                    
                    demand_analysis = demand_predictor.predict_optimal_config(
                        zip_code_radius,
                        months_back=24,
                        min_samples=1,  # Lower threshold for radius search
                        radius_miles=radius_miles,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        property_type=property_type_radius
                    )
                    
                    if 'error' in feature_analysis or 'error' in demand_analysis:
                        error_msg = feature_analysis.get('error') or demand_analysis.get('error')
                        st.warning(f"⚠️ Insufficient data within {radius_miles} miles.")
                        st.info(f"**Found:** {feature_analysis.get('property_count', 0)} properties\n\n**Issue:** {error_msg}\n\n💡 Try increasing the radius or choosing a location with more recent sales.")
                    elif 'optimal_config' not in demand_analysis or not demand_analysis['optimal_config']:
                        st.warning(f"⚠️ Found {feature_analysis.get('property_count', 0)} properties but couldn't generate recommendation.")
                        st.info("💡 The properties may not have complete data (beds/baths/size). Try increasing the radius to 1.5 or 2 miles.")
                    else:
                        st.success(f"✅ Found {feature_analysis['property_count']} properties within {radius_miles} miles!")
                        
                        # Display results (similar to Tab 1)
                        market_stats = feature_analysis.get('market_stats', {})
                        optimal = demand_analysis['optimal_config']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Properties", feature_analysis['property_count'])
                        with col2:
                            st.metric("Median Price", format_currency(market_stats.get('median_sale_price', 0)))
                        with col3:
                            st.metric("Median Size", f"{market_stats.get('median_size', 0):,.0f} sqft")
                        with col4:
                            st.metric("Price/SqFt", f"${market_stats.get('median_price_per_sqft', 0):.2f}")
                        
                        st.markdown("---")
                        st.subheader("🎯 Optimal Build for This Location")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Bedrooms", optimal.get('bedrooms', 'N/A'))
                            st.metric("Bathrooms", optimal.get('bathrooms', 'N/A'))
                        with col2:
                            st.metric("Square Feet", f"{optimal.get('sqft', 0):,}")
                            st.metric("Expected Price", format_currency(optimal.get('median_sale_price', 0)))
                        with col3:
                            st.metric("Sales Velocity", f"{optimal.get('sales_velocity', 0):.1f} units/mo")
                            st.metric("Confidence", f"{optimal.get('confidence', 0)*100:.0f}%")
                        
                        st.info(f"**Why:** {optimal.get('rationale', 'Analysis complete')}")
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("💡 **Tips:** Try increasing the radius, or use a different ZIP code with more properties.")


def show_financial_modeling():
    """Show financial modeling page."""
    st.header("💰 Financial Modeling")
    
    st.markdown("Calculate IRR, ROI, and perform sensitivity analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Project Inputs")
        
        land_cost = st.number_input("Land Cost", value=80000, step=5000)
        house_sqft = st.number_input("House Square Feet", value=2000, step=100)
        construction_cost_per_sqft = st.number_input("Construction Cost per SqFt", value=150, step=10)
        timeline_months = st.number_input("Timeline (months)", value=8, step=1)
        sale_price_per_sqft = st.number_input("Sale Price per SqFt", value=180, step=5)
    
    with col2:
        st.subheader("📈 Advanced Settings")
        
        carrying_cost_monthly = st.number_input("Carrying Cost (monthly)", value=500, step=50)
        soft_cost_percentage = st.number_input("Soft Costs %", value=10, step=1) / 100
        discount_rate = st.number_input("Discount Rate %", value=12, step=1) / 100
    
    if st.button("💡 Calculate", type="primary"):
        optimizer = FinancialOptimizer()
        
        construction_cost = house_sqft * construction_cost_per_sqft
        carrying_costs = carrying_cost_monthly * timeline_months
        soft_costs = construction_cost * soft_cost_percentage
        projected_sale = house_sqft * sale_price_per_sqft
        
        financials = optimizer.analyze_project(
            land_cost=land_cost,
            construction_cost=construction_cost,
            carrying_costs=carrying_costs,
            other_costs=soft_costs,
            projected_sale_price=projected_sale,
            timeline_months=timeline_months,
            discount_rate=discount_rate
        )
        
        st.markdown("---")
        st.subheader("📊 Financial Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Investment", format_currency(financials.total_investment))
        with col2:
            st.metric("Projected Sale", format_currency(financials.projected_sale_price))
        with col3:
            st.metric("Gross Profit", format_currency(financials.gross_profit))
        with col4:
            st.metric("Gross Margin", f"{financials.gross_margin:.1f}%")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("IRR (Annual)", f"{financials.irr*100:.2f}%" if financials.irr else "N/A")
        with col2:
            st.metric("ROI", f"{financials.roi:.1f}%")
        with col3:
            st.metric("NPV", format_currency(financials.npv))
        
        # Sensitivity analysis
        st.markdown("---")
        st.subheader("📉 Sensitivity Analysis")
        
        sensitivity = optimizer.sensitivity_analysis(financials)
        
        sens_df = pd.DataFrame(sensitivity)
        fig = px.line(sens_df, x='price_variation', y=['roi', 'irr'],
                     title="ROI & IRR Sensitivity to Sale Price",
                     labels={'value': 'Percentage', 'variable': 'Metric'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(sens_df, use_container_width=True, hide_index=True)


def show_ai_assistant():
    """Show AI chat assistant."""
    st.header("🤖 AI Assistant")
    
    st.markdown("Ask questions about markets, land opportunities, and development strategy in natural language.")
    
    # Initialize RAG if not already done
    if st.session_state.rag_system is None:
        with st.spinner("Initializing AI Assistant..."):
            try:
                st.session_state.rag_system = QdrantRAGSystem()
                st.success("✅ AI Assistant ready!")
            except Exception as e:
                st.error(f"❌ Could not initialize AI Assistant: {e}")
                st.info("Make sure Qdrant is running locally: `docker run -p 6333:6333 qdrant/qdrant`")
                return
    
    # Index data button
    if st.session_state.submarkets or st.session_state.land_listings or st.session_state.products:
        if st.button("📚 Index Data for AI"):
            with st.spinner("Indexing data..."):
                st.session_state.rag_system.bulk_index_data(
                    submarkets=st.session_state.submarkets,
                    land_listings=st.session_state.land_listings,
                    products=st.session_state.products
                )
                st.success("✅ Data indexed!")
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.chat_message("user").write(msg['content'])
        else:
            st.chat_message("assistant").write(msg['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask anything about your real estate data..."):
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        st.chat_message("user").write(prompt)
        
        # Get response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_system.query(prompt)
                answer = response['answer']
                
                # Add assistant message
                st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
                st.chat_message("assistant").write(answer)
                
                # Show sources
                if response.get('sources'):
                    with st.expander("📚 Sources"):
                        for source in response['sources']:
                            st.caption(source)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.chat_history.append({'role': 'assistant', 'content': error_msg})
                st.chat_message("assistant").error(error_msg)


def show_ml_recommendations():
    """Show ML-based build recommendations using seasonality report."""
    st.header("🧠 BuildOptima Recommendation Studio")
    st.caption("Blend current Triad model outputs with historical seasonality context to guide your next spec.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Builder Inputs")

    demo_labels = ["Custom entry"] + [loc["label"] for loc in DEMO_LOCATIONS]
    selected_demo = st.sidebar.selectbox("Hero Locations", demo_labels, index=0, key="preview_demo")

    defaults = {
        "zip": "27410",
        "lat": 36.089,
        "lon": -79.908,
        "subdivision": "",
        "notes": "",
    }
    if selected_demo != "Custom entry":
        chosen = next((loc for loc in DEMO_LOCATIONS if loc["label"] == selected_demo), None)
        if chosen:
            defaults.update(
                {
                    "zip": chosen["zip_code"],
                    "lat": chosen["latitude"],
                    "lon": chosen["longitude"],
                    "subdivision": chosen.get("subdivision") or "",
                    "notes": chosen.get("notes") or "",
                }
            )

    previous_demo = st.session_state.get("_last_selected_demo")
    if selected_demo != "Custom entry" and previous_demo != selected_demo:
        st.session_state["preview_zip"] = defaults["zip"]
        st.session_state["preview_lat"] = defaults["lat"]
        st.session_state["preview_lon"] = defaults["lon"]
        st.session_state["preview_subdivision"] = defaults["subdivision"]
    st.session_state["_last_selected_demo"] = selected_demo

    with st.sidebar.form("builder_form"):
        zip_code = st.text_input("ZIP Code", value=st.session_state.get("preview_zip", defaults["zip"]), key="preview_zip")
        latitude = st.number_input("Latitude", value=st.session_state.get("preview_lat", defaults["lat"]), format="%.6f", key="preview_lat")
        longitude = st.number_input("Longitude", value=st.session_state.get("preview_lon", defaults["lon"]), format="%.6f", key="preview_lon")
        subdivision = st.text_input("Subdivision (optional)", value=st.session_state.get("preview_subdivision", defaults["subdivision"]), key="preview_subdivision")
        property_type = st.selectbox(
            "Property Type",
            ["Any", "Single Family", "Townhome", "Condo"],
            index=0,
            key="preview_property_type",
        )
        generate_clicked = st.form_submit_button("Generate builder recommendation", type="primary")

    if defaults.get("notes"):
        st.sidebar.caption(defaults["notes"])

    def match_rows(zip_code_val: str, lat_val: float, lon_val: float, prop_type_val: str, subdivision_val: str) -> pd.DataFrame:
        df = load_seasonality_report()
        if df.empty:
            return df

        candidates = df[df["zip_code"].astype(str) == str(zip_code_val)].copy()
        if prop_type_val != "Any" and "property_type" in candidates.columns:
            key = prop_type_val.split()[0][:3].upper()
            candidates = candidates[candidates["property_type"].fillna("").str.upper().str.contains(key)]
        if subdivision_val:
            subset = candidates[candidates["subdivision"].fillna("").str.lower() == subdivision_val.lower()]
            if not subset.empty:
                candidates = subset

        if candidates.empty:
            return candidates

        candidates["distance"] = np.sqrt(
            (candidates["latitude"] - lat_val) ** 2 + (candidates["longitude"] - lon_val) ** 2
        )
        candidates["fast_seller_probability"] = candidates.get("fast_seller_probability", 0.0).fillna(0.0)
        candidates.sort_values(["distance", "fast_seller_probability"], ascending=[True, False], inplace=True)
        return candidates.head(3)

    if generate_clicked:
        matches = match_rows(zip_code, latitude, longitude, property_type, subdivision)
        if matches.empty:
            st.warning("No comparable listings found in the seasonality report. Try another ZIP or adjust coordinates.")
            st.session_state.pop("preview_matches", None)
        else:
            st.session_state.preview_matches = matches.to_dict(orient="records")
            st.session_state.triad_model_cache = {}
            st.toast(f"Loaded {len(matches)} nearby baseline comps.")

    matches_state = st.session_state.get("preview_matches", [])
    if not matches_state:
        st.info("Use the controls in the sidebar to load a recommendation.")
        return

    primary = matches_state[0]
    alts = matches_state[1:]

    spec_title = (
        f"{int(primary.get('beds', 0))} BR / {primary.get('baths', 0)} BA · "
        f"{int(primary.get('sqft', 0) or 0):,} sqft"
    )

    zip_code_clean = str(zip_code).strip()
    triad_cache = st.session_state.setdefault("triad_model_cache", {})
    cache_key = "|".join(
        [
            zip_code_clean,
            str(primary.get("property_id") or primary.get("propertyId") or primary.get("id") or ""),
            str(primary.get("latitude") or ""),
            str(primary.get("longitude") or ""),
        ]
    )
    if cache_key not in triad_cache:
        with st.spinner("Scoring with Triad models..."):
            triad_cache[cache_key] = compute_tri_model_predictions(zip_code_clean, primary)
    model_preds = triad_cache.get(cache_key)

    observed_prob = float(primary.get("fast_seller_probability", 0.0) or 0.0)
    observed_dom = float(primary.get("dom_zip_median", primary.get("days_from_list_to_pending", 0)) or 0.0)
    observed_price = primary.get("sale_price") or primary.get("price")
    if isinstance(observed_price, str):
        cleaned = observed_price.strip().replace(",", "").replace("$", "")
        try:
            observed_price = float(cleaned)
        except ValueError:
            observed_price = None

    inv_ratio = safe_float(primary.get("zip_inventory_trend_ratio"))
    triad_prob = observed_prob
    triad_dom = observed_dom
    predicted_price = observed_price
    price_interval = None
    fast_source = "seasonality"
    dom_source = "seasonality"
    price_source = "seasonality"
    if model_preds:
        sp = model_preds.get("sell_probability")
        dom_pred = model_preds.get("expected_dom")
        price_pred = model_preds.get("predicted_price")
        if sp is not None:
            triad_prob = sp
            fast_source = "triad_model"
        if dom_pred is not None:
            triad_dom = dom_pred
            dom_source = "triad_model"
        if price_pred is not None:
            predicted_price = price_pred
            price_source = "triad_model"
        lower = model_preds.get("price_lower")
        upper = model_preds.get("price_upper")
        if lower is not None and upper is not None:
            price_interval = (lower, upper)

    st.markdown(f'<span class="subheader-pill">🎯 {selected_demo}</span>', unsafe_allow_html=True)

    summary_tab, baseline_tab, alternatives_tab, raw_tab = st.tabs(["Summary", "Seasonality Baseline", "Alternatives", "Raw Data"])

    predicted_price_clean = None
    observed_price_clean = None
    if predicted_price is not None:
        pred_val = safe_float(predicted_price)
        if pred_val is not None:
            predicted_price_clean = round(pred_val)
    if observed_price is not None:
        obs_val = safe_float(observed_price)
        if obs_val is not None:
            observed_price_clean = round(obs_val)
    price_interval_clean = (
        (round(price_interval[0]), round(price_interval[1])) if price_interval else None
    )
    inv_ratio_clean = round(inv_ratio, 2) if inv_ratio is not None else None

    narrative_metrics = {
        "lot": {
            "zip_code": primary.get("zip_code"),
            "latitude": primary.get("latitude"),
            "longitude": primary.get("longitude"),
            "subdivision": primary.get("subdivision"),
        },
        "configuration": {
            "beds": primary.get("beds"),
            "baths": primary.get("baths"),
            "sqft": primary.get("sqft"),
            "finish_level": primary.get("finish_level", "standard"),
            "stories": primary.get("stories", 2),
            "garage_spaces": primary.get("garage_spaces", 2),
        },
        "demand": {
            "sell_probability": triad_prob,
            "sell_probability_source": fast_source,
            "expected_dom": triad_dom,
            "expected_dom_source": dom_source,
            "seasonality_fast_probability": observed_prob,
            "seasonality_dom": observed_dom,
        },
        "margin": {
            "gross_margin": 0,
            "gross_margin_pct": 0,
            "roi": 0,
        },
        "pricing": {
            "predicted_sale_price": predicted_price_clean,
            "predicted_sale_price_formatted": format_currency(predicted_price) if predicted_price is not None else None,
            "price_to_zip_median": primary.get("price_to_zip_median"),
            "price_to_subdivision_median": primary.get("price_to_subdivision_median"),
            "observed_sale_price": observed_price_clean,
            "observed_sale_price_formatted": format_currency(observed_price) if observed_price is not None else None,
        },
        "inventory": {
            "zip_sales_count_30d": primary.get("zip_sales_count_30d"),
            "zip_sales_count_90d": primary.get("zip_sales_count_90d"),
            "zip_inventory_trend_ratio": inv_ratio_clean,
        },
    }
    if price_interval_clean:
        narrative_metrics["pricing"]["predicted_price_low"] = price_interval_clean[0]
        narrative_metrics["pricing"]["predicted_price_high"] = price_interval_clean[1]
        narrative_metrics["pricing"]["predicted_price_low_formatted"] = format_currency(price_interval_clean[0])
        narrative_metrics["pricing"]["predicted_price_high_formatted"] = format_currency(price_interval_clean[1])
        narrative_metrics["pricing"]["predicted_price_range"] = (
            f"{format_currency(price_interval_clean[0])} to {format_currency(price_interval_clean[1])}"
        )

    with summary_tab:
        summary_tab.markdown(f"#### {spec_title}")
        summary_tab.markdown('<div class="metric-row">', unsafe_allow_html=True)
        render_metric_card(
            "Triad sell probability",
            f"{triad_prob*100:.0f}%",
            footnote=f"Seasonality baseline {observed_prob*100:.0f}%"
        )
        render_metric_card(
            "Triad expected DOM",
            f"{triad_dom:.0f} days",
            footnote=f"Seasonality baseline {observed_dom:.0f} days"
        )
        observed = safe_float(primary.get("sale_price") or primary.get("price"))
        price_display = format_currency(predicted_price) if predicted_price else (
            format_currency(observed) if observed else "—"
        )
        if predicted_price and predicted_price < 1000 and observed:
            price_display = format_currency(observed)
        price_footnote = "Triad model" if price_source == "triad_model" else "Seasonality baseline"
        render_metric_card(
            "Predicted sale price",
            price_display,
            footnote=price_footnote
        )
        summary_tab.markdown("</div>", unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "Scenario": ["Triad model", "Seasonality baseline"],
            "Probability": [triad_prob * 100, observed_prob * 100],
        })
        prob_fig = px.bar(
            prob_df,
            x="Scenario",
            y="Probability",
            color="Scenario",
            text="Probability",
            height=260,
            title="Sell Probability Comparison (%)"
        )
        prob_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
        prob_fig.update_traces(texttemplate="%{y:.0f}%", textposition="outside")
        summary_tab.plotly_chart(prob_fig, use_container_width=True)

        dom_df = pd.DataFrame({
            "Scenario": ["Triad model", "Seasonality baseline"],
            "DOM": [triad_dom, observed_dom],
        })
        dom_fig = px.bar(
            dom_df,
            x="Scenario",
            y="DOM",
            color="Scenario",
            text="DOM",
            height=260,
            title="Days on Market Comparison"
        )
        dom_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
        dom_fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
        summary_tab.plotly_chart(dom_fig, use_container_width=True)

        if inv_ratio is not None:
            inv_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=inv_ratio,
                    number={"valueformat": ".2f"},
                    gauge={
                        "axis": {"range": [0, max(1.5, inv_ratio * 1.2)], "tickwidth": 1},
                        "bar": {"color": "#38bdf8"},
                        "steps": [
                            {"range": [0, 0.9], "color": "#1e293b"},
                            {"range": [0.9, 1.1], "color": "#0f172a"},
                            {"range": [1.1, max(1.5, inv_ratio * 1.2)], "color": "#172554"},
                        ],
                    },
                    title={"text": "Inventory Trend Ratio (30d / 90d)"}
                )
            )
            inv_fig.update_layout(height=260, margin=dict(l=35, r=35, t=60, b=0))
            summary_tab.plotly_chart(inv_fig, use_container_width=True)

        summary_tab.markdown("##### Narrative")
        summary_tab.info(generate_recommendation_narrative(narrative_metrics))

    with baseline_tab:
        baseline_tab.markdown("#### Historical Seasonality Snapshot")
        baseline_tab.markdown('<div class="metric-row">', unsafe_allow_html=True)
        render_metric_card("Fast-sale odds", f"{observed_prob*100:.0f}%", footnote="Two-year adjusted baseline")
        render_metric_card("Median DOM", f"{observed_dom:.0f} days", footnote="Two-year adjusted baseline")
        if observed_price:
            render_metric_card("Observed sale price", format_currency(observed_price))
        baseline_tab.markdown("</div>", unsafe_allow_html=True)
        baseline_tab.markdown(
            "- Highlights how similar listings performed before the current softening.\n"
            "- Serves as a sanity check when comparing to the live Triad model."
        )

    with alternatives_tab:
        if not alts:
            alternatives_tab.info("No additional nearby comps found.")
        else:
            alternatives_tab.markdown("#### Nearby Alternatives")
            alternatives_tab.markdown('<div class="card-grid">', unsafe_allow_html=True)
            for row in alts:
                alt_price = row.get("sale_price") or row.get("price")
                if isinstance(alt_price, str):
                    try:
                        alt_price = float(alt_price.replace(",", "").replace("$", ""))
                    except ValueError:
                        alt_price = None
                alternatives_tab.markdown(
                    f"""
                    <div class="alt-card">
                        <div class="alt-card-title">{int(row.get('beds', 0))} BR · {row.get('baths', 0)} BA · {int(row.get('sqft', 0) or 0):,} sqft</div>
                        <div class="alt-card-metric"><span>Fast-sale odds</span><span>{row.get('fast_seller_probability', 0)*100:.0f}%</span></div>
                        <div class="alt-card-metric"><span>DOM</span><span>{row.get('dom_zip_median', row.get('days_from_list_to_pending', '?'))}</span></div>
                        <div class="alt-card-metric"><span>Sale price</span><span>{format_currency(alt_price) if alt_price else '—'}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            alternatives_tab.markdown("</div>", unsafe_allow_html=True)

    with raw_tab:
        raw_tab.dataframe(pd.DataFrame(matches_state), use_container_width=True)


def show_listing_popularity():
    """Show listing popularity analysis page."""
    st.header("🔥 Listing Popularity Analysis")
    
    st.markdown("""
    **Analyze active listings from Zillow to identify what makes properties popular:**
    - Which listings get the most attention (views, saves)
    - What features drive popularity
    - DOM to pending analysis (fast-selling properties)
    - Competitive insights for your builds
    """)
    
    st.markdown("---")
    
    # Input section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        zip_code = st.text_input("ZIP Code", value="27410", key="listing_zip")
    
    with col2:
        status = st.selectbox(
            "Listing Status",
            ["active", "pending", "sold"],
            index=0,
            key="listing_status"
        )
    
    with col3:
        max_results = st.number_input(
            "Max Results",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="listing_max_results"
        )
    
    # Source selection
    source = st.radio(
        "Data Source",
        ["auto", "rapidapi_zillow", "attom"],
        index=0,
        help="Auto tries RapidAPI first, falls back to Attom if needed",
        horizontal=True
    )
    
    st.markdown("---")
    
    # Fetch button
    if st.button("🚀 Fetch & Analyze Listings", type="primary", key="fetch_listings"):
        with st.spinner(f"Fetching {status} listings from ZIP {zip_code}..."):
            try:
                # Fetch listings
                listings = safe_listings_scraper.fetch_listings(
                    zip_code=zip_code,
                    status=status,
                    max_results=max_results,
                    source=source
                )
                
                if not listings:
                    st.error("❌ No listings found. Please check:")
                    st.info("💡 Make sure your RapidAPI key is set in .env (or use Attom with --source attom)")
                    return
                
                st.success(f"✅ Found {len(listings)} listings")
                
                # Analyze popularity
                st.markdown("### 📊 Popularity Analysis")
                
                # Determine metric based on available data
                has_views_saves = any(l.get('views') or l.get('saves') for l in listings)
                popularity_metric = 'composite' if has_views_saves else 'fast_dom'
                
                results = popularity_analyzer.analyze_popular_listings(
                    listings=listings,
                    top_n=20,
                    popularity_metric=popularity_metric
                )
                
                # Display top listings
                st.subheader("🏆 Top Popular Listings")
                
                for i, listing in enumerate(results['top_listings'][:10], 1):
                    with st.expander(f"#{i}: {listing.get('address', 'Unknown')[:60]}...", expanded=(i <= 3)):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Price", f"${listing.get('price', 0):,}")
                        with col2:
                            st.metric("Beds/Baths", f"{listing.get('beds', '?')}/{listing.get('baths', '?')}")
                        with col3:
                            st.metric("SqFt", f"{listing.get('sqft', 0):,}")
                        with col4:
                            st.metric("Popularity Score", f"{listing['popularity_score']:.2f}")
                        
                        # Additional metrics
                        col5, col6, col7 = st.columns(3)
                        with col5:
                            dom = listing.get('days_on_zillow', 'N/A')
                            st.metric("Days on Zillow", dom if isinstance(dom, int) else 'N/A')
                        with col6:
                            views = listing.get('views', 'N/A')
                            st.metric("Views", views if views else 'N/A')
                        with col7:
                            saves = listing.get('saves', 'N/A')
                            st.metric("Saves", saves if saves else 'N/A')
                        
                        # Features
                        features = listing.get('features', [])
                        if features:
                            st.write("**Features:**", ", ".join(features[:10]))
                        
                        # Detail URL
                        if listing.get('detail_url'):
                            st.markdown(f"[View on Zillow →]({listing['detail_url']})")
                
                # Feature analysis
                if results['feature_analysis'].get('drivers'):
                    st.subheader("🎯 Popularity Drivers")
                    st.markdown("Features that appear more frequently in popular listings:")
                    
                    drivers = results['feature_analysis']['drivers'][:10]
                    feature_impact = results['feature_analysis'].get('feature_impact', {})
                    
                    for feature in drivers:
                        impact = feature_impact[feature]
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{feature.title()}**")
                            st.caption(
                                f"Appears in {impact['pct_in_top']:.1f}% of top listings "
                                f"(vs {impact['pct_in_all']:.1f}% overall) - "
                                f"{impact['impact_ratio']:.1f}x more common"
                            )
                        with col2:
                            if impact['impact_ratio'] >= 2.0:
                                st.success("🔥 Strong Driver")
                            elif impact['impact_ratio'] >= 1.5:
                                st.info("📈 Driver")
                            else:
                                st.write("✓")
                        st.markdown("---")
                
                # Price analysis
                if results['price_analysis']:
                    st.subheader("💰 Price Analysis")
                    price_data = results['price_analysis']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Median Price", f"${price_data.get('median', 0):,}")
                    with col2:
                        st.metric("Average Price", f"${price_data.get('mean', 0):,}")
                    with col3:
                        st.metric("Price Range", f"${price_data.get('min', 0):,} - ${price_data.get('max', 0):,}")
                    with col4:
                        st.metric("Q25-Q75", f"${price_data.get('q25', 0):,} - ${price_data.get('q75', 0):,}")
                
                # Configuration analysis
                if results['config_analysis']:
                    st.subheader("🏠 Configuration Analysis")
                    config_data = results['config_analysis']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Most Common Configurations:**")
                        for config, count in list(config_data.get('most_common_configs', {}).items())[:5]:
                            st.write(f"- {config}: {count} listings")
                    
                    with col2:
                        if config_data.get('avg_sqft'):
                            st.metric("Avg SqFt", f"{config_data['avg_sqft']:,}")
                        if config_data.get('avg_price_per_sqft'):
                            st.metric("Avg $/SqFt", f"${config_data['avg_price_per_sqft']:,}")
                
                # Pending analysis (if we have both active and pending)
                if status == 'pending' and len(listings) > 0:
                    st.subheader("⏱️ DOM to Pending Analysis")
                    st.info("💡 For DOM to pending analysis, fetch active listings first, then pending to compare.")
                    
                    # Try to analyze if we can
                    try:
                        dom_values = [l.get('days_on_zillow') for l in listings if l.get('days_on_zillow')]
                        if dom_values:
                            st.write(f"**Median DOM to Pending:** {int(np.median(dom_values))} days")
                            st.write(f"**Average DOM to Pending:** {np.mean(dom_values):.1f} days")
                            st.write(f"**Range:** {min(dom_values)} - {max(dom_values)} days")
                    except:
                        pass
                
                # Store in session state for download
                st.session_state['listing_analysis_results'] = results
                st.session_state['listing_data'] = listings
                
                # Download button
                st.markdown("---")
                if st.button("📥 Download Analysis Results (JSON)", key="download_listing_analysis"):
                    import json
                    download_data = {
                        'analysis': results,
                        'listings': listings,
                        'zip_code': zip_code,
                        'status': status,
                        'analyzed_at': datetime.now().isoformat()
                    }
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(download_data, indent=2, default=str),
                        file_name=f"listing_analysis_{zip_code}_{status}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # Show cached results if available
    if 'listing_analysis_results' in st.session_state:
        st.markdown("---")
        st.info("💡 Previous analysis results are cached. Fetch new listings to update.")


if __name__ == "__main__":
    main()






