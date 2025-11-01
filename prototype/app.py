"""
Streamlit prototype for Real Estate Intelligence Platform.
Rapid prototype for client demos with dashboard, analysis, and AI chat.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data_collectors.market_data import MarketAnalysisCollector
from backend.data_collectors.land_scraper import LandScraperOrchestrator
from backend.data_collectors.sales_data import ProductOptimizationAnalyzer
from backend.analyzers.submarket_ranker import SubmarketRanker
from backend.analyzers.land_analyzer import LandOpportunityAnalyzer
from backend.analyzers.financial_optimizer import FinancialOptimizer
from backend.ai_engine.rag_system import QdrantRAGSystem
from config.settings import settings

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


def main():
    """Main application."""
    
    st.markdown('<div class="main-header">🏘️ Real Estate Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("*AI-Powered Market Analysis & Development Optimization for North Carolina*")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏘️ RE Intel")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📊 Market Analysis", "🏞️ Land Opportunities", 
             "🏗️ Product Intelligence", "💰 Financial Modeling", "🤖 AI Assistant"]
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
    elif page == "🏞️ Land Opportunities":
        show_land_opportunities()
    elif page == "🏗️ Product Intelligence":
        show_product_intelligence()
    elif page == "💰 Financial Modeling":
        show_financial_modeling()
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
            max_price = st.number_input("Max Price", min_value=0, value=150000, step=10000)
        
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


if __name__ == "__main__":
    main()

