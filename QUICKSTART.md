# 🚀 Quick Start Guide

## Real Estate Intelligence Platform - Prototype

Get up and running in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- Docker (optional, for Qdrant vector database)
- API Keys:
  - OpenAI API key (required for AI features)
  - GreatSchools API key (optional)
  - Census API key (optional)

## Installation

### 1. Set Up Environment

The virtual environment and dependencies should already be installed. If not:

```bash
cd real-estate-intel
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Start Qdrant (Optional but Recommended)

The AI Assistant requires Qdrant for vector search:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Leave this running in a separate terminal.

### 4. Launch the Prototype

```bash
./run_prototype.sh
```

Or manually:

```bash
source venv/bin/activate
streamlit run prototype/app.py
```

## 🎯 Features

The prototype includes 6 main sections:

### 1. 🏠 Dashboard
- Executive overview with key metrics
- Top submarkets visualization
- Recent activity feed

### 2. 📊 Market Analysis
- Analyze submarkets based on schools, crime, growth, and pricing
- Compare multiple locations
- Detailed scoring breakdown

### 3. 🏞️ Land Opportunities
- Search and discover land listings
- Filter by city, price, zoning
- Track listings over time

### 4. 🏗️ Product Intelligence
- Determine optimal house size and features
- Analyze what sells best in each market
- Incentive effectiveness analysis

### 5. 💰 Financial Modeling
- Calculate IRR and ROI
- Sensitivity analysis
- Break-even calculations

### 6. 🤖 AI Assistant
- Natural language querying of all data
- Ask questions like:
  - "What are the best submarkets in Wake County?"
  - "Show me land under $100k in Apex"
  - "What house size sells fastest in Cary?"

## 📝 Usage Examples

### Analyze a New Submarket

1. Go to **📊 Market Analysis**
2. Enter cities (one per line):
   ```
   Cary
   Apex
   Durham
   ```
3. Enter corresponding counties
4. Click **🚀 Run Analysis**
5. View ranked results and detailed breakdowns

### Find Land Opportunities

1. Go to **🏞️ Land Opportunities**
2. Select target cities
3. Set max price
4. Click **🔎 Search Land Listings**
5. Filter and sort results

### Calculate Project Financials

1. Go to **💰 Financial Modeling**
2. Enter:
   - Land cost: $80,000
   - House size: 2,000 sqft
   - Construction cost/sqft: $150
   - Sale price/sqft: $180
3. Click **💡 Calculate**
4. Review IRR, ROI, NPV, and sensitivity analysis

### Use AI Assistant

1. Go to **🤖 AI Assistant**
2. First, index your data: Click **📚 Index Data for AI**
3. Ask questions in natural language:
   - "Which submarket has the best schools?"
   - "What's the average price per square foot in Apex?"
   - "Show me the cheapest land listings"

## ⚙️ Configuration

Edit `.env` to customize:

### Analysis Weights

Adjust how much each factor contributes to submarket scoring:

```
SCHOOL_WEIGHT=0.30    # 30%
CRIME_WEIGHT=0.25     # 25%
GROWTH_WEIGHT=0.25    # 25%
PRICE_WEIGHT=0.20     # 20%
```

### Target Markets

```
TARGET_STATE=NC
TARGET_COUNTIES=Wake,Durham,Mecklenburg,Forsyth,Guilford
```

### Financial Parameters

```
DEFAULT_CONSTRUCTION_COST_PER_SQFT=150
DEFAULT_CARRYING_COST_MONTHLY=500
DEFAULT_BUILD_TIME_MONTHS=6
DEFAULT_SALE_TIME_MONTHS=2
DEFAULT_DISCOUNT_RATE=0.12
```

## 🔧 Troubleshooting

### "Module not found" errors

Make sure you've activated the virtual environment:

```bash
source venv/bin/activate
```

### Qdrant connection errors

The app works without Qdrant, but AI features will be limited. To fix:

1. Install Docker
2. Run: `docker run -p 6333:6333 qdrant/qdrant`
3. Restart the app

### OpenAI API errors

Make sure your `.env` file has a valid API key:

```
OPENAI_API_KEY=sk-...your-key-here...
```

## 📊 Mock Data

The prototype includes mock data for demonstration. To use real data:

1. Add actual API keys to `.env`
2. The system will automatically use real APIs when available
3. Mock data is used as fallback when APIs are unavailable

## 🚀 Next Steps

### Phase 2: Production System

The current prototype is designed for rapid demos. For production:

1. **Database**: Migrate from file storage to PostgreSQL
2. **FastAPI Backend**: Build RESTful API layer
3. **React Frontend**: Modern production UI
4. **Authentication**: Add user management
5. **Deployment**: Deploy to AWS/DigitalOcean

See `README.md` for full development roadmap.

## 🆘 Support

For issues or questions:

1. Check the logs in the Streamlit interface
2. Review the README.md for detailed documentation
3. Ensure all dependencies are installed correctly

## 📄 License

Proprietary - All rights reserved

