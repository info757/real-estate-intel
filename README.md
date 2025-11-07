# Real Estate Intelligence Platform

An AI-powered platform for identifying optimal submarkets, land acquisition opportunities, and product configurations for residential real estate development in North Carolina.

## Features

- **Market Analysis**: Analyze submarkets based on schools, crime, growth metrics, and pricing
- **Land Acquisition**: Automated scraping and scoring of land listings
- **Product Optimization**: Identify optimal house sizes, features, and incentives
- **Financial Modeling**: IRR calculations and SG&A optimization
- **AI Query Interface**: Natural language querying of all data using RAG

## Architecture

### Phase 1: Rapid Prototype
- **Frontend**: Streamlit for rapid prototyping and client demos
- **Backend**: Python data collectors and analyzers
- **Vector DB**: Qdrant for semantic search and RAG
- **AI**: LangChain + OpenAI for natural language queries

### Phase 2: Production System
- **Frontend**: React with Material-UI
- **Backend**: FastAPI with PostgreSQL
- **Deployment**: Docker containers on AWS/DigitalOcean

## Project Structure

```
real-estate-intel/
├── backend/
│   ├── data_collectors/     # Data collection modules
│   ├── analyzers/           # Analysis engines
│   ├── models/              # Data models
│   ├── ai_engine/           # LangChain/RAG setup
│   └── utils/               # Utility functions
├── prototype/               # Streamlit app
├── frontend/                # React app (Phase 2)
├── data/                    # Local data storage
├── config/                  # Configuration files
└── tests/                   # Test suite
```

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (for production)
- Qdrant (can run locally via Docker)
- API Keys: OpenAI, GreatSchools, Census

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd real-estate-intel
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

5. Start Qdrant locally (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Running the Prototype

```bash
streamlit run prototype/app.py
```

## Branch Strategy & Demo Freeze

- `main` is frozen for the upcoming demo; only ship critical demo-blocking fixes directly here.
- Use `feature/realestateapi-integration` for the RealEstateApi MCP workstream. Create child branches off it as needed, then merge via PRs back into the integration branch before promoting to `main`.

## Data Sources

### Free APIs
- **US Census API**: Population, demographics, economic data
- **FBI Crime Data**: Crime statistics
- **Bureau of Labor Statistics**: Employment data
- **NC Public Data**: School and public records

### Paid/Premium APIs (Optional)
- **GreatSchools API**: School ratings and data
- **Zillow API**: Real estate listings and pricing
- **Attom Data**: Property data and analytics

### Web Scraping
- Zillow, Realtor.com, LandWatch, Lands of America
- Respectful scraping with rate limiting

## Configuration

Edit `.env` file to customize:
- **Analysis Weights**: Adjust importance of schools, crime, growth, pricing
- **Target Markets**: Specify counties and regions to analyze
- **Financial Parameters**: Set construction costs, timelines, discount rates

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black backend/ prototype/
flake8 backend/ prototype/
```

## Deployment

### Prototype (Streamlit)
```bash
docker build -t real-estate-intel-prototype .
docker run -p 8501:8501 real-estate-intel-prototype
```

### Production (Coming in Phase 2)
- FastAPI backend on AWS ECS or DigitalOcean
- React frontend on Vercel/Netlify
- PostgreSQL on managed service
- Qdrant Cloud for vector database

## Usage Examples

### Natural Language Queries

- "What are the top 3 submarkets in Wake County for family homes?"
- "Show me land under $100k in high-growth areas with good schools"
- "What's the optimal house size for Cary, NC based on recent sales?"
- "Calculate IRR if I buy this lot for $80k and build a 2000 sqft house"

## Success Metrics

- Time to identify viable submarket: **<5 minutes** (vs hours manually)
- Land opportunities surfaced per week: **50+ with scoring**
- Accuracy of price/sqft predictions: **±10%**
- User queries answered via AI: **80%+ success rate**
- Cost per land acquisition: **Reduce by 70%** through automation

## License

Proprietary - All rights reserved

## Support

For questions or issues, please contact the development team.

