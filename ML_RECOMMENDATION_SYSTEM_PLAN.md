# ML-Based Lot Recommendation System

## Overview

Transform the current statistical analysis (descriptive: "what sold") into a production ML system (predictive: "what YOUR house will sell for"). The system predicts sale prices for proposed houses based on lot features, house configuration, and specific features (granite, hardwood, etc.), plus demand risk and costs. Includes backtesting, guardrails, human-in-the-loop, and LLM explainability.

**Key Innovation:** Instead of reporting historical medians ("3BR homes sold for $385K"), the ML model predicts the sale price for YOUR specific lot and feature choices ("YOUR 3BR with granite + hardwood on this 0.25-acre lot will sell for $392K ±$31K").

**Critical Addition:** Current listings data pipeline provides real-time competitive context and demand signals, improving prediction accuracy and identifying trending features.

## Architecture

### Core Components:
1. **Training Data Pipeline** - Collect historical sales data + current listings
2. **Pricing Model** - XGBoost/LightGBM to predict sale prices
3. **Demand Model** - Predict probability of selling within X days
4. **Cost Estimation Engine** - Rules-based cost modeling
5. **Ranking & Scoring** - Combine margin + demand risk
6. **Reliability Layer** - Backtesting, guardrails, HITL
7. **LLM Explainability** - Human-readable reasoning

---

## Phase 1: Training Data Collection & Pipeline

### 1.1 Enhanced Data Collector (`backend/data_collectors/training_data_collector.py`)

**Purpose**: Collect comprehensive training dataset from Attom + external sources

**Features to collect from Attom:**
- ✅ Lot size (acres, sqft) - `lot.lotsize1`, `lot.lotsize2`
- ✅ Beds/baths/sqft - `building.rooms.*`, `building.size.*`
- ✅ Stories, garage - `building.summary.*`, `building.parking.*`
- ✅ Quality/condition - `building.summary.quality`, `building.construction.condition`
- ✅ Sale price - `sale.amount.saleamt`
- ✅ Sale date - `sale.saleTransDate`
- ✅ Subdivision - `area.subdname`
- ✅ Lat/long - `location.latitude`, `location.longitude`

**External data sources to add:**
- School district (Census API or web scraping)
- Neighborhood income (Census API ACS)
- Distance to roads (Google Maps Distance Matrix API)
- Zoning (County GIS or manual lookup)
- List price/DOM (Zillow/Realtor scraping for recent sales)

**Output**: Structured CSV/Parquet with all features for each historical sale

### 1.2 Feature Engineering Module (`backend/ml/feature_engineering.py`)

**Create derived features:**
- Price per sqft
- Price per bedroom/bath
- Lot size ratio (sqft / house sqft)
- Age (current year - year_built)
- Market context features (median income, school ratings)
- Distance features (to downtown, to schools, to major roads)
- Temporal features (month of sale, seasonality)

### 1.3 Current Listings Data Pipeline (`backend/data_collectors/listings_scraper.py`) **NEW**

**Purpose**: Collect real-time listings data to enhance predictions with competitive context and demand signals

**Data sources:**
- Zillow API (RapidAPI) - recommended for MVP
- Realtor.com API (RapidAPI) - alternative/supplement
- Web scraping (fallback) - BeautifulSoup4 + Selenium

**Core listing data to collect:**
- Address, ZIP, subdivision, lat/long
- List price, beds/baths, sqft, lot size
- Days on market (DOM)
- Status (active, pending, contingent)
- List date, last updated date
- Property type (SFR, townhome, condo)

**Demand signals (if available):**
- Page views count (Zillow provides)
- Saves/favorites count
- "Hot home" badge/indicator
- Price reductions (count, amount, dates)
- Photo count (quality indicator)
- Virtual tour availability

**Functions to implement:**

1. `scrape_zillow_listings(zip_code, radius_miles, status='active')`
   - Fetch active listings via Zillow API
   - Parse listing details, photos, description
   - Extract demand signals (views, DOM, hot home status)
   - Return structured listing data

2. `extract_features_from_text(description_text)`
   - NLP/text mining on listing descriptions
   - Identify mentioned features: granite, hardwood, stainless, quartz, LVP, smart home, etc.
   - Use regex + keyword matching
   - Return feature list with frequencies

3. `calculate_competitive_context(zip_code, proposed_config, radius_miles=1.0)`
   - Count active listings similar to proposed config (±20% price, same beds/baths)
   - Calculate total inventory level in area
   - Determine price positioning (percentile)
   - Calculate absorption rate (recent sales / active listings)
   - Return competitive metrics dict

4. `track_listing_outcomes(listings_db, closed_sales_db)`
   - Match historical listings to closed sales (by address)
   - Calculate list-to-sale ratio (sale_price / list_price)
   - Track DOM from list date to pending/sold date
   - Identify patterns (fast sellers vs slow sellers)
   - Return list-to-sale patterns

5. `analyze_feature_trends(listings, days_back=30)`
   - Aggregate feature mentions from recent listings
   - Compare to previous period (trending up/down)
   - Identify "hot" features (increasing mentions + fast sales)
   - Return feature popularity report

**Output formats:**

```python
# Listing data structure
listing = {
    'address': '456 Elm St',
    'zip': '27410',
    'subdivision': 'Hamilton Lakes',
    'list_price': 395000,
    'beds': 3,
    'baths': 2.5,
    'sqft': 1850,
    'dom': 12,
    'status': 'active',
    'views': 1247,
    'hot_home': True,
    'features': ['granite', 'hardwood', 'stainless', 'smart home'],
    'scraped_date': '2025-11-04'
}

# Competitive context structure
competitive_context = {
    'active_listings_similar': 8,
    'total_active_listings': 47,
    'inventory_level': 'medium',  # low/medium/high
    'absorption_rate': 0.42,  # monthly (sales/listings)
    'price_percentile': 0.62,
    'avg_dom_active': 28,
    'price_reduction_rate': 0.18
}

# Feature trends structure
feature_trends = {
    'luxury_vinyl_plank': {'count': 76, 'change_pct': 22, 'trending': 'up'},
    'quartz_countertops': {'count': 89, 'change_pct': 8, 'trending': 'up'},
    'granite_countertops': {'count': 112, 'change_pct': -5, 'trending': 'down'},
    'smart_home': {'count': 127, 'change_pct': 15, 'trending': 'hot'}
}
```

**Database schema:**

```sql
-- Active listings table (updated daily/weekly)
CREATE TABLE active_listings (
    id SERIAL PRIMARY KEY,
    address TEXT,
    zip_code TEXT,
    subdivision TEXT,
    list_price DECIMAL,
    beds INT,
    baths DECIMAL,
    sqft INT,
    dom INT,
    status TEXT,
    views INT,
    hot_home BOOLEAN,
    features JSONB,
    list_date DATE,
    scraped_date DATE
);

-- Listing outcomes table (for list-to-sale analysis)
CREATE TABLE listing_outcomes (
    listing_id INT REFERENCES active_listings(id),
    sale_price DECIMAL,
    sale_date DATE,
    list_to_sale_ratio DECIMAL,
    dom_to_pending INT,
    dom_to_sold INT
);
```

**Collection frequency:**
- **Daily:** Active listings count, competitive inventory
- **Weekly:** New listings analysis, feature trends, price reductions
- **Monthly:** List-to-sale outcomes, demand signal validation

**Cost estimate:** $50-200/month for API access (Zillow/Realtor RapidAPI)

**ROI:** Improve pricing accuracy by 2% on $400K home = $8K per home. For 10 homes/year = $80K additional margin vs $1,200/year cost = **66x ROI**

---

## Phase 2: ML Model Development

### 2.1 Pricing Model (`backend/ml/pricing_model.py`)

**Model**: XGBoost or LightGBM regression

**Features** (feature engineering output):
- Lot features: size, frontage (if available), location
- House features: beds, baths, sqft, stories, garage, quality
- Market context: subdivision, ZIP, neighborhood income, school district
- Temporal: sale date, seasonality
- **Competitive context (NEW):** active listings count, inventory level, price positioning

**Target**: Sale price (continuous)

**Output**: 
- Predicted sale price
- Prediction intervals (confidence bounds)
- Feature importance (for explainability)

**Key enhancement:** Model learns how competitive context affects prices (e.g., high inventory → lower prices)

### 2.2 Demand Model (`backend/ml/demand_model.py`)

**Model**: XGBoost/LightGBM classification or regression

**Option A - Classification**: Predict probability of selling within X days (e.g., 90 days)
- Target: Binary (1 = sold within 90 days, 0 = not)
- Output: Probability score

**Option B - Regression**: Predict expected DOM
- Target: Days on market (continuous)
- Output: Expected DOM + confidence interval

**Features**: Same as pricing model + price relative to market + **current inventory metrics (NEW)**

**Recommendation**: Start with Option A (classification) for simplicity

**Key enhancement:** Model uses current inventory level and absorption rate to predict demand

### 2.3 Model Training Pipeline (`backend/ml/train_models.py`)

**Functions**:
- `train_pricing_model(zip_codes, train_start_date, train_end_date, test_end_date)`
- `train_demand_model(zip_codes, train_start_date, train_end_date, test_end_date)`
- Cross-validation, hyperparameter tuning
- Model persistence (save/load)
- Feature importance export

**Data splitting**:
- Train: Historical data through 2023
- Test: 2024 data for backtesting
- Validation: Holdout set for hyperparameter tuning

---

## Phase 3: Cost Estimation Engine

### 3.1 Cost Model (`backend/analyzers/cost_estimator.py`)

**Rules-based cost function**:

**Base costs**:
- Base cost per sqft by house type (SFR, Townhome, Condo)
- Base cost per sqft by finish level (entry, mid, high)

**Adjustments**:
- Lot conditions: slope penalties, retaining walls, utility connections
- Regional factors: labor costs, material costs by ZIP
- Scale factors: lot size, foundation type

**Output**: Total construction cost estimate

**Future**: Can be replaced with ML model if you collect cost data

---

## Phase 4: Ranking & Recommendation Engine

### 4.1 Enhanced Build Recommender (`backend/analyzers/ml_build_recommender.py`)

**Core logic**:

```python
def recommend_for_lot(lot_features, candidate_configs):
    # Get current market conditions (NEW)
    current_listings = scrape_zillow_listings(lot_features['zip'], radius_miles=1)
    competitive_context = calculate_competitive_context(
        lot_features['zip'], 
        candidate_configs,
        radius_miles=1
    )
    
    recommendations = []
    
    for config in candidate_configs:
        # 1. Predict sale price (with competitive context)
        predicted_price = pricing_model.predict(
            lot_features + config + competitive_context
        )
        
        # 2. Predict demand (with current inventory)
        demand_prob = demand_model.predict(
            lot_features + config + predicted_price + competitive_context
        )
        
        # 3. Estimate cost
        estimated_cost = cost_estimator.estimate(config, lot_features)
        
        # 4. Calculate margin
        margin = predicted_price - estimated_cost - lot_price - sga_allocation
        
        # 5. Score
        score = combine_margin_and_demand(margin, demand_prob)
        
        recommendations.append({
            'config': config,
            'predicted_price': predicted_price,
            'demand_probability': demand_prob,
            'estimated_cost': estimated_cost,
            'margin': margin,
            'margin_pct': margin / predicted_price,
            'score': score,
            'competitive_context': competitive_context  # For transparency
        })
    
    # Rank by score (or margin subject to demand constraint)
    return rank_recommendations(recommendations, min_demand_prob=0.70)
```

**Candidate generation**:
- Generate plausible configs (3BR/2BA, 3BR/2.5BA, 4BR/3BA, etc.)
- Constrain by lot size, zoning, market norms
- Vary finish levels (entry, mid, high)
- **Consider trending features from listings analysis (NEW)**

**Output format**:

```python
{
    'recommendations': [
        {
            'plan': '4BR/3BA, ~2,200 SF, mid-level finishes',
            'predicted_sale_price': 515000,
            'estimated_cost': 375000,
            'gross_margin': 140000,
            'margin_pct': 0.27,
            'demand_probability': 0.78,
            'confidence': 'high',
            'competitive_context': {
                'similar_listings': 5,
                'inventory_level': 'medium',
                'recommendation': 'Good timing - moderate competition'
            },
            'rationale': '...'  # LLM-generated
        },
        # ... more recommendations
    ],
    'confidence': 'high',
    'guardrails': []
}
```

---

## Phase 5: Reliability & Guardrails

### 5.1 Backtesting Module (`backend/ml/backtesting.py`)

**Functions**:
- `backtest_pricing_model(test_data, model)` → MAPE, RMSE, R²
- `backtest_demand_model(test_data, model)` → Accuracy, Precision, Recall, ROC-AUC
- `generate_backtest_report()` → Summary statistics

**Output**: 
- "On last 200 homes in this micro-market, pricing model was within ±8% of actual sale price in 75% of cases"
- "Demand model correctly predicted fast vs slow sellers 82% of the time"

### 5.2 Guardrails Module (`backend/ml/guardrails.py`)

**Checks**:
- Sample size validation (micro-market too small?)
- Out-of-distribution detection (unusual lot/features?)
- Prediction confidence thresholds
- Model version validation
- **Market condition alerts (NEW):** Extreme inventory levels, rapid price changes

**Actions**:
- Flag low confidence recommendations
- Fall back to simpler heuristics if model unreliable
- Require human review for edge cases
- **Alert on market shifts (NEW):** "Inventory up 40% in 30 days - demand predictions may be conservative"

**Output**: Confidence level + warnings

### 5.3 Human-in-the-Loop (HITL) (`backend/ml/feedback_loop.py`)

**Features**:
- Log all recommendations (config, predictions, actual outcomes)
- Track user overrides (accept, tweak, reject)
- Store override reasons
- Use feedback to improve models

**Database schema**:

```python
recommendations_table:
    - lot_id
    - recommended_config
    - predicted_price
    - competitive_context (snapshot)
    - user_action (accept/tweak/reject)
    - actual_outcome (if built)
    - actual_sale_price (if sold)
    - actual_dom (if sold)
    - override_reason
```

---

## Phase 6: LLM Explainability Layer

### 6.1 Explanation Generator (`backend/ml/explainability.py`)

**Input**: Model predictions + feature importance + competitive context + backtest results

**LLM prompt**:

```
Generate a human-readable explanation for why this configuration is recommended:

Lot: {lot_address}
Recommended: {config}
Predicted Price: ${predicted_price}
Predicted Demand: {demand_prob}% chance to sell within 90 days

Key drivers:
- Feature importance: {top_features}
- Market context: {market_stats}
- Competitive context: {active_listings}, {inventory_level}
- Feature trends: {trending_features}
- Historical performance: {backtest_stats}

Write 2-3 sentences explaining the recommendation in plain English, including competitive context.
```

**Output**: Natural language rationale (e.g., "We recommend 4BR/3BA here because 70% of recent sales in this school district between $480-550k were 4BR, and 4BR homes sold on average 18 days faster than 3BR. Currently, there are only 3 similar homes listed (low competition), and 4BR demand is trending up based on listing views.")

---

## Phase 7: Integration & UI

### 7.1 New "ML Lot Recommender" Page (`prototype/app.py`)

**UI Flow**:
1. Input lot details (address, ZIP, price, size, zoning)
2. Select candidate configs or auto-generate
3. **Display current market snapshot (NEW):** Active listings, trending features, inventory level
4. Run ML models
5. Display ranked recommendations with:
   - Predicted price, cost, margin
   - Demand probability
   - **Competitive context summary (NEW)**
   - **Feature trend insights (NEW)**
   - LLM explanation
   - Confidence level
   - Guardrails/warnings
6. HITL interface (accept/tweak/reject)
7. Backtest report (if available)

### 7.2 Model Management UI

- View model performance (backtest results)
- Train new models
- Compare model versions
- View feature importance

### 7.3 Market Intelligence Dashboard (NEW)

- Current listings overview (count, DOM trends)
- Feature popularity trends (what's hot)
- Competitive inventory heatmap (by ZIP/subdivision)
- List-to-sale ratio trends
- Absorption rate tracking

---

## Implementation Order

### Sprint 1: Foundation (Week 1-2)
- Data collection pipeline (Attom + Census API)
- **Listings scraper (Zillow/Realtor API) - NEW**
- Feature engineering
- Training data export

### Sprint 2: ML Models (Week 3-4)
- Pricing model (XGBoost with competitive context)
- Demand model (XGBoost classification with inventory metrics)
- Model training pipeline
- Basic backtesting

### Sprint 3: Integration (Week 5-6)
- Cost estimation engine
- Ranking/scoring system
- Enhanced build recommender
- Guardrails

### Sprint 4: Polish (Week 7-8)
- LLM explainability
- HITL interface
- UI integration
- **Market intelligence dashboard - NEW**
- Comprehensive backtesting

---

## Data Requirements

### Minimum viable dataset:
- 500+ sales across target markets
- 12+ months of data
- Representative mix of configurations
- **100+ current listings for competitive context - NEW**

### Ideal dataset:
- 2,000+ sales
- 24+ months
- Multiple ZIP codes/subdivisions
- Complete feature coverage
- **500+ tracked listings with outcomes - NEW**

---

## Technical Stack

- **ML**: XGBoost or LightGBM (via scikit-learn)
- **Data**: pandas, numpy
- **Feature stores**: pandas DataFrames (future: Feast)
- **Model persistence**: joblib or pickle
- **Backtesting**: scikit-learn metrics
- **LLM**: OpenAI GPT-4 (existing integration)
- **Listings APIs**: Zillow/Realtor (RapidAPI) - NEW
- **Web scraping**: BeautifulSoup4, Selenium (fallback) - NEW
- **NLP**: NLTK or spaCy for text mining - NEW

---

## Success Metrics

1. **Pricing accuracy**: MAPE < 10% on test set (target: 8% with competitive context)
2. **Demand prediction**: ROC-AUC > 0.75 (target: 0.80 with inventory metrics)
3. **Recommendation quality**: User acceptance rate > 60%
4. **Reliability**: Confidence flags correctly identify edge cases
5. **Explainability**: Users rate explanations as "clear" > 80%
6. **Competitive accuracy (NEW)**: Predictions within ±5% when accounting for current inventory

---

## Cost Structure

### One-time costs:
- Development: 7-8 weeks (internal time)
- Census API setup: Free
- Google Maps API: ~$200 credits (free tier)

### Recurring costs:
- Zillow/Realtor API: $50-200/month
- Google Maps API: $0-50/month (depending on volume)
- OpenAI API: ~$50-100/month (for LLM explanations)
- **Total: ~$150-350/month**

### ROI:
- Improve pricing accuracy by 2% on $400K home = $8K per home
- 10 homes/year = $80K additional margin
- vs $4,200/year operating cost
- **ROI: 19x**

---

## Future Enhancements

- Replace cost model with ML (if cost data available)
- Active learning (continuous model improvement)
- Transfer learning from other markets
- Ensemble methods (combine multiple models)
- Real-time market updates (adjust predictions dynamically)
- **Predictive inventory forecasting (forecast future competition) - PHASE 2**
- **Buyer sentiment analysis (social media, search trends) - PHASE 2**
- **Macro trend integration (interest rates, employment) - PHASE 2**

---

## Implementation Todos

### Sprint 1: Foundation
1. **data-collection-pipeline**: Build training data collector: extract all features from Attom (lot size, beds/baths, sale price, etc.) and structure into training dataset. Add external data sources (Census API for income, Google Maps for distance calculations).

2. **listings-scraper**: Build current listings data pipeline: integrate Zillow/Realtor API, scrape active listings, extract demand signals (views, DOM, hot homes), implement text mining for features, calculate competitive context metrics.

3. **feature-engineering**: Create feature engineering module: derive features (price/sqft, lot ratios, temporal features, market context, competitive context). Prepare features for ML models.

### Sprint 2: ML Models
4. **pricing-model**: Build pricing model: XGBoost/LightGBM regression to predict sale prices. Include competitive context features. Train/test split (2023/2024), hyperparameter tuning, model persistence. Output: predicted price + confidence intervals + feature importance.

5. **demand-model**: Build demand model: XGBoost/LightGBM classification to predict probability of selling within 90 days. Include current inventory metrics. Train/test split, evaluation metrics. Output: demand probability score.

### Sprint 3: Integration
6. **cost-estimator**: Create cost estimation engine: rules-based model with base costs per sqft by house type/finish level, lot condition adjustments, regional factors. Output: total construction cost estimate.

7. **ranking-engine**: Build ML-based ranking system: generate candidate configs (include trending features), predict price/demand/cost for each with competitive context, calculate margin, rank by score (margin + demand constraint). Output: ranked recommendations with all metrics.

### Sprint 4: Polish
8. **backtesting**: Implement backtesting module: test pricing model (MAPE, RMSE, R²) and demand model (accuracy, ROC-AUC) on 2024 holdout data. Generate performance reports showing competitive context impact.

9. **guardrails**: Build guardrails system: validate sample size, detect out-of-distribution, check prediction confidence, alert on market shifts (extreme inventory changes). Flag low-confidence recommendations, fall back to heuristics when needed.

10. **hitl-interface**: Create human-in-the-loop system: log recommendations with competitive context snapshot, track user actions (accept/tweak/reject), store override reasons. Database schema for feedback loop.

11. **llm-explainability**: Build LLM explainability layer: use GPT-4 to generate human-readable explanations from model predictions + feature importance + competitive context. Integrate with ranking engine.

12. **ui-integration**: Create new 'ML Lot Recommender' page in Streamlit: input lot details, display current market snapshot, run models, display ranked recommendations with predictions/explanations/competitive context, HITL interface. Add Market Intelligence Dashboard showing listings trends, feature popularity, inventory levels.

