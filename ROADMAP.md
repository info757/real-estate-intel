# Real Estate Intelligence Platform - Product Roadmap

## Current State: Phase 1 - Proof of Concept ✅

### What We've Built (November 2025)

**Data Integration:**
- ✅ Attom Data API integration (free tier)
- ✅ 500 properties per ZIP code
- ✅ Rate limiting & caching (500 calls/day, 7-30 day cache)
- ✅ Real property data from NC (validated against Zillow)

**Analytics Engine:**
- ✅ Feature Analyzer: Identifies which interior/exterior features drive sales
- ✅ Demand Predictor: Predicts optimal bed/bath/sqft configurations
- ✅ Build Recommender: Lot-specific recommendations with ROI/IRR
- ✅ Micro-Market Analysis:
  - Subdivision-level filtering
  - Radius search (0.5-2 mile circles)
  - Subdivision comparison

**Data Quality:**
- 84.8% have sale price
- 99.0% have size/beds
- 98.6% have subdivision
- Most recent data: 31 days old

**Example Insights:**
- Same ZIP (27410), different subdivisions:
  - Victorian/PK: $263k, 850 sqft → Build 2BR/3BA townhome
  - Brooks: $231k, 1161 sqft → Build 2BR/2BA
  - Dresden Woods: $435k, 1792 sqft → Build 4BR/3BA SFH
  - Hamilton Lakes: $302k-$761k (established, low turnover)

---

## Future State: Phase 2 - Comprehensive Historical Analysis & Trend Detection

### Vision
**Transform from snapshot analysis to predictive time-series intelligence that identifies market saturation and emerging opportunities.**

### Core Concept
Instead of just asking "What sells best now?", answer:
1. "What has sold best over the last 5-10 years?"
2. "Is that configuration still performing or is it oversaturated?"
3. "What's the emerging opportunity that competitors are missing?"

### Technical Enhancements

#### 1. Comprehensive Data Collection
**Goal:** Analyze ALL available properties, not just 500

**Implementation:**
- Increase pagination from 5 pages → 100+ pages per ZIP
- Fetch all 10,000 properties in ZIP 27410
- Store in local database (PostgreSQL) for historical analysis
- Schedule daily/weekly refreshes

**Trade-offs:**
- More API calls (manage with longer cache)
- Longer initial load (acceptable for analysis)
- Need local storage (already have PostgreSQL)

**Estimated API Usage:**
- 10,000 properties = 100 pages × 1 call = 100 API calls per ZIP
- With 5 target ZIPs = 500 calls (exactly our daily limit)
- Run once per week = sustainable on free tier

#### 2. Historical Sales Tracking
**Goal:** Track each property's complete sales history, not just most recent sale

**Data to Track:**
- All sales for each property (2004-2025 in our data)
- Price appreciation over time
- Days on market trends
- Buyer type (cash vs mortgage, owner vs investor)

**Analysis Enabled:**
- "This 3BR/2BA sold for $250k in 2020, $310k in 2022, $285k in 2024"
- Identify price peaks and corrections
- Understand market cycles

**Implementation:**
- Attom provides `sale.vintage.lastModified` - track when records change
- Store each property's full sales history in database
- Link properties by `identifier.attomId` across time

#### 3. Time-Series Trend Analysis
**Goal:** Identify configuration saturation and emerging trends

**Key Metrics to Track Over Time:**

```
Configuration Performance by Year:
- Sales velocity (units/month) by year
- Price appreciation by year
- Days on market by year
- Market share by year
```

**Example Analysis:**

```
4BR/3BA in ZIP 27410:
  2020: 15 sales, 12 days DOM, +8% YoY price growth  → HOT
  2021: 28 sales, 14 days DOM, +12% YoY             → VERY HOT
  2022: 35 sales, 18 days DOM, +5% YoY              → PEAK
  2023: 32 sales, 28 days DOM, +1% YoY              → COOLING
  2024: 18 sales, 45 days DOM, -2% YoY              → OVERSATURATED ⚠️
  
Insight: "4BR/3BA peaked in 2022. Now oversaturated. 
         Look at 3BR/2BA instead - gaining momentum."

3BR/2BA in ZIP 27410:
  2020: 8 sales, 45 days DOM
  2021: 10 sales, 38 days DOM
  2022: 12 sales, 32 days DOM
  2023: 18 sales, 25 days DOM  → EMERGING
  2024: 25 sales, 18 days DOM  → OPPORTUNITY ✅
```

**Statistical Methods:**
- Moving averages (3-month, 6-month, 12-month)
- Linear regression for trend detection
- Seasonality adjustment
- Market saturation index: `(Current DOM / Historical Avg DOM) × (Price Change Rate)`

#### 4. Predictive Recommendations

**Current (Phase 1):**
> "Build 4BR/3BA - it sells at 0.5 units/month"

**Future (Phase 2):**
> "⚠️ DON'T build 4BR/3BA:
> - Peaked in 2022 (35 sales)
> - Down 49% in 2024 (18 sales)
> - Days on market: 12 → 45 days (+275%)
> - Oversaturation index: 2.8 (high risk)
> 
> ✅ BUILD 3BR/2BA instead:
> - Growing: 8 sales (2020) → 25 sales (2024) +213%
> - Accelerating: DOM improved 45 → 18 days
> - Underserved: Only 8% of new construction vs 18% of demand
> - Momentum score: 8.5/10 (strong buy signal)"

#### 5. Feature Trend Analysis

**Track feature popularity over time:**

```
Granite Countertops:
  2020: 45% of sales had it
  2021: 62% of sales
  2022: 78% of sales
  2023: 85% of sales
  2024: 91% of sales  → NOW STANDARD (not a differentiator)

Quartz Countertops:
  2020: 8% of sales
  2021: 12% of sales
  2022: 18% of sales
  2023: 28% of sales
  2024: 35% of sales  → EMERGING PREMIUM FEATURE ✅

Smart Home Features:
  2020: 5% of sales
  2024: 34% of sales  → RAPID ADOPTION, INSTALL NOW
```

**Recommendation Engine Enhancement:**
> "Skip granite (now standard) → Install quartz (+$8k value)
> Add smart thermostat (+$2k value, low cost)
> Skip pool (declining: 12% → 8% in 2024)"

#### 6. Subdivision Life Cycle Analysis

**Identify subdivision maturity stages:**

```
Hamilton Lakes:
  Peak: 2004-2005 (63 sales + 36 sales)
  Mature: 2010-2020 (12-22 sales/year)
  Established: 2021-2024 (0-3 sales/year)
  Status: LOW TURNOVER, HIGH-END, STABLE
  
Victorian/PK:
  Active turnover: 15-20 sales/year consistently
  Status: ACTIVE RESALE MARKET
  
Oak Ridge Meadows:
  Growing: Recent development with active sales
  Status: EMERGING, HIGH VELOCITY
```

**Competitive Intelligence:**
> "Avoid Hamilton Lakes - established with low turnover (0-3 sales/year)
> Target Oak Ridge Meadows - active with 15+ sales/year
> Opportunity: Fill gap between entry ($200k) and mid ($435k)"

### Implementation Priority

**Phase 2A: Data Foundation (Month 1)**
- [ ] Increase pagination to fetch all properties per ZIP
- [ ] Set up PostgreSQL tables for historical tracking
- [ ] Create data refresh job (weekly)
- [ ] Validate data completeness (10,000 properties)

**Phase 2B: Time-Series Analysis (Month 2)**
- [ ] Build year-over-year comparison engine
- [ ] Calculate trend metrics (velocity, DOM, price growth)
- [ ] Identify saturation vs opportunity
- [ ] Create "momentum score" algorithm

**Phase 2C: Predictive Intelligence (Month 3)**
- [ ] Enhanced recommendation engine with trend data
- [ ] "Don't build X, build Y instead" logic
- [ ] Feature trend tracking and recommendations
- [ ] Subdivision life cycle classification

**Phase 2D: UI Enhancements (Month 4)**
- [ ] Trend charts (sales velocity over time)
- [ ] Saturation heatmaps
- [ ] "What's Hot / What's Not" dashboard
- [ ] Historical performance comparisons

### Success Metrics

**Phase 1 (Current):**
- ✅ Accurate current market snapshot
- ✅ Subdivision-level recommendations
- ✅ Feature impact analysis

**Phase 2 (Future):**
- 🎯 Predict market saturation 6-12 months ahead
- 🎯 Identify emerging configurations before competitors
- 🎯 Track prediction accuracy (% of recommendations that outperform)
- 🎯 ROI improvement: Recommendations that avoid oversaturated markets

### Data Requirements for Phase 2

**Current:**
- 500 properties per ZIP
- Most recent sale only
- 24-month lookback

**Phase 2:**
- 10,000+ properties per ZIP
- Full sales history per property (2004-2025)
- 5-10 year trend analysis
- Multi-ZIP comparison (5+ ZIPs in NC)

**Storage:**
- Current: ~10MB cached JSON per ZIP
- Phase 2: ~500MB+ per ZIP in PostgreSQL
- Total: ~2.5GB for 5 ZIPs (very manageable)

### Why This Matters

**Current Analysis:**
> "Build 4BR/3BA - it's selling"

**Phase 2 Analysis:**
> "DON'T build 4BR/3BA - everyone else is building it. Market is oversaturated (45 days DOM, down from 12 days in 2022). Build 3BR/2BA instead - underserved and accelerating. Your competition will flood the market with 4BR while you capture the growing 3BR segment."

**This is your competitive moat:** Knowing what NOT to build is as valuable as knowing what to build.

---

## Phase 3 (Future Future): Multi-Market Intelligence

- Expand beyond NC to multiple states
- Cross-market trend comparison
- National trend detection (features gaining traction in CA that haven't hit NC yet)
- Competitor tracking (what are other builders doing?)
- Economic indicator integration (interest rates, employment, etc.)

---

## Technical Debt & Improvements

### Current Known Issues
1. Hamilton Lakes shows $0 sales in some records (data quality from Attom)
2. Only analyzing properties with recent sales (missing established neighborhoods)
3. Cache invalidation strategy needs refinement
4. No automated data quality checks

### Planned Improvements
- Add data validation layer (flag suspicious prices, dates, sizes)
- Implement anomaly detection (outliers in pricing)
- Add confidence intervals to all predictions
- Create data quality dashboard

---

**Last Updated:** November 1, 2025
**Status:** Phase 1 Complete, Phase 2 Planning

