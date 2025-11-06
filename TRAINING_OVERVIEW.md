# Fast-Seller Model Training Overview

## What We're Training

**Model Name:** Fast-Seller Prediction Model  
**Type:** Dual XGBoost Models
1. **Binary Classifier**: Predicts if a property will sell fast (DOM to pending ≤ 14 days)
2. **Regressor**: Predicts actual DOM to pending (continuous days)

## Training Data Source

**Primary Source:** Zillow API (via RapidAPI)
- **What we fetch:** Recently sold listings with full detail pages
- **Data points per listing:**
  - Property details: beds, baths, sqft, lot size, year built, stories
  - Pricing: list price, sale price, price history
  - Timeline: datePosted, pending date (from priceHistory), dateSold
  - Description: Full property description text (for LLM feature extraction)
  - Location: ZIP code, subdivision, lat/long

**Geographic Scope:** All 13 core Greensboro, NC ZIP codes
- 27401, 27402, 27403, 27404, 27405, 27406, 27407, 27408, 27409, 27410, 27411, 27412, 27455

**Time Period:** 24 months (730 days) of historical data

**Expected Sample Size:** 3,200-4,000 usable listings (after filtering)

## Target Variables

### 1. Fast-Seller Classification (Binary)
- **Target:** `is_fast_seller` (1 = fast, 0 = slow)
- **Definition:** DOM to pending ≤ 14 days = fast seller
- **Purpose:** Predict probability of selling quickly

### 2. DOM to Pending Regression (Continuous)
- **Target:** `dom_to_pending` (number of days)
- **Definition:** Days from listing date to pending date
- **Source:** Calculated from `datePosted` and `priceHistory` events
- **Purpose:** Predict expected time to pending

## Features Used for Training

### Basic Property Features
- Beds, baths, square footage
- Lot size (acres, sqft)
- Year built, stories, garage spaces
- Property type (SFR, townhome, condo)

### Pricing Features
- List price, sale price
- Price per sqft
- Price vs neighborhood median
- Price ending pattern

### Location Features
- ZIP code, subdivision
- Latitude, longitude
- Subdivision size
- School district (if available)

### Market Context Features
- Active inventory count
- Median DOM in ZIP
- Price trends (last 90 days)
- Subdivision sales velocity

### Timing Features
- List day of week
- List month, season
- Days since last sale in subdivision

### LLM-Extracted Features (Binary Flags)
- Interior features: granite, quartz, hardwood, tile, stainless steel, etc.
- Exterior features: deck, patio, pool, fencing, landscaping, etc.
- Upgrades: renovated kitchen, updated bathrooms, new roof, HVAC, etc.
- Condition: move-in ready, turnkey, updated, etc.

**Total Features:** ~50-100+ (depending on unique features found)

## Training Process

### Step 1: Data Collection (4-6 hours)
1. Fetch sold listings from Zillow API for each ZIP code
2. Fetch detail pages for each listing (to get priceHistory and descriptions)
3. Calculate DOM metrics from timeline data
4. Extract features from descriptions using GPT-4 LLM

### Step 2: Feature Engineering (30-60 min)
1. Create derived features (price per sqft, ratios, etc.)
2. Encode categorical features
3. Create binary flags for LLM-extracted features
4. Handle missing values

### Step 3: Model Training (45-90 min)
1. Split data: 60% train, 20% validation, 20% test
2. Train binary classifier (fast-seller prediction)
3. Train regressor (DOM prediction)
4. Hyperparameter tuning (if enabled)
5. Evaluate performance

### Step 4: Model Evaluation
- **Classifier Metrics:** AUC, Accuracy, Precision, Recall, F1
- **Regressor Metrics:** MAE, MAPE, R², RMSE

## Performance Expectations

**With 3,200-4,000 samples:**
- Classifier AUC: >0.75 (good), >0.80 (excellent)
- Regressor MAPE: <20% (good), <15% (excellent)
- Feature importance: Top features should make business sense

## Ways to Speed Up Training

### 1. Parallel Data Collection
- **Current:** Sequential ZIP code fetching
- **Optimization:** Fetch multiple ZIPs in parallel (threading/multiprocessing)
- **Speedup:** 2-3x faster (if API rate limits allow)

### 2. Batch LLM Feature Extraction
- **Current:** One listing at a time
- **Optimization:** Batch multiple listings in single LLM call
- **Speedup:** 5-10x faster, lower cost
- **Tradeoff:** Slightly less precise feature extraction

### 3. Skip Hyperparameter Tuning (for initial training)
- **Current:** Grid search over parameter space
- **Optimization:** Use default/known good parameters
- **Speedup:** 3-5x faster training
- **Tradeoff:** Slightly lower model performance

### 4. Cache API Responses
- **Current:** Fetch all data fresh
- **Optimization:** Cache fetched listings to disk
- **Speedup:** Instant on re-runs, protects against API failures

### 5. Incremental Training
- **Current:** Train on all data at once
- **Optimization:** Start with 12 months, add 12 months incrementally
- **Speedup:** Get working model faster, improve later
- **Tradeoff:** Two training runs needed

### 6. Reduce Feature Extraction Scope
- **Current:** Extract all features from all listings
- **Optimization:** Only extract features from fast sellers + sample of slow sellers
- **Speedup:** 50% fewer LLM calls
- **Tradeoff:** Less feature diversity

## Recommended Speed Optimizations

**For Demo Timeline (Priority Order):**

1. **Batch LLM Feature Extraction** (High impact, easy to implement)
   - Process 10-20 listings per LLM call
   - Saves 5-10x time and cost

2. **Cache API Responses** (High impact, easy to implement)
   - Save fetched listings to JSON files
   - Resume from cache if interrupted

3. **Skip Hyperparameter Tuning** (Medium impact, easy)
   - Use default XGBoost parameters
   - Can tune later if needed

4. **Parallel ZIP Fetching** (Medium impact, moderate complexity)
   - Use threading for API calls
   - Respect rate limits

5. **Incremental Training** (Fallback option)
   - Start with 12 months, expand to 24 if time allows

## Cost Estimates

**API Costs (24 months, 4,000 listings):**
- RapidAPI Zillow: ~$0.01-0.02 per listing = $40-80
- OpenAI GPT-4: ~$0.01-0.02 per listing = $40-80
- **Total: ~$80-160**

**With Batch LLM Extraction:**
- OpenAI GPT-4: ~$0.002-0.005 per listing = $8-20
- **Total: ~$48-100**

## Success Criteria

**Model is ready for demo if:**
- ✅ Classifier AUC > 0.70
- ✅ Regressor MAPE < 25%
- ✅ Feature importance makes business sense
- ✅ Predictions align with market intuition
- ✅ Model loads and runs in recommendation engine

## Next Steps

1. Review this overview - confirm we're aligned
2. Update training script with optimizations
3. Create monitoring script for progress tracking
4. Start training with Option 2 (24 months)
5. Monitor progress, have Option 1 as backup
