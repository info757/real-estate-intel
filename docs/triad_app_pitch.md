# Triad Builder Intelligence: Fast Decisions, Faster Sales

## 1. Why Builders Care
- You need to decide **what to build**, **where to build it**, and **how fast it will move**.
- The Triad Builder Intelligence app gives you a single screen to answer all three, backed by fresh MLS data—not gut feel.
- You get clear narratives (“sell probability 45% today versus 77% last season”) so you can adjust price, specs, or cadence before the market does.

## 2. Plain-Language Model Stack
| Layer | What it Does | Why it Matters |
| --- | --- | --- |
| **Pricing model** | Compares your target spec to recent closed sales and estimates a price range | Sets realistic expectations and ensures your margin math starts on solid ground |
| **Demand model** | Tracks how fast similar homes actually went pending or sold, using inventory, relist history, and price positioning | Signals risk: “expect 45 days right now” so you can tweak incentives or timings |
| **Recommendation engine** | Combines pricing + demand and overlays neighborhood insights to suggest the spec (beds/baths/sqft) most likely to sell fast | Turns analytics into a build-ready plan with instant LLM explanations |

## 3. What the App is Seeing Now
- **Tougher absorption**: Triad is running cooler—model says ~45% fast-sale odds and ~48 DOM versus the old 77% / 35 DOM baseline. It’s detecting higher price positioning and more supply than last year.
- **Feature guidance**: The engine points to smaller 4BR/3BA footprints in Green Valley (≈2,500 sqft), priced mid-pack, as the sweet spot. It flags inventory trend ratios so you know if the ZIP is tightening or loosening.
- **Narratives you can use**: “Triad model shows 45% / 48 days; seasonality baseline was 77% / 35 days” becomes a talking point for investors, conveying both current caution and historical context.

## 4. Where We’re Going Next: Live Pipeline View
- Today’s model ingests the closed-sale universe (sold listings). It already reads inventory pressure via 30/90-day sold ratios.
- **Next upgrade: Current listings context**
  1. Scrape active & pending listings with the same feature schema (price, beds, DOM-so-far, status dates).
  2. Re-run the feature engineering to capture “days on market right now” and real-time price competition.
  3. Feed that into the recommendation engine’s `current_listings_context` so the probability instantly reflects the live pipeline, not just trailing sales.
- **Listing popularity signal**: Layer in portal engagement metrics (views, saves, shares per day) so the app can highlight specs attracting real-time buyer attention.
- **Payoff**: faster reaction to price wars, cleaner identification of niches with thin supply, and more confidence pitching investors (“we’re indexing off today’s actives, not just last quarter’s closings”).

## 5. How to Use This in Your Pitch
- Start with the value: “One screen to pick, price, and defend our next spec.”
- Show the model outputs vs. baseline to prove the app sees the slowdown others miss.
- Walk through the recommended spec, the probability/DOM, and the narrative to demonstrate defensibility.
- Close on the roadmap: “Next, we plug in the live pipeline so this stays real-time. The hooks are already in place—we just need to switch them on.”
