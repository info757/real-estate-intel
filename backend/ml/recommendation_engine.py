"""
Recommendation Engine
Combines pricing, demand, and cost models to generate optimal build recommendations.
"""

import datetime
import json
import logging
import math
from itertools import product
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.ml.pricing_model import PricingModel
from backend.ml.demand_model import DemandModel
from backend.ml.cost_estimator import cost_estimator, CostBreakdown
from backend.ml.feature_engineering import feature_engineer
try:
    from backend.ml.fast_seller_model import fast_seller_model
    from backend.ml.insights_generator import insights_generator
    FAST_SELLER_AVAILABLE = True
except ImportError:
    FAST_SELLER_AVAILABLE = False
    logger.warning("Fast-seller model modules not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Main recommendation engine that generates optimal build recommendations.
    
    Process:
    1. Generate candidate house configurations
    2. For each configuration:
       - Predict sale price
       - Predict demand (sell probability + DOM)
       - Estimate construction cost
       - Calculate margin
    3. Rank by margin, subject to demand constraints
    4. Return top recommendations with explanations
    """
    
    def __init__(
        self,
        min_sell_probability: float = 0.70,  # Must have ≥70% chance to sell within threshold
        max_dom: int = 90,  # Maximum acceptable days on market
        min_margin_pct: float = 0.0,  # Minimum margin % (0% default to allow placeholder costs)
        sga_allocation: float = 0.10  # 10% of price for SG&A
    ):
        """
        Initialize recommendation engine.
        
        Args:
            min_sell_probability: Minimum probability of selling within threshold (0-1)
            max_dom: Maximum acceptable days on market
            min_margin_pct: Minimum gross margin percentage
            sga_allocation: SG&A allocation as fraction of sale price
        """
        self.min_sell_probability = min_sell_probability
        self.max_dom = max_dom
        self.min_margin_pct = min_margin_pct
        self.sga_allocation = sga_allocation
        self._model_dir = Path("models/triad_latest")
        self._pricing_model_name = "triad_pricing_model"
        self._demand_model_name = "triad_demand_model"
        self._pricing_model: Optional[PricingModel] = None
        self._demand_model: Optional[DemandModel] = None
        self._new_build_cache_path = Path("data/cache/new_build_specs.json")
        self._new_build_cache: Optional[Dict[str, Any]] = None
        self._premium_subdivision_keys = {
            "new irving park",
            "irving park",
            "starmount forest",
            "sedgefield",
            "summerfield",
        }
        self._last_candidate_source: Optional[str] = None
        
    def generate_recommendations(
        self,
        lot_features: Dict[str, Any],
        property_type: str = 'Single Family Home',
        candidate_configs: Optional[List[Dict[str, Any]]] = None,
        historical_data: Optional[List[Dict[str, Any]]] = None,
        current_listings_context: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
        use_market_insights: bool = True
    ) -> Dict[str, Any]:
        """
        Generate optimal build recommendations for a specific lot.
        
        Args:
            lot_features: Dictionary with lot information:
                - zip_code: str
                - latitude: float
                - longitude: float
                - lot_size_acres: float (optional)
                - lot_condition: str (optional, default: 'flat')
                - utilities_status: str (optional, default: 'all_utilities')
                - subdivision: str (optional)
            property_type: 'Single Family Home', 'Townhome', or 'Condo'
            candidate_configs: Optional list of specific configurations to evaluate.
                             If None, generates candidates automatically.
            historical_data: Optional historical sales data for feature engineering context
            current_listings_context: Optional competitive context from listings scraper
            top_n: Number of top recommendations to return
            
        Returns:
            Dictionary with recommendations and metadata
        """
        logger.info(
            "Generating recommendations for lot in %s",
            lot_features.get("zip_code", "unknown ZIP"),
        )
        print(  # noqa: T201 - surfaced in Streamlit console
            f"[DEBUG] Starting recommendation generation for ZIP {lot_features.get('zip_code', 'unknown')}"
        )

        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        triad_model_dir = project_root / "models" / "triad_latest"
        self._model_dir = triad_model_dir

        pricing_model_path = triad_model_dir / f"{self._pricing_model_name}.pkl"
        demand_model_prob_path = triad_model_dir / f"{self._demand_model_name}_probability.pkl"
        demand_model_dom_path = triad_model_dir / f"{self._demand_model_name}_dom.pkl"

        missing: List[str] = []
        if not pricing_model_path.exists():
            missing.append(str(pricing_model_path))
        if not demand_model_prob_path.exists():
            missing.append(str(demand_model_prob_path))
        if not demand_model_dom_path.exists():
            missing.append(str(demand_model_dom_path))

        if missing:
            error_msg = (
                "Triad model artifacts not found. Missing files: " + ", ".join(missing)
            )
            logger.error(error_msg)
            return {
                "error": error_msg,
                "recommendations": [],
                "total_evaluated": 0,
            }

        # Lazily load Triad pricing/demand models
        if self._pricing_model is None:
            try:
                self._pricing_model = PricingModel(model_dir=str(triad_model_dir))
                self._pricing_model.load(model_name=self._pricing_model_name)
                logger.info("Loaded Triad pricing model from %s", triad_model_dir)
            except Exception as exc:  # pragma: no cover - defensive logging
                error_msg = f"Could not load Triad pricing model: {exc}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "recommendations": [],
                    "total_evaluated": 0,
                }

        if self._demand_model is None:
            try:
                self._demand_model = DemandModel(model_dir=str(triad_model_dir))
                self._demand_model.load(model_name=self._demand_model_name)
                logger.info("Loaded Triad demand model from %s", triad_model_dir)
            except Exception as exc:  # pragma: no cover - defensive logging
                error_msg = f"Could not load Triad demand model: {exc}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "recommendations": [],
                    "total_evaluated": 0,
                }

        # Load fast-seller model if available in Triad directory
        if use_market_insights and FAST_SELLER_AVAILABLE:
            try:
                fast_seller_model.model_dir = triad_model_dir
                fast_seller_model.load()
                logger.info("Fast-seller model loaded from %s", triad_model_dir)
            except Exception as exc:
                logger.warning(
                    "Fast-seller model unavailable in Triad directory (%s); "
                    "continuing without market insights.",
                    exc,
                )
                use_market_insights = False

        print("[DEBUG] Models loaded successfully, generating candidate configurations...")  # noqa: T201
        
        # Generate candidate configurations if not provided
        candidate_source = "provided" if candidate_configs is not None else None
        if candidate_configs is None:
            candidate_configs = self._generate_candidate_configs(
                property_type,
                lot_features,
                historical_data=historical_data,
            )
            candidate_source = self._last_candidate_source or "generated"
        else:
            self._last_candidate_source = "provided"
        
        logger.info(f"Evaluating {len(candidate_configs)} candidate configurations")
        
        # Evaluate each configuration
        results = []
        evaluation_errors = []
        for i, config in enumerate(candidate_configs):
            try:
                evaluation = self._evaluate_configuration(
                    config=config,
                    lot_features=lot_features,
                    property_type=property_type,
                    historical_data=historical_data,
                    current_listings_context=current_listings_context,
                    use_market_insights=use_market_insights
                )
                
                if evaluation:
                    results.append(evaluation)
                else:
                    # Log first few failures for debugging
                    if len(evaluation_errors) < 3:
                        evaluation_errors.append(f"Config {i+1}: {config.get('beds', '?')}BR/{config.get('baths', '?')}BA - evaluation returned None")
            except Exception as e:
                error_msg = f"Config {i+1}: {config.get('beds', '?')}BR/{config.get('baths', '?')}BA - {str(e)}"
                logger.warning(f"Error evaluating {error_msg}")
                if len(evaluation_errors) < 3:
                    evaluation_errors.append(error_msg)
                continue
        
        logger.info(f"Successfully evaluated {len(results)} configurations out of {len(candidate_configs)} candidates")
        
        if not results:
            error_msg = f'No valid configurations evaluated after trying {len(candidate_configs)} candidates.'
            if evaluation_errors:
                error_msg += f' Sample errors: {"; ".join(evaluation_errors)}'
            else:
                error_msg += ' All evaluations returned None - check logs for details.'
            return {
                'error': error_msg,
                'recommendations': [],
                'total_evaluated': len(candidate_configs),
                'successful_evaluations': 0
            }
        
        # Sort by combined score (margin + fast-seller probability if available)
        def get_combined_score(rec):
            margin_score = float(rec.get('margin', {}).get('gross_margin', 0.0) or 0.0)
            demand_block = rec.get('demand', {}) or {}
            sell_prob = demand_block.get('sell_probability')
            fast_prob = demand_block.get('fast_seller_probability')
            expected_dom = demand_block.get('expected_dom')
            config_sqft = self._coerce_float(rec.get('configuration', {}).get('sqft'))
            predicted_price = self._coerce_float(rec.get('predicted_price'))

            score = margin_score

            primary_prob = None
            if sell_prob is not None:
                primary_prob = float(sell_prob)
            elif fast_prob is not None:
                primary_prob = float(fast_prob)

            if primary_prob is not None:
                clipped_prob = min(max(primary_prob, 0.0), 1.0)
                score *= 0.5 + clipped_prob  # scale margin by demand velocity

            if expected_dom is not None:
                dom_value = float(expected_dom)
                score -= dom_value * 1000.0  # penalise slower absorption

            if predicted_price is not None:
                score += float(predicted_price) * 0.05

            if sqft_target is not None and config_sqft is not None and config_sqft < sqft_target:
                shortfall_ratio = (sqft_target - config_sqft) / max(sqft_target, 1.0)
                if shortfall_ratio > 0:
                    score -= shortfall_ratio * margin_score * 0.6

            return score

        sqft_target = None
        if historical_data:
            sqft_values: List[float] = []
            for listing in historical_data:
                sqft_val = self._coerce_float(
                    listing.get("sqft")
                    or listing.get("square_feet")
                    or (listing.get("summary") or {}).get("universalsize")
                    or (listing.get("summary") or {}).get("squareFeet")
                )
                if sqft_val is not None:
                    sqft_values.append(float(sqft_val))
            if sqft_values:
                percentile_value = 85 if len(sqft_values) >= 20 else 75
                sqft_target = float(np.percentile(sqft_values, percentile_value))

        filtered_results = sorted(results, key=get_combined_score, reverse=True)
        passing_count = sum(1 for rec in results if rec.get('passes_constraints'))
        
        # Get top N
        top_recommendations = filtered_results[:top_n]
        
        # Debug: Log unique configurations to verify they're different
        if top_recommendations:
            logger.info(f"Top {len(top_recommendations)} recommendations:")
            for i, rec in enumerate(top_recommendations, 1):
                config = rec.get('configuration', {})
                logger.info(f"  {i}. {config.get('beds', '?')}BR/{config.get('baths', '?')}BA, {config.get('sqft', '?')} sqft - Price: ${rec.get('predicted_price', 0):,.0f}, Margin: {rec.get('margin', {}).get('gross_margin_pct', 0):.1f}%")
        
        # Generate explanations and insights for top recommendations
        for rec in top_recommendations:
            rec['explanation'] = self._generate_explanation(rec, filtered_results)
            # Add fast-seller insights if available
            if use_market_insights and FAST_SELLER_AVAILABLE:
                try:
                    fast_prob = rec.get('demand', {}).get('fast_seller_probability', 0)
                    fast_dom = rec.get('demand', {}).get('fast_seller_dom', 0)
                    margin_pct = rec.get('margin', {}).get('gross_margin_pct', 0)
                    
                    # Get feature importance if available
                    feature_importance = None
                    if fast_seller_model.fast_seller_classifier is not None:
                        feature_importance = fast_seller_model.get_feature_importance('classifier')
                    
                    # Generate insights
                    if insights_generator:
                        rec['insights'] = insights_generator.generate_recommendation_insights(
                            configuration=rec['configuration'],
                            fast_seller_prob=fast_prob,
                            predicted_dom=fast_dom,
                            margin_pct=margin_pct,
                            feature_importance=feature_importance
                        )
                except Exception as e:
                    logger.warning(f"Failed to generate insights: {e}")
        
        return {
            'recommendations': top_recommendations,
            'total_evaluated': len(results),
            'total_passing_constraints': passing_count,
            'constraints': {
                'min_sell_probability': self.min_sell_probability,
                'max_dom': self.max_dom,
                'min_margin_pct': self.min_margin_pct
            },
            'lot_features': lot_features,
            'property_type': property_type,
            'candidate_source': candidate_source,
            'evaluation_errors': evaluation_errors,
        }
    
    def _generate_candidate_configs(
        self,
        property_type: str,
        lot_features: Dict[str, Any],
        *,
        historical_data: Optional[List[Dict[str, Any]]] = None,
        max_configs: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate house configurations to evaluate.

        Preference order:
        1. Use historical sold listings (Triad data) to derive real-world combos.
        2. Fall back to a static lattice when insufficient data is available.
        """
        data_configs = self._generate_configs_from_historical(
            property_type=property_type,
            lot_features=lot_features,
            historical_data=historical_data,
            max_configs=max_configs,
        )
        if data_configs:
            self._last_candidate_source = "historical"
            logger.info(
                "Generated %s data-driven candidate configurations for %s",
                len(data_configs),
                property_type,
            )
            return data_configs

        fallback_configs = self._generate_fallback_configs(
            property_type=property_type,
            lot_features=lot_features,
            max_configs=max_configs,
        )
        self._last_candidate_source = "fallback_lattice"
        logger.info(
            "Historical combos unavailable; using fallback lattice (%s configs)",
            len(fallback_configs),
        )
        return fallback_configs

    def _generate_configs_from_historical(
        self,
        *,
        property_type: str,
        lot_features: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]],
        max_configs: int,
    ) -> List[Dict[str, Any]]:
        subdivision_key: Optional[str] = None
        raw_subdivision = lot_features.get("subdivision")
        if raw_subdivision:
            subdivision_key = str(raw_subdivision).strip().lower()

        new_build_rows: List[Dict[str, Any]] = []
        legacy_rows: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []

        if historical_data:
            for listing in historical_data:
                label_text = self._extract_property_labels(listing)
                if not self._property_type_matches(property_type, label_text):
                    continue

                if subdivision_key:
                    listing_subdivision = (
                        listing.get("subdivision")
                        or (listing.get("summary") or {}).get("neighborhood", {}).get("name")
                        or (listing.get("summary") or {}).get("subdivision")
                        or (listing.get("property_detail_raw") or {}).get("neighborhood", {}).get("name")
                    )
                    if (
                        not listing_subdivision
                        or str(listing_subdivision).strip().lower() != subdivision_key
                    ):
                        continue

                year_built_value = (
                    listing.get("year_built")
                    or listing.get("yearBuilt")
                    or (listing.get("summary") or {}).get("yearBuilt")
                    or (listing.get("property_detail_raw") or {}).get("yearBuilt")
                    or (listing.get("metadata") or {}).get("year_built")
                )
                is_new_build = False
                if year_built_value is not None:
                    try:
                        year_numeric = int(float(year_built_value))
                        if year_numeric >= datetime.datetime.now().year - 3:
                            is_new_build = True
                    except Exception:
                        is_new_build = False

                beds = self._coerce_int(
                    listing.get("beds")
                    or (listing.get("summary") or {}).get("beds")
                    or (listing.get("summary") or {}).get("bedrooms")
                )
                baths = self._coerce_float(
                    listing.get("baths")
                    or listing.get("bathrooms")
                    or (listing.get("summary") or {}).get("baths")
                    or (listing.get("summary") or {}).get("bathstotal")
                )
                sqft = self._coerce_int(
                    listing.get("sqft")
                    or listing.get("square_feet")
                    or (listing.get("summary") or {}).get("universalsize")
                    or (listing.get("summary") or {}).get("squareFeet")
                )

                if beds is None or baths is None or sqft is None:
                    continue
                if beds <= 0 or sqft <= 700:
                    continue
                if beds > 7:
                    continue
                if baths < 2.0 or baths > beds + 1:
                    continue

                sqft_per_bed = sqft / max(beds, 1)
                if sqft_per_bed < 300 or sqft_per_bed > 1750:
                    continue
                sqft_per_bath = sqft / max(baths, 1.0)
                if sqft_per_bath < 400 or sqft_per_bath > 2000:
                    continue

                record = {
                    "beds": beds,
                    "baths": float(baths),
                    "sqft": sqft,
                }
                if is_new_build:
                    new_build_rows.append(record)
                else:
                    legacy_rows.append(record)

        if not (new_build_rows or legacy_rows):
            region_new_builds = self._load_regional_new_build_configs(property_type)
            if subdivision_key:
                subset = [row for row in region_new_builds if row.get("subdivision_key") == subdivision_key]
                if not subset:
                    subset = [
                        row
                        for row in region_new_builds
                        if row.get("subdivision_key") in self._premium_subdivision_keys
                    ]
                if subset:
                    new_build_rows.extend(
                        {"beds": row["beds"], "baths": row["baths"], "sqft": row["sqft"]}
                        for row in subset
                    )
                else:
                    new_build_rows.extend(
                        {"beds": row["beds"], "baths": row["baths"], "sqft": row["sqft"]}
                        for row in region_new_builds
                    )
            else:
                prioritized = [
                    {"beds": row["beds"], "baths": row["baths"], "sqft": row["sqft"]}
                    for row in region_new_builds
                    if row.get("subdivision_key") in self._premium_subdivision_keys
                ]
                if prioritized:
                    new_build_rows.extend(prioritized)
                else:
                    new_build_rows.extend(
                        {"beds": row["beds"], "baths": row["baths"], "sqft": row["sqft"]}
                        for row in region_new_builds
                    )

        rows = new_build_rows + legacy_rows
        if not rows:
            return []

        df = pd.DataFrame(rows)
        df = df[(df["beds"] > 0) & (df["sqft"] >= 700)]
        if df.empty:
            return []

        df["beds"] = df["beds"].astype(int)
        df["baths"] = df["baths"].round(1)
        df["sqft"] = df["sqft"].astype(int)

        grouped = df.groupby(["beds", "baths"])

        candidate_configs: List[Dict[str, Any]] = []
        seen: set[Tuple[int, float, int, str]] = set()

        summaries: List[Tuple[int, float, pd.Series]] = []
        for (beds, baths), group in grouped:
            if group.empty:
                continue
            summaries.append((beds, baths, group["sqft"]))

        # Sort by frequency (descending) so dominant combos appear first
        summaries.sort(key=lambda tup: len(tup[2]), reverse=True)

        for beds, baths, sqft_series in summaries:
            if len(candidate_configs) >= max_configs:
                break

            quantiles = sqft_series.quantile([0.25, 0.5, 0.75]).to_dict()
            sqft_candidates = {
                self._coerce_int(sqft_series.min()),
                self._coerce_int(quantiles.get(0.25)),
                self._coerce_int(quantiles.get(0.5)),
                self._coerce_int(quantiles.get(0.75)),
                self._coerce_int(sqft_series.max()),
            }
            sqft_candidates = {
                sqft for sqft in sqft_candidates if sqft is not None and sqft >= 700
            }

            if not sqft_candidates:
                continue

            for sqft in sorted(sqft_candidates):
                for finish in ("standard", "premium"):
                    key = (beds, float(baths), sqft, finish)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate_configs.append(
                        {
                            "beds": beds,
                            "baths": float(baths),
                            "sqft": sqft,
                            "finish_level": finish,
                            "property_type": property_type,
                            "stories": 1 if sqft < 2000 else 2,
                            "garage_spaces": 2
                            if property_type == "Single Family Home"
                            else 1,
                        }
                    )
                    if len(candidate_configs) >= max_configs:
                        break
                if len(candidate_configs) >= max_configs:
                    break

        return candidate_configs

    def _generate_fallback_configs(
        self,
        *,
        property_type: str,
        lot_features: Dict[str, Any],
        max_configs: int,
    ) -> List[Dict[str, Any]]:
        configs: List[Dict[str, Any]] = []

        if property_type == "Single Family Home":
            bed_range = [3, 4, 5]
            bath_range = [2.0, 2.5, 3.0, 3.5, 4.0]
            sqft_ranges = [
                (1500, 1800),
                (1800, 2200),
                (2200, 2600),
                (2600, 3000),
                (3000, 3600),
            ]
        elif property_type == "Townhome":
            bed_range = [2, 3, 4]
            bath_range = [2.0, 2.5, 3.0]
            sqft_ranges = [
                (1200, 1600),
                (1600, 2000),
                (2000, 2400),
            ]
        else:  # Condo or other
            bed_range = [1, 2, 3]
            bath_range = [1.0, 1.5, 2.0, 2.5]
            sqft_ranges = [
                (800, 1200),
                (1200, 1600),
                (1600, 2000),
            ]

        finish_levels = ["standard", "premium"]

        for beds, baths, (sqft_min, sqft_max), finish in product(
            bed_range, bath_range, sqft_ranges, finish_levels
        ):
            if baths > beds + 1:
                continue
            sqft = int((sqft_min + sqft_max) // 2)
            configs.append(
                {
                    "beds": beds,
                    "baths": baths,
                    "sqft": sqft,
                    "finish_level": finish,
                    "property_type": property_type,
                    "stories": 1 if sqft < 2000 else 2,
                    "garage_spaces": 2
                    if property_type == "Single Family Home"
                    else 1,
                }
            )
            if len(configs) >= max_configs:
                break
        return configs[:max_configs]

    def _load_regional_new_build_configs(self, property_type: str) -> List[Dict[str, Any]]:
        cache = self._load_new_build_cache()
        if not cache:
            return []
        property_key = property_type.lower()
        records: List[Dict[str, Any]] = []
        entries = cache.get(property_key, [])
        for entry in entries:
            beds = self._coerce_int(entry.get("beds"))
            baths = self._coerce_float(entry.get("baths"))
            sqft = self._coerce_int(entry.get("sqft"))
            if beds is None or baths is None or sqft is None:
                continue
            records.append(
                {
                    "beds": beds,
                    "baths": baths,
                    "sqft": sqft,
                    "subdivision_key": entry.get("subdivision_key"),
                }
            )
        return records

    def _load_new_build_cache(self) -> Dict[str, Any]:
        if self._new_build_cache is not None:
            return self._new_build_cache
        if self._new_build_cache_path.exists():
            try:
                with self._new_build_cache_path.open("r") as fh:
                    self._new_build_cache = json.load(fh)
            except Exception:
                logger.warning("Failed to load new build cache from %s", self._new_build_cache_path)
                self._new_build_cache = {}
        else:
            self._new_build_cache = {}
        return self._new_build_cache

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if math.isnan(number):
                return None
            return number
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            cleaned = cleaned.replace(",", "")
            try:
                number = float(cleaned)
                if math.isnan(number):
                    return None
                return number
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        num = RecommendationEngine._coerce_float(value)
        if num is None:
            return None
        rounded = int(round(num))
        if rounded <= 0:
            return None
        return rounded

    @staticmethod
    def _extract_sale_price(listing: Dict[str, Any]) -> Optional[float]:
        summary = listing.get("summary") or {}
        candidates = [
            listing.get("sale_price"),
            listing.get("price"),
            summary.get("mlsSoldPrice"),
            summary.get("lastSaleAmount"),
            summary.get("price"),
            summary.get("listingAmount"),
        ]
        for candidate in candidates:
            value = RecommendationEngine._coerce_float(candidate)
            if value is not None and value > 0:
                return value
        for history in listing.get("priceHistory") or []:
            history_price = RecommendationEngine._coerce_float((history or {}).get("price"))
            if history_price and history_price > 0:
                event = (history or {}).get("event")
                if isinstance(event, str) and event.lower() == "sold":
                    return history_price
        return None

    def _compute_psf_anchor(
        self,
        historical_data: Optional[List[Dict[str, Any]]],
        lot_features: Dict[str, Any],
        percentile: float = 85.0,
    ) -> Optional[float]:
        if not historical_data:
            return None
        psf_values: List[float] = []
        for listing in historical_data:
            sale_price = self._extract_sale_price(listing)
            sqft = self._coerce_float(
                listing.get("sqft")
                or listing.get("square_feet")
                or (listing.get("summary") or {}).get("universalsize")
                or (listing.get("summary") or {}).get("squareFeet")
            )
            if sale_price is None or sqft is None or sqft <= 0:
                continue
            psf = sale_price / sqft
            if psf > 0:
                psf_values.append(psf)
        if not psf_values:
            return None
        psf_array = np.array(psf_values, dtype=float)
        if psf_array.size == 0:
            return None
        percentile = max(0.0, min(100.0, percentile))
        return float(np.percentile(psf_array, percentile))

    @staticmethod
    def _extract_property_labels(listing: Dict[str, Any]) -> str:
        labels: List[str] = []
        metadata = listing.get("metadata") or {}
        for key in (
            "property_type_category",
            "property_type_label",
            "raw_property_use_code",
        ):
            val = metadata.get(key)
            if val:
                labels.append(str(val))

        for key in ("property_type", "propertyType"):
            val = listing.get(key)
            if val:
                labels.append(str(val))

        summary = listing.get("summary") or {}
        for key in ("propertyType", "propertyUse", "propertyUseType"):
            val = summary.get(key)
            if val:
                labels.append(str(val))

        return " ".join(labels).lower()

    @staticmethod
    def _property_type_matches(requested: str, label_text: str) -> bool:
        if not label_text:
            return True

        mapping = {
            "Single Family Home": ("single", "detached", "sfr", "residence"),
            "Townhome": ("town", "row", "attached"),
            "Condo": ("condo", "apartment", "flat"),
        }
        keywords = mapping.get(requested)
        if not keywords:
            return True
        return any(keyword in label_text for keyword in keywords)
    
    def _evaluate_configuration(
        self,
        config: Dict[str, Any],
        lot_features: Dict[str, Any],
        property_type: str,
        historical_data: Optional[List[Dict[str, Any]]] = None,
        current_listings_context: Optional[Dict[str, Any]] = None,
        use_market_insights: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single configuration.
        
        Returns:
            Evaluation dictionary or None if evaluation fails
        """
        try:
            if self._pricing_model is None or self._pricing_model.model is None:
                logger.error("Triad pricing model not available when evaluating configuration.")
                return None

            if (
                self._demand_model is None
                or self._demand_model.sell_probability_model is None
                or self._demand_model.dom_model is None
            ):
                logger.error("Triad demand model not available when evaluating configuration.")
                return None

            # 1. Create feature vector for ML models
            # Ensure property_type is in lot_features for feature vector creation
            lot_features_with_type = lot_features.copy()
            lot_features_with_type['property_type'] = property_type
            
            feature_vector = self._create_feature_vector(
                config=config,
                lot_features=lot_features_with_type,
                historical_data=historical_data,
                current_listings_context=current_listings_context
            )
            
            if feature_vector is None or feature_vector.empty:
                logger.warning(f"Feature vector is None or empty for config {config}")
                return None

            pricing_features = feature_vector
            demand_features = feature_vector

            if self._pricing_model.feature_names:
                pricing_features = feature_vector.reindex(
                    columns=self._pricing_model.feature_names,
                    fill_value=0.0,
                )

            if self._demand_model.feature_names:
                demand_features = feature_vector.reindex(
                    columns=self._demand_model.feature_names,
                    fill_value=0.0,
                )
            
            # 2. Predict price
            predicted_price = self._pricing_model.predict(pricing_features)[0]
            original_model_price = float(predicted_price)
            config_sqft = self._coerce_float(
                config.get('sqft')
                or config.get('square_feet')
                or config.get('universalsize')
            )
            psf_anchor = self._compute_psf_anchor(historical_data, lot_features)
            anchored_price = None
            if (
                psf_anchor is not None
                and config_sqft is not None
                and config_sqft > 0
            ):
                anchored_price = psf_anchor * config_sqft
                if anchored_price > 0:
                    anchor_weight = 0.6
                    predicted_price = (
                        (1 - anchor_weight) * predicted_price
                        + anchor_weight * anchored_price
                    )
            
            # 3. Predict demand
            demand_pred = self._demand_model.predict(demand_features)
            sell_probability = demand_pred['sell_probability'][0]
            expected_dom = demand_pred['expected_dom'][0]
            
            # 3b. Fast-seller model predictions (if using market insights)
            fast_seller_prob = sell_probability  # Default to demand model
            fast_seller_dom = expected_dom  # Default to demand model
            if use_market_insights and FAST_SELLER_AVAILABLE:
                try:
                    if fast_seller_model.fast_seller_classifier is not None:
                        fast_seller_prob = fast_seller_model.predict_fast_seller_probability(feature_vector)[0]
                        lot_zip = (
                            lot_features.get('zip_code')
                            or lot_features.get('postal_code')
                            or lot_features.get('zipCode')
                        )
                        fast_seller_dom = fast_seller_model.predict_dom(
                            feature_vector,
                            zip_codes=[lot_zip]
                        )[0]
                        logger.debug(f"Fast-seller prob: {fast_seller_prob:.3f}, DOM: {fast_seller_dom:.1f}")
                except Exception as e:
                    logger.warning(f"Fast-seller prediction failed: {e}")
            
            # 4. Estimate cost
            cost_breakdown = cost_estimator.estimate_cost_for_config(
                config=config,
                lot_features=lot_features
            )
            
            # 5. Calculate margin
            margin = cost_estimator.estimate_margin(
                predicted_price=predicted_price,
                cost_breakdown=cost_breakdown,
                sga_allocation=self.sga_allocation
            )
            
            # 6. Calculate risk score (lower is better) - use fast-seller if available
            effective_sell_prob = fast_seller_prob if (use_market_insights and FAST_SELLER_AVAILABLE) else sell_probability
            effective_dom = fast_seller_dom if (use_market_insights and FAST_SELLER_AVAILABLE) else expected_dom
            risk_score = self._calculate_risk_score(
                sell_probability=effective_sell_prob,
                expected_dom=effective_dom,
                margin_pct=margin['gross_margin_pct']
            )
            
            # Create a completely new dictionary with all values copied (deep copy for nested structures)
            import copy
            return {
                'configuration': copy.deepcopy(config),  # Deep copy config dict
                'predicted_price': float(predicted_price),
                'demand': {
                    'sell_probability': float(sell_probability),
                    'expected_dom': float(expected_dom),
                    'fast_seller_probability': float(fast_seller_prob),
                    'fast_seller_dom': float(fast_seller_dom),
                    'meets_demand_threshold': sell_probability >= self.min_sell_probability
                },
                'pricing_details': {
                    'model_price': original_model_price,
                    'psf_anchor': float(psf_anchor) if psf_anchor is not None else None,
                    'anchored_price': float(anchored_price) if anchored_price is not None else None,
                },
                'cost': {
                    'construction_cost': float(cost_breakdown.total_cost),
                    'cost_per_sqft': float(cost_breakdown.cost_per_sqft),
                    'cost_breakdown': {
                        'base_cost': float(cost_breakdown.base_cost),
                        'lot_adjustment': float(cost_breakdown.lot_adjustment),
                        'size_adjustment': float(cost_breakdown.size_adjustment),
                        'location_adjustment': float(cost_breakdown.location_adjustment),
                    }
                },
                'margin': copy.deepcopy(margin),  # Deep copy margin dict to avoid any reference issues
                'risk_score': float(risk_score),
                'passes_constraints': (
                    sell_probability >= self.min_sell_probability and
                    expected_dom <= self.max_dom and
                    margin['gross_margin_pct'] >= self.min_margin_pct
                )
            }
            
        except Exception as e:
            logger.error(f"Error evaluating configuration {config}: {e}")
            return None
    
    def _create_feature_vector(
        self,
        config: Dict[str, Any],
        lot_features: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None,
        current_listings_context: Optional[Dict[str, Any]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Create feature vector for ML models with all required features.
        """
        try:
            import pandas as pd
            import numpy as np
            from datetime import datetime
            
            # Basic features from config
            beds = config.get('beds', config.get('bedrooms', 3))
            baths = config.get('baths', config.get('bathrooms', 2.5))
            sqft = config.get('sqft', config.get('square_feet', 2000))
            lot_size_acres = lot_features.get('lot_size_acres', 0.25)
            lot_size_sqft = lot_size_acres * 43560  # Convert acres to sqft
            year_built = 2024  # New construction
            stories = config.get('stories', 1 if sqft < 2000 else 2)
            latitude = lot_features.get('latitude', 36.089)
            longitude = lot_features.get('longitude', -79.908)
            subdivision = lot_features.get('subdivision', '')
            property_type = lot_features.get('property_type', config.get('property_type', 'Single Family Home'))
            
            # Calculate market-based features from historical data if available
            # These are needed for price_per_sqft, price_per_bedroom, etc.
            market_price_per_sqft = 150.0  # Default fallback
            if historical_data:
                try:
                    from backend.ml.feature_engineering import feature_engineer
                    # Create a temporary DataFrame from historical data to calculate market stats
                    hist_df = feature_engineer.engineer_features(historical_data)
                    if not hist_df.empty and 'price_per_sqft' in hist_df.columns:
                        market_price_per_sqft = hist_df['price_per_sqft'].median()
                        if pd.isna(market_price_per_sqft) or market_price_per_sqft <= 0:
                            market_price_per_sqft = 150.0
                except Exception as e:
                    logger.warning(f"Could not calculate market stats from historical data: {e}")
            
            # Calculate subdivision size (number of properties in subdivision)
            subdivision_size = 50  # Default
            if historical_data and subdivision:
                try:
                    subdivision_size = len([p for p in historical_data 
                                          if self._safe_get(p, ['area', 'subdname']) == subdivision])
                    subdivision_size = max(subdivision_size, 10)  # Minimum 10
                except:
                    subdivision_size = 50
            
            # Finish level to quality score mapping
            finish_scores = {'starter': 4, 'standard': 5, 'premium': 7, 'luxury': 9}
            quality_score = finish_scores.get(config.get('finish_level', 'standard'), 5)
            
            # Property type indicators
            is_sfr = 1 if 'single' in property_type.lower() or property_type == 'SFR' else 0
            is_townhome = 1 if 'town' in property_type.lower() or 'townhouse' in property_type.lower() else 0
            is_condo = 1 if 'condo' in property_type.lower() else 0
            
            # Temporal features (using current date for new construction)
            now = datetime.now()
            age_years = 0  # New construction
            sale_year = now.year
            sale_month = now.month
            sale_quarter = (sale_month - 1) // 3 + 1
            is_spring_summer = 1 if sale_month in [3, 4, 5, 6, 7, 8] else 0
            days_since_sale = 0  # New construction, not sold yet
            
            # Calculate derived features
            price_per_sqft = market_price_per_sqft  # Will be adjusted by model, but need base value
            price_per_bedroom = price_per_sqft * sqft / beds if beds > 0 else 0
            price_per_bathroom = price_per_sqft * sqft / baths if baths > 0 else 0
            lot_coverage_ratio = sqft / lot_size_sqft if lot_size_sqft > 0 else 0
            beds_per_1000sqft = (beds / sqft) * 1000 if sqft > 0 else 0
            bath_bed_ratio = baths / beds if beds > 0 else 0
            condition_score = 5  # New construction = good condition
            overall_quality = quality_score  # Use quality_score as overall_quality
            
            # Create feature dictionary with ALL required features
            features = {
                'beds': beds,
                'baths': baths,
                'sqft': sqft,
                'lot_size_acres': lot_size_acres,
                'lot_size_sqft': lot_size_sqft,
                'year_built': year_built,
                'stories': stories,
                'latitude': latitude,
                'longitude': longitude,
                'price_per_sqft': price_per_sqft,
                'price_per_bedroom': price_per_bedroom,
                'price_per_bathroom': price_per_bathroom,
                'lot_coverage_ratio': lot_coverage_ratio,
                'beds_per_1000sqft': beds_per_1000sqft,
                'bath_bed_ratio': bath_bed_ratio,
                'age_years': age_years,
                'sale_year': sale_year,
                'sale_month': sale_month,
                'sale_quarter': sale_quarter,
                'is_spring_summer': is_spring_summer,
                'days_since_sale': days_since_sale,
                'is_sfr': is_sfr,
                'is_townhome': is_townhome,
                'is_condo': is_condo,
                'subdivision_size': subdivision_size,
                'quality_score': quality_score,
                'condition_score': condition_score,
                'overall_quality': overall_quality,
            }
            
            # Convert to DataFrame (single row)
            df = pd.DataFrame([features])
            
            # Ensure all numeric columns are numeric
            numeric_cols = [col for col in df.columns if col not in ['subdivision', 'proptype']]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill any NaN values with defaults
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _safe_get(self, obj: Any, keys: List[str]) -> Any:
        """Safely get nested dictionary value."""
        for key in keys:
            if isinstance(obj, dict):
                obj = obj.get(key, None)
            else:
                return None
            if obj is None:
                return None
        return obj
    
    def _calculate_risk_score(
        self,
        sell_probability: float,
        expected_dom: float,
        margin_pct: float
    ) -> float:
        """
        Calculate risk score (lower is better).
        
        Factors:
        - Lower sell probability = higher risk
        - Higher DOM = higher risk
        - Lower margin = higher risk
        """
        # Normalize components (0-1 scale, higher = worse)
        prob_risk = 1.0 - sell_probability  # Low prob = high risk
        dom_risk = min(expected_dom / 180.0, 1.0)  # Max DOM risk at 180 days
        margin_risk = max((15.0 - margin_pct) / 15.0, 0.0)  # Negative margin = max risk
        
        # Weighted combination
        risk_score = (0.4 * prob_risk) + (0.3 * dom_risk) + (0.3 * margin_risk)
        
        return float(risk_score)
    
    def _apply_constraints(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by constraints."""
        filtered = []
        for result in results:
            if result.get('passes_constraints', False):
                filtered.append(result)
        return filtered
    
    def _generate_explanation(
        self,
        recommendation: Dict[str, Any],
        all_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable explanation for a recommendation.
        
        In MVP, this is rule-based. In full version, LLM would be used.
        """
        config = recommendation['configuration']
        margin = recommendation['margin']
        demand = recommendation['demand']
        
        explanation_parts = []
        
        # Why this configuration
        explanation_parts.append(
            f"Recommended: {config['beds']}BR/{config['baths']}BA, "
            f"{config['sqft']:,} sqft, {config['finish_level']} finishes"
        )
        
        # Financial justification
        explanation_parts.append(
            f"Predicted sale price: ${margin['predicted_price']:,.0f} | "
            f"Construction cost: ${margin['construction_cost']:,.0f} | "
            f"Gross margin: ${margin['gross_margin']:,.0f} ({margin['gross_margin_pct']:.1f}%)"
        )
        
        # Demand justification
        explanation_parts.append(
            f"Demand: {demand['sell_probability']*100:.0f}% probability of selling within 90 days, "
            f"expected {demand['expected_dom']:.0f} days on market"
        )
        
        # Ranking context
        rank = all_results.index(recommendation) + 1
        if rank == 1:
            explanation_parts.append("Top recommendation based on highest margin while meeting demand constraints.")
        else:
            explanation_parts.append(f"Ranked #{rank} out of {len(all_results)} configurations.")
        
        return " | ".join(explanation_parts)


# Singleton instance
recommendation_engine = RecommendationEngine()
