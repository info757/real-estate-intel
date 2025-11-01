"""
Feature Correlation Analyzer
Analyzes which interior and exterior features drive faster sales and higher prices.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics

from backend.data_collectors.attom_client import attom_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """
    Analyzes property features to determine which ones correlate with:
    - Faster sales (lower days on market - estimated from sale date vs list assumptions)
    - Higher sale prices
    - Better price per square foot
    """
    
    def __init__(self):
        self.attom = attom_client
    
    def analyze_feature_impact(
        self,
        zip_code: str,
        months_back: int = 12,
        min_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze feature impact for a ZIP code.
        
        Args:
            zip_code: ZIP code to analyze
            months_back: How many months of sales to analyze
            min_samples: Minimum number of samples for reliable analysis
        
        Returns:
            Feature impact analysis with scores and recommendations
        """
        logger.info(f"Analyzing features for ZIP {zip_code}")
        
        # Get recent sales data
        cutoff_date = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")
        properties = self.attom.get_all_sales_paginated(
            zip_code,
            max_pages=5,
            min_sale_date=cutoff_date
        )
        
        if not properties or len(properties) < min_samples:
            logger.warning(f"Insufficient data for ZIP {zip_code}: {len(properties) if properties else 0} properties")
            return {"error": "Insufficient data", "property_count": len(properties) if properties else 0}
        
        logger.info(f"Analyzing {len(properties)} properties")
        
        # Extract features and metrics
        feature_data = self._extract_features(properties)
        
        # Analyze interior features
        interior_analysis = self._analyze_feature_category(
            feature_data,
            "interior",
            min_samples
        )
        
        # Analyze exterior features
        exterior_analysis = self._analyze_feature_category(
            feature_data,
            "exterior",
            min_samples
        )
        
        # Analyze utilities/systems
        utilities_analysis = self._analyze_feature_category(
            feature_data,
            "utilities",
            min_samples
        )
        
        # Analyze configuration (beds/baths)
        config_analysis = self._analyze_configurations(feature_data, min_samples)
        
        # Overall market stats
        market_stats = self._calculate_market_stats(feature_data)
        
        return {
            "zip_code": zip_code,
            "analysis_date": datetime.now().isoformat(),
            "property_count": len(properties),
            "months_analyzed": months_back,
            "interior_features": interior_analysis,
            "exterior_features": exterior_analysis,
            "utilities": utilities_analysis,
            "configurations": config_analysis,
            "market_stats": market_stats
        }
    
    def _extract_features(self, properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relevant features and metrics from properties."""
        feature_data = []
        
        for prop in properties:
            try:
                sale = prop.get('sale', {})
                building = prop.get('building', {})
                summary = prop.get('summary', {})
                utilities = prop.get('utilities', {})
                
                # Skip if no sale data
                if not sale or 'amount' not in sale:
                    continue
                
                sale_amount = sale.get('amount', {}).get('saleamt')
                if not sale_amount or sale_amount == 0:
                    continue
                
                # Extract building details
                size_info = building.get('size', {})
                rooms_info = building.get('rooms', {})
                interior_info = building.get('interior', {})
                construction_info = building.get('construction', {})
                building_summary = building.get('summary', {})
                
                living_size = size_info.get('livingsize') or size_info.get('universalsize') or size_info.get('bldgsize')
                
                if not living_size or living_size == 0:
                    continue
                
                price_per_sqft = sale_amount / living_size
                
                # Extract features
                data = {
                    # Metrics
                    'sale_price': sale_amount,
                    'price_per_sqft': price_per_sqft,
                    'living_size': living_size,
                    
                    # Sale info
                    'sale_date': sale.get('saleTransDate'),
                    'sale_type': sale.get('amount', {}).get('saletranstype'),
                    
                    # Configuration
                    'beds': rooms_info.get('beds'),
                    'baths_full': rooms_info.get('bathsfull'),
                    'baths_total': rooms_info.get('bathstotal'),
                    
                    # Property type
                    'property_type': summary.get('proptype'),
                    'property_class': summary.get('propclass'),
                    'year_built': summary.get('yearbuilt'),
                    
                    # Interior features
                    'fireplace': interior_info.get('fplcind') == 'Y',
                    'floor_type': interior_info.get('floors'),
                    
                    # Exterior/Construction
                    'roof_type': construction_info.get('roofcover'),
                    'wall_type': construction_info.get('wallType'),
                    'construction_type': construction_info.get('constructiontype'),
                    'foundation_type': construction_info.get('foundationtype'),
                    'condition': construction_info.get('condition'),
                    
                    # Building summary
                    'quality': building_summary.get('quality'),
                    'levels': building_summary.get('levels'),
                    'arch_style': building_summary.get('archStyle'),
                    
                    # Utilities
                    'cooling_type': utilities.get('coolingtype'),
                    'heating_type': utilities.get('heatingtype'),
                    'heating_fuel': utilities.get('heatingfuel'),
                    
                    # Lot
                    'lot_size': prop.get('lot', {}).get('lotsize2'),  # sq ft
                    'pool': prop.get('lot', {}).get('pooltype') not in ['NO POOL', None, '']
                }
                
                feature_data.append(data)
                
            except Exception as e:
                logger.debug(f"Error extracting features: {e}")
                continue
        
        logger.info(f"Extracted features from {len(feature_data)} properties")
        return feature_data
    
    def _analyze_feature_category(
        self,
        feature_data: List[Dict[str, Any]],
        category: str,
        min_samples: int
    ) -> List[Dict[str, Any]]:
        """Analyze a category of features."""
        
        # Define feature mappings by category
        feature_fields = {
            "interior": [
                ('fireplace', 'Fireplace'),
                ('floor_type', 'Floor Type'),
            ],
            "exterior": [
                ('roof_type', 'Roof Type'),
                ('wall_type', 'Wall Type'),
                ('construction_type', 'Construction Type'),
                ('foundation_type', 'Foundation Type'),
                ('condition', 'Condition'),
            ],
            "utilities": [
                ('cooling_type', 'Cooling Type'),
                ('heating_type', 'Heating Type'),
                ('heating_fuel', 'Heating Fuel'),
            ]
        }
        
        if category not in feature_fields:
            return []
        
        results = []
        
        for field, display_name in feature_fields[category]:
            analysis = self._analyze_single_feature(
                feature_data,
                field,
                display_name,
                min_samples
            )
            if analysis:
                results.append(analysis)
        
        # Sort by impact score
        results.sort(key=lambda x: x.get('impact_score', 0), reverse=True)
        
        return results
    
    def _analyze_single_feature(
        self,
        feature_data: List[Dict[str, Any]],
        field: str,
        display_name: str,
        min_samples: int
    ) -> Optional[Dict[str, Any]]:
        """Analyze impact of a single feature."""
        
        # Separate properties with and without this feature
        with_feature = []
        without_feature = []
        feature_values = defaultdict(list)
        
        for data in feature_data:
            value = data.get(field)
            
            if value is None or value == '':
                without_feature.append(data)
            elif isinstance(value, bool):
                if value:
                    with_feature.append(data)
                else:
                    without_feature.append(data)
            else:
                # Categorical feature - group by value
                feature_values[str(value)].append(data)
        
        # For boolean features
        if with_feature and len(with_feature) >= min_samples:
            return self._compare_groups(
                display_name,
                "Yes",
                with_feature,
                without_feature if len(without_feature) >= min_samples else None
            )
        
        # For categorical features
        if feature_values:
            # Find most common values
            value_counts = {v: len(props) for v, props in feature_values.items()}
            if not value_counts:
                return None
            
            most_common_value = max(value_counts, key=value_counts.get)
            
            if value_counts[most_common_value] >= min_samples:
                other_props = []
                for v, props in feature_values.items():
                    if v != most_common_value:
                        other_props.extend(props)
                
                return self._compare_groups(
                    display_name,
                    most_common_value,
                    feature_values[most_common_value],
                    other_props if len(other_props) >= min_samples else None
                )
        
        return None
    
    def _compare_groups(
        self,
        feature_name: str,
        feature_value: str,
        group_with: List[Dict[str, Any]],
        group_without: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Compare properties with and without a feature."""
        
        # Calculate metrics for group with feature
        prices_with = [d['sale_price'] for d in group_with]
        price_per_sqft_with = [d['price_per_sqft'] for d in group_with]
        
        avg_price_with = statistics.mean(prices_with)
        avg_price_sqft_with = statistics.mean(price_per_sqft_with)
        frequency = len(group_with)
        
        # Calculate comparison if we have a control group
        price_premium = 0
        price_sqft_premium = 0
        frequency_pct = 0
        
        if group_without and len(group_without) >= 5:
            prices_without = [d['sale_price'] for d in group_without]
            price_per_sqft_without = [d['price_per_sqft'] for d in group_without]
            
            avg_price_without = statistics.mean(prices_without)
            avg_price_sqft_without = statistics.mean(price_per_sqft_without)
            
            price_premium = avg_price_with - avg_price_without
            price_sqft_premium = avg_price_sqft_with - avg_price_sqft_without
            frequency_pct = frequency / (frequency + len(group_without))
        else:
            # No control group - use market average
            frequency_pct = 1.0
        
        # Calculate impact score (higher is better)
        # Factors: price premium, frequency, sample size
        impact_score = 0
        if price_premium > 0:
            impact_score += (price_premium / 1000) * 0.5  # $1k premium = 0.5 points
        if price_sqft_premium > 0:
            impact_score += price_sqft_premium * 2  # $1/sqft premium = 2 points
        impact_score += frequency_pct * 20  # Frequency adds up to 20 points
        impact_score += min(frequency / 10, 10)  # Sample size adds up to 10 points
        
        # Determine priority
        if impact_score > 30 and frequency_pct > 0.6:
            priority = "must_have"
        elif impact_score > 15 or frequency_pct > 0.4:
            priority = "recommended"
        else:
            priority = "optional"
        
        # Generate rationale
        rationale_parts = []
        if frequency_pct > 0:
            rationale_parts.append(f"Present in {frequency_pct:.0%} of sales")
        if price_premium > 0:
            rationale_parts.append(f"adds ${price_premium:,.0f} to sale price")
        if price_sqft_premium > 0:
            rationale_parts.append(f"adds ${price_sqft_premium:.2f}/sqft")
        
        rationale = ", ".join(rationale_parts) if rationale_parts else "Limited data available"
        
        return {
            "feature": feature_name,
            "value": feature_value,
            "frequency": frequency,
            "frequency_pct": frequency_pct,
            "avg_sale_price": avg_price_with,
            "avg_price_per_sqft": avg_price_sqft_with,
            "price_premium": price_premium,
            "price_sqft_premium": price_sqft_premium,
            "impact_score": impact_score,
            "priority": priority,
            "rationale": rationale
        }
    
    def _analyze_configurations(
        self,
        feature_data: List[Dict[str, Any]],
        min_samples: int
    ) -> List[Dict[str, Any]]:
        """Analyze bed/bath configurations."""
        
        # Group by configuration
        configs = defaultdict(list)
        
        for data in feature_data:
            beds = data.get('beds')
            baths_total = data.get('baths_total')
            
            if beds and baths_total:
                config_key = f"{beds}BR/{baths_total:.1f}BA"
                configs[config_key].append(data)
        
        results = []
        
        for config, props in configs.items():
            if len(props) >= min_samples:
                prices = [p['sale_price'] for p in props]
                price_per_sqft = [p['price_per_sqft'] for p in props]
                sizes = [p['living_size'] for p in props]
                
                results.append({
                    "configuration": config,
                    "count": len(props),
                    "avg_sale_price": statistics.mean(prices),
                    "median_sale_price": statistics.median(prices),
                    "avg_price_per_sqft": statistics.mean(price_per_sqft),
                    "avg_size": statistics.mean(sizes),
                    "median_size": statistics.median(sizes)
                })
        
        # Sort by count (popularity)
        results.sort(key=lambda x: x['count'], reverse=True)
        
        return results
    
    def _calculate_market_stats(self, feature_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall market statistics."""
        
        if not feature_data:
            return {}
        
        prices = [d['sale_price'] for d in feature_data]
        price_per_sqft = [d['price_per_sqft'] for d in feature_data]
        sizes = [d['living_size'] for d in feature_data]
        
        return {
            "median_sale_price": statistics.median(prices),
            "avg_sale_price": statistics.mean(prices),
            "median_price_per_sqft": statistics.median(price_per_sqft),
            "avg_price_per_sqft": statistics.mean(price_per_sqft),
            "median_size": statistics.median(sizes),
            "avg_size": statistics.mean(sizes),
            "total_sales": len(feature_data)
        }
    
    def get_top_features(
        self,
        zip_code: str,
        price_range: Optional[Tuple[float, float]] = None,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top features for a given ZIP code and price range.
        
        Args:
            zip_code: ZIP code
            price_range: Optional (min, max) price tuple
            top_n: Number of top features to return
        
        Returns:
            List of top features with impact scores
        """
        analysis = self.analyze_feature_impact(zip_code)
        
        if 'error' in analysis:
            return []
        
        # Combine all features
        all_features = []
        all_features.extend(analysis.get('interior_features', []))
        all_features.extend(analysis.get('exterior_features', []))
        all_features.extend(analysis.get('utilities', []))
        
        # Filter by price range if specified
        if price_range:
            min_price, max_price = price_range
            all_features = [
                f for f in all_features
                if min_price <= f.get('avg_sale_price', 0) <= max_price
            ]
        
        # Sort by impact score
        all_features.sort(key=lambda x: x.get('impact_score', 0), reverse=True)
        
        return all_features[:top_n]


# Global instance
feature_analyzer = FeatureAnalyzer()

