"""
Feature Engineering Module
Creates derived features for ML models from raw property and market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Transforms raw property data into ML-ready features.
    """
    
    def __init__(self):
        pass

    def _to_float(self, value: Any) -> Optional[float]:
        """Convert value to float if possible."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "").replace("$", "")
            if cleaned in {"", "NA", "N/A", "null"}:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _to_int(self, value: Any) -> Optional[int]:
        """Convert value to integer if possible."""
        float_value = self._to_float(value)
        if float_value is None:
            return None
        try:
            return int(float_value)
        except (TypeError, ValueError):
            return None

    def _extract_from_candidates(self, prop: Dict[str, Any], candidates: List[Any]) -> Any:
        """Return the first non-null value from a list of candidate lookups."""
        for candidate in candidates:
            result = None
            if callable(candidate):
                try:
                    result = candidate(prop)
                except Exception:
                    result = None
            elif isinstance(candidate, (list, tuple)):
                result = self._safe_get(prop, list(candidate))
            else:
                result = prop.get(candidate)
            if result not in (None, "", [], {}):
                return result
        return None

    def _extract_sale_price(self, prop: Dict[str, Any]) -> Optional[float]:
        """Extract sale price from multiple possible RealEstateApi / ATTOM fields."""
        summary = prop.get("summary") or {}
        detail = prop.get("property_detail_raw") or {}
        candidates = [
            ["sale", "amount", "saleamt"],
            "sale_price",
            "price",
            ["sale", "price"],
            ["sale", "salePrice"],
            ["sale", "listPrice"],
            ["summary", "salePrice"],
            ["summary", "price"],
            ["summary", "soldPrice"],
            ["summary", "lastSaleAmount"],
            ["summary", "mlsSoldPrice"],
            ["summary", "lastSalePrice"],
            lambda _: (summary.get("lastSale") or {}).get("amount"),
            lambda _: (summary.get("sale") or {}).get("amount"),
            ["property_detail_raw", "mlsSoldPrice"],
            ["property_detail_raw", "lastSalePrice"],
            ["property_detail_raw", "lastSaleAmount"],
            lambda _: (detail.get("lastSale") or {}).get("saleAmount"),
        ]
        value = self._extract_from_candidates(prop, candidates)
        return self._to_float(value)

    def _extract_date(self, prop: Dict[str, Any], candidates: List[Any]) -> Optional[str]:
        """Extract date-like string from candidates."""
        value = self._extract_from_candidates(prop, candidates)
        if isinstance(value, (str, bytes)):
            return value
        return None

    def _extract_numeric(self, prop: Dict[str, Any], candidates: List[Any]) -> Optional[float]:
        """Extract numeric value (float) from candidates."""
        value = self._extract_from_candidates(prop, candidates)
        return self._to_float(value)

    def _extract_text(self, prop: Dict[str, Any], candidates: List[Any]) -> Optional[str]:
        """Extract text value from candidates."""
        value = self._extract_from_candidates(prop, candidates)
        if value is None:
            return None
        return str(value)
    
    def engineer_features(
        self,
        properties: List[Dict[str, Any]],
        competitive_context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set from property data.
        
        Args:
            properties: List of property dictionaries (from Attom)
            competitive_context: Optional competitive context data (from listings scraper)
            
        Returns:
            DataFrame with engineered features
        """
        if not properties:
            return pd.DataFrame()
        
        # Extract sale_price and other nested fields before DataFrame creation
        # This is needed because pandas doesn't handle deeply nested dicts well
        extracted_data = []
        for prop in properties:
            row = {}
            # Extract sale_price from nested structure
            sale_price = self._extract_sale_price(prop)
            sale_date = self._extract_date(prop, [
                ['sale', 'saleTransDate'],
                'sale_date',
                'sold_date',
                ['summary', 'soldDate'],
                ['summary', 'lastSaleDate'],
                ['summary', 'mlsLastSaleDate'],
                ['summary', 'mlsLastStatusDate'],
                ['property_detail_raw', 'lastSaleDate'],
                ['property_detail_raw', 'mlsLastSaleDate'],
            ])
            
            # Extract other nested fields
            row['sale_price'] = sale_price
            row['sale_date'] = sale_date
            row['list_date'] = self._extract_date(prop, [
                'listing_date',
                'list_date',
                ['summary', 'listDate'],
                ['summary', 'mlsListingDate'],
                ['property_detail_raw', 'mlsListingDate'],
                ['property_detail_raw', 'listDate'],
            ])
            row['pending_date'] = self._extract_date(prop, [
                'pending_date',
                ['summary', 'pendingDate'],
                ['property_detail_raw', 'pendingDate'],
                ['property_detail_raw', 'mlsPendingDate'],
            ])
            row['dom_to_pending'] = self._extract_numeric(prop, [
                'dom_to_pending',
                ['summary', 'mlsDaysOnMarket'],
                ['property_detail_raw', 'mlsDaysOnMarket'],
                lambda p: (p.get('timeline') or {}).get('dom_to_pending'),
            ])
            row['dom_to_sold'] = self._extract_numeric(prop, [
                'dom_to_sold',
                lambda p: (p.get('timeline') or {}).get('dom_to_sold'),
            ])
            row['pending_to_sold'] = self._extract_numeric(prop, [
                'pending_to_sold',
                lambda p: (p.get('timeline') or {}).get('pending_to_sold'),
            ])
            
            # Flatten common fields
            row['beds'] = self._extract_numeric(prop, [
                'beds',
                ['summary', 'beds'],
                ['summary', 'bedrooms'],
                ['property_detail_raw', 'summary', 'beds'],
                ['property_detail_raw', 'propertyInfo', 'bedRooms'],
            ])
            row['baths'] = self._extract_numeric(prop, [
                'baths',
                ['summary', 'bathstotal'],
                ['summary', 'bathrooms'],
                ['property_detail_raw', 'summary', 'baths'],
                ['property_detail_raw', 'propertyInfo', 'bathRoomsFull'],
            ])
            row['sqft'] = self._extract_numeric(prop, [
                'sqft',
                ['summary', 'universalsize'],
                ['summary', 'squareFeet'],
                ['summary', 'universalsize'],
                ['property_detail_raw', 'summary', 'universalsize'],
                ['property_detail_raw', 'propertyInfo', 'squareFeet'],
            ])
            row['lot_size_acres'] = self._extract_numeric(prop, [
                'lot_size_acres',
                ['summary', 'lotSizeAcres'],
                ['property_detail_raw', 'lotInfo', 'lotSizeAcres'],
            ])
            row['lot_size_sqft'] = self._extract_numeric(prop, [
                'lot_size_sqft',
                ['summary', 'lotSquareFeet'],
                ['property_detail_raw', 'lotInfo', 'lotSquareFeet'],
            ])
            row['year_built'] = self._extract_numeric(prop, [
                'year_built',
                ['summary', 'yearbuilt'],
                ['summary', 'yearBuilt'],
                ['property_detail_raw', 'summary', 'yearBuilt'],
                ['property_detail_raw', 'propertyInfo', 'yearBuilt'],
            ])
            row['stories'] = self._extract_numeric(prop, [
                'stories',
                ['summary', 'stories'],
                ['property_detail_raw', 'summary', 'stories'],
                ['property_detail_raw', 'propertyInfo', 'stories'],
            ])
            row['zip_code'] = self._extract_text(prop, [
                'zip_code',
                'zipCode',
                ['address', 'zip'],
                ['address', 'postal1'],
                ['summary', 'postalCode'],
                ['summary', 'address', 'postalCode'],
            ])
            row['subdivision'] = self._extract_text(prop, [
                'subdivision',
                ['summary', 'subdivision'],
                lambda p: ((p.get('summary') or {}).get('neighborhood') or {}).get('name'),
                lambda p: ((p.get('property_detail_raw') or {}).get('neighborhood') or {}).get('name'),
            ])
            row['latitude'] = self._extract_numeric(prop, [
                'latitude',
                ['summary', 'latitude'],
                ['location', 'latitude'],
                ['geocode', 'lat'],
                lambda p: (p.get('address') or {}).get('latitude'),
            ])
            row['longitude'] = self._extract_numeric(prop, [
                'longitude',
                ['summary', 'longitude'],
                ['location', 'longitude'],
                ['geocode', 'lon'],
                ['geocode', 'lng'],
                lambda p: (p.get('address') or {}).get('longitude'),
            ])
            row['property_type'] = self._extract_text(prop, [
                'property_type',
                ['summary', 'propertyType'],
                ['property_detail_raw', 'propertyType'],
                ['summary', 'proptype'],
            ])
            
            # Store original property for any other extractions
            row['_original_property'] = prop
            
            extracted_data.append(row)
        
        df = pd.DataFrame(extracted_data)
        
        logger.info(f"Engineering features for {len(df)} properties")
        
        # Basic property features
        df = self._create_basic_features(df)
        
        # Derived ratio features
        df = self._create_ratio_features(df)
        
        # Temporal features
        df = self._create_temporal_features(df)
        
        # Location features
        df = self._create_location_features(df)
        
        # Market-normalized features
        df = self._create_zip_normalized_features(df)
        
        # Quality/condition features
        df = self._create_quality_features(df)
        
        # Competitive context features (if available)
        if competitive_context:
            df = self._add_competitive_context(df, competitive_context)
        
        logger.info(f"Created {len(df.columns)} features")
        
        return df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features from raw property data."""
        # Fields are already extracted before DataFrame creation, so just ensure they're numeric
        # Only extract from original property if field is missing
        basic_fields = {
            'beds': ['building', 'rooms', 'beds'],
            'baths': ['building', 'rooms', 'bathstotal'],
            'sqft': ['building', 'size', 'universalsize'],
            'lot_size_acres': ['lot', 'lotsize1'],
            'lot_size_sqft': ['lot', 'lotsize2'],
            'year_built': ['summary', 'yearbuilt'],
            'stories': ['building', 'summary', 'levels'],
            'property_type': ['summary', 'proptype'],
            'subdivision': ['area', 'subdname'],
            'zip_code': ['address', 'postal1'],
            'latitude': ['location', 'latitude'],
            'longitude': ['location', 'longitude'],
        }
        
        # Fill in any missing fields from original property
        for field, path in basic_fields.items():
            if field not in df.columns or df[field].isna().all():
                df[field] = df.apply(
                    lambda x: self._safe_get(x.get('_original_property', {}), path) if pd.isna(x.get(field)) else x.get(field),
                    axis=1
                )
        
        # Sale price and date (already extracted, but ensure they're there)
        if 'sale_price' not in df.columns or df['sale_price'].isna().all():
            df['sale_price'] = df.apply(
                lambda x: self._safe_get(x.get('_original_property', {}), ['sale', 'amount', 'saleamt']),
                axis=1
            )
        
        if 'sale_date' not in df.columns or df['sale_date'].isna().all():
            df['sale_date'] = df.apply(
                lambda x: self._safe_get(x.get('_original_property', {}), ['sale', 'saleTransDate']),
                axis=1
            )
        
        # Convert to numeric where appropriate
        numeric_cols = [
            'beds', 'baths', 'sqft', 'lot_size_acres', 'lot_size_sqft',
            'year_built', 'stories', 'sale_price', 'dom_to_pending',
            'dom_to_sold', 'pending_to_sold'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived ratio features."""
        # Price per square foot
        df['price_per_sqft'] = df['sale_price'] / df['sqft'].replace(0, np.nan)
        
        # Price per bedroom
        df['price_per_bedroom'] = df['sale_price'] / df['beds'].replace(0, np.nan)
        
        # Price per bathroom
        df['price_per_bathroom'] = df['sale_price'] / df['baths'].replace(0, np.nan)
        
        # Lot size ratio (house sqft / lot sqft)
        df['lot_coverage_ratio'] = df['sqft'] / df['lot_size_sqft'].replace(0, np.nan)
        
        # Bedrooms per 1000 sqft (density indicator)
        df['beds_per_1000sqft'] = (df['beds'] / df['sqft'].replace(0, np.nan)) * 1000
        
        # Bath to bed ratio
        df['bath_bed_ratio'] = df['baths'] / df['beds'].replace(0, np.nan)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        current_year = datetime.now().year

        # Age of property
        df['age_years'] = current_year - df['year_built']

        # Sale date features
        sale_dates = pd.to_datetime(df['sale_date'], errors='coerce', utc=True)
        df['sale_date_dt'] = sale_dates.dt.tz_localize(None)
        df['sale_year'] = df['sale_date_dt'].dt.year
        df['sale_month'] = df['sale_date_dt'].dt.month
        df['sale_quarter'] = df['sale_date_dt'].dt.quarter
        sale_month_angle = 2 * np.pi * ((df['sale_month'].fillna(1) - 1) / 12.0)
        df['sale_month_sin'] = np.sin(sale_month_angle)
        df['sale_month_cos'] = np.cos(sale_month_angle)
        sale_quarter_angle = 2 * np.pi * ((df['sale_quarter'].fillna(1) - 1) / 4.0)
        df['sale_quarter_sin'] = np.sin(sale_quarter_angle)
        df['sale_quarter_cos'] = np.cos(sale_quarter_angle)

        # Seasonality (spring selling season = higher prices?)
        df['is_spring_summer'] = df['sale_month'].isin([3, 4, 5, 6, 7, 8]).astype(int)

        # Days since sale (for temporal modeling)
        current_ts = pd.Timestamp.utcnow().tz_localize(None)
        df['days_since_sale'] = (current_ts - df['sale_date_dt']).dt.days

        return df
    
    def _create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features."""
        # Convert latitude/longitude to numeric
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # One-hot encode property type
        if 'property_type' in df.columns:
            df['is_sfr'] = df['property_type'].str.contains('SFR', case=False, na=False).astype(int)
            df['is_townhome'] = df['property_type'].str.contains('TOWNHOUSE|ROWHOUSE', case=False, na=False).astype(int)
            df['is_condo'] = df['property_type'].str.contains('CONDO', case=False, na=False).astype(int)
        
        # Subdivision frequency encoding (proxy for neighborhood quality/size)
        if 'subdivision' in df.columns:
            subdiv_counts = df['subdivision'].value_counts()
            df['subdivision_size'] = df['subdivision'].map(subdiv_counts).fillna(0)
        
        return df
    
    def _create_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create quality/condition indicator features."""
        # Extract quality and condition from nested data (using original property)
        df['quality'] = df.apply(lambda x: self._safe_get(x.get('_original_property', {}), ['building', 'summary', 'quality']), axis=1)
        df['condition'] = df.apply(lambda x: self._safe_get(x.get('_original_property', {}), ['building', 'construction', 'condition']), axis=1)
        
        # Convert quality to numeric score
        quality_map = {
            'EXCELLENT': 5,
            'VERY GOOD': 4,
            'GOOD': 3,
            'AVERAGE': 2,
            'FAIR': 1,
            'POOR': 0
        }
        df['quality_score'] = df['quality'].map(quality_map).fillna(2)  # Default to AVERAGE
        
        # Convert condition to numeric score
        condition_map = {
            'EXCELLENT': 5,
            'VERY GOOD': 4,
            'GOOD': 3,
            'AVERAGE': 2,
            'FAIR': 1,
            'POOR': 0
        }
        df['condition_score'] = df['condition'].map(condition_map).fillna(2)
        
        # Combined quality indicator
        df['overall_quality'] = (df['quality_score'] + df['condition_score']) / 2
        
        return df
    
    def _add_competitive_context(
        self,
        df: pd.DataFrame,
        competitive_context: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add competitive context features from listings data."""
        # Add competitive metrics as features
        df['active_listings_similar'] = competitive_context.get('active_listings_similar', 0)
        df['total_active_listings'] = competitive_context.get('total_active_listings', 0)
        df['price_percentile'] = competitive_context.get('price_percentile', 0.5)
        df['avg_dom_active'] = competitive_context.get('avg_dom_active', 0)
        df['price_reduction_rate'] = competitive_context.get('price_reduction_rate', 0)
        df['hot_home_percentage'] = competitive_context.get('hot_home_percentage', 0)
        
        # Encode inventory level
        inventory_map = {'low': 0, 'medium': 1, 'high': 2}
        df['inventory_level_encoded'] = inventory_map.get(
            competitive_context.get('inventory_level', 'medium'), 1
        )
        
        return df

    def _create_zip_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features normalized by ZIP-level market stats."""
        if 'zip_code' not in df.columns:
            return df
        
        zip_series = df['zip_code'].astype(str).str.extract(r'(\d{5})')[0]
        df['zip_code'] = zip_series.fillna(df['zip_code'])
        df['zip_numeric'] = pd.to_numeric(df['zip_code'], errors='coerce')
        
        # Median sale price per ZIP
        if 'sale_price' in df.columns:
            df['zip_sale_price_median'] = df.groupby('zip_code')['sale_price'].transform('median')
            df['price_to_zip_median'] = df['sale_price'] / df['zip_sale_price_median'].replace(0, np.nan)
            df['is_above_zip_median_price'] = (df['price_to_zip_median'] > 1.1).astype(float)
            df['is_below_zip_median_price'] = (df['price_to_zip_median'] < 0.9).astype(float)
        
        if 'price_per_sqft' in df.columns:
            df['zip_price_per_sqft_median'] = df.groupby('zip_code')['price_per_sqft'].transform('median')
            df['price_per_sqft_to_zip'] = df['price_per_sqft'] / df['zip_price_per_sqft_median'].replace(0, np.nan)
        
        if 'sqft' in df.columns:
            df['zip_sqft_median'] = df.groupby('zip_code')['sqft'].transform('median')
            df['sqft_to_zip_median'] = df['sqft'] / df['zip_sqft_median'].replace(0, np.nan)
        
        return df
    
    def _safe_get(self, obj: Any, keys: List[str]) -> Any:
        """Safely get nested dictionary value."""
        for key in keys:
            if isinstance(obj, dict):
                obj = obj.get(key, None)
            else:
                return None
        return obj
    def _create_market_context_features(
        self,
        df: pd.DataFrame,
        market_context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Add market context features for fast-seller prediction."""
        # Active inventory count (default to 0 if not available)
        df['active_inventory_count'] = market_context.get('active_inventory_count', 0) if market_context else 0
        
        # Median DOM in ZIP code
        df['median_dom_in_zip'] = market_context.get('median_dom_in_zip', 30) if market_context else 30
        
        # Price trend (1 = increasing, 0 = stable, -1 = decreasing)
        price_trend = market_context.get('price_trend_last_90_days', 0) if market_context else 0
        df['price_trend_last_90_days'] = price_trend
        
        # Subdivision sales velocity (days between sales in subdivision)
        df['subdivision_sales_velocity'] = market_context.get('subdivision_sales_velocity', 60) if market_context else 60
        
        return df
    
    def _create_pricing_strategy_features(
        self,
        df: pd.DataFrame,
        market_context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Add pricing strategy features."""
        # Price vs neighborhood median (as percentage)
        if 'sale_price' in df.columns and market_context:
            neighborhood_median = market_context.get('neighborhood_median_price', df['sale_price'].median())
            df['price_vs_neighborhood_median'] = (
                (df['sale_price'] - neighborhood_median) / neighborhood_median * 100
            ).fillna(0)
        else:
            df['price_vs_neighborhood_median'] = 0
        
        # Price ending pattern (last 3 digits)
        if 'sale_price' in df.columns:
            df['price_ending_pattern'] = (df['sale_price'] % 1000).fillna(0)
        else:
            df['price_ending_pattern'] = 0
        
        # Price per sqft vs market
        if 'price_per_sqft' in df.columns and market_context:
            market_price_per_sqft = market_context.get('market_price_per_sqft', df['price_per_sqft'].median())
            df['price_per_sqft_vs_market'] = (
                (df['price_per_sqft'] - market_price_per_sqft) / market_price_per_sqft * 100
            ).fillna(0)
        else:
            df['price_per_sqft_vs_market'] = 0
        
        return df
    
    def _create_timing_features(
        self,
        df: pd.DataFrame,
        date_column: str = 'sale_date'
    ) -> pd.DataFrame:
        """Add timing features (list day of week, month, season)."""
        if date_column not in df.columns:
            # Use current date as fallback
            df['list_day_of_week'] = datetime.now().weekday()
            df['list_month'] = datetime.now().month
            df['season'] = ((datetime.now().month - 1) // 3) + 1
            df['days_since_last_sale_in_subdivision'] = 0
            return df
        
        # Convert to datetime if string
        if df[date_column].dtype == 'object':
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Day of week (0 = Monday, 6 = Sunday)
        df['list_day_of_week'] = df[date_column].dt.dayofweek.fillna(0)
        
        # Month (1-12)
        df['list_month'] = df[date_column].dt.month.fillna(1)
        list_month_angle = 2 * np.pi * ((df['list_month'] - 1) / 12.0)
        df['list_month_sin'] = np.sin(list_month_angle)
        df['list_month_cos'] = np.cos(list_month_angle)
        list_dow_angle = 2 * np.pi * (df['list_day_of_week'] / 7.0)
        df['list_day_of_week_sin'] = np.sin(list_dow_angle)
        df['list_day_of_week_cos'] = np.cos(list_dow_angle)
        
        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        df['season'] = ((df[date_column].dt.month - 1) // 3 + 1).fillna(1)
        
        # Days since last sale in subdivision (simplified - would need grouping)
        df['days_since_last_sale_in_subdivision'] = 0  # Placeholder - would need subdivision grouping
        
        return df
    
    def _create_llm_extracted_features(
        self,
        df: pd.DataFrame,
        extracted_features_list: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """Add LLM-extracted features as binary flags."""
        if not extracted_features_list:
            return df
        
        # Get all unique features across all listings
        all_features = set()
        for features_dict in extracted_features_list:
            if isinstance(features_dict, dict):
                all_features.update(features_dict.get('interior', []))
                all_features.update(features_dict.get('exterior', []))
                all_features.update(features_dict.get('upgrades', []))
        
        # Create binary feature columns for each feature
        for feature in all_features:
            # Clean feature name for column name
            col_name = f"has_{feature.lower().replace(' ', '_').replace('-', '_')[:30]}"
            
            # Check if feature is present in each listing
            has_feature = []
            for features_dict in extracted_features_list:
                if isinstance(features_dict, dict):
                    feature_present = (
                        feature in features_dict.get('interior', []) or
                        feature in features_dict.get('exterior', []) or
                        feature in features_dict.get('upgrades', [])
                    )
                    has_feature.append(1 if feature_present else 0)
                else:
                    has_feature.append(0)
            
            # Add column if we have data
            if len(has_feature) == len(df):
                df[col_name] = has_feature
            else:
                # Pad with zeros if length mismatch
                df[col_name] = 0
        
        return df
    
    def engineer_features_for_fast_seller(
        self,
        listings: List[Dict[str, Any]],
        market_context: Optional[Dict[str, Any]] = None,
        extracted_features_list: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Engineer features specifically for fast-seller prediction model.
        
        Args:
            listings: List of listing dictionaries
            market_context: Market context data (inventory, DOM, trends)
            extracted_features_list: LLM-extracted features for each listing
            
        Returns:
            DataFrame with fast-seller model features
        """
        # Start with basic feature engineering
        df = self.engineer_features(listings, competitive_context=market_context)
        
        # Add fast-seller specific features
        df = self._create_market_context_features(df, market_context)
        df = self._create_pricing_strategy_features(df, market_context)
        df = self._create_timing_features(df, date_column='list_date')
        df = self._create_llm_extracted_features(df, extracted_features_list)
        
        # Remove leakage-prone target columns before modeling
        leakage_columns = [
            'dom_to_pending',
            'dom_to_sold',
            'pending_to_sold',
            'dom_relative_to_zip',
            'zip_dom_median',
            'zip_dom_p90'
        ]
        df = df.drop(columns=[col for col in leakage_columns if col in df.columns], errors='ignore')

        # Keep only numeric features for modeling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols].copy()
        
        return df


    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names that will be created.
        Useful for model training.
        """
        return [
            # Basic features
            'beds', 'baths', 'sqft', 'lot_size_acres', 'lot_size_sqft',
            'year_built', 'stories', 'latitude', 'longitude',
            
            # Ratio features
            'price_per_sqft', 'price_per_bedroom', 'price_per_bathroom',
            'lot_coverage_ratio', 'beds_per_1000sqft', 'bath_bed_ratio',
            
            # Temporal features
            'age_years', 'sale_year', 'sale_month', 'sale_quarter',
            'sale_month_sin', 'sale_month_cos', 'sale_quarter_sin', 'sale_quarter_cos',
            'is_spring_summer', 'days_since_sale',
            'list_month_sin', 'list_month_cos', 'list_day_of_week_sin', 'list_day_of_week_cos',
            
            # Location features
            'is_sfr', 'is_townhome', 'is_condo', 'subdivision_size',
            
            # Quality features
            'quality_score', 'condition_score', 'overall_quality',
            
            # Competitive context features (if available)
            'active_listings_similar', 'total_active_listings',
            'price_percentile', 'avg_dom_active', 'price_reduction_rate',
            'hot_home_percentage', 'inventory_level_encoded',
        ]
    
    def get_target_name(self) -> str:
        """Get name of target variable."""
        return 'sale_price'
    
    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_col: str = 'sale_price'
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare DataFrame for ML model training.
        
        Args:
            df: Engineered features DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (X features, y target)
        """
        # Select feature columns (drop target and non-feature columns)
        feature_cols = self.get_feature_names()
        
        # Filter to only columns that exist
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Remove rows with missing target
        df_clean = df[df[target_col].notna()].copy()
        
        # Separate features and target
        X = df_clean[available_features].copy()
        y = df_clean[target_col].copy()
        
        # Handle missing values in features
        X = X.fillna(X.median())
        
        logger.info(f"Prepared {len(X)} samples with {len(available_features)} features")
        
        return X, y


# Singleton instance
feature_engineer = FeatureEngineer()

