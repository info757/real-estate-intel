"""
Demand Prediction Model
Predicts optimal bed/bath/size configurations based on sales velocity and market data.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from backend.data_collectors.attom_client import attom_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemandPredictor:
    """
    Predicts optimal house configurations based on:
    - Sales velocity (how fast homes sell)
    - Price per square foot
    - List-to-sale price ratio
    - Market inventory
    """
    
    def __init__(self):
        self.attom = attom_client
    
    def predict_optimal_config(
        self,
        zip_code: str,
        lot_size: Optional[float] = None,
        months_back: int = 12,
        min_samples: int = 5,
        subdivision: Optional[str] = None,
        radius_miles: Optional[float] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        property_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict the optimal house configuration for a ZIP code or micro-market.
        
        Args:
            zip_code: ZIP code to analyze
            lot_size: Lot size in acres (optional, for filtering)
            months_back: Months of historical data
            min_samples: Minimum samples for reliable prediction
            subdivision: Optional subdivision name to filter by
            radius_miles: Optional radius in miles (requires center_lat/center_lon)
            center_lat: Center latitude for radius search
            center_lon: Center longitude for radius search
            property_type: Optional property type filter ("SFR", "TOWNHOUSE/ROWHOUSE", "CONDOMINIUM", "ALL")
        
        Returns:
            Optimal configuration with supporting data
        """
        market_desc = f"ZIP {zip_code}"
        if subdivision:
            market_desc += f", subdivision '{subdivision}'"
        if radius_miles and center_lat and center_lon:
            market_desc += f", within {radius_miles} miles"
        
        logger.info(f"Predicting optimal configuration for {market_desc}")
        
        # Get sales data
        cutoff_date = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")
        properties = self.attom.get_all_sales_paginated(
            zip_code,
            max_pages=5,
            min_sale_date=cutoff_date
        )
        
        # Filter by subdivision if specified
        if subdivision and properties:
            properties = self._filter_by_subdivision(properties, subdivision)
            logger.info(f"Filtered to {len(properties)} properties in subdivision")
        
        # Filter by radius if specified
        if radius_miles and center_lat and center_lon and properties:
            properties = self._filter_by_radius(properties, center_lat, center_lon, radius_miles)
            logger.info(f"Filtered to {len(properties)} properties within {radius_miles} miles")
        
        # Filter by property type if specified
        if property_type and property_type != "ALL" and properties:
            properties = self._filter_by_property_type(properties, property_type)
            logger.info(f"Filtered to {len(properties)} properties of type {property_type}")
        
        if not properties or len(properties) < min_samples:
            logger.warning(f"Insufficient data for ZIP {zip_code}")
            return {
                "error": "Insufficient data",
                "property_count": len(properties) if properties else 0
            }
        
        logger.info(f"Analyzing {len(properties)} properties")
        
        # Analyze by configuration
        config_analysis = self._analyze_configurations(properties, min_samples)
        
        # Analyze by size brackets
        size_analysis = self._analyze_sizes(properties, min_samples)
        
        # Find optimal configuration
        optimal_config = self._determine_optimal(config_analysis, size_analysis)
        
        # Calculate market context
        market_context = self._calculate_market_context(properties)
        
        return {
            "zip_code": zip_code,
            "analysis_date": datetime.now().isoformat(),
            "property_count": len(properties),
            "months_analyzed": months_back,
            "optimal_config": optimal_config,
            "all_configurations": config_analysis,
            "size_analysis": size_analysis,
            "market_context": market_context
        }
    
    def _analyze_configurations(
        self,
        properties: List[Dict[str, Any]],
        min_samples: int
    ) -> List[Dict[str, Any]]:
        """Analyze performance by bed/bath configuration."""
        
        # Group by configuration
        configs = defaultdict(list)
        
        for prop in properties:
            try:
                building = prop.get('building', {})
                rooms = building.get('rooms', {})
                size_info = building.get('size', {})
                sale = prop.get('sale', {})
                
                beds = rooms.get('beds')
                baths_total = rooms.get('bathstotal')
                living_size = (size_info.get('livingsize') or 
                              size_info.get('universalsize') or 
                              size_info.get('bldgsize'))
                sale_amount = sale.get('amount', {}).get('saleamt')
                sale_date = sale.get('saleTransDate')
                
                if not all([beds, baths_total, living_size, sale_amount, sale_date]):
                    continue
                
                if sale_amount == 0 or living_size == 0:
                    continue
                
                config_key = f"{beds}BR/{baths_total:.1f}BA"
                
                configs[config_key].append({
                    'beds': beds,
                    'baths': baths_total,
                    'size': living_size,
                    'price': sale_amount,
                    'price_per_sqft': sale_amount / living_size,
                    'sale_date': sale_date
                })
                
            except Exception as e:
                logger.debug(f"Error processing property: {e}")
                continue
        
        # Analyze each configuration
        results = []
        
        for config_key, sales in configs.items():
            if len(sales) < min_samples:
                continue
            
            # Extract metrics
            prices = [s['price'] for s in sales]
            sizes = [s['size'] for s in sales]
            price_per_sqft = [s['price_per_sqft'] for s in sales]
            
            # Calculate sales velocity (sales per month)
            date_range_days = self._calculate_date_range(sales)
            sales_velocity = len(sales) / max(date_range_days / 30, 1)
            
            # Calculate demand score
            demand_score = self._calculate_demand_score(
                sales_count=len(sales),
                sales_velocity=sales_velocity,
                price_per_sqft=statistics.median(price_per_sqft),
                size=statistics.median(sizes)
            )
            
            results.append({
                "configuration": config_key,
                "bedrooms": sales[0]['beds'],
                "bathrooms": sales[0]['baths'],
                "sales_count": len(sales),
                "sales_velocity": sales_velocity,  # sales per month
                "median_price": statistics.median(prices),
                "avg_price": statistics.mean(prices),
                "median_size": statistics.median(sizes),
                "avg_size": statistics.mean(sizes),
                "median_price_per_sqft": statistics.median(price_per_sqft),
                "avg_price_per_sqft": statistics.mean(price_per_sqft),
                "demand_score": demand_score
            })
        
        # Sort by demand score
        results.sort(key=lambda x: x['demand_score'], reverse=True)
        
        return results
    
    def _analyze_sizes(
        self,
        properties: List[Dict[str, Any]],
        min_samples: int
    ) -> List[Dict[str, Any]]:
        """Analyze performance by size brackets."""
        
        # Define size brackets
        size_brackets = [
            (0, 1500, "<1500sf"),
            (1500, 2000, "1500-2000sf"),
            (2000, 2500, "2000-2500sf"),
            (2500, 3000, "2500-3000sf"),
            (3000, 10000, "3000sf+")
        ]
        
        bracket_data = {name: [] for _, _, name in size_brackets}
        
        for prop in properties:
            try:
                building = prop.get('building', {})
                size_info = building.get('size', {})
                sale = prop.get('sale', {})
                
                living_size = (size_info.get('livingsize') or 
                              size_info.get('universalsize') or 
                              size_info.get('bldgsize'))
                sale_amount = sale.get('amount', {}).get('saleamt')
                sale_date = sale.get('saleTransDate')
                
                if not all([living_size, sale_amount, sale_date]):
                    continue
                
                if sale_amount == 0 or living_size == 0:
                    continue
                
                # Find bracket
                for min_size, max_size, bracket_name in size_brackets:
                    if min_size <= living_size < max_size:
                        bracket_data[bracket_name].append({
                            'size': living_size,
                            'price': sale_amount,
                            'price_per_sqft': sale_amount / living_size,
                            'sale_date': sale_date
                        })
                        break
                        
            except Exception as e:
                logger.debug(f"Error processing property: {e}")
                continue
        
        # Analyze each bracket
        results = []
        
        for bracket_name, sales in bracket_data.items():
            if len(sales) < min_samples:
                continue
            
            prices = [s['price'] for s in sales]
            sizes = [s['size'] for s in sales]
            price_per_sqft = [s['price_per_sqft'] for s in sales]
            
            # Calculate sales velocity
            date_range_days = self._calculate_date_range(sales)
            sales_velocity = len(sales) / max(date_range_days / 30, 1)
            
            results.append({
                "size_bracket": bracket_name,
                "sales_count": len(sales),
                "sales_velocity": sales_velocity,
                "median_price": statistics.median(prices),
                "median_size": statistics.median(sizes),
                "median_price_per_sqft": statistics.median(price_per_sqft)
            })
        
        # Sort by sales velocity
        results.sort(key=lambda x: x['sales_velocity'], reverse=True)
        
        return results
    
    def _determine_optimal(
        self,
        config_analysis: List[Dict[str, Any]],
        size_analysis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Determine the optimal configuration."""
        
        if not config_analysis:
            return {"error": "No configurations analyzed"}
        
        # Best configuration = highest demand score
        best_config = config_analysis[0]
        
        # Find best size bracket
        best_size_bracket = size_analysis[0] if size_analysis else None
        
        # Generate confidence score
        confidence = self._calculate_confidence(
            best_config['sales_count'],
            best_config['sales_velocity']
        )
        
        # Generate rationale
        rationale = self._generate_rationale(
            best_config,
            config_analysis,
            size_analysis
        )
        
        return {
            "bedrooms": best_config['bedrooms'],
            "bathrooms": best_config['bathrooms'],
            "sqft": int(best_config['median_size']),
            "configuration": best_config['configuration'],
            "median_sale_price": best_config['median_price'],
            "median_price_per_sqft": best_config['median_price_per_sqft'],
            "sales_velocity": best_config['sales_velocity'],
            "sales_count_last_year": best_config['sales_count'],
            "demand_score": best_config['demand_score'],
            "confidence": confidence,
            "rationale": rationale
        }
    
    def _calculate_date_range(self, sales: List[Dict[str, Any]]) -> int:
        """Calculate date range in days."""
        dates = [s['sale_date'] for s in sales if s.get('sale_date')]
        
        if not dates or len(dates) < 2:
            return 365  # Default to 1 year
        
        dates.sort()
        earliest = datetime.strptime(dates[0], "%Y-%m-%d")
        latest = datetime.strptime(dates[-1], "%Y-%m-%d")
        
        return max((latest - earliest).days, 30)  # At least 30 days
    
    def _calculate_demand_score(
        self,
        sales_count: int,
        sales_velocity: float,
        price_per_sqft: float,
        size: float
    ) -> float:
        """
        Calculate demand score.
        Higher score = higher demand.
        
        Factors:
        - Sales velocity (weight: 40%)
        - Sales count (weight: 30%)
        - Price per sqft relative to size (weight: 30%)
        """
        # Velocity score (normalized, higher is better)
        velocity_score = min(sales_velocity * 5, 40)  # Max 40 points
        
        # Count score (normalized, higher is better)
        count_score = min(sales_count / 2, 30)  # Max 30 points
        
        # Value score (considers both price and size efficiency)
        # Prefer larger homes that still maintain good price/sqft
        value_score = min((size / 100) * (price_per_sqft / 10), 30)  # Max 30 points
        
        total_score = velocity_score + count_score + value_score
        
        return round(total_score, 2)
    
    def _calculate_confidence(self, sales_count: int, sales_velocity: float) -> float:
        """Calculate confidence score (0.0 to 1.0)."""
        
        # More sales = higher confidence
        count_factor = min(sales_count / 30, 1.0) * 0.6
        
        # Higher velocity = higher confidence
        velocity_factor = min(sales_velocity / 5, 1.0) * 0.4
        
        confidence = count_factor + velocity_factor
        
        return round(confidence, 2)
    
    def _generate_rationale(
        self,
        best_config: Dict[str, Any],
        all_configs: List[Dict[str, Any]],
        size_analysis: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable rationale."""
        
        rationale_parts = []
        
        # Sales velocity
        velocity = best_config['sales_velocity']
        rationale_parts.append(
            f"{best_config['configuration']} homes sell at {velocity:.1f} units/month"
        )
        
        # Comparison to other configs
        if len(all_configs) > 1:
            second_best = all_configs[1]
            velocity_diff = ((velocity - second_best['sales_velocity']) / 
                           second_best['sales_velocity'] * 100)
            
            if velocity_diff > 10:
                rationale_parts.append(
                    f"{velocity_diff:.0f}% faster than {second_best['configuration']}"
                )
        
        # Sales count
        sales_count = best_config['sales_count']
        rationale_parts.append(
            f"{sales_count} sales in past year"
        )
        
        # Price point
        median_price = best_config['median_price']
        rationale_parts.append(
            f"median price ${median_price:,.0f}"
        )
        
        return "; ".join(rationale_parts)
    
    def _calculate_market_context(self, properties: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall market context."""
        
        prices = []
        sizes = []
        price_per_sqft = []
        
        for prop in properties:
            try:
                building = prop.get('building', {})
                size_info = building.get('size', {})
                sale = prop.get('sale', {})
                
                living_size = (size_info.get('livingsize') or 
                              size_info.get('universalsize') or 
                              size_info.get('bldgsize'))
                sale_amount = sale.get('amount', {}).get('saleamt')
                
                if living_size and sale_amount and sale_amount > 0 and living_size > 0:
                    prices.append(sale_amount)
                    sizes.append(living_size)
                    price_per_sqft.append(sale_amount / living_size)
                    
            except Exception as e:
                continue
        
        if not prices:
            return {}
        
        return {
            "total_sales": len(prices),
            "median_price": statistics.median(prices),
            "avg_price": statistics.mean(prices),
            "median_size": statistics.median(sizes),
            "avg_size": statistics.mean(sizes),
            "median_price_per_sqft": statistics.median(price_per_sqft),
            "avg_price_per_sqft": statistics.mean(price_per_sqft)
        }
    
    def _filter_by_property_type(
        self,
        properties: List[Dict[str, Any]],
        property_type: str
    ) -> List[Dict[str, Any]]:
        """Filter properties by property type."""
        # Map UI-friendly names to Attom's values
        type_mapping = {
            "SFR": "SFR",
            "Single Family Home": "SFR",
            "TOWNHOUSE": "TOWNHOUSE/ROWHOUSE",
            "Townhome": "TOWNHOUSE/ROWHOUSE",
            "Townhouse": "TOWNHOUSE/ROWHOUSE",
            "CONDO": "CONDOMINIUM",
            "Condo": "CONDOMINIUM"
        }
        
        # Get Attom's value
        attom_value = type_mapping.get(property_type, property_type)
        
        return [
            p for p in properties
            if attom_value in p.get('summary', {}).get('proptype', '')
        ]
    
    def _filter_by_subdivision(
        self,
        properties: List[Dict[str, Any]],
        subdivision: str
    ) -> List[Dict[str, Any]]:
        """Filter properties by subdivision name (case-insensitive partial match)."""
        subdivision_upper = subdivision.upper()
        return [
            p for p in properties
            if subdivision_upper in p.get('area', {}).get('subdname', '').upper()
        ]
    
    def _filter_by_radius(
        self,
        properties: List[Dict[str, Any]],
        center_lat: float,
        center_lon: float,
        radius_miles: float
    ) -> List[Dict[str, Any]]:
        """Filter properties within radius of center point."""
        import math
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance in miles between two points."""
            R = 3959  # Earth's radius in miles
            
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            return R * c
        
        filtered = []
        for p in properties:
            try:
                lat = float(p.get('location', {}).get('latitude', 0))
                lon = float(p.get('location', {}).get('longitude', 0))
                
                if lat and lon:
                    distance = haversine_distance(center_lat, center_lon, lat, lon)
                    if distance <= radius_miles:
                        filtered.append(p)
            except (ValueError, TypeError):
                continue
        
        return filtered


# Global instance
demand_predictor = DemandPredictor()

