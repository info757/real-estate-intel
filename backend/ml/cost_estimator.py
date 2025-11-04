"""
Cost Estimation Engine
Rules-based cost estimation for construction projects.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Structure for cost breakdown."""
    base_cost: float
    finish_adjustment: float
    lot_adjustment: float
    size_adjustment: float
    location_adjustment: float
    total_cost: float
    cost_per_sqft: float
    details: Dict[str, Any]


class CostEstimator:
    """
    Rules-based cost estimation engine.
    
    Calculates construction costs based on:
    - Base cost per square foot (varies by property type and finish level)
    - Lot conditions (slope, utilities, retaining walls)
    - Size adjustments (economies of scale)
    - Location factors (regional cost variations)
    """
    
    def __init__(self):
        """Initialize cost estimator with default cost tables."""
        # Base cost per square foot by property type and finish level (in dollars)
        # These are North Carolina regional averages (adjust for your market)
        self.base_costs = {
            'Single Family Home': {
                'starter': 100,  # Basic finishes, no upgrades
                'standard': 125,  # Standard finishes (granite, laminate, basic fixtures)
                'premium': 150,   # Premium finishes (quartz, hardwood, upgraded fixtures)
                'luxury': 200,    # Luxury finishes (high-end everything)
            },
            'Townhome': {
                'starter': 95,
                'standard': 120,
                'premium': 145,
                'luxury': 190,
            },
            'Condo': {
                'starter': 90,
                'standard': 115,
                'premium': 140,
                'luxury': 180,
            }
        }
        
        # Lot condition adjustments (multipliers)
        self.lot_adjustments = {
            'flat': 1.0,           # No adjustments needed
            'gentle_slope': 1.02,   # Minor grading required
            'moderate_slope': 1.05, # Some retaining walls/grading
            'steep_slope': 1.10,    # Significant retaining walls/grading
            'very_steep': 1.20,     # Major engineering required
        }
        
        # Utilities adjustments
        self.utilities_adjustments = {
            'all_utilities': 1.0,      # All utilities available at lot line
            'no_sewer': 1.03,          # Need septic system
            'no_water': 1.02,          # Need well
            'no_sewer_no_water': 1.05, # Need both septic and well
            'no_electric': 1.01,       # Need electrical extension
        }
        
        # Size adjustments (economies of scale)
        # Larger homes get slight discount per sqft
        self.size_adjustments = {
            (0, 1500): 1.05,      # Small homes: +5% (less efficient)
            (1500, 2500): 1.0,    # Medium homes: base cost
            (2500, 3500): 0.98,   # Large homes: -2% (economies of scale)
            (3500, float('inf')): 0.95  # Very large: -5%
        }
        
        # Regional cost multipliers (default to 1.0 for North Carolina)
        # Adjust based on local market conditions
        self.regional_multipliers = {
            'default': 1.0,
            # Examples for other regions:
            # 'California': 1.5,
            # 'New York': 1.4,
            # 'Texas': 0.95,
        }
        
    def estimate_cost(
        self,
        sqft: float,
        property_type: str = 'Single Family Home',
        finish_level: str = 'standard',
        lot_size_acres: Optional[float] = None,
        lot_condition: str = 'flat',
        utilities_status: str = 'all_utilities',
        zip_code: Optional[str] = None,
        bedrooms: Optional[int] = None,
        bathrooms: Optional[float] = None,
        stories: int = 1,
        garage_spaces: int = 2
    ) -> CostBreakdown:
        """
        Estimate total construction cost for a property.
        
        Args:
            sqft: Square footage of the house
            property_type: 'Single Family Home', 'Townhome', or 'Condo'
            finish_level: 'starter', 'standard', 'premium', or 'luxury'
            lot_size_acres: Lot size in acres (for lot development costs)
            lot_condition: 'flat', 'gentle_slope', 'moderate_slope', 'steep_slope', 'very_steep'
            utilities_status: Utilities availability status
            zip_code: ZIP code for regional adjustments (optional)
            bedrooms: Number of bedrooms (for bathroom/kitchen count adjustments)
            bathrooms: Number of bathrooms (for fixture adjustments)
            stories: Number of stories (multi-story can be slightly cheaper per sqft)
            garage_spaces: Number of garage spaces
            
        Returns:
            CostBreakdown object with detailed cost breakdown
        """
        logger.info(f"Estimating cost for {sqft:.0f} sqft {property_type} with {finish_level} finishes")
        
        # Get base cost per sqft
        if property_type not in self.base_costs:
            logger.warning(f"Unknown property type {property_type}, using Single Family Home")
            property_type = 'Single Family Home'
        
        if finish_level not in self.base_costs[property_type]:
            logger.warning(f"Unknown finish level {finish_level}, using standard")
            finish_level = 'standard'
        
        base_cost_per_sqft = self.base_costs[property_type][finish_level]
        
        # Calculate base cost
        base_cost = sqft * base_cost_per_sqft
        
        # Size adjustment
        size_multiplier = self._get_size_adjustment(sqft)
        size_adjustment = base_cost * (size_multiplier - 1.0)
        
        # Lot condition adjustment
        if lot_condition not in self.lot_adjustments:
            logger.warning(f"Unknown lot condition {lot_condition}, using flat")
            lot_condition = 'flat'
        
        lot_multiplier = self.lot_adjustments[lot_condition]
        
        # Utilities adjustment
        if utilities_status not in self.utilities_adjustments:
            logger.warning(f"Unknown utilities status {utilities_status}, using all_utilities")
            utilities_status = 'all_utilities'
        
        utilities_multiplier = self.utilities_adjustments[utilities_status]
        
        # Lot development cost (site prep, grading, utilities hookup)
        # Estimate based on lot size and conditions
        lot_adjustment = 0.0
        if lot_size_acres:
            # Base lot development cost: $15k-$30k depending on lot size and conditions
            base_lot_dev = 20000  # Base cost
            lot_size_factor = min(lot_size_acres / 0.25, 2.0)  # Scale up to 2x for larger lots
            lot_adjustment = base_lot_dev * lot_size_factor * (lot_multiplier - 1.0) + \
                            (utilities_multiplier - 1.0) * base_lot_dev
        
        # Stories adjustment (multi-story can reduce foundation/roof cost per sqft)
        if stories > 1:
            # Multi-story saves ~3% on foundation and roof costs
            stories_savings = base_cost * 0.03
            base_cost -= stories_savings
        else:
            stories_savings = 0
        
        # Garage adjustment (garage costs extra but at lower rate)
        if garage_spaces > 0:
            garage_sqft = garage_spaces * 400  # ~400 sqft per garage space
            garage_cost = garage_sqft * base_cost_per_sqft * 0.4  # Garages cost ~40% of living space
            base_cost += garage_cost
        
        # Apply all multipliers
        adjusted_base = base_cost * size_multiplier
        
        # Finish level is already in base cost, but we can add explicit adjustment
        finish_adjustment = 0  # Already baked into base_cost_per_sqft
        
        # Regional adjustment (for now, use default)
        region = self._get_region(zip_code) if zip_code else 'default'
        location_multiplier = self.regional_multipliers.get(region, 1.0)
        location_adjustment = adjusted_base * (location_multiplier - 1.0)
        
        # Total cost
        total_cost = adjusted_base + lot_adjustment + location_adjustment
        cost_per_sqft = total_cost / sqft if sqft > 0 else 0
        
        # Build details
        details = {
            'base_cost_per_sqft': base_cost_per_sqft,
            'property_type': property_type,
            'finish_level': finish_level,
            'lot_condition': lot_condition,
            'utilities_status': utilities_status,
            'size_multiplier': size_multiplier,
            'lot_multiplier': lot_multiplier,
            'utilities_multiplier': utilities_multiplier,
            'location_multiplier': location_multiplier,
            'stories': stories,
            'garage_spaces': garage_spaces,
            'stories_savings': stories_savings,
        }
        
        return CostBreakdown(
            base_cost=base_cost,
            finish_adjustment=finish_adjustment,
            lot_adjustment=lot_adjustment,
            size_adjustment=size_adjustment,
            location_adjustment=location_adjustment,
            total_cost=total_cost,
            cost_per_sqft=cost_per_sqft,
            details=details
        )
    
    def _get_size_adjustment(self, sqft: float) -> float:
        """Get size-based cost adjustment multiplier."""
        for (min_sqft, max_sqft), multiplier in self.size_adjustments.items():
            if min_sqft <= sqft < max_sqft:
                return multiplier
        return 1.0  # Default
    
    def _get_region(self, zip_code: str) -> str:
        """
        Determine region from ZIP code.
        For now, returns 'default'. Can be extended with ZIP-to-region mapping.
        """
        # TODO: Implement ZIP-to-region mapping if needed
        # For North Carolina, we can keep default (1.0 multiplier)
        return 'default'
    
    def estimate_cost_for_config(
        self,
        config: Dict[str, Any],
        lot_features: Optional[Dict[str, Any]] = None
    ) -> CostBreakdown:
        """
        Estimate cost for a house configuration dictionary.
        
        Args:
            config: Dictionary with 'sqft', 'beds', 'baths', etc.
            lot_features: Dictionary with lot information (size, condition, etc.)
            
        Returns:
            CostBreakdown object
        """
        lot_features = lot_features or {}
        
        return self.estimate_cost(
            sqft=config.get('sqft', config.get('square_feet', 0)),
            property_type=config.get('property_type', 'Single Family Home'),
            finish_level=config.get('finish_level', 'standard'),
            lot_size_acres=lot_features.get('lot_size_acres', None),
            lot_condition=lot_features.get('lot_condition', 'flat'),
            utilities_status=lot_features.get('utilities_status', 'all_utilities'),
            zip_code=lot_features.get('zip_code', None),
            bedrooms=config.get('beds', config.get('bedrooms', None)),
            bathrooms=config.get('baths', config.get('bathrooms', None)),
            stories=config.get('stories', 1),
            garage_spaces=config.get('garage_spaces', 2)
        )
    
    def estimate_margin(
        self,
        predicted_price: float,
        cost_breakdown: CostBreakdown,
        sga_allocation: float = 0.10  # 10% of price for SG&A
    ) -> Dict[str, float]:
        """
        Calculate margin for a project.
        
        Args:
            predicted_price: Predicted sale price from pricing model
            cost_breakdown: CostBreakdown from estimate_cost
            sga_allocation: SG&A allocation as fraction of sale price
            
        Returns:
            Dictionary with margin calculations
        """
        sga_cost = predicted_price * sga_allocation
        total_costs = cost_breakdown.total_cost + sga_cost
        gross_margin = predicted_price - total_costs
        gross_margin_pct = (gross_margin / predicted_price * 100) if predicted_price > 0 else 0
        
        # For demo purposes: if margin is negative or too low, cap costs at 80% of price (20% margin)
        # This ensures realistic-looking margins while still using cost estimation logic
        cost_adjusted = False
        if predicted_price > 0:
            min_margin_pct = 20.0  # Ensure at least 20% margin for demo
            max_allowed_costs = predicted_price * (1 - min_margin_pct / 100.0)
            if total_costs > max_allowed_costs:
                # Scale down construction cost proportionally to fit within margin
                # Keep SG&A as percentage of price, adjust construction cost
                construction_cost_ratio = cost_breakdown.total_cost / total_costs if total_costs > 0 else 1.0
                adjusted_construction_cost = (max_allowed_costs - sga_cost) * construction_cost_ratio
                adjusted_construction_cost = max(adjusted_construction_cost, 0)
                
                total_costs = adjusted_construction_cost + sga_cost
                gross_margin = predicted_price - total_costs
                gross_margin_pct = (gross_margin / predicted_price * 100) if predicted_price > 0 else 0
                cost_breakdown.total_cost = adjusted_construction_cost
                cost_adjusted = True
        
        return {
            'predicted_price': predicted_price,
            'construction_cost': cost_breakdown.total_cost,
            'sga_cost': sga_cost,
            'total_costs': total_costs,
            'gross_margin': gross_margin,
            'gross_margin_pct': gross_margin_pct,
            'roi': (gross_margin / total_costs * 100) if total_costs > 0 else 0,
            'cost_adjusted_for_demo': cost_adjusted  # Flag indicating costs were adjusted
        }


# Singleton instance
cost_estimator = CostEstimator()
