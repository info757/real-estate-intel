"""
Financial modeling with IRR calculations and SG&A optimization.
"""

import numpy as np
from scipy.optimize import newton
from typing import Optional, List
from datetime import datetime
import logging
from backend.models.schemas import ProjectFinancials, SGAMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialOptimizer:
    """Financial analysis and optimization tools."""
    
    def calculate_irr(self, cash_flows: List[float]) -> Optional[float]:
        """Calculate Internal Rate of Return."""
        try:
            # IRR is the rate where NPV = 0
            # Using scipy's newton method to find the root
            def npv(rate, cash_flows):
                return sum([cf / (1 + rate) ** i for i, cf in enumerate(cash_flows)])
            
            # Initial guess of 10%
            irr = newton(lambda r: npv(r, cash_flows), 0.1)
            return irr
        except:
            logger.warning("Could not calculate IRR")
            return None
    
    def calculate_npv(self, cash_flows: List[float], discount_rate: float) -> float:
        """Calculate Net Present Value."""
        npv = sum([cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows)])
        return npv
    
    def analyze_project(self, 
                        land_cost: float,
                        construction_cost: float,
                        carrying_costs: float,
                        other_costs: float,
                        projected_sale_price: float,
                        timeline_months: int,
                        discount_rate: float = None) -> ProjectFinancials:
        """Perform comprehensive financial analysis of a project."""
        
        if discount_rate is None:
            discount_rate = 0.12  # Default 12% annual
        
        # Convert monthly discount rate from annual
        monthly_rate = discount_rate / 12
        
        # Total investment
        total_investment = land_cost + construction_cost + carrying_costs + other_costs
        
        # Cash flows (monthly)
        # Month 0: Land acquisition
        # Months 1-N: Construction and carrying costs
        # Month N: Sale proceeds
        
        cash_flows = []
        cash_flows.append(-land_cost)  # Month 0
        
        # Spread construction and carrying costs over timeline
        monthly_expense = (construction_cost + carrying_costs + other_costs) / timeline_months
        for month in range(1, timeline_months):
            cash_flows.append(-monthly_expense)
        
        # Final month: last expense + sale proceeds
        cash_flows.append(-monthly_expense + projected_sale_price)
        
        # Calculate metrics
        gross_profit = projected_sale_price - total_investment
        gross_margin = (gross_profit / projected_sale_price * 100) if projected_sale_price > 0 else 0
        roi = (gross_profit / total_investment * 100) if total_investment > 0 else 0
        
        # IRR
        irr = self.calculate_irr(cash_flows)
        if irr:
            irr_annual = (1 + irr) ** 12 - 1  # Convert monthly to annual
        else:
            irr_annual = None
        
        # NPV
        npv = self.calculate_npv(cash_flows, monthly_rate)
        
        # Break-even analysis
        break_even_price = total_investment
        margin_of_safety = ((projected_sale_price - break_even_price) / projected_sale_price * 100) if projected_sale_price > 0 else 0
        
        return ProjectFinancials(
            land_cost=land_cost,
            construction_cost=construction_cost,
            carrying_costs=carrying_costs,
            other_costs=other_costs,
            total_investment=total_investment,
            build_time=timeline_months,
            total_timeline=timeline_months,
            projected_sale_price=projected_sale_price,
            gross_profit=gross_profit,
            gross_margin=gross_margin,
            irr=irr_annual,
            roi=roi,
            discount_rate=discount_rate,
            npv=npv,
            break_even_price=break_even_price,
            margin_of_safety=margin_of_safety,
            created_date=datetime.now()
        )
    
    def sensitivity_analysis(self, 
                            base_financials: ProjectFinancials,
                            sale_price_variations: List[float] = None) -> List[dict]:
        """Perform sensitivity analysis on sale price."""
        if sale_price_variations is None:
            # Default: -20%, -10%, 0%, +10%, +20%
            sale_price_variations = [-0.20, -0.10, 0.0, 0.10, 0.20]
        
        results = []
        base_price = base_financials.projected_sale_price
        
        for variation in sale_price_variations:
            adjusted_price = base_price * (1 + variation)
            
            # Recalculate with adjusted price
            adjusted = self.analyze_project(
                land_cost=base_financials.land_cost,
                construction_cost=base_financials.construction_cost,
                carrying_costs=base_financials.carrying_costs,
                other_costs=base_financials.other_costs,
                projected_sale_price=adjusted_price,
                timeline_months=base_financials.total_timeline,
                discount_rate=base_financials.discount_rate
            )
            
            results.append({
                "price_variation": f"{variation * 100:+.0f}%",
                "sale_price": adjusted_price,
                "gross_profit": adjusted.gross_profit,
                "roi": adjusted.roi,
                "irr": adjusted.irr,
                "npv": adjusted.npv
            })
        
        return results
    
    def calculate_sga_metrics(self,
                             period_start: datetime,
                             period_end: datetime,
                             land_acquisition_costs: float,
                             staff_costs: float,
                             technology_costs: float,
                             other_sga: float,
                             projects_completed: int,
                             manual_hours: float = 0,
                             automated_hours: float = 0) -> SGAMetrics:
        """Calculate SG&A and operational efficiency metrics."""
        
        total_sga = land_acquisition_costs + staff_costs + technology_costs + other_sga
        sga_per_project = total_sga / projects_completed if projects_completed > 0 else 0
        
        # Calculate automation savings
        # Assume manual labor costs $50/hour, automation is fixed cost
        total_hours = manual_hours + automated_hours
        if total_hours > 0:
            automation_percentage = automated_hours / total_hours
            # Estimated savings: (hours automated * hourly rate) - (technology costs allocated to automation)
            automation_savings = (automated_hours * 50) - (technology_costs * 0.5)  # Rough estimate
        else:
            automation_percentage = 0
            automation_savings = 0
        
        # Projects per staff (assume staff_costs / 60k = number of staff)
        estimated_staff = staff_costs / 60000 if staff_costs > 0 else 1
        projects_per_staff = projects_completed / estimated_staff
        
        return SGAMetrics(
            period_start=period_start,
            period_end=period_end,
            land_acquisition_costs=land_acquisition_costs,
            staff_costs=staff_costs,
            technology_costs=technology_costs,
            other_sga=other_sga,
            total_sga=total_sga,
            projects_completed=projects_completed,
            sga_per_project=sga_per_project,
            automation_savings=automation_savings if automation_savings > 0 else None,
            projects_per_staff=projects_per_staff,
            automation_percentage=automation_percentage
        )
    
    def optimize_for_target_irr(self, 
                                target_irr: float,
                                land_cost: float,
                                construction_cost_per_sqft: float,
                                house_sqft: int,
                                timeline_months: int) -> dict:
        """Calculate required sale price for target IRR."""
        
        construction_cost = house_sqft * construction_cost_per_sqft
        carrying_costs = 500 * timeline_months
        soft_costs = construction_cost * 0.10
        total_investment = land_cost + construction_cost + carrying_costs + soft_costs
        
        # Iteratively find the sale price that gives target IRR
        # Start with a reasonable assumption
        min_price = total_investment * 1.1  # At least 10% profit
        max_price = total_investment * 2.0  # Cap at 100% profit
        
        for _ in range(20):  # Max iterations
            mid_price = (min_price + max_price) / 2
            
            financials = self.analyze_project(
                land_cost=land_cost,
                construction_cost=construction_cost,
                carrying_costs=carrying_costs,
                other_costs=soft_costs,
                projected_sale_price=mid_price,
                timeline_months=timeline_months
            )
            
            if financials.irr is None:
                break
            
            if abs(financials.irr - target_irr) < 0.001:  # Close enough
                break
            elif financials.irr < target_irr:
                min_price = mid_price
            else:
                max_price = mid_price
        
        return {
            "target_irr": target_irr,
            "required_sale_price": mid_price,
            "price_per_sqft": mid_price / house_sqft,
            "total_investment": total_investment,
            "gross_profit": mid_price - total_investment
        }

