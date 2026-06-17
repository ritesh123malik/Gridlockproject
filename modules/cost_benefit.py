"""
modules/cost_benefit.py
-----------------------
Cost-benefit analysis dashboard. Computes ROI, payback periods, and projected savings
by comparing active traffic delay reductions to the annual cost of the platform.
"""

class CostBenefitAnalyzer:
    @staticmethod
    def compute_roi(savings_per_week: float, system_cost_per_year: float) -> float:
        """
        Calculates ROI (%).
        Formula: ((Annual Savings - Platform Cost) / Platform Cost) * 100
        """
        annual_savings = savings_per_week * 52.0
        if system_cost_per_year <= 0:
            return 0.0
        roi = ((annual_savings - system_cost_per_year) / system_cost_per_year) * 100.0
        return round(roi, 1)

    @staticmethod
    def get_savings_projections(savings_per_week: float, system_cost_per_year: float) -> dict:
        """
        Computes weekly, monthly, and yearly savings and net returns.
        """
        weekly_savings = savings_per_week
        monthly_savings = savings_per_week * (52 / 12)
        annual_savings = savings_per_week * 52.0
        net_annual_benefit = annual_savings - system_cost_per_year
        roi = CostBenefitAnalyzer.compute_roi(savings_per_week, system_cost_per_year)
        
        # Payback period in months
        payback_months = (system_cost_per_year / (weekly_savings * (52 / 12))) if weekly_savings > 0 else 999.0
        
        return {
            "weekly_savings": round(weekly_savings, 2),
            "monthly_savings": round(monthly_savings, 2),
            "annual_savings": round(annual_savings, 2),
            "net_annual_benefit": round(net_annual_benefit, 2),
            "roi": roi,
            "payback_months": round(payback_months, 1)
        }
