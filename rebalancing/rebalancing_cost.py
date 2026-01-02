# -*- coding: utf-8 -*-
"""
Rebalancing Cost Calculator

Calculates transaction costs and verifies cost efficiency.

File: portfolio/rebalancing/rebalancing_cost.py
Created: 2025-12-29
"""

import logging
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Tuple

from rebalancing.models import (
    SellCandidate,
    BuyCandidate,
    TransactionCost,
    CostEfficiencyResult,
    AnalysisResult,
    RebalancingAction,
)
from rebalancing.config import (
    KR_TRANSACTION_COST,
    US_TRANSACTION_COST,
    COST_EFFICIENCY_MULTIPLIER,
    ANNUALIZATION_FACTOR,
    TURNOVER_REFERENCE,
    calculate_total_cost,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RebalancingCostCalculator:
    """
    Calculates transaction costs and verifies cost efficiency.

    Cost components:
    - Commission (buy/sell)
    - Tax (sell only, KR)
    - Exchange fee (US only)
    - Slippage
    """

    def __init__(self, country: str):
        """
        Initialize cost calculator.

        Args:
            country: 'KR' or 'US'
        """
        self.country = country
        self.costs = KR_TRANSACTION_COST if country == "KR" else US_TRANSACTION_COST

    def calculate_sell_cost(
        self,
        amount: float,
        include_slippage: bool = True
    ) -> TransactionCost:
        """
        Calculate cost for selling.

        Args:
            amount: Sell amount
            include_slippage: Include slippage in calculation

        Returns:
            TransactionCost breakdown
        """
        if self.country == "KR":
            commission = amount * self.costs["sell_commission"]
            tax = amount * (self.costs["sell_tax"] + self.costs["sell_agri_tax"])
            exchange_fee = 0.0
            slippage = amount * self.costs["slippage_min"] if include_slippage else 0.0
        else:  # US
            commission = amount * self.costs["sell_commission"]
            tax = amount * self.costs["sec_fee"]
            exchange_fee = amount * self.costs["exchange_fee"]
            slippage = amount * self.costs["slippage_min"] if include_slippage else 0.0

        total = commission + tax + exchange_fee + slippage

        return TransactionCost(
            symbol="",  # To be filled by caller
            action="SELL",
            amount=amount,
            commission=commission,
            tax=tax,
            exchange_fee=exchange_fee,
            slippage=slippage,
            total_cost=total
        )

    def calculate_buy_cost(
        self,
        amount: float,
        include_slippage: bool = True
    ) -> TransactionCost:
        """
        Calculate cost for buying.

        Args:
            amount: Buy amount
            include_slippage: Include slippage in calculation

        Returns:
            TransactionCost breakdown
        """
        if self.country == "KR":
            commission = amount * self.costs["buy_commission"]
            tax = 0.0
            exchange_fee = 0.0
            slippage = amount * self.costs["slippage_min"] if include_slippage else 0.0
        else:  # US
            commission = amount * self.costs["buy_commission"]
            tax = 0.0
            exchange_fee = amount * self.costs["exchange_fee"]
            slippage = amount * self.costs["slippage_min"] if include_slippage else 0.0

        total = commission + tax + exchange_fee + slippage

        return TransactionCost(
            symbol="",
            action="BUY",
            amount=amount,
            commission=commission,
            tax=tax,
            exchange_fee=exchange_fee,
            slippage=slippage,
            total_cost=total
        )

    def calculate_total_rebalancing_cost(
        self,
        sell_candidates: List[SellCandidate],
        buy_candidates: List[BuyCandidate]
    ) -> Tuple[float, List[TransactionCost]]:
        """
        Calculate total cost for all rebalancing transactions.

        Args:
            sell_candidates: List of sell candidates
            buy_candidates: List of buy candidates

        Returns:
            Tuple of (total_cost, list of cost breakdowns)
        """
        costs = []
        total = 0.0

        # Sell costs
        for s in sell_candidates:
            cost = self.calculate_sell_cost(s.expected_amount)
            cost.symbol = s.symbol
            costs.append(cost)
            total += cost.total_cost

        # Buy costs
        for b in buy_candidates:
            cost = self.calculate_buy_cost(b.expected_amount)
            cost.symbol = b.symbol
            costs.append(cost)
            total += cost.total_cost

        return total, costs

    def calculate_expected_improvement(
        self,
        sell_candidates: List[SellCandidate],
        buy_candidates: List[BuyCandidate],
        old_scores: Dict[str, float],
        new_scores: Dict[str, float]
    ) -> float:
        """
        Calculate expected improvement from rebalancing.

        Expected Improvement = Sum of (new_score - old_score) * amount / 100 * annualization

        Args:
            sell_candidates: Stocks to sell
            buy_candidates: Stocks to buy
            old_scores: Current stock scores (symbol -> score)
            new_scores: New stock scores (symbol -> score)

        Returns:
            Expected improvement amount
        """
        improvement = 0.0

        # Improvement from selling low-score stocks
        for s in sell_candidates:
            old_score = old_scores.get(s.symbol, 50)
            # Assume we're replacing with average new stock score
            avg_new_score = sum(new_scores.values()) / len(new_scores) if new_scores else 50
            score_diff = avg_new_score - old_score

            # Convert score difference to expected return (simplified)
            # Assume 1 point = 0.5% annual return
            expected_return_diff = score_diff * 0.005
            improvement += s.expected_amount * expected_return_diff

        # Improvement from buying high-score stocks
        for b in buy_candidates:
            new_score = new_scores.get(b.symbol, 50)
            # Compare to portfolio average
            avg_old_score = sum(old_scores.values()) / len(old_scores) if old_scores else 50
            score_diff = new_score - avg_old_score

            expected_return_diff = score_diff * 0.005
            improvement += b.expected_amount * expected_return_diff

        return improvement

    def verify_cost_efficiency(
        self,
        analysis_result: AnalysisResult,
        old_scores: Dict[str, float],
        new_scores: Dict[str, float],
        force_execute: bool = False
    ) -> CostEfficiencyResult:
        """
        Verify if rebalancing is cost efficient.

        Rule: Expected Improvement > Total Cost x COST_EFFICIENCY_MULTIPLIER

        Args:
            analysis_result: Analysis result with candidates
            old_scores: Current stock scores
            new_scores: New stock scores
            force_execute: Skip efficiency check

        Returns:
            CostEfficiencyResult
        """
        # Calculate total cost
        total_cost, cost_breakdown = self.calculate_total_rebalancing_cost(
            analysis_result.sell_candidates,
            analysis_result.buy_candidates
        )

        # Calculate expected improvement
        expected_improvement = self.calculate_expected_improvement(
            analysis_result.sell_candidates,
            analysis_result.buy_candidates,
            old_scores,
            new_scores
        )

        # Check efficiency
        min_required_ratio = COST_EFFICIENCY_MULTIPLIER
        if total_cost > 0:
            cost_ratio = expected_improvement / total_cost
        else:
            cost_ratio = float('inf') if expected_improvement > 0 else 0

        is_efficient = force_execute or (expected_improvement > total_cost * min_required_ratio)

        return CostEfficiencyResult(
            is_efficient=is_efficient,
            expected_improvement=expected_improvement,
            total_cost=total_cost,
            cost_ratio=cost_ratio,
            min_required_ratio=min_required_ratio,
            details={
                "sell_count": len(analysis_result.sell_candidates),
                "buy_count": len(analysis_result.buy_candidates),
                "total_sell_amount": analysis_result.total_sell_amount,
                "total_buy_amount": analysis_result.total_buy_amount,
                "cost_breakdown": [
                    {
                        "symbol": c.symbol,
                        "action": c.action,
                        "amount": c.amount,
                        "cost": c.total_cost
                    }
                    for c in cost_breakdown
                ]
            }
        )

    def calculate_turnover(
        self,
        sell_amount: float,
        portfolio_value: float,
        period_days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate portfolio turnover rate.

        Turnover = (Sell Amount / Portfolio Value) * (365 / period_days) * 100

        Args:
            sell_amount: Total sell amount in period
            portfolio_value: Average portfolio value
            period_days: Period in days

        Returns:
            Dict with turnover metrics
        """
        if portfolio_value <= 0:
            return {
                "period_turnover_pct": 0,
                "annualized_turnover_pct": 0,
                "period_days": period_days
            }

        period_turnover = (sell_amount / portfolio_value) * 100
        annualized_turnover = period_turnover * (365 / period_days)

        return {
            "period_turnover_pct": round(period_turnover, 2),
            "annualized_turnover_pct": round(annualized_turnover, 2),
            "period_days": period_days
        }

    def get_turnover_status(
        self,
        annualized_turnover: float,
        risk_level: str
    ) -> Dict[str, Any]:
        """
        Get turnover status compared to reference.

        Args:
            annualized_turnover: Annualized turnover percentage
            risk_level: Portfolio risk level

        Returns:
            Status dict with reference comparison
        """
        reference = TURNOVER_REFERENCE.get(risk_level, 200)

        if annualized_turnover <= reference * 0.8:
            status = "LOW"
            message = "Turnover is below reference range"
        elif annualized_turnover <= reference * 1.2:
            status = "NORMAL"
            message = "Turnover is within reference range"
        else:
            status = "HIGH"
            message = "Turnover exceeds reference range"

        return {
            "status": status,
            "message": message,
            "annualized_turnover": annualized_turnover,
            "reference": reference,
            "ratio_to_reference": round(annualized_turnover / reference, 2) if reference > 0 else 0
        }

    def estimate_net_proceeds(
        self,
        gross_amount: float,
        action: str
    ) -> float:
        """
        Estimate net proceeds after costs.

        Args:
            gross_amount: Gross transaction amount
            action: 'BUY' or 'SELL'

        Returns:
            Net amount after costs
        """
        if action == "SELL":
            cost = self.calculate_sell_cost(gross_amount)
            return gross_amount - cost.total_cost
        else:  # BUY
            cost = self.calculate_buy_cost(gross_amount)
            return gross_amount + cost.total_cost  # Total cost to buyer

    def get_cost_summary(
        self,
        sell_candidates: List[SellCandidate],
        buy_candidates: List[BuyCandidate]
    ) -> Dict[str, Any]:
        """
        Get summary of all costs.

        Args:
            sell_candidates: Sell candidates
            buy_candidates: Buy candidates

        Returns:
            Cost summary dict
        """
        total_cost, costs = self.calculate_total_rebalancing_cost(
            sell_candidates, buy_candidates
        )

        sell_costs = [c for c in costs if c.action == "SELL"]
        buy_costs = [c for c in costs if c.action == "BUY"]

        total_sell = sum(s.expected_amount for s in sell_candidates)
        total_buy = sum(b.expected_amount for b in buy_candidates)
        total_volume = total_sell + total_buy

        return {
            "total_cost": round(total_cost, 2),
            "cost_pct_of_volume": round((total_cost / total_volume * 100), 3) if total_volume > 0 else 0,
            "sell_summary": {
                "count": len(sell_candidates),
                "total_amount": total_sell,
                "total_cost": sum(c.total_cost for c in sell_costs),
                "avg_cost_pct": round(
                    sum(c.total_cost for c in sell_costs) / total_sell * 100, 3
                ) if total_sell > 0 else 0
            },
            "buy_summary": {
                "count": len(buy_candidates),
                "total_amount": total_buy,
                "total_cost": sum(c.total_cost for c in buy_costs),
                "avg_cost_pct": round(
                    sum(c.total_cost for c in buy_costs) / total_buy * 100, 3
                ) if total_buy > 0 else 0
            },
            "breakdown_by_type": {
                "commission": sum(c.commission for c in costs),
                "tax": sum(c.tax for c in costs),
                "exchange_fee": sum(c.exchange_fee for c in costs),
                "slippage": sum(c.slippage for c in costs)
            }
        }
