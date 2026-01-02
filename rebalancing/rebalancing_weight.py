# -*- coding: utf-8 -*-
"""
Rebalancing Weight Allocator

Calculates weight adjustments for rebalancing.

File: portfolio/rebalancing/rebalancing_weight.py
Created: 2025-12-29
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from rebalancing.models import (
    HoldingStatus,
    SellCandidate,
    BuyCandidate,
    CandidateStock,
    RebalancingAction,
)
from rebalancing.config import (
    MAX_WEIGHT_PER_STOCK,
    MAX_WEIGHT_PER_SECTOR,
    VAR_LIMIT,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Risk parity blend ratios by risk level
RISK_PARITY_BLEND = {
    "conservative": 0.7,
    "balanced": 0.5,
    "aggressive": 0.3,
}


class RebalancingWeightAllocator:
    """
    Allocates weights for rebalancing.

    Uses combination of:
    - Risk parity (volatility-inverse)
    - Score-based allocation
    - Constraint satisfaction
    """

    def __init__(self, risk_level: str):
        """
        Initialize weight allocator.

        Args:
            risk_level: Portfolio risk level
        """
        self.risk_level = risk_level
        self.max_stock_weight = MAX_WEIGHT_PER_STOCK.get(risk_level, 0.15)
        self.max_sector_weight = MAX_WEIGHT_PER_SECTOR.get(risk_level, 0.30)
        self.risk_parity_blend = RISK_PARITY_BLEND.get(risk_level, 0.5)

    def calculate_post_rebalancing_weights(
        self,
        holdings: List[HoldingStatus],
        sell_candidates: List[SellCandidate],
        buy_candidates: List[BuyCandidate],
        total_portfolio_value: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate weights after rebalancing.

        Args:
            holdings: Current holdings
            sell_candidates: Stocks being sold
            buy_candidates: Stocks being bought
            total_portfolio_value: Total portfolio value

        Returns:
            Dict mapping symbol to weight info
        """
        if total_portfolio_value <= 0:
            return {}

        result = {}
        sell_map = {s.symbol: s for s in sell_candidates}
        buy_map = {b.symbol: b for b in buy_candidates}

        # Process existing holdings
        for h in holdings:
            symbol = h.symbol
            current_value = h.current_value
            current_weight = h.current_weight

            if symbol in sell_map:
                sell = sell_map[symbol]
                if sell.action == RebalancingAction.SELL:
                    # Full exit - remove from portfolio
                    continue
                else:  # DECREASE
                    remaining_shares = h.shares - sell.shares_to_sell
                    new_value = remaining_shares * h.current_price
            else:
                new_value = current_value

            # Check for increase
            if symbol in buy_map:
                buy = buy_map[symbol]
                new_value += buy.expected_amount

            new_weight = new_value / total_portfolio_value

            result[symbol] = {
                "before_value": current_value,
                "after_value": new_value,
                "before_weight": current_weight,
                "after_weight": new_weight,
                "weight_change": new_weight - current_weight,
                "sector": h.sector
            }

        # Process new entries
        for b in buy_candidates:
            if b.symbol not in result and b.action == RebalancingAction.BUY:
                new_weight = b.expected_amount / total_portfolio_value
                result[b.symbol] = {
                    "before_value": 0,
                    "after_value": b.expected_amount,
                    "before_weight": 0,
                    "after_weight": new_weight,
                    "weight_change": new_weight,
                    "sector": None  # Will be filled later
                }

        return result

    def allocate_new_entry_weights(
        self,
        candidates: List[CandidateStock],
        available_amount: float,
        current_sector_weights: Dict[str, float]
    ) -> List[BuyCandidate]:
        """
        Allocate weights for new entries using risk parity blend.

        Args:
            candidates: New entry candidates
            available_amount: Available cash for buying
            current_sector_weights: Current sector weight distribution

        Returns:
            List of BuyCandidates with allocated weights
        """
        if not candidates or available_amount <= 0:
            return []

        # Step 1: Calculate score-based weights
        total_score = sum(c.final_score for c in candidates)
        if total_score <= 0:
            # Equal weight if no scores
            score_weights = {c.symbol: 1.0 / len(candidates) for c in candidates}
        else:
            score_weights = {c.symbol: c.final_score / total_score for c in candidates}

        # Step 2: Calculate risk parity weights (volatility inverse)
        volatilities = {}
        for c in candidates:
            vol = c.volatility if c.volatility and c.volatility > 0 else 0.25  # Default 25%
            volatilities[c.symbol] = vol

        total_inv_vol = sum(1 / v for v in volatilities.values())
        rp_weights = {s: (1 / v) / total_inv_vol for s, v in volatilities.items()}

        # Step 3: Blend weights
        blend = self.risk_parity_blend
        blended_weights = {}
        for c in candidates:
            s = c.symbol
            blended = blend * rp_weights[s] + (1 - blend) * score_weights[s]
            blended_weights[s] = blended

        # Step 4: Apply constraints
        constrained_weights = self._apply_weight_constraints(
            blended_weights, candidates, current_sector_weights
        )

        # Step 5: Normalize to sum to 1
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            normalized = {s: w / total_weight for s, w in constrained_weights.items()}
        else:
            normalized = constrained_weights

        # Step 6: Convert to BuyCandidates
        result = []
        for c in candidates:
            weight = normalized.get(c.symbol, 0)
            if weight <= 0:
                continue

            amount = available_amount * weight
            shares = int(amount / c.current_price)
            if shares < 1:
                continue

            actual_amount = shares * c.current_price

            result.append(BuyCandidate(
                symbol=c.symbol,
                stock_name=c.stock_name,
                action=RebalancingAction.BUY,
                shares_to_buy=shares,
                current_shares=0,
                expected_amount=actual_amount,
                target_weight=weight,
                reason=f"New entry: score={c.final_score:.1f}, {c.consecutive_buy_days} days",
                priority=int(c.final_score)
            ))

        return result

    def _apply_weight_constraints(
        self,
        weights: Dict[str, float],
        candidates: List[CandidateStock],
        current_sector_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply weight constraints.

        Args:
            weights: Raw weights
            candidates: Candidate stocks
            current_sector_weights: Current sector weights

        Returns:
            Constrained weights
        """
        result = {}
        candidate_map = {c.symbol: c for c in candidates}
        sector_new_weights: Dict[str, float] = {}

        for symbol, weight in weights.items():
            candidate = candidate_map.get(symbol)
            if not candidate:
                continue

            sector = candidate.sector or "Unknown"

            # Apply max stock weight
            constrained_weight = min(weight, self.max_stock_weight)

            # Apply max sector weight
            current_sector = current_sector_weights.get(sector, 0)
            new_sector = sector_new_weights.get(sector, 0)
            available_sector = self.max_sector_weight - current_sector - new_sector

            if available_sector <= 0:
                constrained_weight = 0
            else:
                constrained_weight = min(constrained_weight, available_sector)

            result[symbol] = constrained_weight
            sector_new_weights[sector] = sector_new_weights.get(sector, 0) + constrained_weight

        return result

    def calculate_decrease_weight(
        self,
        holding: HoldingStatus,
        severity: str = "normal"
    ) -> float:
        """
        Calculate weight decrease ratio.

        Args:
            holding: Holding status
            severity: 'normal' (30%) or 'severe' (50%)

        Returns:
            Decrease ratio (0.3 or 0.5)
        """
        if severity == "severe":
            return 0.5  # 50% decrease
        return 0.3  # 30% decrease

    def calculate_increase_weight(
        self,
        holding: HoldingStatus,
        available_cash: float,
        target_weight_increase: float = 0.05
    ) -> Tuple[float, int]:
        """
        Calculate weight increase amount.

        Args:
            holding: Holding status
            available_cash: Available cash
            target_weight_increase: Target weight increase (default 5%)

        Returns:
            Tuple of (increase_amount, shares_to_buy)
        """
        # Calculate target increase based on current weight
        target_increase = holding.current_value * (target_weight_increase / holding.current_weight) if holding.current_weight > 0 else 0

        # Limit by available cash and max weight
        max_by_constraint = (self.max_stock_weight - holding.current_weight) * holding.current_value / holding.current_weight if holding.current_weight > 0 else available_cash
        increase_amount = min(target_increase, available_cash, max_by_constraint)

        if increase_amount <= 0:
            return 0, 0

        shares = int(increase_amount / holding.current_price)
        actual_amount = shares * holding.current_price

        return actual_amount, shares

    def validate_sector_constraints(
        self,
        post_weights: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Validate sector constraints after rebalancing.

        Args:
            post_weights: Post-rebalancing weight info

        Returns:
            List of violation messages
        """
        violations = []
        sector_weights: Dict[str, float] = {}

        for symbol, info in post_weights.items():
            sector = info.get("sector", "Unknown")
            weight = info.get("after_weight", 0)
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        for sector, weight in sector_weights.items():
            if weight > self.max_sector_weight:
                violations.append(
                    f"Sector '{sector}' weight {weight:.1%} exceeds max {self.max_sector_weight:.1%}"
                )

        return violations

    def validate_stock_constraints(
        self,
        post_weights: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Validate individual stock constraints.

        Args:
            post_weights: Post-rebalancing weight info

        Returns:
            List of violation messages
        """
        violations = []

        for symbol, info in post_weights.items():
            weight = info.get("after_weight", 0)
            if weight > self.max_stock_weight:
                violations.append(
                    f"Stock '{symbol}' weight {weight:.1%} exceeds max {self.max_stock_weight:.1%}"
                )

        return violations

    def get_weight_summary(
        self,
        post_weights: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get summary of weight distribution.

        Args:
            post_weights: Post-rebalancing weight info

        Returns:
            Summary dict
        """
        if not post_weights:
            return {"count": 0}

        weights = [info["after_weight"] for info in post_weights.values()]
        sector_weights: Dict[str, float] = {}

        for symbol, info in post_weights.items():
            sector = info.get("sector", "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0) + info["after_weight"]

        return {
            "count": len(weights),
            "total_weight": sum(weights),
            "max_weight": max(weights),
            "min_weight": min(weights),
            "avg_weight": np.mean(weights),
            "std_weight": np.std(weights),
            "sector_distribution": sector_weights,
            "max_sector_weight": max(sector_weights.values()) if sector_weights else 0,
            "sector_count": len(sector_weights)
        }

    def rebalance_to_target_weights(
        self,
        holdings: List[HoldingStatus],
        target_weights: Dict[str, float],
        total_value: float
    ) -> Tuple[List[SellCandidate], List[BuyCandidate]]:
        """
        Calculate trades needed to reach target weights.

        Args:
            holdings: Current holdings
            target_weights: Target weight per symbol
            total_value: Total portfolio value

        Returns:
            Tuple of (sell_candidates, buy_candidates)
        """
        sells = []
        buys = []

        holding_map = {h.symbol: h for h in holdings}
        current_symbols = set(holding_map.keys())
        target_symbols = set(target_weights.keys())

        # Stocks to sell (in current but not in target, or reduce weight)
        for symbol in current_symbols:
            h = holding_map[symbol]
            target_weight = target_weights.get(symbol, 0)
            current_weight = h.current_weight

            if target_weight == 0:
                # Full exit
                sells.append(SellCandidate(
                    symbol=symbol,
                    stock_name=h.stock_name,
                    action=RebalancingAction.SELL,
                    shares_to_sell=h.shares,
                    current_shares=h.shares,
                    sell_ratio=1.0,
                    expected_amount=h.current_value,
                    reason="Removed from target portfolio",
                    priority=50
                ))
            elif target_weight < current_weight:
                # Reduce position
                target_value = total_value * target_weight
                reduce_value = h.current_value - target_value
                reduce_shares = int(reduce_value / h.current_price)

                if reduce_shares > 0:
                    sells.append(SellCandidate(
                        symbol=symbol,
                        stock_name=h.stock_name,
                        action=RebalancingAction.DECREASE,
                        shares_to_sell=reduce_shares,
                        current_shares=h.shares,
                        sell_ratio=reduce_shares / h.shares,
                        expected_amount=reduce_shares * h.current_price,
                        reason=f"Reduce weight {current_weight:.1%} -> {target_weight:.1%}",
                        priority=30
                    ))

        # Stocks to buy (new or increase)
        for symbol, target_weight in target_weights.items():
            if target_weight <= 0:
                continue

            target_value = total_value * target_weight

            if symbol in holding_map:
                h = holding_map[symbol]
                if target_weight > h.current_weight:
                    # Increase position
                    increase_value = target_value - h.current_value
                    increase_shares = int(increase_value / h.current_price)

                    if increase_shares > 0:
                        buys.append(BuyCandidate(
                            symbol=symbol,
                            stock_name=h.stock_name,
                            action=RebalancingAction.INCREASE,
                            shares_to_buy=increase_shares,
                            current_shares=h.shares,
                            expected_amount=increase_shares * h.current_price,
                            target_weight=target_weight,
                            reason=f"Increase weight {h.current_weight:.1%} -> {target_weight:.1%}",
                            priority=30
                        ))
            else:
                # New entry - need price info (would be passed in real implementation)
                buys.append(BuyCandidate(
                    symbol=symbol,
                    stock_name=None,
                    action=RebalancingAction.BUY,
                    shares_to_buy=0,  # To be calculated with price
                    current_shares=0,
                    expected_amount=target_value,
                    target_weight=target_weight,
                    reason="New entry to target portfolio",
                    priority=50
                ))

        return sells, buys
