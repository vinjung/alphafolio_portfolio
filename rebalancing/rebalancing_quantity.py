# -*- coding: utf-8 -*-
"""
Rebalancing Quantity Converter

Converts weights to actual share quantities.

File: portfolio/rebalancing/rebalancing_quantity.py
Created: 2025-12-29
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple

from rebalancing.models import (
    SellCandidate,
    BuyCandidate,
    HoldingStatus,
    TradeOrder,
    RebalancingAction,
)
from rebalancing.config import (
    KR_TRANSACTION_COST,
    US_TRANSACTION_COST,
)
from rebalancing.rebalancing_cost import RebalancingCostCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lot sizes
KR_LOT_SIZE = 1  # Korea: 1 share
US_LOT_SIZE = 1  # US: 1 share (no fractional)


class RebalancingQuantityConverter:
    """
    Converts weights and amounts to share quantities.

    Handles:
    - Lot size constraints
    - Slippage adjustment
    - Budget constraints
    - Minimum quantities
    """

    def __init__(self, country: str):
        """
        Initialize quantity converter.

        Args:
            country: 'KR' or 'US'
        """
        self.country = country
        self.lot_size = KR_LOT_SIZE if country == "KR" else US_LOT_SIZE
        self.costs = KR_TRANSACTION_COST if country == "KR" else US_TRANSACTION_COST
        self.cost_calculator = RebalancingCostCalculator(country)

    def calculate_sell_quantity(
        self,
        current_shares: int,
        sell_ratio: float,
        full_exit: bool = False
    ) -> int:
        """
        Calculate sell quantity.

        Args:
            current_shares: Current shares held
            sell_ratio: Ratio to sell (0.0 ~ 1.0)
            full_exit: If True, sell all shares

        Returns:
            Number of shares to sell
        """
        if full_exit:
            return current_shares

        # Calculate shares to sell
        raw_shares = current_shares * sell_ratio
        shares_to_sell = int(math.floor(raw_shares))

        # Ensure at least 1 share if selling
        if shares_to_sell < 1 and sell_ratio > 0:
            shares_to_sell = 1

        # Ensure at least 1 share remains (unless full exit)
        if not full_exit and shares_to_sell >= current_shares:
            shares_to_sell = current_shares - 1

        # Apply lot size
        shares_to_sell = (shares_to_sell // self.lot_size) * self.lot_size

        return max(0, shares_to_sell)

    def calculate_buy_quantity(
        self,
        amount: float,
        price: float,
        include_slippage: bool = True
    ) -> Tuple[int, float]:
        """
        Calculate buy quantity with slippage adjustment.

        Args:
            amount: Amount to invest
            price: Current stock price
            include_slippage: Include slippage in price

        Returns:
            Tuple of (shares, actual_amount)
        """
        if price <= 0 or amount <= 0:
            return 0, 0.0

        # Adjust price for slippage
        if include_slippage:
            slippage = self.costs["slippage_min"]
            adjusted_price = price * (1 + slippage)
        else:
            adjusted_price = price

        # Calculate shares
        raw_shares = amount / adjusted_price
        shares = int(math.floor(raw_shares))

        # Apply lot size
        shares = (shares // self.lot_size) * self.lot_size

        if shares < 1:
            return 0, 0.0

        # Calculate actual amount (using original price)
        actual_amount = shares * price

        return shares, actual_amount

    def calculate_available_cash(
        self,
        cash_balance: float,
        sell_candidates: List[SellCandidate]
    ) -> float:
        """
        Calculate available cash after sells.

        Available = Cash Balance + Sell Proceeds - Sell Costs

        Args:
            cash_balance: Current cash balance
            sell_candidates: Stocks to sell

        Returns:
            Available cash for buying
        """
        total_sell_amount = sum(s.expected_amount for s in sell_candidates)

        # Calculate sell costs
        total_sell_cost = 0.0
        for s in sell_candidates:
            cost = self.cost_calculator.calculate_sell_cost(s.expected_amount)
            total_sell_cost += cost.total_cost

        net_sell_proceeds = total_sell_amount - total_sell_cost

        return cash_balance + net_sell_proceeds

    def adjust_quantities_to_budget(
        self,
        buy_candidates: List[BuyCandidate],
        available_cash: float,
        prices: Dict[str, float]
    ) -> List[BuyCandidate]:
        """
        Adjust buy quantities to fit within budget.

        Args:
            buy_candidates: Original buy candidates
            available_cash: Available cash
            prices: Current prices by symbol

        Returns:
            Adjusted buy candidates
        """
        if not buy_candidates or available_cash <= 0:
            return []

        # Calculate total requested amount
        total_requested = sum(b.expected_amount for b in buy_candidates)

        if total_requested <= available_cash:
            # No adjustment needed
            return buy_candidates

        # Scale down proportionally
        scale_factor = available_cash / total_requested
        adjusted = []

        remaining_cash = available_cash

        for b in buy_candidates:
            if remaining_cash <= 0:
                break

            price = prices.get(b.symbol, 0)
            if price <= 0:
                continue

            # Scale amount
            scaled_amount = min(b.expected_amount * scale_factor, remaining_cash)

            # Recalculate shares
            shares, actual_amount = self.calculate_buy_quantity(scaled_amount, price)

            if shares < 1:
                continue

            remaining_cash -= actual_amount

            adjusted.append(BuyCandidate(
                symbol=b.symbol,
                stock_name=b.stock_name,
                action=b.action,
                shares_to_buy=shares,
                current_shares=b.current_shares,
                expected_amount=actual_amount,
                target_weight=b.target_weight * scale_factor,
                reason=b.reason,
                priority=b.priority
            ))

        return adjusted

    def finalize_sell_orders(
        self,
        sell_candidates: List[SellCandidate],
        prices: Dict[str, float]
    ) -> List[TradeOrder]:
        """
        Convert sell candidates to final trade orders.

        Args:
            sell_candidates: Sell candidates
            prices: Current prices by symbol

        Returns:
            List of TradeOrder
        """
        orders = []

        for s in sell_candidates:
            price = prices.get(s.symbol, 0)
            if price <= 0:
                # Use expected amount to estimate price
                price = s.expected_amount / s.shares_to_sell if s.shares_to_sell > 0 else 0

            if price <= 0 or s.shares_to_sell <= 0:
                continue

            # Recalculate amount with actual price
            amount = s.shares_to_sell * price

            # Calculate cost
            cost = self.cost_calculator.calculate_sell_cost(amount)

            orders.append(TradeOrder(
                symbol=s.symbol,
                stock_name=s.stock_name,
                action=s.action,
                shares=s.shares_to_sell,
                expected_price=price,
                expected_amount=amount,
                expected_cost=cost.total_cost,
                reason=s.reason
            ))

        return orders

    def finalize_buy_orders(
        self,
        buy_candidates: List[BuyCandidate],
        prices: Dict[str, float]
    ) -> List[TradeOrder]:
        """
        Convert buy candidates to final trade orders.

        Args:
            buy_candidates: Buy candidates
            prices: Current prices by symbol

        Returns:
            List of TradeOrder
        """
        orders = []

        for b in buy_candidates:
            price = prices.get(b.symbol, 0)
            if price <= 0:
                # Use expected amount to estimate price
                price = b.expected_amount / b.shares_to_buy if b.shares_to_buy > 0 else 0

            if price <= 0 or b.shares_to_buy <= 0:
                continue

            # Recalculate with finalized quantity
            shares, amount = self.calculate_buy_quantity(
                b.expected_amount, price, include_slippage=False
            )

            if shares < 1:
                continue

            # Calculate cost
            cost = self.cost_calculator.calculate_buy_cost(amount)

            orders.append(TradeOrder(
                symbol=b.symbol,
                stock_name=b.stock_name,
                action=b.action,
                shares=shares,
                expected_price=price,
                expected_amount=amount,
                expected_cost=cost.total_cost,
                reason=b.reason
            ))

        return orders

    def optimize_quantities(
        self,
        sell_candidates: List[SellCandidate],
        buy_candidates: List[BuyCandidate],
        cash_balance: float,
        prices: Dict[str, float]
    ) -> Tuple[List[TradeOrder], List[TradeOrder], Dict[str, Any]]:
        """
        Optimize and finalize all quantities.

        Args:
            sell_candidates: Sell candidates
            buy_candidates: Buy candidates
            cash_balance: Current cash balance
            prices: Current prices

        Returns:
            Tuple of (sell_orders, buy_orders, summary)
        """
        # Step 1: Finalize sell orders
        sell_orders = self.finalize_sell_orders(sell_candidates, prices)

        # Step 2: Calculate available cash
        total_sell_proceeds = sum(o.expected_amount - o.expected_cost for o in sell_orders)
        available_cash = cash_balance + total_sell_proceeds

        # Step 3: Adjust buy quantities to budget
        adjusted_buys = self.adjust_quantities_to_budget(
            buy_candidates, available_cash, prices
        )

        # Step 4: Finalize buy orders
        buy_orders = self.finalize_buy_orders(adjusted_buys, prices)

        # Step 5: Calculate summary
        total_sell = sum(o.expected_amount for o in sell_orders)
        total_buy = sum(o.expected_amount for o in buy_orders)
        total_sell_cost = sum(o.expected_cost for o in sell_orders)
        total_buy_cost = sum(o.expected_cost for o in buy_orders)
        total_cost = total_sell_cost + total_buy_cost

        net_cashflow = total_sell - total_sell_cost - total_buy - total_buy_cost
        final_cash = cash_balance + net_cashflow

        summary = {
            "sell_order_count": len(sell_orders),
            "buy_order_count": len(buy_orders),
            "total_sell_amount": total_sell,
            "total_buy_amount": total_buy,
            "total_sell_cost": total_sell_cost,
            "total_buy_cost": total_buy_cost,
            "total_cost": total_cost,
            "net_cashflow": net_cashflow,
            "initial_cash": cash_balance,
            "final_cash": final_cash,
            "utilization_rate": (total_buy / available_cash * 100) if available_cash > 0 else 0
        }

        return sell_orders, buy_orders, summary

    def validate_quantities(
        self,
        sell_orders: List[TradeOrder],
        buy_orders: List[TradeOrder],
        holdings: List[HoldingStatus],
        cash_balance: float
    ) -> List[str]:
        """
        Validate trade orders.

        Args:
            sell_orders: Sell orders
            buy_orders: Buy orders
            holdings: Current holdings
            cash_balance: Current cash

        Returns:
            List of validation error messages
        """
        errors = []
        holding_map = {h.symbol: h for h in holdings}

        # Validate sell orders
        for order in sell_orders:
            holding = holding_map.get(order.symbol)
            if not holding:
                errors.append(f"SELL {order.symbol}: Not in holdings")
                continue

            if order.shares > holding.shares:
                errors.append(
                    f"SELL {order.symbol}: Quantity {order.shares} exceeds holdings {holding.shares}"
                )

            if order.shares <= 0:
                errors.append(f"SELL {order.symbol}: Invalid quantity {order.shares}")

        # Validate buy orders
        total_buy_amount = sum(o.expected_amount + o.expected_cost for o in buy_orders)
        total_sell_proceeds = sum(
            o.expected_amount - o.expected_cost for o in sell_orders
        )
        available = cash_balance + total_sell_proceeds

        if total_buy_amount > available * 1.01:  # 1% tolerance
            errors.append(
                f"Insufficient funds: Need {total_buy_amount:,.0f}, Available {available:,.0f}"
            )

        for order in buy_orders:
            if order.shares <= 0:
                errors.append(f"BUY {order.symbol}: Invalid quantity {order.shares}")

            if order.expected_price <= 0:
                errors.append(f"BUY {order.symbol}: Invalid price {order.expected_price}")

        return errors

    def get_quantity_summary(
        self,
        sell_orders: List[TradeOrder],
        buy_orders: List[TradeOrder]
    ) -> str:
        """
        Generate human-readable quantity summary.

        Args:
            sell_orders: Sell orders
            buy_orders: Buy orders

        Returns:
            Summary string
        """
        lines = ["Quantity Conversion Summary", "=" * 50]

        if sell_orders:
            lines.append(f"\nSELL Orders ({len(sell_orders)}):")
            for o in sell_orders:
                lines.append(
                    f"  {o.symbol}: {o.shares} shares @ {o.expected_price:,.0f} = {o.expected_amount:,.0f} (cost: {o.expected_cost:,.0f})"
                )

        if buy_orders:
            lines.append(f"\nBUY Orders ({len(buy_orders)}):")
            for o in buy_orders:
                lines.append(
                    f"  {o.symbol}: {o.shares} shares @ {o.expected_price:,.0f} = {o.expected_amount:,.0f} (cost: {o.expected_cost:,.0f})"
                )

        total_sell = sum(o.expected_amount for o in sell_orders)
        total_buy = sum(o.expected_amount for o in buy_orders)
        total_cost = sum(o.expected_cost for o in sell_orders + buy_orders)

        lines.extend([
            "",
            f"Total Sell: {total_sell:,.0f}",
            f"Total Buy: {total_buy:,.0f}",
            f"Total Cost: {total_cost:,.0f}",
            f"Net: {total_sell - total_buy - total_cost:,.0f}"
        ])

        return "\n".join(lines)
