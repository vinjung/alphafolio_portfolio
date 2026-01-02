# -*- coding: utf-8 -*-
"""
Rebalancing Executor

Executes rebalancing plan and updates database.

File: portfolio/rebalancing/rebalancing_executor.py
Created: 2025-12-29
"""

import logging
import uuid
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Tuple

from rebalancing.models import (
    RebalancingType,
    RebalancingStatus,
    RebalancingAction,
    TriggerType,
    TriggerCheckResult,
    AnalysisResult,
    CostEfficiencyResult,
    RebalancingPlan,
    RebalancingResult,
    TradeOrder,
    HoldingStatus,
)
from rebalancing.rebalancing_db import RebalancingDBManager
from rebalancing.rebalancing_trigger import RebalancingTriggerChecker
from rebalancing.rebalancing_analyzer import RebalancingAnalyzer
from rebalancing.rebalancing_cost import RebalancingCostCalculator
from rebalancing.rebalancing_quantity import RebalancingQuantityConverter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RebalancingExecutor:
    """
    Executes rebalancing plans.

    Orchestrates:
    - Plan creation
    - Cost efficiency verification
    - Trade execution
    - Database updates
    - Result generation
    """

    def __init__(self, db: RebalancingDBManager):
        """
        Initialize executor.

        Args:
            db: RebalancingDBManager instance
        """
        self.db = db

    async def create_rebalancing_plan(
        self,
        portfolio_id: str,
        rebalancing_type: RebalancingType,
        trigger_result: Optional[TriggerCheckResult] = None,
        analysis_result: Optional[AnalysisResult] = None,
        cost_result: Optional[CostEfficiencyResult] = None,
        sell_orders: Optional[List[TradeOrder]] = None,
        buy_orders: Optional[List[TradeOrder]] = None
    ) -> RebalancingPlan:
        """
        Create a rebalancing plan.

        Args:
            portfolio_id: Portfolio ID
            rebalancing_type: Type of rebalancing
            trigger_result: Trigger check result
            analysis_result: Analysis result
            cost_result: Cost efficiency result
            sell_orders: Finalized sell orders
            buy_orders: Finalized buy orders

        Returns:
            RebalancingPlan object
        """
        rebalancing_id = f"RB_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        plan_date = date.today()

        # Extract trigger info
        trigger_type = None
        trigger_details = {}
        if trigger_result and trigger_result.triggered_conditions:
            # Use highest severity trigger
            sorted_triggers = sorted(
                trigger_result.triggered_conditions,
                key=lambda t: t.severity.value,
                reverse=True
            )
            trigger_type = sorted_triggers[0].trigger_type
            trigger_details = {
                "conditions": [
                    {
                        "type": t.trigger_type.value,
                        "severity": t.severity.value,
                        "affected": t.affected_symbols
                    }
                    for t in trigger_result.triggered_conditions
                ]
            }

        # Calculate totals
        total_sell = sum(o.expected_amount for o in (sell_orders or []))
        total_buy = sum(o.expected_amount for o in (buy_orders or []))
        total_fee = sum(o.expected_cost for o in (sell_orders or []) + (buy_orders or []))
        net_cashflow = total_sell - total_buy - total_fee

        # Cost efficiency info
        expected_improvement = cost_result.expected_improvement if cost_result else 0
        cost_efficiency_ratio = cost_result.cost_ratio if cost_result else 0
        is_cost_efficient = cost_result.is_efficient if cost_result else True

        return RebalancingPlan(
            rebalancing_id=rebalancing_id,
            portfolio_id=portfolio_id,
            rebalancing_type=rebalancing_type,
            plan_date=plan_date,
            trigger_type=trigger_type,
            trigger_details=trigger_details,
            sell_orders=sell_orders or [],
            buy_orders=buy_orders or [],
            total_sell_amount=total_sell,
            total_buy_amount=total_buy,
            total_fee=total_fee,
            net_cashflow=net_cashflow,
            expected_improvement=expected_improvement,
            cost_efficiency_ratio=cost_efficiency_ratio,
            is_cost_efficient=is_cost_efficient,
            status=RebalancingStatus.PLANNED
        )

    async def execute_rebalancing(
        self,
        plan: RebalancingPlan,
        dry_run: bool = False
    ) -> RebalancingResult:
        """
        Execute a rebalancing plan.

        Args:
            plan: RebalancingPlan to execute
            dry_run: If True, don't actually update DB

        Returns:
            RebalancingResult
        """
        started_at = datetime.now()
        logger.info(f"Executing rebalancing {plan.rebalancing_id} (dry_run={dry_run})")

        try:
            if not dry_run:
                # Update status to IN_PROGRESS
                await self._save_rebalancing_record(plan, RebalancingStatus.IN_PROGRESS)

            # Execute sell orders
            executed_sells = 0
            total_sell_amount = 0.0

            for order in plan.sell_orders:
                success = await self._execute_sell_order(
                    plan.portfolio_id,
                    plan.rebalancing_id,
                    order,
                    dry_run
                )
                if success:
                    executed_sells += 1
                    total_sell_amount += order.expected_amount
                    order.executed = True
                    order.executed_price = order.expected_price
                    order.executed_amount = order.expected_amount
                    order.executed_at = datetime.now()

            # Execute buy orders
            executed_buys = 0
            total_buy_amount = 0.0

            for order in plan.buy_orders:
                success = await self._execute_buy_order(
                    plan.portfolio_id,
                    plan.rebalancing_id,
                    order,
                    dry_run
                )
                if success:
                    executed_buys += 1
                    total_buy_amount += order.expected_amount
                    order.executed = True
                    order.executed_price = order.expected_price
                    order.executed_amount = order.expected_amount
                    order.executed_at = datetime.now()

            # Update portfolio master
            if not dry_run:
                await self._update_portfolio_after_rebalancing(
                    plan.portfolio_id,
                    plan.rebalancing_id
                )

                # Update status to COMPLETED
                await self.db.update_rebalancing_status(
                    plan.rebalancing_id,
                    RebalancingStatus.COMPLETED.value,
                    datetime.now()
                )

            completed_at = datetime.now()

            # Get updated portfolio info
            portfolio = await self.db.get_portfolio_master(plan.portfolio_id)
            holdings = await self.db.get_portfolio_holdings(plan.portfolio_id)

            return RebalancingResult(
                success=True,
                rebalancing_id=plan.rebalancing_id,
                portfolio_id=plan.portfolio_id,
                rebalancing_type=plan.rebalancing_type,
                executed_sell_count=executed_sells,
                executed_buy_count=executed_buys,
                total_sell_amount=total_sell_amount,
                total_buy_amount=total_buy_amount,
                total_fee=plan.total_fee,
                new_holdings_count=len(holdings) if holdings else 0,
                new_total_value=float(portfolio["current_budget"]) if portfolio else 0,
                new_cash_balance=0,  # TODO: Track cash
                started_at=started_at,
                completed_at=completed_at
            )

        except Exception as e:
            logger.exception(f"Rebalancing execution failed: {e}")

            if not dry_run:
                await self.db.update_rebalancing_status(
                    plan.rebalancing_id,
                    RebalancingStatus.CANCELLED.value,
                    datetime.now()
                )

            return RebalancingResult(
                success=False,
                rebalancing_id=plan.rebalancing_id,
                portfolio_id=plan.portfolio_id,
                rebalancing_type=plan.rebalancing_type,
                started_at=started_at,
                completed_at=datetime.now(),
                error_code="EXECUTION_ERROR",
                error_message=str(e)
            )

    async def _save_rebalancing_record(
        self,
        plan: RebalancingPlan,
        status: RebalancingStatus
    ) -> bool:
        """Save rebalancing record to database."""
        return await self.db.save_rebalancing(
            rebalancing_id=plan.rebalancing_id,
            portfolio_id=plan.portfolio_id,
            rebalancing_type=plan.rebalancing_type.value,
            trigger_type=plan.trigger_type.value if plan.trigger_type else None,
            status=status.value,
            plan_date=plan.plan_date,
            total_sell_amount=plan.total_sell_amount,
            total_buy_amount=plan.total_buy_amount,
            total_fee=plan.total_fee,
            net_cashflow=plan.net_cashflow,
            expected_improvement=plan.expected_improvement
        )

    async def _execute_sell_order(
        self,
        portfolio_id: str,
        rebalancing_id: str,
        order: TradeOrder,
        dry_run: bool
    ) -> bool:
        """Execute a single sell order."""
        logger.info(f"  SELL {order.symbol}: {order.shares} shares @ {order.expected_price:,.0f}")

        if dry_run:
            return True

        try:
            # Get current holding
            holdings = await self.db.get_portfolio_holdings(portfolio_id)
            holding = next((h for h in holdings if h["symbol"] == order.symbol), None)

            if not holding:
                logger.error(f"  Holding not found: {order.symbol}")
                return False

            current_shares = holding["shares"]
            if order.shares > current_shares:
                logger.error(f"  Insufficient shares: {order.shares} > {current_shares}")
                return False

            # Save transaction
            await self.db.save_transaction(
                portfolio_id=portfolio_id,
                symbol=order.symbol,
                transaction_type="SELL",
                shares=order.shares,
                price=order.expected_price,
                amount=order.expected_amount,
                fee=order.expected_cost,
                transaction_date=date.today(),
                rebalancing_id=rebalancing_id
            )

            # Save rebalancing detail
            before_weight = float(holding["current_weight"]) if holding["current_weight"] else 0
            await self.db.save_rebalancing_detail(
                rebalancing_id=rebalancing_id,
                symbol=order.symbol,
                action=order.action.value,
                shares=order.shares,
                price=order.expected_price,
                amount=order.expected_amount,
                fee=order.expected_cost,
                reason=order.reason,
                before_weight=before_weight,
                after_weight=0 if order.action == RebalancingAction.SELL else before_weight * 0.5
            )

            # Update or delete holding
            remaining_shares = current_shares - order.shares
            if remaining_shares <= 0:
                await self.db.delete_holding(portfolio_id, order.symbol)
            else:
                avg_price = float(holding["avg_price"])
                current_price = order.expected_price
                invested = remaining_shares * avg_price
                current_value = remaining_shares * current_price

                await self.db.update_holding(
                    portfolio_id=portfolio_id,
                    symbol=order.symbol,
                    shares=remaining_shares,
                    avg_price=avg_price,
                    current_price=current_price,
                    invested_amount=invested,
                    current_value=current_value,
                    current_weight=0  # Will be recalculated
                )

                # Update scale-out stage if this is a DECREASE (partial sell)
                if order.action == RebalancingAction.DECREASE:
                    # Parse scale-out stage from reason if available
                    # Reason format: "1R profit reached..." or "2R profit reached..."
                    if "1R profit" in order.reason:
                        await self.db.update_scale_out_stage(portfolio_id, order.symbol, 1)
                        logger.info(f"  Updated {order.symbol} to scale-out stage 1")
                    elif "2R profit" in order.reason:
                        await self.db.update_scale_out_stage(portfolio_id, order.symbol, 2)
                        logger.info(f"  Updated {order.symbol} to scale-out stage 2")

            return True

        except Exception as e:
            logger.error(f"  Sell order failed: {e}")
            return False

    async def _execute_buy_order(
        self,
        portfolio_id: str,
        rebalancing_id: str,
        order: TradeOrder,
        dry_run: bool
    ) -> bool:
        """Execute a single buy order."""
        logger.info(f"  BUY {order.symbol}: {order.shares} shares @ {order.expected_price:,.0f}")

        if dry_run:
            return True

        try:
            # Check if already holding
            holdings = await self.db.get_portfolio_holdings(portfolio_id)
            holding = next((h for h in holdings if h["symbol"] == order.symbol), None)

            # Save transaction
            await self.db.save_transaction(
                portfolio_id=portfolio_id,
                symbol=order.symbol,
                transaction_type="BUY",
                shares=order.shares,
                price=order.expected_price,
                amount=order.expected_amount,
                fee=order.expected_cost,
                transaction_date=date.today(),
                rebalancing_id=rebalancing_id
            )

            # Save rebalancing detail
            before_weight = float(holding["current_weight"]) if holding and holding["current_weight"] else 0
            await self.db.save_rebalancing_detail(
                rebalancing_id=rebalancing_id,
                symbol=order.symbol,
                action=order.action.value,
                shares=order.shares,
                price=order.expected_price,
                amount=order.expected_amount,
                fee=order.expected_cost,
                reason=order.reason,
                before_weight=before_weight,
                after_weight=0  # Will be calculated later
            )

            # Update or create holding
            if holding:
                # Increase existing position
                current_shares = holding["shares"]
                current_avg = float(holding["avg_price"])
                current_invested = float(holding["invested_amount"]) if holding["invested_amount"] else current_shares * current_avg

                new_shares = current_shares + order.shares
                new_invested = current_invested + order.expected_amount
                new_avg = new_invested / new_shares
                new_value = new_shares * order.expected_price

                await self.db.update_holding(
                    portfolio_id=portfolio_id,
                    symbol=order.symbol,
                    shares=new_shares,
                    avg_price=new_avg,
                    current_price=order.expected_price,
                    invested_amount=new_invested,
                    current_value=new_value,
                    current_weight=0  # Will be recalculated
                )
            else:
                # New entry
                await self.db.update_holding(
                    portfolio_id=portfolio_id,
                    symbol=order.symbol,
                    shares=order.shares,
                    avg_price=order.expected_price,
                    current_price=order.expected_price,
                    invested_amount=order.expected_amount,
                    current_value=order.expected_amount,
                    current_weight=0  # Will be recalculated
                )

            return True

        except Exception as e:
            logger.error(f"  Buy order failed: {e}")
            return False

    async def _update_portfolio_after_rebalancing(
        self,
        portfolio_id: str,
        rebalancing_id: str
    ) -> bool:
        """Update portfolio master and recalculate weights."""
        try:
            holdings = await self.db.get_portfolio_holdings(portfolio_id)
            if not holdings:
                return True

            # Calculate total value
            total_value = sum(float(h["current_value"]) for h in holdings)

            # Update weights
            for h in holdings:
                current_value = float(h["current_value"])
                new_weight = current_value / total_value if total_value > 0 else 0

                await self.db.update_holding(
                    portfolio_id=portfolio_id,
                    symbol=h["symbol"],
                    shares=h["shares"],
                    avg_price=float(h["avg_price"]),
                    current_price=float(h["current_price"]),
                    invested_amount=float(h["invested_amount"]) if h["invested_amount"] else 0,
                    current_value=current_value,
                    current_weight=new_weight
                )

            # Update portfolio master
            await self.db.update_portfolio_master(
                portfolio_id=portfolio_id,
                current_budget=total_value,
                current_stock_count=len(holdings)
            )

            return True

        except Exception as e:
            logger.error(f"Failed to update portfolio after rebalancing: {e}")
            return False

    def generate_trade_summary_json(self, plan: RebalancingPlan) -> Dict[str, Any]:
        """
        Generate JSON trade summary.

        Args:
            plan: RebalancingPlan

        Returns:
            JSON-serializable dict
        """
        return {
            "rebalancing_id": plan.rebalancing_id,
            "portfolio_id": plan.portfolio_id,
            "type": plan.rebalancing_type.value,
            "plan_date": plan.plan_date.isoformat(),
            "trigger": plan.trigger_type.value if plan.trigger_type else None,
            "actions": [
                {
                    "symbol": o.symbol,
                    "action": o.action.value,
                    "shares": o.shares,
                    "price": o.expected_price,
                    "amount": o.expected_amount,
                    "cost": o.expected_cost,
                    "reason": o.reason
                }
                for o in plan.sell_orders + plan.buy_orders
            ],
            "summary": {
                "total_sell": plan.total_sell_amount,
                "total_buy": plan.total_buy_amount,
                "total_fee": plan.total_fee,
                "net_cashflow": plan.net_cashflow,
                "expected_improvement": plan.expected_improvement,
                "cost_efficiency_ratio": plan.cost_efficiency_ratio,
                "is_cost_efficient": plan.is_cost_efficient
            },
            "status": plan.status.value
        }

    async def cancel_rebalancing(
        self,
        rebalancing_id: str,
        reason: str = "User cancelled"
    ) -> bool:
        """
        Cancel a planned rebalancing.

        Args:
            rebalancing_id: Rebalancing ID
            reason: Cancellation reason

        Returns:
            True if successful
        """
        logger.info(f"Cancelling rebalancing {rebalancing_id}: {reason}")

        return await self.db.update_rebalancing_status(
            rebalancing_id,
            RebalancingStatus.CANCELLED.value,
            datetime.now()
        )
