# -*- coding: utf-8 -*-
"""
Rebalancing Trigger Checker

Checks trigger conditions for emergency rebalancing.

File: portfolio/rebalancing/rebalancing_trigger.py
Created: 2025-12-29
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncpg

from rebalancing.models import (
    TriggerCondition,
    TriggerCheckResult,
    TriggerType,
    AlertSeverity,
    HoldingStatus,
    RebalancingAction,
)
from rebalancing.config import (
    GRADE_NUMERIC,
    GRADE_DROP_TRIGGER,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MDD_LIMIT,
    MARKET_CRASH_THRESHOLD,
    MIN_HOLDING_DAYS,
    get_recommended_action,
)
from rebalancing.rebalancing_db import RebalancingDBManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RebalancingTriggerChecker:
    """
    Checks various trigger conditions for rebalancing.

    Trigger types:
    - GRADE_DROP: Quant grade dropped 2+ levels
    - STOP_LOSS: Stock loss reached stop-loss threshold
    - TAKE_PROFIT: Stock gain reached take-profit threshold
    - MDD_LIMIT: Portfolio MDD limit reached
    - SUSPENDED: Trading halt detected
    - MARKET_CRASH: Benchmark dropped 5%+ in a day
    """

    def __init__(self, db: RebalancingDBManager):
        """
        Initialize trigger checker.

        Args:
            db: RebalancingDBManager instance
        """
        self.db = db

    async def check_all_triggers(
        self,
        portfolio_id: str,
        check_date: Optional[date] = None
    ) -> TriggerCheckResult:
        """
        Check all trigger conditions for a portfolio.

        Args:
            portfolio_id: Portfolio ID
            check_date: Check date (default: today)

        Returns:
            TriggerCheckResult with all triggered conditions
        """
        if check_date is None:
            check_date = date.today()

        logger.info(f"Checking triggers for portfolio {portfolio_id} on {check_date}")

        # Get portfolio info
        portfolio = await self.db.get_portfolio_master(portfolio_id)
        if not portfolio:
            return TriggerCheckResult(
                portfolio_id=portfolio_id,
                check_date=check_date,
                needs_rebalancing=False,
                summary="Portfolio not found"
            )

        country = portfolio["country"]
        risk_level = portfolio["risk_level"]

        # Get holdings with current status
        holdings = await self._get_holdings_status(portfolio_id, country, check_date)

        if not holdings:
            return TriggerCheckResult(
                portfolio_id=portfolio_id,
                check_date=check_date,
                needs_rebalancing=False,
                summary="No holdings in portfolio"
            )

        # Check each trigger type
        triggered_conditions: List[TriggerCondition] = []

        # 1. Grade drop check
        grade_trigger = await self._check_grade_drop(holdings)
        if grade_trigger.is_triggered:
            triggered_conditions.append(grade_trigger)

        # 2. Stop loss check (ATR-based)
        stop_loss_trigger = self._check_stop_loss(holdings, risk_level)
        if stop_loss_trigger.is_triggered:
            triggered_conditions.append(stop_loss_trigger)

        # 3. Trailing stop check (Chandelier Exit)
        trailing_trigger = self._check_trailing_stop(holdings)
        if trailing_trigger.is_triggered:
            triggered_conditions.append(trailing_trigger)

        # 4. Scale-out check (partial profit taking)
        scale_out_trigger = self._check_scale_out(holdings)
        if scale_out_trigger.is_triggered:
            triggered_conditions.append(scale_out_trigger)

        # 5. MDD limit check
        mdd_trigger = await self._check_mdd_limit(portfolio_id, risk_level)
        if mdd_trigger.is_triggered:
            triggered_conditions.append(mdd_trigger)

        # 6. Trading halt check
        halt_trigger = self._check_trading_halt(holdings)
        if halt_trigger.is_triggered:
            triggered_conditions.append(halt_trigger)

        # 7. Market crash check
        market_trigger = await self._check_market_crash(country, portfolio.get("benchmark"))
        if market_trigger.is_triggered:
            triggered_conditions.append(market_trigger)

        # Build summary
        needs_rebalancing = len(triggered_conditions) > 0
        if needs_rebalancing:
            critical_count = sum(
                1 for t in triggered_conditions
                if t.severity == AlertSeverity.CRITICAL
            )
            warning_count = len(triggered_conditions) - critical_count
            summary = f"Triggered: {len(triggered_conditions)} conditions ({critical_count} critical, {warning_count} warning)"
        else:
            summary = "No triggers activated"

        return TriggerCheckResult(
            portfolio_id=portfolio_id,
            check_date=check_date,
            needs_rebalancing=needs_rebalancing,
            triggered_conditions=triggered_conditions,
            summary=summary
        )

    async def _get_holdings_status(
        self,
        portfolio_id: str,
        country: str,
        check_date: date
    ) -> List[HoldingStatus]:
        """
        Get holdings with current status including grade info.

        Args:
            portfolio_id: Portfolio ID
            country: Country code
            check_date: Check date

        Returns:
            List of HoldingStatus objects
        """
        # Get raw holdings
        holdings_raw = await self.db.get_portfolio_holdings(portfolio_id)
        if not holdings_raw:
            return []

        symbols = [h["symbol"] for h in holdings_raw]

        # Get current grades
        grades = await self.db.get_stock_grades(symbols, country, check_date)

        # Get current prices
        prices = await self.db.get_current_prices(symbols, country)

        # Check trading halts
        halts = await self.db.check_trading_halt(symbols, country)

        # Build HoldingStatus list
        result = []
        for h in holdings_raw:
            symbol = h["symbol"]

            # Get current price
            price_info = prices.get(symbol, {})
            current_price = price_info.get("price", h["current_price"] or h["avg_price"])

            # Calculate current values
            shares = h["shares"]
            avg_price = float(h["avg_price"])
            invested = float(h["invested_amount"]) if h["invested_amount"] else shares * avg_price
            current_value = shares * current_price
            pnl = current_value - invested
            pnl_pct = (pnl / invested * 100) if invested > 0 else 0

            # Get grade info
            grade_info = grades.get(symbol, {})
            current_grade = grade_info.get("grade")
            entry_grade = h.get("entry_grade")

            # Calculate grade change
            grade_change = 0
            if current_grade and entry_grade:
                current_num = GRADE_NUMERIC.get(current_grade, 3)
                entry_num = GRADE_NUMERIC.get(entry_grade, 5)
                grade_change = current_num - entry_num

            # Calculate holding days
            entry_date = h.get("entry_date")
            holding_days = 0
            if entry_date:
                if isinstance(entry_date, date):
                    holding_days = (check_date - entry_date).days
                else:
                    holding_days = (check_date - entry_date.date()).days

            # Get recommended action
            recommended = RebalancingAction.HOLD
            action_reason = None
            if current_grade and entry_grade:
                action_str = get_recommended_action(current_grade, entry_grade)
                recommended = RebalancingAction(action_str) if action_str in [a.value for a in RebalancingAction] else RebalancingAction.HOLD
                if recommended != RebalancingAction.HOLD:
                    action_reason = f"Grade changed from {entry_grade} to {current_grade}"

            # Calculate total portfolio value for weight
            total_value = sum(
                float(hh["shares"]) * prices.get(hh["symbol"], {}).get("price", float(hh["avg_price"]))
                for hh in holdings_raw
            )
            current_weight = current_value / total_value if total_value > 0 else 0

            # ATR-based risk management fields
            atr_pct = float(h.get("atr_pct")) if h.get("atr_pct") else None
            dynamic_stop_pct = float(h.get("dynamic_stop_pct")) if h.get("dynamic_stop_pct") else None
            dynamic_take_pct = float(h.get("dynamic_take_pct")) if h.get("dynamic_take_pct") else None
            peak_price = float(h.get("peak_price")) if h.get("peak_price") else current_price
            peak_date_raw = h.get("peak_date")
            peak_date_val = peak_date_raw if isinstance(peak_date_raw, date) else (peak_date_raw.date() if peak_date_raw else None)
            trailing_stop_price = float(h.get("trailing_stop_price")) if h.get("trailing_stop_price") else None
            scale_out_stage = h.get("scale_out_stage") or 0
            profit_protection_mode = h.get("profit_protection_mode") or False

            # Calculate drawdown from peak
            drawdown_from_peak_pct = None
            if peak_price and peak_price > 0:
                drawdown_from_peak_pct = ((current_price - peak_price) / peak_price) * 100

            status = HoldingStatus(
                symbol=symbol,
                stock_name=h.get("stock_name"),
                sector=h.get("sector"),
                shares=shares,
                avg_price=avg_price,
                current_price=current_price,
                invested_amount=invested,
                current_value=current_value,
                current_weight=current_weight,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
                entry_grade=entry_grade,
                current_grade=current_grade,
                grade_change=grade_change,
                entry_date=entry_date if isinstance(entry_date, date) else (entry_date.date() if entry_date else None),
                holding_days=holding_days,
                atr_pct=atr_pct,
                dynamic_stop_pct=dynamic_stop_pct,
                dynamic_take_pct=dynamic_take_pct,
                peak_price=peak_price,
                peak_date=peak_date_val,
                trailing_stop_price=trailing_stop_price,
                drawdown_from_peak_pct=drawdown_from_peak_pct,
                scale_out_stage=scale_out_stage,
                profit_protection_mode=profit_protection_mode,
                is_tradable=not halts.get(symbol, False),
                is_suspended=halts.get(symbol, False),
                recommended_action=recommended,
                action_reason=action_reason
            )
            result.append(status)

        return result

    async def _check_grade_drop(
        self,
        holdings: List[HoldingStatus]
    ) -> TriggerCondition:
        """
        Check for grade drop trigger (2+ levels).

        Args:
            holdings: List of holding status

        Returns:
            TriggerCondition result
        """
        affected = []
        details = {}

        for h in holdings:
            # Grade dropped 2+ levels (grade_change is negative for drops)
            if h.grade_change <= -GRADE_DROP_TRIGGER:
                affected.append(h.symbol)
                details[h.symbol] = {
                    "entry_grade": h.entry_grade,
                    "current_grade": h.current_grade,
                    "grade_change": h.grade_change,
                    "holding_days": h.holding_days,
                }

        is_triggered = len(affected) > 0
        return TriggerCondition(
            trigger_type=TriggerType.GRADE_DROP,
            is_triggered=is_triggered,
            severity=AlertSeverity.WARNING if is_triggered else AlertSeverity.INFO,
            details={"dropped_stocks": details},
            affected_symbols=affected
        )

    def _check_stop_loss(
        self,
        holdings: List[HoldingStatus],
        risk_level: str
    ) -> TriggerCondition:
        """
        Check for stop loss trigger using ATR-based dynamic thresholds.

        Uses dynamic_stop_pct (2x ATR) when available, falls back to fixed threshold.

        Args:
            holdings: List of holding status
            risk_level: Portfolio risk level

        Returns:
            TriggerCondition result
        """
        fallback_threshold = STOP_LOSS_PCT.get(risk_level, -12.0)
        affected = []
        details = {}

        for h in holdings:
            # Use ATR-based dynamic stop if available, else fallback
            threshold = h.dynamic_stop_pct if h.dynamic_stop_pct else fallback_threshold

            if h.unrealized_pnl_pct <= threshold:
                affected.append(h.symbol)
                details[h.symbol] = {
                    "pnl_pct": round(h.unrealized_pnl_pct, 2),
                    "threshold": threshold,
                    "atr_based": h.dynamic_stop_pct is not None,
                    "atr_pct": h.atr_pct,
                    "current_value": h.current_value,
                    "invested": h.invested_amount,
                }

        is_triggered = len(affected) > 0
        return TriggerCondition(
            trigger_type=TriggerType.STOP_LOSS,
            is_triggered=is_triggered,
            severity=AlertSeverity.CRITICAL if is_triggered else AlertSeverity.INFO,
            details={"fallback_threshold": fallback_threshold, "stocks": details},
            affected_symbols=affected
        )

    def _check_trailing_stop(
        self,
        holdings: List[HoldingStatus]
    ) -> TriggerCondition:
        """
        Check for trailing stop trigger (Chandelier Exit).

        Chandelier Exit: Sell when price drops below (peak_price - 3 x ATR).
        Only applies to profitable positions to protect gains.

        Args:
            holdings: List of holding status

        Returns:
            TriggerCondition result
        """
        affected = []
        details = {}

        for h in holdings:
            # Only check stocks with ATR data and trailing stop price
            if not h.atr_pct or not h.peak_price:
                continue

            # Calculate trailing stop if not already set
            # Chandelier Exit: peak - 3x ATR
            if h.trailing_stop_price:
                trailing_stop = h.trailing_stop_price
            else:
                atr_amount = h.peak_price * (h.atr_pct / 100)
                trailing_stop = h.peak_price - (3 * atr_amount)

            # Check if current price broke trailing stop
            if h.current_price < trailing_stop:
                # Only trigger for positions that were in profit (protecting gains)
                if h.peak_price > h.avg_price:
                    affected.append(h.symbol)
                    details[h.symbol] = {
                        "current_price": h.current_price,
                        "peak_price": h.peak_price,
                        "trailing_stop": round(trailing_stop, 2),
                        "atr_pct": h.atr_pct,
                        "drawdown_from_peak_pct": round(h.drawdown_from_peak_pct, 2) if h.drawdown_from_peak_pct else None,
                        "unrealized_pnl_pct": round(h.unrealized_pnl_pct, 2),
                        "profit_protected": True,
                    }

        is_triggered = len(affected) > 0
        return TriggerCondition(
            trigger_type=TriggerType.TRAILING_STOP,
            is_triggered=is_triggered,
            severity=AlertSeverity.WARNING if is_triggered else AlertSeverity.INFO,
            details={"chandelier_multiplier": 3, "stocks": details},
            affected_symbols=affected
        )

    def _check_scale_out(
        self,
        holdings: List[HoldingStatus]
    ) -> TriggerCondition:
        """
        Check for scale-out trigger (partial profit taking).

        Scale-out stages:
        - Stage 0 -> 1: Sell 33% when profit reaches 1x risk (ATR)
        - Stage 1 -> 2: Sell 33% when profit reaches 2x risk (2x ATR)
        - Stage 2 -> 3: Remaining 34% managed by trailing stop only

        Args:
            holdings: List of holding status

        Returns:
            TriggerCondition result
        """
        affected = []
        details = {}

        for h in holdings:
            # Only check stocks with ATR data
            if not h.atr_pct:
                continue

            # Calculate profit in terms of risk multiples (ATR)
            profit_pct = h.unrealized_pnl_pct
            risk_pct = h.atr_pct  # 1R = 1x ATR

            if risk_pct <= 0:
                continue

            risk_multiple = profit_pct / risk_pct  # How many R in profit

            # Determine if scale-out is due based on current stage
            scale_out_due = False
            target_stage = h.scale_out_stage
            sell_ratio = 0.0
            reason = ""

            if h.scale_out_stage == 0 and risk_multiple >= 1.0:
                # First scale-out: 1R profit reached, sell 33%
                scale_out_due = True
                target_stage = 1
                sell_ratio = 0.33
                reason = f"1R profit reached ({profit_pct:.1f}% >= {risk_pct:.1f}%)"

            elif h.scale_out_stage == 1 and risk_multiple >= 2.0:
                # Second scale-out: 2R profit reached, sell another 33%
                scale_out_due = True
                target_stage = 2
                sell_ratio = 0.33
                reason = f"2R profit reached ({profit_pct:.1f}% >= {2*risk_pct:.1f}%)"

            if scale_out_due:
                affected.append(h.symbol)
                details[h.symbol] = {
                    "current_stage": h.scale_out_stage,
                    "target_stage": target_stage,
                    "profit_pct": round(profit_pct, 2),
                    "risk_pct": round(risk_pct, 2),
                    "risk_multiple": round(risk_multiple, 2),
                    "sell_ratio": sell_ratio,
                    "reason": reason,
                    "current_shares": h.shares,
                    "shares_to_sell": int(h.shares * sell_ratio),
                }

        is_triggered = len(affected) > 0
        return TriggerCondition(
            trigger_type=TriggerType.SCALE_OUT,
            is_triggered=is_triggered,
            severity=AlertSeverity.INFO,
            details={"scale_out_stages": "33%-33%-34%", "stocks": details},
            affected_symbols=affected
        )

    def _check_take_profit(
        self,
        holdings: List[HoldingStatus],
        risk_level: str
    ) -> TriggerCondition:
        """
        Check for take profit trigger.

        Args:
            holdings: List of holding status
            risk_level: Portfolio risk level

        Returns:
            TriggerCondition result
        """
        threshold = TAKE_PROFIT_PCT.get(risk_level, 25.0)
        affected = []
        details = {}

        for h in holdings:
            if h.unrealized_pnl_pct >= threshold:
                affected.append(h.symbol)
                details[h.symbol] = {
                    "pnl_pct": round(h.unrealized_pnl_pct, 2),
                    "threshold": threshold,
                    "current_value": h.current_value,
                    "profit": h.unrealized_pnl,
                }

        is_triggered = len(affected) > 0
        return TriggerCondition(
            trigger_type=TriggerType.TAKE_PROFIT,
            is_triggered=is_triggered,
            severity=AlertSeverity.INFO,
            details={"take_profit_threshold": threshold, "stocks": details},
            affected_symbols=affected
        )

    async def _check_mdd_limit(
        self,
        portfolio_id: str,
        risk_level: str
    ) -> TriggerCondition:
        """
        Check for portfolio MDD limit trigger.

        Args:
            portfolio_id: Portfolio ID
            risk_level: Portfolio risk level

        Returns:
            TriggerCondition result
        """
        threshold = MDD_LIMIT.get(risk_level, -15.0)

        # Get portfolio performance history to calculate MDD
        # For now, simplified: check current drawdown from peak
        portfolio = await self.db.get_portfolio_master(portfolio_id)
        if not portfolio:
            return TriggerCondition(
                trigger_type=TriggerType.MDD_LIMIT,
                is_triggered=False,
                severity=AlertSeverity.INFO,
                details={}
            )

        initial_budget = float(portfolio["initial_budget"])
        current_budget = float(portfolio["current_budget"]) if portfolio["current_budget"] else initial_budget

        # Simple drawdown calculation (from initial)
        # TODO: Implement proper MDD calculation from peak
        drawdown_pct = ((current_budget - initial_budget) / initial_budget) * 100

        is_triggered = drawdown_pct <= threshold

        return TriggerCondition(
            trigger_type=TriggerType.MDD_LIMIT,
            is_triggered=is_triggered,
            severity=AlertSeverity.CRITICAL if is_triggered else AlertSeverity.INFO,
            details={
                "current_drawdown": round(drawdown_pct, 2),
                "threshold": threshold,
                "initial_budget": initial_budget,
                "current_budget": current_budget,
            },
            affected_symbols=[]
        )

    def _check_trading_halt(
        self,
        holdings: List[HoldingStatus]
    ) -> TriggerCondition:
        """
        Check for trading halt trigger.

        Args:
            holdings: List of holding status

        Returns:
            TriggerCondition result
        """
        affected = [h.symbol for h in holdings if h.is_suspended]
        details = {
            s: {"stock_name": h.stock_name}
            for h in holdings
            for s in [h.symbol]
            if h.is_suspended
        }

        is_triggered = len(affected) > 0
        return TriggerCondition(
            trigger_type=TriggerType.SUSPENDED,
            is_triggered=is_triggered,
            severity=AlertSeverity.CRITICAL if is_triggered else AlertSeverity.INFO,
            details={"suspended_stocks": details},
            affected_symbols=affected
        )

    async def _check_market_crash(
        self,
        country: str,
        benchmark: Optional[str] = None
    ) -> TriggerCondition:
        """
        Check for market crash trigger (benchmark -5%+).

        Args:
            country: Country code
            benchmark: Benchmark index name

        Returns:
            TriggerCondition result
        """
        # Determine benchmark table and column
        if country == "KR":
            table = "kr_benchmark_index"
            if not benchmark:
                benchmark = "KOSPI"
        else:
            table = "market_index"
            if not benchmark:
                benchmark = "S&P 500"

        try:
            # Get latest 2 days of benchmark data
            query = f"""
            SELECT date, close
            FROM {table}
            WHERE index_name LIKE $1
            ORDER BY date DESC
            LIMIT 2
            """
            async with self.db.pool.acquire() as conn:
                rows = await conn.fetch(query, f"%{benchmark}%")

            if len(rows) < 2:
                return TriggerCondition(
                    trigger_type=TriggerType.MARKET_CRASH,
                    is_triggered=False,
                    severity=AlertSeverity.INFO,
                    details={"message": "Insufficient benchmark data"}
                )

            latest = float(rows[0]["close"])
            previous = float(rows[1]["close"])
            daily_return = ((latest - previous) / previous) * 100

            is_triggered = daily_return <= MARKET_CRASH_THRESHOLD

            return TriggerCondition(
                trigger_type=TriggerType.MARKET_CRASH,
                is_triggered=is_triggered,
                severity=AlertSeverity.CRITICAL if is_triggered else AlertSeverity.INFO,
                details={
                    "benchmark": benchmark,
                    "daily_return": round(daily_return, 2),
                    "threshold": MARKET_CRASH_THRESHOLD,
                    "latest_date": rows[0]["date"].isoformat() if rows[0]["date"] else None,
                },
                affected_symbols=[]
            )

        except Exception as e:
            logger.error(f"Failed to check market crash: {e}")
            return TriggerCondition(
                trigger_type=TriggerType.MARKET_CRASH,
                is_triggered=False,
                severity=AlertSeverity.INFO,
                details={"error": str(e)}
            )

    def should_skip_due_to_holding_period(
        self,
        holding: HoldingStatus,
        trigger_type: TriggerType
    ) -> bool:
        """
        Check if action should be skipped due to minimum holding period.

        Emergency triggers (STOP_LOSS, SUSPENDED) bypass holding period.

        Args:
            holding: Holding status
            trigger_type: Trigger type

        Returns:
            True if should skip (holding period not met)
        """
        # Emergency triggers bypass holding period
        emergency_triggers = {TriggerType.STOP_LOSS, TriggerType.SUSPENDED}
        if trigger_type in emergency_triggers:
            return False

        # Check holding period
        return holding.holding_days < MIN_HOLDING_DAYS

    def get_holdings_status(
        self,
        holdings: List[HoldingStatus]
    ) -> Dict[str, Any]:
        """
        Get summary of holdings status.

        Args:
            holdings: List of holding status

        Returns:
            Summary dict
        """
        if not holdings:
            return {"total": 0}

        total_value = sum(h.current_value for h in holdings)
        total_invested = sum(h.invested_amount for h in holdings)
        total_pnl = total_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        grade_up = [h for h in holdings if h.grade_change > 0]
        grade_down = [h for h in holdings if h.grade_change < 0]
        grade_same = [h for h in holdings if h.grade_change == 0]

        profitable = [h for h in holdings if h.unrealized_pnl > 0]
        losing = [h for h in holdings if h.unrealized_pnl < 0]

        return {
            "total": len(holdings),
            "total_value": total_value,
            "total_invested": total_invested,
            "total_pnl": total_pnl,
            "total_pnl_pct": round(total_pnl_pct, 2),
            "grade_improved": len(grade_up),
            "grade_dropped": len(grade_down),
            "grade_unchanged": len(grade_same),
            "profitable_count": len(profitable),
            "losing_count": len(losing),
            "suspended_count": len([h for h in holdings if h.is_suspended]),
        }
