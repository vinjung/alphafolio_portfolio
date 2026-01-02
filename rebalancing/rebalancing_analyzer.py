# -*- coding: utf-8 -*-
"""
Rebalancing Analyzer

Analyzes portfolio for sell/buy candidates and weight adjustments.

File: portfolio/rebalancing/rebalancing_analyzer.py
Created: 2025-12-29
"""

import logging
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Tuple, Set

from rebalancing.models import (
    HoldingStatus,
    CandidateStock,
    SellCandidate,
    BuyCandidate,
    AnalysisResult,
    RebalancingAction,
    TriggerCheckResult,
    TriggerType,
)
from rebalancing.config import (
    GRADE_NUMERIC,
    GRADE_DROP_TRIGGER,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MIN_HOLDING_DAYS,
    MIN_CONSECUTIVE_BUY_DAYS,
    MIN_ENTRY_GRADE,
    MAX_WEIGHT_PER_SECTOR,
    get_recommended_action,
)
from rebalancing.rebalancing_db import RebalancingDBManager
from rebalancing.rebalancing_trigger import RebalancingTriggerChecker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RebalancingAnalyzer:
    """
    Analyzes portfolio for rebalancing decisions.

    Identifies:
    - Stocks to sell (full exit or partial)
    - Stocks to buy (new entry or increase)
    - Weight adjustments needed
    """

    def __init__(self, db: RebalancingDBManager):
        """
        Initialize analyzer.

        Args:
            db: RebalancingDBManager instance
        """
        self.db = db
        self.trigger_checker = RebalancingTriggerChecker(db)

    async def analyze_portfolio(
        self,
        portfolio_id: str,
        trigger_result: Optional[TriggerCheckResult] = None,
        analysis_date: Optional[date] = None
    ) -> AnalysisResult:
        """
        Perform full portfolio analysis for rebalancing.

        Args:
            portfolio_id: Portfolio ID
            trigger_result: Optional trigger check result
            analysis_date: Analysis date (default: today)

        Returns:
            AnalysisResult with all recommendations
        """
        if analysis_date is None:
            analysis_date = date.today()

        logger.info(f"Analyzing portfolio {portfolio_id} for rebalancing")

        # Get portfolio info
        portfolio = await self.db.get_portfolio_master(portfolio_id)
        if not portfolio:
            return AnalysisResult(
                portfolio_id=portfolio_id,
                analysis_date=analysis_date,
                total_value=0,
                cash_balance=0,
                holdings_count=0,
                has_actions=False
            )

        country = portfolio["country"]
        risk_level = portfolio["risk_level"]

        # Get holdings status
        holdings = await self.trigger_checker._get_holdings_status(
            portfolio_id, country, analysis_date
        )

        if not holdings:
            return AnalysisResult(
                portfolio_id=portfolio_id,
                analysis_date=analysis_date,
                total_value=float(portfolio.get("current_budget", 0)),
                cash_balance=float(portfolio.get("current_budget", 0)),
                holdings_count=0,
                has_actions=False
            )

        # Calculate totals
        total_value = sum(h.current_value for h in holdings)
        # Assume cash_balance is tracked separately or is 0
        cash_balance = 0.0  # TODO: Track cash balance in portfolio

        # Run trigger check if not provided
        if trigger_result is None:
            trigger_result = await self.trigger_checker.check_all_triggers(
                portfolio_id, analysis_date
            )

        # Analyze sell candidates
        sell_candidates = self._analyze_sell_candidates(
            holdings, risk_level, trigger_result
        )

        # Calculate available cash after sells
        total_sell_amount = sum(s.expected_amount for s in sell_candidates)
        available_for_buy = cash_balance + total_sell_amount

        # Get current sectors for diversification
        current_sectors = self._get_sector_distribution(holdings)

        # Analyze buy candidates
        buy_candidates = await self._analyze_buy_candidates(
            portfolio_id,
            country,
            risk_level,
            holdings,
            current_sectors,
            available_for_buy,
            analysis_date
        )

        total_buy_amount = sum(b.expected_amount for b in buy_candidates)
        net_cashflow = total_sell_amount - total_buy_amount

        has_actions = len(sell_candidates) > 0 or len(buy_candidates) > 0

        return AnalysisResult(
            portfolio_id=portfolio_id,
            analysis_date=analysis_date,
            total_value=total_value,
            cash_balance=cash_balance,
            holdings_count=len(holdings),
            sell_candidates=sell_candidates,
            buy_candidates=buy_candidates,
            total_sell_amount=total_sell_amount,
            total_buy_amount=total_buy_amount,
            net_cashflow=net_cashflow,
            has_actions=has_actions
        )

    def _analyze_sell_candidates(
        self,
        holdings: List[HoldingStatus],
        risk_level: str,
        trigger_result: TriggerCheckResult
    ) -> List[SellCandidate]:
        """
        Analyze holdings for sell candidates.

        Sell triggers:
        1. Grade dropped 2+ levels -> SELL or DECREASE
        2. Stop loss reached -> SELL
        3. Take profit reached -> DECREASE (partial)
        4. Trading halted -> SELL (when possible)
        5. Grade is "매도" -> SELL

        Args:
            holdings: List of holding status
            risk_level: Portfolio risk level
            trigger_result: Trigger check result

        Returns:
            List of sell candidates
        """
        candidates = []
        stop_loss_threshold = STOP_LOSS_PCT.get(risk_level, -12.0)
        take_profit_threshold = TAKE_PROFIT_PCT.get(risk_level, 25.0)

        # Build triggered symbols set for quick lookup
        triggered_symbols: Dict[str, Set[TriggerType]] = {}
        for cond in trigger_result.triggered_conditions:
            for symbol in cond.affected_symbols:
                if symbol not in triggered_symbols:
                    triggered_symbols[symbol] = set()
                triggered_symbols[symbol].add(cond.trigger_type)

        for h in holdings:
            action = None
            sell_ratio = 0.0
            reason = ""
            priority = 0

            triggers = triggered_symbols.get(h.symbol, set())

            # Priority 1: Trading halted (CRITICAL)
            if h.is_suspended or TriggerType.SUSPENDED in triggers:
                action = RebalancingAction.SELL
                sell_ratio = 1.0
                reason = "Trading halted - full exit required"
                priority = 100

            # Priority 2: Stop loss reached (CRITICAL)
            elif TriggerType.STOP_LOSS in triggers or h.unrealized_pnl_pct <= stop_loss_threshold:
                action = RebalancingAction.SELL
                sell_ratio = 1.0
                reason = f"Stop loss triggered ({h.unrealized_pnl_pct:.1f}% <= {stop_loss_threshold}%)"
                priority = 90

            # Priority 3: Grade is "매도" (SELL grade)
            elif h.current_grade == "매도":
                action = RebalancingAction.SELL
                sell_ratio = 1.0
                reason = f"Quant grade is SELL"
                priority = 80

            # Priority 4: Grade dropped 3+ levels
            elif h.grade_change <= -3:
                action = RebalancingAction.SELL
                sell_ratio = 1.0
                reason = f"Grade dropped 3+ levels ({h.entry_grade} -> {h.current_grade})"
                priority = 70

            # Priority 5: Grade dropped 2 levels
            elif h.grade_change == -2 or TriggerType.GRADE_DROP in triggers:
                # Check holding period for non-emergency
                if h.holding_days >= MIN_HOLDING_DAYS:
                    action = RebalancingAction.DECREASE
                    sell_ratio = 0.5  # Sell 50%
                    reason = f"Grade dropped 2 levels ({h.entry_grade} -> {h.current_grade})"
                    priority = 50
                else:
                    # Skip due to holding period
                    continue

            # Priority 6: Take profit reached
            elif TriggerType.TAKE_PROFIT in triggers or h.unrealized_pnl_pct >= take_profit_threshold:
                if h.holding_days >= MIN_HOLDING_DAYS:
                    action = RebalancingAction.DECREASE
                    sell_ratio = 0.3  # Sell 30% to lock in profits
                    reason = f"Take profit triggered ({h.unrealized_pnl_pct:.1f}% >= {take_profit_threshold}%)"
                    priority = 30
                else:
                    continue

            # Priority 7: Grade is "매도 고려"
            elif h.current_grade == "매도 고려":
                if h.holding_days >= MIN_HOLDING_DAYS:
                    action = RebalancingAction.DECREASE
                    sell_ratio = 0.3
                    reason = "Quant grade is SELL_CONSIDER"
                    priority = 20
                else:
                    continue

            if action:
                shares_to_sell = int(h.shares * sell_ratio)
                if shares_to_sell < 1:
                    shares_to_sell = 1  # At least 1 share

                # For full exit, sell all
                if action == RebalancingAction.SELL:
                    shares_to_sell = h.shares

                expected_amount = shares_to_sell * h.current_price

                candidates.append(SellCandidate(
                    symbol=h.symbol,
                    stock_name=h.stock_name,
                    action=action,
                    shares_to_sell=shares_to_sell,
                    current_shares=h.shares,
                    sell_ratio=sell_ratio,
                    expected_amount=expected_amount,
                    reason=reason,
                    priority=priority
                ))

        # Sort by priority (highest first)
        candidates.sort(key=lambda x: x.priority, reverse=True)

        return candidates

    async def _analyze_buy_candidates(
        self,
        portfolio_id: str,
        country: str,
        risk_level: str,
        holdings: List[HoldingStatus],
        current_sectors: Dict[str, float],
        available_cash: float,
        analysis_date: date
    ) -> List[BuyCandidate]:
        """
        Analyze potential buy candidates.

        Buy criteria:
        1. Quant grade >= "매수 고려"
        2. Consecutive buy grade >= 5 days
        3. Not already in portfolio (or increase existing)
        4. Sector diversification check
        5. Sufficient liquidity

        Args:
            portfolio_id: Portfolio ID
            country: Country code
            risk_level: Risk level
            holdings: Current holdings
            current_sectors: Current sector distribution
            available_cash: Available cash for buying
            analysis_date: Analysis date

        Returns:
            List of buy candidates
        """
        if available_cash <= 0:
            return []

        candidates = []
        current_symbols = {h.symbol for h in holdings}
        max_sector_weight = MAX_WEIGHT_PER_SECTOR.get(risk_level, 0.30)

        # Get buy grade candidates from grade table
        buy_grades = ["강력 매수", "매수", "매수 고려"]
        min_grade_num = GRADE_NUMERIC.get(MIN_ENTRY_GRADE, 4)

        # Query candidates with buy grades
        grade_table = "kr_stock_grade" if country == "KR" else "us_stock_grade"
        price_table = "kr_intraday_total" if country == "KR" else "us_daily"
        basic_table = "kr_stock_basic" if country == "KR" else "us_stock_basic"

        query = f"""
        WITH latest_grades AS (
            SELECT DISTINCT ON (symbol)
                g.symbol,
                g.final_grade,
                g.final_score,
                g.conviction_score,
                g.date
            FROM {grade_table} g
            WHERE g.date <= $1
              AND g.final_grade IN ('강력 매수', '매수', '매수 고려')
            ORDER BY g.symbol, g.date DESC
        ),
        latest_prices AS (
            SELECT DISTINCT ON (symbol)
                symbol, close, volume
            FROM {price_table}
            ORDER BY symbol, date DESC
        )
        SELECT
            lg.symbol,
            lg.final_grade,
            lg.final_score,
            lg.conviction_score,
            lp.close as current_price,
            lp.volume
        FROM latest_grades lg
        JOIN latest_prices lp ON lg.symbol = lp.symbol
        WHERE lp.volume > 0
        ORDER BY lg.final_score DESC
        LIMIT 50
        """

        try:
            async with self.db.pool.acquire() as conn:
                rows = await conn.fetch(query, analysis_date)
        except Exception as e:
            logger.error(f"Failed to query buy candidates: {e}")
            return []

        for row in rows:
            symbol = row["symbol"]

            # Skip if already holding
            if symbol in current_symbols:
                # Could consider INCREASE for existing holdings
                continue

            # Check consecutive buy days
            consecutive_days = await self.db.get_consecutive_buy_days(
                symbol, country, analysis_date
            )
            if consecutive_days < MIN_CONSECUTIVE_BUY_DAYS:
                continue

            # Get stock info for sector
            try:
                info_query = f"""
                SELECT stock_name, sector
                FROM {basic_table}
                WHERE symbol = $1
                """
                async with self.db.pool.acquire() as conn:
                    info_row = await conn.fetchrow(info_query, symbol)

                stock_name = info_row["stock_name"] if info_row else None
                sector = info_row["sector"] if info_row else "Unknown"
            except Exception:
                stock_name = None
                sector = "Unknown"

            # Check sector weight constraint
            current_sector_weight = current_sectors.get(sector, 0.0)
            if current_sector_weight >= max_sector_weight:
                continue  # Sector already at max

            current_price = float(row["current_price"])
            final_score = float(row["final_score"]) if row["final_score"] else 0

            candidates.append(CandidateStock(
                symbol=symbol,
                stock_name=stock_name,
                sector=sector,
                current_grade=row["final_grade"],
                consecutive_buy_days=consecutive_days,
                final_score=final_score,
                current_price=current_price
            ))

        # Sort by score and consecutive days
        candidates.sort(
            key=lambda x: (x.final_score, x.consecutive_buy_days),
            reverse=True
        )

        # Convert to BuyCandidates with allocation
        buy_candidates = []
        remaining_cash = available_cash

        for c in candidates[:10]:  # Top 10 candidates
            if remaining_cash <= 0:
                break

            # Allocate proportionally (simplified)
            allocation = min(remaining_cash * 0.2, remaining_cash)  # Max 20% per stock
            shares = int(allocation / c.current_price)

            if shares < 1:
                continue

            expected_amount = shares * c.current_price
            remaining_cash -= expected_amount

            buy_candidates.append(BuyCandidate(
                symbol=c.symbol,
                stock_name=c.stock_name,
                action=RebalancingAction.BUY,
                shares_to_buy=shares,
                current_shares=0,
                expected_amount=expected_amount,
                target_weight=0.0,  # Will be calculated in weight module
                reason=f"Buy grade ({c.current_grade}), {c.consecutive_buy_days} consecutive days, score={c.final_score:.1f}",
                priority=int(c.final_score)
            ))

        return buy_candidates

    def _analyze_increase_candidates(
        self,
        holdings: List[HoldingStatus],
        available_cash: float
    ) -> List[BuyCandidate]:
        """
        Analyze existing holdings for position increase.

        Increase criteria:
        1. Grade improved from entry
        2. Current weight < target weight
        3. Sufficient cash available

        Args:
            holdings: Current holdings
            available_cash: Available cash

        Returns:
            List of increase candidates
        """
        candidates = []

        for h in holdings:
            # Only consider grade improvements
            if h.grade_change <= 0:
                continue

            # Skip if already at high weight
            if h.current_weight >= 0.15:  # 15% max
                continue

            # Calculate increase amount
            increase_amount = min(available_cash * 0.1, available_cash)
            shares = int(increase_amount / h.current_price)

            if shares < 1:
                continue

            candidates.append(BuyCandidate(
                symbol=h.symbol,
                stock_name=h.stock_name,
                action=RebalancingAction.INCREASE,
                shares_to_buy=shares,
                current_shares=h.shares,
                expected_amount=shares * h.current_price,
                target_weight=h.current_weight + 0.05,
                reason=f"Grade improved ({h.entry_grade} -> {h.current_grade})",
                priority=h.grade_change * 10
            ))

        candidates.sort(key=lambda x: x.priority, reverse=True)
        return candidates

    def _get_sector_distribution(
        self,
        holdings: List[HoldingStatus]
    ) -> Dict[str, float]:
        """
        Calculate current sector weight distribution.

        Args:
            holdings: Current holdings

        Returns:
            Dict mapping sector to weight
        """
        total_value = sum(h.current_value for h in holdings)
        if total_value == 0:
            return {}

        sectors: Dict[str, float] = {}
        for h in holdings:
            sector = h.sector or "Unknown"
            if sector not in sectors:
                sectors[sector] = 0.0
            sectors[sector] += h.current_value / total_value

        return sectors

    def summarize_analysis(self, result: AnalysisResult) -> str:
        """
        Generate human-readable summary of analysis.

        Args:
            result: Analysis result

        Returns:
            Summary string
        """
        lines = [
            f"Portfolio Analysis Summary ({result.analysis_date})",
            f"=" * 50,
            f"Holdings: {result.holdings_count} stocks",
            f"Total Value: {result.total_value:,.0f}",
            f"Cash Balance: {result.cash_balance:,.0f}",
            "",
        ]

        if result.sell_candidates:
            lines.append(f"SELL Candidates ({len(result.sell_candidates)}):")
            for s in result.sell_candidates:
                lines.append(
                    f"  - {s.symbol}: {s.action.value} {s.shares_to_sell} shares "
                    f"({s.expected_amount:,.0f}) - {s.reason}"
                )
            lines.append("")

        if result.buy_candidates:
            lines.append(f"BUY Candidates ({len(result.buy_candidates)}):")
            for b in result.buy_candidates:
                lines.append(
                    f"  - {b.symbol}: {b.action.value} {b.shares_to_buy} shares "
                    f"({b.expected_amount:,.0f}) - {b.reason}"
                )
            lines.append("")

        lines.extend([
            f"Total Sell: {result.total_sell_amount:,.0f}",
            f"Total Buy: {result.total_buy_amount:,.0f}",
            f"Net Cashflow: {result.net_cashflow:,.0f}",
        ])

        return "\n".join(lines)
