# -*- coding: utf-8 -*-
"""
Rebalancing Database Operations

Database CRUD operations for portfolio rebalancing system.

File: portfolio/rebalancing/rebalancing_db.py
Created: 2025-12-29
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncpg

from db.db_manager import SharedDatabaseManager
from rebalancing.config import DB_TABLES, TRADING_HALT_DAYS, GRADE_NUMERIC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RebalancingDBManager:
    """
    Database manager for rebalancing operations.

    Handles all database read/write operations for the rebalancing system.
    Uses SharedDatabaseManager for connection pooling.
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        """
        Initialize database manager.

        Args:
            db_pool: Existing connection pool (optional)
        """
        self.pool = db_pool
        self._shared_db: Optional[SharedDatabaseManager] = None

    async def initialize(self) -> None:
        """Initialize database connection pool if not provided"""
        if self.pool is None:
            self._shared_db = SharedDatabaseManager()
            await self._shared_db.initialize()
            self.pool = self._shared_db.pool

    async def close(self) -> None:
        """Close database connection"""
        if self._shared_db:
            await self._shared_db.close()

    # ========================================================================
    # Portfolio Read Operations
    # ========================================================================

    async def get_portfolio_master(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """
        Get portfolio master information.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Portfolio master record or None
        """
        query = """
        SELECT
            portfolio_id,
            portfolio_name,
            country,
            risk_level,
            initial_budget,
            current_budget,
            current_stock_count,
            benchmark,
            max_weight_per_stock,
            max_weight_per_sector,
            rebalancing_frequency,
            created_at,
            analysis_date,
            next_rebalancing_date,
            status
        FROM portfolio_master
        WHERE portfolio_id = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, portfolio_id)
            return dict(row) if row else None

    async def get_all_active_portfolios(self) -> List[Dict[str, Any]]:
        """
        Get all active portfolios.

        Returns:
            List of active portfolio master records
        """
        query = """
        SELECT
            portfolio_id,
            portfolio_name,
            country,
            risk_level,
            initial_budget,
            current_budget,
            current_stock_count,
            benchmark,
            created_at,
            next_rebalancing_date,
            status
        FROM portfolio_master
        WHERE status = 'ACTIVE'
        ORDER BY created_at DESC
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]

    async def get_portfolio_holdings(
        self,
        portfolio_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get current holdings for a portfolio.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            List of holding records
        """
        query = """
        SELECT
            symbol,
            stock_name,
            country,
            sector,
            shares,
            avg_price,
            current_price,
            current_weight,
            invested_amount,
            current_value,
            unrealized_pnl,
            unrealized_pnl_pct,
            entry_date,
            entry_score,
            atr_pct,
            dynamic_stop_pct,
            dynamic_take_pct,
            peak_price,
            peak_date,
            trailing_stop_price,
            scale_out_stage,
            profit_protection_mode
        FROM portfolio_holdings
        WHERE portfolio_id = $1
        ORDER BY current_weight DESC
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, portfolio_id)
            return [dict(row) for row in rows]

    # ========================================================================
    # Grade Read Operations
    # ========================================================================

    async def get_stock_grades(
        self,
        symbols: List[str],
        country: str,
        target_date: Optional[date] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get quant grades for stocks.

        Args:
            symbols: List of stock symbols
            country: 'KR' or 'US'
            target_date: Target date (default: latest)

        Returns:
            Dict mapping symbol to grade info
        """
        if not symbols:
            return {}

        table = DB_TABLES["kr_grade"] if country == "KR" else DB_TABLES["us_grade"]

        if target_date:
            query = f"""
            SELECT DISTINCT ON (symbol)
                symbol,
                date,
                final_grade,
                final_score,
                conviction_score
            FROM {table}
            WHERE symbol = ANY($1) AND date <= $2
            ORDER BY symbol, date DESC
            """
            params = [symbols, target_date]
        else:
            query = f"""
            SELECT DISTINCT ON (symbol)
                symbol,
                date,
                final_grade,
                final_score,
                conviction_score
            FROM {table}
            WHERE symbol = ANY($1)
            ORDER BY symbol, date DESC
            """
            params = [symbols]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return {
            row["symbol"]: {
                "grade": row["final_grade"],
                "score": float(row["final_score"]) if row["final_score"] else 0,
                "conviction": float(row["conviction_score"]) if row["conviction_score"] else 0,
                "date": row["date"],
            }
            for row in rows
        }

    async def get_consecutive_buy_days(
        self,
        symbol: str,
        country: str,
        as_of_date: Optional[date] = None
    ) -> int:
        """
        Get consecutive days with buy grade for a stock.

        Args:
            symbol: Stock symbol
            country: 'KR' or 'US'
            as_of_date: Reference date (default: latest)

        Returns:
            Number of consecutive buy grade days
        """
        table = DB_TABLES["kr_grade"] if country == "KR" else DB_TABLES["us_grade"]
        buy_grades = ("강력 매수", "매수", "매수 고려")

        if as_of_date is None:
            as_of_date = date.today()

        query = f"""
        SELECT date, final_grade
        FROM {table}
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT 30
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, as_of_date)

        consecutive = 0
        for row in rows:
            if row["final_grade"] in buy_grades:
                consecutive += 1
            else:
                break

        return consecutive

    # ========================================================================
    # Price Read Operations
    # ========================================================================

    async def get_current_prices(
        self,
        symbols: List[str],
        country: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get current prices for stocks.

        Args:
            symbols: List of stock symbols
            country: 'KR' or 'US'

        Returns:
            Dict mapping symbol to price info
        """
        if not symbols:
            return {}

        table = DB_TABLES["kr_price"] if country == "KR" else DB_TABLES["us_price"]

        # Get latest date first
        date_query = f"SELECT MAX(date) as latest_date FROM {table}"
        async with self.pool.acquire() as conn:
            date_row = await conn.fetchrow(date_query)
            if not date_row or not date_row["latest_date"]:
                return {}

            latest_date = date_row["latest_date"]

            query = f"""
            SELECT symbol, date, close, volume
            FROM {table}
            WHERE symbol = ANY($1) AND date = $2
            """
            rows = await conn.fetch(query, symbols, latest_date)

        return {
            row["symbol"]: {
                "price": float(row["close"]),
                "volume": int(row["volume"]) if row["volume"] else 0,
                "date": row["date"],
            }
            for row in rows
        }

    async def get_exchange_rate(
        self,
        target_date: Optional[date] = None
    ) -> float:
        """
        Get USD/KRW exchange rate.

        Args:
            target_date: Target date (default: latest)

        Returns:
            Exchange rate (fallback: 1450.0)
        """
        try:
            if target_date:
                query = """
                SELECT data_value
                FROM exchange_rate
                WHERE (item_name1 LIKE '%미국%' OR item_name1 LIKE '%달러%')
                  AND time_value <= $1
                ORDER BY time_value DESC
                LIMIT 1
                """
                params = [target_date]
            else:
                query = """
                SELECT data_value
                FROM exchange_rate
                WHERE (item_name1 LIKE '%미국%' OR item_name1 LIKE '%달러%')
                ORDER BY time_value DESC
                LIMIT 1
                """
                params = []

            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, *params)

            if row and row["data_value"]:
                return float(row["data_value"])
            return 1450.0

        except Exception as e:
            logger.error(f"Failed to get exchange rate: {e}")
            return 1450.0

    # ========================================================================
    # Trading Halt Check
    # ========================================================================

    async def check_trading_halt(
        self,
        symbols: List[str],
        country: str
    ) -> Dict[str, bool]:
        """
        Check if stocks are trading halted.

        Trading halt is detected if:
        - Last data is 2+ days old, OR
        - Volume is 0 for 2+ consecutive days

        Args:
            symbols: List of stock symbols
            country: 'KR' or 'US'

        Returns:
            Dict mapping symbol to halt status (True = halted)
        """
        if not symbols:
            return {}

        table = DB_TABLES["kr_price"] if country == "KR" else DB_TABLES["us_price"]
        halt_days = TRADING_HALT_DAYS

        # Get latest trading date
        date_query = f"SELECT MAX(date) as latest_date FROM {table}"
        async with self.pool.acquire() as conn:
            date_row = await conn.fetchrow(date_query)
            if not date_row or not date_row["latest_date"]:
                return {s: True for s in symbols}  # No data = all halted

            market_latest = date_row["latest_date"]

            # Get each stock's latest data
            query = f"""
            SELECT DISTINCT ON (symbol)
                symbol, date, volume
            FROM {table}
            WHERE symbol = ANY($1)
            ORDER BY symbol, date DESC
            """
            rows = await conn.fetch(query, symbols)

        result = {}
        stock_data = {row["symbol"]: row for row in rows}

        for symbol in symbols:
            if symbol not in stock_data:
                result[symbol] = True  # No data = halted
                continue

            row = stock_data[symbol]
            days_behind = (market_latest - row["date"]).days

            # Check if data is too old
            if days_behind >= halt_days:
                result[symbol] = True
            # Check zero volume (would need more days check)
            elif row["volume"] == 0:
                result[symbol] = True
            else:
                result[symbol] = False

        return result

    # ========================================================================
    # Rebalancing Write Operations
    # ========================================================================

    async def save_rebalancing(
        self,
        rebalancing_id: str,
        portfolio_id: str,
        rebalancing_type: str,
        trigger_type: Optional[str],
        status: str,
        plan_date: date,
        total_sell_amount: float,
        total_buy_amount: float,
        total_fee: float,
        net_cashflow: float,
        expected_improvement: float = 0.0
    ) -> bool:
        """
        Save rebalancing event record.

        Args:
            Various rebalancing fields

        Returns:
            True if successful
        """
        query = """
        INSERT INTO portfolio_rebalancing (
            rebalancing_id,
            portfolio_id,
            rebalancing_type,
            trigger_type,
            status,
            plan_date,
            total_sell_amount,
            total_buy_amount,
            total_fee,
            net_cashflow,
            expected_improvement,
            created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (rebalancing_id) DO UPDATE SET
            status = EXCLUDED.status,
            total_sell_amount = EXCLUDED.total_sell_amount,
            total_buy_amount = EXCLUDED.total_buy_amount,
            total_fee = EXCLUDED.total_fee,
            net_cashflow = EXCLUDED.net_cashflow,
            updated_at = NOW()
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    rebalancing_id,
                    portfolio_id,
                    rebalancing_type,
                    trigger_type,
                    status,
                    plan_date,
                    total_sell_amount,
                    total_buy_amount,
                    total_fee,
                    net_cashflow,
                    expected_improvement,
                    datetime.now()
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save rebalancing: {e}")
            return False

    async def save_rebalancing_detail(
        self,
        rebalancing_id: str,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        amount: float,
        fee: float,
        reason: str,
        before_weight: float = 0.0,
        after_weight: float = 0.0
    ) -> bool:
        """
        Save rebalancing detail record.

        Args:
            Various detail fields

        Returns:
            True if successful
        """
        query = """
        INSERT INTO portfolio_rebalancing_detail (
            rebalancing_id,
            symbol,
            action,
            shares,
            price,
            amount,
            fee,
            reason,
            before_weight,
            after_weight,
            created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    rebalancing_id,
                    symbol,
                    action,
                    shares,
                    price,
                    amount,
                    fee,
                    reason,
                    before_weight,
                    after_weight,
                    datetime.now()
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save rebalancing detail: {e}")
            return False

    async def update_rebalancing_status(
        self,
        rebalancing_id: str,
        status: str,
        executed_at: Optional[datetime] = None
    ) -> bool:
        """
        Update rebalancing status.

        Args:
            rebalancing_id: Rebalancing ID
            status: New status
            executed_at: Execution timestamp

        Returns:
            True if successful
        """
        query = """
        UPDATE portfolio_rebalancing
        SET status = $1, executed_at = $2, updated_at = NOW()
        WHERE rebalancing_id = $3
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(query, status, executed_at, rebalancing_id)
            return True
        except Exception as e:
            logger.error(f"Failed to update rebalancing status: {e}")
            return False

    # ========================================================================
    # Transaction Write Operations
    # ========================================================================

    async def save_transaction(
        self,
        portfolio_id: str,
        symbol: str,
        transaction_type: str,
        shares: int,
        price: float,
        amount: float,
        fee: float,
        transaction_date: date,
        rebalancing_id: Optional[str] = None
    ) -> bool:
        """
        Save transaction record.

        Args:
            Various transaction fields

        Returns:
            True if successful
        """
        query = """
        INSERT INTO portfolio_transactions (
            portfolio_id,
            symbol,
            transaction_type,
            shares,
            price,
            amount,
            fee,
            transaction_date,
            rebalancing_id,
            created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    portfolio_id,
                    symbol,
                    transaction_type,
                    shares,
                    price,
                    amount,
                    fee,
                    transaction_date,
                    rebalancing_id,
                    datetime.now()
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save transaction: {e}")
            return False

    # ========================================================================
    # Holdings Update Operations
    # ========================================================================

    async def update_holding(
        self,
        portfolio_id: str,
        symbol: str,
        shares: int,
        avg_price: float,
        current_price: float,
        invested_amount: float,
        current_value: float,
        current_weight: float
    ) -> bool:
        """
        Update or insert holding record.

        Args:
            Various holding fields

        Returns:
            True if successful
        """
        query = """
        INSERT INTO portfolio_holdings (
            portfolio_id,
            symbol,
            shares,
            avg_price,
            current_price,
            invested_amount,
            current_value,
            current_weight,
            unrealized_pnl,
            unrealized_pnl_pct,
            updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (portfolio_id, symbol) DO UPDATE SET
            shares = EXCLUDED.shares,
            avg_price = EXCLUDED.avg_price,
            current_price = EXCLUDED.current_price,
            invested_amount = EXCLUDED.invested_amount,
            current_value = EXCLUDED.current_value,
            current_weight = EXCLUDED.current_weight,
            unrealized_pnl = EXCLUDED.unrealized_pnl,
            unrealized_pnl_pct = EXCLUDED.unrealized_pnl_pct,
            updated_at = EXCLUDED.updated_at
        """
        unrealized_pnl = current_value - invested_amount
        unrealized_pnl_pct = (unrealized_pnl / invested_amount * 100) if invested_amount > 0 else 0

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    portfolio_id,
                    symbol,
                    shares,
                    avg_price,
                    current_price,
                    invested_amount,
                    current_value,
                    current_weight,
                    unrealized_pnl,
                    unrealized_pnl_pct,
                    datetime.now()
                )
            return True
        except Exception as e:
            logger.error(f"Failed to update holding: {e}")
            return False

    async def delete_holding(
        self,
        portfolio_id: str,
        symbol: str
    ) -> bool:
        """
        Delete holding record (full exit).

        Args:
            portfolio_id: Portfolio ID
            symbol: Stock symbol

        Returns:
            True if successful
        """
        query = """
        DELETE FROM portfolio_holdings
        WHERE portfolio_id = $1 AND symbol = $2
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(query, portfolio_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to delete holding: {e}")
            return False

    async def update_scale_out_stage(
        self,
        portfolio_id: str,
        symbol: str,
        new_stage: int
    ) -> bool:
        """
        Update scale-out stage for a holding.

        Scale-out stages:
        - 0: No scale-out yet
        - 1: 33% sold at 1R profit
        - 2: 66% sold at 2R profit (33% remaining)
        - 3: Full exit via trailing stop

        Args:
            portfolio_id: Portfolio ID
            symbol: Stock symbol
            new_stage: New scale-out stage

        Returns:
            True if successful
        """
        query = """
        UPDATE portfolio_holdings
        SET scale_out_stage = $1,
            profit_protection_mode = TRUE,
            updated_at = NOW()
        WHERE portfolio_id = $2 AND symbol = $3
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(query, new_stage, portfolio_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to update scale-out stage: {e}")
            return False

    async def update_portfolio_master(
        self,
        portfolio_id: str,
        current_budget: float,
        current_stock_count: int,
        next_rebalancing_date: Optional[date] = None
    ) -> bool:
        """
        Update portfolio master after rebalancing.

        Args:
            portfolio_id: Portfolio ID
            current_budget: Current portfolio value
            current_stock_count: Number of stocks held
            next_rebalancing_date: Optional next rebalancing date

        Returns:
            True if successful
        """
        if next_rebalancing_date:
            query = """
            UPDATE portfolio_master
            SET
                current_budget = $1,
                current_stock_count = $2,
                next_rebalancing_date = $3,
                updated_at = NOW()
            WHERE portfolio_id = $4
            """
            params = [current_budget, current_stock_count, next_rebalancing_date, portfolio_id]
        else:
            query = """
            UPDATE portfolio_master
            SET
                current_budget = $1,
                current_stock_count = $2,
                updated_at = NOW()
            WHERE portfolio_id = $3
            """
            params = [current_budget, current_stock_count, portfolio_id]

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(query, *params)
            return True
        except Exception as e:
            logger.error(f"Failed to update portfolio master: {e}")
            return False

    # ========================================================================
    # Alert Operations
    # ========================================================================

    async def save_alert(
        self,
        alert_id: str,
        portfolio_id: str,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        symbol: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save alert record.

        Args:
            Various alert fields

        Returns:
            True if successful
        """
        import json

        query = """
        INSERT INTO portfolio_alerts (
            alert_id,
            portfolio_id,
            alert_type,
            severity,
            title,
            message,
            symbol,
            data,
            created_at,
            is_read,
            is_resolved
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    alert_id,
                    portfolio_id,
                    alert_type,
                    severity,
                    title,
                    message,
                    symbol,
                    json.dumps(data) if data else None,
                    datetime.now(),
                    False,
                    False
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
            return False

    async def get_unread_alerts(
        self,
        portfolio_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get unread alerts.

        Args:
            portfolio_id: Filter by portfolio (optional)

        Returns:
            List of unread alerts
        """
        if portfolio_id:
            query = """
            SELECT *
            FROM portfolio_alerts
            WHERE portfolio_id = $1 AND is_read = FALSE
            ORDER BY created_at DESC
            """
            params = [portfolio_id]
        else:
            query = """
            SELECT *
            FROM portfolio_alerts
            WHERE is_read = FALSE
            ORDER BY created_at DESC
            """
            params = []

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    # ========================================================================
    # Rebalancing History
    # ========================================================================

    async def get_rebalancing_history(
        self,
        portfolio_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get rebalancing history for a portfolio.

        Args:
            portfolio_id: Portfolio ID
            limit: Maximum records to return

        Returns:
            List of rebalancing records
        """
        query = """
        SELECT
            rebalancing_id,
            rebalancing_type,
            trigger_type,
            status,
            plan_date,
            executed_at,
            total_sell_amount,
            total_buy_amount,
            total_fee,
            net_cashflow,
            expected_improvement
        FROM portfolio_rebalancing
        WHERE portfolio_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, portfolio_id, limit)
            return [dict(row) for row in rows]

    async def get_rebalancing_details(
        self,
        rebalancing_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get rebalancing detail records.

        Args:
            rebalancing_id: Rebalancing ID

        Returns:
            List of detail records
        """
        query = """
        SELECT
            symbol,
            action,
            shares,
            price,
            amount,
            fee,
            reason,
            before_weight,
            after_weight
        FROM portfolio_rebalancing_detail
        WHERE rebalancing_id = $1
        ORDER BY action, symbol
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, rebalancing_id)
            return [dict(row) for row in rows]
