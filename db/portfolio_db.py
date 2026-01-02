# -*- coding: utf-8 -*-
"""
Portfolio Database Operations

Saves portfolio data to PostgreSQL database.

File: create_portfolio/portfolio_db.py
Created: 2025-12-24
"""

import os
import asyncio
import asyncpg
import logging
from datetime import datetime, date
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

from models import (
    PortfolioResponse,
    PortfolioStock,
    ProcessedInput,
    Country,
)

# Load .env from quant directory
env_path = Path(__file__).parent.parent / 'quant' / '.env'
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioDBManager:
    """
    Portfolio database manager.

    Saves portfolio data to:
    - portfolio_master
    - portfolio_holdings
    - portfolio_transactions
    """

    def __init__(self, db_pool=None):
        """
        Initialize database manager.

        Args:
            db_pool: Optional asyncpg connection pool
        """
        self.pool = db_pool
        self._owns_pool = False

    async def initialize(self):
        """Initialize database connection if not provided"""
        if self.pool is None:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise ValueError("DATABASE_URL not found")

            db_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
            self.pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
            self._owns_pool = True
            logger.info("Database pool initialized for portfolio saving")

    async def close(self):
        """Close database connection if owned"""
        if self._owns_pool and self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    async def save_portfolio(
        self,
        response: PortfolioResponse,
        processed_input: ProcessedInput,
        selection_data: dict = None
    ) -> bool:
        """
        Save complete portfolio to database.

        Args:
            response: PortfolioResponse from generator
            processed_input: ProcessedInput with constraints
            selection_data: Optional dict with selection details (consecutive_buy_days, etc.)

        Returns:
            True if successful, False otherwise
        """
        if not response.success:
            logger.warning("Cannot save failed portfolio generation")
            return False

        await self.initialize()

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 1. Save portfolio_master
                    await self._save_master(conn, response, processed_input)

                    # 2. Save portfolio_holdings
                    await self._save_holdings(
                        conn, response, processed_input, selection_data
                    )

                    # 3. Save portfolio_transactions (initial buy)
                    await self._save_transactions(conn, response, processed_input)

            logger.info(f"Portfolio {response.portfolio_id} saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save portfolio: {e}")
            return False

    async def _save_master(
        self,
        conn: asyncpg.Connection,
        response: PortfolioResponse,
        processed_input: ProcessedInput
    ):
        """Save to portfolio_master table"""
        request = response.request
        constraints = processed_input.constraints

        # Determine benchmark (user input > default by country)
        if request.benchmark:
            benchmark = request.benchmark
        elif request.country == Country.KR:
            benchmark = 'KOSPI200'
        elif request.country == Country.US:
            benchmark = 'S&P500'
        else:
            benchmark = 'KOSPI200'  # MIXED default to KOSPI200 if not specified

        query = """
        INSERT INTO portfolio_master (
            portfolio_id,
            portfolio_name,
            portfolio_description,
            status,
            country,
            risk_level,
            initial_budget,
            current_budget,
            target_stock_count,
            current_stock_count,
            benchmark,
            max_weight_per_stock,
            max_weight_per_sector,
            min_consecutive_buy_days,
            created_by,
            created_at,
            activated_at,
            analysis_date,
            updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19
        )
        """

        await conn.execute(
            query,
            response.portfolio_id,
            response.portfolio_name,
            f"Auto-generated {request.risk_level.value} portfolio for {request.country.value}",
            'ACTIVE',
            request.country.value,
            request.risk_level.value,
            request.budget,
            int(response.summary.total_investment) if response.summary else request.budget,
            request.num_stocks,
            len(response.stocks),
            benchmark,
            constraints.weight_constraints.max_weight_per_stock,
            constraints.weight_constraints.max_weight_per_sector,
            request.min_consecutive_buy_days if request.min_consecutive_buy_days else 5,
            'system',
            response.created_at,
            response.created_at,
            processed_input.analysis_date,
            response.created_at
        )

        logger.info(f"  Saved portfolio_master: {response.portfolio_id}")

    async def _save_holdings(
        self,
        conn: asyncpg.Connection,
        response: PortfolioResponse,
        processed_input: ProcessedInput,
        selection_data: dict = None
    ):
        """Save to portfolio_holdings table"""
        if not response.stocks:
            return

        # Get consecutive_buy_days from selection_data if available
        consecutive_days_map = {}
        if selection_data and 'consecutive_buy_days' in selection_data:
            consecutive_days_map = selection_data['consecutive_buy_days']

        query = """
        INSERT INTO portfolio_holdings (
            portfolio_id,
            symbol,
            stock_name,
            country,
            sector,
            status,
            shares,
            avg_price,
            current_price,
            entry_weight,
            current_weight,
            invested_amount,
            current_value,
            unrealized_pnl,
            unrealized_pnl_pct,
            entry_date,
            entry_score,
            entry_reason,
            consecutive_buy_days,
            price_updated_at,
            updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21
        )
        """

        for stock in response.stocks:
            # Get consecutive_buy_days for this stock
            consec_days = consecutive_days_map.get(stock.symbol, 0)

            # Calculate values
            current_value = stock.shares * stock.current_price
            invested_amount = stock.amount
            unrealized_pnl = current_value - invested_amount
            unrealized_pnl_pct = (unrealized_pnl / invested_amount * 100) if invested_amount > 0 else 0

            # Build entry reason from selection_reasons
            entry_reason = '; '.join(stock.selection_reasons) if stock.selection_reasons else None

            await conn.execute(
                query,
                response.portfolio_id,
                stock.symbol,
                stock.stock_name,
                stock.country.value,
                stock.sector,
                'ACTIVE',
                stock.shares,
                stock.current_price,  # avg_price = current_price at entry
                stock.current_price,
                stock.weight,  # entry_weight
                stock.weight,  # current_weight (same at creation)
                invested_amount,
                current_value,
                unrealized_pnl,
                unrealized_pnl_pct,
                processed_input.analysis_date,
                stock.final_score,
                entry_reason,
                consec_days,
                datetime.now(),
                datetime.now()
            )

        logger.info(f"  Saved {len(response.stocks)} holdings")

    async def _save_transactions(
        self,
        conn: asyncpg.Connection,
        response: PortfolioResponse,
        processed_input: ProcessedInput
    ):
        """Save initial buy transactions to portfolio_transactions table"""
        if not response.stocks:
            return

        query = """
        INSERT INTO portfolio_transactions (
            portfolio_id,
            rebalancing_id,
            symbol,
            stock_name,
            country,
            transaction_type,
            transaction_date,
            shares,
            price,
            amount,
            fee,
            tax,
            net_amount,
            shares_before,
            shares_after,
            avg_price_before,
            avg_price_after,
            realized_pnl,
            transaction_reason,
            notes,
            created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21
        )
        """

        # analysis_date를 거래일로 사용 (백테스트 모드)
        transaction_date = processed_input.analysis_date

        for stock in response.stocks:
            amount = stock.shares * stock.current_price

            # Estimate fee (0.015% for KR)
            fee = amount * 0.00015 if stock.country == Country.KR else 0

            await conn.execute(
                query,
                response.portfolio_id,
                None,  # rebalancing_id (NULL for initial)
                stock.symbol,
                stock.stock_name,
                stock.country.value,
                'BUY',
                transaction_date,
                stock.shares,
                stock.current_price,
                amount,
                fee,
                0,  # tax (no tax on buy)
                amount + fee,  # net_amount
                0,  # shares_before
                stock.shares,  # shares_after
                None,  # avg_price_before
                stock.current_price,  # avg_price_after
                None,  # realized_pnl (no PnL on buy)
                'INITIAL',
                'Initial portfolio creation',
                datetime.now()
            )

        logger.info(f"  Saved {len(response.stocks)} transactions")


async def save_portfolio(
    response: PortfolioResponse,
    processed_input: ProcessedInput,
    selection_data: dict = None,
    pool: 'asyncpg.Pool' = None
) -> bool:
    """
    Convenience function to save portfolio.

    Args:
        response: PortfolioResponse from generator
        processed_input: ProcessedInput with constraints
        selection_data: Optional dict with selection details
        pool: Optional shared database pool. If None, creates own pool.

    Returns:
        True if successful, False otherwise
    """
    if pool is not None:
        # Use shared pool (don't close it)
        db_manager = PortfolioDBManager(db_pool=pool)
        db_manager._owns_pool = False
        return await db_manager.save_portfolio(response, processed_input, selection_data)
    else:
        # Create own pool (backward compatibility)
        db_manager = PortfolioDBManager()
        try:
            return await db_manager.save_portfolio(response, processed_input, selection_data)
        finally:
            await db_manager.close()


# ============================================================================
# Test
# ============================================================================

async def test_db_connection():
    """Test database connection"""
    print("=" * 70)
    print("Portfolio DB Manager - Connection Test")
    print("=" * 70)

    db_manager = PortfolioDBManager()
    try:
        await db_manager.initialize()

        async with db_manager.pool.acquire() as conn:
            # Test query
            result = await conn.fetchval("SELECT COUNT(*) FROM portfolio_master")
            print(f"  portfolio_master count: {result}")

            result = await conn.fetchval("SELECT COUNT(*) FROM portfolio_holdings")
            print(f"  portfolio_holdings count: {result}")

            result = await conn.fetchval("SELECT COUNT(*) FROM portfolio_transactions")
            print(f"  portfolio_transactions count: {result}")

        print("\n  Connection test successful!")

    except Exception as e:
        print(f"\n  Connection test failed: {e}")
    finally:
        await db_manager.close()

    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(test_db_connection())
