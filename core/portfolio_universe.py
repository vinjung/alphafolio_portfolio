# -*- coding: utf-8 -*-
"""
Portfolio Universe Filter

Filters stock universe based on liquidity, grade, and risk criteria.
Returns candidate stocks for portfolio selection.

File: create_portfolio/portfolio_universe.py
Created: 2025-12-24
"""

import os
import asyncio
import asyncpg
import logging
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal
from dotenv import load_dotenv

from models import (
    ProcessedInput,
    StockCandidate,
    Country,
    UniverseFilter as UniverseFilterModel,
)
from config import (
    UNIVERSE_FILTER,
    get_universe_filter,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Simple async database manager for portfolio creation"""

    def __init__(self):
        self.pool = None

    async def initialize(self):
        """Initialize connection pool"""
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL not found in environment variables")

        db_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

        self.pool = await asyncpg.create_pool(
            db_url,
            min_size=5,
            max_size=20,
            command_timeout=120,
        )
        logger.info("Database connection pool initialized")

    async def execute_query(self, query: str, *params) -> List[Dict]:
        """Execute query and return results as list of dicts"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")


class UniverseFilter:
    """
    Filter stock universe based on criteria.

    Applies filters:
    1. Minimum trading value (liquidity)
    2. Quant grade (매수 고려 이상)
    3. Confidence score >= 50
    4. risk_flag != EXTREME_RISK
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize with database manager.

        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager

    async def filter_universe(
        self,
        processed_input: ProcessedInput
    ) -> List[StockCandidate]:
        """
        Filter universe and return candidate stocks.

        Args:
            processed_input: Processed input with constraints

        Returns:
            List of StockCandidate objects
        """
        country = processed_input.request.country
        analysis_date = processed_input.analysis_date
        universe_filter = processed_input.constraints.universe_filter

        logger.info(f"Filtering universe for {country.value}, date: {analysis_date}")

        candidates = []

        if country == Country.KR:
            candidates = await self._filter_kr_universe(analysis_date, universe_filter)
        elif country == Country.US:
            candidates = await self._filter_us_universe(analysis_date, universe_filter)
        else:  # MIXED
            kr_candidates = await self._filter_kr_universe(analysis_date, universe_filter)
            us_candidates = await self._filter_us_universe(analysis_date, universe_filter)
            candidates = kr_candidates + us_candidates

        logger.info(f"Universe filter complete: {len(candidates)} candidates found")
        return candidates

    async def _filter_kr_universe(
        self,
        analysis_date: date,
        universe_filter: UniverseFilterModel
    ) -> List[StockCandidate]:
        """
        Filter Korean stock universe.

        Args:
            analysis_date: Analysis date
            universe_filter: Filter parameters

        Returns:
            List of StockCandidate for KR stocks
        """
        # Check if data exists for the exact analysis_date
        date_query = """
        SELECT date FROM kr_stock_grade WHERE date = $1 LIMIT 1
        """
        date_result = await self.db.execute_query(date_query, analysis_date)

        if not date_result:
            raise ValueError(f"KR data not available for date {analysis_date}")

        actual_date = analysis_date
        logger.info(f"KR Universe: Using date {actual_date}")

        # Build allowed grades list for SQL
        allowed_grades = universe_filter.allowed_grades
        if not allowed_grades:
            allowed_grades = ['강력 매수', '매수', '매수 고려']

        # Main query with trading value filter
        query = """
        WITH trading_values AS (
            SELECT
                symbol,
                AVG(trading_value) as avg_trading_value
            FROM kr_intraday_total
            WHERE date >= ($1::date - INTERVAL '20 days')::date
                AND date <= $1
            GROUP BY symbol
        )
        SELECT
            g.symbol,
            g.stock_name,
            g.final_score,
            g.final_grade,
            g.value_score,
            g.quality_score,
            g.momentum_score,
            g.growth_score,
            g.confidence_score,
            g.conviction_score,
            g.volatility_annual,
            g.max_drawdown_1y,
            g.beta,
            g.outlier_risk_score,
            g.risk_flag,
            g.sector_rotation_score,
            d.industry as sector,
            tv.avg_trading_value,
            g.foreign_net_30d,
            g.inst_net_30d
        FROM kr_stock_grade g
        LEFT JOIN kr_stock_detail d ON g.symbol = d.symbol
        LEFT JOIN trading_values tv ON g.symbol = tv.symbol
        WHERE g.date = $1
            AND g.final_grade = ANY($2)
            AND g.confidence_score >= $3
            AND (g.risk_flag IS NULL OR g.risk_flag != 'EXTREME_RISK')
            AND tv.avg_trading_value >= $4
            AND g.final_score IS NOT NULL
        ORDER BY g.final_score DESC
        """

        try:
            rows = await self.db.execute_query(
                query,
                actual_date,
                allowed_grades,
                universe_filter.min_confidence_score,
                universe_filter.min_trading_value
            )

            candidates = []
            for row in rows:
                candidate = StockCandidate(
                    symbol=row['symbol'],
                    stock_name=row['stock_name'] or row['symbol'],
                    country=Country.KR,
                    sector=row['sector'],
                    final_score=self._to_float(row['final_score']),
                    value_score=self._to_float(row['value_score']),
                    quality_score=self._to_float(row['quality_score']),
                    momentum_score=self._to_float(row['momentum_score']),
                    growth_score=self._to_float(row['growth_score']),
                    conviction_score=self._to_float(row['conviction_score']),
                    volatility_annual=self._to_float(row['volatility_annual']),
                    max_drawdown_1y=self._to_float(row['max_drawdown_1y']),
                    beta=self._to_float(row['beta']),
                    outlier_risk_score=self._to_float(row['outlier_risk_score']),
                    risk_flag=row['risk_flag'],
                    sector_rotation_score=self._to_float(row['sector_rotation_score']),
                    # 센티먼트/수급 지표 (4단계)
                    foreign_net_30d=self._to_float(row.get('foreign_net_30d')),
                    inst_net_30d=self._to_float(row.get('inst_net_30d')),
                )
                candidates.append(candidate)

            logger.info(f"KR Universe: {len(candidates)} stocks passed filter")
            return candidates

        except Exception as e:
            logger.error(f"KR Universe filter error: {e}")
            raise

    async def _filter_us_universe(
        self,
        analysis_date: date,
        universe_filter: UniverseFilterModel
    ) -> List[StockCandidate]:
        """
        Filter US stock universe.

        Args:
            analysis_date: Analysis date
            universe_filter: Filter parameters

        Returns:
            List of StockCandidate for US stocks
        """
        # Check if data exists for the exact analysis_date
        date_query = """
        SELECT date FROM us_stock_grade WHERE date = $1 LIMIT 1
        """
        date_result = await self.db.execute_query(date_query, analysis_date)

        if not date_result:
            raise ValueError(f"US data not available for date {analysis_date}")

        actual_date = analysis_date
        logger.info(f"US Universe: Using date {actual_date}")

        # Build allowed grades list for SQL
        allowed_grades = universe_filter.allowed_grades
        if not allowed_grades:
            allowed_grades = ['강력 매수', '매수', '매수 고려']

        # Get US-specific filter config
        us_config = get_universe_filter('US')
        min_trading_value_usd = us_config.get('min_trading_value', 1_000_000)

        # Main query with trading value filter
        # US uses volume * close as trading value proxy
        query = """
        WITH trading_values AS (
            SELECT
                symbol,
                AVG(volume * close) as avg_trading_value
            FROM us_daily
            WHERE date >= ($1::date - INTERVAL '20 days')::date
                AND date <= $1
            GROUP BY symbol
        )
        SELECT
            g.symbol,
            g.stock_name,
            g.final_score,
            g.final_grade,
            g.value_score,
            g.quality_score,
            g.momentum_score,
            g.growth_score,
            g.confidence_score,
            g.conviction_score,
            g.volatility_annual,
            g.max_drawdown_1y,
            g.beta,
            g.outlier_risk_score,
            g.risk_flag,
            g.sector_rotation_score,
            b.sector,
            tv.avg_trading_value,
            g.insider_signal,
            -- 애널리스트 평균 등급 계산 (1=Strong Buy ~ 5=Strong Sell, 낮을수록 좋음)
            CASE
                WHEN (COALESCE(b.analystratingstrongbuy,0) + COALESCE(b.analystratingstrongbuy,0) +
                      COALESCE(b.analystratingbuy,0) + COALESCE(b.analystratinghold,0) +
                      COALESCE(b.analystratingsell,0) + COALESCE(b.analystratingstrongsell,0)) > 0
                THEN (
                    1.0 * COALESCE(b.analystratingstrongbuy,0) +
                    2.0 * COALESCE(b.analystratingbuy,0) +
                    3.0 * COALESCE(b.analystratinghold,0) +
                    4.0 * COALESCE(b.analystratingsell,0) +
                    5.0 * COALESCE(b.analystratingstrongsell,0)
                ) / NULLIF(
                    COALESCE(b.analystratingstrongbuy,0) + COALESCE(b.analystratingbuy,0) +
                    COALESCE(b.analystratinghold,0) + COALESCE(b.analystratingsell,0) +
                    COALESCE(b.analystratingstrongsell,0), 0
                )
                ELSE NULL
            END as analyst_rating
        FROM us_stock_grade g
        LEFT JOIN us_stock_basic b ON g.symbol = b.symbol
        LEFT JOIN trading_values tv ON g.symbol = tv.symbol
        WHERE g.date = $1
            AND g.final_grade = ANY($2)
            AND g.confidence_score >= $3
            AND (g.risk_flag IS NULL OR g.risk_flag != 'EXTREME_RISK')
            AND tv.avg_trading_value >= $4
            AND g.final_score IS NOT NULL
        ORDER BY g.final_score DESC
        """

        try:
            rows = await self.db.execute_query(
                query,
                actual_date,
                allowed_grades,
                universe_filter.min_confidence_score,
                min_trading_value_usd
            )

            candidates = []
            for row in rows:
                candidate = StockCandidate(
                    symbol=row['symbol'],
                    stock_name=row['stock_name'] or row['symbol'],
                    country=Country.US,
                    sector=row['sector'],
                    final_score=self._to_float(row['final_score']),
                    value_score=self._to_float(row['value_score']),
                    quality_score=self._to_float(row['quality_score']),
                    momentum_score=self._to_float(row['momentum_score']),
                    growth_score=self._to_float(row['growth_score']),
                    conviction_score=self._to_float(row['conviction_score']),
                    volatility_annual=self._to_float(row['volatility_annual']),
                    max_drawdown_1y=self._to_float(row['max_drawdown_1y']),
                    beta=self._to_float(row['beta']),
                    outlier_risk_score=self._to_float(row['outlier_risk_score']),
                    risk_flag=row['risk_flag'],
                    sector_rotation_score=self._to_float(row['sector_rotation_score']),
                    # 센티먼트/수급 지표 (4단계)
                    insider_signal=self._to_float(row.get('insider_signal')),
                    analyst_rating=self._to_float(row.get('analyst_rating')),
                )
                candidates.append(candidate)

            logger.info(f"US Universe: {len(candidates)} stocks passed filter")
            return candidates

        except Exception as e:
            logger.error(f"US Universe filter error: {e}")
            raise

    def _to_float(self, value) -> Optional[float]:
        """Safely convert to float"""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


async def filter_universe(
    processed_input: ProcessedInput,
    pool: 'asyncpg.Pool' = None
) -> List[StockCandidate]:
    """
    Convenience function to filter universe.

    Args:
        processed_input: Processed input from PortfolioInputProcessor
        pool: Optional shared database pool. If None, creates own pool.

    Returns:
        List of StockCandidate objects
    """
    if pool is not None:
        # Use shared pool
        db = DatabaseManager()
        db.pool = pool
        universe_filter = UniverseFilter(db)
        candidates = await universe_filter.filter_universe(processed_input)
        return candidates
    else:
        # Create own pool (backward compatibility)
        db = DatabaseManager()
        await db.initialize()
        try:
            universe_filter = UniverseFilter(db)
            candidates = await universe_filter.filter_universe(processed_input)
            return candidates
        finally:
            await db.close()


# ============================================================================
# Test
# ============================================================================

async def test_universe_filter():
    """Test universe filtering"""
    from portfolio_input import process_portfolio_request

    print("=" * 70)
    print("Universe Filter - Test")
    print("=" * 70)

    # Test 1: KR Balanced
    print("\n[Test 1] KR Balanced - 10 stocks")
    processed = process_portfolio_request(
        budget=10_000_000,
        country='KR',
        risk_level='balanced',
        num_stocks=10
    )

    if not processed.validation.is_valid:
        print("  Input validation failed!")
        return

    db = DatabaseManager()
    await db.initialize()

    try:
        universe_filter = UniverseFilter(db)
        candidates = await universe_filter.filter_universe(processed)

        print(f"  Total candidates: {len(candidates)}")
        print(f"  Top 5 candidates:")
        for i, c in enumerate(candidates[:5], 1):
            print(f"    {i}. {c.symbol} {c.stock_name}: "
                  f"Score={c.final_score:.1f}, "
                  f"Sector={c.sector or 'N/A'}")

        # Sector distribution
        sectors = {}
        for c in candidates:
            sector = c.sector or 'Unknown'
            sectors[sector] = sectors.get(sector, 0) + 1

        print(f"\n  Sector distribution (top 5):")
        for sector, count in sorted(sectors.items(), key=lambda x: -x[1])[:5]:
            print(f"    {sector}: {count}")

    finally:
        await db.close()

    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(test_universe_filter())
