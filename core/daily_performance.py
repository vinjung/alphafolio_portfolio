# -*- coding: utf-8 -*-
"""
Daily Performance Calculator

Calculates and stores daily portfolio performance including:
- Daily/cumulative returns
- Benchmark returns
- Excess returns

File: portfolio/core/daily_performance.py
Created: 2025-12-30
"""

import os
import asyncio
import asyncpg
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from decimal import Decimal
import numpy as np
from dotenv import load_dotenv

# Load .env from quant directory (../../quant/.env from core/)
env_path = Path(__file__).parent.parent.parent / 'quant' / '.env'
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Benchmark mapping
# market_index has: KOSDAQ, NASDAQ, KOSPI, DOW, S&P500
# kr_benchmark_index has: 코스피 200, etc.
BENCHMARK_MAPPING = {
    # Korean benchmarks - use market_index
    'KOSPI': ('market_index', 'KOSPI'),
    'KOSDAQ': ('market_index', 'KOSDAQ'),
    # Korean benchmarks - use kr_benchmark_index
    'KOSPI200': ('kr_benchmark_index', '코스피 200'),
    'KRX100': ('kr_benchmark_index', 'KRX 100'),
    # US benchmarks - use market_index
    'S&P500': ('market_index', 'S&P500'),
    'SPX': ('market_index', 'S&P500'),
    'NASDAQ': ('market_index', 'NASDAQ'),
    'DOW': ('market_index', 'DOW'),
    'DOW30': ('market_index', 'DOW'),
}


class DailyPerformanceCalculator:
    """
    Calculates and stores daily portfolio performance.

    Data sources:
    - Korean stock prices: kr_intraday_total
    - US stock prices: us_daily
    - Korean benchmark: kr_benchmark_index
    - US benchmark: market_index
    """

    def __init__(self, db_pool=None):
        """Initialize calculator with optional database pool."""
        self.pool = db_pool
        self._owns_pool = False

    async def initialize(self):
        """Initialize database connection if not provided."""
        if self.pool is None:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise ValueError("DATABASE_URL not found")

            db_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
            self.pool = await asyncpg.create_pool(
                db_url,
                min_size=2,
                max_size=10,
                command_timeout=300
            )
            self._owns_pool = True
            logger.info("Database pool initialized for daily performance")

    async def close(self):
        """Close database connection if owned."""
        if self._owns_pool and self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    async def get_active_portfolios(self) -> List[Dict[str, Any]]:
        """Get all active portfolios."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT portfolio_id, portfolio_name, country, benchmark,
                       analysis_date, initial_budget
                FROM portfolio_master
                WHERE status = 'ACTIVE'
                ORDER BY analysis_date DESC
            """)
            return [dict(r) for r in rows]

    async def get_portfolio_holdings(
        self,
        portfolio_id: str
    ) -> List[Dict[str, Any]]:
        """Get holdings for a portfolio."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT symbol, stock_name, country, shares, avg_price,
                       invested_amount, entry_date,
                       atr_pct, dynamic_stop_pct, dynamic_take_pct,
                       peak_price, peak_date, trailing_stop_price,
                       scale_out_stage, profit_protection_mode
                FROM portfolio_holdings
                WHERE portfolio_id = $1 AND status = 'ACTIVE'
            """, portfolio_id)
            return [dict(r) for r in rows]

    async def get_stock_prices(
        self,
        symbols: List[str],
        country: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, Dict[date, Decimal]]:
        """
        Get historical stock prices.

        Returns: {symbol: {date: close_price}}
        """
        result: Dict[str, Dict[date, Decimal]] = {}

        if not symbols:
            return result

        async with self.pool.acquire() as conn:
            if country == 'KR':
                rows = await conn.fetch("""
                    SELECT symbol, date, close
                    FROM kr_intraday_total
                    WHERE symbol = ANY($1)
                    AND date >= $2 AND date <= $3
                    ORDER BY symbol, date
                """, symbols, start_date, end_date)
            else:  # US or MIXED
                rows = await conn.fetch("""
                    SELECT symbol, date, close
                    FROM us_daily
                    WHERE symbol = ANY($1)
                    AND date >= $2 AND date <= $3
                    ORDER BY symbol, date
                """, symbols, start_date, end_date)

            for row in rows:
                symbol = row['symbol']
                if symbol not in result:
                    result[symbol] = {}
                result[symbol][row['date']] = row['close']

        return result

    async def get_benchmark_prices(
        self,
        benchmark: str,
        country: str,
        start_date: date,
        end_date: date
    ) -> Dict[date, Decimal]:
        """
        Get benchmark index prices from appropriate table.

        Returns: {date: close_price}
        """
        result: Dict[date, Decimal] = {}

        # Get table and column name from mapping
        mapping = BENCHMARK_MAPPING.get(benchmark)
        if mapping:
            table_name, index_name = mapping
        else:
            # Default to market_index with benchmark as exchange name
            table_name = 'market_index'
            index_name = benchmark

        async with self.pool.acquire() as conn:
            if table_name == 'kr_benchmark_index':
                rows = await conn.fetch("""
                    SELECT date, close
                    FROM kr_benchmark_index
                    WHERE index_name = $1
                    AND date >= $2 AND date <= $3
                    ORDER BY date
                """, index_name, start_date, end_date)
            else:  # market_index
                rows = await conn.fetch("""
                    SELECT date, close
                    FROM market_index
                    WHERE exchange = $1
                    AND date >= $2 AND date <= $3
                    ORDER BY date
                """, index_name, start_date, end_date)

            for row in rows:
                result[row['date']] = row['close']

        return result

    async def get_last_performance_date(
        self,
        portfolio_id: str
    ) -> Optional[date]:
        """Get the last date with performance data."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT MAX(date) as last_date
                FROM portfolio_daily_performance
                WHERE portfolio_id = $1
            """, portfolio_id)
            return row['last_date'] if row and row['last_date'] else None

    def calculate_daily_values(
        self,
        holdings: List[Dict[str, Any]],
        stock_prices: Dict[str, Dict[date, Decimal]],
        dates: List[date]
    ) -> Dict[date, Dict[str, Any]]:
        """
        Calculate portfolio value for each date.

        Returns: {date: {total_value, invested, stock_values}}
        """
        result = {}

        # Calculate total invested amount (constant)
        total_invested = sum(
            float(h['invested_amount'] or h['shares'] * h['avg_price'])
            for h in holdings
        )

        for dt in dates:
            stock_values = {}
            total_value = 0.0
            winning = 0
            losing = 0
            best_symbol = None
            best_return = float('-inf')
            worst_symbol = None
            worst_return = float('inf')

            for h in holdings:
                symbol = h['symbol']
                shares = h['shares']

                # Handle None avg_price
                avg_price_raw = h['avg_price']
                if avg_price_raw is None:
                    continue  # Skip holdings without avg_price
                avg_price = float(avg_price_raw)

                invested_raw = h.get('invested_amount')
                invested = float(invested_raw) if invested_raw else shares * avg_price

                # Get price for this date
                if symbol in stock_prices and dt in stock_prices[symbol]:
                    price_raw = stock_prices[symbol][dt]
                    price = float(price_raw) if price_raw else avg_price
                else:
                    # Use average price if no market price
                    price = avg_price

                value = shares * price
                pnl_pct = ((price / avg_price) - 1) * 100 if avg_price > 0 else 0

                stock_values[symbol] = {
                    'value': value,
                    'price': price,
                    'pnl_pct': pnl_pct
                }
                total_value += value

                if pnl_pct > 0:
                    winning += 1
                elif pnl_pct < 0:
                    losing += 1

                if pnl_pct > best_return:
                    best_return = pnl_pct
                    best_symbol = symbol
                if pnl_pct < worst_return:
                    worst_return = pnl_pct
                    worst_symbol = symbol

            result[dt] = {
                'total_value': total_value,
                'total_invested': total_invested,
                'stock_count': len(holdings),
                'winning_stocks': winning,
                'losing_stocks': losing,
                'best_performer_symbol': best_symbol,
                'best_performer_return': best_return if best_return != float('-inf') else 0,
                'worst_performer_symbol': worst_symbol,
                'worst_performer_return': worst_return if worst_return != float('inf') else 0,
                'stock_values': stock_values
            }

        return result

    def calculate_returns(
        self,
        daily_values: Dict[date, Dict[str, Any]],
        benchmark_prices: Dict[date, Decimal],
        analysis_date: date
    ) -> List[Dict[str, Any]]:
        """
        Calculate returns for each date.

        Returns list of performance records sorted by date.
        """
        results = []
        sorted_dates = sorted(daily_values.keys())

        if not sorted_dates:
            return results

        # Get initial values
        initial_value = daily_values[sorted_dates[0]]['total_invested']
        initial_benchmark_raw = benchmark_prices.get(analysis_date)
        initial_benchmark = float(initial_benchmark_raw) if initial_benchmark_raw else 0

        # If no benchmark on analysis_date, use first available
        if initial_benchmark == 0 and benchmark_prices:
            for dt in sorted(benchmark_prices.keys()):
                if dt >= analysis_date:
                    val = benchmark_prices[dt]
                    if val:
                        initial_benchmark = float(val)
                        break

        prev_value = initial_value
        peak_value = initial_value

        for i, dt in enumerate(sorted_dates):
            data = daily_values[dt]
            current_value = data['total_value']

            # Daily return
            if prev_value > 0:
                daily_return = ((current_value / prev_value) - 1) * 100
            else:
                daily_return = 0.0

            # Cumulative return from initial investment
            if initial_value > 0:
                cumulative_return = ((current_value / initial_value) - 1) * 100
            else:
                cumulative_return = 0.0

            # Benchmark returns
            current_benchmark_raw = benchmark_prices.get(dt)
            current_benchmark = float(current_benchmark_raw) if current_benchmark_raw else initial_benchmark
            if initial_benchmark > 0:
                benchmark_cumulative = ((current_benchmark / initial_benchmark) - 1) * 100
            else:
                benchmark_cumulative = 0.0

            # Daily benchmark return
            if i > 0:
                prev_date = sorted_dates[i - 1]
                prev_benchmark_raw = benchmark_prices.get(prev_date)
                prev_benchmark = float(prev_benchmark_raw) if prev_benchmark_raw else current_benchmark
                if prev_benchmark > 0:
                    benchmark_daily = ((current_benchmark / prev_benchmark) - 1) * 100
                else:
                    benchmark_daily = 0.0
            else:
                benchmark_daily = 0.0

            # Excess return
            excess_return = cumulative_return - benchmark_cumulative

            # Drawdown
            if current_value > peak_value:
                peak_value = current_value

            if peak_value > 0:
                current_drawdown = ((current_value / peak_value) - 1) * 100
            else:
                current_drawdown = 0.0

            results.append({
                'date': dt,
                'total_invested': initial_value,
                'total_value': current_value,
                'cash_balance': 0,  # Assuming fully invested
                'total_portfolio_value': current_value,
                'daily_return': daily_return,
                'cumulative_return': cumulative_return,
                'benchmark_daily_return': benchmark_daily,
                'benchmark_cumulative_return': benchmark_cumulative,
                'excess_return': excess_return,
                'current_drawdown': current_drawdown,
                'stock_count': data['stock_count'],
                'winning_stocks': data['winning_stocks'],
                'losing_stocks': data['losing_stocks'],
                'best_performer_symbol': data['best_performer_symbol'],
                'best_performer_return': data['best_performer_return'],
                'worst_performer_symbol': data['worst_performer_symbol'],
                'worst_performer_return': data['worst_performer_return'],
            })

            prev_value = current_value

        # Calculate rolling metrics (30-day volatility, max drawdown, sharpe)
        if len(results) >= 2:
            returns = [r['daily_return'] for r in results]

            for i, r in enumerate(results):
                # 30-day volatility
                if i >= 29:
                    window = returns[i-29:i+1]
                    r['volatility_30d'] = float(np.std(window)) * np.sqrt(252)
                else:
                    r['volatility_30d'] = None

                # Max drawdown up to this point
                max_dd = min(r2['current_drawdown'] for r2 in results[:i+1])
                r['max_drawdown'] = max_dd

                # 30-day Sharpe ratio (assuming 3% risk-free rate)
                if i >= 29 and r['volatility_30d'] and r['volatility_30d'] > 0:
                    window_returns = returns[i-29:i+1]
                    avg_return = np.mean(window_returns) * 252  # Annualized
                    r['sharpe_ratio_30d'] = (avg_return - 3) / r['volatility_30d']
                else:
                    r['sharpe_ratio_30d'] = None

        return results

    def _safe_round(self, value: Any, decimals: int = 4, max_val: float = 9999.9999) -> Optional[float]:
        """
        Safely round a numeric value to prevent overflow.

        Args:
            value: Value to round
            decimals: Decimal places
            max_val: Maximum allowed value

        Returns:
            Rounded value or None
        """
        if value is None:
            return None
        try:
            rounded = round(float(value), decimals)
            # Clamp to prevent numeric overflow
            if rounded > max_val:
                return max_val
            if rounded < -max_val:
                return -max_val
            return rounded
        except (TypeError, ValueError):
            return None

    async def update_holdings_risk_data(
        self,
        portfolio_id: str,
        holdings: List[Dict[str, Any]],
        stock_prices: Dict[str, Dict[date, Decimal]],
        target_date: date
    ) -> int:
        """
        Update peak prices and trailing stops for all holdings.

        This implements the Chandelier Exit trailing stop:
        - peak_price: Highest price since entry
        - trailing_stop_price: peak_price - (3 x ATR)

        Args:
            portfolio_id: Portfolio ID
            holdings: List of holdings
            stock_prices: {symbol: {date: price}}
            target_date: Date to update for

        Returns:
            Number of holdings updated
        """
        updated_count = 0

        async with self.pool.acquire() as conn:
            for h in holdings:
                symbol = h['symbol']

                # Get current price
                if symbol in stock_prices and target_date in stock_prices[symbol]:
                    current_price = float(stock_prices[symbol][target_date])
                else:
                    continue  # No price data for today

                # Get existing peak price
                old_peak = float(h['peak_price']) if h.get('peak_price') else float(h['avg_price'])
                atr_pct = float(h['atr_pct']) if h.get('atr_pct') else None

                # Update peak if current price is higher
                new_peak = old_peak
                new_peak_date = h.get('peak_date')
                if current_price > old_peak:
                    new_peak = current_price
                    new_peak_date = target_date

                # Calculate trailing stop (Chandelier Exit: peak - 3x ATR)
                trailing_stop = None
                if atr_pct and new_peak:
                    atr_amount = new_peak * (atr_pct / 100)
                    trailing_stop = new_peak - (3 * atr_amount)

                # Check if in profit protection mode (profitable + past 1R)
                avg_price = float(h['avg_price'])
                profit_pct = ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0
                profit_protection = False
                if atr_pct and profit_pct > atr_pct:  # Profit > 1R
                    profit_protection = True

                # Update database
                await conn.execute("""
                    UPDATE portfolio_holdings
                    SET current_price = $1,
                        peak_price = $2,
                        peak_date = $3,
                        trailing_stop_price = $4,
                        profit_protection_mode = $5,
                        updated_at = NOW()
                    WHERE portfolio_id = $6 AND symbol = $7
                """,
                    current_price,
                    new_peak,
                    new_peak_date,
                    trailing_stop,
                    profit_protection,
                    portfolio_id,
                    symbol
                )
                updated_count += 1

        return updated_count

    async def save_performance_records(
        self,
        portfolio_id: str,
        records: List[Dict[str, Any]]
    ) -> int:
        """Save performance records to database."""
        if not records:
            return 0

        async with self.pool.acquire() as conn:
            # Delete existing records for these dates
            dates = [r['date'] for r in records]
            await conn.execute("""
                DELETE FROM portfolio_daily_performance
                WHERE portfolio_id = $1 AND date = ANY($2)
            """, portfolio_id, dates)

            # Insert new records
            query = """
                INSERT INTO portfolio_daily_performance (
                    portfolio_id, date, total_invested, total_value,
                    cash_balance, total_portfolio_value,
                    daily_return, cumulative_return,
                    benchmark_daily_return, benchmark_cumulative_return,
                    excess_return, volatility_30d, max_drawdown,
                    current_drawdown, sharpe_ratio_30d,
                    stock_count, winning_stocks, losing_stocks,
                    best_performer_symbol, best_performer_return,
                    worst_performer_symbol, worst_performer_return,
                    created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23
                )
            """

            for r in records:
                await conn.execute(
                    query,
                    portfolio_id,
                    r['date'],
                    self._safe_round(r['total_invested'], 2, 999999999999.99),
                    self._safe_round(r['total_value'], 2, 999999999999.99),
                    self._safe_round(r['cash_balance'], 2, 999999999999.99),
                    self._safe_round(r['total_portfolio_value'], 2, 999999999999.99),
                    self._safe_round(r['daily_return']),
                    self._safe_round(r['cumulative_return']),
                    self._safe_round(r['benchmark_daily_return']),
                    self._safe_round(r['benchmark_cumulative_return']),
                    self._safe_round(r['excess_return']),
                    self._safe_round(r.get('volatility_30d')),
                    self._safe_round(r.get('max_drawdown')),
                    self._safe_round(r['current_drawdown']),
                    self._safe_round(r.get('sharpe_ratio_30d')),
                    r['stock_count'],
                    r['winning_stocks'],
                    r['losing_stocks'],
                    r['best_performer_symbol'],
                    self._safe_round(r['best_performer_return']),
                    r['worst_performer_symbol'],
                    self._safe_round(r['worst_performer_return']),
                    datetime.now()
                )

        return len(records)

    async def calculate_portfolio_performance(
        self,
        portfolio_id: str,
        force_recalculate: bool = False
    ) -> int:
        """
        Calculate and store daily performance for a portfolio.

        Args:
            portfolio_id: Portfolio ID
            force_recalculate: If True, recalculate all dates

        Returns:
            Number of records saved
        """
        # Get portfolio info
        async with self.pool.acquire() as conn:
            portfolio = await conn.fetchrow("""
                SELECT portfolio_id, portfolio_name, country, benchmark,
                       analysis_date, initial_budget
                FROM portfolio_master
                WHERE portfolio_id = $1
            """, portfolio_id)

            if not portfolio:
                logger.warning(f"Portfolio {portfolio_id} not found")
                return 0

        portfolio = dict(portfolio)
        analysis_date = portfolio['analysis_date']
        country = portfolio['country']
        benchmark = portfolio['benchmark']

        logger.info(f"Processing portfolio: {portfolio['portfolio_name']} ({portfolio_id})")
        logger.info(f"  Country: {country}, Benchmark: {benchmark}, Analysis date: {analysis_date}")

        # Determine start date
        if force_recalculate:
            start_date = analysis_date
        else:
            last_date = await self.get_last_performance_date(portfolio_id)
            if last_date:
                start_date = last_date + timedelta(days=1)
            else:
                start_date = analysis_date

        end_date = date.today()

        if start_date > end_date:
            logger.info(f"  Already up to date")
            return 0

        logger.info(f"  Calculating from {start_date} to {end_date}")

        # Get holdings
        holdings = await self.get_portfolio_holdings(portfolio_id)
        if not holdings:
            logger.warning(f"  No holdings found")
            return 0

        logger.info(f"  Holdings: {len(holdings)} stocks")

        # Separate by country for MIXED portfolios
        kr_symbols = [h['symbol'] for h in holdings if h['country'] == 'KR']
        us_symbols = [h['symbol'] for h in holdings if h['country'] == 'US']

        # Get stock prices
        stock_prices: Dict[str, Dict[date, Decimal]] = {}

        if kr_symbols:
            kr_prices = await self.get_stock_prices(
                kr_symbols, 'KR', analysis_date, end_date
            )
            stock_prices.update(kr_prices)

        if us_symbols:
            us_prices = await self.get_stock_prices(
                us_symbols, 'US', analysis_date, end_date
            )
            stock_prices.update(us_prices)

        # Get benchmark prices
        benchmark_prices = await self.get_benchmark_prices(
            benchmark, country, analysis_date, end_date
        )

        if not benchmark_prices:
            logger.warning(f"  No benchmark prices found for {benchmark}")

        # Get all dates with stock data
        all_dates = set()
        for symbol_prices in stock_prices.values():
            all_dates.update(symbol_prices.keys())

        # Filter to dates >= start_date
        dates = sorted([d for d in all_dates if d >= start_date])

        if not dates:
            logger.info(f"  No trading dates found")
            return 0

        logger.info(f"  Processing {len(dates)} trading days")

        # Calculate daily values
        daily_values = self.calculate_daily_values(holdings, stock_prices, dates)

        # Calculate returns
        records = self.calculate_returns(daily_values, benchmark_prices, analysis_date)

        # Save to database
        saved = await self.save_performance_records(portfolio_id, records)

        logger.info(f"  Saved {saved} performance records")

        # Update holdings risk data (peak price, trailing stop)
        if dates:
            latest_date = max(dates)
            risk_updated = await self.update_holdings_risk_data(
                portfolio_id, holdings, stock_prices, latest_date
            )
            logger.info(f"  Updated risk data for {risk_updated} holdings")

        return saved

    async def run_batch(
        self,
        portfolio_ids: Optional[List[str]] = None,
        force_recalculate: bool = False
    ) -> Dict[str, int]:
        """
        Run batch calculation for all or specified portfolios.

        Args:
            portfolio_ids: Optional list of portfolio IDs to process
            force_recalculate: If True, recalculate all dates

        Returns:
            Dict mapping portfolio_id to number of records saved
        """
        await self.initialize()

        results = {}

        try:
            if portfolio_ids:
                portfolios = [{'portfolio_id': pid} for pid in portfolio_ids]
            else:
                portfolios = await self.get_active_portfolios()

            logger.info(f"Processing {len(portfolios)} portfolios")

            for p in portfolios:
                portfolio_id = p['portfolio_id']
                try:
                    count = await self.calculate_portfolio_performance(
                        portfolio_id, force_recalculate
                    )
                    results[portfolio_id] = count
                except Exception as e:
                    logger.error(f"Error processing {portfolio_id}: {e}")
                    results[portfolio_id] = -1

            total = sum(c for c in results.values() if c > 0)
            logger.info(f"Batch complete. Total records saved: {total}")

        finally:
            await self.close()

        return results


async def run_daily_performance_batch(
    portfolio_ids: Optional[List[str]] = None,
    force_recalculate: bool = False
) -> Dict[str, int]:
    """
    Convenience function to run daily performance batch.

    Args:
        portfolio_ids: Optional list of portfolio IDs
        force_recalculate: If True, recalculate all history

    Returns:
        Dict mapping portfolio_id to records saved
    """
    calculator = DailyPerformanceCalculator()
    return await calculator.run_batch(portfolio_ids, force_recalculate)


# ============================================================================
# Test / CLI
# ============================================================================

async def main():
    """Run daily performance batch."""
    import argparse

    parser = argparse.ArgumentParser(description='Daily Performance Calculator')
    parser.add_argument(
        '--portfolio', '-p',
        type=str,
        help='Specific portfolio ID to process'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force recalculate all history'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Daily Performance Calculator")
    print("=" * 70)

    portfolio_ids = [args.portfolio] if args.portfolio else None

    results = await run_daily_performance_batch(
        portfolio_ids=portfolio_ids,
        force_recalculate=args.force
    )

    print("\nResults:")
    for pid, count in results.items():
        status = f"{count} records" if count >= 0 else "ERROR"
        print(f"  {pid}: {status}")

    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(main())
