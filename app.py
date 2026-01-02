# -*- coding: utf-8 -*-
"""
Portfolio Generator Web Application

FastAPI web server for portfolio generation and management.
Local admin GUI for creating and viewing portfolios.

File: create_portfolio/app.py
Created: 2025-12-24
"""

import asyncio
import logging
from datetime import date, datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from create.portfolio_generator import generate_portfolio
from db.db_manager import SharedDatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Alpha Portfolio Generator")

# Templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ============================================================================
# Helper Functions
# ============================================================================

async def get_exchange_rate(db, target_date=None) -> float:
    """
    Get USD/KRW exchange rate from exchange_rate table.

    Args:
        db: SharedDatabaseManager instance
        target_date: Target date for exchange rate. If None, get latest.

    Returns:
        Exchange rate (USD to KRW). Returns 1350.0 as fallback.
    """
    try:
        if target_date:
            # Get exchange rate for target_date or earlier
            query = """
            SELECT data_value
            FROM exchange_rate
            WHERE (item_name1 LIKE '%미국%' OR item_name1 LIKE '%달러%' OR item_name1 LIKE '%USD%')
              AND time_value <= $1
            ORDER BY time_value DESC
            LIMIT 1
            """
            result = await db.execute_one(query, target_date)
        else:
            # Get latest exchange rate
            query = """
            SELECT data_value
            FROM exchange_rate
            WHERE (item_name1 LIKE '%미국%' OR item_name1 LIKE '%달러%' OR item_name1 LIKE '%USD%')
            ORDER BY time_value DESC
            LIMIT 1
            """
            result = await db.execute_one(query)

        if result and result['data_value']:
            return float(result['data_value'])
        else:
            logger.warning("Exchange rate not found, using fallback 1450.0")
            return 1450.0
    except Exception as e:
        logger.error(f"Failed to fetch exchange rate: {e}")
        return 1450.0


async def get_latest_prices(db, symbols: list, country: str) -> tuple:
    """
    Get latest prices for given symbols from us_daily or kr_intraday_total table.
    For US stocks, also returns USD prices and exchange rate.
    For MIXED portfolios, queries both tables.

    Args:
        db: SharedDatabaseManager instance
        symbols: List of stock symbols
        country: 'US', 'KR', or 'MIXED'

    Returns:
        Tuple of (prices_krw_dict, prices_usd_dict, exchange_rate)
        - prices_krw_dict: {symbol: price_in_krw}
        - prices_usd_dict: {symbol: price_in_usd} (empty for KR-only)
        - exchange_rate: float (1.0 for KR-only)
    """
    if not symbols:
        return {}, {}, 1.0

    prices_krw = {}
    prices_usd = {}
    exchange_rate = 1.0

    # Helper to check if symbol is KR (numeric) or US (alphabetic)
    def is_kr_symbol(sym: str) -> bool:
        return sym.isdigit()

    # Determine which symbols belong to which market
    if country == 'MIXED':
        kr_symbols = [s for s in symbols if is_kr_symbol(s)]
        us_symbols = [s for s in symbols if not is_kr_symbol(s)]
    elif country == 'KR':
        kr_symbols = symbols
        us_symbols = []
    else:  # US
        kr_symbols = []
        us_symbols = symbols

    # Query KR prices
    if kr_symbols:
        date_query = "SELECT MAX(date) as latest_date FROM kr_intraday_total"
        date_result = await db.execute_one(date_query)
        if date_result and date_result['latest_date']:
            latest_date = date_result['latest_date']
            placeholders = ', '.join([f'${i+1}' for i in range(len(kr_symbols))])
            price_query = f"""
            SELECT symbol, close FROM kr_intraday_total
            WHERE date = '{latest_date}' AND symbol IN ({placeholders})
            """
            rows = await db.execute_query(price_query, *kr_symbols)
            for row in rows:
                prices_krw[row['symbol']] = float(row['close'])

    # Query US prices
    if us_symbols:
        date_query = "SELECT MAX(date) as latest_date FROM us_daily"
        date_result = await db.execute_one(date_query)
        if date_result and date_result['latest_date']:
            latest_date = date_result['latest_date']
            exchange_rate = await get_exchange_rate(db)
            placeholders = ', '.join([f'${i+1}' for i in range(len(us_symbols))])
            price_query = f"""
            SELECT symbol, close FROM us_daily
            WHERE date = '{latest_date}' AND symbol IN ({placeholders})
            """
            rows = await db.execute_query(price_query, *us_symbols)
            for row in rows:
                usd_price = float(row['close'])
                prices_usd[row['symbol']] = usd_price
                prices_krw[row['symbol']] = usd_price * exchange_rate

    return prices_krw, prices_usd, exchange_rate


async def get_portfolio_list():
    """Fetch portfolio list from database with real-time return calculation"""
    db = SharedDatabaseManager()
    try:
        await db.initialize()

        query = """
        SELECT
            portfolio_id,
            portfolio_name,
            country,
            risk_level,
            initial_budget,
            current_budget,
            current_stock_count,
            created_at,
            analysis_date,
            status
        FROM portfolio_master
        ORDER BY created_at DESC
        """
        rows = await db.execute_query(query)

        # Calculate real-time return rate for each portfolio
        for row in rows:
            portfolio_id = row['portfolio_id']
            country = row['country']

            # Get holdings for this portfolio
            holdings_query = """
            SELECT symbol, shares, avg_price, invested_amount
            FROM portfolio_holdings
            WHERE portfolio_id = $1
            """
            holdings = await db.execute_query(holdings_query, portfolio_id)

            if holdings:
                # Get current prices (with exchange rate for US stocks)
                symbols = [h['symbol'] for h in holdings]
                prices_krw, prices_usd, current_exchange_rate = await get_latest_prices(db, symbols, country)

                # For US stocks, get analysis_date exchange rate for accurate return calculation
                analysis_exchange_rate = 1.0
                if country == 'US' and row.get('analysis_date'):
                    analysis_exchange_rate = await get_exchange_rate(db, row['analysis_date'])

                # Calculate current total value
                total_current_value = 0
                total_invested = 0
                total_invested_usd = 0
                total_current_value_usd = 0

                for h in holdings:
                    symbol = h['symbol']
                    shares = h['shares']
                    invested = float(h['invested_amount']) if h['invested_amount'] else 0

                    if symbol in prices_krw:
                        current_value = shares * prices_krw[symbol]
                        total_current_value += current_value

                        # For US stocks, track USD values for return calculation
                        if country == 'US' and symbol in prices_usd:
                            invested_usd = invested / analysis_exchange_rate if analysis_exchange_rate > 0 else 0
                            current_value_usd = shares * prices_usd[symbol]
                            total_invested_usd += invested_usd
                            total_current_value_usd += current_value_usd
                    else:
                        # Fallback to invested amount if current price not found
                        total_current_value += invested

                    total_invested += invested

                # Update row with real-time values
                row['current_budget'] = total_current_value

                # Return rate: USD-based for US stocks, KRW-based for KR stocks
                if country == 'US' and total_invested_usd > 0:
                    # US portfolio: return rate based on USD (exclude exchange rate fluctuation)
                    row['return_rate'] = ((total_current_value_usd - total_invested_usd)
                                          / total_invested_usd * 100)
                elif total_invested > 0:
                    # KR portfolio: return rate based on KRW
                    row['return_rate'] = ((total_current_value - total_invested)
                                          / total_invested * 100)
                else:
                    row['return_rate'] = 0.0

                # Get latest benchmark return from portfolio_daily_performance
                benchmark_query = """
                SELECT benchmark_cumulative_return
                FROM portfolio_daily_performance
                WHERE portfolio_id = $1
                ORDER BY date DESC
                LIMIT 1
                """
                benchmark_result = await db.execute_one(benchmark_query, portfolio_id)
                if benchmark_result and benchmark_result['benchmark_cumulative_return'] is not None:
                    row['benchmark_return'] = float(benchmark_result['benchmark_cumulative_return'])
                else:
                    row['benchmark_return'] = None
            else:
                row['return_rate'] = 0.0
                row['benchmark_return'] = None

        return rows
    except Exception as e:
        logger.error(f"Failed to fetch portfolio list: {e}")
        return []
    finally:
        await db.close()


async def get_portfolio_detail(portfolio_id: str):
    """Fetch portfolio detail from database with real-time price calculation"""
    db = SharedDatabaseManager()
    try:
        await db.initialize()

        # Get master info
        master_query = """
        SELECT *
        FROM portfolio_master
        WHERE portfolio_id = $1
        """
        master = await db.execute_one(master_query, portfolio_id)

        if not master:
            return None, None

        country = master['country']

        # Get holdings
        holdings_query = """
        SELECT
            symbol,
            stock_name,
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
            entry_score
        FROM portfolio_holdings
        WHERE portfolio_id = $1
        ORDER BY current_weight DESC
        """
        holdings = await db.execute_query(holdings_query, portfolio_id)

        if holdings:
            # Get current prices for all symbols (with exchange rate for US stocks)
            symbols = [h['symbol'] for h in holdings]
            prices_krw, prices_usd, current_exchange_rate = await get_latest_prices(db, symbols, country)

            # For US stocks, get analysis_date exchange rate for accurate P/L calculation
            analysis_exchange_rate = 1.0
            if country == 'US' and master.get('analysis_date'):
                analysis_exchange_rate = await get_exchange_rate(db, master['analysis_date'])

            total_current_value = 0
            total_invested = 0
            total_invested_usd = 0
            total_current_value_usd = 0

            # Update each holding with real-time values
            for h in holdings:
                symbol = h['symbol']
                shares = h['shares']
                avg_price = float(h['avg_price']) if h['avg_price'] else 0
                invested = float(h['invested_amount']) if h['invested_amount'] else 0

                # Set country flag for template
                h['is_us_stock'] = (country == 'US')

                if symbol in prices_krw:
                    # Update with real-time price (KRW)
                    h['current_price'] = prices_krw[symbol]
                    h['current_value'] = shares * prices_krw[symbol]
                    h['unrealized_pnl'] = h['current_value'] - invested

                    # For US stocks, calculate P/L based on USD prices (exclude exchange rate fluctuation)
                    if country == 'US' and symbol in prices_usd:
                        h['current_price_usd'] = prices_usd[symbol]
                        # Convert avg_price (KRW) back to USD using analysis_date exchange rate
                        avg_price_usd = avg_price / analysis_exchange_rate if analysis_exchange_rate > 0 else 0
                        h['avg_price_usd'] = avg_price_usd

                        # P/L calculation based on USD (pure stock price change)
                        if avg_price_usd > 0:
                            h['unrealized_pnl_pct'] = ((prices_usd[symbol] - avg_price_usd) / avg_price_usd) * 100
                        else:
                            h['unrealized_pnl_pct'] = 0.0

                        # Track USD totals for portfolio return calculation
                        invested_usd = invested / analysis_exchange_rate if analysis_exchange_rate > 0 else 0
                        current_value_usd = shares * prices_usd[symbol]
                        total_invested_usd += invested_usd
                        total_current_value_usd += current_value_usd
                    else:
                        h['current_price_usd'] = None
                        h['avg_price_usd'] = None
                        # KR stocks: P/L based on KRW
                        if invested > 0:
                            h['unrealized_pnl_pct'] = (h['unrealized_pnl'] / invested) * 100
                        else:
                            h['unrealized_pnl_pct'] = 0.0
                else:
                    h['current_price_usd'] = None
                    h['avg_price_usd'] = None

                total_current_value += h['current_value'] if h['current_value'] else 0
                total_invested += invested

            # Recalculate weights based on current values
            if total_current_value > 0:
                for h in holdings:
                    if h['current_value']:
                        h['current_weight'] = h['current_value'] / total_current_value

            # Update master with real-time values
            master['current_budget'] = total_current_value
            master['exchange_rate'] = current_exchange_rate if country == 'US' else None

            # Portfolio return rate: USD-based for US stocks, KRW-based for KR stocks
            if country == 'US' and total_invested_usd > 0:
                # US portfolio: return rate based on USD (exclude exchange rate fluctuation)
                master['return_rate'] = ((total_current_value_usd - total_invested_usd)
                                         / total_invested_usd * 100)
            elif total_invested > 0:
                # KR portfolio: return rate based on KRW
                master['return_rate'] = ((total_current_value - total_invested)
                                         / total_invested * 100)
            else:
                master['return_rate'] = 0.0

            # Get latest benchmark return from portfolio_daily_performance
            benchmark_query = """
            SELECT benchmark_cumulative_return
            FROM portfolio_daily_performance
            WHERE portfolio_id = $1
            ORDER BY date DESC
            LIMIT 1
            """
            benchmark_result = await db.execute_one(benchmark_query, portfolio_id)
            if benchmark_result and benchmark_result['benchmark_cumulative_return'] is not None:
                master['benchmark_return'] = float(benchmark_result['benchmark_cumulative_return'])
            else:
                master['benchmark_return'] = None
        else:
            master['return_rate'] = 0.0
            master['benchmark_return'] = None

        return master, holdings
    except Exception as e:
        logger.error(f"Failed to fetch portfolio detail: {e}")
        return None, None
    finally:
        await db.close()


# ============================================================================
# Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def create_page(request: Request):
    """Render create portfolio page"""
    return templates.TemplateResponse("create.html", {
        "request": request,
        "active_menu": "create",
        "today": date.today().isoformat()
    })


@app.get("/list", response_class=HTMLResponse)
async def list_page(request: Request):
    """Render portfolio list page"""
    portfolios = await get_portfolio_list()
    return templates.TemplateResponse("list.html", {
        "request": request,
        "active_menu": "list",
        "portfolios": portfolios
    })


@app.get("/detail/{portfolio_id}", response_class=HTMLResponse)
async def detail_page(request: Request, portfolio_id: str):
    """Render portfolio detail page"""
    master, holdings = await get_portfolio_detail(portfolio_id)

    if not master:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    return templates.TemplateResponse("detail.html", {
        "request": request,
        "active_menu": "list",
        "portfolio": master,
        "holdings": holdings
    })


@app.post("/generate")
async def generate_portfolio_api(
    portfolio_name: str = Form(...),
    country: str = Form(...),
    budget: int = Form(...),
    risk_level: str = Form(...),
    num_stocks: int = Form(...),
    benchmark: str = Form(...),
    analysis_date: str = Form(...),
    max_weight_per_stock: Optional[float] = Form(None),
    max_weight_per_sector: Optional[float] = Form(None),
    min_consecutive_buy_days: Optional[int] = Form(None),
    rebalancing_frequency: Optional[str] = Form(None)
):
    """Generate portfolio API endpoint"""
    try:
        logger.info(f"Generating portfolio: {portfolio_name}")

        # Parse analysis date
        if analysis_date:
            parsed_date = datetime.strptime(analysis_date, "%Y-%m-%d").date()
        else:
            parsed_date = None

        # Call generate_portfolio
        response = await generate_portfolio(
            budget=budget,
            country=country,
            risk_level=risk_level,
            num_stocks=num_stocks,
            portfolio_name=portfolio_name,
            analysis_date=parsed_date,
            benchmark=benchmark,
            max_weight_per_stock=max_weight_per_stock,
            max_weight_per_sector=max_weight_per_sector,
            min_consecutive_buy_days=min_consecutive_buy_days,
            rebalancing_frequency=rebalancing_frequency
        )

        if response.success:
            return JSONResponse({
                "success": True,
                "message": "Portfolio generated successfully",
                "portfolio_id": response.portfolio_id,
                "portfolio_name": response.portfolio_name,
                "stock_count": len(response.stocks) if response.stocks else 0
            })
        else:
            return JSONResponse({
                "success": False,
                "message": response.message,
                "error_code": response.error_code,
                "error_detail": response.error_detail
            }, status_code=400)

    except Exception as e:
        logger.exception("Portfolio generation failed")
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)


@app.delete("/delete/{portfolio_id}")
async def delete_portfolio_api(portfolio_id: str):
    """Delete portfolio and all related data from database"""
    db = SharedDatabaseManager()
    try:
        await db.initialize()

        # Check if portfolio exists
        check_query = "SELECT portfolio_id FROM portfolio_master WHERE portfolio_id = $1"
        exists = await db.execute_one(check_query, portfolio_id)

        if not exists:
            return JSONResponse({
                "success": False,
                "message": "Portfolio not found"
            }, status_code=404)

        # Delete in order (child tables first, then master)
        # Only delete from portfolio_* tables, never touch kr_* or us_* tables
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                # 1. Delete transactions
                await conn.execute(
                    "DELETE FROM portfolio_transactions WHERE portfolio_id = $1",
                    portfolio_id
                )

                # 2. Delete stock daily performance
                await conn.execute(
                    "DELETE FROM portfolio_stock_daily WHERE portfolio_id = $1",
                    portfolio_id
                )

                # 3. Delete daily performance
                await conn.execute(
                    "DELETE FROM portfolio_daily_performance WHERE portfolio_id = $1",
                    portfolio_id
                )

                # 4. Delete holdings
                await conn.execute(
                    "DELETE FROM portfolio_holdings WHERE portfolio_id = $1",
                    portfolio_id
                )

                # 5. Delete master (last)
                await conn.execute(
                    "DELETE FROM portfolio_master WHERE portfolio_id = $1",
                    portfolio_id
                )

        logger.info(f"Portfolio {portfolio_id} deleted successfully")

        return JSONResponse({
            "success": True,
            "message": "Portfolio deleted successfully",
            "portfolio_id": portfolio_id
        })

    except Exception as e:
        logger.exception(f"Failed to delete portfolio {portfolio_id}")
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)
    finally:
        await db.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alpha Portfolio Generator - Web UI")
    print("=" * 60)
    print()
    print("  Access: http://localhost:7000")
    print("  Press Ctrl+C to stop")
    print()
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=7000)
