# -*- coding: utf-8 -*-
"""
Portfolio Quantity Converter

Converts portfolio weights to actual share quantities.
Applies lot size and minimum investment constraints.

File: create_portfolio/portfolio_quantity.py
Created: 2025-12-24
"""

import os
import asyncio
import asyncpg
import logging
from decimal import Decimal
from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from models import (
    ProcessedInput,
    PortfolioStock,
    Country,
)
from core.portfolio_weight import WeightedStock, WeightAllocationResult

# Load .env from quant directory
env_path = Path(__file__).parent.parent / 'quant' / '.env'
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# 한국 매매 단위 (1주)
KR_LOT_SIZE = 1

# 미국 매매 단위 (1주, 소수점 가능 브로커도 있으나 정수 기준)
US_LOT_SIZE = 1


@dataclass
class PriceInfo:
    """종목 가격 정보"""
    symbol: str
    current_price: float  # Price in KRW (converted if US stock)
    currency: str  # Always 'KRW' for consistency
    price_date: date
    price_usd: float = None  # Original USD price (for US stocks only)
    exchange_rate: float = 1.0  # USD/KRW exchange rate used


@dataclass
class QuantityResult:
    """수량 환산 결과"""
    symbol: str
    stock_name: str
    country: Country
    sector: Optional[str]

    # 가격 정보
    current_price: float
    currency: str

    # 비중 및 수량
    target_weight: float
    target_amount: float
    shares: int
    actual_amount: float
    actual_weight: float

    # 점수 정보
    final_score: float
    conviction_score: Optional[float]

    # 등급 일관성
    consecutive_buy_days: int

    # 선정 이유
    selection_reasons: List[str] = field(default_factory=list)


@dataclass
class QuantityConversionResult:
    """수량 환산 전체 결과"""
    budget: int
    total_invested: float
    cash_remainder: float
    utilization_rate: float
    stocks: List[QuantityResult]
    currency: str
    # Result validation fields
    stocks_requested: int = 0         # 요청 종목 수
    stocks_allocated: int = 0         # 실제 배분 종목 수 (1주 이상)
    zero_share_count: int = 0         # 0주 종목 수
    excluded_stocks: List[str] = field(default_factory=list)  # 제외된 종목
    allocation_success: bool = True   # 성공 여부
    allocation_message: str = ""      # 결과 메시지


class QuantityConverter:
    """
    수량 환산기

    1. 현재가 조회
    2. 비중 -> 목표 금액 변환
    3. 주식 수량 계산 (정수 단위)
    4. 잔액 재배분
    """

    def __init__(self, db_pool=None):
        """
        Initialize converter.

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
            self.pool = await asyncpg.create_pool(db_url, min_size=5, max_size=20)
            self._owns_pool = True
            logger.info("Database pool initialized for quantity conversion")

    async def close(self):
        """Close database connection if owned"""
        if self._owns_pool and self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    async def convert_to_quantities(
        self,
        weight_result: WeightAllocationResult,
        processed_input: ProcessedInput
    ) -> QuantityConversionResult:
        """
        비중을 수량으로 환산.

        Args:
            weight_result: 비중 배분 결과
            processed_input: 처리된 입력

        Returns:
            QuantityConversionResult
        """
        await self.initialize()

        budget = processed_input.request.budget
        country = processed_input.request.country
        analysis_date = processed_input.analysis_date
        requested_count = len(weight_result.stocks)

        logger.info(f"Converting weights to quantities. Budget: {budget:,}, Date: {analysis_date}")

        # Step 1: analysis_date 기준 가격 조회
        symbols = [s.symbol for s in weight_result.stocks]
        prices = await self._fetch_prices(symbols, country, analysis_date)

        # Step 2: Phase 1 - 최소 1주 확보 (종목 수 보장)
        results, excluded = self._allocate_minimum_shares(
            weight_result.stocks, prices, budget
        )

        # Step 3: Phase 2 - 잔액으로 추가 매수 (비중 gap 기준)
        results, remainder = self._distribute_remainder_by_gap(results, prices, budget)

        # Step 4: 결과 검증
        allocated_count = sum(1 for r in results if r.shares > 0)
        zero_count = sum(1 for r in results if r.shares == 0)

        # 0주 종목 제거
        results = [r for r in results if r.shares > 0]

        # 결과 계산
        total_invested = sum(r.actual_amount for r in results)
        utilization = total_invested / budget if budget > 0 else 0

        # Always use KRW - US stocks are converted to KRW at analysis_date exchange rate
        currency = 'KRW'

        # 성공 여부 판정
        success, message = self._validate_allocation_result(
            requested_count, allocated_count, utilization
        )

        logger.info(f"Quantity conversion complete. Requested: {requested_count}, "
                   f"Allocated: {allocated_count}, Excluded: {len(excluded)}, "
                   f"Invested: {total_invested:,.0f}, Utilization: {utilization:.1%}")

        return QuantityConversionResult(
            budget=budget,
            total_invested=total_invested,
            cash_remainder=remainder,
            utilization_rate=utilization,
            stocks=results,
            currency=currency,
            stocks_requested=requested_count,
            stocks_allocated=allocated_count,
            zero_share_count=zero_count,
            excluded_stocks=excluded,
            allocation_success=success,
            allocation_message=message
        )

    async def _fetch_exchange_rate(self, target_date: date) -> float:
        """
        Fetch USD/KRW exchange rate for the given date.

        Args:
            target_date: Target date for exchange rate

        Returns:
            Exchange rate (USD to KRW). Returns 1350.0 as fallback.
        """
        try:
            async with self.pool.acquire() as conn:
                query = """
                SELECT data_value
                FROM exchange_rate
                WHERE (item_name1 LIKE '%미국%' OR item_name1 LIKE '%달러%' OR item_name1 LIKE '%USD%')
                  AND time_value <= $1
                ORDER BY time_value DESC
                LIMIT 1
                """
                row = await conn.fetchrow(query, target_date)

                if row and row['data_value']:
                    return float(row['data_value'])
                else:
                    logger.warning(f"Exchange rate not found for {target_date}, using fallback 1450.0")
                    return 1450.0
        except Exception as e:
            logger.error(f"Failed to fetch exchange rate: {e}")
            return 1450.0

    async def _fetch_prices(
        self,
        symbols: List[str],
        country: Country,
        analysis_date: date
    ) -> Dict[str, PriceInfo]:
        """
        analysis_date 기준 가격 조회.

        Args:
            symbols: 종목 코드 리스트
            country: 국가
            analysis_date: 분석 기준일

        Returns:
            symbol -> PriceInfo 매핑
        """
        prices = {}

        if country == Country.KR:
            prices.update(await self._fetch_kr_prices(symbols, analysis_date))
        elif country == Country.US:
            prices.update(await self._fetch_us_prices(symbols, analysis_date))
        else:  # MIXED
            kr_symbols = [s for s in symbols if s.isdigit()]
            us_symbols = [s for s in symbols if not s.isdigit()]
            prices.update(await self._fetch_kr_prices(kr_symbols, analysis_date))
            prices.update(await self._fetch_us_prices(us_symbols, analysis_date))

        return prices

    async def _fetch_kr_prices(
        self,
        symbols: List[str],
        analysis_date: date
    ) -> Dict[str, PriceInfo]:
        """한국 종목 analysis_date 기준 가격 조회"""
        if not symbols:
            return {}

        async with self.pool.acquire() as conn:
            # analysis_date 이하의 최신 거래일 조회
            date_query = """
            SELECT MAX(date) as latest_date
            FROM kr_intraday_total
            WHERE symbol = ANY($1)
              AND date <= $2
            """
            row = await conn.fetchrow(date_query, symbols, analysis_date)
            latest_date = row['latest_date'] if row else analysis_date

            # 종가 조회
            price_query = """
            SELECT symbol, close, date
            FROM kr_intraday_total
            WHERE symbol = ANY($1)
              AND date = $2
            """
            rows = await conn.fetch(price_query, symbols, latest_date)

        prices = {}
        for row in rows:
            prices[row['symbol']] = PriceInfo(
                symbol=row['symbol'],
                current_price=float(row['close']),
                currency='KRW',
                price_date=row['date']
            )

        logger.info(f"Fetched {len(prices)} KR prices for date {latest_date} (analysis_date: {analysis_date})")
        return prices

    async def _fetch_us_prices(
        self,
        symbols: List[str],
        analysis_date: date
    ) -> Dict[str, PriceInfo]:
        """미국 종목 analysis_date 기준 가격 조회 (원화 환산)"""
        if not symbols:
            return {}

        # Get exchange rate for analysis_date
        exchange_rate = await self._fetch_exchange_rate(analysis_date)
        logger.info(f"Using exchange rate {exchange_rate:.2f} for US stocks (date: {analysis_date})")

        async with self.pool.acquire() as conn:
            # analysis_date 이하의 최신 거래일 조회
            date_query = """
            SELECT MAX(date) as latest_date
            FROM us_daily
            WHERE symbol = ANY($1)
              AND date <= $2
            """
            row = await conn.fetchrow(date_query, symbols, analysis_date)
            latest_date = row['latest_date'] if row else analysis_date

            # 종가 조회
            price_query = """
            SELECT symbol, close, date
            FROM us_daily
            WHERE symbol = ANY($1)
              AND date = $2
            """
            rows = await conn.fetch(price_query, symbols, latest_date)

        prices = {}
        for row in rows:
            price_usd = float(row['close'])
            price_krw = price_usd * exchange_rate  # Convert to KRW

            prices[row['symbol']] = PriceInfo(
                symbol=row['symbol'],
                current_price=price_krw,  # Store KRW price
                currency='KRW',  # Always KRW for consistency
                price_date=row['date'],
                price_usd=price_usd,  # Store original USD price
                exchange_rate=exchange_rate
            )

        logger.info(f"Fetched {len(prices)} US prices for date {latest_date} (analysis_date: {analysis_date}), "
                   f"converted to KRW at rate {exchange_rate:.2f}")
        return prices

    def _allocate_minimum_shares(
        self,
        stocks: List[WeightedStock],
        prices: Dict[str, PriceInfo],
        budget: int
    ) -> Tuple[List[QuantityResult], List[str]]:
        """
        Phase 1: 최소 1주 확보 (종목 수 보장).

        모든 종목에 최소 1주를 배분하고, 예산 초과 시 저점수 종목 제외.

        Args:
            stocks: 비중 배분된 종목
            prices: 가격 정보
            budget: 총 예산

        Returns:
            (QuantityResult 리스트, 제외된 종목 리스트)
        """
        # 가격 정보가 있는 종목만 필터링하고 점수 순 정렬
        valid_stocks = []
        for stock in stocks:
            price_info = prices.get(stock.symbol)
            if price_info:
                valid_stocks.append((stock, price_info))
            else:
                logger.warning(f"No price found for {stock.symbol}, skipping")

        # 점수 높은 순 정렬
        valid_stocks.sort(key=lambda x: -x[0].selection_score)

        # 최소 1주씩 배분 가능한 종목 선택
        selected = []
        excluded = []
        total_min_cost = 0

        for stock, price_info in valid_stocks:
            price = price_info.current_price
            if total_min_cost + price <= budget:
                selected.append((stock, price_info))
                total_min_cost += price
            else:
                excluded.append(stock.symbol)
                logger.info(f"Excluded {stock.symbol} (price: {price:,.0f}, "
                           f"would exceed budget)")

        logger.info(f"Phase 1: {len(selected)} stocks can get minimum 1 share, "
                   f"{len(excluded)} excluded")

        # 선택된 종목에 1주씩 배분
        results = []
        for stock, price_info in selected:
            price = price_info.current_price
            target_amount = budget * stock.final_weight

            result = QuantityResult(
                symbol=stock.symbol,
                stock_name=stock.stock_name,
                country=stock.country,
                sector=stock.sector,
                current_price=price,
                currency=price_info.currency,
                target_weight=stock.final_weight,
                target_amount=target_amount,
                shares=1,  # 최소 1주 보장
                actual_amount=price,
                actual_weight=price / budget if budget > 0 else 0,
                final_score=stock.selection_score,
                conviction_score=stock.conviction_score,
                consecutive_buy_days=stock.consecutive_buy_days,
                selection_reasons=stock.selection_reasons.copy()
            )
            results.append(result)

        return results, excluded

    def _distribute_remainder_by_gap(
        self,
        results: List[QuantityResult],
        prices: Dict[str, PriceInfo],
        budget: int
    ) -> Tuple[List[QuantityResult], float]:
        """
        Phase 2: 잔액으로 추가 매수 (비중 gap 기준).

        목표 비중과 현재 비중의 차이가 큰 종목 우선으로 1주씩 추가.

        Args:
            results: Phase 1 결과
            prices: 가격 정보
            budget: 총 예산

        Returns:
            (업데이트된 결과, 최종 잔액)
        """
        total_invested = sum(r.actual_amount for r in results)
        remainder = budget - total_invested

        iterations = 0
        max_iterations = 100000  # Safety net only; actual termination by 'if not added: break'

        while remainder > 0 and iterations < max_iterations:
            # 현재 비중 재계산
            current_total = sum(r.actual_amount for r in results)
            for r in results:
                r.actual_weight = r.actual_amount / budget if budget > 0 else 0

            # 비중 gap 계산하여 정렬 (목표 - 현재)
            results_with_gap = []
            for r in results:
                gap = r.target_weight - r.actual_weight
                results_with_gap.append((r, gap))

            results_with_gap.sort(key=lambda x: -x[1])  # gap 큰 순

            added = False
            for result, gap in results_with_gap:
                price = result.current_price
                if remainder >= price:
                    result.shares += 1
                    result.actual_amount = result.shares * price
                    result.actual_weight = result.actual_amount / budget if budget > 0 else 0
                    remainder -= price
                    added = True
                    break

            if not added:
                break

            iterations += 1

        logger.info(f"Phase 2: Distributed remainder in {iterations} iterations, "
                   f"final remainder: {remainder:,.0f}")

        return results, remainder

    def _validate_allocation_result(
        self,
        requested: int,
        allocated: int,
        utilization: float
    ) -> Tuple[bool, str]:
        """
        결과 검증.

        Args:
            requested: 요청 종목 수
            allocated: 배분된 종목 수
            utilization: 예산 활용률

        Returns:
            (성공 여부, 메시지)
        """
        if requested == 0:
            return False, "FAILED: No stocks requested"

        ratio = allocated / requested

        if ratio >= 0.8 and utilization >= 0.5:
            return True, f"SUCCESS: {allocated}/{requested} stocks allocated ({ratio:.0%})"
        elif ratio >= 0.5:
            return True, f"WARNING: {allocated}/{requested} stocks allocated ({ratio:.0%}), some stocks excluded due to budget"
        else:
            return False, f"FAILED: Only {allocated}/{requested} stocks could be allocated ({ratio:.0%})"


async def convert_to_quantities(
    weight_result: WeightAllocationResult,
    processed_input: ProcessedInput,
    pool: 'asyncpg.Pool' = None
) -> QuantityConversionResult:
    """
    수량 환산 편의 함수.

    Args:
        weight_result: 비중 배분 결과
        processed_input: 처리된 입력
        pool: Optional shared database pool. If None, creates own pool.

    Returns:
        QuantityConversionResult
    """
    if pool is not None:
        # Use shared pool (don't close it)
        converter = QuantityConverter(db_pool=pool)
        converter._owns_pool = False
        return await converter.convert_to_quantities(weight_result, processed_input)
    else:
        # Create own pool (backward compatibility)
        converter = QuantityConverter()
        try:
            return await converter.convert_to_quantities(weight_result, processed_input)
        finally:
            await converter.close()


def create_portfolio_stocks(
    quantity_result: QuantityConversionResult
) -> List[PortfolioStock]:
    """
    QuantityResult를 PortfolioStock 모델로 변환.

    Args:
        quantity_result: 수량 환산 결과

    Returns:
        PortfolioStock 리스트
    """
    stocks = []

    for q in quantity_result.stocks:
        stock = PortfolioStock(
            symbol=q.symbol,
            stock_name=q.stock_name,
            country=q.country,
            sector=q.sector,
            weight=q.actual_weight,
            shares=q.shares,
            amount=q.actual_amount,
            current_price=q.current_price,
            currency=q.currency,
            final_score=q.final_score,
            conviction_score=q.conviction_score,
            volatility_annual=None,  # 추후 추가 가능
            beta=None,
            selection_reasons=q.selection_reasons
        )
        stocks.append(stock)

    return stocks


# ============================================================================
# Test
# ============================================================================

async def test_quantity_converter():
    """Test quantity conversion with real data"""
    from portfolio_input import process_portfolio_request
    from portfolio_universe import filter_universe
    from portfolio_risk_filter import filter_by_risk
    from portfolio_selector import select_stocks
    from portfolio_weight import allocate_weights

    print("=" * 70)
    print("Quantity Converter - Test")
    print("=" * 70)

    # Full pipeline test
    print("\n[Test] Full Pipeline - KR Balanced 10M KRW")

    processed = process_portfolio_request(
        budget=10_000_000,
        country='KR',
        risk_level='balanced',
        num_stocks=10
    )

    if not processed.validation.is_valid:
        print("  Input validation failed!")
        return

    # Step 1-3: Universe -> Risk -> Selection
    print("\n  Step 1-3: Filtering and Selection...")
    candidates = await filter_universe(processed)
    risk_result = filter_by_risk(candidates, processed)
    selection = await select_stocks(risk_result.candidates, processed)

    print(f"    Selected {selection.final_selected} stocks")

    # Step 4: Weight allocation
    print("\n  Step 4: Weight Allocation...")
    weight_result = allocate_weights(selection.selected_stocks, processed)
    print(f"    Weights allocated")

    # Step 5: Quantity conversion
    print("\n  Step 5: Quantity Conversion...")
    quantity_result = await convert_to_quantities(weight_result, processed)

    print(f"\n  Results:")
    print(f"    Budget: {quantity_result.budget:,} KRW")
    print(f"    Invested: {quantity_result.total_invested:,.0f} KRW")
    print(f"    Remainder: {quantity_result.cash_remainder:,.0f} KRW")
    print(f"    Utilization: {quantity_result.utilization_rate:.1%}")
    print(f"    Requested: {quantity_result.stocks_requested}")
    print(f"    Allocated: {quantity_result.stocks_allocated}")
    print(f"    Excluded: {quantity_result.excluded_stocks}")
    print(f"    Success: {quantity_result.allocation_success}")
    print(f"    Message: {quantity_result.allocation_message}")

    print(f"\n  Portfolio:")
    print(f"  {'No':<3} {'Symbol':<10} {'Name':<14} {'Price':>10} {'Shares':>6} {'Amount':>12} {'Weight':>7}")
    print(f"  {'-'*3} {'-'*10} {'-'*14} {'-'*10} {'-'*6} {'-'*12} {'-'*7}")

    for i, q in enumerate(quantity_result.stocks, 1):
        name = (q.stock_name or '')[:12]
        print(f"  {i:<3} {q.symbol:<10} {name:<14} "
              f"{q.current_price:>10,.0f} {q.shares:>6} "
              f"{q.actual_amount:>12,.0f} {q.actual_weight:>6.1%}")

    # Convert to PortfolioStock models
    print("\n  PortfolioStock conversion:")
    portfolio_stocks = create_portfolio_stocks(quantity_result)
    print(f"    Created {len(portfolio_stocks)} PortfolioStock objects")

    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(test_quantity_converter())
