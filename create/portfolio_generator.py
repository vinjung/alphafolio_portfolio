# -*- coding: utf-8 -*-
"""
Portfolio Generator

Main entry point for alpha portfolio generation.
Orchestrates all modules and generates final portfolio.

File: create_portfolio/portfolio_generator.py
Created: 2025-12-24
"""

import asyncio
import logging
import uuid
from datetime import datetime, date
from typing import Optional

from models import (
    PortfolioRequest,
    PortfolioResponse,
    PortfolioSummary,
    PortfolioStock,
    ProcessedInput,
    Country,
)
from create.portfolio_input import process_portfolio_request
from core.portfolio_universe import filter_universe
from core.portfolio_risk_filter import filter_by_risk
from core.portfolio_selector import select_stocks
from core.portfolio_weight import allocate_weights
from core.portfolio_quantity import convert_to_quantities, create_portfolio_stocks
from db.portfolio_db import save_portfolio
from db.db_manager import SharedDatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def generate_portfolio(
    budget: int,
    country: str = 'KR',
    risk_level: str = 'balanced',
    num_stocks: int = 10,
    portfolio_name: Optional[str] = None,
    analysis_date: Optional[date] = None,
    benchmark: Optional[str] = None,
    max_weight_per_stock: Optional[float] = None,
    max_weight_per_sector: Optional[float] = None,
    min_consecutive_buy_days: Optional[int] = None,
    rebalancing_frequency: Optional[str] = None
) -> PortfolioResponse:
    """
    알파 포트폴리오 생성.

    전체 파이프라인:
    1. 입력 처리 및 검증
    2. 유니버스 필터링
    3. 리스크 필터링
    4. 종목 선정 (등급 일관성 포함)
    5. 비중 배분
    6. 수량 환산
    7. 데이터베이스 저장

    Args:
        budget: 투자 예산
        country: 국가 (KR, US, MIXED)
        risk_level: 리스크 레벨 (conservative, balanced, aggressive)
        num_stocks: 종목 수 (5-20)
        portfolio_name: 포트폴리오 이름 (선택)
        analysis_date: 분석 기준일 (선택)
        benchmark: 벤치마크 지수 (선택)
        max_weight_per_stock: 종목당 최대 비중 (선택)
        max_weight_per_sector: 섹터당 최대 비중 (선택)
        min_consecutive_buy_days: 최소 연속 매수등급 일수 (선택)
        rebalancing_frequency: 리밸런싱 주기 (선택)

    Returns:
        PortfolioResponse with complete portfolio
    """
    logger.info("=" * 70)
    logger.info("Alpha Portfolio Generation Started")
    logger.info(f"Budget: {budget:,}, Country: {country}, Risk: {risk_level}")
    logger.info("=" * 70)

    portfolio_id = str(uuid.uuid4())[:8]
    created_at = datetime.now()

    # Step 1: 입력 처리 및 검증
    logger.info("\n[Step 1] Input Processing & Validation")
    processed = process_portfolio_request(
        budget=budget,
        country=country,
        risk_level=risk_level,
        num_stocks=num_stocks,
        portfolio_name=portfolio_name,
        analysis_date=analysis_date,
        benchmark=benchmark,
        max_weight_per_stock=max_weight_per_stock,
        max_weight_per_sector=max_weight_per_sector,
        min_consecutive_buy_days=min_consecutive_buy_days,
        rebalancing_frequency=rebalancing_frequency
    )

    if not processed.validation.is_valid:
        logger.error("Input validation failed")
        return PortfolioResponse(
            success=False,
            message="Input validation failed",
            portfolio_id=portfolio_id,
            created_at=created_at,
            request=processed.request,
            validation=processed.validation,
            error_code="VALIDATION_ERROR",
            error_detail="; ".join([e.message for e in processed.validation.errors])
        )

    logger.info(f"  Analysis date: {processed.analysis_date}")

    # Initialize shared database pool
    db = SharedDatabaseManager()
    try:
        await db.initialize()

        # Step 2: 유니버스 필터링
        logger.info("\n[Step 2] Universe Filtering")
        try:
            candidates = await filter_universe(processed, db.pool)
        except ValueError as e:
            await db.close()
            return _create_error_response(
                portfolio_id, created_at, processed,
                "NO_DATA_FOR_DATE",
                str(e)
            )
        logger.info(f"  Candidates after universe filter: {len(candidates)}")

        if not candidates:
            await db.close()
            return _create_error_response(
                portfolio_id, created_at, processed,
                "NO_CANDIDATES",
                "No stocks passed universe filter"
            )

        # Step 3: 리스크 필터링
        logger.info("\n[Step 3] Risk Filtering")
        risk_result = filter_by_risk(candidates, processed)
        logger.info(f"  Candidates after risk filter: {len(risk_result.candidates)}")

        if not risk_result.candidates:
            await db.close()
            return _create_error_response(
                portfolio_id, created_at, processed,
                "NO_CANDIDATES_AFTER_RISK",
                "No stocks passed risk filter"
            )

        # Step 4: 종목 선정 (등급 일관성 포함)
        logger.info("\n[Step 4] Stock Selection (with Grade Consistency)")
        selection = await select_stocks(risk_result.candidates, processed, db.pool)
        logger.info(f"  After consistency filter: {selection.after_consistency_filter}")
        logger.info(f"  Final selected: {selection.final_selected}")

        if not selection.selected_stocks:
            await db.close()
            return _create_error_response(
                portfolio_id, created_at, processed,
                "NO_SELECTION",
                "No stocks selected after consistency filter"
            )

        # Step 5: 비중 배분
        logger.info("\n[Step 5] Weight Allocation")
        weight_result = allocate_weights(selection.selected_stocks, processed)
        logger.info(f"  Max weight: {weight_result.max_stock_weight:.1%}")
        logger.info(f"  Min weight: {weight_result.min_stock_weight:.1%}")

        # Step 6: 수량 환산
        logger.info("\n[Step 6] Quantity Conversion")
        quantity_result = await convert_to_quantities(weight_result, processed, db.pool)
        logger.info(f"  Invested: {quantity_result.total_invested:,.0f}")
        logger.info(f"  Utilization: {quantity_result.utilization_rate:.1%}")

        # 최종 PortfolioStock 모델 생성
        portfolio_stocks = create_portfolio_stocks(quantity_result)

        # Summary 생성
        summary = _create_summary(portfolio_stocks, quantity_result)

        # 응답 객체 생성
        response = PortfolioResponse(
            success=True,
            message="Portfolio generated successfully",
            portfolio_id=portfolio_id,
            portfolio_name=portfolio_name or f"Alpha_{country}_{risk_level}_{created_at.strftime('%Y%m%d')}",
            created_at=created_at,
            request=processed.request,
            summary=summary,
            stocks=portfolio_stocks,
            validation=processed.validation
        )

        # Step 7: 데이터베이스 저장
        logger.info("\n[Step 7] Saving to Database")

        # consecutive_buy_days 정보 추출
        selection_data = {
            'consecutive_buy_days': {
                q.symbol: q.consecutive_buy_days
                for q in quantity_result.stocks
            }
        }

        try:
            saved = await save_portfolio(response, processed, selection_data, db.pool)
            if saved:
                logger.info("  Portfolio saved to database successfully")
            else:
                logger.warning("  Failed to save portfolio to database")
        except Exception as db_error:
            logger.error(f"  Database save error: {db_error}")
            # DB 저장 실패해도 포트폴리오 생성 결과는 반환

        # 성공 응답
        logger.info("\n" + "=" * 70)
        logger.info("Portfolio Generation Completed Successfully")
        logger.info("=" * 70)

        return response

    except Exception as e:
        logger.exception("Portfolio generation failed")
        return PortfolioResponse(
            success=False,
            message="Portfolio generation failed",
            portfolio_id=portfolio_id,
            created_at=created_at,
            error_code="GENERATION_ERROR",
            error_detail=str(e)
        )
    finally:
        await db.close()


def _create_error_response(
    portfolio_id: str,
    created_at: datetime,
    processed: ProcessedInput,
    error_code: str,
    error_detail: str
) -> PortfolioResponse:
    """에러 응답 생성"""
    return PortfolioResponse(
        success=False,
        message=error_detail,
        portfolio_id=portfolio_id,
        created_at=created_at,
        request=processed.request,
        validation=processed.validation,
        error_code=error_code,
        error_detail=error_detail
    )


def _create_summary(
    stocks: list,
    quantity_result
) -> PortfolioSummary:
    """포트폴리오 요약 생성"""
    total_stocks = len(stocks)
    total_investment = quantity_result.total_invested

    # 가중 평균 계산
    if total_investment > 0:
        avg_final_score = sum(
            s.final_score * s.weight for s in stocks
        )
        avg_volatility = sum(
            (s.volatility_annual or 0) * s.weight for s in stocks
        ) or None
        avg_beta = sum(
            (s.beta or 0) * s.weight for s in stocks
        ) or None
    else:
        avg_final_score = 0
        avg_volatility = None
        avg_beta = None

    # 섹터 분포
    sector_weights = {}
    for s in stocks:
        sector = s.sector or 'Unknown'
        sector_weights[sector] = sector_weights.get(sector, 0) + s.weight

    return PortfolioSummary(
        total_stocks=total_stocks,
        total_investment=total_investment,
        currency=quantity_result.currency,
        avg_final_score=avg_final_score,
        avg_volatility=avg_volatility,
        avg_beta=avg_beta,
        sector_weights=sector_weights
    )


def print_portfolio(response: PortfolioResponse):
    """포트폴리오 출력"""
    print("\n" + "=" * 80)
    print("ALPHA PORTFOLIO RESULT")
    print("=" * 80)

    if not response.success:
        print(f"\n[ERROR] {response.error_code}: {response.error_detail}")
        return

    print(f"\nPortfolio ID: {response.portfolio_id}")
    print(f"Portfolio Name: {response.portfolio_name}")
    print(f"Created: {response.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    if response.request:
        print(f"\nRequest:")
        print(f"  Budget: {response.request.budget:,}")
        print(f"  Country: {response.request.country.value}")
        print(f"  Risk Level: {response.request.risk_level.value}")
        print(f"  Num Stocks: {response.request.num_stocks}")

    if response.summary:
        print(f"\nSummary:")
        print(f"  Total Stocks: {response.summary.total_stocks}")
        print(f"  Total Investment: {response.summary.total_investment:,.0f} {response.summary.currency}")
        print(f"  Avg Final Score: {response.summary.avg_final_score:.1f}")

        print(f"\n  Sector Weights:")
        for sector, weight in sorted(response.summary.sector_weights.items(), key=lambda x: -x[1]):
            print(f"    {sector}: {weight:.1%}")

    if response.stocks:
        print(f"\nPortfolio Holdings:")
        print(f"  {'No':<3} {'Symbol':<10} {'Name':<16} {'Price':>10} {'Shares':>6} {'Amount':>12} {'Weight':>7}")
        print(f"  {'-'*3} {'-'*10} {'-'*16} {'-'*10} {'-'*6} {'-'*12} {'-'*7}")

        for i, s in enumerate(response.stocks, 1):
            name = (s.stock_name or '')[:14]
            print(f"  {i:<3} {s.symbol:<10} {name:<16} "
                  f"{s.current_price:>10,.0f} {s.shares:>6} "
                  f"{s.amount:>12,.0f} {s.weight:>6.1%}")

    if response.validation and response.validation.warnings:
        print(f"\nWarnings:")
        for w in response.validation.warnings:
            print(f"  - {w}")

    print("\n" + "=" * 80)


# ============================================================================
# Test / Main
# ============================================================================

async def main():
    """Main entry point for testing"""
    print("=" * 80)
    print("Alpha Portfolio Generator - Test")
    print("=" * 80)

    # Test 1: KR Balanced
    print("\n[Test 1] KR Balanced - 10M KRW, 10 stocks")
    response = await generate_portfolio(
        budget=10_000_000,
        country='KR',
        risk_level='balanced',
        num_stocks=10,
        portfolio_name="Test Portfolio 1"
    )
    print_portfolio(response)

    # Test 2: KR Conservative
    print("\n[Test 2] KR Conservative - 5M KRW, 8 stocks")
    response2 = await generate_portfolio(
        budget=5_000_000,
        country='KR',
        risk_level='conservative',
        num_stocks=8
    )
    print_portfolio(response2)

    # Test 3: KR Aggressive
    print("\n[Test 3] KR Aggressive - 20M KRW, 12 stocks")
    response3 = await generate_portfolio(
        budget=20_000_000,
        country='KR',
        risk_level='aggressive',
        num_stocks=12
    )
    print_portfolio(response3)


if __name__ == '__main__':
    asyncio.run(main())
