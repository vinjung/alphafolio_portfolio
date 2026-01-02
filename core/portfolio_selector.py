# -*- coding: utf-8 -*-
"""
Portfolio Stock Selector

Selects top stocks based on selection score with grade consistency filter.
Applies diversification constraints.

File: create_portfolio/portfolio_selector.py
Created: 2025-12-24
"""

import os
import asyncio
import asyncpg
import logging
import numpy as np
from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from models import (
    ProcessedInput,
    StockCandidate,
    Country,
    RiskLevel,
)

# Load .env from quant directory
env_path = Path(__file__).parent.parent / 'quant' / '.env'
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Grade Consistency Configuration
# ============================================================================

# 연속 매수 등급 유지 최소 일수 (기본값: 5일)
MIN_CONSECUTIVE_BUY_DAYS = 5

# 리스크 레벨별 등급 일관성 가중치
GRADE_CONSISTENCY_WEIGHTS = {
    'conservative': 0.20,   # 보수적: 일관성 중요
    'balanced': 0.15,       # 균형: 적당한 가중치
    'aggressive': 0.10,     # 공격적: 성장 우선
}

# ============================================================================
# Momentum Factor Configuration (US 수익률 개선)
# ============================================================================

# 리스크 레벨별 모멘텀 가중치 및 룩백 비율
MOMENTUM_CONFIG = {
    'conservative': {'weight': 0.05, 'lookback_1m': 0.3, 'lookback_3m': 0.7},
    'balanced': {'weight': 0.10, 'lookback_1m': 0.4, 'lookback_3m': 0.6},
    'aggressive': {'weight': 0.20, 'lookback_1m': 0.5, 'lookback_3m': 0.5},
}

# US-specific: 모멘텀 가중치 강화 (US 시장은 모멘텀 효과가 더 강함)
US_MOMENTUM_CONFIG = {
    'conservative': {'weight': 0.10, 'lookback_1m': 0.4, 'lookback_3m': 0.6},
    'balanced': {'weight': 0.15, 'lookback_1m': 0.5, 'lookback_3m': 0.5},
    'aggressive': {'weight': 0.25, 'lookback_1m': 0.6, 'lookback_3m': 0.4},
}

# US-specific Selection Score weights
# US 시장: Sector Rotation 약화, Final Score/Conviction 강화
US_SELECTION_SCORE_WEIGHTS = {
    'conservative': {'alpha': 0.45, 'beta': 0.30, 'gamma': 0.05},
    'balanced': {'alpha': 0.50, 'beta': 0.25, 'gamma': 0.10},
    'aggressive': {'alpha': 0.55, 'beta': 0.20, 'gamma': 0.10},
}

# 매수 등급 목록
KR_BUY_GRADES = ['강력 매수', '매수', '매수 고려']
US_BUY_GRADES = ['강력 매수', '매수', '매수 고려']

# ============================================================================
# Sector Diversification Configuration (2단계)
# ============================================================================

# 리스크 레벨별 섹터 분산 설정
SECTOR_DIVERSIFICATION_CONFIG = {
    'conservative': {
        'min_sectors': 5,       # 최소 5개 섹터에 분산
        'max_per_sector': 2,    # 섹터당 최대 2종목
    },
    'balanced': {
        'min_sectors': 4,       # 최소 4개 섹터에 분산
        'max_per_sector': 3,    # 섹터당 최대 3종목
    },
    'aggressive': {
        'min_sectors': 3,       # 최소 3개 섹터에 분산
        'max_per_sector': 4,    # 섹터당 최대 4종목
    },
}

# ============================================================================
# Market Cap Tiering Configuration (3단계)
# ============================================================================

# 시가총액 티어 기준 (USD 기준, US/KR 통합 처리)
# US: 달러, KR: 원화 (내부에서 환율 적용)
MARKET_CAP_TIERS = {
    'large': 7_000_000_000,    # 대형주: > $7B (~10조원)
    'mid': 700_000_000,        # 중형주: $700M ~ $7B (~1조~10조원)
    # small: < $700M (~1조원 미만)
}

# 리스크 레벨별 시가총액 티어 할당 비율
MARKET_CAP_TIER_ALLOCATION = {
    'conservative': {
        'large': 0.60,     # 대형주 60%
        'mid': 0.30,       # 중형주 30%
        'small': 0.10,     # 소형주 10%
    },
    'balanced': {
        'large': 0.40,     # 대형주 40%
        'mid': 0.40,       # 중형주 40%
        'small': 0.20,     # 소형주 20%
    },
    'aggressive': {
        'large': 0.20,     # 대형주 20%
        'mid': 0.40,       # 중형주 40%
        'small': 0.40,     # 소형주 40%
    },
}

# KRW -> USD 환율 (시총 비교용)
KRW_TO_USD_RATE = 1400


@dataclass
class GradeConsistency:
    """등급 일관성 정보"""
    symbol: str
    consecutive_buy_days: int
    total_buy_days: int
    total_days: int
    buy_ratio: float
    passes_filter: bool


@dataclass
class SelectionResult:
    """종목 선정 결과"""
    symbol: str
    stock_name: str
    country: Country
    sector: Optional[str]

    # 원본 점수
    final_score: float
    conviction_score: Optional[float]
    sector_rotation_score: Optional[float]

    # 등급 일관성
    consecutive_buy_days: int
    grade_consistency_score: float

    # 최종 선정 점수
    selection_score: float
    z_score: float

    # 다단계 정렬용 추가 지표
    sharpe_ratio: Optional[float] = None
    liquidity_score: Optional[float] = None
    market_cap: Optional[float] = None

    # 모멘텀 지표 (US 수익률 개선)
    return_1m: Optional[float] = None           # 1개월 수익률 (%)
    return_3m: Optional[float] = None           # 3개월 수익률 (%)
    momentum_return_score: Optional[float] = None  # 모멘텀 점수 (0-100)

    # 리스크 지표 (5단계: 리스크 패리티 비중용)
    volatility_annual: Optional[float] = None

    # 선정 이유
    selection_reasons: List[str] = field(default_factory=list)


@dataclass
class SelectorSummary:
    """선정 결과 요약"""
    total_candidates: int
    after_consistency_filter: int
    final_selected: int
    avg_selection_score: float
    sector_distribution: Dict[str, int]
    selected_stocks: List[SelectionResult]


class StockSelector:
    """
    종목 선정기

    1. 등급 일관성 필터 적용
    2. Selection Score 계산 (Z-score 정규화)
    3. 섹터 다변화 적용
    4. Top-N 선정
    """

    def __init__(self, db_pool=None):
        """
        Initialize selector.

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
            logger.info("Database pool initialized")

    async def close(self):
        """Close database connection if owned"""
        if self._owns_pool and self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    async def select_stocks(
        self,
        candidates: List[StockCandidate],
        processed_input: ProcessedInput
    ) -> SelectorSummary:
        """
        종목 선정 수행.

        Args:
            candidates: Risk filter 통과한 후보 종목
            processed_input: 처리된 입력

        Returns:
            SelectorSummary with selected stocks
        """
        await self.initialize()

        country = processed_input.request.country
        risk_level = processed_input.request.risk_level
        num_stocks = processed_input.request.num_stocks
        analysis_date = processed_input.analysis_date

        logger.info(f"Selecting {num_stocks} stocks from {len(candidates)} candidates")
        logger.info(f"Risk level: {risk_level.value}, Country: {country.value}")

        total_candidates = len(candidates)

        # Step 1: 등급 일관성 계산 및 필터링
        consistency_map = await self._calculate_grade_consistency(
            candidates, country, analysis_date
        )

        filtered_candidates = []
        for c in candidates:
            consistency = consistency_map.get(c.symbol)
            if consistency and consistency.passes_filter:
                filtered_candidates.append((c, consistency))

        logger.info(f"After consistency filter: {len(filtered_candidates)} candidates")

        if not filtered_candidates:
            logger.warning("No candidates passed consistency filter!")
            return SelectorSummary(
                total_candidates=total_candidates,
                after_consistency_filter=0,
                final_selected=0,
                avg_selection_score=0,
                sector_distribution={},
                selected_stocks=[]
            )

        # Step 2: 추가 지표 조회 (sharpe_ratio, market_cap, liquidity)
        additional_metrics = await self._fetch_additional_metrics(
            [c for c, _ in filtered_candidates], country, analysis_date
        )

        # Step 3: Selection Score 계산
        selection_results = self._calculate_selection_scores(
            filtered_candidates, risk_level, additional_metrics
        )

        # Step 4: Z-score 정규화 및 다단계 정렬
        selection_results = self._normalize_with_zscore(selection_results)

        # Step 5: 섹터 분산 강제 적용 후 Top-N 선정
        max_per_sector = processed_input.constraints.weight_constraints.max_weight_per_sector
        selected = self._select_with_diversification(
            selection_results, num_stocks, max_per_sector, risk_level
        )

        # Summary 생성
        sector_dist = {}
        for s in selected:
            sector = s.sector or 'Unknown'
            sector_dist[sector] = sector_dist.get(sector, 0) + 1

        avg_score = sum(s.selection_score for s in selected) / len(selected) if selected else 0

        return SelectorSummary(
            total_candidates=total_candidates,
            after_consistency_filter=len(filtered_candidates),
            final_selected=len(selected),
            avg_selection_score=avg_score,
            sector_distribution=sector_dist,
            selected_stocks=selected
        )

    async def _calculate_grade_consistency(
        self,
        candidates: List[StockCandidate],
        country: Country,
        analysis_date: date
    ) -> Dict[str, GradeConsistency]:
        """
        각 후보 종목의 등급 일관성 계산.

        Args:
            candidates: 후보 종목 리스트
            country: 국가
            analysis_date: 분석 기준일

        Returns:
            symbol -> GradeConsistency 매핑
        """
        result = {}

        # MIXED: KR/US 분리해서 각각 처리
        if country == Country.MIXED:
            kr_candidates = [c for c in candidates if c.country == Country.KR]
            us_candidates = [c for c in candidates if c.country == Country.US]

            if kr_candidates:
                kr_result = await self._calculate_grade_consistency(
                    kr_candidates, Country.KR, analysis_date
                )
                result.update(kr_result)

            if us_candidates:
                us_result = await self._calculate_grade_consistency(
                    us_candidates, Country.US, analysis_date
                )
                result.update(us_result)

            passed = sum(1 for c in result.values() if c.passes_filter)
            logger.info(f"Grade consistency: {passed}/{len(result)} passed (>= {MIN_CONSECUTIVE_BUY_DAYS} days)")
            return result

        if country == Country.US:
            buy_grades = US_BUY_GRADES
            table_name = 'us_stock_grade'
        else:
            buy_grades = KR_BUY_GRADES
            table_name = 'kr_stock_grade'

        # 실제 데이터가 있는 최신 날짜 조회
        async with self.pool.acquire() as conn:
            date_query = f"""
            SELECT MAX(date) as latest_date
            FROM {table_name}
            WHERE date <= $1
            """
            row = await conn.fetchrow(date_query, analysis_date)
            actual_date = row['latest_date'] if row and row['latest_date'] else analysis_date

        # 각 종목별 등급 이력 조회 (analysis_date 이전 데이터만)
        symbols = [c.symbol for c in candidates]

        async with self.pool.acquire() as conn:
            # 한 번에 모든 종목의 이력 조회 (analysis_date 이하만)
            history_query = f"""
            SELECT symbol, date, final_grade
            FROM {table_name}
            WHERE symbol = ANY($1)
              AND final_grade IS NOT NULL
              AND date <= $2
            ORDER BY symbol, date DESC
            """
            rows = await conn.fetch(history_query, symbols, analysis_date)

        # 종목별 그룹핑
        from collections import defaultdict
        history_by_symbol = defaultdict(list)
        for row in rows:
            history_by_symbol[row['symbol']].append({
                'date': row['date'],
                'grade': row['final_grade']
            })

        # 각 종목별 일관성 계산
        for symbol in symbols:
            history = history_by_symbol.get(symbol, [])

            if not history:
                result[symbol] = GradeConsistency(
                    symbol=symbol,
                    consecutive_buy_days=0,
                    total_buy_days=0,
                    total_days=0,
                    buy_ratio=0,
                    passes_filter=False
                )
                continue

            # 연속 매수 등급 일수 계산 (최신일부터)
            consecutive_days = 0
            for h in history:
                if h['grade'] in buy_grades:
                    consecutive_days += 1
                else:
                    break

            total_buy_days = sum(1 for h in history if h['grade'] in buy_grades)
            total_days = len(history)
            buy_ratio = total_buy_days / total_days if total_days > 0 else 0

            result[symbol] = GradeConsistency(
                symbol=symbol,
                consecutive_buy_days=consecutive_days,
                total_buy_days=total_buy_days,
                total_days=total_days,
                buy_ratio=buy_ratio,
                passes_filter=consecutive_days >= MIN_CONSECUTIVE_BUY_DAYS
            )

        passed = sum(1 for c in result.values() if c.passes_filter)
        logger.info(f"Grade consistency: {passed}/{len(result)} passed (>= {MIN_CONSECUTIVE_BUY_DAYS} days)")

        return result

    async def _fetch_additional_metrics(
        self,
        candidates: List[StockCandidate],
        country: Country,
        analysis_date: date
    ) -> Dict[str, Dict[str, float]]:
        """
        다단계 정렬용 추가 지표 조회.

        Args:
            candidates: 후보 종목 리스트
            country: 국가
            analysis_date: 분석 기준일

        Returns:
            symbol -> {sharpe_ratio, market_cap, liquidity_score} 매핑
        """
        result = {}

        if country == Country.MIXED:
            kr_candidates = [c for c in candidates if c.country == Country.KR]
            us_candidates = [c for c in candidates if c.country == Country.US]

            if kr_candidates:
                kr_result = await self._fetch_additional_metrics(
                    kr_candidates, Country.KR, analysis_date
                )
                result.update(kr_result)

            if us_candidates:
                us_result = await self._fetch_additional_metrics(
                    us_candidates, Country.US, analysis_date
                )
                result.update(us_result)

            return result

        symbols = [c.symbol for c in candidates]
        if not symbols:
            return result

        async with self.pool.acquire() as conn:
            if country == Country.US:
                # US: us_stock_grade (sharpe_ratio) + us_stock_basic (market_cap) + returns
                query = """
                WITH latest_prices AS (
                    SELECT DISTINCT ON (symbol) symbol, date, close
                    FROM us_daily
                    WHERE symbol = ANY($1) AND date <= $2
                    ORDER BY symbol, date DESC
                ),
                price_1m AS (
                    SELECT DISTINCT ON (symbol) symbol, close as close_1m
                    FROM us_daily
                    WHERE symbol = ANY($1) AND date <= ($2::date - INTERVAL '21 days')::date
                    ORDER BY symbol, date DESC
                ),
                price_3m AS (
                    SELECT DISTINCT ON (symbol) symbol, close as close_3m
                    FROM us_daily
                    WHERE symbol = ANY($1) AND date <= ($2::date - INTERVAL '63 days')::date
                    ORDER BY symbol, date DESC
                )
                SELECT
                    g.symbol,
                    g.sharpe_ratio,
                    b.market_cap,
                    CASE
                        WHEN b.market_cap > 0 THEN (lp.close * 1000000) / b.market_cap
                        ELSE 0
                    END as liquidity_score,
                    CASE
                        WHEN p1.close_1m > 0 THEN ((lp.close / p1.close_1m) - 1) * 100
                        ELSE NULL
                    END as return_1m,
                    CASE
                        WHEN p3.close_3m > 0 THEN ((lp.close / p3.close_3m) - 1) * 100
                        ELSE NULL
                    END as return_3m
                FROM us_stock_grade g
                LEFT JOIN us_stock_basic b ON g.symbol = b.symbol
                LEFT JOIN latest_prices lp ON g.symbol = lp.symbol
                LEFT JOIN price_1m p1 ON g.symbol = p1.symbol
                LEFT JOIN price_3m p3 ON g.symbol = p3.symbol
                WHERE g.symbol = ANY($1)
                  AND g.date = (
                      SELECT MAX(date) FROM us_stock_grade
                      WHERE symbol = ANY($1) AND date <= $2
                  )
                """
            else:
                # KR: kr_stock_grade (sharpe_ratio) + kr_intraday_total (market_cap, trading_value) + returns
                query = """
                WITH latest_prices AS (
                    SELECT DISTINCT ON (symbol) symbol, date, close
                    FROM kr_intraday_total
                    WHERE symbol = ANY($1) AND date <= $2
                    ORDER BY symbol, date DESC
                ),
                price_1m AS (
                    SELECT DISTINCT ON (symbol) symbol, close as close_1m
                    FROM kr_intraday_total
                    WHERE symbol = ANY($1) AND date <= ($2::date - INTERVAL '21 days')::date
                    ORDER BY symbol, date DESC
                ),
                price_3m AS (
                    SELECT DISTINCT ON (symbol) symbol, close as close_3m
                    FROM kr_intraday_total
                    WHERE symbol = ANY($1) AND date <= ($2::date - INTERVAL '63 days')::date
                    ORDER BY symbol, date DESC
                )
                SELECT
                    g.symbol,
                    g.sharpe_ratio,
                    lp.market_cap,
                    CASE
                        WHEN lp.market_cap > 0 THEN lp.trading_value / lp.market_cap
                        ELSE 0
                    END as liquidity_score,
                    CASE
                        WHEN p1.close_1m > 0 THEN ((lp.close / p1.close_1m) - 1) * 100
                        ELSE NULL
                    END as return_1m,
                    CASE
                        WHEN p3.close_3m > 0 THEN ((lp.close / p3.close_3m) - 1) * 100
                        ELSE NULL
                    END as return_3m
                FROM kr_stock_grade g
                LEFT JOIN (
                    SELECT DISTINCT ON (symbol) symbol, market_cap, trading_value, close
                    FROM kr_intraday_total
                    WHERE symbol = ANY($1) AND date <= $2
                    ORDER BY symbol, date DESC
                ) lp ON g.symbol = lp.symbol
                LEFT JOIN price_1m p1 ON g.symbol = p1.symbol
                LEFT JOIN price_3m p3 ON g.symbol = p3.symbol
                WHERE g.symbol = ANY($1)
                  AND g.date = (
                      SELECT MAX(date) FROM kr_stock_grade
                      WHERE symbol = ANY($1) AND date <= $2
                  )
                """

            rows = await conn.fetch(query, symbols, analysis_date)

        for row in rows:
            result[row['symbol']] = {
                'sharpe_ratio': float(row['sharpe_ratio']) if row['sharpe_ratio'] else None,
                'market_cap': float(row['market_cap']) if row['market_cap'] else None,
                'liquidity_score': float(row['liquidity_score']) if row['liquidity_score'] else None,
                'return_1m': float(row['return_1m']) if row['return_1m'] else None,
                'return_3m': float(row['return_3m']) if row['return_3m'] else None,
            }

        logger.info(f"Fetched additional metrics for {len(result)} symbols")
        return result

    def _calculate_selection_scores(
        self,
        candidates: List[Tuple[StockCandidate, GradeConsistency]],
        risk_level: RiskLevel,
        additional_metrics: Dict[str, Dict[str, float]] = None
    ) -> List[SelectionResult]:
        """
        Selection Score 계산 (모멘텀 팩터 포함).

        공식: Selection Score = α × Final Score
                              + β × Conviction Score
                              + γ × Sector Rotation Score
                              + δ × Grade Consistency Score
                              + ε × Momentum Return Score (US 강화)

        Args:
            candidates: (StockCandidate, GradeConsistency) 튜플 리스트
            risk_level: 리스크 레벨
            additional_metrics: 추가 지표 (sharpe_ratio, market_cap, liquidity_score, return_1m, return_3m)

        Returns:
            SelectionResult 리스트
        """
        if additional_metrics is None:
            additional_metrics = {}

        # Check if all candidates are US (for US-specific weights)
        is_us_portfolio = all(c.country == Country.US for c, _ in candidates)

        # 리스크 레벨별 가중치 (US 전용 가중치 적용)
        if is_us_portfolio and risk_level.value in US_SELECTION_SCORE_WEIGHTS:
            us_weights = US_SELECTION_SCORE_WEIGHTS[risk_level.value]
            alpha, beta, gamma = us_weights['alpha'], us_weights['beta'], us_weights['gamma']
            logger.info(f"Using US-specific selection weights: alpha={alpha}, beta={beta}, gamma={gamma}")
        elif risk_level == RiskLevel.CONSERVATIVE:
            alpha, beta, gamma = 0.40, 0.30, 0.10
        elif risk_level == RiskLevel.AGGRESSIVE:
            alpha, beta, gamma = 0.50, 0.15, 0.25
        else:  # BALANCED
            alpha, beta, gamma = 0.45, 0.25, 0.15

        delta = GRADE_CONSISTENCY_WEIGHTS.get(risk_level.value, 0.15)

        # 모멘텀 가중치 설정 (US 강화)
        if is_us_portfolio:
            momentum_config = US_MOMENTUM_CONFIG.get(risk_level.value, US_MOMENTUM_CONFIG['balanced'])
            logger.info(f"Using US-specific momentum config: weight={momentum_config['weight']}")
        else:
            momentum_config = MOMENTUM_CONFIG.get(risk_level.value, MOMENTUM_CONFIG['balanced'])

        epsilon = momentum_config['weight']  # 모멘텀 가중치

        # 가중치 합이 1이 되도록 정규화
        total = alpha + beta + gamma + delta + epsilon
        alpha = alpha / total
        beta = beta / total
        gamma = gamma / total
        delta = delta / total
        epsilon = epsilon / total

        # 모멘텀 점수 계산을 위해 먼저 수익률 수집
        returns_1m = []
        returns_3m = []
        for candidate, _ in candidates:
            metrics = additional_metrics.get(candidate.symbol, {})
            r1 = metrics.get('return_1m')
            r3 = metrics.get('return_3m')
            if r1 is not None:
                returns_1m.append(r1)
            if r3 is not None:
                returns_3m.append(r3)

        # 수익률 통계 (Z-score 변환용)
        r1_mean = np.mean(returns_1m) if returns_1m else 0
        r1_std = np.std(returns_1m) if len(returns_1m) > 1 else 1
        r3_mean = np.mean(returns_3m) if returns_3m else 0
        r3_std = np.std(returns_3m) if len(returns_3m) > 1 else 1

        results = []

        for candidate, consistency in candidates:
            # 각 점수 준비 (None이면 50점 기본값)
            final_score = candidate.final_score or 50
            conviction_score = candidate.conviction_score or 50
            sector_rotation_score = candidate.sector_rotation_score or 50

            # 등급 일관성 점수 (0-100)
            grade_consistency_score = min(100, consistency.consecutive_buy_days * 100 / 30)
            ratio_bonus = consistency.buy_ratio * 20
            grade_consistency_score = min(100, grade_consistency_score + ratio_bonus)

            # 추가 지표 조회
            metrics = additional_metrics.get(candidate.symbol, {})
            return_1m = metrics.get('return_1m')
            return_3m = metrics.get('return_3m')

            # 모멘텀 점수 계산 (0-100 스케일)
            momentum_return_score = 50  # 기본값
            if return_1m is not None or return_3m is not None:
                # Z-score를 0-100 점수로 변환
                z1 = (return_1m - r1_mean) / r1_std if return_1m is not None and r1_std > 0 else 0
                z3 = (return_3m - r3_mean) / r3_std if return_3m is not None and r3_std > 0 else 0

                # 가중 평균 (설정에 따라)
                w1 = momentum_config['lookback_1m']
                w3 = momentum_config['lookback_3m']
                combined_z = w1 * z1 + w3 * z3

                # Z-score를 0-100으로 변환 (Z=-3 -> 0, Z=0 -> 50, Z=3 -> 100)
                momentum_return_score = max(0, min(100, 50 + combined_z * 16.67))

            # Selection Score 계산 (모멘텀 포함)
            selection_score = (
                alpha * final_score +
                beta * conviction_score +
                gamma * sector_rotation_score +
                delta * grade_consistency_score +
                epsilon * momentum_return_score
            )

            # 선정 이유 생성
            reasons = []
            if final_score >= 75:
                reasons.append(f"높은 종합점수 ({final_score:.1f})")
            if conviction_score and conviction_score >= 70:
                reasons.append(f"강한 확신도 ({conviction_score:.1f})")
            if consistency.consecutive_buy_days >= 15:
                reasons.append(f"우수한 등급 일관성 ({consistency.consecutive_buy_days}일 연속)")
            elif consistency.consecutive_buy_days >= 10:
                reasons.append(f"양호한 등급 일관성 ({consistency.consecutive_buy_days}일 연속)")
            if consistency.buy_ratio >= 0.5:
                reasons.append(f"높은 매수등급 비율 ({consistency.buy_ratio:.0%})")
            if momentum_return_score >= 70:
                reasons.append(f"강한 모멘텀 ({return_1m:.1f}%/1M)" if return_1m else "강한 모멘텀")

            results.append(SelectionResult(
                symbol=candidate.symbol,
                stock_name=candidate.stock_name,
                country=candidate.country,
                sector=candidate.sector,
                final_score=final_score,
                conviction_score=conviction_score,
                sector_rotation_score=sector_rotation_score,
                consecutive_buy_days=consistency.consecutive_buy_days,
                grade_consistency_score=grade_consistency_score,
                selection_score=selection_score,
                z_score=0,  # 다음 단계에서 계산
                sharpe_ratio=metrics.get('sharpe_ratio'),
                liquidity_score=metrics.get('liquidity_score'),
                market_cap=metrics.get('market_cap'),
                return_1m=return_1m,
                return_3m=return_3m,
                momentum_return_score=momentum_return_score,
                volatility_annual=candidate.volatility_annual,
                selection_reasons=reasons
            ))

        return results

    def _normalize_with_zscore(
        self,
        results: List[SelectionResult]
    ) -> List[SelectionResult]:
        """
        Z-score로 정규화하고 다단계 우선순위로 정렬.

        정렬 기준 (내림차순):
        1. z_score: 선정 점수 (정규화)
        2. conviction_score: 확신도 (팩터 일치도)
        3. sharpe_ratio: 위험 대비 수익 효율
        4. liquidity_score: 유동성 (거래대금/시총)
        5. market_cap: 시가총액 (대형주 우선)

        Args:
            results: SelectionResult 리스트

        Returns:
            다단계 정렬된 리스트
        """
        if not results:
            return results

        scores = [r.selection_score for r in results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        if std_score == 0:
            std_score = 1  # 분산이 0인 경우 처리

        for r in results:
            r.z_score = (r.selection_score - mean_score) / std_score

        # 다단계 우선순위 정렬 (모두 내림차순)
        results.sort(key=lambda x: (
            -x.z_score,                              # 1차: 선정점수 (정규화)
            -(x.conviction_score or 0),              # 2차: 확신도
            -(x.sharpe_ratio or -999),               # 3차: 샤프비율 (None은 최하위)
            -(x.liquidity_score or 0),               # 4차: 유동성
            -(x.market_cap or 0)                     # 5차: 시가총액
        ))

        logger.info("Applied multi-level priority sorting: z_score > conviction > sharpe > liquidity > market_cap")

        return results

    def _get_market_cap_tier(
        self,
        market_cap: Optional[float],
        country: Country
    ) -> str:
        """
        시가총액 티어 분류.

        Args:
            market_cap: 시가총액 (US: USD, KR: KRW)
            country: 국가

        Returns:
            티어 문자열: 'large', 'mid', 'small'
        """
        if market_cap is None:
            return 'small'  # 정보 없으면 소형주로 분류

        # KR은 USD로 환산
        if country == Country.KR:
            market_cap_usd = market_cap / KRW_TO_USD_RATE
        else:
            market_cap_usd = market_cap

        if market_cap_usd >= MARKET_CAP_TIERS['large']:
            return 'large'
        elif market_cap_usd >= MARKET_CAP_TIERS['mid']:
            return 'mid'
        else:
            return 'small'

    def _select_with_diversification(
        self,
        results: List[SelectionResult],
        num_stocks: int,
        max_sector_weight: float,
        risk_level: RiskLevel = RiskLevel.BALANCED
    ) -> List[SelectionResult]:
        """
        섹터 분산 + 시가총액 티어링 적용하여 Top-N 선정.

        3라운드 선정 방식:
        - Round 1: 각 섹터에서 최고 점수 1개씩 선정 (min_sectors 확보, 티어 균형 고려)
        - Round 2: 남은 슬롯을 점수 순으로 채움 (섹터 + 티어 제한)
        - Round 3: 목표 미달 시 제한 완화

        Args:
            results: 정렬된 SelectionResult 리스트
            num_stocks: 선정할 종목 수
            max_sector_weight: 섹터당 최대 비중 (기존 호환성)
            risk_level: 리스크 레벨

        Returns:
            선정된 종목 리스트
        """
        if not results:
            return []

        # 리스크 레벨별 섹터 분산 설정 가져오기
        sector_config = SECTOR_DIVERSIFICATION_CONFIG.get(
            risk_level.value,
            SECTOR_DIVERSIFICATION_CONFIG['balanced']
        )
        min_sectors = sector_config['min_sectors']
        max_per_sector = sector_config['max_per_sector']

        # 기존 max_sector_weight 기반 계산과 비교하여 더 엄격한 값 사용
        weight_based_max = max(1, int(num_stocks * max_sector_weight))
        max_per_sector = min(max_per_sector, weight_based_max)

        # 시가총액 티어 할당 설정 가져오기
        tier_config = MARKET_CAP_TIER_ALLOCATION.get(
            risk_level.value,
            MARKET_CAP_TIER_ALLOCATION['balanced']
        )
        # 티어별 목표 종목 수 계산 (올림 처리로 유연성 확보)
        import math
        tier_targets = {
            'large': max(1, math.ceil(num_stocks * tier_config['large'])),
            'mid': max(1, math.ceil(num_stocks * tier_config['mid'])),
            'small': max(1, math.ceil(num_stocks * tier_config['small'])),
        }
        # 합이 num_stocks를 초과하지 않도록 조정
        total_target = sum(tier_targets.values())
        if total_target > num_stocks:
            # small부터 줄이기
            excess = total_target - num_stocks
            for tier in ['small', 'mid', 'large']:
                if excess <= 0:
                    break
                reduce = min(excess, tier_targets[tier] - 1)
                tier_targets[tier] -= reduce
                excess -= reduce

        selected = []
        sector_counts = {}
        tier_counts = {'large': 0, 'mid': 0, 'small': 0}
        selected_symbols = set()

        # 각 종목에 티어 정보 추가
        for r in results:
            r._tier = self._get_market_cap_tier(r.market_cap, r.country)

        # 섹터별 종목 그룹핑 (이미 점수순 정렬됨)
        sector_stocks = {}
        for r in results:
            sector = r.sector or 'Unknown'
            if sector not in sector_stocks:
                sector_stocks[sector] = []
            sector_stocks[sector].append(r)

        # ────────────────────────────────────────────────────
        # Round 1: 각 섹터에서 최고 점수 1개씩 선정 (min_sectors 확보)
        # 티어 균형을 고려하여 부족한 티어 우선
        # ────────────────────────────────────────────────────
        sorted_sectors = sorted(
            sector_stocks.keys(),
            key=lambda s: sector_stocks[s][0].z_score if sector_stocks[s] else -999,
            reverse=True
        )

        for sector in sorted_sectors:
            if len(selected) >= num_stocks:
                break
            if len(sector_counts) >= min_sectors and len(selected) >= min_sectors:
                break

            stocks = sector_stocks[sector]
            # 티어 부족 순서대로 후보 탐색
            tier_priority = sorted(
                ['large', 'mid', 'small'],
                key=lambda t: tier_counts[t] - tier_targets[t]
            )

            selected_stock = None
            for tier in tier_priority:
                for stock in stocks:
                    if stock.symbol not in selected_symbols and stock._tier == tier:
                        selected_stock = stock
                        break
                if selected_stock:
                    break

            # 티어 무관하게 가장 높은 점수 선택 (fallback)
            if not selected_stock:
                for stock in stocks:
                    if stock.symbol not in selected_symbols:
                        selected_stock = stock
                        break

            if selected_stock:
                selected.append(selected_stock)
                selected_symbols.add(selected_stock.symbol)
                sector_counts[sector] = 1
                tier_counts[selected_stock._tier] += 1

        logger.info(f"Round 1: {len(selected)} stocks, {len(sector_counts)} sectors, "
                    f"tiers: L={tier_counts['large']}/M={tier_counts['mid']}/S={tier_counts['small']}")

        # ────────────────────────────────────────────────────
        # Round 2: 남은 슬롯을 점수 순으로 채움 (섹터 + 티어 제한)
        # ────────────────────────────────────────────────────
        for r in results:
            if len(selected) >= num_stocks:
                break
            if r.symbol in selected_symbols:
                continue

            sector = r.sector or 'Unknown'
            tier = r._tier
            current_sector_count = sector_counts.get(sector, 0)
            current_tier_count = tier_counts[tier]

            # 섹터 제한 확인
            if current_sector_count >= max_per_sector:
                continue

            # 티어 제한 확인 (소프트 제한: 목표의 1.5배까지 허용)
            tier_limit = int(tier_targets[tier] * 1.5)
            if current_tier_count >= tier_limit:
                continue

            selected.append(r)
            selected_symbols.add(r.symbol)
            sector_counts[sector] = current_sector_count + 1
            tier_counts[tier] += 1

        logger.info(f"Round 2: {len(selected)} stocks, {len(sector_counts)} sectors, "
                    f"tiers: L={tier_counts['large']}/M={tier_counts['mid']}/S={tier_counts['small']}")

        # ────────────────────────────────────────────────────
        # Round 3: 목표 미달 시 제한 완화하여 추가 선정
        # ────────────────────────────────────────────────────
        if len(selected) < num_stocks:
            for r in results:
                if len(selected) >= num_stocks:
                    break
                if r.symbol not in selected_symbols:
                    selected.append(r)
                    selected_symbols.add(r.symbol)
                    sector = r.sector or 'Unknown'
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    tier_counts[r._tier] += 1

            logger.info(f"Round 3: Relaxed, {len(selected)} stocks")

        # 티어 속성 제거 (임시 속성)
        for r in results:
            if hasattr(r, '_tier'):
                delattr(r, '_tier')

        logger.info(f"Selected {len(selected)} stocks with diversification "
                    f"(sectors={len(sector_counts)}, tiers: L={tier_counts['large']}/M={tier_counts['mid']}/S={tier_counts['small']})")

        return selected


async def select_stocks(
    candidates: List[StockCandidate],
    processed_input: ProcessedInput,
    pool: 'asyncpg.Pool' = None
) -> SelectorSummary:
    """
    종목 선정 편의 함수.

    Args:
        candidates: Risk filter 통과한 후보 종목
        processed_input: 처리된 입력
        pool: Optional shared database pool. If None, creates own pool.

    Returns:
        SelectorSummary
    """
    if pool is not None:
        # Use shared pool (don't close it)
        selector = StockSelector(db_pool=pool)
        selector._owns_pool = False
        return await selector.select_stocks(candidates, processed_input)
    else:
        # Create own pool (backward compatibility)
        selector = StockSelector()
        try:
            return await selector.select_stocks(candidates, processed_input)
        finally:
            await selector.close()


# ============================================================================
# Test
# ============================================================================

async def test_stock_selector():
    """Test stock selection"""
    from portfolio_input import process_portfolio_request
    from portfolio_universe import filter_universe
    from portfolio_risk_filter import filter_by_risk

    print("=" * 70)
    print("Stock Selector - Test")
    print("=" * 70)

    # Test: KR Balanced
    print("\n[Test] KR Balanced - 10 stocks")
    processed = process_portfolio_request(
        budget=10_000_000,
        country='KR',
        risk_level='balanced',
        num_stocks=10
    )

    if not processed.validation.is_valid:
        print("  Input validation failed!")
        return

    # Step 1: Universe filter
    print("\n  Step 1: Universe filtering...")
    candidates = await filter_universe(processed)
    print(f"    Universe candidates: {len(candidates)}")

    # Step 2: Risk filter
    print("\n  Step 2: Risk filtering...")
    risk_result = filter_by_risk(candidates, processed)
    print(f"    After risk filter: {len(risk_result.candidates)}")

    # Step 3: Stock selection
    print("\n  Step 3: Stock selection...")
    selection = await select_stocks(risk_result.candidates, processed)

    print(f"\n  Results:")
    print(f"    Total candidates: {selection.total_candidates}")
    print(f"    After consistency filter: {selection.after_consistency_filter}")
    print(f"    Final selected: {selection.final_selected}")
    print(f"    Avg selection score: {selection.avg_selection_score:.2f}")

    print(f"\n  Sector distribution:")
    for sector, count in sorted(selection.sector_distribution.items(), key=lambda x: -x[1]):
        print(f"    {sector}: {count}")

    print(f"\n  Selected stocks:")
    print(f"  {'No':<3} {'Symbol':<10} {'Name':<14} {'Score':>6} {'Z':>5} {'Conv':>5} {'Sharpe':>6} {'MCap':>10}")
    print(f"  {'-'*3} {'-'*10} {'-'*14} {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*10}")

    for i, s in enumerate(selection.selected_stocks, 1):
        name = (s.stock_name or '')[:12]
        conv = f"{s.conviction_score:.0f}" if s.conviction_score else "-"
        sharpe = f"{s.sharpe_ratio:.2f}" if s.sharpe_ratio else "-"
        mcap = f"{s.market_cap/1e12:.1f}T" if s.market_cap and s.market_cap > 1e12 else \
               f"{s.market_cap/1e9:.0f}B" if s.market_cap and s.market_cap > 1e9 else "-"
        print(f"  {i:<3} {s.symbol:<10} {name:<14} "
              f"{s.selection_score:>6.1f} {s.z_score:>5.2f} {conv:>5} {sharpe:>6} {mcap:>10}")
        if s.selection_reasons:
            print(f"      └─ {', '.join(s.selection_reasons[:2])}")

    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)


if __name__ == '__main__':
    asyncio.run(test_stock_selector())
