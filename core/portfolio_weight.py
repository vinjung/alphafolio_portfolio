# -*- coding: utf-8 -*-
"""
Portfolio Weight Allocator

Allocates portfolio weights based on selection scores.
Applies sector and single stock constraints.
Implements risk parity weighting (5단계).

File: create_portfolio/portfolio_weight.py
Created: 2025-12-24
Updated: 2025-12-29 (5단계: 리스크 패리티 비중)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from models import (
    ProcessedInput,
    WeightConstraints,
    Country,
    RiskLevel,
)
from core.portfolio_selector import SelectionResult
from config import get_max_weight_per_stock

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Risk Parity Configuration (5단계)
# ============================================================================

# 리스크 레벨별 리스크 패리티 적용 강도
# 1.0 = 완전 리스크 패리티, 0.0 = 점수 기반만
RISK_PARITY_BLEND = {
    'conservative': 0.7,    # 보수적: 리스크 분산 중시 (70% 리스크 패리티)
    'balanced': 0.5,        # 균형: 반반 혼합
    'aggressive': 0.3,      # 공격적: 알파 추구 중시 (30% 리스크 패리티)
}

# US-specific: 리스크 패리티 비율 하향 (알파 추구 강화)
RISK_PARITY_BLEND_BY_COUNTRY = {
    'US': {
        'conservative': 0.5,    # US 보수적: 50% (기본 70%)
        'balanced': 0.3,        # US 균형: 30% (기본 50%)
        'aggressive': 0.15,     # US 공격적: 15% (기본 30%)
    },
    'KR': None,     # Use default RISK_PARITY_BLEND
    'MIXED': None,  # Use default RISK_PARITY_BLEND
}

# 변동성 데이터 없을 때 기본값 (%)
DEFAULT_VOLATILITY = 30.0

# ============================================================================
# Portfolio VaR Configuration (6단계)
# ============================================================================

# 리스크 레벨별 포트폴리오 VaR 95% 한도 (일간, %)
PORTFOLIO_VAR_LIMITS = {
    'conservative': 2.0,    # 보수적: VaR < 2%
    'balanced': 3.0,        # 균형: VaR < 3%
    'aggressive': 5.0,      # 공격적: VaR < 5%
}

# US-specific: VaR 한도 완화 (더 높은 리스크 허용)
PORTFOLIO_VAR_LIMITS_BY_COUNTRY = {
    'US': {
        'conservative': 2.5,    # US 보수적: 2.5% (기본 2%)
        'balanced': 4.0,        # US 균형: 4% (기본 3%)
        'aggressive': 7.0,      # US 공격적: 7% (기본 5%)
    },
    'KR': None,     # Use default PORTFOLIO_VAR_LIMITS
    'MIXED': None,  # Use default PORTFOLIO_VAR_LIMITS
}

# VaR 계산 상수
VAR_CONFIDENCE_Z = 1.65  # 95% 신뢰수준 Z-value

# VaR 초과 시 비중 조정 상수
VAR_ADJUSTMENT_FACTOR = 0.9  # 10%씩 비중 감소

# ============================================================================
# Correlation-based Diversification Configuration (7단계)
# ============================================================================

# 동일 섹터 내 종목 쌍 최대 합산 비중 제한
# 같은 섹터 = 높은 상관관계 가정 (섹터를 상관관계 프록시로 사용)
MAX_CORRELATED_PAIR_WEIGHT = {
    'conservative': 0.20,    # 보수적: 섹터 내 쌍당 최대 20%
    'balanced': 0.30,        # 균형: 섹터 내 쌍당 최대 30%
    'aggressive': 0.40,      # 공격적: 섹터 내 쌍당 최대 40%
}

# 상관관계 조정 적용 임계값 (이 비중 이상인 종목 쌍만 조정)
CORRELATION_ADJUSTMENT_THRESHOLD = 0.08  # 8%


@dataclass
class WeightedStock:
    """비중 배분된 종목"""
    symbol: str
    stock_name: str
    country: Country
    sector: Optional[str]

    # 점수 정보
    selection_score: float
    z_score: float
    conviction_score: Optional[float]

    # 비중 정보
    raw_weight: float          # 초기 계산 비중
    adjusted_weight: float     # 제약 조건 적용 후 비중
    final_weight: float        # 최종 비중 (정규화 후)

    # 등급 일관성
    consecutive_buy_days: int

    # 리스크 지표 (5단계)
    volatility_annual: Optional[float] = None

    # 선정 이유
    selection_reasons: List[str] = field(default_factory=list)


@dataclass
class WeightAllocationResult:
    """비중 배분 결과"""
    total_weight: float
    stocks: List[WeightedStock]
    sector_weights: Dict[str, float]
    max_stock_weight: float
    min_stock_weight: float
    weight_concentration: float  # HHI (Herfindahl-Hirschman Index)

    # VaR 정보 (6단계)
    portfolio_var_95: float = 0.0       # 포트폴리오 VaR 95% (일간, %)
    var_limit: float = 0.0              # VaR 한도
    var_within_limit: bool = True       # 한도 준수 여부


class WeightAllocator:
    """
    비중 배분기

    1. Selection Score 기반 초기 비중 계산
    2. Conviction Score 보너스 적용
    3. 섹터/종목 제약 조건 적용
    4. 최종 정규화
    """

    def __init__(
        self,
        constraints: WeightConstraints,
        country: Country = None,
        risk_level: RiskLevel = None
    ):
        """
        Initialize allocator.

        Args:
            constraints: WeightConstraints from processed input
            country: Country for country-specific override
            risk_level: RiskLevel for country-specific override
        """
        self.constraints = constraints
        self.country = country  # Store for country-specific settings
        self.max_weight_per_sector = constraints.max_weight_per_sector
        self.min_weight_per_stock = constraints.min_weight_per_stock

        # Apply country-specific max_weight_per_stock override
        if country and risk_level:
            self.max_weight_per_stock = get_max_weight_per_stock(
                risk_level.value, country.value
            )
            logger.info(f"Max weight per stock: {self.max_weight_per_stock:.0%} "
                        f"(country={country.value}, risk={risk_level.value})")
        else:
            self.max_weight_per_stock = constraints.max_weight_per_stock

    def allocate_weights(
        self,
        selected_stocks: List[SelectionResult],
        risk_level: RiskLevel
    ) -> WeightAllocationResult:
        """
        비중 배분 수행.

        Args:
            selected_stocks: 선정된 종목 리스트
            risk_level: 리스크 레벨

        Returns:
            WeightAllocationResult
        """
        if not selected_stocks:
            return WeightAllocationResult(
                total_weight=0,
                stocks=[],
                sector_weights={},
                max_stock_weight=0,
                min_stock_weight=0,
                weight_concentration=0
            )

        logger.info(f"Allocating weights for {len(selected_stocks)} stocks")

        # Step 1: 초기 비중 계산 (리스크 패리티 + Score 기반)
        weighted_stocks = self._calculate_initial_weights(selected_stocks, risk_level)

        # Step 2: Conviction 보너스 적용
        weighted_stocks = self._apply_conviction_bonus(weighted_stocks, risk_level)

        # Step 3: 종목별 최대/최소 비중 제약 적용
        weighted_stocks = self._apply_stock_constraints(weighted_stocks)

        # Step 4: 섹터별 비중 제약 적용
        weighted_stocks = self._apply_sector_constraints(weighted_stocks)

        # Step 5: 최종 정규화
        weighted_stocks = self._normalize_weights(weighted_stocks)

        # Step 6: 상관관계 기반 분산 조정 (7단계 - VaR 전에 적용)
        weighted_stocks = self._apply_correlation_diversification(
            weighted_stocks, risk_level
        )

        # Step 7: 포트폴리오 VaR 검증 및 조정 (6단계)
        # Check country-specific VaR limit
        var_limit = PORTFOLIO_VAR_LIMITS.get(risk_level.value, 3.0)
        if self.country:
            country_var_config = PORTFOLIO_VAR_LIMITS_BY_COUNTRY.get(self.country.value)
            if country_var_config:
                var_limit = country_var_config.get(risk_level.value, var_limit)
                logger.info(f"Using country-specific VaR limit: {var_limit}% (country={self.country.value})")

        weighted_stocks, portfolio_var = self._verify_and_adjust_var(
            weighted_stocks, var_limit
        )

        # Step 8: 최종 종목별 비중 제약 재적용 (버그 수정)
        weighted_stocks = self._apply_final_stock_constraints(weighted_stocks)

        # 결과 계산
        sector_weights = self._calculate_sector_weights(weighted_stocks)
        max_weight = max(s.final_weight for s in weighted_stocks)
        min_weight = min(s.final_weight for s in weighted_stocks)
        hhi = sum(s.final_weight ** 2 for s in weighted_stocks)

        logger.info(f"Weight allocation complete. Max: {max_weight:.1%}, Min: {min_weight:.1%}")
        logger.info(f"Portfolio VaR 95%: {portfolio_var:.2f}% (limit: {var_limit:.1f}%)")

        return WeightAllocationResult(
            total_weight=sum(s.final_weight for s in weighted_stocks),
            stocks=weighted_stocks,
            sector_weights=sector_weights,
            max_stock_weight=max_weight,
            min_stock_weight=min_weight,
            weight_concentration=hhi,
            portfolio_var_95=portfolio_var,
            var_limit=var_limit,
            var_within_limit=portfolio_var <= var_limit
        )

    def _calculate_initial_weights(
        self,
        selected_stocks: List[SelectionResult],
        risk_level: RiskLevel = RiskLevel.BALANCED
    ) -> List[WeightedStock]:
        """
        초기 비중 계산 (리스크 패리티 적용).

        공식:
        - Score-based: (z_score + shift) / sum(z_score + shift)
        - Risk parity: (1/volatility) / sum(1/volatility)
        - Blended: blend * risk_parity + (1-blend) * score_based

        Args:
            selected_stocks: 선정된 종목
            risk_level: 리스크 레벨 (블렌딩 비율 결정)

        Returns:
            WeightedStock 리스트
        """
        n = len(selected_stocks)

        # 1. Score-based weights
        min_z = min(s.z_score for s in selected_stocks)
        shifted_scores = [s.z_score - min_z + 1 for s in selected_stocks]
        total_score = sum(shifted_scores)
        score_weights = [s / total_score if total_score > 0 else 1/n for s in shifted_scores]

        # 2. Risk parity weights (inverse volatility)
        volatilities = []
        for stock in selected_stocks:
            vol = stock.volatility_annual if stock.volatility_annual else DEFAULT_VOLATILITY
            vol = max(vol, 5.0)  # 최소 5% 변동성으로 제한 (극단값 방지)
            volatilities.append(vol)

        inv_vols = [1.0 / v for v in volatilities]
        total_inv_vol = sum(inv_vols)
        risk_parity_weights = [iv / total_inv_vol for iv in inv_vols]

        # 3. Blended weights (country-specific override)
        # Check country-specific risk parity blend first
        blend = RISK_PARITY_BLEND.get(risk_level.value, 0.5)
        if self.country:
            country_config = RISK_PARITY_BLEND_BY_COUNTRY.get(self.country.value)
            if country_config:
                country_blend = country_config.get(risk_level.value)
                if country_blend is not None:
                    blend = country_blend
                    logger.info(f"Using country-specific risk parity blend: {blend:.0%} "
                                f"(country={self.country.value})")

        weighted_stocks = []
        for i, stock in enumerate(selected_stocks):
            # 블렌딩: 리스크 패리티 + 점수 기반
            raw_weight = blend * risk_parity_weights[i] + (1 - blend) * score_weights[i]

            weighted = WeightedStock(
                symbol=stock.symbol,
                stock_name=stock.stock_name,
                country=stock.country,
                sector=stock.sector,
                selection_score=stock.selection_score,
                z_score=stock.z_score,
                conviction_score=stock.conviction_score,
                raw_weight=raw_weight,
                adjusted_weight=raw_weight,
                final_weight=raw_weight,
                consecutive_buy_days=stock.consecutive_buy_days,
                volatility_annual=volatilities[i],
                selection_reasons=stock.selection_reasons.copy()
            )
            weighted_stocks.append(weighted)

        logger.info(f"Applied risk parity weighting (blend={blend:.0%} risk parity, "
                    f"{1-blend:.0%} score-based)")

        return weighted_stocks

    def _apply_conviction_bonus(
        self,
        stocks: List[WeightedStock],
        risk_level: RiskLevel
    ) -> List[WeightedStock]:
        """
        Conviction Score 기반 보너스 적용.

        보수적: 최대 20% 보너스
        균형: 최대 35% 보너스
        공격적: 최대 50% 보너스

        Args:
            stocks: WeightedStock 리스트
            risk_level: 리스크 레벨

        Returns:
            보너스 적용된 리스트
        """
        if risk_level == RiskLevel.CONSERVATIVE:
            max_bonus = 0.20
        elif risk_level == RiskLevel.AGGRESSIVE:
            max_bonus = 0.50
        else:
            max_bonus = 0.35

        for stock in stocks:
            if stock.conviction_score and stock.conviction_score > 50:
                # Conviction 50~100을 0~max_bonus로 매핑
                bonus_ratio = ((stock.conviction_score - 50) / 50) * max_bonus
                stock.adjusted_weight = stock.raw_weight * (1 + bonus_ratio)
            else:
                stock.adjusted_weight = stock.raw_weight

        # 재정규화
        total = sum(s.adjusted_weight for s in stocks)
        for stock in stocks:
            stock.adjusted_weight = stock.adjusted_weight / total if total > 0 else stock.adjusted_weight

        return stocks

    def _apply_stock_constraints(
        self,
        stocks: List[WeightedStock]
    ) -> List[WeightedStock]:
        """
        종목별 최대/최소 비중 제약 적용.

        Args:
            stocks: WeightedStock 리스트

        Returns:
            제약 적용된 리스트
        """
        iterations = 0
        max_iterations = 10

        while iterations < max_iterations:
            adjusted = False

            for stock in stocks:
                # 최대 비중 초과 시 조정
                if stock.adjusted_weight > self.max_weight_per_stock:
                    excess = stock.adjusted_weight - self.max_weight_per_stock
                    stock.adjusted_weight = self.max_weight_per_stock
                    adjusted = True

                    # 초과분을 다른 종목에 배분
                    others = [s for s in stocks if s.symbol != stock.symbol]
                    if others:
                        dist_per_stock = excess / len(others)
                        for other in others:
                            other.adjusted_weight += dist_per_stock

                # 최소 비중 미달 시 조정
                if stock.adjusted_weight < self.min_weight_per_stock:
                    deficit = self.min_weight_per_stock - stock.adjusted_weight
                    stock.adjusted_weight = self.min_weight_per_stock
                    adjusted = True

                    # 부족분을 다른 종목에서 차감
                    others = [s for s in stocks if s.symbol != stock.symbol and
                              s.adjusted_weight > self.min_weight_per_stock]
                    if others:
                        deduct_per_stock = deficit / len(others)
                        for other in others:
                            other.adjusted_weight = max(
                                self.min_weight_per_stock,
                                other.adjusted_weight - deduct_per_stock
                            )

            if not adjusted:
                break

            iterations += 1

        return stocks

    def _apply_sector_constraints(
        self,
        stocks: List[WeightedStock]
    ) -> List[WeightedStock]:
        """
        섹터별 비중 제약 적용.

        Args:
            stocks: WeightedStock 리스트

        Returns:
            제약 적용된 리스트
        """
        iterations = 0
        max_iterations = 10

        while iterations < max_iterations:
            # 섹터별 비중 계산
            sector_weights = {}
            for stock in stocks:
                sector = stock.sector or 'Unknown'
                sector_weights[sector] = sector_weights.get(sector, 0) + stock.adjusted_weight

            # 초과 섹터 확인 및 조정
            adjusted = False
            for sector, weight in sector_weights.items():
                if weight > self.max_weight_per_sector:
                    excess = weight - self.max_weight_per_sector
                    adjusted = True

                    # 해당 섹터 종목들의 비중 비례 감소
                    sector_stocks = [s for s in stocks if (s.sector or 'Unknown') == sector]
                    for stock in sector_stocks:
                        reduction = (stock.adjusted_weight / weight) * excess
                        stock.adjusted_weight -= reduction

                    # 초과분을 다른 섹터에 배분
                    other_stocks = [s for s in stocks if (s.sector or 'Unknown') != sector]
                    if other_stocks:
                        other_total = sum(s.adjusted_weight for s in other_stocks)
                        for stock in other_stocks:
                            stock.adjusted_weight += (stock.adjusted_weight / other_total) * excess if other_total > 0 else 0

            if not adjusted:
                break

            iterations += 1

        return stocks

    def _normalize_weights(
        self,
        stocks: List[WeightedStock]
    ) -> List[WeightedStock]:
        """
        비중 합이 1이 되도록 정규화.

        Args:
            stocks: WeightedStock 리스트

        Returns:
            정규화된 리스트
        """
        total = sum(s.adjusted_weight for s in stocks)

        if total > 0:
            for stock in stocks:
                stock.final_weight = stock.adjusted_weight / total
        else:
            equal_weight = 1 / len(stocks) if stocks else 0
            for stock in stocks:
                stock.final_weight = equal_weight

        return stocks

    def _apply_correlation_diversification(
        self,
        stocks: List[WeightedStock],
        risk_level: RiskLevel
    ) -> List[WeightedStock]:
        """
        상관관계 기반 분산 조정.

        동일 섹터 내 고비중 종목 쌍의 합산 비중이 한도를 초과하면
        비중을 분산하여 포트폴리오 리스크를 낮춤.

        Args:
            stocks: 비중 배분된 종목 리스트
            risk_level: 리스크 레벨

        Returns:
            조정된 종목 리스트
        """
        max_pair_weight = MAX_CORRELATED_PAIR_WEIGHT.get(risk_level.value, 0.30)
        threshold = CORRELATION_ADJUSTMENT_THRESHOLD

        # 섹터별 종목 그룹핑
        sector_stocks = {}
        for stock in stocks:
            sector = stock.sector or 'Unknown'
            if sector not in sector_stocks:
                sector_stocks[sector] = []
            sector_stocks[sector].append(stock)

        adjusted = False

        for sector, sector_stock_list in sector_stocks.items():
            if len(sector_stock_list) < 2:
                continue

            # 비중 높은 순으로 정렬
            sorted_stocks = sorted(
                sector_stock_list,
                key=lambda s: s.final_weight,
                reverse=True
            )

            # 고비중 종목 쌍 확인 및 조정
            for i in range(len(sorted_stocks) - 1):
                for j in range(i + 1, len(sorted_stocks)):
                    stock_i = sorted_stocks[i]
                    stock_j = sorted_stocks[j]

                    # 둘 다 임계값 이상인 경우만 조정
                    if stock_i.final_weight < threshold or stock_j.final_weight < threshold:
                        continue

                    pair_weight = stock_i.final_weight + stock_j.final_weight

                    if pair_weight > max_pair_weight:
                        # 초과분 계산
                        excess = pair_weight - max_pair_weight

                        # 비중 비례로 감소
                        reduction_i = excess * (stock_i.final_weight / pair_weight)
                        reduction_j = excess * (stock_j.final_weight / pair_weight)

                        stock_i.final_weight -= reduction_i
                        stock_j.final_weight -= reduction_j

                        adjusted = True

        if adjusted:
            # 재정규화
            total = sum(s.final_weight for s in stocks)
            for stock in stocks:
                stock.final_weight /= total

            logger.info(f"Applied correlation-based diversification "
                        f"(max pair weight: {max_pair_weight:.0%})")

        return stocks

    def _calculate_sector_weights(
        self,
        stocks: List[WeightedStock]
    ) -> Dict[str, float]:
        """섹터별 비중 계산"""
        sector_weights = {}
        for stock in stocks:
            sector = stock.sector or 'Unknown'
            sector_weights[sector] = sector_weights.get(sector, 0) + stock.final_weight
        return sector_weights

    def _apply_final_stock_constraints(
        self,
        stocks: List[WeightedStock]
    ) -> List[WeightedStock]:
        """
        최종 단계에서 종목별 max_weight 재적용.

        정규화, 상관관계 조정, VaR 조정 후 max_weight를 초과한 종목이
        있을 수 있으므로 최종적으로 다시 제약을 적용.

        Args:
            stocks: 비중 배분된 종목 리스트

        Returns:
            제약 적용된 종목 리스트
        """
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            adjusted = False
            excess_total = 0.0

            # 초과 비중 확인 및 제한
            for stock in stocks:
                if stock.final_weight > self.max_weight_per_stock:
                    excess = stock.final_weight - self.max_weight_per_stock
                    excess_total += excess
                    stock.final_weight = self.max_weight_per_stock
                    adjusted = True

            if not adjusted:
                break

            # 초과분을 max_weight 미만인 종목에 비례 배분
            under_limit_stocks = [
                s for s in stocks
                if s.final_weight < self.max_weight_per_stock
            ]

            if under_limit_stocks and excess_total > 0:
                # 현재 비중 비례로 배분
                under_total = sum(s.final_weight for s in under_limit_stocks)
                for stock in under_limit_stocks:
                    if under_total > 0:
                        add_weight = excess_total * (stock.final_weight / under_total)
                        stock.final_weight += add_weight

            iteration += 1

        # 최종 정규화 (합이 1.0이 되도록)
        total = sum(s.final_weight for s in stocks)
        if total > 0 and abs(total - 1.0) > 0.001:
            for stock in stocks:
                stock.final_weight /= total

        # 정규화 후 다시 한번 max_weight 초과 확인 및 조정
        for stock in stocks:
            if stock.final_weight > self.max_weight_per_stock:
                stock.final_weight = self.max_weight_per_stock

        # 마지막 정규화
        total = sum(s.final_weight for s in stocks)
        if total > 0:
            for stock in stocks:
                stock.final_weight /= total

        return stocks

    def _calculate_portfolio_var(
        self,
        stocks: List[WeightedStock]
    ) -> float:
        """
        포트폴리오 VaR 95% 계산.

        공식: VaR_p = Z * sqrt(sum(w_i^2 * vol_i^2))
        (상관관계 0 가정 - 보수적 추정)

        Args:
            stocks: 비중 배분된 종목 리스트

        Returns:
            일간 VaR 95% (%)
        """
        # 연간 변동성(%)을 일간(%)으로 변환 (vol_daily = vol_annual / sqrt(252))
        # 예: 연간 20% -> 일간 1.26%
        variance_sum = 0.0

        for stock in stocks:
            vol_annual = stock.volatility_annual or DEFAULT_VOLATILITY
            vol_daily = vol_annual / np.sqrt(252)  # % 단위 유지
            # vol_daily를 비율로 변환 (1.26% -> 0.0126)
            vol_daily_ratio = vol_daily / 100
            variance_sum += (stock.final_weight ** 2) * (vol_daily_ratio ** 2)

        portfolio_std = np.sqrt(variance_sum)
        portfolio_var = VAR_CONFIDENCE_Z * portfolio_std * 100  # % 단위로 반환

        return portfolio_var

    def _verify_and_adjust_var(
        self,
        stocks: List[WeightedStock],
        var_limit: float
    ) -> Tuple[List[WeightedStock], float]:
        """
        포트폴리오 VaR 검증 및 조정.

        VaR 초과 시 고변동성 종목의 비중을 줄여서 VaR 한도 내로 조정.

        Args:
            stocks: 비중 배분된 종목 리스트
            var_limit: VaR 한도 (%)

        Returns:
            (조정된 종목 리스트, 최종 VaR)
        """
        portfolio_var = self._calculate_portfolio_var(stocks)

        if portfolio_var <= var_limit:
            logger.info(f"Portfolio VaR {portfolio_var:.2f}% within limit {var_limit}%")
            return stocks, portfolio_var

        logger.info(f"Portfolio VaR {portfolio_var:.2f}% exceeds limit {var_limit}%, adjusting...")

        # 반복적으로 고변동성 종목 비중 감소
        max_iterations = 20
        iteration = 0

        while portfolio_var > var_limit and iteration < max_iterations:
            # 변동성 높은 순으로 정렬
            high_vol_stocks = sorted(
                stocks,
                key=lambda s: s.volatility_annual or 0,
                reverse=True
            )

            # 상위 30% 고변동성 종목 비중 감소
            n_adjust = max(1, len(stocks) // 3)
            reduced_weight = 0.0

            for i in range(n_adjust):
                stock = high_vol_stocks[i]
                reduction = stock.final_weight * (1 - VAR_ADJUSTMENT_FACTOR)
                stock.final_weight *= VAR_ADJUSTMENT_FACTOR
                reduced_weight += reduction

            # 저변동성 종목에 재배분
            low_vol_stocks = high_vol_stocks[-n_adjust:]
            if low_vol_stocks and reduced_weight > 0:
                add_per_stock = reduced_weight / len(low_vol_stocks)
                for stock in low_vol_stocks:
                    stock.final_weight += add_per_stock

            # 재정규화
            total = sum(s.final_weight for s in stocks)
            for stock in stocks:
                stock.final_weight /= total

            portfolio_var = self._calculate_portfolio_var(stocks)
            iteration += 1

        if iteration > 0:
            logger.info(f"VaR adjustment complete after {iteration} iterations. "
                        f"Final VaR: {portfolio_var:.2f}%")

        return stocks, portfolio_var


def allocate_weights(
    selected_stocks: List[SelectionResult],
    processed_input: ProcessedInput
) -> WeightAllocationResult:
    """
    비중 배분 편의 함수.

    Args:
        selected_stocks: 선정된 종목 리스트
        processed_input: 처리된 입력

    Returns:
        WeightAllocationResult
    """
    allocator = WeightAllocator(
        processed_input.constraints.weight_constraints,
        country=processed_input.request.country,
        risk_level=processed_input.request.risk_level
    )
    return allocator.allocate_weights(
        selected_stocks,
        processed_input.request.risk_level
    )


# ============================================================================
# Test
# ============================================================================

def test_weight_allocator():
    """Test weight allocation"""
    from core.portfolio_selector import SelectionResult
    from models import Country, RiskLevel, WeightConstraints

    print("=" * 70)
    print("Weight Allocator - Test")
    print("=" * 70)

    # Mock selected stocks
    mock_stocks = [
        SelectionResult(
            symbol='005930', stock_name='Samsung', country=Country.KR,
            sector='Electronics', final_score=78.0, conviction_score=75.0,
            sector_rotation_score=60.0, consecutive_buy_days=20,
            grade_consistency_score=80.0, selection_score=72.0, z_score=2.5,
            selection_reasons=['높은 종합점수']
        ),
        SelectionResult(
            symbol='000660', stock_name='SK Hynix', country=Country.KR,
            sector='Electronics', final_score=75.0, conviction_score=70.0,
            sector_rotation_score=55.0, consecutive_buy_days=18,
            grade_consistency_score=75.0, selection_score=68.0, z_score=1.8,
            selection_reasons=['높은 종합점수']
        ),
        SelectionResult(
            symbol='035420', stock_name='NAVER', country=Country.KR,
            sector='IT Services', final_score=72.0, conviction_score=65.0,
            sector_rotation_score=70.0, consecutive_buy_days=15,
            grade_consistency_score=70.0, selection_score=65.0, z_score=1.2,
            selection_reasons=['양호한 등급 일관성']
        ),
        SelectionResult(
            symbol='005380', stock_name='Hyundai Motor', country=Country.KR,
            sector='Automotive', final_score=70.0, conviction_score=60.0,
            sector_rotation_score=50.0, consecutive_buy_days=12,
            grade_consistency_score=65.0, selection_score=62.0, z_score=0.5,
            selection_reasons=['양호한 등급 일관성']
        ),
        SelectionResult(
            symbol='051910', stock_name='LG Chem', country=Country.KR,
            sector='Chemicals', final_score=68.0, conviction_score=55.0,
            sector_rotation_score=45.0, consecutive_buy_days=10,
            grade_consistency_score=60.0, selection_score=58.0, z_score=-0.2,
            selection_reasons=[]
        ),
    ]

    # Test with balanced constraints
    constraints = WeightConstraints(
        max_weight_per_stock=0.25,
        max_weight_per_sector=0.35,
        min_weight_per_stock=0.05
    )

    print("\n[Test 1] Balanced Weight Allocation")
    allocator = WeightAllocator(constraints)
    result = allocator.allocate_weights(mock_stocks, RiskLevel.BALANCED)

    print(f"\n  Total weight: {result.total_weight:.2%}")
    print(f"  Max stock weight: {result.max_stock_weight:.2%}")
    print(f"  Min stock weight: {result.min_stock_weight:.2%}")
    print(f"  Concentration (HHI): {result.weight_concentration:.4f}")

    print(f"\n  Stock weights:")
    print(f"  {'Symbol':<10} {'Name':<18} {'Raw':>8} {'Adj':>8} {'Final':>8}")
    print(f"  {'-'*10} {'-'*18} {'-'*8} {'-'*8} {'-'*8}")
    for s in result.stocks:
        print(f"  {s.symbol:<10} {s.stock_name[:16]:<18} "
              f"{s.raw_weight:>7.1%} {s.adjusted_weight:>7.1%} {s.final_weight:>7.1%}")

    print(f"\n  Sector weights:")
    for sector, weight in sorted(result.sector_weights.items(), key=lambda x: -x[1]):
        print(f"    {sector}: {weight:.1%}")

    # Test 2: Conservative (lower max weight)
    print("\n[Test 2] Conservative Weight Allocation")
    constraints2 = WeightConstraints(
        max_weight_per_stock=0.15,
        max_weight_per_sector=0.30,
        min_weight_per_stock=0.08
    )
    allocator2 = WeightAllocator(constraints2)
    result2 = allocator2.allocate_weights(mock_stocks, RiskLevel.CONSERVATIVE)

    print(f"  Max stock weight: {result2.max_stock_weight:.2%} (limit: 15%)")
    print(f"  Min stock weight: {result2.min_stock_weight:.2%} (limit: 8%)")

    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)


if __name__ == '__main__':
    test_weight_allocator()
