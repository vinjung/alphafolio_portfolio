# -*- coding: utf-8 -*-
"""
Portfolio Configuration

Constants and constraints for alpha portfolio generation.
Designed for admin GUI modification.

File: create_portfolio/config.py
Created: 2025-12-24
"""

from typing import Dict, Any

# ============================================================================
# 1. Budget Constraints (예산 제약)
# ============================================================================

BUDGET_LIMITS = {
    'KR': {
        'min': 500_000,           # 50만원
        'max': 100_000_000,       # 1억원
        'currency': 'KRW',
        'display_unit': '만원'
    },
    'US': {
        'min': 500_000,           # 50만원
        'max': 100_000_000,       # 1억원
        'currency': 'USD',
        'display_unit': '만원'
    },
    'MIXED': {
        'min': 500_000,           # 50만원
        'max': 100_000_000,       # 1억원
        'currency': 'MIXED',
        'display_unit': '만원'
    }
}

# ============================================================================
# 2. Stock Count Constraints (종목 수 제약)
# ============================================================================

STOCK_COUNT_LIMITS = {
    'min': 5,
    'max': 20,
    'default': 10
}

# ============================================================================
# 3. Risk Level Constraints (리스크 수준별 제약조건)
# ============================================================================
# Based on 기획안 섹션 3.2, 4.1

RISK_LEVEL_CONSTRAINTS = {
    'conservative': {
        'name_kr': '안정형',
        'description': '변동성과 손실을 최소화하는 보수적 전략',
        # Filtering constraints (섹션 3.2)
        'volatility_max': 40.0,          # 연환산 변동성 상한 (%)
        'mdd_max': -20.0,                 # 최대 낙폭 상한 (%)
        'beta_min': 0.5,                  # 베타 하한
        'beta_max': 1.2,                  # 베타 상한
        'growth_score_min': None,         # Growth Score 하한 (안정형은 제한 없음)
        # Weight constraints (섹션 4.1)
        'max_weight_per_stock': 0.15,    # 단일 종목 최대 비중 (15%)
        'max_weight_per_sector': 0.30,   # 단일 섹터 최대 비중 (30%)
        # Quality filter (섹션 3.2.2)
        'quality_score_min': 50,         # Quality Score 안전장치
    },
    'balanced': {
        'name_kr': '균형형',
        'description': '수익과 리스크의 균형을 추구하는 전략',
        # Filtering constraints
        'volatility_max': 60.0,
        'mdd_max': -30.0,
        'beta_min': 0.3,
        'beta_max': 1.5,
        'growth_score_min': None,
        # Weight constraints
        'max_weight_per_stock': 0.20,    # 20%
        'max_weight_per_sector': 0.40,   # 40%
        # Quality filter
        'quality_score_min': 40,
    },
    'aggressive': {
        'name_kr': '공격형',
        'description': '높은 수익을 추구하는 적극적 전략',
        # Filtering constraints
        'volatility_max': None,          # 제한 없음
        'mdd_max': None,                 # 제한 없음
        'beta_min': None,                # 제한 없음
        'beta_max': None,                # 제한 없음
        'growth_score_min': 60,          # Growth Score 60 이상
        # Weight constraints
        'max_weight_per_stock': 0.50,    # 50%
        'max_weight_per_sector': 0.50,   # 50%
        # Quality filter
        'quality_score_min': 30,
    }
}

# ============================================================================
# 3-1. Country-specific Weight Override (국가별 비중 제약 오버라이드)
# ============================================================================
# US 시장은 개별 종목 집중도를 낮추어 분산 효과 강화

COUNTRY_WEIGHT_OVERRIDE = {
    'US': {
        'conservative': {'max_weight_per_stock': 0.15},
        'balanced': {'max_weight_per_stock': 0.20},
        'aggressive': {'max_weight_per_stock': 0.35},  # 50% -> 35%
    },
    'KR': None,     # 기본값 사용
    'MIXED': None,  # 기본값 사용
}

# ============================================================================
# 4. Universe Filter Constraints (유니버스 필터링 조건)
# ============================================================================
# Based on 기획안 섹션 3.1

UNIVERSE_FILTER = {
    'KR': {
        'min_trading_value': 500_000_000,    # 일평균 5억원 이상
        'min_final_grade': ['강력 매수', '매수', '매수 고려'],  # 매수 고려 이상
        'min_confidence_score': 50,           # confidence_score >= 50
        'exclude_risk_flags': ['EXTREME_RISK'],  # 극단적 위험 종목 제외
        'lookback_days': 20,                  # 거래대금 평균 산출 기간
    },
    'US': {
        'min_trading_value': 500_000,        # 일평균 $500K 이상 (USD) - 기존 $1M에서 완화
        'min_final_grade': ['강력 매수', '매수', '매수 고려'],
        'min_confidence_score': 50,
        'exclude_risk_flags': ['EXTREME_RISK'],
        'lookback_days': 20,
        # US specific (섹션 3.2.1)
        'max_gap_down': -0.10,               # Gap-down 제한 (-10%)
    }
}

# ============================================================================
# 5. Weight Allocation Parameters (비중 배분 파라미터)
# ============================================================================
# Based on 기획안 섹션 4.1

WEIGHT_ALLOCATION = {
    # Hybrid approach weights
    'score_weight': 0.6,          # 점수 기반 비중 (60%)
    'volatility_weight': 0.4,     # 변동성 역가중 비중 (40%)

    # Minimum weight
    'min_weight_per_stock': 0.03,  # 최소 비중 (3%)
}

# ============================================================================
# 6. Cost Parameters (비용 파라미터)
# ============================================================================
# Based on 기획안 섹션 5

COST_PARAMS = {
    'KR': {
        'commission_rate': 0.00015,    # 수수료 0.015% (매수+매도)
        'tax_rate': 0.0018,            # 거래세 0.18% (매도 시)
        'slippage_base': 0.001,        # 기본 슬리피지 0.1%
        'slippage_low_liquidity': 0.005,  # 저유동성 추가 슬리피지 0.4%
    },
    'US': {
        'commission_rate': 0.0,        # 수수료 무료 (대부분 브로커)
        'tax_rate': 0.0,               # 매도세 없음 (양도세는 별도)
        'slippage_base': 0.001,        # 기본 슬리피지 0.1%
        'slippage_low_liquidity': 0.005,
        'sec_fee_rate': 0.0000278,     # SEC Fee (매도 시)
    }
}

# ============================================================================
# 7. Backtest Parameters (백테스트 파라미터)
# ============================================================================
# Based on 기획안 섹션 6

BACKTEST_PARAMS = {
    'period_months': 3,            # 백테스트 기간 (3개월)
    'benchmark': {
        'KR': 'KOSPI200',
        'US': 'SPY'
    },
    'min_trading_days': 60,        # 최소 거래일 수
    'rebalance_frequency': None,   # Phase 1에서는 리밸런싱 없음
}

# ============================================================================
# 8. Sector Mapping (섹터 매핑)
# ============================================================================

SECTOR_MAPPING = {
    'KR': {
        'source_column': 'industry',      # kr_stock_detail.industry
        'table': 'kr_stock_detail',
    },
    'US': {
        'source_column': 'sector',        # us_stock_basic.sector
        'table': 'us_stock_basic',
    }
}

# ============================================================================
# 9. Default Values (기본값)
# ============================================================================

DEFAULTS = {
    'country': 'KR',
    'risk_level': 'balanced',
    'num_stocks': 10,
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_risk_constraints(risk_level: str) -> Dict[str, Any]:
    """Get constraints for a specific risk level"""
    return RISK_LEVEL_CONSTRAINTS.get(risk_level, RISK_LEVEL_CONSTRAINTS['balanced'])


def get_universe_filter(country: str) -> Dict[str, Any]:
    """Get universe filter for a specific country"""
    return UNIVERSE_FILTER.get(country, UNIVERSE_FILTER['KR'])


def get_budget_limits(country: str) -> Dict[str, Any]:
    """Get budget limits for a specific country"""
    return BUDGET_LIMITS.get(country, BUDGET_LIMITS['KR'])


def get_cost_params(country: str) -> Dict[str, Any]:
    """Get cost parameters for a specific country"""
    return COST_PARAMS.get(country, COST_PARAMS['KR'])


def get_max_weight_per_stock(risk_level: str, country: str) -> float:
    """
    Get max_weight_per_stock with country-specific override.

    Args:
        risk_level: 'conservative', 'balanced', or 'aggressive'
        country: 'KR', 'US', or 'MIXED'

    Returns:
        max_weight_per_stock value (0.0 ~ 1.0)
    """
    # Get base value from risk level constraints
    base_constraints = RISK_LEVEL_CONSTRAINTS.get(risk_level, RISK_LEVEL_CONSTRAINTS['balanced'])
    base_value = base_constraints.get('max_weight_per_stock', 0.20)

    # Check for country-specific override
    country_override = COUNTRY_WEIGHT_OVERRIDE.get(country)
    if country_override:
        risk_override = country_override.get(risk_level)
        if risk_override and 'max_weight_per_stock' in risk_override:
            return risk_override['max_weight_per_stock']

    return base_value
