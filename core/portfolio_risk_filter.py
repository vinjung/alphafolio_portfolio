# -*- coding: utf-8 -*-
"""
Portfolio Risk Filter

Applies risk-level specific filters to candidate stocks.
Filters based on volatility, MDD, beta, growth score, quality score,
and sentiment/supply-demand indicators.

File: create_portfolio/portfolio_risk_filter.py
Created: 2025-12-24
Updated: 2025-12-29 (4단계: 센티먼트/수급 필터 추가)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from models import (
    ProcessedInput,
    StockCandidate,
    FilterConstraints,
    Country,
    RiskLevel,
)
from config import get_risk_constraints

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Sentiment/Supply-Demand Filter Configuration (4단계)
# ============================================================================

# 리스크 레벨별 센티먼트 필터 설정
SENTIMENT_FILTER_CONFIG = {
    'conservative': {
        # KR: 외국인/기관 순매수 필터 (엄격)
        'kr_require_positive_flow': True,     # 외국인+기관 순매수 양수 필요
        'kr_min_combined_net': 0,             # 최소 합산 순매수
        # US: 애널리스트 등급 필터 비활성화 (수익률 검증 결과 역효과)
        'us_max_analyst_rating': None,        # 제한 없음
        'us_prefer_positive_insider': True,   # 양수 내부자 신호 선호
    },
    'balanced': {
        'kr_require_positive_flow': False,
        'kr_min_combined_net': -10,           # 약간의 유출 허용
        'us_max_analyst_rating': None,        # 제한 없음
        'us_prefer_positive_insider': False,
    },
    'aggressive': {
        'kr_require_positive_flow': False,
        'kr_min_combined_net': None,          # 제한 없음
        'us_max_analyst_rating': None,        # 제한 없음
        'us_prefer_positive_insider': False,
    },
}


@dataclass
class FilterResult:
    """Result of a single filter application"""
    filter_name: str
    before_count: int
    after_count: int
    removed_count: int
    removed_symbols: List[str]


@dataclass
class RiskFilterSummary:
    """Summary of all risk filtering"""
    total_before: int
    total_after: int
    filter_results: List[FilterResult]
    candidates: List[StockCandidate]


class RiskFilter:
    """
    Apply risk-level specific filters to candidate stocks.

    Filters applied (based on risk level):
    1. Volatility filter (annual volatility max)
    2. MDD filter (max drawdown limit)
    3. Beta filter (beta range)
    4. Growth filter (min growth score for aggressive)
    5. Quality filter (safety net)
    """

    def __init__(self):
        """Initialize risk filter"""
        self.filter_results: List[FilterResult] = []

    def filter_by_risk(
        self,
        candidates: List[StockCandidate],
        processed_input: ProcessedInput
    ) -> RiskFilterSummary:
        """
        Apply all risk filters based on risk level.

        Args:
            candidates: List of StockCandidate from universe filter
            processed_input: Processed input with constraints

        Returns:
            RiskFilterSummary with filtered candidates and stats
        """
        risk_level = processed_input.request.risk_level
        constraints = processed_input.constraints.filter_constraints

        logger.info(f"Applying risk filters for {risk_level.value} level")
        logger.info(f"Starting with {len(candidates)} candidates")

        self.filter_results = []
        total_before = len(candidates)
        filtered = candidates.copy()

        # 1. Volatility filter
        if constraints.volatility_max is not None:
            filtered = self._apply_volatility_filter(
                filtered,
                constraints.volatility_max
            )

        # 2. MDD filter
        if constraints.mdd_max is not None:
            filtered = self._apply_mdd_filter(
                filtered,
                constraints.mdd_max
            )

        # 3. Beta filter
        if constraints.beta_min is not None or constraints.beta_max is not None:
            filtered = self._apply_beta_filter(
                filtered,
                constraints.beta_min,
                constraints.beta_max
            )

        # 4. Growth filter (for aggressive)
        if constraints.growth_score_min is not None:
            filtered = self._apply_growth_filter(
                filtered,
                constraints.growth_score_min
            )

        # 5. Quality filter (safety net)
        if constraints.quality_score_min is not None:
            filtered = self._apply_quality_filter(
                filtered,
                constraints.quality_score_min
            )

        # 6. Sentiment/Supply-demand filter (4단계)
        filtered = self._apply_sentiment_filter(
            filtered,
            risk_level
        )

        logger.info(f"Risk filtering complete: {len(filtered)} candidates remain")

        return RiskFilterSummary(
            total_before=total_before,
            total_after=len(filtered),
            filter_results=self.filter_results,
            candidates=filtered
        )

    def _apply_volatility_filter(
        self,
        candidates: List[StockCandidate],
        max_volatility: float
    ) -> List[StockCandidate]:
        """
        Filter by maximum annual volatility.

        Args:
            candidates: List of candidates
            max_volatility: Maximum allowed volatility (%)

        Returns:
            Filtered list
        """
        before_count = len(candidates)

        filtered = []
        removed = []

        for c in candidates:
            # Allow if volatility is None (data missing) or within limit
            if c.volatility_annual is None or c.volatility_annual <= max_volatility:
                filtered.append(c)
            else:
                removed.append(c.symbol)

        self._log_filter_result(
            'Volatility',
            before_count,
            len(filtered),
            removed,
            f"<= {max_volatility}%"
        )

        return filtered

    def _apply_mdd_filter(
        self,
        candidates: List[StockCandidate],
        max_mdd: float
    ) -> List[StockCandidate]:
        """
        Filter by maximum drawdown.

        Args:
            candidates: List of candidates
            max_mdd: Maximum allowed MDD (negative %, e.g., -20)

        Returns:
            Filtered list
        """
        before_count = len(candidates)

        filtered = []
        removed = []

        for c in candidates:
            # MDD is negative, so we check if it's >= limit (less negative)
            # e.g., -15% >= -20% means it passes
            if c.max_drawdown_1y is None or c.max_drawdown_1y >= max_mdd:
                filtered.append(c)
            else:
                removed.append(c.symbol)

        self._log_filter_result(
            'MDD',
            before_count,
            len(filtered),
            removed,
            f">= {max_mdd}%"
        )

        return filtered

    def _apply_beta_filter(
        self,
        candidates: List[StockCandidate],
        beta_min: Optional[float],
        beta_max: Optional[float]
    ) -> List[StockCandidate]:
        """
        Filter by beta range.

        Args:
            candidates: List of candidates
            beta_min: Minimum beta (or None)
            beta_max: Maximum beta (or None)

        Returns:
            Filtered list
        """
        before_count = len(candidates)

        filtered = []
        removed = []

        for c in candidates:
            # Allow if beta is None (data missing)
            if c.beta is None:
                filtered.append(c)
                continue

            passes = True

            if beta_min is not None and c.beta < beta_min:
                passes = False

            if beta_max is not None and c.beta > beta_max:
                passes = False

            if passes:
                filtered.append(c)
            else:
                removed.append(c.symbol)

        range_str = f"{beta_min or '*'} ~ {beta_max or '*'}"
        self._log_filter_result(
            'Beta',
            before_count,
            len(filtered),
            removed,
            range_str
        )

        return filtered

    def _apply_growth_filter(
        self,
        candidates: List[StockCandidate],
        min_growth: float
    ) -> List[StockCandidate]:
        """
        Filter by minimum growth score.

        Args:
            candidates: List of candidates
            min_growth: Minimum growth score

        Returns:
            Filtered list
        """
        before_count = len(candidates)

        filtered = []
        removed = []

        for c in candidates:
            # Require growth score for aggressive strategy
            if c.growth_score is not None and c.growth_score >= min_growth:
                filtered.append(c)
            else:
                removed.append(c.symbol)

        self._log_filter_result(
            'Growth Score',
            before_count,
            len(filtered),
            removed,
            f">= {min_growth}"
        )

        return filtered

    def _apply_quality_filter(
        self,
        candidates: List[StockCandidate],
        min_quality: float
    ) -> List[StockCandidate]:
        """
        Filter by minimum quality score (safety net).

        Args:
            candidates: List of candidates
            min_quality: Minimum quality score

        Returns:
            Filtered list
        """
        before_count = len(candidates)

        filtered = []
        removed = []

        for c in candidates:
            # Allow if quality score is None or meets minimum
            if c.quality_score is None or c.quality_score >= min_quality:
                filtered.append(c)
            else:
                removed.append(c.symbol)

        self._log_filter_result(
            'Quality Score (Safety)',
            before_count,
            len(filtered),
            removed,
            f">= {min_quality}"
        )

        return filtered

    def _apply_sentiment_filter(
        self,
        candidates: List[StockCandidate],
        risk_level: RiskLevel
    ) -> List[StockCandidate]:
        """
        Apply sentiment/supply-demand filters based on risk level.

        KR stocks: foreign_net_30d + inst_net_30d (외국인/기관 순매수)
        US stocks: analyst_rating, insider_signal

        Args:
            candidates: List of candidates
            risk_level: Risk level for filter intensity

        Returns:
            Filtered list
        """
        before_count = len(candidates)

        config = SENTIMENT_FILTER_CONFIG.get(
            risk_level.value,
            SENTIMENT_FILTER_CONFIG['balanced']
        )

        filtered = []
        removed = []

        # KR과 US를 분리하여 처리
        kr_candidates = [c for c in candidates if c.country == Country.KR]
        us_candidates = [c for c in candidates if c.country == Country.US]

        # KR 필터링
        kr_filtered, kr_removed = self._filter_kr_sentiment(kr_candidates, config)
        filtered.extend(kr_filtered)
        removed.extend(kr_removed)

        # US 필터링
        us_filtered, us_removed = self._filter_us_sentiment(us_candidates, config)
        filtered.extend(us_filtered)
        removed.extend(us_removed)

        filter_desc = f"KR: net>={config.get('kr_min_combined_net', 'N/A')}, " \
                      f"US: rating<={config.get('us_max_analyst_rating', 'N/A')}"

        self._log_filter_result(
            'Sentiment',
            before_count,
            len(filtered),
            removed,
            filter_desc
        )

        return filtered

    def _filter_kr_sentiment(
        self,
        candidates: List[StockCandidate],
        config: Dict
    ) -> Tuple[List[StockCandidate], List[str]]:
        """Filter KR stocks by sentiment (foreign + institutional flow)."""
        filtered = []
        removed = []

        min_combined = config.get('kr_min_combined_net')
        require_positive = config.get('kr_require_positive_flow', False)

        for c in candidates:
            # 데이터 없으면 통과 (soft filter)
            foreign_net = c.foreign_net_30d or 0
            inst_net = c.inst_net_30d or 0
            combined = foreign_net + inst_net

            passes = True

            if require_positive and combined < 0:
                passes = False

            if min_combined is not None and combined < min_combined:
                passes = False

            if passes:
                filtered.append(c)
            else:
                removed.append(c.symbol)

        return filtered, removed

    def _filter_us_sentiment(
        self,
        candidates: List[StockCandidate],
        config: Dict
    ) -> Tuple[List[StockCandidate], List[str]]:
        """Filter US stocks by sentiment (analyst rating, insider signal)."""
        filtered = []
        removed = []

        max_rating = config.get('us_max_analyst_rating')
        prefer_positive_insider = config.get('us_prefer_positive_insider', False)

        for c in candidates:
            passes = True

            # 애널리스트 등급 필터 (1=Best, 5=Worst)
            if max_rating is not None and c.analyst_rating is not None:
                if c.analyst_rating > max_rating:
                    passes = False

            # 내부자 신호 필터 (선호 모드만)
            if prefer_positive_insider:
                if c.insider_signal is not None and c.insider_signal < 0:
                    passes = False

            if passes:
                filtered.append(c)
            else:
                removed.append(c.symbol)

        return filtered, removed

    def _log_filter_result(
        self,
        filter_name: str,
        before: int,
        after: int,
        removed: List[str],
        condition: str
    ) -> None:
        """Log and store filter result"""
        removed_count = before - after

        result = FilterResult(
            filter_name=filter_name,
            before_count=before,
            after_count=after,
            removed_count=removed_count,
            removed_symbols=removed[:10]  # Store max 10 for reference
        )
        self.filter_results.append(result)

        if removed_count > 0:
            logger.info(f"  {filter_name} ({condition}): {before} -> {after} (-{removed_count})")
        else:
            logger.info(f"  {filter_name} ({condition}): {before} -> {after} (no change)")


def filter_by_risk(
    candidates: List[StockCandidate],
    processed_input: ProcessedInput
) -> RiskFilterSummary:
    """
    Convenience function to apply risk filters.

    Args:
        candidates: List of StockCandidate from universe filter
        processed_input: Processed input with constraints

    Returns:
        RiskFilterSummary with filtered candidates
    """
    risk_filter = RiskFilter()
    return risk_filter.filter_by_risk(candidates, processed_input)


# ============================================================================
# Test
# ============================================================================

def test_risk_filter():
    """Test risk filtering with mock data"""
    print("=" * 70)
    print("Risk Filter - Test")
    print("=" * 70)

    # Create mock candidates
    mock_candidates = [
        StockCandidate(
            symbol='005930', stock_name='Samsung', country=Country.KR,
            sector='Electronics', final_score=75.0,
            volatility_annual=25.0, max_drawdown_1y=-15.0, beta=1.0,
            quality_score=70.0, growth_score=55.0
        ),
        StockCandidate(
            symbol='000660', stock_name='SK Hynix', country=Country.KR,
            sector='Semiconductors', final_score=70.0,
            volatility_annual=45.0, max_drawdown_1y=-25.0, beta=1.3,
            quality_score=65.0, growth_score=60.0
        ),
        StockCandidate(
            symbol='035720', stock_name='Kakao', country=Country.KR,
            sector='IT', final_score=68.0,
            volatility_annual=55.0, max_drawdown_1y=-35.0, beta=1.1,
            quality_score=60.0, growth_score=70.0
        ),
        StockCandidate(
            symbol='TEST01', stock_name='HighVol Stock', country=Country.KR,
            sector='Bio', final_score=72.0,
            volatility_annual=80.0, max_drawdown_1y=-45.0, beta=1.8,
            quality_score=45.0, growth_score=80.0
        ),
        StockCandidate(
            symbol='TEST02', stock_name='LowQuality Stock', country=Country.KR,
            sector='Etc', final_score=65.0,
            volatility_annual=35.0, max_drawdown_1y=-18.0, beta=0.8,
            quality_score=25.0, growth_score=40.0
        ),
    ]

    from portfolio_input import process_portfolio_request

    # Test 1: Conservative
    print("\n[Test 1] Conservative Risk Filter")
    processed = process_portfolio_request(
        budget=10_000_000,
        country='KR',
        risk_level='conservative',
        num_stocks=10
    )

    result = filter_by_risk(mock_candidates, processed)

    print(f"  Before: {result.total_before}, After: {result.total_after}")
    print(f"  Filters applied:")
    for fr in result.filter_results:
        print(f"    - {fr.filter_name}: {fr.before_count} -> {fr.after_count}")
    print(f"  Remaining: {[c.symbol for c in result.candidates]}")

    # Test 2: Balanced
    print("\n[Test 2] Balanced Risk Filter")
    processed2 = process_portfolio_request(
        budget=10_000_000,
        country='KR',
        risk_level='balanced',
        num_stocks=10
    )

    result2 = filter_by_risk(mock_candidates, processed2)

    print(f"  Before: {result2.total_before}, After: {result2.total_after}")
    print(f"  Remaining: {[c.symbol for c in result2.candidates]}")

    # Test 3: Aggressive
    print("\n[Test 3] Aggressive Risk Filter")
    processed3 = process_portfolio_request(
        budget=10_000_000,
        country='KR',
        risk_level='aggressive',
        num_stocks=10
    )

    result3 = filter_by_risk(mock_candidates, processed3)

    print(f"  Before: {result3.total_before}, After: {result3.total_after}")
    print(f"  Filters applied:")
    for fr in result3.filter_results:
        print(f"    - {fr.filter_name}: {fr.before_count} -> {fr.after_count}")
    print(f"  Remaining: {[c.symbol for c in result3.candidates]}")

    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)


if __name__ == '__main__':
    test_risk_filter()
