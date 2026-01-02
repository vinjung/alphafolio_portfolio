# -*- coding: utf-8 -*-
"""
Portfolio Input Processor

Validates and processes input for alpha portfolio generation.
Designed for admin GUI integration.

File: create_portfolio/portfolio_input.py
Created: 2025-12-24
"""

import logging
from datetime import date, datetime
from typing import Dict, Any, Optional, Tuple

from models import (
    PortfolioRequest,
    PortfolioConstraints,
    FilterConstraints,
    WeightConstraints,
    UniverseFilter,
    ValidationResult,
    ProcessedInput,
    Country,
    RiskLevel,
)
from config import (
    BUDGET_LIMITS,
    STOCK_COUNT_LIMITS,
    RISK_LEVEL_CONSTRAINTS,
    UNIVERSE_FILTER,
    WEIGHT_ALLOCATION,
    DEFAULTS,
    get_risk_constraints,
    get_universe_filter,
    get_budget_limits,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioInputProcessor:
    """
    Process and validate portfolio generation input.

    Usage:
        processor = PortfolioInputProcessor(request)
        result = processor.process()

        if result.validation.is_valid:
            # Use result.constraints, result.analysis_date, etc.
            pass
        else:
            # Handle errors
            for error in result.validation.errors:
                print(f"{error.field}: {error.message}")
    """

    def __init__(self, request: PortfolioRequest):
        """
        Initialize processor with request.

        Args:
            request: PortfolioRequest from admin GUI
        """
        self.request = request
        self.validation = ValidationResult(is_valid=True)

    def process(self) -> ProcessedInput:
        """
        Process and validate input, return ProcessedInput.

        Returns:
            ProcessedInput with constraints and validation result
        """
        logger.info(f"Processing portfolio request: {self.request.country.value}, "
                    f"{self.request.risk_level.value}, {self.request.num_stocks} stocks")

        # Step 1: Validate input
        self._validate_budget()
        self._validate_num_stocks()
        self._validate_country_risk_combination()

        # Step 2: Build constraints (even if validation failed, for error display)
        constraints = self._build_constraints()

        # Step 3: Determine analysis date
        analysis_date = self._determine_analysis_date()

        # Step 4: Calculate budget per country (for MIXED)
        budget_per_country = self._calculate_budget_allocation()

        # Step 5: Add warnings if applicable
        self._add_warnings()

        # Build result
        result = ProcessedInput(
            request=self.request,
            constraints=constraints,
            analysis_date=analysis_date,
            budget_per_country=budget_per_country,
            validation=self.validation
        )

        if self.validation.is_valid:
            logger.info("Input validation passed")
        else:
            logger.warning(f"Input validation failed with {len(self.validation.errors)} errors")

        return result

    def _validate_budget(self) -> None:
        """Validate budget against country limits"""
        country = self.request.country.value
        budget = self.request.budget
        limits = get_budget_limits(country)

        if budget < limits['min']:
            self.validation.add_error(
                field='budget',
                message=f"Budget must be at least {limits['min']:,} ({limits['display_unit']})",
                code='BUDGET_TOO_LOW'
            )

        if budget > limits['max']:
            self.validation.add_error(
                field='budget',
                message=f"Budget cannot exceed {limits['max']:,} ({limits['display_unit']})",
                code='BUDGET_TOO_HIGH'
            )

    def _validate_num_stocks(self) -> None:
        """Validate number of stocks"""
        num_stocks = self.request.num_stocks

        if num_stocks < STOCK_COUNT_LIMITS['min']:
            self.validation.add_error(
                field='num_stocks',
                message=f"Minimum {STOCK_COUNT_LIMITS['min']} stocks required",
                code='STOCKS_TOO_FEW'
            )

        if num_stocks > STOCK_COUNT_LIMITS['max']:
            self.validation.add_error(
                field='num_stocks',
                message=f"Maximum {STOCK_COUNT_LIMITS['max']} stocks allowed",
                code='STOCKS_TOO_MANY'
            )

        # Note: Budget-to-stock-count validation removed
        # Actual validation happens in portfolio_quantity.py based on real prices
        pass

    def _validate_country_risk_combination(self) -> None:
        """Validate country and risk level combination"""
        # Currently no restrictions, but placeholder for future rules
        # e.g., US aggressive might need higher budget

        if (self.request.country == Country.US and
            self.request.risk_level == RiskLevel.AGGRESSIVE and
            self.request.budget < 5_000_000):
            self.validation.add_warning(
                "US aggressive portfolio with low budget may have limited stock selection"
            )

    def _build_constraints(self) -> PortfolioConstraints:
        """Build PortfolioConstraints from request"""
        country = self.request.country.value
        risk_level = self.request.risk_level.value

        # Get risk-level constraints
        risk_config = get_risk_constraints(risk_level)

        # Build FilterConstraints
        filter_constraints = FilterConstraints(
            volatility_max=risk_config.get('volatility_max'),
            mdd_max=risk_config.get('mdd_max'),
            beta_min=risk_config.get('beta_min'),
            beta_max=risk_config.get('beta_max'),
            growth_score_min=risk_config.get('growth_score_min'),
            quality_score_min=risk_config.get('quality_score_min'),
        )

        # Build WeightConstraints (user override > risk level default)
        weight_constraints = WeightConstraints(
            max_weight_per_stock=(
                self.request.max_weight_per_stock
                if self.request.max_weight_per_stock is not None
                else risk_config.get('max_weight_per_stock')
            ),
            max_weight_per_sector=(
                self.request.max_weight_per_sector
                if self.request.max_weight_per_sector is not None
                else risk_config.get('max_weight_per_sector')
            ),
            min_weight_per_stock=WEIGHT_ALLOCATION.get('min_weight_per_stock', 0.03),
        )

        # Build UniverseFilter
        universe_config = get_universe_filter(country)

        # For MIXED, use the more restrictive filter (KR)
        if country == 'MIXED':
            universe_config = get_universe_filter('KR')

        universe_filter = UniverseFilter(
            min_trading_value=universe_config.get('min_trading_value'),
            min_confidence_score=universe_config.get('min_confidence_score'),
            allowed_grades=universe_config.get('min_final_grade', []),
            exclude_risk_flags=universe_config.get('exclude_risk_flags', ['EXTREME_RISK']),
            lookback_days=universe_config.get('lookback_days', 20),
            max_gap_down=universe_config.get('max_gap_down'),
        )

        return PortfolioConstraints(
            filter_constraints=filter_constraints,
            weight_constraints=weight_constraints,
            universe_filter=universe_filter,
        )

    def _determine_analysis_date(self) -> date:
        """Determine analysis date"""
        if self.request.analysis_date:
            return self.request.analysis_date

        # Default to today (actual data date will be determined by DB query)
        return date.today()

    def _calculate_budget_allocation(self) -> Dict[str, int]:
        """Calculate budget allocation per country"""
        budget = self.request.budget
        country = self.request.country

        if country == Country.KR:
            return {'KR': budget}
        elif country == Country.US:
            return {'US': budget}
        else:  # MIXED
            # Default: 60% KR, 40% US
            return {
                'KR': int(budget * 0.6),
                'US': int(budget * 0.4)
            }

    def _add_warnings(self) -> None:
        """Add warnings based on input analysis"""
        # Warning for small budget
        if self.request.budget < 3_000_000:
            self.validation.add_warning(
                "Small budget may result in concentrated positions due to lot size constraints"
            )

        # Warning for aggressive + many stocks
        if (self.request.risk_level == RiskLevel.AGGRESSIVE and
            self.request.num_stocks > 15):
            self.validation.add_warning(
                "Aggressive strategy with many stocks may dilute alpha potential"
            )

        # Warning for conservative + few stocks
        if (self.request.risk_level == RiskLevel.CONSERVATIVE and
            self.request.num_stocks < 8):
            self.validation.add_warning(
                "Conservative strategy with few stocks increases concentration risk"
            )


def process_portfolio_request(
    budget: int,
    country: str = 'KR',
    risk_level: str = 'balanced',
    num_stocks: int = 10,
    analysis_date: Optional[date] = None,
    portfolio_name: Optional[str] = None,
    benchmark: Optional[str] = None,
    max_weight_per_stock: Optional[float] = None,
    max_weight_per_sector: Optional[float] = None,
    min_consecutive_buy_days: Optional[int] = None,
    rebalancing_frequency: Optional[str] = None
) -> ProcessedInput:
    """
    Convenience function to process portfolio request.

    Args:
        budget: Investment budget
        country: 'KR', 'US', or 'MIXED'
        risk_level: 'conservative', 'balanced', or 'aggressive'
        num_stocks: Number of stocks (5-20)
        analysis_date: Optional analysis date
        portfolio_name: Optional portfolio name
        benchmark: Optional benchmark index
        max_weight_per_stock: Optional max weight per stock (overrides default)
        max_weight_per_sector: Optional max weight per sector (overrides default)
        min_consecutive_buy_days: Optional min consecutive buy days
        rebalancing_frequency: Optional rebalancing frequency

    Returns:
        ProcessedInput with validation and constraints
    """
    request = PortfolioRequest(
        budget=budget,
        country=Country(country),
        risk_level=RiskLevel(risk_level),
        num_stocks=num_stocks,
        analysis_date=analysis_date,
        portfolio_name=portfolio_name,
        benchmark=benchmark,
        max_weight_per_stock=max_weight_per_stock,
        max_weight_per_sector=max_weight_per_sector,
        min_consecutive_buy_days=min_consecutive_buy_days,
        rebalancing_frequency=rebalancing_frequency,
    )

    processor = PortfolioInputProcessor(request)
    return processor.process()


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Portfolio Input Processor - Test Cases")
    print("=" * 70)

    # Test 1: Valid KR balanced
    print("\n[Test 1] Valid KR Balanced")
    result1 = process_portfolio_request(
        budget=10_000_000,
        country='KR',
        risk_level='balanced',
        num_stocks=10
    )
    print(f"  Valid: {result1.validation.is_valid}")
    print(f"  Errors: {len(result1.validation.errors)}")
    print(f"  Warnings: {result1.validation.warnings}")
    print(f"  Volatility Max: {result1.constraints.filter_constraints.volatility_max}")
    print(f"  Max Weight/Stock: {result1.constraints.weight_constraints.max_weight_per_stock}")

    # Test 2: Budget too low
    print("\n[Test 2] Budget Too Low")
    result2 = process_portfolio_request(
        budget=100_000,  # 10만원 - too low
        country='KR',
        risk_level='balanced',
        num_stocks=10
    )
    print(f"  Valid: {result2.validation.is_valid}")
    for error in result2.validation.errors:
        print(f"  Error: [{error.code}] {error.field}: {error.message}")

    # Test 3: Too many stocks
    print("\n[Test 3] Too Many Stocks")
    result3 = process_portfolio_request(
        budget=10_000_000,
        country='KR',
        risk_level='balanced',
        num_stocks=25  # max is 20
    )
    print(f"  Valid: {result3.validation.is_valid}")
    for error in result3.validation.errors:
        print(f"  Error: [{error.code}] {error.field}: {error.message}")

    # Test 4: US Aggressive
    print("\n[Test 4] US Aggressive")
    result4 = process_portfolio_request(
        budget=50_000_000,
        country='US',
        risk_level='aggressive',
        num_stocks=8
    )
    print(f"  Valid: {result4.validation.is_valid}")
    print(f"  Growth Score Min: {result4.constraints.filter_constraints.growth_score_min}")
    print(f"  Volatility Max: {result4.constraints.filter_constraints.volatility_max}")
    print(f"  Max Weight/Stock: {result4.constraints.weight_constraints.max_weight_per_stock}")

    # Test 5: MIXED
    print("\n[Test 5] MIXED Portfolio")
    result5 = process_portfolio_request(
        budget=30_000_000,
        country='MIXED',
        risk_level='balanced',
        num_stocks=15
    )
    print(f"  Valid: {result5.validation.is_valid}")
    print(f"  Budget Allocation: {result5.budget_per_country}")

    # Test 6: Conservative with warnings
    print("\n[Test 6] Conservative Low Budget Few Stocks")
    result6 = process_portfolio_request(
        budget=2_000_000,
        country='KR',
        risk_level='conservative',
        num_stocks=5
    )
    print(f"  Valid: {result6.validation.is_valid}")
    print(f"  Warnings: {result6.validation.warnings}")
    print(f"  Beta Range: {result6.constraints.filter_constraints.beta_min} ~ {result6.constraints.filter_constraints.beta_max}")

    print("\n" + "=" * 70)
    print("All tests completed")
    print("=" * 70)
