# -*- coding: utf-8 -*-
"""
Portfolio Models

Pydantic models for alpha portfolio generation.
Designed for FastAPI integration and admin GUI.

File: create_portfolio/models.py
Created: 2025-12-24
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import date, datetime
from enum import Enum


# ============================================================================
# 1. Enums
# ============================================================================

class Country(str, Enum):
    KR = "KR"
    US = "US"
    MIXED = "MIXED"


class RiskLevel(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class RiskFlag(str, Enum):
    NORMAL = "NORMAL"
    MODERATE_RISK = "MODERATE_RISK"
    HIGH_RISK = "HIGH_RISK"
    EXTREME_RISK = "EXTREME_RISK"


# ============================================================================
# 2. Input Models (Request)
# ============================================================================

class PortfolioRequest(BaseModel):
    """Portfolio generation request from admin GUI"""

    budget: int = Field(
        ...,
        gt=0,
        description="Investment budget in KRW (or USD equivalent for US)"
    )
    country: Country = Field(
        default=Country.KR,
        description="Target market: KR, US, or MIXED"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.BALANCED,
        description="Risk level: conservative, balanced, aggressive"
    )
    num_stocks: int = Field(
        default=10,
        ge=5,
        le=20,
        description="Number of stocks in portfolio (5-20)"
    )
    analysis_date: Optional[date] = Field(
        default=None,
        description="Analysis date (None = latest available)"
    )
    portfolio_name: Optional[str] = Field(
        default=None,
        description="Optional portfolio name for identification"
    )
    benchmark: Optional[str] = Field(
        default=None,
        description="Benchmark index (e.g., KOSPI200, S&P500)"
    )
    max_weight_per_stock: Optional[float] = Field(
        default=None,
        ge=0.05,
        le=0.50,
        description="Max weight per stock (overrides risk level default)"
    )
    max_weight_per_sector: Optional[float] = Field(
        default=None,
        ge=0.10,
        le=0.80,
        description="Max weight per sector (overrides risk level default)"
    )
    min_consecutive_buy_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=30,
        description="Min consecutive buy grade days"
    )
    rebalancing_frequency: Optional[str] = Field(
        default=None,
        description="Rebalancing frequency (WEEKLY, MONTHLY, QUARTERLY, NONE)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "budget": 10000000,
                "country": "KR",
                "risk_level": "balanced",
                "num_stocks": 10,
                "analysis_date": None,
                "portfolio_name": "2024-12 Balanced Portfolio"
            }
        }


# ============================================================================
# 3. Constraint Models
# ============================================================================

class FilterConstraints(BaseModel):
    """Stock filtering constraints based on risk level"""

    volatility_max: Optional[float] = Field(
        default=None,
        description="Maximum annual volatility (%)"
    )
    mdd_max: Optional[float] = Field(
        default=None,
        description="Maximum drawdown (%, negative value)"
    )
    beta_min: Optional[float] = Field(
        default=None,
        description="Minimum beta"
    )
    beta_max: Optional[float] = Field(
        default=None,
        description="Maximum beta"
    )
    growth_score_min: Optional[float] = Field(
        default=None,
        description="Minimum growth score (for aggressive)"
    )
    quality_score_min: Optional[float] = Field(
        default=None,
        description="Minimum quality score (safety filter)"
    )


class WeightConstraints(BaseModel):
    """Portfolio weight allocation constraints"""

    max_weight_per_stock: float = Field(
        ...,
        gt=0,
        le=1,
        description="Maximum weight per single stock (0-1)"
    )
    max_weight_per_sector: float = Field(
        ...,
        gt=0,
        le=1,
        description="Maximum weight per sector (0-1)"
    )
    min_weight_per_stock: float = Field(
        default=0.03,
        gt=0,
        le=1,
        description="Minimum weight per stock (0-1)"
    )


class UniverseFilter(BaseModel):
    """Universe filtering parameters"""

    min_trading_value: float = Field(
        ...,
        description="Minimum daily trading value"
    )
    min_confidence_score: float = Field(
        default=50,
        description="Minimum confidence score"
    )
    allowed_grades: List[str] = Field(
        ...,
        description="Allowed final grades"
    )
    exclude_risk_flags: List[str] = Field(
        default=["EXTREME_RISK"],
        description="Risk flags to exclude"
    )
    lookback_days: int = Field(
        default=20,
        description="Days for trading value average"
    )
    max_gap_down: Optional[float] = Field(
        default=None,
        description="Maximum gap-down (US only)"
    )


class PortfolioConstraints(BaseModel):
    """Combined constraints for portfolio generation"""

    filter_constraints: FilterConstraints
    weight_constraints: WeightConstraints
    universe_filter: UniverseFilter


# ============================================================================
# 4. Validation Result Models
# ============================================================================

class ValidationError(BaseModel):
    """Single validation error"""

    field: str
    message: str
    code: str


class ValidationResult(BaseModel):
    """Input validation result"""

    is_valid: bool
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def add_error(self, field: str, message: str, code: str):
        self.errors.append(ValidationError(field=field, message=message, code=code))
        self.is_valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)


# ============================================================================
# 5. Processing Result Models
# ============================================================================

class ProcessedInput(BaseModel):
    """Processed and validated input ready for portfolio generation"""

    # Original request
    request: PortfolioRequest

    # Derived values
    constraints: PortfolioConstraints
    analysis_date: date
    budget_per_country: Dict[str, int] = Field(
        default_factory=dict,
        description="Budget allocation per country (for MIXED)"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    validation: ValidationResult


# ============================================================================
# 6. Stock Selection Models (for later use)
# ============================================================================

class StockCandidate(BaseModel):
    """Stock candidate for portfolio inclusion"""

    symbol: str
    stock_name: str
    country: Country
    sector: Optional[str] = None

    # Scores
    final_score: float
    value_score: Optional[float] = None
    quality_score: Optional[float] = None
    momentum_score: Optional[float] = None
    growth_score: Optional[float] = None
    conviction_score: Optional[float] = None

    # Risk metrics
    volatility_annual: Optional[float] = None
    max_drawdown_1y: Optional[float] = None
    beta: Optional[float] = None
    outlier_risk_score: Optional[float] = None
    risk_flag: Optional[str] = None

    # Selection metrics
    selection_score: Optional[float] = None  # Normalized score for ranking
    sector_rotation_score: Optional[float] = None

    # Sentiment/Supply-demand metrics (4단계)
    # KR only
    foreign_net_30d: Optional[float] = None  # 외국인 30일 순매수
    inst_net_30d: Optional[float] = None     # 기관 30일 순매수
    # US only
    insider_signal: Optional[float] = None   # 내부자 거래 신호
    analyst_rating: Optional[float] = None   # 애널리스트 평균 등급


class PortfolioStock(BaseModel):
    """Stock included in final portfolio"""

    symbol: str
    stock_name: str
    country: Country
    sector: Optional[str] = None

    # Allocation
    weight: float = Field(..., description="Portfolio weight (0-1)")
    shares: int = Field(..., description="Number of shares")
    amount: float = Field(..., description="Investment amount")

    # Current price info
    current_price: float
    currency: str

    # Scores (carried over)
    final_score: float
    conviction_score: Optional[float] = None

    # Risk metrics (carried over)
    volatility_annual: Optional[float] = None
    beta: Optional[float] = None

    # XAI explanation
    selection_reasons: List[str] = Field(default_factory=list)


class PortfolioSummary(BaseModel):
    """Portfolio summary statistics"""

    total_stocks: int
    total_investment: float
    currency: str

    # Weighted averages
    avg_final_score: float
    avg_volatility: Optional[float] = None
    avg_beta: Optional[float] = None

    # Sector distribution
    sector_weights: Dict[str, float] = Field(default_factory=dict)

    # Risk distribution
    risk_flag_counts: Dict[str, int] = Field(default_factory=dict)


# ============================================================================
# 7. API Response Models
# ============================================================================

class PortfolioResponse(BaseModel):
    """Complete portfolio generation response"""

    success: bool
    message: str

    # Portfolio data
    portfolio_id: Optional[str] = None
    portfolio_name: Optional[str] = None
    created_at: Optional[datetime] = None

    # Request echo
    request: Optional[PortfolioRequest] = None

    # Results
    summary: Optional[PortfolioSummary] = None
    stocks: List[PortfolioStock] = Field(default_factory=list)

    # Validation info
    validation: Optional[ValidationResult] = None

    # Error details (if failed)
    error_code: Optional[str] = None
    error_detail: Optional[str] = None
