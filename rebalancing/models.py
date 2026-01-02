# -*- coding: utf-8 -*-
"""
Rebalancing Data Models

Pydantic models for portfolio rebalancing system.

File: portfolio/rebalancing/models.py
Created: 2025-12-29
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class RebalancingType(str, Enum):
    """Rebalancing execution type"""
    SCHEDULED = "SCHEDULED"    # Monthly scheduled
    TRIGGERED = "TRIGGERED"    # Trigger-based (emergency)
    MANUAL = "MANUAL"          # User manual execution


class TriggerType(str, Enum):
    """Trigger types for emergency rebalancing"""
    GRADE_DROP = "GRADE_DROP"          # 2+ grade level drop
    STOP_LOSS = "STOP_LOSS"            # Stop loss threshold reached (ATR-based)
    TRAILING_STOP = "TRAILING_STOP"    # Trailing stop triggered (peak - 3x ATR)
    TAKE_PROFIT = "TAKE_PROFIT"        # Take profit for scale-out
    SCALE_OUT = "SCALE_OUT"            # Partial profit taking
    MDD_LIMIT = "MDD_LIMIT"            # Portfolio MDD limit reached
    SUSPENDED = "SUSPENDED"            # Trading halt detected
    MARKET_CRASH = "MARKET_CRASH"      # Benchmark daily -5% or more
    WEIGHT_DRIFT = "WEIGHT_DRIFT"      # Weight drift over threshold


class RebalancingAction(str, Enum):
    """Actions for individual stocks"""
    BUY = "BUY"            # New entry
    SELL = "SELL"          # Full exit
    INCREASE = "INCREASE"  # Increase position
    DECREASE = "DECREASE"  # Decrease position
    HOLD = "HOLD"          # No action


class RebalancingStatus(str, Enum):
    """Rebalancing execution status"""
    PLANNED = "PLANNED"          # Analysis done, waiting for execution
    IN_PROGRESS = "IN_PROGRESS"  # Executing trades
    COMPLETED = "COMPLETED"      # All trades completed
    CANCELLED = "CANCELLED"      # User cancelled or conditions not met


class AlertType(str, Enum):
    """Alert types"""
    GRADE_CHANGE = "GRADE_CHANGE"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TAKE_PROFIT = "TAKE_PROFIT"
    SCALE_OUT = "SCALE_OUT"
    REBALANCING_DUE = "REBALANCING_DUE"
    WEIGHT_DRIFT = "WEIGHT_DRIFT"
    MDD_WARNING = "MDD_WARNING"
    SUSPENDED = "SUSPENDED"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ============================================================================
# Stock Evaluation Models
# ============================================================================

class HoldingStatus(BaseModel):
    """Current status of a holding stock"""
    symbol: str
    stock_name: Optional[str] = None
    sector: Optional[str] = None

    # Position info
    shares: int
    avg_price: float
    current_price: float
    invested_amount: float
    current_value: float
    current_weight: float

    # P&L
    unrealized_pnl: float
    unrealized_pnl_pct: float

    # Grade info
    entry_grade: Optional[str] = None
    current_grade: Optional[str] = None
    grade_change: int = 0  # Positive = improved, Negative = dropped
    consecutive_grade_days: int = 0

    # Holding period
    entry_date: Optional[date] = None
    holding_days: int = 0

    # ATR-based risk management
    atr_pct: Optional[float] = None
    dynamic_stop_pct: Optional[float] = None  # ATR-based stop loss
    dynamic_take_pct: Optional[float] = None  # ATR-based take profit
    peak_price: Optional[float] = None  # Highest price since entry
    peak_date: Optional[date] = None
    trailing_stop_price: Optional[float] = None
    drawdown_from_peak_pct: Optional[float] = None  # Current price vs peak
    scale_out_stage: int = 0  # 0=none, 1=1st, 2=2nd, 3=trailing only
    profit_protection_mode: bool = False

    # Flags
    is_tradable: bool = True
    is_suspended: bool = False

    # Recommended action
    recommended_action: RebalancingAction = RebalancingAction.HOLD
    action_reason: Optional[str] = None


class CandidateStock(BaseModel):
    """Candidate stock for new entry"""
    symbol: str
    stock_name: Optional[str] = None
    sector: Optional[str] = None

    # Grade info
    current_grade: str
    consecutive_buy_days: int
    final_score: float

    # Price info
    current_price: float
    volatility: Optional[float] = None
    beta: Optional[float] = None

    # Allocation
    suggested_weight: float = 0.0
    suggested_shares: int = 0
    suggested_amount: float = 0.0


# ============================================================================
# Trigger Models
# ============================================================================

class TriggerCondition(BaseModel):
    """Trigger condition check result"""
    trigger_type: TriggerType
    is_triggered: bool
    severity: AlertSeverity
    details: Dict[str, Any] = {}
    affected_symbols: List[str] = []


class TriggerCheckResult(BaseModel):
    """Result of trigger check for a portfolio"""
    portfolio_id: str
    check_date: date
    needs_rebalancing: bool
    triggered_conditions: List[TriggerCondition] = []
    summary: str = ""


# ============================================================================
# Analysis Models
# ============================================================================

class SellCandidate(BaseModel):
    """Stock recommended for selling"""
    symbol: str
    stock_name: Optional[str] = None
    action: RebalancingAction  # SELL or DECREASE
    shares_to_sell: int
    current_shares: int
    sell_ratio: float  # 0.0 ~ 1.0
    expected_amount: float
    reason: str
    priority: int = 0  # Higher = more urgent


class BuyCandidate(BaseModel):
    """Stock recommended for buying"""
    symbol: str
    stock_name: Optional[str] = None
    action: RebalancingAction  # BUY or INCREASE
    shares_to_buy: int
    current_shares: int = 0
    expected_amount: float
    target_weight: float
    reason: str
    priority: int = 0  # Higher = more preferred


class AnalysisResult(BaseModel):
    """Result of rebalancing analysis"""
    portfolio_id: str
    analysis_date: date

    # Current status
    total_value: float
    cash_balance: float
    holdings_count: int

    # Recommendations
    sell_candidates: List[SellCandidate] = []
    buy_candidates: List[BuyCandidate] = []

    # Summary
    total_sell_amount: float = 0.0
    total_buy_amount: float = 0.0
    net_cashflow: float = 0.0

    # Flags
    has_actions: bool = False


# ============================================================================
# Cost Models
# ============================================================================

class TransactionCost(BaseModel):
    """Transaction cost breakdown"""
    symbol: str
    action: str  # BUY or SELL
    amount: float
    commission: float
    tax: float = 0.0
    exchange_fee: float = 0.0
    slippage: float
    total_cost: float


class CostEfficiencyResult(BaseModel):
    """Cost efficiency verification result"""
    is_efficient: bool
    expected_improvement: float
    total_cost: float
    cost_ratio: float  # improvement / cost
    min_required_ratio: float
    details: Dict[str, Any] = {}


# ============================================================================
# Execution Models
# ============================================================================

class TradeOrder(BaseModel):
    """Individual trade order"""
    symbol: str
    stock_name: Optional[str] = None
    action: RebalancingAction
    shares: int
    expected_price: float
    expected_amount: float
    expected_cost: float
    reason: str

    # Execution result (filled after execution)
    executed: bool = False
    executed_price: Optional[float] = None
    executed_amount: Optional[float] = None
    executed_at: Optional[datetime] = None


class RebalancingPlan(BaseModel):
    """Complete rebalancing execution plan"""
    rebalancing_id: str
    portfolio_id: str
    rebalancing_type: RebalancingType
    plan_date: date

    # Trigger info
    trigger_type: Optional[TriggerType] = None
    trigger_details: Dict[str, Any] = {}

    # Orders
    sell_orders: List[TradeOrder] = []
    buy_orders: List[TradeOrder] = []

    # Summary
    total_sell_amount: float = 0.0
    total_buy_amount: float = 0.0
    total_fee: float = 0.0
    net_cashflow: float = 0.0

    # Cost efficiency
    expected_improvement: float = 0.0
    cost_efficiency_ratio: float = 0.0
    is_cost_efficient: bool = True

    # Status
    status: RebalancingStatus = RebalancingStatus.PLANNED
    created_at: datetime = Field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None


# ============================================================================
# Result Models
# ============================================================================

class RebalancingResult(BaseModel):
    """Final result of rebalancing execution"""
    success: bool
    rebalancing_id: str
    portfolio_id: str
    rebalancing_type: RebalancingType

    # Execution summary
    executed_sell_count: int = 0
    executed_buy_count: int = 0
    total_sell_amount: float = 0.0
    total_buy_amount: float = 0.0
    total_fee: float = 0.0

    # Portfolio after rebalancing
    new_holdings_count: int = 0
    new_total_value: float = 0.0
    new_cash_balance: float = 0.0

    # Timestamps
    started_at: datetime
    completed_at: datetime

    # Error info
    error_code: Optional[str] = None
    error_message: Optional[str] = None


# ============================================================================
# Alert Models
# ============================================================================

class Alert(BaseModel):
    """Alert notification"""
    alert_id: str
    portfolio_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    symbol: Optional[str] = None
    data: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    is_read: bool = False
    is_resolved: bool = False


# ============================================================================
# Request/Response Models (for API)
# ============================================================================

class RebalancingRequest(BaseModel):
    """Request to execute rebalancing"""
    portfolio_id: str
    rebalancing_type: RebalancingType = RebalancingType.MANUAL
    force_execute: bool = False  # Skip cost efficiency check
    dry_run: bool = False  # Analysis only, no execution


class RebalancingResponse(BaseModel):
    """Response from rebalancing execution"""
    success: bool
    message: str
    rebalancing_id: Optional[str] = None
    plan: Optional[RebalancingPlan] = None
    result: Optional[RebalancingResult] = None
    alerts: List[Alert] = []
