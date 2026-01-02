# -*- coding: utf-8 -*-
"""
Rebalancing Configuration

Configuration constants for portfolio rebalancing system.

File: portfolio/rebalancing/config.py
Created: 2025-12-29
"""

from enum import Enum
from typing import Dict, Any


# ============================================================================
# Quant Grade System
# ============================================================================

class QuantGrade(Enum):
    """Quant grade levels (Korean/US common)"""
    STRONG_BUY = "강력 매수"
    BUY = "매수"
    BUY_CONSIDER = "매수 고려"
    NEUTRAL = "중립"
    SELL_CONSIDER = "매도 고려"
    SELL = "매도"


# Grade numeric mapping for comparison (higher = better)
GRADE_NUMERIC = {
    "강력 매수": 6,
    "매수": 5,
    "매수 고려": 4,
    "중립": 3,
    "매도 고려": 2,
    "매도": 1,
}

# Reverse mapping
NUMERIC_TO_GRADE = {v: k for k, v in GRADE_NUMERIC.items()}


# ============================================================================
# Rebalancing Trigger Conditions
# ============================================================================

# Grade drop trigger: N levels or more
GRADE_DROP_TRIGGER = 2  # 2 levels or more drop triggers review

# Stop loss / Take profit by risk level (%)
STOP_LOSS_PCT = {
    "conservative": -8.0,
    "balanced": -12.0,
    "aggressive": -15.0,
}

TAKE_PROFIT_PCT = {
    "conservative": 15.0,
    "balanced": 25.0,
    "aggressive": 40.0,
}

# Portfolio MDD limits by risk level (%)
MDD_LIMIT = {
    "conservative": -10.0,
    "balanced": -15.0,
    "aggressive": -20.0,
}

# Market crash trigger: benchmark daily return threshold (%)
MARKET_CRASH_THRESHOLD = -5.0

# Trading halt detection: days without data
TRADING_HALT_DAYS = 2

# Minimum holding period (days) - except emergency
MIN_HOLDING_DAYS = 30


# ============================================================================
# Transaction Costs
# ============================================================================

# Korean market costs (%)
KR_TRANSACTION_COST = {
    "buy_commission": 0.00015,      # 0.015%
    "sell_commission": 0.00015,     # 0.015%
    "sell_tax": 0.0018,             # 0.18% (securities transaction tax)
    "sell_agri_tax": 0.0015,        # 0.15% (agricultural special tax)
    "slippage_min": 0.001,          # 0.1%
    "slippage_max": 0.003,          # 0.3%
}

# US market costs (%)
US_TRANSACTION_COST = {
    "buy_commission": 0.0025,       # 0.25%
    "sell_commission": 0.0025,      # 0.25%
    "exchange_fee": 0.0025,         # 0.25% (currency exchange)
    "sec_fee": 0.0000278,           # SEC fee (per dollar)
    "slippage_min": 0.001,          # 0.1%
    "slippage_max": 0.003,          # 0.3%
}


def calculate_total_cost(country: str, action: str, amount: float) -> float:
    """
    Calculate total transaction cost.

    Args:
        country: 'KR' or 'US'
        action: 'BUY' or 'SELL'
        amount: Transaction amount

    Returns:
        Total cost including commission, tax, and slippage
    """
    if country == "KR":
        costs = KR_TRANSACTION_COST
        if action == "BUY":
            cost_rate = costs["buy_commission"] + costs["slippage_min"]
        else:  # SELL
            cost_rate = (costs["sell_commission"] + costs["sell_tax"] +
                        costs["sell_agri_tax"] + costs["slippage_min"])
    else:  # US
        costs = US_TRANSACTION_COST
        if action == "BUY":
            cost_rate = costs["buy_commission"] + costs["exchange_fee"] + costs["slippage_min"]
        else:  # SELL
            cost_rate = (costs["sell_commission"] + costs["exchange_fee"] +
                        costs["slippage_min"])
            # Add SEC fee
            return amount * cost_rate + amount * costs["sec_fee"]

    return amount * cost_rate


# ============================================================================
# Cost Efficiency Verification
# ============================================================================

# Minimum improvement ratio over cost (Expected Improvement > Cost x N)
COST_EFFICIENCY_MULTIPLIER = 2.0

# Annualization factor for expected return calculation
ANNUALIZATION_FACTOR = 12  # Monthly to annual


# ============================================================================
# Turnover Monitoring (Reference only, no limit)
# ============================================================================

TURNOVER_REFERENCE = {
    "conservative": 150,  # 150% annual
    "balanced": 200,      # 200% annual
    "aggressive": 250,    # 250% annual
}


# ============================================================================
# Weight Constraints (Same as portfolio creation)
# ============================================================================

MAX_WEIGHT_PER_STOCK = {
    "conservative": 0.12,  # 12%
    "balanced": 0.15,      # 15%
    "aggressive": 0.20,    # 20%
}

MAX_WEIGHT_PER_SECTOR = {
    "conservative": 0.25,  # 25%
    "balanced": 0.30,      # 30%
    "aggressive": 0.40,    # 40%
}

# Portfolio VaR limits by risk level (%)
VAR_LIMIT = {
    "conservative": 0.02,  # 2%
    "balanced": 0.05,      # 5%
    "aggressive": 0.07,    # 7%
}


# ============================================================================
# New Stock Entry Conditions
# ============================================================================

# Minimum consecutive buy grade days for entry
MIN_CONSECUTIVE_BUY_DAYS = 5

# Minimum grade for entry
MIN_ENTRY_GRADE = "매수 고려"


# ============================================================================
# Grade Action Matrix
# ============================================================================

def get_recommended_action(current_grade: str, entry_grade: str) -> str:
    """
    Get recommended action based on grade change.

    Args:
        current_grade: Current quant grade
        entry_grade: Grade at entry time

    Returns:
        Recommended action: HOLD, INCREASE, DECREASE, SELL, MONITOR
    """
    current_num = GRADE_NUMERIC.get(current_grade, 3)
    entry_num = GRADE_NUMERIC.get(entry_grade, 5)

    grade_change = current_num - entry_num

    # Grade improved
    if grade_change > 0:
        return "INCREASE"

    # Grade maintained
    elif grade_change == 0:
        return "HOLD"

    # 1 level drop
    elif grade_change == -1:
        return "MONITOR"

    # 2 levels drop
    elif grade_change == -2:
        return "DECREASE"

    # 3+ levels drop or SELL grade
    else:
        return "SELL"


# ============================================================================
# Alert Configuration
# ============================================================================

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


ALERT_CONFIG = {
    "GRADE_CHANGE": {
        "severity": AlertSeverity.WARNING,
        "threshold": 2,  # 2+ grade change
    },
    "STOP_LOSS": {
        "severity": AlertSeverity.CRITICAL,
    },
    "TAKE_PROFIT": {
        "severity": AlertSeverity.INFO,
    },
    "REBALANCING_DUE": {
        "severity": AlertSeverity.INFO,
    },
    "WEIGHT_DRIFT": {
        "severity": AlertSeverity.WARNING,
        "threshold": 0.10,  # 10% drift
    },
    "MDD_WARNING": {
        "severity": AlertSeverity.CRITICAL,
    },
    "SUSPENDED": {
        "severity": AlertSeverity.CRITICAL,
    },
}


# ============================================================================
# Database Tables
# ============================================================================

DB_TABLES = {
    # Input tables
    "portfolio_master": "portfolio_master",
    "portfolio_holdings": "portfolio_holdings",
    "kr_grade": "kr_stock_grade",
    "us_grade": "us_stock_grade",
    "kr_price": "kr_intraday_total",
    "us_price": "us_daily",
    "kr_indicators": "kr_indicators",
    "us_indicators": "us_indicators",
    "kr_trading": "kr_individual_investor_daily_trading",
    "exchange_rate": "exchange_rate",

    # Output tables
    "rebalancing": "portfolio_rebalancing",
    "rebalancing_detail": "portfolio_rebalancing_detail",
    "transactions": "portfolio_transactions",
    "alerts": "portfolio_alerts",
}
