# -*- coding: utf-8 -*-
"""
Alert Generator

Generates alerts for portfolio monitoring and rebalancing.

File: portfolio/rebalancing/alert_generator.py
Created: 2025-12-29
"""

import logging
import uuid
from datetime import date, datetime
from typing import List, Dict, Any, Optional

from rebalancing.models import (
    Alert,
    AlertType,
    AlertSeverity,
    TriggerCheckResult,
    TriggerCondition,
    TriggerType,
    HoldingStatus,
)
from rebalancing.config import (
    ALERT_CONFIG,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MDD_LIMIT,
)
from rebalancing.rebalancing_db import RebalancingDBManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertGenerator:
    """
    Generates and manages portfolio alerts.

    Alert types:
    - GRADE_CHANGE: Quant grade changed 2+ levels
    - STOP_LOSS: Stop loss threshold reached
    - TAKE_PROFIT: Take profit threshold reached
    - REBALANCING_DUE: Scheduled rebalancing due
    - WEIGHT_DRIFT: Weight drift over threshold
    - MDD_WARNING: MDD limit approaching
    - SUSPENDED: Trading halt detected
    """

    def __init__(self, db: RebalancingDBManager):
        """
        Initialize alert generator.

        Args:
            db: RebalancingDBManager instance
        """
        self.db = db

    def generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        return f"ALT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"

    async def generate_alerts_from_triggers(
        self,
        portfolio_id: str,
        trigger_result: TriggerCheckResult,
        save_to_db: bool = True
    ) -> List[Alert]:
        """
        Generate alerts from trigger check result.

        Args:
            portfolio_id: Portfolio ID
            trigger_result: Trigger check result
            save_to_db: Save alerts to database

        Returns:
            List of generated alerts
        """
        alerts = []

        for condition in trigger_result.triggered_conditions:
            alert = self._create_alert_from_trigger(portfolio_id, condition)
            if alert:
                alerts.append(alert)

                if save_to_db:
                    await self.db.save_alert(
                        alert_id=alert.alert_id,
                        portfolio_id=alert.portfolio_id,
                        alert_type=alert.alert_type.value,
                        severity=alert.severity.value,
                        title=alert.title,
                        message=alert.message,
                        symbol=alert.symbol,
                        data=alert.data
                    )

        return alerts

    def _create_alert_from_trigger(
        self,
        portfolio_id: str,
        condition: TriggerCondition
    ) -> Optional[Alert]:
        """Create alert from trigger condition."""
        alert_id = self.generate_alert_id()

        if condition.trigger_type == TriggerType.GRADE_DROP:
            return self._create_grade_change_alert(
                alert_id, portfolio_id, condition
            )
        elif condition.trigger_type == TriggerType.STOP_LOSS:
            return self._create_stop_loss_alert(
                alert_id, portfolio_id, condition
            )
        elif condition.trigger_type == TriggerType.TAKE_PROFIT:
            return self._create_take_profit_alert(
                alert_id, portfolio_id, condition
            )
        elif condition.trigger_type == TriggerType.MDD_LIMIT:
            return self._create_mdd_warning_alert(
                alert_id, portfolio_id, condition
            )
        elif condition.trigger_type == TriggerType.SUSPENDED:
            return self._create_suspended_alert(
                alert_id, portfolio_id, condition
            )
        elif condition.trigger_type == TriggerType.MARKET_CRASH:
            return self._create_market_crash_alert(
                alert_id, portfolio_id, condition
            )

        return None

    def _create_grade_change_alert(
        self,
        alert_id: str,
        portfolio_id: str,
        condition: TriggerCondition
    ) -> Alert:
        """Create grade change alert."""
        affected = condition.affected_symbols
        details = condition.details.get("dropped_stocks", {})

        if len(affected) == 1:
            symbol = affected[0]
            info = details.get(symbol, {})
            title = f"Grade Drop: {symbol}"
            message = (
                f"{symbol} grade dropped from {info.get('entry_grade', 'N/A')} "
                f"to {info.get('current_grade', 'N/A')} "
                f"({info.get('grade_change', 0)} levels)"
            )
        else:
            title = f"Grade Drop: {len(affected)} stocks"
            message = f"Multiple stocks experienced grade drops: {', '.join(affected)}"

        return Alert(
            alert_id=alert_id,
            portfolio_id=portfolio_id,
            alert_type=AlertType.GRADE_CHANGE,
            severity=AlertSeverity.WARNING,
            title=title,
            message=message,
            symbol=affected[0] if len(affected) == 1 else None,
            data={"affected_stocks": details}
        )

    def _create_stop_loss_alert(
        self,
        alert_id: str,
        portfolio_id: str,
        condition: TriggerCondition
    ) -> Alert:
        """Create stop loss alert."""
        affected = condition.affected_symbols
        details = condition.details.get("stocks", {})
        threshold = condition.details.get("stop_loss_threshold", -12)

        if len(affected) == 1:
            symbol = affected[0]
            info = details.get(symbol, {})
            title = f"Stop Loss: {symbol}"
            message = (
                f"{symbol} reached stop loss threshold "
                f"({info.get('pnl_pct', 0):.1f}% <= {threshold}%)"
            )
        else:
            title = f"Stop Loss: {len(affected)} stocks"
            message = f"Multiple stocks hit stop loss: {', '.join(affected)}"

        return Alert(
            alert_id=alert_id,
            portfolio_id=portfolio_id,
            alert_type=AlertType.STOP_LOSS,
            severity=AlertSeverity.CRITICAL,
            title=title,
            message=message,
            symbol=affected[0] if len(affected) == 1 else None,
            data={"threshold": threshold, "stocks": details}
        )

    def _create_take_profit_alert(
        self,
        alert_id: str,
        portfolio_id: str,
        condition: TriggerCondition
    ) -> Alert:
        """Create take profit alert."""
        affected = condition.affected_symbols
        details = condition.details.get("stocks", {})
        threshold = condition.details.get("take_profit_threshold", 25)

        if len(affected) == 1:
            symbol = affected[0]
            info = details.get(symbol, {})
            title = f"Take Profit: {symbol}"
            message = (
                f"{symbol} reached take profit threshold "
                f"({info.get('pnl_pct', 0):.1f}% >= {threshold}%)"
            )
        else:
            title = f"Take Profit: {len(affected)} stocks"
            message = f"Multiple stocks hit take profit: {', '.join(affected)}"

        return Alert(
            alert_id=alert_id,
            portfolio_id=portfolio_id,
            alert_type=AlertType.TAKE_PROFIT,
            severity=AlertSeverity.INFO,
            title=title,
            message=message,
            symbol=affected[0] if len(affected) == 1 else None,
            data={"threshold": threshold, "stocks": details}
        )

    def _create_mdd_warning_alert(
        self,
        alert_id: str,
        portfolio_id: str,
        condition: TriggerCondition
    ) -> Alert:
        """Create MDD warning alert."""
        details = condition.details
        current_dd = details.get("current_drawdown", 0)
        threshold = details.get("threshold", -15)

        return Alert(
            alert_id=alert_id,
            portfolio_id=portfolio_id,
            alert_type=AlertType.MDD_WARNING,
            severity=AlertSeverity.CRITICAL,
            title="MDD Limit Reached",
            message=(
                f"Portfolio drawdown {current_dd:.1f}% "
                f"reached limit of {threshold}%"
            ),
            data=details
        )

    def _create_suspended_alert(
        self,
        alert_id: str,
        portfolio_id: str,
        condition: TriggerCondition
    ) -> Alert:
        """Create trading suspended alert."""
        affected = condition.affected_symbols
        details = condition.details.get("suspended_stocks", {})

        if len(affected) == 1:
            symbol = affected[0]
            title = f"Trading Halted: {symbol}"
            message = f"{symbol} trading has been halted"
        else:
            title = f"Trading Halted: {len(affected)} stocks"
            message = f"Multiple stocks halted: {', '.join(affected)}"

        return Alert(
            alert_id=alert_id,
            portfolio_id=portfolio_id,
            alert_type=AlertType.SUSPENDED,
            severity=AlertSeverity.CRITICAL,
            title=title,
            message=message,
            symbol=affected[0] if len(affected) == 1 else None,
            data={"suspended_stocks": list(affected)}
        )

    def _create_market_crash_alert(
        self,
        alert_id: str,
        portfolio_id: str,
        condition: TriggerCondition
    ) -> Alert:
        """Create market crash alert."""
        details = condition.details
        benchmark = details.get("benchmark", "Market")
        daily_return = details.get("daily_return", 0)

        return Alert(
            alert_id=alert_id,
            portfolio_id=portfolio_id,
            alert_type=AlertType.GRADE_CHANGE,  # Using GRADE_CHANGE as proxy
            severity=AlertSeverity.CRITICAL,
            title=f"Market Crash: {benchmark}",
            message=(
                f"{benchmark} dropped {abs(daily_return):.1f}% today. "
                f"Portfolio review recommended."
            ),
            data=details
        )

    async def generate_rebalancing_due_alert(
        self,
        portfolio_id: str,
        portfolio_name: str,
        last_rebalancing_date: Optional[date],
        save_to_db: bool = True
    ) -> Alert:
        """
        Generate rebalancing due alert.

        Args:
            portfolio_id: Portfolio ID
            portfolio_name: Portfolio name
            last_rebalancing_date: Last rebalancing date
            save_to_db: Save to database

        Returns:
            Alert object
        """
        alert_id = self.generate_alert_id()

        if last_rebalancing_date:
            days_since = (date.today() - last_rebalancing_date).days
            message = (
                f"Portfolio '{portfolio_name}' is due for monthly rebalancing. "
                f"Last rebalancing: {last_rebalancing_date} ({days_since} days ago)"
            )
        else:
            message = (
                f"Portfolio '{portfolio_name}' is due for monthly rebalancing. "
                f"No previous rebalancing on record."
            )

        alert = Alert(
            alert_id=alert_id,
            portfolio_id=portfolio_id,
            alert_type=AlertType.REBALANCING_DUE,
            severity=AlertSeverity.INFO,
            title=f"Rebalancing Due: {portfolio_name}",
            message=message,
            data={"last_rebalancing_date": last_rebalancing_date.isoformat() if last_rebalancing_date else None}
        )

        if save_to_db:
            await self.db.save_alert(
                alert_id=alert.alert_id,
                portfolio_id=alert.portfolio_id,
                alert_type=alert.alert_type.value,
                severity=alert.severity.value,
                title=alert.title,
                message=alert.message,
                data=alert.data
            )

        return alert

    async def generate_weight_drift_alert(
        self,
        portfolio_id: str,
        holdings: List[HoldingStatus],
        target_weights: Dict[str, float],
        threshold: float = 0.10,
        save_to_db: bool = True
    ) -> Optional[Alert]:
        """
        Generate weight drift alert if any stock exceeds threshold.

        Args:
            portfolio_id: Portfolio ID
            holdings: Current holdings
            target_weights: Target weights by symbol
            threshold: Drift threshold (default 10%)
            save_to_db: Save to database

        Returns:
            Alert if drift detected, None otherwise
        """
        drifted = []

        for h in holdings:
            target = target_weights.get(h.symbol, h.current_weight)
            drift = abs(h.current_weight - target)

            if drift >= threshold:
                drifted.append({
                    "symbol": h.symbol,
                    "current_weight": h.current_weight,
                    "target_weight": target,
                    "drift": drift
                })

        if not drifted:
            return None

        alert_id = self.generate_alert_id()

        if len(drifted) == 1:
            d = drifted[0]
            title = f"Weight Drift: {d['symbol']}"
            message = (
                f"{d['symbol']} weight drifted {d['drift']:.1%} "
                f"(current: {d['current_weight']:.1%}, target: {d['target_weight']:.1%})"
            )
        else:
            title = f"Weight Drift: {len(drifted)} stocks"
            symbols = [d["symbol"] for d in drifted]
            message = f"Multiple stocks with weight drift: {', '.join(symbols)}"

        alert = Alert(
            alert_id=alert_id,
            portfolio_id=portfolio_id,
            alert_type=AlertType.WEIGHT_DRIFT,
            severity=AlertSeverity.WARNING,
            title=title,
            message=message,
            data={"drifted_stocks": drifted, "threshold": threshold}
        )

        if save_to_db:
            await self.db.save_alert(
                alert_id=alert.alert_id,
                portfolio_id=alert.portfolio_id,
                alert_type=alert.alert_type.value,
                severity=alert.severity.value,
                title=alert.title,
                message=alert.message,
                data=alert.data
            )

        return alert

    async def get_active_alerts(
        self,
        portfolio_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active (unread) alerts.

        Args:
            portfolio_id: Filter by portfolio (optional)

        Returns:
            List of alert dicts
        """
        return await self.db.get_unread_alerts(portfolio_id)

    def format_alert_message(self, alert: Alert) -> str:
        """
        Format alert for display.

        Args:
            alert: Alert object

        Returns:
            Formatted string
        """
        severity_icon = {
            AlertSeverity.INFO: "[INFO]",
            AlertSeverity.WARNING: "[WARNING]",
            AlertSeverity.CRITICAL: "[CRITICAL]"
        }

        icon = severity_icon.get(alert.severity, "[?]")
        timestamp = alert.created_at.strftime("%Y-%m-%d %H:%M")

        return f"{icon} {timestamp} - {alert.title}\n  {alert.message}"

    def format_alerts_summary(self, alerts: List[Alert]) -> str:
        """
        Format multiple alerts as summary.

        Args:
            alerts: List of alerts

        Returns:
            Formatted summary string
        """
        if not alerts:
            return "No active alerts"

        lines = [f"Active Alerts ({len(alerts)})", "=" * 40]

        critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        warning = [a for a in alerts if a.severity == AlertSeverity.WARNING]
        info = [a for a in alerts if a.severity == AlertSeverity.INFO]

        if critical:
            lines.append(f"\nCRITICAL ({len(critical)}):")
            for a in critical:
                lines.append(f"  - {a.title}")

        if warning:
            lines.append(f"\nWARNING ({len(warning)}):")
            for a in warning:
                lines.append(f"  - {a.title}")

        if info:
            lines.append(f"\nINFO ({len(info)}):")
            for a in info:
                lines.append(f"  - {a.title}")

        return "\n".join(lines)
