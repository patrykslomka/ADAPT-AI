"""Alerting system for LLMOps."""
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    severity: str  # 'critical', 'warning', 'info'
    title: str
    message: str
    metric: str
    threshold: float
    current_value: float
    timestamp: str
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class AlertRule:
    """Rule for triggering alerts."""
    name: str
    metric: str
    condition: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    threshold: float
    severity: str
    message: str
    cooldown_seconds: int = 300  # Don't re-alert within this period


class AlertManager:
    """Manage alerts and notifications."""

    def __init__(self, alerts_file: Path = None):
        if alerts_file is None:
            alerts_file = Path("./logs/alerts.jsonl")
        self.alerts_file = alerts_file
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)

        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.active_alerts: List[Alert] = []
        self.last_alert_times: Dict[str, float] = {}

        # Define alert rules
        self.rules: List[AlertRule] = [
            AlertRule(
                name='high_cost',
                metric='total_cost',
                condition='gt',
                threshold=1.0,
                severity='warning',
                message='Query cost exceeded ${threshold}'
            ),
            AlertRule(
                name='high_latency',
                metric='response_time',
                condition='gt',
                threshold=10.0,
                severity='warning',
                message='Response time exceeded {threshold}s'
            ),
            AlertRule(
                name='compliance_failure',
                metric='compliance_passed',
                condition='eq',
                threshold=0.0,  # False = 0
                severity='critical',
                message='Compliance check failed'
            ),
            AlertRule(
                name='quality_failure',
                metric='quality_passed',
                condition='eq',
                threshold=0.0,
                severity='critical',
                message='Quality check failed'
            ),
            AlertRule(
                name='high_hallucination',
                metric='hallucination_score',
                condition='gt',
                threshold=0.3,
                severity='warning',
                message='High hallucination risk detected ({current_value:.0%})'
            ),
            AlertRule(
                name='low_confidence',
                metric='confidence_score',
                condition='lt',
                threshold=0.5,
                severity='warning',
                message='Low confidence score ({current_value:.0%})'
            ),
            AlertRule(
                name='error_rate_high',
                metric='error_rate',
                condition='gt',
                threshold=0.1,
                severity='critical',
                message='Error rate exceeded 10%'
            )
        ]

    def add_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def check_metrics(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Check metrics against alert rules."""
        alerts = []
        current_time = datetime.now().timestamp()

        for rule in self.rules:
            metric_value = metrics.get(rule.metric)

            if metric_value is None:
                continue

            # Check cooldown
            last_alert = self.last_alert_times.get(rule.name, 0)
            if current_time - last_alert < rule.cooldown_seconds:
                continue

            # Evaluate condition
            triggered = self._evaluate_condition(metric_value, rule.condition, rule.threshold)

            if triggered:
                alert = Alert(
                    alert_id=f"{rule.name}-{int(current_time)}",
                    severity=rule.severity,
                    title=rule.name.replace('_', ' ').title(),
                    message=rule.message.format(
                        threshold=rule.threshold,
                        current_value=metric_value
                    ),
                    metric=rule.metric,
                    threshold=rule.threshold,
                    current_value=float(metric_value) if isinstance(metric_value, (int, float, bool)) else 0,
                    timestamp=datetime.now().isoformat()
                )

                alerts.append(alert)
                self.active_alerts.append(alert)
                self.last_alert_times[rule.name] = current_time

                # Write to file
                self._write_alert(alert)

                logger.warning(f"Alert triggered: {alert.title} - {alert.message}")

                # Call handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler failed: {e}")

        return alerts

    def _evaluate_condition(self, value: Any, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        # Convert boolean to numeric
        if isinstance(value, bool):
            value = 1.0 if value else 0.0

        try:
            value = float(value)
        except (ValueError, TypeError):
            return False

        if condition == 'gt':
            return value > threshold
        elif condition == 'lt':
            return value < threshold
        elif condition == 'gte':
            return value >= threshold
        elif condition == 'lte':
            return value <= threshold
        elif condition == 'eq':
            return value == threshold
        else:
            return False

    def register_handler(self, handler: Callable[[Alert], None]):
        """Register alert handler."""
        self.alert_handlers.append(handler)
        logger.info(f"Registered alert handler: {handler.__name__}")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self.active_alerts.remove(alert)
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False

    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by severity."""
        if severity:
            return [a for a in self.active_alerts if a.severity == severity]
        return self.active_alerts.copy()

    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get alert history from file."""
        alerts = []
        try:
            with open(self.alerts_file, 'r') as f:
                for line in f:
                    try:
                        alerts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return []

        # Return most recent
        return alerts[-limit:]

    def _write_alert(self, alert: Alert):
        """Write alert to file."""
        try:
            with open(self.alerts_file, 'a') as f:
                f.write(json.dumps(asdict(alert)) + '\n')
        except Exception as e:
            logger.error(f"Failed to write alert: {e}")

    def get_alert_stats(self, hours: int = 24) -> Dict[str, int]:
        """Get alert statistics."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=hours)
        stats = {
            'total': 0,
            'critical': 0,
            'warning': 0,
            'info': 0,
            'acknowledged': 0,
            'resolved': 0
        }

        try:
            with open(self.alerts_file, 'r') as f:
                for line in f:
                    try:
                        alert = json.loads(line)
                        alert_time = datetime.fromisoformat(alert['timestamp'])
                        if alert_time >= cutoff:
                            stats['total'] += 1
                            stats[alert.get('severity', 'info')] += 1
                            if alert.get('acknowledged'):
                                stats['acknowledged'] += 1
                            if alert.get('resolved'):
                                stats['resolved'] += 1
                    except (json.JSONDecodeError, ValueError):
                        continue
        except FileNotFoundError:
            pass

        return stats


# Global alert manager instance
alert_manager = AlertManager()


# Default handlers

def console_alert_handler(alert: Alert):
    """Print alert to console."""
    symbols = {
        'critical': '\u274c',  # Red X
        'warning': '\u26a0\ufe0f',  # Warning sign
        'info': '\u2139\ufe0f'  # Info
    }
    symbol = symbols.get(alert.severity, '\u2139\ufe0f')
    print(f"\n{symbol} ALERT: {alert.title}")
    print(f"   Severity: {alert.severity.upper()}")
    print(f"   Message: {alert.message}")
    print(f"   Value: {alert.current_value} (threshold: {alert.threshold})")
    print(f"   Time: {alert.timestamp}\n")


def logging_alert_handler(alert: Alert):
    """Log alert using logging module."""
    log_level = {
        'critical': logging.CRITICAL,
        'warning': logging.WARNING,
        'info': logging.INFO
    }.get(alert.severity, logging.INFO)

    logger.log(
        log_level,
        f"ALERT [{alert.severity.upper()}] {alert.title}: {alert.message} "
        f"(value={alert.current_value}, threshold={alert.threshold})"
    )


# Register default handlers
alert_manager.register_handler(logging_alert_handler)
