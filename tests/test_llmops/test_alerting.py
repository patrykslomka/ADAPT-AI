"""Tests for alerting system."""
import pytest
import tempfile
from pathlib import Path
from src.llmops.alerting import AlertManager, Alert, AlertRule


class TestAlertManager:
    """Tests for AlertManager class."""

    @pytest.fixture
    def alert_manager(self, tmp_path):
        """Create alert manager with temporary file."""
        alerts_file = tmp_path / "alerts.jsonl"
        manager = AlertManager(alerts_file=alerts_file)
        # Clear default rules for testing
        manager.rules = []
        manager.alert_handlers = []
        return manager

    def test_add_rule(self, alert_manager):
        """Test adding alert rules."""
        rule = AlertRule(
            name='test_rule',
            metric='test_metric',
            condition='gt',
            threshold=10.0,
            severity='warning',
            message='Test alert'
        )

        alert_manager.add_rule(rule)

        assert len(alert_manager.rules) == 1
        assert alert_manager.rules[0].name == 'test_rule'

    def test_check_metrics_no_trigger(self, alert_manager):
        """Test that metrics below threshold don't trigger."""
        alert_manager.add_rule(AlertRule(
            name='high_latency',
            metric='response_time',
            condition='gt',
            threshold=10.0,
            severity='warning',
            message='High latency'
        ))

        alerts = alert_manager.check_metrics({'response_time': 5.0})

        assert len(alerts) == 0

    def test_check_metrics_trigger(self, alert_manager):
        """Test that metrics above threshold trigger alert."""
        alert_manager.add_rule(AlertRule(
            name='high_latency',
            metric='response_time',
            condition='gt',
            threshold=10.0,
            severity='warning',
            message='High latency'
        ))

        alerts = alert_manager.check_metrics({'response_time': 15.0})

        assert len(alerts) == 1
        assert alerts[0].title == 'High Latency'
        assert alerts[0].severity == 'warning'

    def test_check_metrics_eq_condition(self, alert_manager):
        """Test equality condition."""
        alert_manager.add_rule(AlertRule(
            name='compliance_fail',
            metric='compliance_passed',
            condition='eq',
            threshold=0.0,
            severity='critical',
            message='Compliance failed'
        ))

        # False should equal 0
        alerts = alert_manager.check_metrics({'compliance_passed': False})

        assert len(alerts) == 1
        assert alerts[0].severity == 'critical'

    def test_check_metrics_lt_condition(self, alert_manager):
        """Test less-than condition."""
        alert_manager.add_rule(AlertRule(
            name='low_confidence',
            metric='confidence',
            condition='lt',
            threshold=0.5,
            severity='warning',
            message='Low confidence'
        ))

        alerts = alert_manager.check_metrics({'confidence': 0.3})

        assert len(alerts) == 1

    def test_cooldown_prevents_duplicate_alerts(self, alert_manager):
        """Test that cooldown prevents rapid re-alerting."""
        alert_manager.add_rule(AlertRule(
            name='test_alert',
            metric='value',
            condition='gt',
            threshold=10.0,
            severity='warning',
            message='Test',
            cooldown_seconds=60
        ))

        # First alert should trigger
        alerts1 = alert_manager.check_metrics({'value': 20.0})
        assert len(alerts1) == 1

        # Second alert within cooldown should not trigger
        alerts2 = alert_manager.check_metrics({'value': 20.0})
        assert len(alerts2) == 0

    def test_register_handler(self, alert_manager):
        """Test registering alert handlers."""
        handler_called = []

        def test_handler(alert):
            handler_called.append(alert)

        alert_manager.register_handler(test_handler)

        alert_manager.add_rule(AlertRule(
            name='test',
            metric='value',
            condition='gt',
            threshold=5.0,
            severity='info',
            message='Test'
        ))

        alert_manager.check_metrics({'value': 10.0})

        assert len(handler_called) == 1

    def test_acknowledge_alert(self, alert_manager):
        """Test acknowledging an alert."""
        alert_manager.add_rule(AlertRule(
            name='test',
            metric='value',
            condition='gt',
            threshold=5.0,
            severity='warning',
            message='Test'
        ))

        alerts = alert_manager.check_metrics({'value': 10.0})
        alert_id = alerts[0].alert_id

        result = alert_manager.acknowledge_alert(alert_id)

        assert result is True
        assert alert_manager.active_alerts[0].acknowledged is True

    def test_resolve_alert(self, alert_manager):
        """Test resolving an alert."""
        alert_manager.add_rule(AlertRule(
            name='test',
            metric='value',
            condition='gt',
            threshold=5.0,
            severity='warning',
            message='Test'
        ))

        alerts = alert_manager.check_metrics({'value': 10.0})
        alert_id = alerts[0].alert_id

        result = alert_manager.resolve_alert(alert_id)

        assert result is True
        assert len(alert_manager.active_alerts) == 0

    def test_get_active_alerts_by_severity(self, alert_manager):
        """Test filtering active alerts by severity."""
        alert_manager.add_rule(AlertRule(
            name='warning1',
            metric='metric1',
            condition='gt',
            threshold=5.0,
            severity='warning',
            message='Warning'
        ))
        alert_manager.add_rule(AlertRule(
            name='critical1',
            metric='metric2',
            condition='gt',
            threshold=5.0,
            severity='critical',
            message='Critical'
        ))

        alert_manager.check_metrics({'metric1': 10.0, 'metric2': 10.0})

        critical = alert_manager.get_active_alerts(severity='critical')
        warning = alert_manager.get_active_alerts(severity='warning')

        assert len(critical) == 1
        assert len(warning) == 1

    def test_get_alert_history(self, alert_manager):
        """Test getting alert history."""
        alert_manager.add_rule(AlertRule(
            name='test',
            metric='value',
            condition='gt',
            threshold=5.0,
            severity='warning',
            message='Test',
            cooldown_seconds=0  # No cooldown
        ))

        # Trigger multiple alerts
        alert_manager.check_metrics({'value': 10.0})
        alert_manager.last_alert_times = {}  # Reset cooldown
        alert_manager.check_metrics({'value': 15.0})

        history = alert_manager.get_alert_history()

        assert len(history) >= 2

    def test_get_alert_stats(self, alert_manager):
        """Test getting alert statistics."""
        alert_manager.add_rule(AlertRule(
            name='warning',
            metric='value',
            condition='gt',
            threshold=5.0,
            severity='warning',
            message='Warning'
        ))

        alert_manager.check_metrics({'value': 10.0})

        stats = alert_manager.get_alert_stats(hours=24)

        assert stats['total'] >= 1
        assert 'warning' in stats


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            alert_id='test-001',
            severity='critical',
            title='Test Alert',
            message='This is a test',
            metric='test_metric',
            threshold=10.0,
            current_value=15.0,
            timestamp='2024-01-01T00:00:00'
        )

        assert alert.alert_id == 'test-001'
        assert alert.severity == 'critical'
        assert not alert.acknowledged
        assert not alert.resolved


class TestAlertRule:
    """Tests for AlertRule dataclass."""

    def test_rule_creation(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            name='test_rule',
            metric='cpu_usage',
            condition='gt',
            threshold=90.0,
            severity='critical',
            message='CPU usage too high'
        )

        assert rule.name == 'test_rule'
        assert rule.condition == 'gt'
        assert rule.cooldown_seconds == 300  # Default
