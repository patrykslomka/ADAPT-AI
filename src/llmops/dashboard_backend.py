"""Backend API for metrics dashboard."""
from typing import Dict, List, Optional, Any
from src.llmops.metrics_collector import metrics_collector
from src.llmops.alerting import alert_manager
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DashboardBackend:
    """Provide data for metrics dashboard."""

    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = Path("./data/metrics.db")
        self.db_path = db_path

    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent queries with key metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    query_id,
                    timestamp,
                    query_text,
                    patient_id,
                    total_response_time,
                    total_cost,
                    compliance_passed,
                    quality_passed,
                    confidence_score
                FROM query_metrics
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    'query_id': row[0],
                    'timestamp': row[1],
                    'query': row[2][:50] + '...' if len(row[2] or '') > 50 else row[2],
                    'patient_id': row[3],
                    'response_time': round(row[4] or 0, 2),
                    'cost': f"${row[5]:.4f}" if row[5] else "$0.0000",
                    'compliance': 'Pass' if row[6] else 'Fail',
                    'quality': 'Pass' if row[7] else 'Fail',
                    'confidence': f"{(row[8] or 0):.0%}"
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get recent queries: {e}")
            return []

    def get_time_series(self, hours: int = 24, metric: str = 'total_response_time') -> Dict:
        """Get time series data for charts."""
        valid_metrics = [
            'total_response_time', 'total_cost', 'confidence_score',
            'hallucination_score', 'total_input_tokens', 'total_output_tokens'
        ]

        if metric not in valid_metrics:
            metric = 'total_response_time'

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    AVG({metric}) as avg_value,
                    MIN({metric}) as min_value,
                    MAX({metric}) as max_value,
                    COUNT(*) as count
                FROM query_metrics
                WHERE timestamp >= datetime('now', '-{hours} hours')
                GROUP BY hour
                ORDER BY hour
            """)

            rows = cursor.fetchall()
            conn.close()

            return {
                'metric': metric,
                'labels': [row[0] for row in rows],
                'average': [round(row[1], 2) if row[1] else 0 for row in rows],
                'min': [round(row[2], 2) if row[2] else 0 for row in rows],
                'max': [round(row[3], 2) if row[3] else 0 for row in rows],
                'count': [row[4] for row in rows]
            }
        except Exception as e:
            logger.error(f"Failed to get time series: {e}")
            return {
                'metric': metric,
                'labels': [],
                'average': [],
                'min': [],
                'max': [],
                'count': []
            }

    def get_agent_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Get agent performance metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT
                    AVG(primary_agent_time) as avg_primary,
                    AVG(compliance_agent_time) as avg_compliance,
                    AVG(quality_agent_time) as avg_quality,
                    AVG(total_response_time) as avg_total,
                    SUM(primary_agent_tokens) as total_primary_tokens,
                    SUM(quality_agent_tokens) as total_quality_tokens
                FROM query_metrics
                WHERE timestamp >= datetime('now', '-{hours} hours')
            """)

            row = cursor.fetchone()
            conn.close()

            return {
                'primary_agent': {
                    'avg_time': round(row[0] or 0, 2),
                    'total_tokens': row[4] or 0
                },
                'compliance_agent': {
                    'avg_time': round(row[1] or 0, 2),
                    'total_tokens': 0  # Rule-based
                },
                'quality_agent': {
                    'avg_time': round(row[2] or 0, 2),
                    'total_tokens': row[5] or 0
                },
                'total': {
                    'avg_time': round(row[3] or 0, 2)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get agent performance: {e}")
            return {}

    def get_quality_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get quality-related metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT
                    AVG(CASE WHEN compliance_passed THEN 1.0 ELSE 0.0 END) * 100 as compliance_rate,
                    AVG(CASE WHEN quality_passed THEN 1.0 ELSE 0.0 END) * 100 as quality_rate,
                    AVG(confidence_score) as avg_confidence,
                    AVG(hallucination_score) as avg_hallucination,
                    SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) as error_count,
                    COUNT(*) as total_queries
                FROM query_metrics
                WHERE timestamp >= datetime('now', '-{hours} hours')
            """)

            row = cursor.fetchone()
            conn.close()

            total = row[5] or 1
            return {
                'compliance_pass_rate': round(row[0] or 0, 1),
                'quality_pass_rate': round(row[1] or 0, 1),
                'avg_confidence': round(row[2] or 0, 2),
                'avg_hallucination_risk': round(row[3] or 0, 2),
                'error_count': row[4] or 0,
                'error_rate': round((row[4] or 0) / total * 100, 1),
                'total_queries': total
            }
        except Exception as e:
            logger.error(f"Failed to get quality metrics: {e}")
            return {}

    def get_cost_breakdown(self, hours: int = 24) -> Dict[str, float]:
        """Get cost breakdown by component."""
        summary = metrics_collector.get_metrics_summary(hours)

        # Haiku pricing: $0.80/MTok input, $4.00/MTok output
        input_cost = summary.get("total_input_tokens", 0) * 0.8 / 1_000_000
        output_cost = summary.get("total_output_tokens", 0) * 4.0 / 1_000_000

        return {
            "input_tokens": summary.get("total_input_tokens", 0),
            "output_tokens": summary.get("total_output_tokens", 0),
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total_cost": round(input_cost + output_cost, 4),
            "recorded_cost": summary.get("total_cost", 0)
        }

    def get_dashboard_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get complete dashboard summary."""
        summary = metrics_collector.get_metrics_summary(hours)
        quality = self.get_quality_metrics(hours)
        cost = self.get_cost_breakdown(hours)
        alerts = alert_manager.get_alert_stats(hours)

        return {
            'period_hours': hours,
            'overview': {
                'total_queries': summary.get('total_queries', 0),
                'avg_response_time': summary.get('avg_response_time', 0),
                'total_cost': cost['total_cost'],
                'error_rate': quality.get('error_rate', 0)
            },
            'quality': {
                'compliance_rate': quality.get('compliance_pass_rate', 0),
                'quality_rate': quality.get('quality_pass_rate', 0),
                'avg_confidence': quality.get('avg_confidence', 0),
                'hallucination_risk': quality.get('avg_hallucination_risk', 0)
            },
            'cost': cost,
            'alerts': alerts
        }

    def get_patient_query_stats(self, patient_id: str) -> Dict[str, Any]:
        """Get query statistics for a specific patient."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    AVG(total_response_time) as avg_response_time,
                    AVG(confidence_score) as avg_confidence,
                    MIN(timestamp) as first_query,
                    MAX(timestamp) as last_query
                FROM query_metrics
                WHERE patient_id = ?
            """, (patient_id,))

            row = cursor.fetchone()
            conn.close()

            return {
                'patient_id': patient_id,
                'total_queries': row[0] or 0,
                'avg_response_time': round(row[1] or 0, 2),
                'avg_confidence': round(row[2] or 0, 2),
                'first_query': row[3],
                'last_query': row[4]
            }
        except Exception as e:
            logger.error(f"Failed to get patient stats: {e}")
            return {'patient_id': patient_id, 'total_queries': 0}

    def export_metrics(self, hours: int = 24, format: str = 'json') -> str:
        """Export metrics data."""
        import json

        data = {
            'exported_at': datetime.now().isoformat(),
            'period_hours': hours,
            'summary': self.get_dashboard_summary(hours),
            'recent_queries': self.get_recent_queries(100),
            'agent_performance': self.get_agent_performance(hours)
        }

        if format == 'json':
            return json.dumps(data, indent=2)
        else:
            return json.dumps(data)


# Global dashboard instance
dashboard = DashboardBackend()
