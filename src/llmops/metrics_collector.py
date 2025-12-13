"""LLMOps Metrics Collector.

Collects and stores metrics for monitoring AI system performance.
"""
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    timestamp: str
    query_text: str
    patient_id: Optional[str]

    # Timing metrics
    total_response_time: float
    mcp_time: float
    primary_agent_time: float
    compliance_agent_time: float
    quality_agent_time: float
    rag_time: Optional[float]
    rat_time: Optional[float]

    # Token metrics
    total_input_tokens: int
    total_output_tokens: int
    primary_agent_tokens: int
    compliance_agent_tokens: int
    quality_agent_tokens: int

    # Cost metrics
    total_cost: float

    # Quality metrics
    compliance_passed: bool
    quality_passed: bool
    hallucination_score: float
    confidence_score: float

    # Status
    response_generated: bool
    error_occurred: bool
    error_message: Optional[str] = None


class MetricsCollector:
    """Collect and store LLMOps metrics."""

    def __init__(self, db_path: Path = None):
        """Initialize metrics collector.

        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            db_path = Path("./data/metrics.db")

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info(f"Metrics collector initialized: {db_path}")

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_metrics (
                query_id TEXT PRIMARY KEY,
                timestamp TEXT,
                query_text TEXT,
                patient_id TEXT,

                total_response_time REAL,
                mcp_time REAL,
                primary_agent_time REAL,
                compliance_agent_time REAL,
                quality_agent_time REAL,
                rag_time REAL,
                rat_time REAL,

                total_input_tokens INTEGER,
                total_output_tokens INTEGER,
                primary_agent_tokens INTEGER,
                compliance_agent_tokens INTEGER,
                quality_agent_tokens INTEGER,

                total_cost REAL,

                compliance_passed BOOLEAN,
                quality_passed BOOLEAN,
                hallucination_score REAL,
                confidence_score REAL,

                response_generated BOOLEAN,
                error_occurred BOOLEAN,
                error_message TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                total_queries INTEGER,
                total_tokens INTEGER,
                total_cost REAL,
                avg_response_time REAL,
                compliance_pass_rate REAL,
                quality_pass_rate REAL,
                avg_confidence REAL,
                error_count INTEGER
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON query_metrics(timestamp)
        """)

        conn.commit()
        conn.close()

    def record_metrics(self, metrics: QueryMetrics):
        """Record metrics for a query.

        Args:
            metrics: QueryMetrics instance
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data = asdict(metrics)

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])

        cursor.execute(
            f"INSERT OR REPLACE INTO query_metrics ({columns}) VALUES ({placeholders})",
            list(data.values())
        )

        conn.commit()
        conn.close()

        logger.debug(f"Recorded metrics for query: {metrics.query_id}")

    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary metrics for time period.

        Args:
            hours: Number of hours to look back

        Returns:
            Dict with summary statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        cursor.execute("""
            SELECT
                COUNT(*) as total_queries,
                COALESCE(AVG(total_response_time), 0) as avg_response_time,
                COALESCE(SUM(total_cost), 0) as total_cost,
                COALESCE(SUM(total_input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(total_output_tokens), 0) as total_output_tokens,
                COALESCE(AVG(CASE WHEN compliance_passed THEN 1.0 ELSE 0.0 END) * 100, 0) as compliance_pass_rate,
                COALESCE(AVG(CASE WHEN quality_passed THEN 1.0 ELSE 0.0 END) * 100, 0) as quality_pass_rate,
                COALESCE(AVG(confidence_score), 0) as avg_confidence,
                COALESCE(SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END), 0) as error_count
            FROM query_metrics
            WHERE timestamp >= ?
        """, (cutoff,))

        row = cursor.fetchone()
        conn.close()

        return {
            "period_hours": hours,
            "total_queries": row[0],
            "avg_response_time": round(row[1], 2),
            "total_cost": round(row[2], 4),
            "total_input_tokens": row[3],
            "total_output_tokens": row[4],
            "compliance_pass_rate": round(row[5], 1),
            "quality_pass_rate": round(row[6], 1),
            "avg_confidence": round(row[7], 2),
            "error_count": row[8]
        }

    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent queries with metrics.

        Args:
            limit: Number of queries to return

        Returns:
            List of query metrics dicts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM query_metrics
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_cost_breakdown(self, hours: int = 24) -> Dict[str, float]:
        """Get cost breakdown by component.

        Args:
            hours: Number of hours to look back

        Returns:
            Dict with cost per component
        """
        # For this implementation, we estimate based on token usage
        # In production, you'd track actual API costs
        summary = self.get_metrics_summary(hours)

        # Haiku pricing: $0.80/MTok input, $4.00/MTok output
        input_cost = summary["total_input_tokens"] * 0.8 / 1_000_000
        output_cost = summary["total_output_tokens"] * 4.0 / 1_000_000

        return {
            "input_tokens_cost": round(input_cost, 4),
            "output_tokens_cost": round(output_cost, 4),
            "total_estimated_cost": round(input_cost + output_cost, 4),
            "total_recorded_cost": summary["total_cost"]
        }

    def update_daily_summary(self):
        """Update daily summary table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().strftime("%Y-%m-%d")
        start = f"{today} 00:00:00"
        end = f"{today} 23:59:59"

        cursor.execute("""
            SELECT
                COUNT(*) as total_queries,
                COALESCE(SUM(total_input_tokens + total_output_tokens), 0) as total_tokens,
                COALESCE(SUM(total_cost), 0) as total_cost,
                COALESCE(AVG(total_response_time), 0) as avg_response_time,
                COALESCE(AVG(CASE WHEN compliance_passed THEN 1.0 ELSE 0.0 END), 0) as compliance_pass_rate,
                COALESCE(AVG(CASE WHEN quality_passed THEN 1.0 ELSE 0.0 END), 0) as quality_pass_rate,
                COALESCE(AVG(confidence_score), 0) as avg_confidence,
                COALESCE(SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END), 0) as error_count
            FROM query_metrics
            WHERE timestamp BETWEEN ? AND ?
        """, (start, end))

        row = cursor.fetchone()

        cursor.execute("""
            INSERT OR REPLACE INTO daily_summary
            (date, total_queries, total_tokens, total_cost, avg_response_time,
             compliance_pass_rate, quality_pass_rate, avg_confidence, error_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (today, *row))

        conn.commit()
        conn.close()

        logger.info(f"Updated daily summary for {today}")

    def get_daily_trends(self, days: int = 7) -> List[Dict]:
        """Get daily trends.

        Args:
            days: Number of days to look back

        Returns:
            List of daily summaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        cursor.execute("""
            SELECT *
            FROM daily_summary
            WHERE date >= ?
            ORDER BY date DESC
        """, (cutoff,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]


# Global metrics collector instance
metrics_collector = MetricsCollector()
