"""Test LLMOps metrics collector."""
import pytest
import sys
from pathlib import Path
from datetime import datetime
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llmops.metrics_collector import MetricsCollector, QueryMetrics


@pytest.fixture
def temp_db():
    """Create temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_metrics.db"
        yield db_path


@pytest.fixture
def collector(temp_db):
    """Create metrics collector with temp database."""
    return MetricsCollector(db_path=temp_db)


@pytest.fixture
def sample_metrics():
    """Create sample metrics."""
    return QueryMetrics(
        query_id="test-query-001",
        timestamp=datetime.now().isoformat(),
        query_text="What are TB symptoms?",
        patient_id="P-0001",
        total_response_time=2.5,
        mcp_time=0.1,
        primary_agent_time=1.5,
        compliance_agent_time=0.3,
        quality_agent_time=0.6,
        rag_time=0.5,
        rat_time=None,
        total_input_tokens=500,
        total_output_tokens=200,
        primary_agent_tokens=400,
        compliance_agent_tokens=50,
        quality_agent_tokens=250,
        total_cost=0.01,
        compliance_passed=True,
        quality_passed=True,
        hallucination_score=0.1,
        confidence_score=0.9,
        response_generated=True,
        error_occurred=False,
        error_message=None
    )


class TestMetricsRecording:
    """Test metrics recording."""

    def test_record_metrics(self, collector, sample_metrics):
        """Test recording metrics."""
        collector.record_metrics(sample_metrics)

        # Verify recorded
        recent = collector.get_recent_queries(limit=1)
        assert len(recent) == 1
        assert recent[0]['query_id'] == 'test-query-001'

    def test_record_multiple_metrics(self, collector, sample_metrics):
        """Test recording multiple metrics."""
        # Record multiple
        for i in range(5):
            metrics = QueryMetrics(
                query_id=f"test-query-{i:03d}",
                timestamp=datetime.now().isoformat(),
                query_text=f"Test query {i}",
                patient_id=None,
                total_response_time=1.0 + i * 0.5,
                mcp_time=0.1,
                primary_agent_time=0.5,
                compliance_agent_time=0.2,
                quality_agent_time=0.2,
                rag_time=0.3,
                rat_time=None,
                total_input_tokens=100 * (i + 1),
                total_output_tokens=50 * (i + 1),
                primary_agent_tokens=80 * (i + 1),
                compliance_agent_tokens=10 * (i + 1),
                quality_agent_tokens=60 * (i + 1),
                total_cost=0.001 * (i + 1),
                compliance_passed=True,
                quality_passed=True,
                hallucination_score=0.1,
                confidence_score=0.9,
                response_generated=True,
                error_occurred=False
            )
            collector.record_metrics(metrics)

        # Verify count
        recent = collector.get_recent_queries(limit=10)
        assert len(recent) == 5


class TestMetricsSummary:
    """Test metrics summary."""

    def test_get_metrics_summary_empty(self, collector):
        """Test summary with no data."""
        summary = collector.get_metrics_summary(hours=24)

        assert summary['total_queries'] == 0
        assert summary['avg_response_time'] == 0
        assert summary['total_cost'] == 0

    def test_get_metrics_summary(self, collector, sample_metrics):
        """Test summary with data."""
        collector.record_metrics(sample_metrics)

        summary = collector.get_metrics_summary(hours=24)

        assert summary['total_queries'] == 1
        assert summary['avg_response_time'] > 0
        assert summary['total_cost'] > 0
        assert summary['compliance_pass_rate'] == 100.0
        assert summary['error_count'] == 0


class TestCostBreakdown:
    """Test cost breakdown."""

    def test_cost_breakdown(self, collector, sample_metrics):
        """Test cost breakdown calculation."""
        collector.record_metrics(sample_metrics)

        breakdown = collector.get_cost_breakdown(hours=24)

        assert 'input_tokens_cost' in breakdown
        assert 'output_tokens_cost' in breakdown
        assert 'total_estimated_cost' in breakdown
        assert breakdown['total_estimated_cost'] >= 0


class TestRecentQueries:
    """Test recent queries retrieval."""

    def test_get_recent_queries_empty(self, collector):
        """Test getting recent queries with no data."""
        recent = collector.get_recent_queries(limit=10)
        assert len(recent) == 0

    def test_get_recent_queries_limit(self, collector):
        """Test limit on recent queries."""
        # Add 10 records
        for i in range(10):
            metrics = QueryMetrics(
                query_id=f"test-{i}",
                timestamp=datetime.now().isoformat(),
                query_text=f"Query {i}",
                patient_id=None,
                total_response_time=1.0,
                mcp_time=0.1,
                primary_agent_time=0.5,
                compliance_agent_time=0.2,
                quality_agent_time=0.2,
                rag_time=0.3,
                rat_time=None,
                total_input_tokens=100,
                total_output_tokens=50,
                primary_agent_tokens=80,
                compliance_agent_tokens=10,
                quality_agent_tokens=60,
                total_cost=0.01,
                compliance_passed=True,
                quality_passed=True,
                hallucination_score=0.1,
                confidence_score=0.9,
                response_generated=True,
                error_occurred=False
            )
            collector.record_metrics(metrics)

        # Get only 5
        recent = collector.get_recent_queries(limit=5)
        assert len(recent) == 5


class TestErrorTracking:
    """Test error tracking in metrics."""

    def test_error_tracking(self, collector):
        """Test tracking errors."""
        metrics = QueryMetrics(
            query_id="error-test",
            timestamp=datetime.now().isoformat(),
            query_text="Failed query",
            patient_id=None,
            total_response_time=0.5,
            mcp_time=0.1,
            primary_agent_time=0.0,
            compliance_agent_time=0.0,
            quality_agent_time=0.0,
            rag_time=None,
            rat_time=None,
            total_input_tokens=50,
            total_output_tokens=0,
            primary_agent_tokens=50,
            compliance_agent_tokens=0,
            quality_agent_tokens=0,
            total_cost=0.001,
            compliance_passed=False,
            quality_passed=False,
            hallucination_score=0.0,
            confidence_score=0.0,
            response_generated=False,
            error_occurred=True,
            error_message="API connection failed"
        )
        collector.record_metrics(metrics)

        summary = collector.get_metrics_summary(hours=24)
        assert summary['error_count'] == 1
