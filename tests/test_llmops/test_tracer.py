"""Tests for distributed tracing."""
import pytest
import tempfile
from pathlib import Path
import time
from src.llmops.tracer import DistributedTracer, TraceSpan, TraceContext, SpanContext


class TestDistributedTracer:
    """Tests for DistributedTracer class."""

    @pytest.fixture
    def tracer(self, tmp_path):
        """Create tracer with temporary file."""
        trace_file = tmp_path / "traces.jsonl"
        return DistributedTracer(trace_file=trace_file)

    def test_start_trace(self, tracer):
        """Test starting a new trace."""
        span = tracer.start_trace("trace-001", "test_operation")

        assert span.trace_id == "trace-001"
        assert span.name == "test_operation"
        assert span.status == "in_progress"
        assert span.start_time > 0

    def test_start_span(self, tracer):
        """Test starting a span within a trace."""
        tracer.start_trace("trace-001", "root")
        span = tracer.start_span("trace-001", "child_operation", "trace-001-root")

        assert span.trace_id == "trace-001"
        assert span.parent_span_id == "trace-001-root"
        assert span.name == "child_operation"

    def test_end_span(self, tracer):
        """Test ending a span."""
        span = tracer.start_trace("trace-001", "test")
        time.sleep(0.01)  # Small delay
        tracer.end_span(span, "success")

        assert span.status == "success"
        assert span.end_time is not None
        assert span.duration > 0

    def test_add_event(self, tracer):
        """Test adding events to a span."""
        span = tracer.start_trace("trace-001", "test")

        tracer.add_event(span, "checkpoint", {"data": "test"})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"
        assert span.events[0]["attributes"]["data"] == "test"

    def test_set_attribute(self, tracer):
        """Test setting attributes on a span."""
        span = tracer.start_trace("trace-001", "test")

        tracer.set_attribute(span, "user_id", "123")

        assert span.attributes["user_id"] == "123"

    def test_end_trace(self, tracer):
        """Test ending all spans in a trace."""
        tracer.start_trace("trace-001", "root")
        tracer.start_span("trace-001", "child1")
        tracer.start_span("trace-001", "child2")

        tracer.end_trace("trace-001", "success")

        # All spans should be removed from active
        assert "trace-001" not in tracer.active_traces

    def test_write_span_to_file(self, tracer):
        """Test that spans are written to file."""
        span = tracer.start_trace("trace-001", "test")
        tracer.end_span(span, "success")

        # Read file
        with open(tracer.trace_file, 'r') as f:
            content = f.read()

        assert "trace-001" in content
        assert "success" in content

    def test_get_trace(self, tracer):
        """Test getting all spans for a trace."""
        span1 = tracer.start_trace("trace-001", "root")
        tracer.end_span(span1)

        span2 = tracer.start_span("trace-001", "child")
        tracer.end_span(span2)

        spans = tracer.get_trace("trace-001")

        assert len(spans) == 2

    def test_get_recent_traces(self, tracer):
        """Test getting recent traces."""
        for i in range(5):
            span = tracer.start_trace(f"trace-{i}", f"op-{i}")
            tracer.end_span(span)

        traces = tracer.get_recent_traces(limit=3)

        assert len(traces) == 3


class TestTraceContext:
    """Tests for TraceContext context manager."""

    @pytest.fixture
    def tracer(self, tmp_path):
        trace_file = tmp_path / "traces.jsonl"
        return DistributedTracer(trace_file=trace_file)

    def test_context_manager_success(self, tracer):
        """Test context manager for successful operation."""
        with TraceContext(tracer, "trace-001", "test_op") as span:
            assert span.status == "in_progress"

        spans = tracer.get_trace("trace-001")
        assert len(spans) == 1
        assert spans[0]["status"] == "success"

    def test_context_manager_error(self, tracer):
        """Test context manager handles errors."""
        try:
            with TraceContext(tracer, "trace-002", "failing_op") as span:
                raise ValueError("Test error")
        except ValueError:
            pass

        spans = tracer.get_trace("trace-002")
        assert len(spans) == 1
        assert spans[0]["status"] == "error"


class TestSpanContext:
    """Tests for SpanContext context manager."""

    @pytest.fixture
    def tracer(self, tmp_path):
        trace_file = tmp_path / "traces.jsonl"
        return DistributedTracer(trace_file=trace_file)

    def test_span_context_success(self, tracer):
        """Test span context for successful operation."""
        root = tracer.start_trace("trace-001", "root")

        with SpanContext(tracer, "trace-001", "child", root.span_id) as span:
            assert span.parent_span_id == root.span_id

        tracer.end_span(root)

    def test_nested_spans(self, tracer):
        """Test nested span contexts."""
        with TraceContext(tracer, "trace-001", "root") as root:
            with SpanContext(tracer, "trace-001", "level1", root.span_id) as s1:
                with SpanContext(tracer, "trace-001", "level2", s1.span_id) as s2:
                    assert s2.parent_span_id == s1.span_id
