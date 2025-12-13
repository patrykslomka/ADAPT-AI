"""Distributed tracing for LLMOps."""
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
import time
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class TraceSpan:
    """Single trace span."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "in_progress"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)


class DistributedTracer:
    """Trace requests through multi-agent system."""

    def __init__(self, trace_file: Path = None):
        if trace_file is None:
            trace_file = Path("./logs/traces.jsonl")
        self.trace_file = trace_file
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self.active_traces: Dict[str, List[TraceSpan]] = {}
        self._lock = threading.Lock()

    def start_trace(self, trace_id: str, name: str, attributes: Dict = None) -> TraceSpan:
        """Start new trace."""
        span = TraceSpan(
            span_id=f"{trace_id}-root",
            trace_id=trace_id,
            parent_span_id=None,
            name=name,
            start_time=time.time(),
            attributes=attributes or {}
        )

        with self._lock:
            if trace_id not in self.active_traces:
                self.active_traces[trace_id] = []
            self.active_traces[trace_id].append(span)

        logger.debug(f"Started trace: {trace_id} - {name}")
        return span

    def start_span(
        self,
        trace_id: str,
        name: str,
        parent_span_id: Optional[str] = None,
        attributes: Dict = None
    ) -> TraceSpan:
        """Start a new span within a trace."""
        span_id = f"{trace_id}-{name}-{int(time.time() * 1000)}"

        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            start_time=time.time(),
            attributes=attributes or {}
        )

        with self._lock:
            if trace_id not in self.active_traces:
                self.active_traces[trace_id] = []
            self.active_traces[trace_id].append(span)

        logger.debug(f"Started span: {span_id}")
        return span

    def add_event(self, span: TraceSpan, name: str, attributes: Dict = None):
        """Add an event to a span."""
        event = {
            'name': name,
            'timestamp': time.time(),
            'attributes': attributes or {}
        }
        span.events.append(event)
        logger.debug(f"Added event '{name}' to span {span.span_id}")

    def set_attribute(self, span: TraceSpan, key: str, value: Any):
        """Set an attribute on a span."""
        span.attributes[key] = value

    def end_span(self, span: TraceSpan, status: str = "success"):
        """End trace span."""
        span.end_time = time.time()
        span.duration = span.end_time - span.start_time
        span.status = status

        # Write to file
        self._write_span(span)
        logger.debug(f"Ended span: {span.span_id} - {status} ({span.duration:.3f}s)")

    def end_trace(self, trace_id: str, status: str = "success"):
        """End all spans in a trace."""
        with self._lock:
            spans = self.active_traces.get(trace_id, [])

            for span in spans:
                if span.end_time is None:
                    span.end_time = time.time()
                    span.duration = span.end_time - span.start_time
                    span.status = status
                    self._write_span(span)

            # Clean up
            if trace_id in self.active_traces:
                del self.active_traces[trace_id]

        logger.info(f"Ended trace: {trace_id} with {len(spans)} spans")

    def _write_span(self, span: TraceSpan):
        """Write span to trace file."""
        try:
            with open(self.trace_file, 'a') as f:
                span_dict = asdict(span)
                # Convert timestamps to ISO format for readability
                span_dict['start_time_iso'] = datetime.fromtimestamp(span.start_time).isoformat()
                if span.end_time:
                    span_dict['end_time_iso'] = datetime.fromtimestamp(span.end_time).isoformat()
                f.write(json.dumps(span_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to write span: {e}")

    def get_trace(self, trace_id: str) -> List[Dict]:
        """Get all spans for a trace from file."""
        spans = []
        try:
            with open(self.trace_file, 'r') as f:
                for line in f:
                    try:
                        span = json.loads(line)
                        if span.get('trace_id') == trace_id:
                            spans.append(span)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

        return spans

    def get_recent_traces(self, limit: int = 10) -> List[Dict]:
        """Get recent traces."""
        traces = {}
        try:
            with open(self.trace_file, 'r') as f:
                for line in f:
                    try:
                        span = json.loads(line)
                        trace_id = span.get('trace_id')
                        if trace_id not in traces:
                            traces[trace_id] = {
                                'trace_id': trace_id,
                                'spans': [],
                                'start_time': span.get('start_time')
                            }
                        traces[trace_id]['spans'].append(span)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return []

        # Sort by start time and return most recent
        sorted_traces = sorted(
            traces.values(),
            key=lambda x: x.get('start_time', 0),
            reverse=True
        )
        return sorted_traces[:limit]


class TraceContext:
    """Context manager for tracing."""

    def __init__(self, tracer: DistributedTracer, trace_id: str, name: str, attributes: Dict = None):
        self.tracer = tracer
        self.trace_id = trace_id
        self.name = name
        self.attributes = attributes or {}
        self.span: Optional[TraceSpan] = None

    def __enter__(self) -> TraceSpan:
        self.span = self.tracer.start_trace(self.trace_id, self.name, self.attributes)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "error" if exc_type else "success"
        if self.span:
            if exc_val:
                self.tracer.add_event(self.span, "exception", {"error": str(exc_val)})
            self.tracer.end_span(self.span, status)
        return False


class SpanContext:
    """Context manager for individual spans."""

    def __init__(
        self,
        tracer: DistributedTracer,
        trace_id: str,
        name: str,
        parent_span_id: str = None,
        attributes: Dict = None
    ):
        self.tracer = tracer
        self.trace_id = trace_id
        self.name = name
        self.parent_span_id = parent_span_id
        self.attributes = attributes or {}
        self.span: Optional[TraceSpan] = None

    def __enter__(self) -> TraceSpan:
        self.span = self.tracer.start_span(
            self.trace_id, self.name, self.parent_span_id, self.attributes
        )
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "error" if exc_type else "success"
        if self.span:
            if exc_val:
                self.tracer.add_event(self.span, "exception", {"error": str(exc_val)})
            self.tracer.end_span(self.span, status)
        return False


# Global tracer instance
tracer = DistributedTracer()
