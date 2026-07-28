"""Thin OpenTelemetry wrapper for pipeline tracing.

Uses the OTel API (already in core deps). When no TracerProvider/exporter
is configured, all spans are no-ops. The OTLP gRPC exporter is an
optional dependency (`pip install hazard-assessment[telemetry]`).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

from opentelemetry import trace

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("hazard_assessment.pipeline")


@contextlib.contextmanager
def pipeline_span(trace_id: str, name: str = "pipeline_run") -> Iterator[trace.Span]:
    """Context manager wrapping a pipeline execution in an OTel span."""
    with _tracer.start_as_current_span(
        name,
        attributes={"pipeline.trace_id": trace_id},
    ) as span:
        yield span


def configure_tracer_provider(otlp_endpoint: str | None = None) -> None:
    """Configure OTel TracerProvider with optional OTLP exporter.

    Called once at app startup. When otlp_endpoint is None, uses the
    default no-op provider (spans are silently discarded).
    """
    if otlp_endpoint is None:
        return

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("OTel tracing configured: endpoint=%s", otlp_endpoint)
    except ImportError:
        logger.warning(
            "opentelemetry-exporter-otlp not installed; tracing disabled. "
            "Install with: pip install hazard-assessment[telemetry]"
        )
