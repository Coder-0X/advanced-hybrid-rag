"""
Unit tests for Distributed Observability, OpenTelemetry Tracing, and Prometheus Metrics.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.observability.tracer import tracer, inject_w3c_carrier, extract_w3c_carrier
from app.observability.metrics import record_stage_latency, get_prometheus_metrics

client = TestClient(app)


def test_opentelemetry_w3c_carrier_injection():
    """Verify trace context can be injected into and extracted from W3C carriers."""
    with tracer.start_as_current_span("test_span") as span:
        carrier = inject_w3c_carrier()
        assert "traceparent" in carrier
        assert carrier["traceparent"].startswith("00-")

        ctx = extract_w3c_carrier(carrier)
        assert ctx is not None


def test_prometheus_histogram_recording():
    """Verify latency recordings are exposed via Prometheus format."""
    record_stage_latency(stage="dense_retrieval", duration_sec=0.015, archetype="lexical_exact")
    record_stage_latency(stage="rerank", duration_sec=0.035, archetype="semantic_paraphrase")

    content, content_type = get_prometheus_metrics()
    assert isinstance(content, bytes)
    text = content.decode("utf-8")
    assert "rag_retrieval_latency_seconds_bucket" in text
    assert 'stage="dense_retrieval"' in text
    assert 'stage="rerank"' in text


def test_fastapi_trace_id_header_and_metrics_endpoint():
    """Verify ASGI middleware injects X-Trace-Id and exposes /metrics."""
    # Check trace ID header on health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert "X-Trace-Id" in health_resp.headers
    assert len(health_resp.headers["X-Trace-Id"]) == 32

    # Check metrics endpoint
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "rag_retrieval_latency_seconds" in metrics_resp.text
