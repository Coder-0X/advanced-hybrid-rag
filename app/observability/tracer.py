"""
Distributed Tracing Engine using OpenTelemetry with W3C Context Propagation.

Propagates traceparent contexts across asynchronous boundaries and thread pools,
instrumenting retrieval, fusion, and reranking stages with GenAI semantic conventions.
"""
from typing import Dict, Any, Optional
from opentelemetry import trace
from opentelemetry.trace import Tracer, SpanKind, Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

_propagator = TraceContextTextMapPropagator()


def setup_telemetry(
    service_name: str = "advanced-hybrid-rag",
    service_version: str = "2.0.0"
) -> Tracer:
    """
    Initializes OpenTelemetry TracerProvider with service resource metadata.
    """
    # Prevent re-registration if already active
    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        return trace.get_tracer(service_name, service_version)

    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
        "deployment.environment": "production"
    })
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name, service_version)


tracer = setup_telemetry()


def inject_w3c_carrier(carrier: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Inject active trace context into W3C traceparent carrier dict."""
    if carrier is None:
        carrier = {}
    _propagator.inject(carrier)
    return carrier


def extract_w3c_carrier(carrier: Dict[str, str]) -> Any:
    """Extract W3C traceparent carrier into OpenTelemetry context."""
    return _propagator.extract(carrier)
