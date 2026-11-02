"""
FastAPI ASGI Middleware for OpenTelemetry W3C Tracing and Latency Instrumentation.

Extracts traceparent headers from client requests, correlates child spans,
and injects X-Trace-Id into response headers for complete end-to-end auditability.
"""
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from app.observability.tracer import tracer, extract_w3c_carrier
from app.observability.metrics import IN_FLIGHT_REQUESTS


class OpenTelemetryMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware instrumenting HTTP requests with OpenTelemetry spans and W3C context.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract existing W3C traceparent context if present
        headers_dict = dict(request.headers)
        extracted_ctx = extract_w3c_carrier(headers_dict)

        span_name = f"{request.method} {request.url.path}"
        IN_FLIGHT_REQUESTS.inc()
        t0 = time.perf_counter()

        with tracer.start_as_current_span(
            span_name,
            context=extracted_ctx,
            kind=SpanKind.SERVER
        ) as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.target", request.url.path)
            if request.client:
                span.set_attribute("http.client_ip", request.client.host)

            trace_id = format(span.get_span_context().trace_id, "032x")

            try:
                response = await call_next(request)
                span.set_attribute("http.status_code", response.status_code)
                if response.status_code >= 500:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                else:
                    span.set_status(Status(StatusCode.OK))
                
                # Expose trace ID for client log correlation
                response.headers["X-Trace-Id"] = trace_id
                return response
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                IN_FLIGHT_REQUESTS.dec()
                duration_ms = (time.perf_counter() - t0) * 1000.0
                span.set_attribute("http.server.duration_ms", duration_ms)
