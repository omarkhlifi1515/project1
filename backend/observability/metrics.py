from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

REQUEST_COUNT = Counter("http_requests_total", "Total number of HTTP requests", ["path", "method"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP latency", ["path"])


def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type="text/plain")
