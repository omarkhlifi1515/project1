from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import admin, auth, chat, health
from backend.config.settings import get_settings
from backend.db.base import Base, engine
from backend.observability.logging import configure_logging
from backend.observability.metrics import REQUEST_COUNT, REQUEST_LATENCY, metrics_endpoint

settings = get_settings()
configure_logging()
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(chat.router, prefix=settings.api_prefix)
app.include_router(admin.router, prefix=settings.api_prefix)


@app.get("/")
def root():
    return {"service": settings.app_name, "status": "running"}


@app.middleware("http")
async def collect_metrics(request: Request, call_next):
    path = request.url.path
    method = request.method
    REQUEST_COUNT.labels(path=path, method=method).inc()
    with REQUEST_LATENCY.labels(path=path).time():
        response = await call_next(request)
    return response


@app.get("/metrics")
def metrics():
    return metrics_endpoint()
