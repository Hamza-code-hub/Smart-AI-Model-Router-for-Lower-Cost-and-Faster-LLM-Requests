import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from gateway.breaker import get_breaker
from gateway.config import get_settings
from gateway.db import close_db, init_db
from gateway.embeddings import warmup as warmup_embeddings
from gateway.logging_setup import configure_logging
from gateway.observability import metrics as obs_metrics
from gateway.observability import tracing as obs_tracing
from gateway.providers import close_providers, init_providers
from gateway.redis_client import close_redis, init_redis
from gateway.routing import init_resolver
from gateway.routes.admin import router as admin_router
from gateway.routes.chat import router as chat_router
from gateway.routes.health import router as health_router
from gateway.routes.models import router as models_router
from gateway.routes.ui import router as ui_router

_STATIC_DIR = Path(__file__).parent / "static"

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(json_logs=settings.log_json, log_level=settings.log_level)
    obs_tracing.init(service_name="llm-gateway", service_version=settings.app_version)
    obs_metrics.attach_breaker(get_breaker())
    await init_db()
    await init_redis()
    init_providers()
    init_resolver()
    await warmup_embeddings()
    logger.info("gateway.started", version=settings.app_version)
    yield
    await close_providers()
    await close_db()
    await close_redis()
    logger.info("gateway.stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="llm-gateway",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}},
        )

    app.include_router(health_router)
    app.include_router(chat_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(admin_router, prefix="/admin")
    app.include_router(ui_router)

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/chat", status_code=302)

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        body, content_type = obs_metrics.render_latest()
        return Response(content=body, media_type=content_type)

    obs_tracing.instrument_app(app)

    return app


app = create_app()
