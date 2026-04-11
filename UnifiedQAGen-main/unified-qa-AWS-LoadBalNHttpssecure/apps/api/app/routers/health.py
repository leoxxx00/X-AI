from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from app.services.queue import queue_depth

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    return {"status": "ready", "queue_depth": queue_depth()}


@router.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)