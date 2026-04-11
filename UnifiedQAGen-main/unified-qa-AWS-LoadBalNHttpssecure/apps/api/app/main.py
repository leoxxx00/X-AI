from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.base import Base
from app.db.session import engine
from app.routers.health import router as health_router
from app.routers.jobs import router as jobs_router
from app.routers.evaluator import router as evaluator_router

setup_logging(settings.LOG_LEVEL)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Unified QA API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(jobs_router)
app.include_router(evaluator_router)