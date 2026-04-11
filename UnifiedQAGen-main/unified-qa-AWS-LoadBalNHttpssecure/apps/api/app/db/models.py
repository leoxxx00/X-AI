import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    url: Mapped[str] = mapped_column(Text, nullable=False)
    strictness: Mapped[str] = mapped_column(String(32), default="Standard")
    auto_mode: Mapped[int] = mapped_column(Integer, default=1)
    requested_pairs: Mapped[int] = mapped_column(Integer, default=10)

    status: Mapped[str] = mapped_column(String(32), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    step: Mapped[str] = mapped_column(String(64), default="queued")

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    capacity_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    results_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    artifacts_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)