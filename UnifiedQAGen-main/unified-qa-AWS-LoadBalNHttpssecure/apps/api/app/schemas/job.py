from pydantic import BaseModel, Field, HttpUrl
from typing import Any


class EvaluationBounds(BaseModel):
    training_grade_pairs: int = Field(ge=1)
    raw_extractable_pairs: int = Field(ge=1)


class CreateJobRequest(BaseModel):
    url: HttpUrl
    strictness: str = Field(default="Standard")
    auto_mode: bool = Field(default=True)
    requested_pairs: int = Field(default=10, ge=1)
    evaluation: EvaluationBounds | None = None


class CreateJobResponse(BaseModel):
    job_id: str
    status: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    step: str
    summary: str | None = None
    error_message: str | None = None


class JobResultsResponse(BaseModel):
    job_id: str
    status: str
    summary: str | None = None
    capacity: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    accepted_pairs: list[dict[str, Any]] | None = None
    artifacts: dict[str, Any] | None = None