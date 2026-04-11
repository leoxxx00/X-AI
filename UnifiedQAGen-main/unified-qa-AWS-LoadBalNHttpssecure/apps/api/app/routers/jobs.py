import csv
import io
import re
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas.job import (
    CreateJobRequest,
    CreateJobResponse,
    JobResponse,
    JobResultsResponse,
)
from app.services.jobs import create_job, get_job

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def validate_requested_pairs(
    requested_pairs: int,
    training_grade_pairs: int,
    raw_extractable_pairs: int,
) -> None:
    if requested_pairs < training_grade_pairs or requested_pairs > raw_extractable_pairs:
        raise ValueError(
            f"requested_pairs must be between {training_grade_pairs} and {raw_extractable_pairs}"
        )


def build_results_payload(job):
    return {
        "job_id": job.id,
        "status": job.status,
        "summary": job.summary,
        "capacity": job.capacity_json,
        "metrics": job.metrics_json,
        "accepted_pairs": job.results_json or [],
        "artifacts": job.artifacts_json,
    }


def build_csv_rows(job):
    rows = job.results_json or []
    normalized_rows = []

    for row in rows:
        normalized_rows.append(
            {
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "context": row.get("context", ""),
                "source": row.get("source", ""),
            }
        )

    return normalized_rows


def slugify_filename(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "job"


def derive_job_name(job) -> str:
    """
    Prefer a human-friendly name for downloads.

    Order:
    1. job.summary (if present)
    2. hostname from job.url
    3. fallback 'job'
    """
    if getattr(job, "summary", None):
        summary = str(job.summary).strip()
        if summary:
            return slugify_filename(summary[:80])

    if getattr(job, "url", None):
        try:
            parsed = urlparse(str(job.url))
            host = parsed.netloc.replace("www.", "").strip()
            path = parsed.path.strip("/").replace("/", "-")
            combined = f"{host}-{path}" if path else host
            if combined:
                return slugify_filename(combined[:80])
        except Exception:
            pass

    return "job"


def build_download_filename(job, ext: str) -> str:
    base_name = derive_job_name(job)
    short_job_id = str(job.id)[:8]
    return f"{base_name}_{short_job_id}_results.{ext}"


@router.post("", response_model=CreateJobResponse)
def create_job_route(payload: CreateJobRequest):
    try:
        if payload.evaluation is not None:
            validate_requested_pairs(
                payload.requested_pairs,
                payload.evaluation.training_grade_pairs,
                payload.evaluation.raw_extractable_pairs,
            )

        job = create_job(
            url=str(payload.url),
            strictness=payload.strictness,
            auto_mode=payload.auto_mode,
            requested_pairs=payload.requested_pairs,
            evaluation=payload.evaluation.model_dump() if payload.evaluation else None,
        )
        return CreateJobResponse(job_id=job.id, status=job.status)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{job_id}", response_model=JobResponse)
def get_job_route(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        step=job.step,
        summary=job.summary,
        error_message=job.error_message,
    )


@router.get("/{job_id}/results", response_model=JobResultsResponse)
def get_job_results_route(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResultsResponse(
        job_id=job.id,
        status=job.status,
        summary=job.summary,
        capacity=job.capacity_json,
        metrics=job.metrics_json,
        accepted_pairs=job.results_json,
        artifacts=job.artifacts_json,
    )


@router.get("/{job_id}/results.json")
def download_results_json(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    payload = build_results_payload(job)
    filename = build_download_filename(job, "json")

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }

    return JSONResponse(content=payload, headers=headers)


@router.get("/{job_id}/results.csv")
def download_results_csv(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    rows = build_csv_rows(job)

    output = io.StringIO()
    fieldnames = ["question", "answer", "context", "source"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for row in rows:
        writer.writerow(row)

    output.seek(0)
    filename = build_download_filename(job, "csv")

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers=headers,
    )