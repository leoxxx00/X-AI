from app.db.session import SessionLocal
from app.db.models import Job
from app.services.queue import enqueue_job


def create_job(
    url: str,
    strictness: str,
    auto_mode: bool,
    requested_pairs: int,
    evaluation: dict | None = None,
) -> Job:
    db = SessionLocal()
    try:
        job = Job(
            url=url,
            strictness=strictness,
            auto_mode=1 if auto_mode else 0,
            requested_pairs=requested_pairs,
            status="queued",
            progress=0,
            step="queued",
            capacity_json=evaluation,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        enqueue_job(job.id)
        return job
    finally:
        db.close()


def get_job(job_id: str) -> Job | None:
    db = SessionLocal()
    try:
        return db.get(Job, job_id)
    finally:
        db.close()