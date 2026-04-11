import json
import logging
import time
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.logging import setup_logging
from app.pipeline.engine import run_job_pipeline

setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger("worker")

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
QUEUE_NAME = "unifiedqa:jobs"


def process_job(job_id: str):
    from sqlalchemy import text

    db = SessionLocal()
    try:
        row = db.execute(
            text(
                "SELECT id, url, strictness, auto_mode, requested_pairs, capacity_json "
                "FROM jobs WHERE id = :id"
            ),
            {"id": job_id},
        ).mappings().first()

        if not row:
            logger.warning("Job not found", extra={"job_id": job_id})
            return

        db.execute(
            text("UPDATE jobs SET status='running', progress=10, step='extracting' WHERE id=:id"),
            {"id": job_id},
        )
        db.commit()

        result = run_job_pipeline(
            job_id=job_id,
            url=row["url"],
            strictness=row["strictness"],
            auto_mode=bool(row["auto_mode"]),
            requested_pairs=row["requested_pairs"],
            capacity=row["capacity_json"],
        )

        db.execute(
            text(
                """
                UPDATE jobs
                SET status='completed',
                    progress=100,
                    step='done',
                    summary=:summary,
                    capacity_json=:capacity,
                    metrics_json=:metrics,
                    results_json=:results,
                    artifacts_json=:artifacts,
                    error_message=NULL
                WHERE id=:id
                """
            ),
            {
                "id": job_id,
                "summary": result["summary"],
                "capacity": json.dumps(result["capacity"]),
                "metrics": json.dumps(result["metrics"]),
                "results": json.dumps(result["accepted_pairs"]),
                "artifacts": json.dumps(result["artifacts"]),
            },
        )
        db.commit()

        logger.info("Job completed", extra={"job_id": job_id})

    except Exception as e:
        db.rollback()
        db.execute(
            text(
                """
                UPDATE jobs
                SET status='failed',
                    progress=100,
                    step='failed',
                    error_message=:error
                WHERE id=:id
                """
            ),
            {"id": job_id, "error": str(e)},
        )
        db.commit()
        logger.exception("Job failed", extra={"job_id": job_id})
    finally:
        db.close()


def main():
    logger.info("Worker started")
    while True:
        item = r.lpop(QUEUE_NAME)
        if not item:
            time.sleep(settings.WORKER_POLL_SECONDS)
            continue

        payload = json.loads(item)
        process_job(payload["job_id"])


if __name__ == "__main__":
    main()