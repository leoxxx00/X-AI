import json
import redis
from app.core.config import settings

QUEUE_NAME = "unifiedqa:jobs"
r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def enqueue_job(job_id: str) -> None:
    r.rpush(QUEUE_NAME, json.dumps({"job_id": job_id}))


def queue_depth() -> int:
    return int(r.llen(QUEUE_NAME))