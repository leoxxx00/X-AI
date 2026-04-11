import os
import io
import csv
import json
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from app.core.config import settings


def to_json(job_id: str, results: List[Dict[str, Any]]) -> str:
    return json.dumps(
        {
            "job_id": job_id,
            "results": results,
        },
        ensure_ascii=False,
        indent=2,
    )


def to_csv(results: List[Dict[str, Any]]) -> str:
    output = io.StringIO()
    fieldnames = ["question", "answer", "context", "source"]

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for row in results:
        writer.writerow(
            {
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "context": row.get("context", ""),
                "source": row.get("source", ""),
            }
        )

    return output.getvalue()


def save_outputs(job_id: str, input_url: str, qa_pairs: List[Dict[str, Any]]) -> Dict[str, str]:
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    base = f"{job_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    csv_path = os.path.join(settings.OUTPUT_DIR, f"{base}.csv")
    json_path = os.path.join(settings.OUTPUT_DIR, f"{base}.json")

    rows = [
        {
            "url": input_url,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "context": item.get("context", ""),
            "source": item.get("source", ""),
            "type": item.get("type", "other"),
            "quality_score": item.get("quality_score"),
        }
        for item in qa_pairs
    ]

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    return {"csv_path": csv_path, "json_path": json_path}