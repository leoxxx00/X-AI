from app.pipeline.extractor import extract_page
from app.pipeline.evaluator import compute_metrics_simple, estimate_capacity_simple
from app.pipeline.rag import build_vectorstore, add_to_rag
from app.pipeline.generator import generate_pairs
from app.pipeline.exporter import save_outputs


def run_job_pipeline(
    job_id: str,
    url: str,
    strictness: str,
    auto_mode: bool,
    requested_pairs: int,
    capacity: dict | None = None,
) -> dict:
    title, text = extract_page(url)

    metrics = compute_metrics_simple(url, title, text)

    if capacity is None:
        capacity = estimate_capacity_simple(metrics, strictness)

    raw_extractable_pairs = max(1, int(capacity["raw_extractable_pairs"]))
    training_grade_pairs = max(1, int(capacity["training_grade_pairs"]))

    target_pairs = training_grade_pairs if auto_mode else requested_pairs
    target_pairs = max(1, min(target_pairs, raw_extractable_pairs))

    vs = build_vectorstore(job_id)
    add_to_rag(vs, url, text)

    qa_pairs = generate_pairs(vs, title, target_pairs)
    artifacts = save_outputs(job_id, url, qa_pairs)

    summary = "\n".join(
        [
            f"URL: {url}",
            f"Title: {title}",
            f"Target pairs: {target_pairs}",
            f"Accepted pairs: {len(qa_pairs)}",
            f"CSV: {artifacts['csv_path']}",
            f"JSON: {artifacts['json_path']}",
        ]
    )

    return {
        "summary": summary,
        "capacity": capacity,
        "metrics": metrics,
        "accepted_pairs": qa_pairs,
        "artifacts": artifacts,
    }