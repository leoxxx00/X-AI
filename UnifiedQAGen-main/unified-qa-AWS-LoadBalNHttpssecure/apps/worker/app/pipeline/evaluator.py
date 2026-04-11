import re


SYSTEM_MAX_PAIRS = 500


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'\-]{1,}", text.lower())


def split_sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]


def lexical_diversity(words: list[str]) -> float:
    return round(len(set(words)) / len(words), 4) if words else 0.0


def sentence_diversity(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    norm = [re.sub(r"\W+", " ", s.lower()).strip() for s in sentences]
    return round(len(set(norm)) / len(norm), 4)


def count_fact_like_sentences(sentences: list[str]) -> int:
    count = 0
    for s in sentences:
        if len(s) < 35:
            continue
        if re.search(
            r"\b(is|are|was|were|refers to|defined as|used to|includes|contains|consists of|can be|means)\b",
            s,
            flags=re.I,
        ):
            count += 1
            continue
        if re.search(r"\b\d+(?:\.\d+)?\b", s):
            count += 1
    return count


def compute_metrics_simple(url: str, title: str, text: str) -> dict:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    fact_like_total = count_fact_like_sentences(sentences)

    return {
        "url": url,
        "title": title,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "lexical_diversity": lexical_diversity(words),
        "sentence_diversity": sentence_diversity(sentences),
        "fact_like_total": fact_like_total,
        "fact_density": round(fact_like_total / len(sentences), 4) if sentences else 0.0,
        "text_length_chars": len(text),
    }


def estimate_capacity_simple(metrics: dict, strictness: str = "Standard") -> dict:
    words = int(metrics["word_count"])
    sentences = int(metrics["sentence_count"])
    fact_like = int(metrics["fact_like_total"])
    lexical = float(metrics.get("lexical_diversity", 0.0))
    sentence_div = float(metrics.get("sentence_diversity", 0.0))
    fact_density = float(metrics.get("fact_density", 0.0))

    by_words = words // 10
    by_sentences = sentences // 1
    by_facts = fact_like
    by_density_bonus = round(fact_density * 20)

    weighted_base = round(
        (by_words * 0.35)
        + (by_sentences * 0.25)
        + (by_facts * 0.30)
        + (by_density_bonus * 0.10)
    )

    diversity_bonus = 0
    if lexical >= 0.45:
        diversity_bonus += 2
    if lexical >= 0.55:
        diversity_bonus += 2
    if sentence_div >= 0.80:
        diversity_bonus += 2
    if sentence_div >= 0.90:
        diversity_bonus += 2

    base = max(5, weighted_base + diversity_bonus)
    base = min(base, SYSTEM_MAX_PAIRS)

    if strictness == "Strict":
        base = max(1, round(base * 0.92))
    elif strictness == "Lenient":
        base = min(SYSTEM_MAX_PAIRS, round(base * 1.18))
    else:
        base = min(SYSTEM_MAX_PAIRS, round(base * 1.08))

    raw_extractable = min(SYSTEM_MAX_PAIRS, round(base * 1.35 + 4))
    predicted_min = max(1, round(base * 0.88))
    predicted_max = min(SYSTEM_MAX_PAIRS, round(base * 1.15 + 3))

    confidence = 0.68
    if fact_density >= 0.35:
        confidence += 0.04
    if lexical >= 0.45:
        confidence += 0.03
    if sentence_div >= 0.85:
        confidence += 0.03

    confidence = min(0.85, round(confidence, 2))

    return {
        "training_grade_pairs": int(base),
        "raw_extractable_pairs": int(raw_extractable),
        "predicted_min": int(predicted_min),
        "predicted_max": int(predicted_max),
        "confidence": confidence,
        "method": "weighted_quality_capacity_estimator_v2",
    }