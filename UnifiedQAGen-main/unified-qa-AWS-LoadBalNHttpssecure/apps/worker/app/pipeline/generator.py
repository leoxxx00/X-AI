from openai import OpenAI
from app.core.config import settings
from app.pipeline.rag import similarity_docs

import json

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
    return _client


def llm_json(prompt: str) -> dict:
    client = get_client()

    response = client.chat.completions.create(
        model=settings.DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return valid JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
        stream=False,
    )

    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def build_seed_queries(title: str) -> list[str]:
    topic = (title or "").strip()

    templates = [
        "{topic} overview, key concepts, and essential facts",
        "{topic} definitions, terminology, and meaning",
        "{topic} core components, structure, and main parts",
        "{topic} types, categories, and classifications",
        "{topic} features, functions, and roles",
        "{topic} how it works, mechanism, and process",
        "{topic} causes, effects, and relationships",
        "{topic} comparisons, distinctions, and contrasts",
        "{topic} step by step process and workflow",
        "{topic} requirements, conditions, and prerequisites",
        "{topic} best practices, recommendations, and guidance",
        "{topic} common problems, errors, and troubleshooting",
        "{topic} numbers, dates, measurements, and statistics",
        "{topic} examples, applications, and use cases",
        "{topic} benefits, strengths, limitations, and risks",
        "{topic} historical background, origin, and development",
    ]

    seen: set[str] = set()
    queries: list[str] = []

    for template in templates:
        query = template.format(topic=topic).strip()
        normalized = query.lower()

        if query and normalized not in seen:
            seen.add(normalized)
            queries.append(query)

    return queries


def generate_pairs(vs, title: str, target_pairs: int) -> list[dict]:
    pairs: list[dict] = []
    seen_questions: set[str] = set()

    if target_pairs <= 0:
        return pairs

    for seed in build_seed_queries(title):
        remaining = target_pairs - len(pairs)
        if remaining <= 0:
            break

        docs = similarity_docs(vs, seed, k=5)
        if not docs:
            continue

        context_text = "\n\n".join(
            f"[CHUNK {doc['chunk_index']}] {doc['text']}" for doc in docs
        )

        sources = sorted(
            {
                doc.get("source_url", "").strip()
                for doc in docs
                if doc.get("source_url", "").strip()
            }
        )

        prompt = f"""
Return valid JSON:
{{
  "items": [
    {{
      "question": "...",
      "answer": "...",
      "context": "...",
      "source": "...",
      "type": "definition|steps|comparison|cause_effect|numbers_dates|how_it_works|examples|other",
      "quality_score": 8
    }}
  ]
}}

TITLE:
{title}

CONTEXT:
\"\"\"{context_text}\"\"\"

RULES:
- Create up to {min(5, remaining)} grounded Q/A pairs.
- Every answer must be directly supported by the provided context.
- Do not invent facts.
- Keep questions specific and non-duplicate.
- Keep answers concise but complete.
- The "context" field should contain a short supporting excerpt or summary from the retrieved chunks.
- The "source" field should be the most relevant source URL when available.
"""

        data = llm_json(prompt)

        for item in data.get("items", []):
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            context = item.get("context", "").strip()
            source = item.get("source", "").strip()

            normalized_question = " ".join(question.lower().split())

            if normalized_question in seen_questions:
                continue

            if len(question) <= 10 or len(answer) <= 20:
                continue

            if not source and sources:
                source = sources[0]

            pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "context": context,
                    "source": source,
                    "type": item.get("type", "other"),
                    "quality_score": item.get("quality_score", 8),
                }
            )
            seen_questions.add(normalized_question)

            if len(pairs) >= target_pairs:
                return pairs

    return pairs