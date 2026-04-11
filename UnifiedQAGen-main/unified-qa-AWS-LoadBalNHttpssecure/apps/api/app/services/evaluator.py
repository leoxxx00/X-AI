import os
import re
import json
import math
import time
from collections import Counter
from html import unescape
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse

import requests
import trafilatura
from openai import OpenAI, AuthenticationError


USER_AGENT = "Mozilla/5.0 (compatible; QA-Capacity-Evaluator/7.0-open-world)"
DEFAULT_TIMEOUT = 25

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

MIN_TEXT_LENGTH = 300
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
LLM_SAMPLE_CHARS = 7000


SPECIALIST_PROFILES: Dict[Tuple[str, str], Dict[str, float]] = {
    ("encyclopedia", "explainer"): {
        "type_multiplier": 1.08,
        "cap_multiplier": 1.05,
        "fact_bonus_multiplier": 1.06,
        "list_floor_multiplier": 0.28,
        "depth_bonus": 1.04,
    },
    ("encyclopedia", "reference"): {
        "type_multiplier": 1.05,
        "cap_multiplier": 1.02,
        "fact_bonus_multiplier": 1.05,
        "list_floor_multiplier": 0.26,
        "depth_bonus": 1.03,
    },
    ("medical_info", "explainer"): {
        "type_multiplier": 1.00,
        "cap_multiplier": 0.96,
        "fact_bonus_multiplier": 1.10,
        "list_floor_multiplier": 0.42,
        "depth_bonus": 1.03,
    },
    ("engineering", "explainer"): {
        "type_multiplier": 1.03,
        "cap_multiplier": 0.99,
        "fact_bonus_multiplier": 1.06,
        "list_floor_multiplier": 0.32,
        "depth_bonus": 1.02,
    },
    ("documentation", "reference"): {
        "type_multiplier": 1.01,
        "cap_multiplier": 0.98,
        "fact_bonus_multiplier": 1.03,
        "list_floor_multiplier": 0.48,
        "depth_bonus": 1.02,
    },
    ("documentation", "explainer"): {
        "type_multiplier": 0.98,
        "cap_multiplier": 0.96,
        "fact_bonus_multiplier": 1.00,
        "list_floor_multiplier": 0.44,
        "depth_bonus": 1.01,
    },
    ("documentation", "article"): {
        "type_multiplier": 0.94,
        "cap_multiplier": 0.92,
        "fact_bonus_multiplier": 0.96,
        "list_floor_multiplier": 0.30,
        "depth_bonus": 0.98,
    },
    ("research", "paper"): {
        "type_multiplier": 1.02,
        "cap_multiplier": 0.99,
        "fact_bonus_multiplier": 1.08,
        "list_floor_multiplier": 0.22,
        "depth_bonus": 1.05,
    },
    ("government_policy", "policy"): {
        "type_multiplier": 0.92,
        "cap_multiplier": 0.90,
        "fact_bonus_multiplier": 0.98,
        "list_floor_multiplier": 0.24,
        "depth_bonus": 0.98,
    },
    ("government_policy", "listing"): {
        "type_multiplier": 0.70,
        "cap_multiplier": 0.68,
        "fact_bonus_multiplier": 0.70,
        "list_floor_multiplier": 0.12,
        "depth_bonus": 0.86,
    },
    ("legal", "policy"): {
        "type_multiplier": 0.86,
        "cap_multiplier": 0.84,
        "fact_bonus_multiplier": 0.92,
        "list_floor_multiplier": 0.22,
        "depth_bonus": 0.96,
    },
    ("legal", "listing"): {
        "type_multiplier": 0.68,
        "cap_multiplier": 0.66,
        "fact_bonus_multiplier": 0.68,
        "list_floor_multiplier": 0.10,
        "depth_bonus": 0.82,
    },
    ("company_info", "overview"): {
        "type_multiplier": 0.76,
        "cap_multiplier": 0.74,
        "fact_bonus_multiplier": 0.72,
        "list_floor_multiplier": 0.18,
        "depth_bonus": 0.90,
    },
    ("finance_business", "report"): {
        "type_multiplier": 0.92,
        "cap_multiplier": 0.90,
        "fact_bonus_multiplier": 0.98,
        "list_floor_multiplier": 0.22,
        "depth_bonus": 0.98,
    },
    ("news", "article"): {
        "type_multiplier": 0.82,
        "cap_multiplier": 0.82,
        "fact_bonus_multiplier": 0.84,
        "list_floor_multiplier": 0.14,
        "depth_bonus": 0.92,
    },
    ("blog", "article"): {
        "type_multiplier": 0.86,
        "cap_multiplier": 0.85,
        "fact_bonus_multiplier": 0.84,
        "list_floor_multiplier": 0.18,
        "depth_bonus": 0.92,
    },
    ("faq_help", "faq"): {
        "type_multiplier": 0.98,
        "cap_multiplier": 0.95,
        "fact_bonus_multiplier": 1.00,
        "list_floor_multiplier": 0.44,
        "depth_bonus": 0.98,
    },
    ("faq_help", "explainer"): {
        "type_multiplier": 0.96,
        "cap_multiplier": 0.94,
        "fact_bonus_multiplier": 0.98,
        "list_floor_multiplier": 0.40,
        "depth_bonus": 0.98,
    },
    ("product_page", "spec_sheet"): {
        "type_multiplier": 0.62,
        "cap_multiplier": 0.60,
        "fact_bonus_multiplier": 0.66,
        "list_floor_multiplier": 0.16,
        "depth_bonus": 0.82,
    },
    ("educational_article", "explainer"): {
        "type_multiplier": 0.99,
        "cap_multiplier": 0.96,
        "fact_bonus_multiplier": 1.00,
        "list_floor_multiplier": 0.30,
        "depth_bonus": 1.00,
    },
    ("general_article", "explainer"): {
        "type_multiplier": 0.92,
        "cap_multiplier": 0.90,
        "fact_bonus_multiplier": 0.94,
        "list_floor_multiplier": 0.24,
        "depth_bonus": 0.96,
    },
    ("general_article", "reference"): {
        "type_multiplier": 0.94,
        "cap_multiplier": 0.92,
        "fact_bonus_multiplier": 0.96,
        "list_floor_multiplier": 0.24,
        "depth_bonus": 0.98,
    },
    ("general_article", "overview"): {
        "type_multiplier": 0.82,
        "cap_multiplier": 0.80,
        "fact_bonus_multiplier": 0.84,
        "list_floor_multiplier": 0.18,
        "depth_bonus": 0.90,
    },
    ("general_article", "article"): {
        "type_multiplier": 0.88,
        "cap_multiplier": 0.86,
        "fact_bonus_multiplier": 0.90,
        "list_floor_multiplier": 0.18,
        "depth_bonus": 0.94,
    },
    ("general_article", "listing"): {
        "type_multiplier": 0.72,
        "cap_multiplier": 0.70,
        "fact_bonus_multiplier": 0.72,
        "list_floor_multiplier": 0.10,
        "depth_bonus": 0.82,
    },
    ("unknown", "mixed"): {
        "type_multiplier": 0.86,
        "cap_multiplier": 0.84,
        "fact_bonus_multiplier": 0.88,
        "list_floor_multiplier": 0.20,
        "depth_bonus": 0.92,
    },
}

DEFAULT_PROFILE = SPECIALIST_PROFILES[("unknown", "mixed")]

OPEN_DOMAIN_TO_PROFILE = {
    "reference_knowledge": "encyclopedia",
    "health_medical": "medical_info",
    "technical_docs": "documentation",
    "scientific_research": "research",
    "government_public_sector": "government_policy",
    "legal_compliance": "legal",
    "company_marketing": "company_info",
    "finance_investor": "finance_business",
    "news_media": "news",
    "blog_editorial": "blog",
    "support_help": "faq_help",
    "commerce_product": "product_page",
    "education_learning": "educational_article",
    "general_information": "general_article",
    "community_forum": "blog",
    "directory_listing": "general_article",
    "dataset_catalog": "documentation",
    "media_entertainment": "news",
}

OPEN_FORM_TO_PROFILE = {
    "reference": "reference",
    "explainer": "explainer",
    "paper": "paper",
    "policy": "policy",
    "listing": "listing",
    "overview": "overview",
    "faq": "faq",
    "spec_sheet": "spec_sheet",
    "article": "article",
    "forum_thread": "article",
    "dataset": "reference",
    "changelog": "article",
    "mixed": "mixed",
    "report": "report",
}


def is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", (text or "").strip(), re.IGNORECASE))


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def dedupe_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        item_n = normalize_spaces(item)
        key = item_n.lower()
        if item_n and key not in seen:
            seen.add(key)
            out.append(item_n)
    return out


def count_regex_hits(text: str, patterns: List[str]) -> int:
    total = 0
    for p in patterns:
        total += len(re.findall(p, text, flags=re.I))
    return total


def fetch_html(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    response = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response.text


def extract_main_text(html: str, url: Optional[str] = None) -> str:
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    return text or ""


def extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return normalize_spaces(unescape(match.group(1)))


def extract_html_headings(html: str) -> List[str]:
    headings = re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = []
    for h in headings:
        h = re.sub(r"<[^>]+>", " ", h)
        h = normalize_spaces(unescape(h))
        if h and len(h) >= 2:
            cleaned.append(h)
    return cleaned


def extract_meta_tags(html: str) -> Dict[str, str]:
    tags = {}

    patterns = [
        (r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', "description"),
        (r'<meta[^>]+property=["\']og:type["\'][^>]+content=["\'](.*?)["\']', "og_type"),
        (r'<meta[^>]+property=["\']og:site_name["\'][^>]+content=["\'](.*?)["\']', "og_site_name"),
        (r'<meta[^>]+name=["\']twitter:card["\'][^>]+content=["\'](.*?)["\']', "twitter_card"),
        (r'<meta[^>]+name=["\']keywords["\'][^>]+content=["\'](.*?)["\']', "keywords"),
        (r'<meta[^>]+property=["\']article:section["\'][^>]+content=["\'](.*?)["\']', "article_section"),
        (r'<meta[^>]+property=["\']article:tag["\'][^>]+content=["\'](.*?)["\']', "article_tag"),
    ]

    for pattern, key in patterns:
        m = re.search(pattern, html, flags=re.I | re.S)
        if m:
            tags[key] = normalize_spaces(unescape(m.group(1)))

    return tags


def get_url_features(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    host_parts = [p for p in host.split(".") if p]
    tld = host_parts[-1] if host_parts else ""
    sld = ".".join(host_parts[-2:]) if len(host_parts) >= 2 else host
    subdomains = host_parts[:-2] if len(host_parts) >= 2 else []
    path_parts = [p for p in path.split("/") if p]

    return {
        "host": host,
        "tld": tld,
        "sld": sld,
        "subdomains": subdomains,
        "path": path,
        "path_parts": path_parts,
    }


def clean_extracted_text(text: str) -> str:
    text = unescape(text or "")
    text = text.replace("\xa0", " ")

    text = re.sub(r"\[(?:\d+|citation needed|note \d+|clarification needed)\]", " ", text, flags=re.I)
    text = re.sub(r"\bedit\b", " ", text, flags=re.I)
    text = re.sub(r"\bjump to\b", " ", text, flags=re.I)

    tail_markers = [
        r"\bReferences\b",
        r"\bExternal links\b",
        r"\bFurther reading\b",
        r"\bSee also\b",
        r"\bNotes\b",
        r"\bCitations\b",
        r"\bBibliography\b",
        r"\bFootnotes\b",
    ]
    for marker in tail_markers:
        m = re.search(marker, text, flags=re.I)
        if m:
            after = text[m.start():]
            if len(after) < max(1400, int(len(text) * 0.22)):
                text = text[:m.start()]
                break

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return []

    protected = text
    protected = re.sub(r"\b([A-Z][a-z]{0,4})\.", r"\1<prd>", protected)
    protected = re.sub(
        r"\b(e\.g|i\.e|vs|etc)\.",
        lambda m: m.group(0).replace(".", "<prd>"),
        protected,
        flags=re.I,
    )
    protected = re.sub(r"(\d)\.(\d)", r"\1<prd>\2", protected)

    parts = re.split(r"(?<=[.!?])\s+", protected)
    out = []
    for p in parts:
        p = p.replace("<prd>", ".").strip()
        if p:
            out.append(p)
    return out


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z'\-]{1,}", (text or "").lower())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            window = text[start:end]
            last_break = max(window.rfind(". "), window.rfind("? "), window.rfind("! "), window.rfind("\n"))
            if last_break > int(chunk_size * 0.6):
                end = start + last_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def split_into_sections(headings: List[str], text: str) -> List[Dict[str, str]]:
    if not headings:
        return [{"heading": "Main", "text": text[:]}]

    text_norm = text
    sections = []
    used_ranges = []

    for i, heading in enumerate(headings[:25]):
        pattern = re.escape(heading)
        match = re.search(pattern, text_norm, flags=re.I)
        if not match:
            continue

        start = match.start()
        next_start = len(text_norm)
        for next_heading in headings[i + 1:]:
            m2 = re.search(re.escape(next_heading), text_norm[start + len(heading):], flags=re.I)
            if m2:
                next_start = start + len(heading) + m2.start()
                break

        if any(start >= a and start < b for a, b in used_ranges):
            continue

        body = text_norm[start:next_start].strip()
        if len(body) >= 80:
            sections.append({"heading": heading, "text": body})
            used_ranges.append((start, next_start))

    if not sections:
        sections = [{"heading": "Main", "text": text[:]}]

    return sections[:20]


def sentence_diversity(sentences: List[str]) -> float:
    if not sentences:
        return 0.0
    normalized = []
    for s in sentences:
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            normalized.append(s)
    if not normalized:
        return 0.0
    return round(len(set(normalized)) / len(normalized), 4)


def lexical_diversity(words: List[str]) -> float:
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def average_sentence_length(words: List[str], sentences: List[str]) -> float:
    if not sentences:
        return 0.0
    return round(len(words) / max(len(sentences), 1), 2)


def entropy_score(words: List[str]) -> float:
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return round(entropy, 4)


def count_numeric_tokens(text: str) -> int:
    return len(re.findall(r"\b\d+(?:[\.,]\d+)?\b", text or ""))


def count_definition_patterns(text: str) -> int:
    patterns = [
        r"\bis defined as\b", r"\brefers to\b", r"\bmeans\b", r"\bis called\b", r"\bis known as\b",
        r"\bcan be defined as\b", r"\bis a test that\b", r"\bis used to\b", r"\bconsists of\b",
        r"\bincludes\b", r"\bdescribes\b",
    ]
    return sum(len(re.findall(p, text or "", flags=re.I)) for p in patterns)


def count_comparison_patterns(text: str) -> int:
    patterns = [
        r"\bcompared to\b", r"\bin contrast\b", r"\bwhereas\b", r"\bhowever\b",
        r"\bon the other hand\b", r"\bdifferent from\b", r"\bsimilar to\b", r"\bversus\b",
    ]
    return sum(len(re.findall(p, text or "", flags=re.I)) for p in patterns)


def count_causal_patterns(text: str) -> int:
    patterns = [
        r"\bbecause\b", r"\bdue to\b", r"\bresults in\b", r"\bleads to\b", r"\bcauses\b",
        r"\btherefore\b", r"\bas a result\b", r"\bso that\b", r"\bdriven by\b",
    ]
    return sum(len(re.findall(p, text or "", flags=re.I)) for p in patterns)


def count_list_signals(text: str) -> int:
    count = 0
    for line in (text or "").splitlines():
        if re.match(r"^\s*(\-|\*|\d+\.)\s+", line):
            count += 1
    return count


def count_fact_like_list_items(text: str) -> int:
    count = 0
    for line in (text or "").splitlines():
        line = line.strip()
        if re.match(r"^(\-|\*|\d+\.)\s+", line):
            content = re.sub(r"^(\-|\*|\d+\.)\s+", "", line).strip()
            if len(content) >= 20:
                count += 1
    return count


def count_long_sentences(sentences: List[str], min_words: int = 12) -> int:
    return sum(1 for s in sentences if len(tokenize_words(s)) >= min_words)


def count_fact_like_sentences(sentences: List[str]) -> int:
    count = 0
    for s in sentences:
        s2 = s.strip()
        if len(s2) < 35:
            continue

        if re.search(
            r"\b(is|are|was|were|refers to|defined as|used to|consists of|includes|contains|called|known as|describes)\b",
            s2,
            flags=re.I,
        ):
            count += 1
            continue

        if re.search(r"\b\d+(?:\.\d+)?\b", s2):
            count += 1
            continue

        if re.search(r"\b(can|may|often|typically|usually|commonly|must|should)\b", s2, flags=re.I):
            count += 1
    return count


def average_chunk_uniqueness(chunks: List[str]) -> float:
    if not chunks:
        return 0.0
    signatures = []
    for c in chunks:
        words = tokenize_words(c)
        sig = " ".join(sorted(set(words[:80])))
        signatures.append(sig)
    if not signatures:
        return 0.0
    return round(len(set(signatures)) / len(signatures), 4)


def clean_headings_for_article(html: str, text: str) -> List[str]:
    raw = extract_html_headings(html)
    text_lower = (text or "").lower()
    cleaned = []
    for h in raw:
        h_norm = normalize_spaces(h)
        if len(h_norm) < 2:
            continue
        if h_norm.lower() in {"contents", "navigation menu", "menu"}:
            continue
        if h_norm.lower() in text_lower or len(h_norm.split()) >= 2:
            cleaned.append(h_norm)
    return dedupe_keep_order(cleaned)


def repetition_penalty(lexical_div: float, sentence_div: float, chunk_uniqueness: float) -> float:
    penalty = 1.0
    if lexical_div < 0.22:
        penalty -= 0.10
    elif lexical_div < 0.28:
        penalty -= 0.05

    if sentence_div < 0.75:
        penalty -= 0.10
    elif sentence_div < 0.85:
        penalty -= 0.04

    if chunk_uniqueness < 0.70:
        penalty -= 0.10
    elif chunk_uniqueness < 0.85:
        penalty -= 0.04

    return round(max(0.72, penalty), 4)


def detect_catalog_page(text: str, headings: List[str], sentences: List[str]) -> Dict[str, Any]:
    short_sentences = sum(1 for s in sentences if len(tokenize_words(s)) < 8)
    date_count = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    heading_density = len(headings) / max(len(sentences), 1)
    repeated_short = 0
    for s in sentences[:300]:
        if re.search(r"\b(regulations?|order|act|notice|update|article|press release|result|listing)\b", s, flags=re.I):
            repeated_short += 1

    score = 0.0
    if sentences:
        if short_sentences > len(sentences) * 0.35:
            score += 0.35
        if heading_density > 0.18:
            score += 0.30
        if date_count > 8:
            score += 0.20
        if repeated_short > max(6, len(sentences) * 0.10):
            score += 0.25

    score = min(1.0, score)

    return {
        "catalog_score": round(score, 3),
        "is_catalog_like": score >= 0.45,
        "catalog_penalty": round(max(0.60, 1.0 - (score * 0.28)), 3),
        "short_sentence_ratio": round(short_sentences / max(len(sentences), 1), 3),
        "date_count": date_count,
        "heading_density": round(heading_density, 3),
    }


def detect_domain_and_form(
    url: str,
    title: str,
    text: str,
    headings: List[str],
    catalog_info: Dict[str, Any],
    html: str = "",
) -> Tuple[str, str, float, Dict[str, float]]:
    meta = extract_meta_tags(html or "")
    urlf = get_url_features(url)

    joined = " ".join([
        url or "",
        title or "",
        " ".join(headings[:20]),
        meta.get("description", ""),
        meta.get("og_type", ""),
        meta.get("og_site_name", ""),
        meta.get("keywords", ""),
        meta.get("article_section", ""),
        meta.get("article_tag", ""),
        text[:12000],
    ]).lower()

    scores: Dict[str, float] = {
        "reference_knowledge": 0.0,
        "health_medical": 0.0,
        "technical_docs": 0.0,
        "scientific_research": 0.0,
        "government_public_sector": 0.0,
        "legal_compliance": 0.0,
        "company_marketing": 0.0,
        "finance_investor": 0.0,
        "news_media": 0.0,
        "blog_editorial": 0.0,
        "support_help": 0.0,
        "commerce_product": 0.0,
        "education_learning": 0.0,
        "community_forum": 0.0,
        "directory_listing": 0.0,
        "dataset_catalog": 0.0,
        "media_entertainment": 0.0,
        "general_information": 0.0,
    }

    host = urlf["host"]
    path = urlf["path"]
    tld = urlf["tld"]
    subdomains = " ".join(urlf["subdomains"])
    og_type = (meta.get("og_type", "") or "").lower()

    if tld in {"gov", "gouv"} or ".gov." in host or ".gov/" in url.lower():
        scores["government_public_sector"] += 4.0
    if tld in {"edu", "ac"} or ".edu" in host:
        scores["education_learning"] += 2.5
    if tld in {"mil"}:
        scores["government_public_sector"] += 4.0
    if ".org" in host:
        scores["general_information"] += 0.6

    path_rules = [
        (r"/docs?/|/documentation/|/api/|/reference/|/sdk/|/manual/", "technical_docs", 3.0),
        (r"/help/|/support/|/faq/|/kb/|/knowledge-base/|/troubleshooting/", "support_help", 3.2),
        (r"/blog/|/posts?/|/articles?/", "blog_editorial", 2.2),
        (r"/news/|/press/|/press-release/|/newsroom/", "news_media", 2.8),
        (r"/product/|/products/|/pricing/|/shop/|/store/|/buy/", "commerce_product", 3.0),
        (r"/research/|/paper/|/journal/|/study/|/publications?/", "scientific_research", 2.8),
        (r"/legal/|/terms/|/privacy/|/agreement/|/compliance/", "legal_compliance", 3.0),
        (r"/investors?/|/earnings/|/annual-report/|/financials?/", "finance_investor", 3.0),
        (r"/about/|/company/|/team/|/leadership/|/careers/", "company_marketing", 2.2),
        (r"/forum/|/community/|/discussion/|/thread/", "community_forum", 3.0),
        (r"/directory/|/listing/|/catalog/|/browse/|/index/", "directory_listing", 3.0),
        (r"/dataset/|/data/|/repository/|/download/", "dataset_catalog", 2.4),
        (r"/learn/|/course/|/lesson/|/tutorial/", "education_learning", 2.8),
        (r"/wiki/|/encyclopedia/", "reference_knowledge", 3.2),
    ]
    for pattern, archetype, weight in path_rules:
        if re.search(pattern, path, flags=re.I):
            scores[archetype] += weight

    if og_type == "article":
        scores["news_media"] += 1.4
        scores["blog_editorial"] += 1.0
    elif og_type == "product":
        scores["commerce_product"] += 2.4
    elif og_type == "website":
        scores["company_marketing"] += 0.8
        scores["general_information"] += 0.6

    content_rules = {
        "reference_knowledge": [
            r"\bencyclopedia\b", r"\boverview\b", r"\bhistory of\b", r"\bbackground\b",
            r"\breferences\b", r"\bfurther reading\b", r"\bsee also\b",
        ],
        "health_medical": [
            r"\bsymptoms?\b", r"\bdiagnosis\b", r"\btreatment\b", r"\bclinical\b", r"\bpatient\b",
            r"\bdisease\b", r"\bcondition\b", r"\bprognosis\b", r"\bprevention\b",
        ],
        "technical_docs": [
            r"\bapi\b", r"\bendpoints?\b", r"\bparameters?\b", r"\bconfiguration\b", r"\binstallation\b",
            r"\busage\b", r"\bexamples?\b", r"\bsdk\b", r"\bdeveloper\b", r"\breference\b",
        ],
        "scientific_research": [
            r"\babstract\b", r"\bmethods?\b", r"\bresults?\b", r"\bdiscussion\b", r"\bconclusion\b",
            r"\bparticipants?\b", r"\btrial\b", r"\bdoi\b", r"\bjournal\b", r"\bstatistically significant\b",
        ],
        "government_public_sector": [
            r"\bguidance\b", r"\bregulation\b", r"\bdepartment\b", r"\bministry\b", r"\bstatutory\b",
            r"\bpublic consultation\b", r"\bofficial\b", r"\bpolicy\b",
        ],
        "legal_compliance": [
            r"\bterms of service\b", r"\bprivacy policy\b", r"\bliability\b", r"\bagreement\b",
            r"\bgoverning law\b", r"\brights?\b", r"\bjurisdiction\b", r"\bcompliance\b",
        ],
        "company_marketing": [
            r"\bour mission\b", r"\bour vision\b", r"\bout team\b", r"\bleadership\b", r"\bcareers\b",
            r"\babout us\b", r"\bour story\b",
        ],
        "finance_investor": [
            r"\brevenue\b", r"\bprofit\b", r"\bebitda\b", r"\bearnings\b", r"\bquarterly\b",
            r"\bshareholder\b", r"\bannual report\b", r"\bcash flow\b",
        ],
        "news_media": [
            r"\breported\b", r"\bpublished\b", r"\bupdated\b", r"\bbreaking\b", r"\bpress release\b",
            r"\baccording to\b",
        ],
        "blog_editorial": [
            r"\btutorial\b", r"\bguide\b", r"\bbest practices\b", r"\bopinion\b", r"\btips\b",
            r"\bhow to\b",
        ],
        "support_help": [
            r"\bfaq\b", r"\btroubleshooting\b", r"\bhow do i\b", r"\bcommon questions\b",
            r"\bsupport\b", r"\bhelp center\b",
        ],
        "commerce_product": [
            r"\bbuy now\b", r"\badd to cart\b", r"\bspecifications?\b", r"\bproduct details\b",
            r"\bshipping\b", r"\bcustomer reviews?\b", r"\bsku\b", r"\bprice\b",
        ],
        "education_learning": [
            r"\bstudents?\b", r"\blesson\b", r"\bteacher\b", r"\bquiz\b", r"\bcourse\b",
            r"\blearn\b", r"\bclassroom\b",
        ],
        "community_forum": [
            r"\breplies\b", r"\bposted by\b", r"\bmember\b", r"\bmoderator\b", r"\bthread\b",
            r"\bdiscussion\b", r"\bjoined\b",
        ],
        "directory_listing": [
            r"\bfilter by\b", r"\bsort by\b", r"\bresults\b", r"\bview all\b", r"\bbrowse\b",
            r"\bshowing \d+\b",
        ],
        "dataset_catalog": [
            r"\bdataset\b", r"\bdownload csv\b", r"\bdownload json\b", r"\brecord count\b",
            r"\bdata dictionary\b", r"\bmetadata\b",
        ],
        "media_entertainment": [
            r"\bepisode\b", r"\bcast\b", r"\btrailer\b", r"\bseason\b", r"\balbum\b", r"\btracklist\b",
        ],
        "general_information": [
            r"\binformation\b", r"\bdetails\b", r"\boverview\b", r"\bguide\b",
        ],
    }

    for archetype, patterns in content_rules.items():
        hits = count_regex_hits(joined, patterns)
        scores[archetype] += min(hits, 8) * 0.65

    if catalog_info["is_catalog_like"]:
        scores["directory_listing"] += 2.5

    if len(headings) >= 6:
        scores["reference_knowledge"] += 0.8
        scores["technical_docs"] += 0.8
        scores["education_learning"] += 0.5

    if re.search(r"\babstract\b", joined) and re.search(r"\bmethods?\b", joined):
        scores["scientific_research"] += 2.0

    if re.search(r"\bfaq\b", joined) or re.search(r"\bq:\s|\ba:\s", joined):
        scores["support_help"] += 2.0

    if re.search(r"\bprivacy policy\b|\bterms of service\b", joined):
        scores["legal_compliance"] += 2.5

    if re.search(r"\bapi\b", joined) and re.search(r"\bparameters?\b|\brequest\b|\bresponse\b", joined):
        scores["technical_docs"] += 2.2

    if re.search(r"\bprice\b|\badd to cart\b|\bshipping\b", joined):
        scores["commerce_product"] += 2.0

    if re.search(r"\bposted\b|\breplies\b|\bthread\b", joined):
        scores["community_forum"] += 1.8

    if re.search(r"\brevenue\b|\bprofit\b|\bebitda\b|\bannual report\b", joined):
        scores["finance_investor"] += 2.2

    if "wikipedia.org" in host or "britannica." in host:
        scores["reference_knowledge"] += 5.0
    if "pubmed" in host or "nejm" in host or "thelancet" in host:
        scores["scientific_research"] += 4.5
        scores["health_medical"] += 2.0
    if "docs." in host or "developer." in host:
        scores["technical_docs"] += 3.0
    if "support." in host or "help." in host:
        scores["support_help"] += 3.0
    if "forum." in host or "community." in host:
        scores["community_forum"] += 3.0
    if "investor" in subdomains:
        scores["finance_investor"] += 2.5

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_domain, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_score < 1.2:
        domain = "general_information"
    else:
        domain = top_domain

    page_form = "explainer"

    if catalog_info["is_catalog_like"]:
        page_form = "listing"
    elif domain == "technical_docs":
        if re.search(r"\bapi\b|\breference\b|\bparameters?\b|\bendpoints?\b", joined):
            page_form = "reference"
        elif re.search(r"\brelease notes\b|\bchangelog\b|\bwhat's new\b", joined):
            page_form = "changelog"
        else:
            page_form = "explainer"
    elif domain == "scientific_research":
        page_form = "paper"
    elif domain == "support_help":
        page_form = "faq" if re.search(r"\bfaq\b|\bcommon questions\b", joined) else "explainer"
    elif domain == "commerce_product":
        page_form = "spec_sheet"
    elif domain in {"government_public_sector", "legal_compliance"}:
        page_form = "policy" if not catalog_info["is_catalog_like"] else "listing"
    elif domain in {"news_media", "blog_editorial", "media_entertainment"}:
        page_form = "article"
    elif domain == "company_marketing":
        page_form = "overview"
    elif domain == "community_forum":
        page_form = "forum_thread"
    elif domain == "dataset_catalog":
        page_form = "dataset"
    elif domain == "reference_knowledge":
        page_form = "reference" if re.search(r"\breferences\b|\bsee also\b", joined) else "explainer"
    elif domain == "finance_investor":
        page_form = "report" if re.search(r"\bannual report\b|\bquarterly\b|\bearnings\b", joined) else "overview"

    gap = top_score - second_score
    classifier_confidence = 0.40 + min(top_score / 10.0, 0.30) + min(gap / 6.0, 0.18)
    if domain == "general_information":
        classifier_confidence -= 0.08
    if catalog_info["is_catalog_like"] and page_form == "listing":
        classifier_confidence += 0.03
    classifier_confidence = round(max(0.30, min(classifier_confidence, 0.94)), 2)

    return domain, page_form, classifier_confidence, scores


def compute_section_metrics(sections: List[Dict[str, str]]) -> Dict[str, Any]:
    section_scores = []
    rich_sections = 0

    for sec in sections[:20]:
        sec_text = sec["text"]
        words = tokenize_words(sec_text)
        sents = split_sentences(sec_text)
        facts = count_fact_like_sentences(sents) + count_fact_like_list_items(sec_text)
        richness = (
            min(len(words) / 120.0, 6.0) +
            min(len(sents) / 8.0, 4.0) +
            min(facts / 6.0, 4.0)
        )
        section_scores.append(round(richness, 3))
        if richness >= 2.0:
            rich_sections += 1

    return {
        "section_count": len(sections),
        "rich_section_count": rich_sections,
        "avg_section_richness": round(sum(section_scores) / max(len(section_scores), 1), 3),
        "section_richness_scores": section_scores[:20],
    }


def compute_base_metrics(url: str, title: str, html: str, text: str) -> Dict[str, Any]:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    chunks = chunk_text(text)
    headings = clean_headings_for_article(html, text)
    sections = split_into_sections(headings, text)
    meta = extract_meta_tags(html)
    url_features = get_url_features(url)

    lexical = lexical_diversity(words)
    sent_div = sentence_diversity(sentences)
    chunk_uni = average_chunk_uniqueness(chunks)
    fact_like_sentences = count_fact_like_sentences(sentences)
    fact_like_list_items = count_fact_like_list_items(text)
    catalog_info = detect_catalog_page(text, headings, sentences)

    domain, page_form, classifier_confidence, page_type_scores = detect_domain_and_form(
        url=url,
        title=title,
        text=text,
        headings=headings,
        catalog_info=catalog_info,
        html=html,
    )

    section_metrics = compute_section_metrics(sections)

    return {
        "title": title,
        "domain": domain,
        "page_form": page_form,
        "classifier_confidence": classifier_confidence,
        "page_type_scores": page_type_scores,
        "meta_tags": meta,
        "url_features": url_features,
        "text_length_chars": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "chunk_count": len(chunks),
        "heading_count": len(headings),
        "avg_sentence_length_words": average_sentence_length(words, sentences),
        "lexical_diversity": lexical,
        "sentence_diversity": sent_div,
        "entropy_score": entropy_score(words),
        "chunk_uniqueness": chunk_uni,
        "repetition_penalty": repetition_penalty(lexical, sent_div, chunk_uni),
        "numeric_token_count": count_numeric_tokens(text),
        "definition_signal_count": count_definition_patterns(text),
        "comparison_signal_count": count_comparison_patterns(text),
        "causal_signal_count": count_causal_patterns(text),
        "list_signal_count": count_list_signals(text),
        "fact_like_sentence_count": fact_like_sentences,
        "fact_like_list_item_count": fact_like_list_items,
        "fact_like_total": fact_like_sentences + fact_like_list_items,
        "long_sentence_count": count_long_sentences(sentences),
        "sample_headings": headings[:12],
        "catalog_info": catalog_info,
        **section_metrics,
    }


def get_profile(domain: str, page_form: str) -> Dict[str, float]:
    profile_domain = OPEN_DOMAIN_TO_PROFILE.get(domain, domain)
    profile_form = OPEN_FORM_TO_PROFILE.get(page_form, page_form)
    return SPECIALIST_PROFILES.get((profile_domain, profile_form), DEFAULT_PROFILE)


def estimate_with_specialist(metrics: Dict[str, Any]) -> Dict[str, Any]:
    chars = metrics["text_length_chars"]
    words = metrics["word_count"]
    sentences = metrics["sentence_count"]
    chunks = metrics["chunk_count"]
    headings = metrics["heading_count"]

    lexical = metrics["lexical_diversity"]
    sent_div = metrics["sentence_diversity"]
    entropy = metrics["entropy_score"]
    chunk_uni = metrics["chunk_uniqueness"]
    rep_penalty = metrics["repetition_penalty"]

    numeric = metrics["numeric_token_count"]
    defs = metrics["definition_signal_count"]
    comps = metrics["comparison_signal_count"]
    causals = metrics["causal_signal_count"]
    lists = metrics["list_signal_count"]
    fact_like = metrics["fact_like_total"]
    long_sent = metrics["long_sentence_count"]

    rich_sections = metrics["rich_section_count"]
    avg_section_richness = metrics["avg_section_richness"]
    catalog_penalty = metrics["catalog_info"]["catalog_penalty"]

    domain = metrics["domain"]
    page_form = metrics["page_form"]
    profile = get_profile(domain, page_form)

    if chars < MIN_TEXT_LENGTH or words < 80 or sentences < 5:
        return {
            "raw_extractable_pairs": 0,
            "training_grade_pairs": 0,
            "heuristic_min": 0,
            "heuristic_max": 2,
            "heuristic_confidence": 0.2,
            "heuristic_reason": "Too little extractable text for reliable Q/A estimation.",
        }

    base_from_volume = (
        (words / 75.0) +
        (sentences / 5.0) +
        (chunks * 0.7) +
        (headings * 0.55)
    )

    fact_bonus = (
        min(fact_like, 90) * 0.32 +
        min(defs, 14) * 0.60 +
        min(comps, 12) * 0.35 +
        min(causals, 12) * 0.35 +
        min(lists, 24) * 0.10 +
        min(numeric, 30) * 0.04 +
        min(long_sent, 80) * 0.05
    ) * profile["fact_bonus_multiplier"]

    section_bonus = (
        min(rich_sections, 12) * 0.30 +
        min(avg_section_richness, 4.0) * 0.80
    ) * profile["depth_bonus"]

    quality_multiplier = (
        0.60 +
        min(lexical, 0.50) * 0.35 +
        min(sent_div, 1.0) * 0.18 +
        min(chunk_uni, 1.0) * 0.15 +
        min(entropy / 8.5, 1.0) * 0.12
    )

    raw_score = (
        (base_from_volume + fact_bonus + section_bonus)
        * quality_multiplier
        * rep_penalty
        * profile["type_multiplier"]
        * catalog_penalty
    )

    base_cap = min(
        max(4, int(words / 45)),
        max(4, int(sentences * 0.90)),
        max(4, int(headings * 2.8 + 6)),
        max(4, int(fact_like * 0.95)),
        max(4, int(rich_sections * 5 + 4)),
    )

    if lists >= 8:
        base_cap = max(base_cap, min(16, int(lists * profile["list_floor_multiplier"])))

    cap = max(3, int(round(base_cap * profile["cap_multiplier"] * catalog_penalty)))

    training_grade_pairs = max(1, min(int(round(raw_score)), cap))
    raw_extractable_pairs = max(training_grade_pairs, min(cap + 8, int(round(training_grade_pairs * 1.22 + 2))))

    spread = max(2, int(round(training_grade_pairs * 0.18)))

    confidence = 0.36
    if words >= 500:
        confidence += 0.08
    if headings >= 4:
        confidence += 0.06
    if lexical >= 0.28:
        confidence += 0.05
    if sent_div >= 0.80:
        confidence += 0.05
    if chunk_uni >= 0.80:
        confidence += 0.05
    if fact_like >= 10:
        confidence += 0.06
    if rich_sections >= 3:
        confidence += 0.05
    if metrics["classifier_confidence"] < 0.55:
        confidence -= 0.06
    if metrics["catalog_info"]["is_catalog_like"]:
        confidence -= 0.05
    confidence = round(max(0.28, min(confidence, 0.86)), 2)

    return {
        "raw_extractable_pairs": raw_extractable_pairs,
        "training_grade_pairs": training_grade_pairs,
        "heuristic_min": max(0, training_grade_pairs - spread),
        "heuristic_max": training_grade_pairs + spread,
        "heuristic_confidence": confidence,
        "heuristic_reason": (
            f"Conservative {domain}/{page_form} estimate based on cleaned content volume, "
            f"section richness, fact-like content, structure, diversity, and repetition control."
        ),
    }


def get_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise ValueError("Missing DEEPSEEK_API_KEY.")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def build_llm_sample(text: str, max_chars: int = LLM_SAMPLE_CHARS) -> str:
    text = normalize_spaces(text)
    if len(text) <= max_chars:
        return text

    part = max_chars // 3
    n = len(text)
    start = text[:part]
    mid_start = max(0, (n // 2) - (part // 2))
    middle = text[mid_start:mid_start + part]
    end = text[-part:]

    return start + "\n\n[...MIDDLE...]\n\n" + middle + "\n\n[...END...]\n\n" + end


def llm_capacity_estimate(title: str, text: str, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not DEEPSEEK_API_KEY:
        return None

    client = get_client()
    sample = build_llm_sample(text)
    metrics_for_prompt = {
        k: v for k, v in metrics.items()
        if k not in {"sample_headings", "page_type_scores", "section_richness_scores", "meta_tags", "url_features"}
    }

    prompt = f"""
You are estimating how many DISTINCT, non-trivial, answerable Q/A pairs can be extracted from one webpage.

Return ONLY valid JSON with this exact schema:
{{
  "raw_extractable_pairs": 0,
  "training_grade_pairs": 0,
  "llm_min": 0,
  "llm_max": 0,
  "confidence": 0.0,
  "reasoning_summary": ""
}}

Rules:
- Count only Q/A pairs directly answerable from the provided page text.
- "raw_extractable_pairs" can include simpler but still valid factual pairs.
- "training_grade_pairs" should count only stronger, non-trivial, reusable educational pairs.
- Prefer useful, factual, non-duplicate Q/A pairs.
- Be conservative.
- Do not hallucinate.
- Confidence must be between 0 and 1.
- Adapt to the domain, page form, catalog-like signals, meta tags, and URL features in METRICS.
- This is an open-world classifier: do not assume the site fits any narrow fixed category.

PAGE TITLE:
{title}

METRICS:
{json.dumps(metrics_for_prompt, ensure_ascii=False)}

TEXT SAMPLE:
\"\"\"{sample}\"\"\"
"""

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise evaluator that returns JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
        stream=False,
    )

    raw = (response.choices[0].message.content or "").strip()
    data = json.loads(raw)

    required = {
        "raw_extractable_pairs",
        "training_grade_pairs",
        "llm_min",
        "llm_max",
        "confidence",
        "reasoning_summary",
    }
    if not isinstance(data, dict) or not required.issubset(set(data.keys())):
        return None

    data["raw_extractable_pairs"] = safe_int(data["raw_extractable_pairs"])
    data["training_grade_pairs"] = safe_int(data["training_grade_pairs"])
    data["llm_min"] = safe_int(data["llm_min"])
    data["llm_max"] = safe_int(data["llm_max"])
    data["confidence"] = max(0.0, min(1.0, safe_float(data["confidence"])))
    return data


def adjusted_volume_cap(metrics: Dict[str, Any]) -> int:
    words = metrics.get("word_count", 0)
    sentences = metrics.get("sentence_count", 0)
    headings = metrics.get("heading_count", 0)
    fact_like = metrics.get("fact_like_total", 0)
    lists = metrics.get("list_signal_count", 0)
    rich_sections = metrics.get("rich_section_count", 0)
    domain = metrics.get("domain", "unknown")
    page_form = metrics.get("page_form", "mixed")
    profile = get_profile(domain, page_form)

    volume_cap = min(
        max(3, int(words / 45)),
        max(3, int(sentences * 0.78 + headings * 0.85)),
        max(3, int(fact_like * 1.05)),
        max(3, int(rich_sections * 5 + 4)),
    )
    volume_cap = max(
        3,
        int(round(volume_cap * profile["cap_multiplier"] * metrics["catalog_info"]["catalog_penalty"]))
    )

    if lists >= 8:
        volume_cap = max(volume_cap, min(16, int(lists * profile["list_floor_multiplier"])))

    return volume_cap


def combine_estimates(
    heuristic: Dict[str, Any],
    llm_estimate: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    volume_cap = adjusted_volume_cap(metrics)

    if llm_estimate is None:
        predicted = min(heuristic["training_grade_pairs"], volume_cap)
        raw_pairs = max(predicted, min(volume_cap + 8, heuristic["raw_extractable_pairs"]))
        return {
            "raw_extractable_pairs": raw_pairs,
            "training_grade_pairs": predicted,
            "predicted_pairs": predicted,
            "predicted_min": max(0, min(heuristic["heuristic_min"], predicted - 2)),
            "predicted_max": min(heuristic["heuristic_max"], volume_cap),
            "confidence": min(0.78, heuristic["heuristic_confidence"]),
            "method": "heuristic_only_open_world",
            "notes": [
                heuristic["heuristic_reason"],
                f"Final estimate capped by cleaned page-volume ceiling ({volume_cap}).",
            ],
        }

    hp = heuristic["training_grade_pairs"]
    lp = llm_estimate["training_grade_pairs"]
    disagreement = abs(hp - lp) / max(hp, lp, 1)

    predicted = min(volume_cap, max(lp, int(round((lp * 0.82) + (min(hp, volume_cap) * 0.18)))))
    raw_pairs = max(
        predicted,
        int(round((llm_estimate["raw_extractable_pairs"] * 0.75) + (heuristic["raw_extractable_pairs"] * 0.25)))
    )

    predicted_min = min(
        predicted,
        max(0, int(round((llm_estimate["llm_min"] * 0.88) + (heuristic["heuristic_min"] * 0.12))))
    )
    predicted_max = min(
        volume_cap,
        max(predicted, int(round((llm_estimate["llm_max"] * 0.88) + (heuristic["heuristic_max"] * 0.12))))
    )

    confidence = 0.54 + (llm_estimate["confidence"] * 0.30) + (heuristic["heuristic_confidence"] * 0.10)
    confidence += (metrics.get("classifier_confidence", 0.5) - 0.5) * 0.10
    if disagreement > 0.40:
        confidence -= 0.12
    elif disagreement > 0.25:
        confidence -= 0.06
    if metrics["catalog_info"]["is_catalog_like"]:
        confidence -= 0.04
    confidence = round(max(0.32, min(confidence, 0.93)), 2)

    return {
        "raw_extractable_pairs": raw_pairs,
        "training_grade_pairs": max(0, predicted),
        "predicted_pairs": max(0, predicted),
        "predicted_min": max(0, predicted_min),
        "predicted_max": max(max(0, predicted_min), predicted_max),
        "confidence": confidence,
        "method": "llm_led_open_world_blend_v7",
        "notes": [
            heuristic["heuristic_reason"],
            llm_estimate.get("reasoning_summary", ""),
            f"Final estimate capped by cleaned page-volume ceiling ({volume_cap}).",
            f"Heuristic/LLM disagreement ratio: {round(disagreement, 3)}",
        ],
        "llm_details": llm_estimate,
    }


def evaluate_url_qa_capacity(url: str) -> Dict[str, Any]:
    if not url or not is_url(url):
        raise ValueError("Please provide a valid http:// or https:// URL.")

    started = time.time()

    html = fetch_html(url)
    title = extract_title(html)
    raw_text = extract_main_text(html, url=url)
    text = clean_extracted_text(raw_text)

    if not text or len(text) < MIN_TEXT_LENGTH:
        return {
            "url": url,
            "title": title,
            "status": "not_enough_content",
            "training_grade_pairs": 0,
            "raw_extractable_pairs": 0,
            "predicted_pairs": 0,
            "predicted_min": 0,
            "predicted_max": 2,
            "confidence": 0.15,
            "method": "insufficient_text",
            "metrics": {
                "raw_text_length_chars": len(raw_text or ""),
                "clean_text_length_chars": len(text or ""),
            },
            "notes": ["The page does not appear to contain enough clean extractable text."],
            "elapsed_seconds": round(time.time() - started, 2),
        }

    metrics = compute_base_metrics(url=url, title=title, html=html, text=text)
    heuristic = estimate_with_specialist(metrics)

    llm_estimate = None
    llm_used = False
    llm_error = None

    try:
        llm_estimate = llm_capacity_estimate(title=title, text=text, metrics=metrics)
        llm_used = llm_estimate is not None
    except AuthenticationError:
        raise
    except Exception as e:
        llm_error = str(e)
        llm_estimate = None

    final_estimate = combine_estimates(heuristic, llm_estimate, metrics)

    result = {
        "url": url,
        "title": title,
        "status": "ok",
        "training_grade_pairs": final_estimate["training_grade_pairs"],
        "raw_extractable_pairs": final_estimate["raw_extractable_pairs"],
        "predicted_pairs": final_estimate["predicted_pairs"],
        "predicted_min": final_estimate["predicted_min"],
        "predicted_max": final_estimate["predicted_max"],
        "confidence": final_estimate["confidence"],
        "method": final_estimate["method"],
        "metrics": metrics,
        "notes": final_estimate["notes"],
        "elapsed_seconds": round(time.time() - started, 2),
        "llm_used": llm_used,
    }

    if llm_error:
        result["llm_error"] = llm_error
    if "llm_details" in final_estimate:
        result["llm_details"] = final_estimate["llm_details"]

    result["quality_view"] = {
        "raw_extractable_pairs": result["raw_extractable_pairs"],
        "training_grade_pairs": result["training_grade_pairs"],
        "repetition_risk": (
            "low" if metrics["repetition_penalty"] >= 0.94
            else "medium" if metrics["repetition_penalty"] >= 0.84
            else "high"
        ),
        "catalog_risk": (
            "high" if metrics["catalog_info"]["catalog_score"] >= 0.65
            else "medium" if metrics["catalog_info"]["catalog_score"] >= 0.45
            else "low"
        ),
        "extraction_cleanliness": "high" if metrics["text_length_chars"] >= MIN_TEXT_LENGTH else "low",
        "domain": metrics["domain"],
        "page_form": metrics["page_form"],
        "classifier_confidence": metrics["classifier_confidence"],
    }

    return result


def apply_strictness(result: Dict[str, Any], strictness: str) -> Dict[str, Any]:
    if result.get("status") != "ok":
        return result

    base = result["training_grade_pairs"]
    min_v = result["predicted_min"]
    max_v = result["predicted_max"]

    if strictness == "Strict":
        result["training_grade_pairs"] = max(0, int(round(base * 0.88)))
        result["predicted_pairs"] = result["training_grade_pairs"]
        result["predicted_min"] = max(0, int(round(min_v * 0.90)))
        result["predicted_max"] = max(result["predicted_pairs"], int(round(max_v * 0.92)))
    elif strictness == "Lenient":
        result["training_grade_pairs"] = int(round(base * 1.08))
        result["predicted_pairs"] = result["training_grade_pairs"]
        result["predicted_min"] = max(0, int(round(min_v * 1.02)))
        result["predicted_max"] = int(round(max_v * 1.10))

    return result


def run_evaluator_service(url_input: str, strictness: str = "Standard") -> Dict[str, Any]:
    url_input = (url_input or "").strip()

    if not url_input:
        raise ValueError("Please enter a URL.")
    if not is_url(url_input):
        raise ValueError("Please enter a valid http:// or https:// URL.")

    result = evaluate_url_qa_capacity(url_input)
    return apply_strictness(result, strictness)