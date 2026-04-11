import re
from html import unescape

import requests
import trafilatura

USER_AGENT = "Mozilla/5.0 (compatible; UnifiedQASystem/1.0-open-world)"
DEFAULT_TIMEOUT = 25
MIN_TEXT_LENGTH = 300


def fetch_html(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response.text


def extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    if not match:
        return ""

    return re.sub(r"\s+", " ", unescape(match.group(1))).strip()


def extract_main_text(html: str, url: str) -> str:
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    return text or ""


def clean_extracted_text(text: str) -> str:
    text = unescape(text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_page(url: str) -> tuple[str, str]:
    html = fetch_html(url)
    title = extract_title(html)
    text = clean_extracted_text(extract_main_text(html, url))

    if len(text) < MIN_TEXT_LENGTH:
        raise ValueError("The page does not contain enough clean extractable text.")

    return title, text