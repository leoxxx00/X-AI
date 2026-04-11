from urllib.parse import urlparse


def validate_http_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must start with http:// or https://")
    if not parsed.netloc:
        raise ValueError("Invalid URL")
    return url.strip()