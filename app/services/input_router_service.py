from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass
class InputRoute:
    input_kind: str
    original_input: str
    normalized_input: str
    requires_extraction: bool
    reason: str


YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

TWITTER_DOMAINS = {
    "twitter.com",
    "www.twitter.com",
    "mobile.twitter.com",
    "x.com",
    "www.x.com",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff")


def _clean(value: Optional[str]) -> str:
    return " ".join(str(value or "").strip().split())


def _looks_like_url(value: str) -> bool:
    value = value.strip()
    if value.startswith(("http://", "https://")):
        return True
    parsed = urlparse("https://" + value)
    return bool(parsed.netloc and "." in parsed.netloc and " " not in value)


def _normalize_url(value: str) -> str:
    value = value.strip()
    if not value.startswith(("http://", "https://")):
        value = "https://" + value
    return value


def detect_input_kind(raw_input: str, source_type: str = "auto") -> InputRoute:
    # Detect whether the input is direct text, URL, YouTube URL, X/Twitter URL, etc.
    original = str(raw_input or "").strip()
    requested = _clean(source_type).lower() or "auto"

    if requested in {"text", "claim", "headline", "article", "plain_text"}:
        return InputRoute("plain_text", original, original, False, "Source type explicitly set to text.")

    if requested in {"youtube", "youtube_url"}:
        return InputRoute("youtube_url", original, _normalize_url(original), True, "Source type explicitly set to YouTube.")

    if requested in {"twitter", "twitter_url", "x", "x_url"}:
        return InputRoute("twitter_url", original, _normalize_url(original), True, "Source type explicitly set to X/Twitter.")

    if requested in {"pdf", "pdf_url"}:
        return InputRoute("pdf_url", original, _normalize_url(original), True, "Source type explicitly set to PDF URL.")

    if requested in {"image", "image_url", "screenshot"}:
        return InputRoute("image_url", original, _normalize_url(original), True, "Source type explicitly set to image URL.")

    if requested in {"url", "web_url", "article_url"}:
        normalized = _normalize_url(original)
        parsed = urlparse(normalized)
        domain = parsed.netloc.lower()
        if domain in YOUTUBE_DOMAINS:
            return InputRoute("youtube_url", original, normalized, True, "URL domain is YouTube.")
        if domain in TWITTER_DOMAINS:
            return InputRoute("twitter_url", original, normalized, True, "URL domain is X/Twitter.")
        return InputRoute("url", original, normalized, True, "Source type explicitly set to URL.")

    if not _looks_like_url(original):
        return InputRoute("plain_text", original, original, False, "Input does not look like a URL.")

    normalized = _normalize_url(original)
    parsed = urlparse(normalized)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    if domain in YOUTUBE_DOMAINS:
        return InputRoute("youtube_url", original, normalized, True, "URL domain is YouTube.")

    if domain in TWITTER_DOMAINS:
        return InputRoute("twitter_url", original, normalized, True, "URL domain is X/Twitter.")

    if path.endswith(".pdf"):
        return InputRoute("pdf_url", original, normalized, True, "URL path ends with .pdf.")

    if path.endswith(IMAGE_EXTENSIONS):
        return InputRoute("image_url", original, normalized, True, "URL path looks like an image.")

    return InputRoute("url", original, normalized, True, "Input looks like a generic web URL.")
