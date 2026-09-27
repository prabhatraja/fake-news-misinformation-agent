import html
import re
from html.parser import HTMLParser
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

from app.services.input_router_service import InputRoute


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


class ReadableTextParser(HTMLParser):
    # Small dependency-free HTML text extractor.

    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "form"}
    BLOCK_TAGS = {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "tr"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self.skip_depth = 0
        self.title = ""
        self._in_title = False
        self.meta: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        attrs_dict = {str(k).lower(): str(v) for k, v in attrs if k and v is not None}

        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return

        if tag == "title":
            self._in_title = True

        if tag == "meta":
            key = (
                attrs_dict.get("property")
                or attrs_dict.get("name")
                or attrs_dict.get("itemprop")
                or ""
            ).lower()
            value = attrs_dict.get("content", "")
            if key and value:
                self.meta[key] = html.unescape(value).strip()

        if tag == "link" and attrs_dict.get("rel", "").lower() == "canonical":
            href = attrs_dict.get("href", "")
            if href:
                self.meta["canonical"] = href

        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth > 0:
            self.skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth > 0:
            return
        data = html.unescape(data or "").strip()
        if not data:
            return
        if self._in_title:
            self.title += " " + data
        self.parts.append(data)
        self.parts.append(" ")

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        lines = [line for line in lines if len(line) > 1]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split()).strip()



def _fix_common_mojibake(value: Any) -> str:
    """Fix common UTF-8/Windows-1252 mojibake seen in extracted web text.

    This intentionally avoids heavy dependencies. It handles the common artifacts
    that were observed during Phase 2 URL extraction, for example:
    48Â°51â²29.6â³N -> 48°51′29.6″N
    """
    text = str(value or "")

    replacements = {
        "Â°": "°",
        "Â±": "±",
        "Â£": "£",
        "Â©": "©",
        "Â®": "®",
        "Â·": "·",
        "Â ": " ",
        "Â": "",
        "â²": "′",
        "â³": "″",
        "â€²": "′",
        "â€³": "″",
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€�": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "…",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã¡": "á",
        "Ã³": "ó",
        "Ã¼": "ü",
        "Ã¶": "ö",
        "Ã±": "ñ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return " ".join(text.replace("\u00a0", " ").split()).strip()


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _first_meta(meta: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        if meta.get(key):
            return _clean(meta[key])
    return ""


def _base_result(route: InputRoute) -> Dict[str, Any]:
    return {
        "status": "not_started",
        "input_kind": route.input_kind,
        "url": route.normalized_input if route.requires_extraction else "",
        "domain": _domain(route.normalized_input) if route.requires_extraction else "",
        "title": "",
        "author": "",
        "published_date": "",
        "text": "",
        "text_length": 0,
        "limitations": [],
        "metadata": {},
    }


def extract_plain_text(route: InputRoute) -> Dict[str, Any]:
    text = _clean(route.normalized_input)
    result = _base_result(route)
    result.update(
        {
            "status": "success" if text else "failed",
            "title": "Direct text input",
            "text": text,
            "text_length": len(text),
            "metadata": {"source_type": "plain_text"},
        }
    )
    return result


def extract_generic_url(route: InputRoute, timeout: int = 20) -> Dict[str, Any]:
    result = _base_result(route)
    url = route.normalized_input

    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "limitations": [f"URL fetch failed: {error}"],
            }
        )
        return result

    content_type = response.headers.get("content-type", "").lower()

    # Let requests use detected encoding where possible. This reduces URL extraction mojibake.
    response.encoding = response.apparent_encoding or response.encoding

    # PDF/image URL extraction enabled in Mega Batch 4.
    if "application/pdf" in content_type or urlparse(url).path.lower().endswith(".pdf"):
        from app.services.file_extraction_service import extract_url_file

        return extract_url_file(
            response.url,
            "pdf_url",
            content_bytes=response.content,
            content_type=content_type,
            final_url=response.url,
        )

    if "image/" in content_type or urlparse(url).path.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")):
        from app.services.file_extraction_service import extract_url_file

        return extract_url_file(
            response.url,
            "image_url",
            content_bytes=response.content,
            content_type=content_type,
            final_url=response.url,
        )

    html_text = response.text or ""
    parser = ReadableTextParser()

    try:
        parser.feed(html_text)
    except Exception:
        pass

    extracted_text = _fix_common_mojibake(parser.text())
    meta = parser.meta

    title = _first_meta(meta, ["og:title", "twitter:title", "headline"]) or _clean(parser.title)
    author = _first_meta(meta, ["author", "article:author", "parsely-author"])
    published = _first_meta(
        meta,
        [
            "article:published_time",
            "datepublished",
            "date",
            "pubdate",
            "dcterms.date",
            "parsely-pub-date",
            "sailthru.date",
        ],
    )

    title = _fix_common_mojibake(title)
    author = _fix_common_mojibake(author)
    published = _fix_common_mojibake(published)

    limitations: List[str] = []
    status = "success"

    if len(extracted_text) < 250:
        status = "partial"
        limitations.append(
            "Extracted text is short. The page may be blocked, dynamic, paywalled, or mostly non-text."
        )

    result.update(
        {
            "status": status,
            "url": response.url,
            "domain": _domain(response.url),
            "title": title,
            "author": author,
            "published_date": published,
            "text": extracted_text,
            "text_length": len(extracted_text),
            "limitations": limitations,
            "metadata": {
                "content_type": content_type,
                "canonical": meta.get("canonical", ""),
                "final_url": response.url,
            },
        }
    )
    return result


def extract_twitter_url(
    route: InputRoute,
    timeout: int = 20,
    manual_text: str = "",
    profile_post_limit: int = 5,
) -> Dict[str, Any]:
    """Delegate X/Twitter extraction to the hardened single-post/profile service."""
    from app.services.twitter_extraction_service import extract_twitter_source

    return extract_twitter_source(
        route,
        manual_text=manual_text,
        profile_post_limit=profile_post_limit,
        timeout=timeout,
    )

def extract_youtube_url(route: InputRoute) -> Dict[str, Any]:
    # Phase 2 Mega Batch 2: YouTube metadata + best-effort transcript extraction.
    # The dedicated service keeps optional youtube-transcript-api dependency isolated.
    try:
        from app.services.youtube_extraction_service import extract_youtube_content

        return extract_youtube_content(route)
    except Exception as error:
        result = _base_result(route)
        result.update(
            {
                "status": "failed",
                "title": "YouTube URL detected",
                "limitations": [
                    f"YouTube extraction failed: {error}",
                    "Paste the video transcript or description for reliable analysis.",
                ],
                "metadata": {"source_type": "youtube_url", "error": str(error)},
            }
        )
        return result


def extract_content(
    route: InputRoute,
    *,
    twitter_manual_text: str = "",
    twitter_profile_post_limit: int = 5,
) -> Dict[str, Any]:
    if route.input_kind == "plain_text":
        return extract_plain_text(route)

    if route.input_kind == "twitter_url":
        return extract_twitter_url(
            route,
            manual_text=twitter_manual_text,
            profile_post_limit=twitter_profile_post_limit,
        )

    if route.input_kind == "youtube_url":
        return extract_youtube_url(route)

    if route.input_kind in {"pdf_url", "image_url"}:
        from app.services.file_extraction_service import extract_url_file

        return extract_url_file(route.normalized_input, route.input_kind)

    if route.input_kind == "url":
        return extract_generic_url(route)

    result = _base_result(route)
    result.update(
        {
            "status": "failed",
            "limitations": [f"Unsupported input kind: {route.input_kind}"],
        }
    )
    return result
