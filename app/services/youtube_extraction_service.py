import html
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


TRANSCRIPT_LANGUAGES = ["en", "en-US", "en-GB", "hi"]
MAX_TRANSCRIPT_CHARS = 14000


def _clean(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if domain in {"youtu.be", "www.youtu.be"}:
        return path.split("/")[0]

    query = parse_qs(parsed.query or "")
    if query.get("v"):
        return query["v"][0]

    parts = path.split("/") if path else []
    for marker in ["shorts", "embed", "live", "v"]:
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]

    # Last fallback for copied URLs that still contain watch/v style fragments.
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{6,})", url)
    return match.group(1) if match else ""


def _base_youtube_result(route, video_id: str = "") -> Dict[str, Any]:
    url = getattr(route, "normalized_input", "")
    return {
        "status": "not_started",
        "input_kind": "youtube_url",
        "url": url,
        "domain": _domain(url),
        "title": "",
        "author": "",
        "published_date": "",
        "text": "",
        "text_length": 0,
        "limitations": [],
        "metadata": {
            "source_type": "youtube_url",
            "video_id": video_id,
            "transcript_available": False,
            "transcript_language": "",
            "transcript_source": "",
            "extraction_note": "YouTube extraction is best-effort. Transcript availability depends on captions, region, and access restrictions.",
        },
    }


def _fetch_oembed(url: str, timeout: int = 15) -> Tuple[Dict[str, Any], List[str]]:
    limitations: List[str] = []
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            headers=DEFAULT_HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json(), limitations
    except Exception as error:
        limitations.append(f"YouTube oEmbed metadata fetch failed: {error}")
        return {}, limitations


def _extract_meta_content(page: str, keys: List[str]) -> str:
    for key in keys:
        patterns = [
            rf'<meta[^>]+property=["\\\']{re.escape(key)}["\\\'][^>]+content=["\\\']([^"\\\']+)["\\\']',
            rf'<meta[^>]+name=["\\\']{re.escape(key)}["\\\'][^>]+content=["\\\']([^"\\\']+)["\\\']',
            rf'<meta[^>]+content=["\\\']([^"\\\']+)["\\\'][^>]+property=["\\\']{re.escape(key)}["\\\']',
            rf'<meta[^>]+content=["\\\']([^"\\\']+)["\\\'][^>]+name=["\\\']{re.escape(key)}["\\\']',
        ]
        for pattern in patterns:
            match = re.search(pattern, page, flags=re.I | re.S)
            if match:
                return _clean(match.group(1))
    return ""


def _fetch_page_metadata(url: str, timeout: int = 15) -> Tuple[Dict[str, str], List[str]]:
    limitations: List[str] = []
    meta: Dict[str, str] = {}
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        page = response.text or ""
        meta["title"] = _extract_meta_content(page, ["og:title", "twitter:title", "title"])
        meta["description"] = _extract_meta_content(page, ["og:description", "description", "twitter:description"])
        meta["published_date"] = _extract_meta_content(page, ["datePublished", "article:published_time", "uploadDate", "datepublished"])
        meta["canonical"] = _extract_meta_content(page, ["og:url"])
    except Exception as error:
        limitations.append(f"YouTube page metadata fetch failed: {error}")
    return meta, limitations


def _normalize_transcript_entries(entries: Any) -> List[Dict[str, Any]]:
    if entries is None:
        return []

    if hasattr(entries, "to_raw_data"):
        entries = entries.to_raw_data()

    normalized = []
    try:
        iterator = list(entries)
    except Exception:
        iterator = []

    for item in iterator:
        if isinstance(item, dict):
            text = item.get("text", "")
            start = item.get("start")
            duration = item.get("duration")
        else:
            text = getattr(item, "text", "")
            start = getattr(item, "start", None)
            duration = getattr(item, "duration", None)
        if text:
            normalized.append({"text": text, "start": start, "duration": duration})
    return normalized


def _fetch_transcript(video_id: str) -> Tuple[str, Dict[str, Any], List[str]]:
    limitations: List[str] = []
    metadata: Dict[str, Any] = {
        "transcript_available": False,
        "transcript_language": "",
        "transcript_source": "",
        "transcript_entries": 0,
    }

    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except Exception as error:
        limitations.append(
            "youtube-transcript-api is not installed. Run: pip install youtube-transcript-api"
        )
        metadata["transcript_error"] = str(error)
        return "", metadata, limitations

    raw_entries: List[Dict[str, Any]] = []
    language_used = ""
    source = ""

    # Older library API.
    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id, languages=TRANSCRIPT_LANGUAGES)
        raw_entries = _normalize_transcript_entries(entries)
        language_used = "preferred"
        source = "get_transcript"
    except Exception as first_error:
        metadata["transcript_first_error"] = str(first_error)

        # Newer API fallback.
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=TRANSCRIPT_LANGUAGES)
            raw_entries = _normalize_transcript_entries(fetched)
            language_used = getattr(fetched, "language_code", "preferred") or "preferred"
            source = "fetch"
        except Exception as second_error:
            metadata["transcript_second_error"] = str(second_error)

            # Last fallback: list available transcripts and choose English/manual/generated if possible.
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                chosen = None
                for lang in TRANSCRIPT_LANGUAGES:
                    try:
                        chosen = transcript_list.find_transcript([lang])
                        break
                    except Exception:
                        continue
                if chosen is None:
                    try:
                        chosen = transcript_list.find_generated_transcript(TRANSCRIPT_LANGUAGES)
                    except Exception:
                        chosen = None
                if chosen is not None:
                    fetched = chosen.fetch()
                    raw_entries = _normalize_transcript_entries(fetched)
                    language_used = getattr(chosen, "language_code", "available") or "available"
                    source = "list_transcripts"
            except Exception as third_error:
                metadata["transcript_third_error"] = str(third_error)

    if not raw_entries:
        limitations.append(
            "Transcript/captions were not available or could not be downloaded. Analysis is limited to title/metadata."
        )
        return "", metadata, limitations

    parts: List[str] = []
    for item in raw_entries:
        text = _clean(item.get("text", ""))
        text = re.sub(r"\[(music|applause|laughter|silence)\]", "", text, flags=re.I).strip()
        if text:
            parts.append(text)

    transcript = _clean(" ".join(parts))
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_TRANSCRIPT_CHARS].rsplit(" ", 1)[0] + "..."
        limitations.append(
            f"Transcript was truncated to {MAX_TRANSCRIPT_CHARS} characters before claim extraction."
        )

    metadata.update(
        {
            "transcript_available": True,
            "transcript_language": language_used,
            "transcript_source": source,
            "transcript_entries": len(raw_entries),
        }
    )
    return transcript, metadata, limitations


def extract_youtube_content(route, timeout: int = 20) -> Dict[str, Any]:
    url = getattr(route, "normalized_input", "")
    video_id = extract_youtube_video_id(url)
    result = _base_youtube_result(route, video_id=video_id)

    if not video_id:
        result.update(
            {
                "status": "failed",
                "limitations": ["Could not extract a YouTube video ID from the URL."],
            }
        )
        return result

    limitations: List[str] = []

    oembed, oembed_limits = _fetch_oembed(url, timeout=timeout)
    limitations.extend(oembed_limits)

    page_meta, page_limits = _fetch_page_metadata(url, timeout=timeout)
    limitations.extend(page_limits)

    title = _clean(oembed.get("title") or page_meta.get("title") or "YouTube video")
    author = _clean(oembed.get("author_name") or "")
    description = _clean(page_meta.get("description") or "")
    published_date = _clean(page_meta.get("published_date") or "")

    transcript, transcript_meta, transcript_limits = _fetch_transcript(video_id)
    limitations.extend(transcript_limits)

    text_parts = []
    if title:
        text_parts.append(f"YouTube video title: {title}.")
    if author:
        text_parts.append(f"Channel/source: {author}.")
    if published_date:
        text_parts.append(f"Published or uploaded date: {published_date}.")
    if description:
        text_parts.append(f"Video description: {description}")
    if transcript:
        text_parts.append(f"Video transcript: {transcript}")

    text = _clean(" ".join(text_parts))
    status = "success" if transcript else "partial" if text else "failed"

    if not transcript:
        limitations.append(
            "No transcript was available, so the agent can only analyze title/metadata. Paste transcript for stronger verification."
        )

    result.update(
        {
            "status": status,
            "title": title,
            "author": author,
            "published_date": published_date,
            "text": text,
            "text_length": len(text),
            "limitations": list(dict.fromkeys(limitations)),
            "metadata": {
                **result.get("metadata", {}),
                **transcript_meta,
                "video_id": video_id,
                "provider": "youtube_oembed+transcript_best_effort",
                "canonical": f"https://www.youtube.com/watch?v={video_id}",
                "oembed_author_url": oembed.get("author_url", ""),
                "thumbnail_url": oembed.get("thumbnail_url", ""),
                "description_available": bool(description),
            },
        }
    )
    return result
