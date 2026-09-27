from __future__ import annotations

import html
import os
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from app.core.config import settings
from app.services.input_router_service import InputRoute


X_API_BASE = "https://api.x.com/2"
OEMBED_URL = "https://publish.twitter.com/oembed"


class TweetEmbedParser(HTMLParser):
    """Extract only the post body and canonical status link from X/Twitter oEmbed HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.in_body = False
        self.body_parts: List[str] = []
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = {str(k).lower(): str(v) for k, v in attrs if k and v is not None}
        if tag.lower() == "p":
            self.in_body = True
        if tag.lower() == "a" and attrs_dict.get("href"):
            self.links.append(html.unescape(attrs_dict["href"]))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "p":
            self.in_body = False

    def handle_data(self, data: str) -> None:
        if self.in_body and data:
            self.body_parts.append(html.unescape(data))

    def body_text(self) -> str:
        return " ".join(" ".join(self.body_parts).split()).strip()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split()).strip()


def _base_result(route: InputRoute, input_kind: str) -> Dict[str, Any]:
    return {
        "status": "not_started",
        "input_kind": input_kind,
        "url": route.normalized_input,
        "domain": (urlparse(route.normalized_input).netloc or "").lower(),
        "title": "",
        "author": "",
        "published_date": "",
        "text": "",
        "text_length": 0,
        "limitations": [],
        "metadata": {"source_type": input_kind},
    }


def parse_twitter_url(url: str) -> Dict[str, Any]:
    """Classify an X/Twitter URL as a single post or a profile URL."""
    try:
        parsed = urlparse(url)
    except Exception:
        return {"kind": "unsupported", "handle": "", "status_id": ""}

    host = (parsed.netloc or "").lower().split(":")[0]
    if host not in {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}:
        return {"kind": "unsupported", "handle": "", "status_id": ""}

    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return {"kind": "unsupported", "handle": "", "status_id": ""}

    handle = parts[0].lstrip("@").strip()
    reserved = {
        "home", "explore", "search", "notifications", "messages", "settings",
        "compose", "intent", "share", "i", "tos", "privacy",
    }
    if not handle or handle.lower() in reserved:
        return {"kind": "unsupported", "handle": handle, "status_id": ""}

    if len(parts) >= 3 and parts[1].lower() in {"status", "statuses"} and parts[2].isdigit():
        return {"kind": "post", "handle": handle, "status_id": parts[2]}

    if len(parts) == 1:
        return {"kind": "profile", "handle": handle, "status_id": ""}

    return {"kind": "unsupported", "handle": handle, "status_id": ""}


def _status_identity_from_links(links: List[str]) -> Dict[str, str]:
    for link in links:
        match = re.search(r"https?://(?:www\.)?(?:x|twitter)\.com/([^/?#]+)/status(?:es)?/(\d+)", link, re.I)
        if match:
            return {"handle": match.group(1), "status_id": match.group(2), "url": link}
    return {"handle": "", "status_id": "", "url": ""}


def _manual_fallback(
    route: InputRoute,
    *,
    requested_handle: str,
    requested_status_id: str,
    manual_text: str,
    reason: str,
) -> Dict[str, Any]:
    text = _clean(manual_text)
    result = _base_result(route, "twitter_url")
    if not text:
        result.update(
            {
                "status": "failed",
                "limitations": [
                    reason,
                    "Paste the visible post text in the manual X/Twitter text fallback to continue safely.",
                ],
                "metadata": {
                    "source_type": "twitter_url",
                    "twitter_source_kind": "post",
                    "requested_handle": requested_handle,
                    "requested_status_id": requested_status_id,
                    "identity_verified": False,
                    "extraction_quality": "failed",
                    "manual_text_fallback": False,
                },
            }
        )
        return result

    result.update(
        {
            "status": "partial",
            "title": (
                f"X/Twitter post text supplied manually (@{requested_handle})"
                if requested_handle
                else "X/Twitter post text supplied manually"
            ),
            "author": requested_handle,
            "text": text,
            "text_length": len(text),
            "limitations": [
                reason,
                "The post text was supplied manually. Its association with the X/Twitter URL was not independently confirmed.",
                "Attached images, video, quoted-post context, replies, and thread context were not verified.",
            ],
            "metadata": {
                "source_type": "twitter_url",
                "twitter_source_kind": "post",
                "requested_handle": requested_handle,
                "requested_status_id": requested_status_id,
                "resolved_handle": "",
                "resolved_status_id": "",
                "identity_verified": False,
                "extraction_quality": "manual_fallback",
                "manual_text_fallback": True,
            },
        }
    )
    return result


def extract_single_post(
    route: InputRoute,
    *,
    manual_text: str = "",
    timeout: int = 15,
) -> Dict[str, Any]:
    parsed = parse_twitter_url(route.normalized_input)
    handle = parsed.get("handle", "")
    status_id = parsed.get("status_id", "")
    result = _base_result(route, "twitter_url")

    if parsed.get("kind") != "post":
        return _manual_fallback(
            route,
            requested_handle=handle,
            requested_status_id=status_id,
            manual_text=manual_text,
            reason="The supplied X/Twitter URL is not a valid single-post URL containing /status/<post_id>.",
        )

    try:
        response = requests.get(
            OEMBED_URL,
            params={"url": route.normalized_input, "omit_script": "true", "dnt": "true"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        embedded_html = payload.get("html", "") or ""
        parser = TweetEmbedParser()
        parser.feed(embedded_html)
        post_text = _clean(parser.body_text())
        resolved = _status_identity_from_links(parser.links)
        resolved_id = resolved.get("status_id", "")
        resolved_handle = resolved.get("handle", "")

        if not resolved_id or resolved_id != status_id:
            return _manual_fallback(
                route,
                requested_handle=handle,
                requested_status_id=status_id,
                manual_text=manual_text,
                reason=(
                    "X/Twitter returned content whose post ID did not match the requested URL; "
                    "the returned content was rejected and was not classified."
                ),
            )

        if len(post_text) < 5:
            return _manual_fallback(
                route,
                requested_handle=handle,
                requested_status_id=status_id,
                manual_text=manual_text,
                reason="X/Twitter returned the requested post but no usable post text could be extracted.",
            )

        limitations = [
            "Post text was extracted from X/Twitter public oEmbed and the returned post ID matched the requested URL.",
            "Attached images, video, quoted-post context, replies, and full thread context were not verified.",
        ]
        handle_matches = not resolved_handle or not handle or resolved_handle.lower() == handle.lower()
        if not handle_matches:
            limitations.append(
                "The account handle in the canonical embed differs from the handle in the supplied URL. "
                "The post ID matched, so the post was accepted; accounts can be renamed."
            )

        result.update(
            {
                "status": "success",
                "title": _clean(payload.get("author_name") or f"X post by @{resolved_handle or handle}"),
                "author": _clean(payload.get("author_name") or resolved_handle or handle),
                "text": post_text,
                "text_length": len(post_text),
                "limitations": limitations,
                "metadata": {
                    "source_type": "twitter_url",
                    "twitter_source_kind": "post",
                    "provider": "publish.twitter.com/oembed",
                    "author_url": payload.get("author_url", ""),
                    "requested_handle": handle,
                    "requested_status_id": status_id,
                    "resolved_handle": resolved_handle,
                    "resolved_status_id": resolved_id,
                    "canonical_post_url": resolved.get("url", ""),
                    "identity_verified": True,
                    "handle_matches": handle_matches,
                    "extraction_quality": "reliable",
                    "manual_text_fallback": False,
                },
            }
        )
        return result
    except Exception as error:
        return _manual_fallback(
            route,
            requested_handle=handle,
            requested_status_id=status_id,
            manual_text=manual_text,
            reason=f"X/Twitter public post extraction failed: {error}",
        )


def _x_bearer_token() -> str:
    return _clean(
        getattr(settings, "x_bearer_token", "")
        or os.getenv("X_BEARER_TOKEN", "")
        or os.getenv("TWITTER_BEARER_TOKEN", "")
    )


def extract_profile(
    route: InputRoute,
    *,
    post_limit: int = 5,
    timeout: int = 15,
) -> Dict[str, Any]:
    parsed = parse_twitter_url(route.normalized_input)
    handle = parsed.get("handle", "")
    result = _base_result(route, "twitter_profile_url")
    limit = max(1, min(int(post_limit or 5), 10))
    token = _x_bearer_token()

    if parsed.get("kind") != "profile":
        result.update(
            {
                "status": "failed",
                "limitations": ["The supplied X/Twitter URL is not a supported profile URL."],
                "metadata": {"source_type": "twitter_profile_url", "twitter_source_kind": "profile"},
            }
        )
        return result

    if not token:
        result.update(
            {
                "status": "failed",
                "title": f"X profile @{handle}",
                "author": handle,
                "limitations": [
                    "X/Twitter profile analysis requires the official X API because public oEmbed supports individual posts, not timelines.",
                    "Configure X_BEARER_TOKEN in .env, then restart FastAPI. "
                    "Single-post X/Twitter URLs can still use public oEmbed without an API token.",
                ],
                "metadata": {
                    "source_type": "twitter_profile_url",
                    "twitter_source_kind": "profile",
                    "requested_handle": handle,
                    "x_api_configured": False,
                    "posts_requested": limit,
                    "profile_posts": [],
                    "extraction_quality": "failed",
                    "failure_reason": "x_api_token_missing",
                    "http_status": None,
                },
            }
        )
        return result

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "FakeNewsMisinformationAgent/1.0",
    }
    try:
        user_response = requests.get(
            f"{X_API_BASE}/users/by/username/{handle}",
            headers=headers,
            params={"user.fields": "id,name,username,protected,verified,description"},
            timeout=timeout,
        )
        user_response.raise_for_status()
        user_payload = user_response.json()
        user = user_payload.get("data") or {}
        user_id = _clean(user.get("id"))
        canonical_username = _clean(user.get("username")) or handle
        if not user_id:
            raise RuntimeError("X API user lookup returned no user ID.")

        if bool(user.get("protected")):
            result.update(
                {
                    "status": "failed",
                    "title": _clean(user.get("name") or f"X profile @{canonical_username}"),
                    "author": canonical_username,
                    "limitations": [
                        "The X/Twitter profile is protected; public timeline analysis is not available."
                    ],
                    "metadata": {
                        "source_type": "twitter_profile_url",
                        "twitter_source_kind": "profile",
                        "requested_handle": handle,
                        "resolved_handle": canonical_username,
                        "user_id": user_id,
                        "x_api_configured": True,
                        "profile_posts": [],
                        "extraction_quality": "failed",
                        "failure_reason": "protected_profile",
                        "http_status": 403,
                    },
                }
            )
            return result

        api_max_results = max(5, limit)
        posts_response = requests.get(
            f"{X_API_BASE}/users/{user_id}/tweets",
            headers=headers,
            params={
                "max_results": api_max_results,
                "exclude": "replies,retweets",
                "post.fields": "created_at,lang,conversation_id,public_metrics,text",
            },
            timeout=timeout,
        )
        posts_response.raise_for_status()
        posts_payload = posts_response.json()
        raw_posts = posts_payload.get("data") or []

        posts: List[Dict[str, Any]] = []
        for item in raw_posts:
            text = _clean(item.get("text"))
            post_id = _clean(item.get("id"))
            if not text or not post_id:
                continue
            posts.append(
                {
                    "post_id": post_id,
                    "url": f"https://x.com/{canonical_username}/status/{post_id}",
                    "text": text,
                    "created_at": _clean(item.get("created_at")),
                    "lang": _clean(item.get("lang")),
                    "conversation_id": _clean(item.get("conversation_id")),
                    "public_metrics": item.get("public_metrics") or {},
                }
            )
            if len(posts) >= limit:
                break

        if not posts:
            result.update(
                {
                    "status": "failed",
                    "title": _clean(user.get("name") or f"X profile @{canonical_username}"),
                    "author": canonical_username,
                    "limitations": [
                        "No recent public original posts were returned for this profile. "
                        "Replies and reposts are excluded from the demo sample."
                    ],
                    "metadata": {
                        "source_type": "twitter_profile_url",
                        "twitter_source_kind": "profile",
                        "requested_handle": handle,
                        "resolved_handle": canonical_username,
                        "user_id": user_id,
                        "x_api_configured": True,
                        "posts_requested": limit,
                        "posts_retrieved": 0,
                        "profile_posts": [],
                        "extraction_quality": "failed",
                        "failure_reason": "no_public_posts",
                        "http_status": 200,
                    },
                }
            )
            return result

        joined_text = "\n\n".join(
            f"Post {index}: {post['text']}" for index, post in enumerate(posts, start=1)
        )
        result.update(
            {
                "status": "success",
                "title": _clean(user.get("name") or f"X profile @{canonical_username}"),
                "author": canonical_username,
                "text": joined_text,
                "text_length": len(joined_text),
                "limitations": [
                    f"Profile analysis is limited to the latest {len(posts)} public original posts retrieved in this sample; it does not characterize the entire account.",
                    "Replies and reposts are excluded. Attached media, quoted-post context, and linked-page contents are not independently verified.",
                ],
                "metadata": {
                    "source_type": "twitter_profile_url",
                    "twitter_source_kind": "profile",
                    "provider": "X API v2",
                    "requested_handle": handle,
                    "resolved_handle": canonical_username,
                    "user_id": user_id,
                    "profile_name": _clean(user.get("name")),
                    "verified": bool(user.get("verified")),
                    "x_api_configured": True,
                    "posts_requested": limit,
                    "posts_retrieved": len(posts),
                    "profile_posts": posts,
                    "extraction_quality": "reliable",
                },
            }
        )
        return result
    except Exception as error:
        http_status = None
        failure_reason = "x_api_error"
        friendly_limitations = []

        if isinstance(error, requests.HTTPError) and error.response is not None:
            http_status = error.response.status_code
            detail = ""
            try:
                payload = error.response.json()
                detail = _clean(payload.get("detail") or payload.get("title") or "")
            except Exception:
                detail = _clean(error.response.text)

            if http_status == 402 or "credits depleted" in detail.lower():
                failure_reason = "x_api_credits_required"
                friendly_limitations = [
                    "X/Twitter profile analysis is configured correctly, but the official X API requires available API credits for timeline retrieval.",
                    "Single public X/Twitter post verification remains available through public oEmbed without paid API credits.",
                ]
            elif http_status == 401:
                failure_reason = "x_api_authentication_failed"
                friendly_limitations = [
                    "X API authentication failed. Check X_BEARER_TOKEN and restart FastAPI after changing .env.",
                    "Single public X/Twitter post verification remains available through public oEmbed.",
                ]
            elif http_status == 403:
                failure_reason = "x_api_access_forbidden"
                friendly_limitations = [
                    "The X API denied access to this profile request for the current developer app/account.",
                    "Single public X/Twitter post verification remains available through public oEmbed.",
                ]
            elif http_status == 429:
                failure_reason = "x_api_rate_limited"
                friendly_limitations = [
                    "The X API rate limit was reached. Retry later or analyze a single public post instead.",
                ]

        if not friendly_limitations:
            friendly_limitations = [
                f"X API profile extraction failed: {error}",
                "Check X_BEARER_TOKEN, X API access/credits, and network availability. Single-post X/Twitter URLs can still use public oEmbed.",
            ]

        result.update(
            {
                "status": "failed",
                "title": f"X profile @{handle}",
                "author": handle,
                "limitations": friendly_limitations,
                "metadata": {
                    "source_type": "twitter_profile_url",
                    "twitter_source_kind": "profile",
                    "requested_handle": handle,
                    "x_api_configured": True,
                    "posts_requested": limit,
                    "profile_posts": [],
                    "extraction_quality": "failed",
                    "failure_reason": failure_reason,
                    "http_status": http_status,
                },
            }
        )
        return result


def extract_twitter_source(
    route: InputRoute,
    *,
    manual_text: str = "",
    profile_post_limit: int = 5,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    timeout_value = int(timeout or getattr(settings, "x_api_timeout_seconds", 15) or 15)
    parsed = parse_twitter_url(route.normalized_input)
    kind = parsed.get("kind")

    if kind == "post":
        return extract_single_post(route, manual_text=manual_text, timeout=timeout_value)
    if kind == "profile":
        return extract_profile(route, post_limit=profile_post_limit, timeout=timeout_value)

    result = _base_result(route, "twitter_url")
    result.update(
        {
            "status": "failed",
            "limitations": [
                "Unsupported X/Twitter URL. Use a single post URL such as https://x.com/user/status/123... "
                "or a profile URL such as https://x.com/user."
            ],
            "metadata": {
                "source_type": "twitter_url",
                "twitter_source_kind": "unsupported",
                "extraction_quality": "failed",
            },
        }
    )
    return result
