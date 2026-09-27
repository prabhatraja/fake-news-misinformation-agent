import re
from typing import Any, Dict, List


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _clip(value: str, max_chars: int = 500) -> str:
    value = _clean(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def _add_unique(items: List[str], value: str) -> None:
    value = _clean(value)
    if value and value not in items:
        items.append(value)


def _contains_term(text: str, term: str) -> bool:
    """Match location/entity terms without substring pollution.

    The previous implementation used ``term in lower_text``. That made the
    short token ``US`` match words such as ``August``. This boundary-aware
    matcher keeps useful aliases while preventing those false positives.
    """
    text = _clean(text).lower()
    term = _clean(term).lower()
    if not text or not term:
        return False

    if re.fullmatch(r"[a-z0-9]+", term):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))

    return term in text


def _extract_possible_locations(text: str) -> List[str]:
    known = [
        "India", "Delhi", "New Delhi", "Bengaluru", "Bangalore", "Mumbai",
        "Kolkata", "Chennai", "Agra", "Jaipur", "Paris", "France", "Iran",
        "Israel", "Ukraine", "Russia", "United States", "US", "USA", "UK",
        "London", "Gaza", "China", "Pakistan", "Mars", "Moon",
    ]
    return [item for item in known if _contains_term(text, item)]


def _claim_has_temporal_relevance(claim: str, temporal_context: Dict[str, Any]) -> bool:
    event_terms = set(temporal_context.get("event_terms") or [])
    relative_cues = list(temporal_context.get("relative_time_cues") or [])
    event_year = temporal_context.get("event_year_guess")
    time_scope = _clean(temporal_context.get("time_scope"))
    return bool(
        event_terms
        or relative_cues
        or event_year
        or time_scope
        in {"historical", "historical_or_recirculated_media", "relative_time_claim"}
    )


def _focused_fact_check_terms(
    *,
    input_kind: str,
    temporal_context: Dict[str, Any],
    recirculation_risk: bool,
    event_terms: List[str],
) -> List[str]:
    """Return only high-value query terms.

    Generic additions such as ``historical context archive`` were useful for
    recirculated-media investigations but degraded ordinary historical facts.
    They are now reserved for actual recirculation/source-context cases.
    """
    terms: List[str] = []

    if recirculation_risk:
        for value in ["fact check", "original footage", "old video", "recirculated"]:
            _add_unique(terms, value)

    if "protest" in event_terms:
        for value in ["fact check", "news report"]:
            _add_unique(terms, value)

    if "conflict" in event_terms:
        for value in ["fact check", "official statement"]:
            _add_unique(terms, value)

    if "health_science" in event_terms:
        for value in ["scientific evidence", "public health"]:
            _add_unique(terms, value)

    if input_kind == "youtube_url" or "video_claim" in event_terms:
        _add_unique(terms, "fact check")

    # Historical alone should not inject generic archive terms. A year is enough.
    if (
        temporal_context.get("requires_historical_evidence")
        and (recirculation_risk or temporal_context.get("source_age_years"))
    ):
        _add_unique(terms, "archive")

    return terms


def build_verification_context(
    claim_text: str,
    source_metadata: Dict[str, Any],
    temporal_context: Dict[str, Any],
    input_kind: str,
) -> Dict[str, Any]:
    """Build a claim-centered retrieval context with conservative enrichment."""
    claim = _clean(claim_text)
    title = _clean(source_metadata.get("title"))
    domain = _clean(source_metadata.get("domain"))
    published_year = temporal_context.get("published_year")
    event_year = temporal_context.get("event_year_guess")
    time_scope = _clean(temporal_context.get("time_scope")) or "unknown"
    event_terms = list(temporal_context.get("event_terms") or [])
    relative_cues = list(temporal_context.get("relative_time_cues") or [])
    recirculation_risk = bool(temporal_context.get("recirculation_risk"))

    keywords: List[str] = []
    query_notes: List[str] = []
    temporal_relevant = _claim_has_temporal_relevance(claim, temporal_context)

    if temporal_relevant and event_year and str(event_year) not in claim:
        _add_unique(keywords, str(event_year))
    if temporal_relevant and published_year and published_year != event_year and str(published_year) not in claim:
        _add_unique(keywords, str(published_year))

    # Locations/entities already present in the claim do not need to be appended.
    # Keeping retrieval claim-centered prevents accidental source-title pollution.

    for value in _focused_fact_check_terms(
        input_kind=input_kind,
        temporal_context=temporal_context,
        recirculation_risk=recirculation_risk,
        event_terms=event_terms,
    ):
        _add_unique(keywords, value)

    if "protest" in event_terms:
        query_notes.append("Protest/news cues detected; retrieval includes a fact-check/news context term.")
    if "conflict" in event_terms:
        query_notes.append("Conflict/security cues detected; independent evidence and official statements are preferred where available.")
    if "health_science" in event_terms:
        query_notes.append("Health/science cues detected; scientific or public-health evidence is preferred where available.")
    if input_kind == "youtube_url" or "video_claim" in event_terms:
        query_notes.append("Video/source-context cues detected; transcript claims are verified against independent evidence.")
    if recirculation_risk:
        query_notes.append("Possible recirculated-media context detected; retrieval searches for original/older context.")

    # Relative expressions remain in the original claim text and therefore do not
    # need to be appended again.

    if title and input_kind in {"url", "youtube_url", "twitter_url"}:
        # Add a compact title only when it contributes terms that are not already in
        # the claim. This helps identify an article/video source without dominating.
        title_slice = re.sub(r"[^A-Za-z0-9 ,.'\-:/]", " ", title)
        title_slice = _clip(title_slice, 70)
        if title_slice and title_slice.lower() not in claim.lower():
            _add_unique(keywords, title_slice)

    retrieval_query_text = _clip(" ".join([claim] + keywords), 650)

    query_focus = "standard"
    if recirculation_risk:
        query_focus = "recirculated_media"
    elif relative_cues:
        query_focus = "relative_time_claim"
    elif temporal_context.get("requires_historical_evidence"):
        query_focus = "historical_event"

    return {
        "query_focus": query_focus,
        "retrieval_query_text": retrieval_query_text,
        "keywords_added": keywords,
        "retrieval_keywords": keywords,
        "query_notes": query_notes,
        "time_scope": time_scope,
        "event_year_guess": event_year,
        "published_year": published_year,
        "domain_or_channel": domain,
        "source_title": title,
        "query_quality": "claim_centered",
    }
