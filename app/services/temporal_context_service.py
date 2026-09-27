import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _lower(value: Any) -> str:
    return _clean(value).lower()


def _dedupe(items: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _analysis_horizon_year() -> int:
    # Allow the current year and a one-year future tolerance for scheduled events.
    return datetime.now(timezone.utc).date().year + 1


def extract_years(text: str) -> List[int]:
    """Extract plausible years while filtering noisy future/reference years.

    Earlier Mega Batch 3 allowed years up to 2100 and selected the maximum year,
    which polluted pages like Wikipedia with unrelated years such as 2092. For
    misinformation verification, useful temporal cues are normally historical or
    near-current. This keeps broad historical support, but rejects far-future noise.
    """
    years: List[int] = []
    max_year = _analysis_horizon_year()
    for match in re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", text or ""):
        try:
            year = int(match)
            if 1500 <= year <= max_year and year not in years:
                years.append(year)
        except Exception:
            continue
    return years


def _parse_year_from_date(value: str) -> Optional[int]:
    value = _clean(value)
    if not value:
        return None
    years = extract_years(value)
    return years[0] if years else None


def _resolve_relative_window(cue: str, today: date) -> Dict[str, str]:
    cue = cue.lower().strip()
    if cue == "today":
        return {"cue": cue, "approx_start": today.isoformat(), "approx_end": today.isoformat()}
    if cue == "yesterday":
        day = today - timedelta(days=1)
        return {"cue": cue, "approx_start": day.isoformat(), "approx_end": day.isoformat()}
    if cue == "last week":
        start = today - timedelta(days=14)
        end = today - timedelta(days=1)
        return {"cue": cue, "approx_start": start.isoformat(), "approx_end": end.isoformat()}
    if cue == "last month":
        start = today - timedelta(days=62)
        end = today - timedelta(days=1)
        return {"cue": cue, "approx_start": start.isoformat(), "approx_end": end.isoformat()}
    if cue == "last year":
        return {"cue": cue, "approx_start": f"{today.year - 1}-01-01", "approx_end": f"{today.year - 1}-12-31"}
    return {"cue": cue, "approx_start": "", "approx_end": ""}


def _contains_any(lower_text: str, terms: List[str]) -> bool:
    for term in terms:
        term = term.lower().strip()
        if " " in term:
            if term in lower_text:
                return True
        else:
            if re.search(rf"\b{re.escape(term)}\b", lower_text):
                return True
    return False


def detect_event_terms(text: str) -> List[str]:
    lower = _lower(text)
    groups = {
        "protest": ["protest", "protests", "rally", "demonstration", "march", "unrest", "riot", "clash", "caa", "nrc", "citizenship amendment"],
        "video_claim": ["video", "clip", "footage", "viral", "shared", "circulated", "recirculated"],
        "election": ["election", "ballot", "vote", "voting", "poll", "candidate"],
        "conflict": ["war", "attack", "missile", "bomb", "nuclear", "military", "strike", "explosion"],
        "health_science": ["vaccine", "covid", "virus", "disease", "medicine", "drug", "treatment", "study"],
        "weather_disaster": ["earthquake", "flood", "cyclone", "storm", "wildfire", "landslide"],
    }
    detected: List[str] = []
    for name, terms in groups.items():
        if _contains_any(lower, terms):
            detected.append(name)
    return detected


def detect_relative_cues(text: str) -> List[str]:
    lower = _lower(text)
    cues = [
        "today", "yesterday", "last week", "last month", "last year", "years ago",
        "old video", "old clip", "viral clip", "viral video", "recirculated",
        "shared as", "claiming to show", "claimed to show", "recently", "now", "current",
    ]
    detected: List[str] = []
    for cue in cues:
        if " " in cue:
            if cue in lower:
                detected.append(cue)
        else:
            if re.search(rf"\b{re.escape(cue)}\b", lower):
                detected.append(cue)
    return detected


def _guess_event_year(years: List[int], published_year: Optional[int], now_year: int) -> Optional[int]:
    if not years:
        return None
    non_publish = [year for year in years if year != published_year]
    candidates = non_publish or years

    # Prefer explicitly historical years over near-current publication/reference years.
    historical = [year for year in candidates if year < now_year]
    if historical:
        return max(historical)

    # Current/near-future remains useful for current claims.
    near_current = [year for year in candidates if year <= now_year + 1]
    if near_current:
        return max(near_current)

    return None


def extract_temporal_context(content: str, source_metadata: Dict[str, Any]) -> Dict[str, Any]:
    today = datetime.now(timezone.utc).date()
    now_year = today.year

    content = _clean(content)
    # Guardrail: never scan an entire long article/page for source-level temporal
    # context. Long pages contain navigation, references, unrelated years, and words
    # like "current" or "today" that pollute every claim. Multi-claim verification
    # computes claim-level context separately.
    if len(content) > 1800:
        content = content[:1800]

    published = _clean(source_metadata.get("published_date"))
    title = _clean(source_metadata.get("title"))
    source_type = _clean(source_metadata.get("source_type"))
    combined = _clean(f"{published} {title} {content}")

    years = extract_years(combined)
    published_year = _parse_year_from_date(published)
    event_terms = detect_event_terms(combined)
    relative_cues = detect_relative_cues(combined)
    relative_windows = [_resolve_relative_window(cue, today) for cue in relative_cues]
    event_year = _guess_event_year(years, published_year, now_year)

    source_age_years = None
    if published_year:
        source_age_years = max(0, now_year - published_year)

    old_media_phrase = any(
        cue in relative_cues
        for cue in ["old video", "old clip", "viral clip", "viral video", "recirculated", "shared as", "claiming to show", "claimed to show"]
    )
    current_framing_cue = any(cue in relative_cues for cue in ["today", "recently", "now", "current", "last week", "last month"])

    recirculation_risk = bool(
        old_media_phrase
        or ("video_claim" in event_terms and source_age_years is not None and source_age_years >= 2)
        or ("video_claim" in event_terms and event_year is not None and event_year < now_year and current_framing_cue)
    )

    if relative_cues and not event_year:
        time_scope = "relative_time_claim"
    elif event_year and event_year < now_year:
        time_scope = "historical"
    elif event_year and event_year >= now_year:
        time_scope = "current_or_future"
    else:
        time_scope = "unknown"

    if recirculation_risk:
        time_scope = "historical_or_recirculated_media"

    requires_historical_evidence = bool(
        time_scope in {"historical", "historical_or_recirculated_media"}
        or (event_year is not None and event_year < now_year)
        or (source_age_years is not None and source_age_years >= 2)
    )

    notes: List[str] = [
        "Temporal context is a supporting signal used to improve evidence retrieval and interpretation."
    ]
    if source_age_years is not None and source_age_years >= 2:
        notes.append("The source appears older than two years; verify whether it is being reused with current-event framing.")
    if relative_cues:
        notes.append("Relative time expressions were detected; current-date interpretation can change over time.")
    if recirculation_risk:
        notes.append("Possible old-media or recirculated-content context detected.")

    return {
        "published_date": published,
        "published_year": published_year,
        "source_age_years": source_age_years,
        "years_mentioned": _dedupe(years),
        "event_year_guess": event_year,
        "time_scope": time_scope,
        "relative_time_cues": relative_cues,
        "relative_time_windows": relative_windows,
        "event_terms": event_terms,
        "recirculation_risk": recirculation_risk,
        "requires_historical_evidence": requires_historical_evidence,
        "source_type": source_type,
        "notes": notes,
    }
