from typing import Any, Dict, List
from urllib.parse import urlparse


DIRECTIONAL_LABELS = {"SUPPORTS", "REFUTES"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _source_identity(item: Dict[str, Any]) -> str:
    url = str(item.get("url") or "").strip()
    if url:
        try:
            host = (urlparse(url).hostname or "").lower()
            if host.startswith("www."):
                host = host[4:]
            if host:
                return host
        except Exception:
            pass

    source = str(item.get("source") or item.get("source_name") or "").strip().lower()
    if source.startswith("www."):
        source = source[4:]
    return source or "unknown"


def select_source_diverse_evidence(
    evidence: List[Dict[str, Any]],
    *,
    limit: int = 3,
    min_relevance: float = 0.5,
) -> List[Dict[str, Any]]:
    """Return a ranked, source-diverse main-view evidence subset.

    The backend may retain several pages from one domain for semantic auditing, but the
    demo UI should not present those pages as independent corroboration. Directional,
    usable evidence is preferred. If no directional usable evidence exists, usable NEI
    evidence is shown. Only when nothing is usable do we fall back to weak evidence.
    """
    limit = max(1, int(limit or 1))
    items = [item for item in evidence if isinstance(item, dict)]
    if not items:
        return []

    usable = [item for item in items if _safe_float(item.get("relevance_score")) >= float(min_relevance)]
    directional = [
        item for item in usable
        if str(item.get("evidence_label") or "").upper() in DIRECTIONAL_LABELS
    ]
    candidates = directional or usable or items

    selected: List[Dict[str, Any]] = []
    seen = set()
    for item in candidates:
        identity = _source_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def build_evidence_coverage_text(summary: Dict[str, Any], fallback_retrieved: int) -> str:
    retrieved = int(summary.get("retrieved_count", fallback_retrieved) or 0)
    usable = int(summary.get("usable_evidence_count", 0) or 0)
    parts = [f"{retrieved} retrieved", f"{usable} usable"]

    consensus = str(summary.get("directional_consensus") or "").upper()
    if consensus == "SUPPORTS":
        count = int(summary.get("supports_count", 0) or 0)
        sources = int(summary.get("support_source_count", 0) or 0)
        parts.append(f"{count} support across {sources} independent source{'s' if sources != 1 else ''}")
    elif consensus == "REFUTES":
        count = int(summary.get("refutes_count", 0) or 0)
        sources = int(summary.get("refute_source_count", 0) or 0)
        parts.append(f"{count} contradict across {sources} independent source{'s' if sources != 1 else ''}")
    elif consensus == "MIXED":
        parts.append(
            f"{int(summary.get('supports_count', 0) or 0)} support / "
            f"{int(summary.get('refutes_count', 0) or 0)} contradict"
        )

    return " • ".join(parts)
