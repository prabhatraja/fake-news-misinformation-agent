from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from tavily import TavilyClient

try:
    from exa_py import Exa
except Exception:
    Exa = None

from app.core.config import settings
from app.services.evidence_policy_loader import load_evidence_policy


_tavily_client: Optional[TavilyClient] = None
_exa_client: Optional[Any] = None


def policy() -> Dict[str, Any]:
    return load_evidence_policy()


def clean_text(value: Optional[Any]) -> str:
    if value is None:
        return ""

    text = str(value)

    if text.lower() == "nan":
        return ""

    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())

    return text.strip()


def normalize_token(token: str) -> str:
    token = token.lower().strip("'\".,;:!?()[]{}")

    if token.endswith("'s"):
        token = token[:-2]

    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("s"):
        token = token[:-1]

    return token


def extract_tokens(text: str) -> List[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", clean_text(text).lower())
    tokens = [normalize_token(token) for token in raw_tokens]
    return [token for token in tokens if token]


def stance_policy() -> Dict[str, Any]:
    return policy()["stance_policy"]


def source_trust_policy() -> Dict[str, Any]:
    return policy()["source_trust"]


def extract_content_tokens(text: str) -> Set[str]:
    stopwords = stance_policy()["stopwords"]

    return {
        token
        for token in extract_tokens(text)
        if len(token) >= 3 and token not in stopwords
    }


def contains_any_phrase(text: str, phrases: Set[str]) -> bool:
    normalized_text = f" {clean_text(text).lower()} "
    return any(phrase in normalized_text for phrase in phrases)


def has_negation(text: str) -> bool:
    tokens = set(extract_tokens(text))
    return bool(tokens.intersection(stance_policy()["negation_cues"]))


def get_tavily_client() -> Optional[TavilyClient]:
    global _tavily_client

    api_key = clean_text(getattr(settings, "tavily_api_key", ""))

    if not api_key:
        return None

    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=api_key)

    return _tavily_client


def get_exa_client() -> Optional[Any]:
    global _exa_client

    if Exa is None:
        return None

    api_key = clean_text(getattr(settings, "exa_api_key", ""))

    if not api_key:
        return None

    if _exa_client is None:
        _exa_client = Exa(api_key=api_key)

    return _exa_client


def get_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain
    except Exception:
        return ""


def domain_matches(domain: str, known_domains: Set[str]) -> bool:
    return domain in known_domains or any(domain.endswith(f".{item}") for item in known_domains)


def score_source_trust(url: str) -> float:
    domain = get_domain(url)
    trust_policy = source_trust_policy()

    if not domain:
        return float(trust_policy["unknown_domain_trust"])

    for suffix, score in trust_policy["domain_suffix_scores"].items():
        if domain.endswith(suffix):
            return float(score)

    source_groups = [
        "official_domains",
        "trusted_fact_check_domains",
        "authority_domains",
        "low_trust_domains",
    ]

    for group_name in source_groups:
        group = trust_policy[group_name]

        if domain_matches(domain, group["domains"]):
            return float(group["trust_score"])

    return float(trust_policy["default_trust"])


def estimate_freshness_score(result: Dict) -> float:
    published_date = clean_text(result.get("published_date"))

    if not published_date:
        return 0.80

    return 1.0


def calculate_final_evidence_score(
    relevance_score: float,
    trust_score: float,
    freshness_score: float,
) -> float:
    final_score = (
        relevance_score * 0.45
        + trust_score * 0.35
        + freshness_score * 0.20
    )

    return round(final_score, 4)


def calculate_claim_evidence_alignment(query_text: str, evidence_text: str) -> Tuple[float, int]:
    query_tokens = extract_content_tokens(query_text)
    evidence_tokens = extract_content_tokens(evidence_text)

    if not query_tokens or not evidence_tokens:
        return 0.0, 0

    overlap_tokens = query_tokens.intersection(evidence_tokens)
    overlap_ratio = len(overlap_tokens) / len(query_tokens)

    return round(overlap_ratio, 4), len(overlap_tokens)


def relation_alignment_exists(query_text: str, evidence_text: str) -> bool:
    query_tokens = set(extract_tokens(query_text))
    evidence_tokens = set(extract_tokens(evidence_text))
    matched_relation_group = False

    for relation in stance_policy()["relation_groups"].values():
        query_relation_tokens = relation["query"]
        evidence_relation_tokens = relation["evidence"]

        if query_tokens.intersection(query_relation_tokens):
            matched_relation_group = True

            if evidence_tokens.intersection(evidence_relation_tokens):
                return True

    return not matched_relation_group


def detect_explicit_verdict_direction(text: str) -> str:
    text = clean_text(text).lower()
    policy_data = stance_policy()

    if contains_any_phrase(text, policy_data["neutral_verdict_cues"]):
        return "NEUTRAL"

    if contains_any_phrase(text, policy_data["explicit_refute_patterns"]):
        return "REFUTES"

    if contains_any_phrase(text, policy_data["explicit_support_patterns"]):
        return "SUPPORTS"

    tokens = set(extract_tokens(text))

    if tokens.intersection(policy_data["refute_verdict_cues"]):
        return "REFUTES"

    if tokens.intersection(policy_data["support_verdict_cues"]):
        return "SUPPORTS"

    return "UNKNOWN"


def infer_support_from_aligned_assertion(query_text: str, evidence_text: str) -> bool:
    overlap_ratio, overlap_count = calculate_claim_evidence_alignment(query_text, evidence_text)
    policy_data = stance_policy()

    if overlap_count < int(policy_data["assertion_overlap_count"]):
        return False

    if overlap_ratio < float(policy_data["assertion_overlap_ratio"]):
        return False

    if not relation_alignment_exists(query_text, evidence_text):
        return False

    evidence_tokens = set(extract_tokens(evidence_text))

    if not evidence_tokens.intersection(policy_data["assertion_cues"]):
        return False

    if has_negation(query_text) != has_negation(evidence_text):
        return False

    direction = detect_explicit_verdict_direction(evidence_text)

    if direction in {"REFUTES", "NEUTRAL"}:
        return False

    return True


def infer_web_stance(query_text: str, title: str, content: str) -> str:
    evidence_text = f"{clean_text(title)} {clean_text(content)}"
    overlap_ratio, overlap_count = calculate_claim_evidence_alignment(query_text, evidence_text)
    policy_data = stance_policy()

    if overlap_count < int(policy_data["minimum_overlap_count"]):
        return "NOT_ENOUGH_INFO"

    if overlap_ratio < float(policy_data["minimum_overlap_ratio"]):
        return "NOT_ENOUGH_INFO"

    if not relation_alignment_exists(query_text, evidence_text):
        return "NOT_ENOUGH_INFO"

    direction = detect_explicit_verdict_direction(evidence_text)

    if direction == "NEUTRAL":
        return "NOT_ENOUGH_INFO"

    if direction == "REFUTES":
        return "REFUTES"

    if direction == "SUPPORTS":
        return "SUPPORTS"

    if infer_support_from_aligned_assertion(query_text, evidence_text):
        return "SUPPORTS"

    return "NOT_ENOUGH_INFO"


def should_use_live_search() -> bool:
    runtime_mode = clean_text(getattr(settings, "runtime_mode", "offline_eval")).lower()

    if runtime_mode != "hybrid":
        return False

    if not bool(getattr(settings, "live_search_enabled", False)):
        return False

    return True


def should_use_tavily_search() -> bool:
    provider = clean_text(getattr(settings, "live_search_provider", "tavily")).lower()

    if provider not in {"tavily", "ensemble", "tavily_exa"}:
        return False

    if not clean_text(getattr(settings, "tavily_api_key", "")):
        return False

    return True


def should_use_exa_search() -> bool:
    if not bool(getattr(settings, "exa_search_enabled", False)):
        return False

    if Exa is None:
        return False

    if not clean_text(getattr(settings, "exa_api_key", "")):
        return False

    return True


def build_search_query(query_text: str) -> str:
    query_text = clean_text(query_text)

    if len(query_text) > 220:
        query_text = query_text[:220]

    return f"{query_text} official source fact check"


def normalize_live_evidence_label(
    evidence_label: str,
    relevance_score: float,
    trust_score: float,
    final_evidence_score: float,
    min_final_score: float,
) -> str:
    if (
        relevance_score < float(getattr(settings, "live_search_min_relevance", 0.55))
        or trust_score < float(getattr(settings, "live_search_min_trust", 0.60))
        or final_evidence_score < min_final_score
    ):
        return "NOT_ENOUGH_INFO"

    return evidence_label


def make_evidence_item(
    provider: str,
    query_text: str,
    title: str,
    url: str,
    content: str,
    relevance_score: float,
    freshness_score: float,
    published_date: Optional[str],
    retrieved_at: str,
    min_final_score: float,
) -> Optional[Dict]:
    title = clean_text(title)
    url = clean_text(url)
    content = clean_text(content)

    if not content and not title:
        return None

    relevance_score = round(max(0.0, min(1.0, float(relevance_score or 0.0))), 4)
    trust_score = score_source_trust(url)

    final_evidence_score = calculate_final_evidence_score(
        relevance_score=relevance_score,
        trust_score=trust_score,
        freshness_score=freshness_score,
    )

    if (
        relevance_score < float(getattr(settings, "live_search_min_relevance", 0.55))
        and final_evidence_score < min_final_score
    ):
        return None

    evidence_label = infer_web_stance(
        query_text=query_text,
        title=title,
        content=content,
    )

    evidence_label = normalize_live_evidence_label(
        evidence_label=evidence_label,
        relevance_score=relevance_score,
        trust_score=trust_score,
        final_evidence_score=final_evidence_score,
        min_final_score=min_final_score,
    )

    source_name = get_domain(url) or provider

    return {
        "source": source_name,
        "source_type": "live_web",
        "source_name": source_name,
        "provider": provider,
        "title": title,
        "url": url,
        "text": content,
        "relevance_score": relevance_score,
        "trust_score": round(trust_score, 4),
        "freshness_score": round(freshness_score, 4),
        "final_evidence_score": final_evidence_score,
        "evidence_label": evidence_label,
        "published_date": clean_text(published_date) or None,
        "retrieved_at": retrieved_at,
    }


def search_tavily_web(query_text: str) -> List[Dict]:
    if not should_use_tavily_search():
        return []

    client = get_tavily_client()

    if client is None:
        return []

    search_query = build_search_query(query_text)

    try:
        response = client.search(
            query=search_query,
            search_depth=clean_text(getattr(settings, "live_search_depth", "basic")) or "basic",
            max_results=int(getattr(settings, "live_search_max_results", 5)),
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as error:
        print(f"Tavily web search failed: {error}")
        return []

    results = response.get("results", []) if isinstance(response, dict) else []
    retrieved_at = datetime.now(timezone.utc).isoformat()

    evidence_items: List[Dict] = []

    for result in results:
        evidence_item = make_evidence_item(
            provider="tavily",
            query_text=query_text,
            title=clean_text(result.get("title")),
            url=clean_text(result.get("url")),
            content=clean_text(result.get("content")),
            relevance_score=float(result.get("score", 0.0) or 0.0),
            freshness_score=estimate_freshness_score(result),
            published_date=clean_text(result.get("published_date")) or None,
            retrieved_at=retrieved_at,
            min_final_score=float(getattr(settings, "live_search_min_final_score", 0.60)),
        )

        if evidence_item is not None:
            evidence_items.append(evidence_item)

    return evidence_items


def extract_exa_result_text(result: Any) -> str:
    highlights = getattr(result, "highlights", None) or []

    highlight_text = " ".join(
        clean_text(highlight)
        for highlight in highlights
        if clean_text(highlight)
    )

    if highlight_text:
        return highlight_text

    text = clean_text(getattr(result, "text", ""))

    if text:
        return text

    summary = clean_text(getattr(result, "summary", ""))

    if summary:
        return summary

    return ""


def call_exa_search(client: Any, search_query: str) -> Any:
    search_type = clean_text(getattr(settings, "exa_search_type", "auto")) or "auto"
    max_results = int(getattr(settings, "exa_max_results", 5))

    try:
        return client.search(
            search_query,
            type=search_type,
            num_results=max_results,
            contents={"highlights": True},
        )
    except TypeError:
        pass

    try:
        return client.search_and_contents(
            search_query,
            type=search_type,
            num_results=max_results,
            highlights=True,
        )
    except TypeError:
        pass

    return client.search(
        search_query,
        type=search_type,
        num_results=max_results,
    )


def search_exa_web(query_text: str) -> List[Dict]:
    if not should_use_exa_search():
        return []

    client = get_exa_client()

    if client is None:
        return []

    search_query = build_search_query(query_text)

    try:
        response = call_exa_search(client=client, search_query=search_query)
    except Exception as error:
        print(f"Exa web search failed: {error}")
        return []

    results = getattr(response, "results", []) or []
    retrieved_at = datetime.now(timezone.utc).isoformat()

    evidence_items: List[Dict] = []

    for result in results:
        raw_score = getattr(result, "score", None)

        if raw_score is None:
            relevance_score = 0.62
        else:
            relevance_score = float(raw_score or 0.0)

        evidence_item = make_evidence_item(
            provider="exa",
            query_text=query_text,
            title=clean_text(getattr(result, "title", "")),
            url=clean_text(getattr(result, "url", "")),
            content=extract_exa_result_text(result),
            relevance_score=relevance_score,
            freshness_score=0.80,
            published_date=None,
            retrieved_at=retrieved_at,
            min_final_score=float(getattr(settings, "exa_min_final_score", 0.58)),
        )

        if evidence_item is not None:
            evidence_items.append(evidence_item)

    return evidence_items


def has_strong_live_evidence(evidence_items: List[Dict]) -> bool:
    directional_items = [
        item
        for item in evidence_items
        if item.get("evidence_label") in {"SUPPORTS", "REFUTES"}
    ]

    if not directional_items:
        return False

    strong_items = [
        item
        for item in directional_items
        if float(item.get("relevance_score") or 0.0) >= 0.65
        and float(item.get("final_evidence_score") or 0.0)
        >= float(getattr(settings, "live_search_min_final_score", 0.60))
    ]

    return len(strong_items) >= 2


def deduplicate_live_evidence(evidence_items: List[Dict]) -> List[Dict]:
    best_by_key: Dict[str, Dict] = {}

    for item in evidence_items:
        url = clean_text(item.get("url"))
        text = clean_text(item.get("text"))

        key = url.lower() if url else text[:300].lower()

        if not key:
            continue

        existing = best_by_key.get(key)

        if existing is None:
            best_by_key[key] = item
            continue

        item_score = float(item.get("final_evidence_score") or item.get("relevance_score") or 0.0)
        existing_score = float(existing.get("final_evidence_score") or existing.get("relevance_score") or 0.0)

        if item_score > existing_score:
            best_by_key[key] = item

    return list(best_by_key.values())


def rank_live_evidence(evidence_items: List[Dict]) -> List[Dict]:
    return sorted(
        evidence_items,
        key=lambda item: float(
            item.get("final_evidence_score")
            or item.get("relevance_score")
            or 0.0
        ),
        reverse=True,
    )


def search_live_web(query_text: str) -> List[Dict]:
    if not should_use_live_search():
        return []

    evidence_items: List[Dict] = []

    tavily_evidence = search_tavily_web(query_text=query_text)
    evidence_items.extend(tavily_evidence)

    if not has_strong_live_evidence(tavily_evidence):
        exa_evidence = search_exa_web(query_text=query_text)
        evidence_items.extend(exa_evidence)

    deduplicated = deduplicate_live_evidence(evidence_items)
    ranked = rank_live_evidence(deduplicated)

    return ranked