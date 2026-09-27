import json
import re
from typing import Any, Dict, List

import requests

from app.services.slm_verifier_service import (
    clean_text,
    get_runtime_slm_config,
    normalize_lmstudio_base_url,
)


FALLBACK_CUES = {
    "is", "are", "was", "were", "has", "have", "had", "will", "said", "announced",
    "reported", "claimed", "located", "born", "died", "contains", "caused", "tested",
}

VERIFICATION_TARGETS = {
    "direct_proposition",
    "embedded_proposition",
    "attribution",
}

ASSERTION_STATUSES = {
    "asserted",
    "denied",
    "questioned",
    "unclear",
}


CLAIM_REPRESENTATION_POLICY = """
Semantic verification-target policy:
- verification_target describes the factual truth question that downstream retrieval and evidence verification should evaluate; it is not a grammatical label for the sentence.
- Use direct_proposition when source_text itself directly states the proposition to verify.
- Use embedded_proposition when removing a reporting/source/communication frame leaves a complete, independently checkable proposition and evidence supporting or refuting that proposition answers the same substantive truth question. A named carrier, source, speaker, or reporting action is not material merely because it appears in the sentence.
- Use attribution only when the identity, authorship, existence, timing, or content of the communication act itself is the factual proposition to verify and there is no independent embedded proposition whose truth answers the substantive question.
- assertion_status describes the status of the proposition proposed in verification_text. Use asserted only when that proposition is presented positively for verification. Use denied, questioned, or unclear when appropriate.
- Never promote a denied, questioned, hypothetical, conditional, possible, planned, intended, merely capable, predicted, or otherwise non-asserted proposition into an asserted fact.
- For embedded_proposition, prefer an extractive deletion of framing words over paraphrasing. Never invent or alter entities, dates, quantities, locations, negation, modality, or relationships.
""".strip()


def _truncate(text: str, max_chars: int = 6000) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _extract_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("No JSON object found in claim extraction response.")


def _call_lmstudio_for_claims(content: str, max_claims: int) -> Dict[str, Any]:
    cfg = get_runtime_slm_config()
    base_url = normalize_lmstudio_base_url(cfg["base_url"])
    url = f"{base_url}/chat/completions"

    prompt = f"""/no_think
Extract up to {max_claims} factual, checkable claims from the content below and represent each claim in a way that separates source framing from the proposition that should be verified.

Rules:
1. Return only claims that can be checked using external evidence.
2. Ignore opinions, emotions, vague slogans, jokes, questions, and generic summaries.
3. Prefer central who/what/when/where claims, event claims, health/science claims, location claims, and material communication/attribution claims. Use numeric/statistical claims only when they are central or materially important.
4. If the content contains multiple independent factual statements, return them as separate claims.
5. source_text must stay faithful to the source wording for that claim.
6. verification_text is the factual proposition that evidence retrieval and verification should evaluate.
7. Apply this shared semantic policy exactly:
{CLAIM_REPRESENTATION_POLICY}
8. Return valid JSON only. No markdown.

Return this JSON shape:
{{
  "claims": [
    {{
      "source_text": "faithful source wording for the claim",
      "verification_text": "atomic proposition to verify",
      "verification_target": "direct_proposition|embedded_proposition|attribution",
      "assertion_status": "asserted|denied|questioned|unclear",
      "representation_confidence": 0.0,
      "claim_type": "event|location|health|political|numeric|quote|other",
      "importance": "high|medium|low",
      "reason_for_selection": "short reason",
      "transformation_reason": "why verification_text differs from source_text, or empty if unchanged"
    }}
  ]
}}

Content:
{_truncate(content)}
""".strip()

    payload = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "system",
                "content": "/no_think You extract and structurally represent factual claims. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": min(max(int(cfg.get("max_tokens", 800)), 700), 1400),
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=int(cfg.get("timeout_seconds", 120)))
    response.raise_for_status()
    data = response.json()
    raw = data["choices"][0]["message"].get("content", "")
    return _extract_json(raw)


def _call_lmstudio_for_claim_representations(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Semantically represent already-isolated short claims without re-segmenting them."""
    cfg = get_runtime_slm_config()
    base_url = normalize_lmstudio_base_url(cfg["base_url"])
    url = f"{base_url}/chat/completions"

    compact_candidates = [
        {
            "candidate_id": index,
            "source_text": clean_text(item.get("source_text") or item.get("claim_text", "")),
        }
        for index, item in enumerate(candidates, start=1)
    ]

    prompt = f"""/no_think
Represent each candidate statement below for evidence verification.

Rules:
1. Return exactly one output for every candidate_id. Do not merge, split, add, or drop candidates.
2. Copy source_text faithfully from the corresponding candidate.
3. verification_text is the factual proposition that evidence retrieval and verification should evaluate.
4. Apply this shared semantic policy exactly:
{CLAIM_REPRESENTATION_POLICY}
5. Return valid JSON only. No markdown.

Return:
{{
  "claims": [
    {{
      "candidate_id": 1,
      "source_text": "copied candidate text",
      "verification_text": "proposition to verify",
      "verification_target": "direct_proposition|embedded_proposition|attribution",
      "assertion_status": "asserted|denied|questioned|unclear",
      "representation_confidence": 0.0,
      "transformation_reason": "short reason or empty if unchanged"
    }}
  ]
}}

Candidates:
{json.dumps(compact_candidates, ensure_ascii=False)}
""".strip()

    payload = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "system",
                "content": "/no_think You structurally represent factual claims. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": min(max(int(cfg.get("max_tokens", 800)), 700), 1400),
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=int(cfg.get("timeout_seconds", 120)))
    response.raise_for_status()
    data = response.json()
    raw = data["choices"][0]["message"].get("content", "")
    return _extract_json(raw)


def _represent_short_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parsed = _call_lmstudio_for_claim_representations(candidates)
    raw_claims = parsed.get("claims", []) if isinstance(parsed, dict) else []
    if not isinstance(raw_claims, list):
        raise ValueError("Claim representation response did not contain a claims list.")

    by_id: Dict[int, Dict[str, Any]] = {}
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        try:
            candidate_id = int(item.get("candidate_id"))
        except Exception:
            continue
        if candidate_id not in by_id:
            by_id[candidate_id] = item

    if len(by_id) != len(candidates):
        raise ValueError("Claim representation response did not cover every short-input candidate exactly once.")

    merged: List[Dict[str, Any]] = []
    for candidate_id, candidate in enumerate(candidates, start=1):
        representation = dict(by_id[candidate_id])
        # The deterministic sentence splitter is authoritative for source boundaries.
        # The SLM may choose the verification proposition but may not rewrite the source span.
        representation["source_text"] = clean_text(candidate.get("source_text") or candidate.get("claim_text", ""))
        representation["claim_type"] = candidate.get("claim_type", "other")
        representation["importance"] = candidate.get("importance", "medium")
        representation["reason_for_selection"] = candidate.get("reason_for_selection", "Short factual statement selected deterministically.")
        merged.append(representation)

    normalized = normalize_claims({"claims": merged}, max_claims=len(candidates))
    if len(normalized) != len(candidates):
        raise ValueError("Normalized claim representation did not preserve short-input candidate count.")
    return normalized


def _sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", clean_text(text))
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip(" .!?\n\t") for p in parts if len(p.strip()) > 20]


def _infer_claim_type(sentence: str) -> str:
    text = clean_text(sentence).lower()
    if re.search(r"\b(19|20)\d{2}\b", text) or any(term in text for term in {"constructed", "started", "completed", "happened", "occurred", "protest", "tested", "announced"}):
        return "event"
    if any(term in text for term in {"located", "location", "city", "country", "paris", "delhi", "france", "india"}):
        return "location"
    if any(term in text for term in {"vaccine", "health", "covid", "disease", "virus", "nuclear", "bomb", "scientific"}):
        return "health" if "nuclear" not in text and "bomb" not in text else "event"
    if re.search(r"\b\d+(?:\.\d+)?%?\b", text):
        return "numeric"
    if any(term in text for term in {"said", "claimed", "claims", "reported", "quote"}):
        return "quote"
    return "other"


def _infer_importance(sentence: str) -> str:
    text = clean_text(sentence).lower()
    high_terms = {"bomb", "nuclear", "vaccine", "covid", "protest", "riot", "war", "election", "killed", "death", "viral video", "misleading"}
    if any(term in text for term in high_terms):
        return "high"
    if re.search(r"\b(19|20)\d{2}\b", text) or any(term in text for term in {"located", "constructed", "completed", "started"}):
        return "high"
    return "medium"


def _fallback_claim_record(sentence: str, reason: str) -> Dict[str, Any]:
    source_text = clean_text(sentence)
    return {
        "claim_text": source_text,
        "source_text": source_text,
        "verification_text": source_text,
        "verification_target": "direct_proposition",
        "assertion_status": "unclear",
        "representation_confidence": 0.0,
        "representation_fallback": True,
        "transformation_reason": "Semantic claim representation was unavailable; source text is verified directly.",
        "claim_type": _infer_claim_type(source_text),
        "importance": _infer_importance(source_text),
        "reason_for_selection": reason,
    }


def _fallback_claims(content: str, max_claims: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for sentence in _sentence_split(content):
        words = sentence.split()
        if len(words) < 6 or len(words) > 45:
            continue

        lower_words = {w.strip(".,:;!?()[]{}\"'").lower() for w in words}
        has_year = bool(re.search(r"\b(19|20)\d{2}\b", sentence))
        has_number = bool(re.search(r"\b\d+(?:\.\d+)?%?\b", sentence))
        has_cue = bool(lower_words.intersection(FALLBACK_CUES))
        has_capital = bool(re.search(r"\b[A-Z][a-z]{2,}\b", sentence))

        if not (has_year or has_number or has_cue or has_capital):
            continue

        candidates.append(
            _fallback_claim_record(
                sentence.strip() + ".",
                "Fallback sentence selection based on factual cues.",
            )
        )

        if len(candidates) >= max_claims:
            break

    if not candidates and clean_text(content):
        candidates.append(
            _fallback_claim_record(
                clean_text(content),
                "Input treated as a single claim because no better extraction was available.",
            )
        )

    return candidates[:max_claims]


def _is_low_value_numeric_claim(claim_text: str, claim_type: str, importance: str) -> bool:
    """Avoid over-prioritizing minor statistics from long pages.

    Numeric claims are useful, but in a multi-format URL analysis they can cause
    the whole page to look fake due to tiny source/version discrepancies.
    """
    text = clean_text(claim_text).lower()
    if claim_type != "numeric":
        return False
    if importance == "high":
        return False
    low_value_terms = {
        "visitor", "visitors", "received", "ranked", "ranking", "metres", "meters",
        "height", "length", "area", "population", "views", "followers", "likes",
    }
    return any(term in text for term in low_value_terms)


def _representation_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", clean_text(text).lower())


def _is_token_subsequence(candidate: str, source: str) -> bool:
    """Require embedded verification text to be extractive, not invented/paraphrased."""
    candidate_tokens = _representation_tokens(candidate)
    source_tokens = _representation_tokens(source)
    if len(candidate_tokens) < 2 or not source_tokens:
        return False

    cursor = 0
    for token in source_tokens:
        if cursor < len(candidate_tokens) and token == candidate_tokens[cursor]:
            cursor += 1
            if cursor == len(candidate_tokens):
                return True
    return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return max(0.0, min(number, 1.0))


def _normalize_claim_representation(item: Dict[str, Any]) -> Dict[str, Any]:
    source_text = clean_text(
        item.get("source_text")
        or item.get("claim_text")
        or item.get("claim")
        or item.get("text")
    )
    requested_verification = clean_text(item.get("verification_text") or source_text)
    target = clean_text(item.get("verification_target") or "direct_proposition").lower()
    assertion_status = clean_text(item.get("assertion_status") or "unclear").lower()
    confidence = _safe_float(item.get("representation_confidence"), default=0.0)
    transformation_reason = clean_text(item.get("transformation_reason") or "")

    if target not in VERIFICATION_TARGETS:
        target = "direct_proposition"
    if assertion_status not in ASSERTION_STATUSES:
        assertion_status = "unclear"

    verification_text = source_text
    representation_fallback = False

    if target == "embedded_proposition":
        safe_embedded = (
            assertion_status == "asserted"
            and requested_verification
            and requested_verification != source_text
            and _is_token_subsequence(requested_verification, source_text)
        )
        if safe_embedded:
            verification_text = requested_verification
        else:
            representation_fallback = True
            target = "direct_proposition"
            verification_text = source_text
            fallback_reason = (
                "Proposed embedded proposition was not a safe extractive assertion; source text is verified directly."
            )
            transformation_reason = " ".join(
                part for part in [transformation_reason, fallback_reason] if part
            )
    else:
        # Direct propositions and material attributions remain faithful to the surface claim.
        verification_text = source_text

    if not transformation_reason and verification_text != source_text:
        transformation_reason = "Source framing was separated from the asserted proposition for verification."

    return {
        "claim_text": verification_text,
        "source_text": source_text,
        "verification_text": verification_text,
        "verification_target": target,
        "assertion_status": assertion_status,
        "representation_confidence": confidence,
        "representation_fallback": representation_fallback,
        "transformation_reason": transformation_reason,
    }


def normalize_claims(raw_claims: Any, max_claims: int) -> List[Dict[str, Any]]:
    if isinstance(raw_claims, dict):
        raw_claims = raw_claims.get("claims", [])

    if not isinstance(raw_claims, list):
        return []

    normalized: List[Dict[str, Any]] = []

    for item in raw_claims:
        if isinstance(item, str):
            source_text = clean_text(item)
            representation = {
                "claim_text": source_text,
                "source_text": source_text,
                "verification_text": source_text,
                "verification_target": "direct_proposition",
                "assertion_status": "unclear",
                "representation_confidence": 0.0,
                "representation_fallback": True,
                "transformation_reason": "Claim returned as plain text; source text is verified directly.",
            }
            claim_type = "other"
            importance = "medium"
            reason = "Claim returned as plain text by extractor."
        elif isinstance(item, dict):
            representation = _normalize_claim_representation(item)
            source_text = representation["source_text"]
            claim_type = clean_text(item.get("claim_type") or "other").lower()
            importance = clean_text(item.get("importance") or "medium").lower()
            reason = clean_text(item.get("reason_for_selection") or "Selected by claim extraction.")
        else:
            continue

        if len(source_text) < 5:
            continue

        if importance not in {"high", "medium", "low"}:
            importance = "medium"

        normalized.append(
            {
                "claim_id": len(normalized) + 1,
                **representation,
                "claim_type": claim_type or "other",
                "importance": importance,
                "reason_for_selection": reason,
            }
        )

        if len(normalized) >= max_claims:
            break

    # Keep central claims ahead of secondary numeric/statistical claims.
    central = [c for c in normalized if not _is_low_value_numeric_claim(c.get("verification_text", c.get("claim_text", "")), c.get("claim_type", "other"), c.get("importance", "medium"))]
    secondary_numeric = [c for c in normalized if _is_low_value_numeric_claim(c.get("verification_text", c.get("claim_text", "")), c.get("claim_type", "other"), c.get("importance", "medium"))]
    reordered = (central + secondary_numeric)[:max_claims]
    for idx, item in enumerate(reordered, start=1):
        item["claim_id"] = idx
    return reordered


def _fallback_after_slm_failure(content: str, max_claims: int, limitations: List[str]) -> Dict[str, Any]:
    sentence_claims = _fallback_claims(content, max_claims=max_claims)
    if len(content.split()) <= 60 and len(sentence_claims) > 1:
        for idx, item in enumerate(sentence_claims, start=1):
            item["claim_id"] = idx
        limitations.append("Short multi-sentence content was split into separate verifiable claims.")
        return {
            "status": "sentence_split_fallback",
            "claims": sentence_claims,
            "limitations": limitations,
        }

    for idx, item in enumerate(sentence_claims, start=1):
        item["claim_id"] = idx
    return {
        "status": "fallback",
        "claims": sentence_claims,
        "limitations": limitations,
    }


def extract_claims(content: str, max_claims: int = 3) -> Dict[str, Any]:
    """Extract checkable claims with semantic representation and deterministic fallback."""
    content = clean_text(content)
    max_claims = max(1, min(int(max_claims or 3), 5))

    if not content:
        return {"status": "failed", "claims": [], "limitations": ["No content available for claim extraction."]}

    limitations: List[str] = []

    if len(content.split()) <= 60:
        # Preserve the existing deterministic sentence isolation for short content.
        # Qwen represents each isolated claim but does not decide how many short
        # claims exist, preventing semantic normalization from merging unrelated facts.
        candidates = _fallback_claims(content, max_claims=max_claims)
        try:
            claims = _represent_short_candidates(candidates)
            return {"status": "semantic_short", "claims": claims, "limitations": limitations}
        except Exception as error:
            limitations.append(f"SLM claim representation failed; deterministic fallback used: {error}")
            return _fallback_after_slm_failure(content, max_claims=max_claims, limitations=limitations)

    try:
        parsed = _call_lmstudio_for_claims(content, max_claims=max_claims)
        claims = normalize_claims(parsed, max_claims=max_claims)
        if claims:
            return {"status": "success", "claims": claims, "limitations": limitations}
        limitations.append("SLM claim extraction returned no usable claims; deterministic fallback used.")
    except Exception as error:
        limitations.append(f"SLM claim extraction failed; deterministic fallback used: {error}")

    return _fallback_after_slm_failure(content, max_claims=max_claims, limitations=limitations)
