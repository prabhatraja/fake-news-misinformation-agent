import re
from urllib.parse import urlparse
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_POLICY_PATH = PROJECT_ROOT / "config" / "evidence_policy.yaml"


DEFAULT_RISK_POLICY = {
    "minimum_evidence_relevance": 0.50,
    "strong_evidence_relevance": 0.65,
    "minimum_aligned_evidence_count": 2,
    "high_classifier_confidence": 0.85,
    "claim_like_inputs": ["claim", "headline"],
    "suspicious_cues": [],
}


def clean_policy_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().lower()


@lru_cache(maxsize=1)
def load_risk_policy() -> Dict[str, Any]:
    if not EVIDENCE_POLICY_PATH.exists():
        return DEFAULT_RISK_POLICY

    try:
        with EVIDENCE_POLICY_PATH.open("r", encoding="utf-8") as file:
            policy = yaml.safe_load(file) or {}
    except Exception:
        return DEFAULT_RISK_POLICY

    raw_risk_policy = policy.get("risk_policy", {}) or {}

    return {
        "minimum_evidence_relevance": float(
            raw_risk_policy.get(
                "minimum_evidence_relevance",
                DEFAULT_RISK_POLICY["minimum_evidence_relevance"],
            )
        ),
        "strong_evidence_relevance": float(
            raw_risk_policy.get(
                "strong_evidence_relevance",
                DEFAULT_RISK_POLICY["strong_evidence_relevance"],
            )
        ),
        "minimum_aligned_evidence_count": int(
            raw_risk_policy.get(
                "minimum_aligned_evidence_count",
                DEFAULT_RISK_POLICY["minimum_aligned_evidence_count"],
            )
        ),
        "high_classifier_confidence": float(
            raw_risk_policy.get(
                "high_classifier_confidence",
                DEFAULT_RISK_POLICY["high_classifier_confidence"],
            )
        ),
        "claim_like_inputs": [
            clean_policy_text(item)
            for item in raw_risk_policy.get(
                "claim_like_inputs",
                DEFAULT_RISK_POLICY["claim_like_inputs"],
            )
            if clean_policy_text(item)
        ],
        "suspicious_cues": [
            clean_policy_text(item)
            for item in raw_risk_policy.get(
                "suspicious_cues",
                DEFAULT_RISK_POLICY["suspicious_cues"],
            )
            if clean_policy_text(item)
        ],
    }


def get_min_evidence_relevance() -> float:
    return float(load_risk_policy()["minimum_evidence_relevance"])


def get_strong_evidence_relevance() -> float:
    return float(load_risk_policy()["strong_evidence_relevance"])


def get_min_aligned_evidence_count() -> int:
    return int(load_risk_policy()["minimum_aligned_evidence_count"])


def get_high_classifier_confidence() -> float:
    return float(load_risk_policy()["high_classifier_confidence"])


def get_claim_like_inputs() -> Set[str]:
    return set(load_risk_policy()["claim_like_inputs"])


def get_suspicious_cues() -> Set[str]:
    return set(load_risk_policy()["suspicious_cues"])


def normalize_text_for_matching(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_suspicious_language(text: str) -> bool:
    normalized_text = normalize_text_for_matching(text)

    if not normalized_text:
        return False

    for cue in get_suspicious_cues():
        normalized_cue = normalize_text_for_matching(cue)

        if not normalized_cue:
            continue

        # Phrase cue, for example "cover up" or "deep state".
        if " " in normalized_cue:
            if normalized_cue in normalized_text:
                return True
            continue

        # Single-word cue must match as a full word, not substring.
        if re.search(rf"\b{re.escape(normalized_cue)}\b", normalized_text):
            return True

    return False


def normalize_evidence_label(label: Any) -> str:
    normalized = str(label or "UNKNOWN").upper().strip()

    if normalized == "NOT ENOUGH INFO":
        return "NOT_ENOUGH_INFO"

    return normalized


def get_relevance_score(item: Dict) -> float:
    try:
        return float(item.get("relevance_score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _independent_source_identity(item: Dict) -> str:
    """Return a stable publisher/domain identity for corroboration counting.

    Two different URLs from the same publisher are not treated as two independent
    sources. Local curated datasets keep their collection/source identity.
    """
    source_type = str(item.get("source_type") or "").lower().strip()
    if source_type.startswith("local_"):
        return str(item.get("source") or item.get("source_name") or source_type).lower().strip()

    for key in ("domain", "source_name", "source"):
        value = str(item.get(key) or "").lower().strip()
        if value and "." in value and " " not in value:
            return value.removeprefix("www.")

    url = str(item.get("url") or "").strip()
    if url:
        try:
            host = (urlparse(url).hostname or "").lower().removeprefix("www.")
            if host:
                return host
        except Exception:
            pass

    return str(item.get("source_name") or item.get("source") or url or "unknown").lower().strip()


def summarize_evidence(evidence: List[Dict]) -> Dict:
    min_evidence_relevance = get_min_evidence_relevance()
    strong_evidence_relevance = get_strong_evidence_relevance()
    min_aligned_evidence_count = get_min_aligned_evidence_count()

    summary = {
        "retrieved_count": len(evidence),
        "usable_evidence_count": 0,
        "weak_evidence_count": 0,
        "supports_count": 0,
        "refutes_count": 0,
        "not_enough_info_count": 0,
        "unknown_count": 0,
        "strong_supports_count": 0,
        "strong_refutes_count": 0,
        "support_max_relevance": 0.0,
        "refute_max_relevance": 0.0,
        "not_enough_info_max_relevance": 0.0,
        "top_evidence_label": "UNKNOWN",
        "top_relevance_score": 0.0,
        "minimum_relevance_required": min_evidence_relevance,
        "strong_evidence_relevance_required": strong_evidence_relevance,
        "minimum_aligned_evidence_count": min_aligned_evidence_count,
        # Backward-compatible directional flags used by downstream decision logic.
        # These can become true either from a threshold-strong item or from independent corroboration.
        "has_strong_support": False,
        "has_strong_refute": False,
        # Explicit metadata removes ambiguity in diagnostics when strong_*_count is zero but
        # independent corroboration still establishes a reliable direction.
        "has_threshold_strong_support": False,
        "has_threshold_strong_refute": False,
        "has_corroborated_support": False,
        "has_corroborated_refute": False,
        "support_basis": "none",
        "refute_basis": "none",
    }

    if not evidence:
        summary["evidence_quality"] = "Insufficient"
        summary["directional_consensus"] = "INSUFFICIENT"
        summary["support_source_count"] = 0
        summary["refute_source_count"] = 0
        return summary

    support_sources = set()
    refute_sources = set()

    top_item = evidence[0]
    summary["top_evidence_label"] = normalize_evidence_label(
        top_item.get("evidence_label", "UNKNOWN")
    )
    summary["top_relevance_score"] = round(get_relevance_score(top_item), 4)

    for item in evidence:
        relevance_score = get_relevance_score(item)
        label = normalize_evidence_label(item.get("evidence_label", "UNKNOWN"))

        if relevance_score < min_evidence_relevance:
            summary["weak_evidence_count"] += 1
            continue

        summary["usable_evidence_count"] += 1

        if label == "SUPPORTS":
            summary["supports_count"] += 1
            support_sources.add(_independent_source_identity(item))
            summary["support_max_relevance"] = max(
                summary["support_max_relevance"],
                relevance_score,
            )

            if relevance_score >= strong_evidence_relevance:
                summary["strong_supports_count"] += 1

        elif label == "REFUTES":
            summary["refutes_count"] += 1
            refute_sources.add(_independent_source_identity(item))
            summary["refute_max_relevance"] = max(
                summary["refute_max_relevance"],
                relevance_score,
            )

            if relevance_score >= strong_evidence_relevance:
                summary["strong_refutes_count"] += 1

        elif label == "NOT_ENOUGH_INFO":
            summary["not_enough_info_count"] += 1
            summary["not_enough_info_max_relevance"] = max(
                summary["not_enough_info_max_relevance"],
                relevance_score,
            )

        else:
            summary["unknown_count"] += 1

    # Directional evidence is considered strong when at least one item crosses
    # the strong threshold OR enough independent usable items align. This keeps
    # a single high-quality source useful while rewarding corroboration.
    summary["support_source_count"] = len({value for value in support_sources if value})
    summary["refute_source_count"] = len({value for value in refute_sources if value})

    summary["has_threshold_strong_support"] = bool(
        summary["strong_supports_count"] >= 1
        and summary["support_max_relevance"] >= strong_evidence_relevance
    )
    summary["has_threshold_strong_refute"] = bool(
        summary["strong_refutes_count"] >= 1
        and summary["refute_max_relevance"] >= strong_evidence_relevance
    )
    summary["has_corroborated_support"] = bool(
        summary["supports_count"] >= min_aligned_evidence_count
        and summary["support_source_count"] >= min_aligned_evidence_count
    )
    summary["has_corroborated_refute"] = bool(
        summary["refutes_count"] >= min_aligned_evidence_count
        and summary["refute_source_count"] >= min_aligned_evidence_count
    )

    # Preserve the existing downstream semantics while making the basis explicit.
    summary["has_strong_support"] = bool(
        summary["has_threshold_strong_support"] or summary["has_corroborated_support"]
    )
    summary["has_strong_refute"] = bool(
        summary["has_threshold_strong_refute"] or summary["has_corroborated_refute"]
    )

    if summary["has_threshold_strong_support"] and summary["has_corroborated_support"]:
        summary["support_basis"] = "strong_item_and_independent_corroboration"
    elif summary["has_threshold_strong_support"]:
        summary["support_basis"] = "strong_item"
    elif summary["has_corroborated_support"]:
        summary["support_basis"] = "independent_corroboration"

    if summary["has_threshold_strong_refute"] and summary["has_corroborated_refute"]:
        summary["refute_basis"] = "strong_item_and_independent_corroboration"
    elif summary["has_threshold_strong_refute"]:
        summary["refute_basis"] = "strong_item"
    elif summary["has_corroborated_refute"]:
        summary["refute_basis"] = "independent_corroboration"

    if summary["has_strong_support"] and summary["has_strong_refute"]:
        summary["directional_consensus"] = "MIXED"
    elif summary["has_strong_support"]:
        summary["directional_consensus"] = "SUPPORTS"
    elif summary["has_strong_refute"]:
        summary["directional_consensus"] = "REFUTES"
    elif summary["usable_evidence_count"] > 0:
        summary["directional_consensus"] = "INCONCLUSIVE"
    else:
        summary["directional_consensus"] = "INSUFFICIENT"

    if summary["directional_consensus"] in {"SUPPORTS", "REFUTES"} and summary["top_relevance_score"] >= strong_evidence_relevance:
        summary["evidence_quality"] = "Strong"
    elif summary["usable_evidence_count"] >= 2:
        summary["evidence_quality"] = "Moderate"
    elif summary["usable_evidence_count"] == 1:
        summary["evidence_quality"] = "Limited"
    else:
        summary["evidence_quality"] = "Insufficient"

    summary["support_max_relevance"] = round(summary["support_max_relevance"], 4)
    summary["refute_max_relevance"] = round(summary["refute_max_relevance"], 4)
    summary["not_enough_info_max_relevance"] = round(
        summary["not_enough_info_max_relevance"],
        4,
    )

    return summary


def derive_final_label(
    classification: Dict,
    evidence_summary: Dict,
    input_type: str,
    text: str = "",
) -> str:
    classifier_label = classification["predicted_label"]
    classifier_confidence = float(classification["confidence_score"])

    supports_count = evidence_summary["supports_count"]
    refutes_count = evidence_summary["refutes_count"]
    usable_evidence_count = evidence_summary["usable_evidence_count"]

    has_strong_support = bool(evidence_summary.get("has_strong_support", False))
    has_strong_refute = bool(evidence_summary.get("has_strong_refute", False))

    support_max_relevance = float(evidence_summary.get("support_max_relevance", 0.0))
    refute_max_relevance = float(evidence_summary.get("refute_max_relevance", 0.0))

    strong_evidence_relevance = get_strong_evidence_relevance()
    high_classifier_confidence = get_high_classifier_confidence()
    claim_like_inputs = get_claim_like_inputs()

    suspicious_language = has_suspicious_language(text)
    normalized_input_type = clean_policy_text(input_type)

    if normalized_input_type in claim_like_inputs:
        # For short claims, do not trust the classifier alone.
        if usable_evidence_count == 0:
            if (
                classifier_label == "Likely Fake"
                and classifier_confidence >= high_classifier_confidence
                and suspicious_language
            ):
                return "Misleading"

            return "Uncertain"

        # Conflicting strong evidence should not force a real/fake decision.
        if has_strong_support and has_strong_refute:
            return "Uncertain"

        # Strong REFUTES evidence can produce Likely Fake.
        if has_strong_refute and not has_strong_support:
            return "Likely Fake"

        # Strong SUPPORTS evidence can produce Likely Real.
        # Weak or entity-only support below the strong threshold cannot flip a claim.
        if has_strong_support and not has_strong_refute:
            if support_max_relevance < strong_evidence_relevance:
                return "Uncertain"

            if (
                classifier_label == "Likely Fake"
                and classifier_confidence >= high_classifier_confidence
                and support_max_relevance < strong_evidence_relevance
            ):
                return "Uncertain"

            return "Likely Real"

        # Evidence exists but is not strong enough to decide.
        if (
            classifier_label == "Likely Fake"
            and classifier_confidence >= high_classifier_confidence
            and suspicious_language
        ):
            return "Misleading"

        # Directional evidence below the strong threshold should not overclaim.
        if supports_count > refutes_count or refutes_count > supports_count:
            return "Uncertain"

        return "Uncertain"

    # For article-style inputs, continue using the classifier as primary signal.
    return classifier_label




def apply_slm_verification_to_label(
    current_final_label: str,
    slm_verification: Dict,
    input_type: str,
    evidence_summary: Dict,
) -> str:
    """Use SLM stance only when it agrees with adequate retrieved evidence.

    The verifier is an evidence interpreter, not an independent source of truth.
    A high-confidence SLM output therefore cannot create a directional verdict when
    the evidence summary itself is weak or points the opposite way.
    """
    if not slm_verification or not bool(slm_verification.get("used", False)):
        return current_final_label

    normalized_input_type = clean_policy_text(input_type)
    if normalized_input_type not in get_claim_like_inputs():
        return current_final_label

    stance = str(slm_verification.get("stance", "NOT_ENOUGH_INFO")).upper().strip().replace(" ", "_")
    confidence = float(slm_verification.get("confidence", 0.0) or 0.0)
    threshold = float(getattr(settings, "slm_min_confidence_to_override", 0.70))
    if confidence < threshold:
        return current_final_label

    has_support = bool(evidence_summary.get("has_strong_support", False))
    has_refute = bool(evidence_summary.get("has_strong_refute", False))
    supports_count = int(evidence_summary.get("supports_count", 0) or 0)
    refutes_count = int(evidence_summary.get("refutes_count", 0) or 0)
    min_aligned = int(evidence_summary.get("minimum_aligned_evidence_count", get_min_aligned_evidence_count()) or 2)

    if stance == "SUPPORTS":
        if has_refute and not has_support:
            return "Uncertain"
        if has_support or supports_count >= min_aligned:
            return "Likely Real"
        return current_final_label

    if stance == "REFUTES":
        if has_support and not has_refute:
            return "Uncertain"
        if has_refute or refutes_count >= min_aligned:
            return "Likely Fake"
        return current_final_label

    if stance in {"NOT_ENOUGH_INFO", "NEI"}:
        # If the verifier cannot establish direction, preserve only a deterministic
        # label already backed by strong, non-conflicting evidence.
        if current_final_label == "Likely Real" and has_support and not has_refute:
            return current_final_label
        if current_final_label == "Likely Fake" and has_refute and not has_support:
            return current_final_label
        return "Uncertain"

    return current_final_label


def calculate_claim_risk_score(
    final_label: str,
    classifier_label: str,
    confidence: float,
    evidence_summary: Dict,
) -> float:
    supports_count = evidence_summary["supports_count"]
    refutes_count = evidence_summary["refutes_count"]
    not_enough_info_count = evidence_summary["not_enough_info_count"]
    usable_evidence_count = evidence_summary["usable_evidence_count"]
    top_relevance_score = float(evidence_summary["top_relevance_score"])

    has_strong_support = bool(evidence_summary.get("has_strong_support", False))
    has_strong_refute = bool(evidence_summary.get("has_strong_refute", False))
    has_threshold_strong_refute = bool(
        evidence_summary.get("has_threshold_strong_refute", evidence_summary.get("strong_refutes_count", 0) >= 1)
    )
    has_corroborated_refute = bool(evidence_summary.get("has_corroborated_refute", False))

    if final_label == "Likely Real":
        risk_score = 0.25

        if refutes_count > 0:
            risk_score += 0.10

        if not_enough_info_count > 0:
            risk_score += 0.05

        if not has_strong_support:
            risk_score += 0.10

        if top_relevance_score < 0.70:
            risk_score += 0.05

        return risk_score

    if final_label == "Likely Fake":
        risk_score = 0.70

        if supports_count > 0:
            risk_score -= 0.10

        if not_enough_info_count > 0:
            risk_score += 0.05

        # High misinformation risk should not hinge on a brittle 0.75 single-item
        # relevance cutoff when the evidence already contains both threshold-strong
        # refutation and independent corroboration. Preserve the old path for a
        # single exceptionally relevant refuting item as well.
        if (
            has_threshold_strong_refute and has_corroborated_refute
        ) or (
            has_strong_refute and top_relevance_score >= 0.75
        ):
            risk_score += 0.05

        return risk_score

    if final_label == "Misleading":
        risk_score = 0.65

        if refutes_count > 0:
            risk_score += 0.05

        if supports_count > 0:
            risk_score -= 0.05

        return risk_score

    # Uncertain
    risk_score = 0.50

    if usable_evidence_count > 0 and top_relevance_score >= 0.60:
        risk_score += 0.03

    return risk_score


def calculate_article_risk_score(
    classifier_label: str,
    confidence: float,
    evidence_summary: Dict,
) -> float:
    supports_count = evidence_summary["supports_count"]
    refutes_count = evidence_summary["refutes_count"]
    not_enough_info_count = evidence_summary["not_enough_info_count"]
    usable_evidence_count = evidence_summary["usable_evidence_count"]

    has_strong_support = bool(evidence_summary.get("has_strong_support", False))
    has_strong_refute = bool(evidence_summary.get("has_strong_refute", False))

    if classifier_label == "Likely Fake":
        risk_score = confidence
    elif classifier_label == "Likely Real":
        risk_score = 1.0 - confidence
    else:
        risk_score = 0.50

    if usable_evidence_count > 0:
        if has_strong_refute and refutes_count > supports_count:
            risk_score += 0.15
        elif has_strong_support and supports_count > refutes_count:
            risk_score -= 0.10
        elif not_enough_info_count >= max(supports_count, refutes_count):
            risk_score += 0.05
    else:
        risk_score += 0.03

    return risk_score


def to_risk_label(risk_score: float) -> str:
    if risk_score >= 0.75:
        return "High Risk"

    if risk_score >= 0.50:
        return "Medium Risk"

    if risk_score >= 0.30:
        return "Low Risk"

    return "Very Low Risk"


def calculate_risk(
    classification: Dict,
    evidence: List[Dict],
    input_type: str = "text",
    text: str = "",
    slm_verification: Dict = None,
) -> Dict:
    confidence = float(classification["confidence_score"])
    classifier_label = classification["predicted_label"]

    evidence_summary = summarize_evidence(evidence)

    final_label = derive_final_label(
        classification=classification,
        evidence_summary=evidence_summary,
        input_type=input_type,
        text=text,
    )

    final_label = apply_slm_verification_to_label(
        current_final_label=final_label,
        slm_verification=slm_verification or {},
        input_type=input_type,
        evidence_summary=evidence_summary,
    )

    normalized_input_type = clean_policy_text(input_type)

    if normalized_input_type in get_claim_like_inputs():
        risk_score = calculate_claim_risk_score(
            final_label=final_label,
            classifier_label=classifier_label,
            confidence=confidence,
            evidence_summary=evidence_summary,
        )
    else:
        risk_score = calculate_article_risk_score(
            classifier_label=classifier_label,
            confidence=confidence,
            evidence_summary=evidence_summary,
        )

    risk_score = max(0.0, min(1.0, risk_score))
    risk_score = round(risk_score, 4)

    return {
        "risk_score": risk_score,
        "risk_label": to_risk_label(risk_score),
        "final_label": final_label,
        "evidence_summary": evidence_summary,
    }
