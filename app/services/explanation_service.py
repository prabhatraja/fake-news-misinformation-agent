from typing import Dict, List

import numpy as np
from lime.lime_text import LimeTextExplainer

from app.core.config import settings
from app.services.classifier_service import load_baseline_model


CLASS_NAMES = ["real", "fake"]

BASIC_STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and", "or", "to", "of",
    "in", "on", "for", "by", "with", "from", "that", "this", "it", "as",
    "at", "be", "has", "have", "had", "not", "but", "into", "after",
}


def _predict_proba_for_lime(texts: List[str]) -> np.ndarray:
    vectorizer, model = load_baseline_model()

    text_vectors = vectorizer.transform(texts)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(text_vectors)

    raise ValueError("Current baseline model does not support predict_proba.")


def get_lime_influential_terms(text: str, num_features: int = 8) -> List[Dict]:
    if not text or len(text.strip()) < 5:
        return []

    explainer = LimeTextExplainer(class_names=CLASS_NAMES)

    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=_predict_proba_for_lime,
        num_features=num_features,
        num_samples=500,
    )

    terms = []

    for term, weight in explanation.as_list():
        impact = "fake" if weight > 0 else "real"

        terms.append(
            {
                "term": term,
                "weight": round(float(weight), 4),
                "impact": impact,
            }
        )

    return terms


def get_basic_content_terms(text: str, max_terms: int = 8) -> List[str]:
    """
    Lightweight fallback for transformer mode.

    This is not a transformer attribution method. It only displays useful content
    tokens from the input so the dashboard still has a readable term summary.
    """

    cleaned_tokens = []

    for raw_token in text.replace(".", " ").replace(",", " ").split():
        token = raw_token.strip().strip(":;!?()[]{}\"'").lower()

        if not token:
            continue

        if token in BASIC_STOPWORDS:
            continue

        if len(token) < 3:
            continue

        if token not in cleaned_tokens:
            cleaned_tokens.append(token)

        if len(cleaned_tokens) >= max_terms:
            break

    return cleaned_tokens


def get_influential_terms(text: str) -> List[str]:
    model_mode = settings.model_mode.lower().strip()

    if model_mode == "transformer":
        return get_basic_content_terms(text)

    lime_terms = get_lime_influential_terms(text)

    return [item["term"] for item in lime_terms]


def generate_explanation(
    classification: Dict,
    risk: Dict,
    evidence: List[Dict],
    inference_mode: str = "distilbert_local_rag",
    slm_verification: Dict = None,
) -> str:
    """Generate an audit-friendly explanation without treating classifier confidence as truth."""
    final_label = risk.get("final_label", classification["predicted_label"])
    classifier_label = classification["predicted_label"]
    confidence = float(classification["confidence_score"])
    risk_label = risk["risk_label"]
    model_used = classification.get("model_used", "baseline model")
    evidence_summary = risk.get("evidence_summary", {}) or {}

    consensus = str(evidence_summary.get("directional_consensus") or "INCONCLUSIVE")
    quality = str(evidence_summary.get("evidence_quality") or "Insufficient")
    supports = int(evidence_summary.get("supports_count", 0) or 0)
    refutes = int(evidence_summary.get("refutes_count", 0) or 0)
    nei = int(evidence_summary.get("not_enough_info_count", 0) or 0)
    usable = int(evidence_summary.get("usable_evidence_count", 0) or 0)

    parts = [
        f"Final assessment: {final_label} with {risk_label.lower()}.",
        f"Base classifier signal: {classifier_label} at {confidence:.1%} confidence using {model_used}.",
    ]

    if str(inference_mode).lower().strip() in {"traditional_lr", "traditional_svm", "distilbert"}:
        parts.append("This comparison mode does not use evidence retrieval or SLM verification.")
    else:
        parts.append(
            f"Evidence quality: {quality}; consensus: {consensus}. "
            f"Usable evidence counts were SUPPORTS={supports}, REFUTES={refutes}, NOT_ENOUGH_INFO={nei} across {usable} usable items."
        )

    if slm_verification and slm_verification.get("used"):
        stance = slm_verification.get("stance", "NOT_ENOUGH_INFO")
        slm_confidence = float(slm_verification.get("confidence", 0.0) or 0.0)
        reason = str(slm_verification.get("reason") or "").strip()
        sentence = f"Evidence verifier stance: {stance} at {slm_confidence:.1%} confidence."
        if reason:
            sentence += f" Reason: {reason}"
        parts.append(sentence)
        judgments = slm_verification.get("evidence_judgments") or []
        if judgments:
            parts.append(
                f"The same grounded verifier also semantically checked {len(judgments)} top evidence item(s); "
                "high-confidence live-web judgments were used to refine heuristic stance labels."
            )

    parts.append(
        "The result is an evidence-grounded misinformation risk assessment, not an absolute truth guarantee."
    )
    return " ".join(parts)

