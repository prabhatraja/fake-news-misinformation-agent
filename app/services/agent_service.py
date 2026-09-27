from typing import Dict, Optional

from app.core.config import settings
from app.services.preprocessing_service import clean_text, detect_input_length_type
from app.services.classifier_service import classify_text
from app.services.retrieval_service import retrieve_evidence
from app.services.risk_service import calculate_risk
from app.services.explanation_service import generate_explanation, get_influential_terms
from app.services.slm_verifier_service import (
    apply_semantic_evidence_judgments,
    verify_claim_with_slm,
)


CLASSIFIER_ONLY_MODES = {
    "traditional_lr",
    "traditional_svm",
    "distilbert",
}

LOCAL_RAG_MODES = {
    "distilbert_local_rag",
    "slm_assisted_local",
}

HYBRID_MODES = {
    "distilbert_hybrid",
    "slm_assisted_hybrid",
}

SLM_MODES = {
    "slm_assisted_local",
    "slm_assisted_hybrid",
}


def normalize_inference_mode(inference_mode: str = None) -> str:
    mode = (inference_mode or settings.inference_mode or "distilbert_local_rag")
    mode = str(mode).lower().strip()
    valid_modes = CLASSIFIER_ONLY_MODES.union(LOCAL_RAG_MODES).union(HYBRID_MODES)
    if mode not in valid_modes:
        return "distilbert_local_rag"
    return mode


def build_classifier_only_risk(classification: Dict) -> Dict:
    label = classification["predicted_label"]
    confidence = float(classification["confidence_score"])

    if label == "Likely Fake":
        risk_score = confidence
    elif label == "Likely Real":
        risk_score = 1.0 - confidence
    else:
        risk_score = 0.50

    risk_score = round(max(0.0, min(1.0, risk_score)), 4)
    if risk_score >= 0.75:
        risk_label = "High Risk"
    elif risk_score >= 0.50:
        risk_label = "Medium Risk"
    elif risk_score >= 0.30:
        risk_label = "Low Risk"
    else:
        risk_label = "Very Low Risk"

    return {
        "risk_score": risk_score,
        "risk_label": risk_label,
        "final_label": label,
        "evidence_summary": {
            "retrieved_count": 0,
            "usable_evidence_count": 0,
            "weak_evidence_count": 0,
            "supports_count": 0,
            "refutes_count": 0,
            "not_enough_info_count": 0,
            "unknown_count": 0,
            "top_evidence_label": "NONE",
            "top_relevance_score": 0.0,
            "minimum_relevance_required": 0.0,
            "strong_evidence_relevance_required": 0.0,
            "has_strong_support": False,
            "has_strong_refute": False,
            "directional_consensus": "NOT_APPLICABLE",
            "evidence_quality": "Not used",
        },
        "slm_verification": None,
    }


def _build_assessment_summary(
    classification: Dict,
    risk: Dict,
    evidence: list,
    slm_verification: Optional[Dict],
    selected_mode: str,
) -> Dict:
    final_label = risk.get("final_label", classification.get("predicted_label", "Uncertain"))
    risk_label = risk.get("risk_label", "Medium Risk")
    evidence_summary = risk.get("evidence_summary") or {}
    classifier_label = classification.get("predicted_label", "Unknown")
    classifier_confidence = float(classification.get("confidence_score", 0.0) or 0.0)

    consensus = str(evidence_summary.get("directional_consensus") or "INCONCLUSIVE")
    evidence_quality = str(evidence_summary.get("evidence_quality") or "Insufficient")

    slm_used = bool(slm_verification and slm_verification.get("used"))
    slm_stance = str((slm_verification or {}).get("stance") or "NOT_ENOUGH_INFO")
    slm_reason = str((slm_verification or {}).get("reason") or "").strip()

    if selected_mode in CLASSIFIER_ONLY_MODES:
        evidence_verdict = "Not used"
        why = (
            f"This comparison mode reports the base classifier signal: {classifier_label} "
            f"with confidence {classifier_confidence:.1%}. Evidence verification is not enabled in this mode."
        )
    else:
        evidence_verdict = (
            "Supports Claim" if consensus == "SUPPORTS" else
            "Contradicts Claim" if consensus == "REFUTES" else
            "Mixed Evidence" if consensus == "MIXED" else
            "Insufficient Evidence"
        )

        if slm_used and slm_reason:
            prefix = {
                "SUPPORTS": "The retrieved evidence supports the claim.",
                "REFUTES": "The retrieved evidence contradicts the claim.",
                "NOT_ENOUGH_INFO": "The retrieved evidence was not sufficient for a firm factual direction.",
            }.get(slm_stance, "The evidence was evaluated conservatively.")
            why = f"{prefix} {slm_reason}".strip()
        elif consensus == "SUPPORTS":
            why = "The strongest usable evidence supports the main factual proposition in the claim."
        elif consensus == "REFUTES":
            why = "The strongest usable evidence contradicts the main factual proposition in the claim."
        elif consensus == "MIXED":
            why = "Retrieved sources point in conflicting directions, so the assessment remains conservative."
        else:
            why = "The system did not find enough strong, aligned evidence to make a more definitive evidence-grounded assessment."

    final_direction = "SUPPORTS" if final_label == "Likely Real" else "REFUTES" if final_label == "Likely Fake" else "MIXED_OR_UNCERTAIN"
    classifier_direction = "SUPPORTS" if classifier_label == "Likely Real" else "REFUTES" if classifier_label == "Likely Fake" else "MIXED_OR_UNCERTAIN"
    signals_aligned = bool(
        selected_mode in CLASSIFIER_ONLY_MODES
        or final_direction == classifier_direction
        or final_label in {"Uncertain", "Misleading"}
    )

    return {
        "final_assessment": final_label,
        "misinformation_risk": risk_label,
        "evidence_verdict": evidence_verdict,
        "evidence_quality": evidence_quality,
        "why": why,
        "base_classifier_label": classifier_label,
        "base_classifier_confidence": round(classifier_confidence, 4),
        "slm_used": slm_used,
        "slm_stance": slm_stance if slm_used else "NOT_USED",
        "signals_aligned": signals_aligned,
        "retrieved_evidence_count": len(evidence),
    }


def run_misinformation_agent(
    text: str,
    input_type: str = "text",
    inference_mode: str = None,
    retrieval_query_text: Optional[str] = None,
) -> dict:
    """Run the single-claim pipeline with evidence-semantic hardening."""
    cleaned_text = clean_text(text)
    retrieval_text = clean_text(retrieval_query_text) if retrieval_query_text else cleaned_text
    if not retrieval_text:
        retrieval_text = cleaned_text

    detected_type = detect_input_length_type(cleaned_text)
    final_input_type = input_type if input_type != "text" else detected_type
    selected_mode = normalize_inference_mode(inference_mode)

    classification = classify_text(cleaned_text, inference_mode=selected_mode)
    evidence = []
    slm_verification = None

    if selected_mode in CLASSIFIER_ONLY_MODES:
        risk = build_classifier_only_risk(classification)
    else:
        evidence_mode = "hybrid" if selected_mode in HYBRID_MODES else "local"
        evidence = retrieve_evidence(
            retrieval_text,
            top_k=settings.top_k_evidence,
            evidence_mode=evidence_mode,
        )

        if selected_mode in SLM_MODES:
            slm_verification = verify_claim_with_slm(
                claim=cleaned_text,
                evidence=evidence,
                inference_mode=selected_mode,
            )
            # One grounded SLM call now also judges each selected live-web evidence
            # snippet semantically. This corrects lexical stance misses without
            # adding multiple model calls or using the SLM as a knowledge source.
            evidence = apply_semantic_evidence_judgments(
                evidence,
                slm_verification,
                min_confidence=0.75,
                min_relevance=0.50,
            )

        risk = calculate_risk(
            classification,
            evidence,
            input_type=final_input_type,
            text=cleaned_text,
            slm_verification=slm_verification,
        )

    explanation = generate_explanation(
        classification,
        risk,
        evidence,
        inference_mode=selected_mode,
        slm_verification=slm_verification,
    )
    influential_terms = get_influential_terms(cleaned_text)
    assessment_summary = _build_assessment_summary(
        classification=classification,
        risk=risk,
        evidence=evidence,
        slm_verification=slm_verification,
        selected_mode=selected_mode,
    )

    response = {
        "input_text": cleaned_text,
        "input_type": final_input_type,
        "predicted_label": risk.get("final_label", classification["predicted_label"]),
        "confidence_score": classification["confidence_score"],
        "risk_score": risk["risk_score"],
        "risk_label": risk["risk_label"],
        "explanation": explanation,
        "assessment_summary": assessment_summary,
        "influential_terms": influential_terms,
        "evidence": evidence,
        "disclaimer": (
            "This system supports misinformation risk analysis but should not be treated "
            "as the final authority on truth. Retrieved evidence and SLM-assisted verification "
            "are supporting signals and should be interpreted with human judgment."
        ),
        "inference_mode": selected_mode,
        "model_used": classification.get("model_used"),
        "raw_class": classification.get("raw_class"),
        "class_probabilities": classification.get("class_probabilities"),
        "evidence_summary": risk.get("evidence_summary"),
        "slm_verification": slm_verification,
    }

    if retrieval_text != cleaned_text:
        response["retrieval_query_text"] = retrieval_text

    return response
