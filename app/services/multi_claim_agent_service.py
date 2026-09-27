from copy import deepcopy
import re
from typing import Any, Dict, List, Optional

from app.services.agent_service import run_misinformation_agent
from app.services.claim_extraction_service import extract_claims
from app.services.content_extraction_service import extract_content
from app.services.evidence_query_builder_service import build_verification_context
from app.services.input_router_service import detect_input_kind
from app.services.temporal_context_service import extract_temporal_context


RISK_RANK = {
    "Very Low Risk": 0,
    "Low Risk": 1,
    "Medium Risk": 2,
    "High Risk": 3,
}

# Sources whose text is extracted from a container/source rather than directly typed
# as a single claim. These require slightly more conservative aggregation and
# guardrails because extraction, transcripts, OCR, and source context can be partial.
SOURCE_TYPES_WITH_EXTRACTION = {
    "url",
    "twitter_url",
    "twitter_profile_url",
    "youtube_url",
    "pdf_url",
    "image_url",
    "text_file_upload",
    "pdf_upload",
    "image_upload",
    "url_file",
    "uploaded_file",
}

MEDIA_CONTEXT_TERMS = {
    "video",
    "viral video",
    "clip",
    "viral clip",
    "shared",
    "circulated",
    "recirculated",
    "old video",
    "misleading video",
    "footage",
    "screenshot",
    "image",
    "photo",
    "post",
    "tweet",
    "x.com",
    "twitter",
}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _source_metadata(extraction: Dict[str, Any], input_kind: str) -> Dict[str, Any]:
    return {
        "source_type": input_kind,
        "url": extraction.get("url", ""),
        "domain": extraction.get("domain", ""),
        "title": extraction.get("title", ""),
        "author": extraction.get("author", ""),
        "published_date": extraction.get("published_date", ""),
        "metadata": extraction.get("metadata", {}),
    }


def _risk_rank(risk: str) -> int:
    return RISK_RANK.get(risk or "Medium Risk", 2)


def _number_from_match(value: str, suffix: str = "") -> Optional[float]:
    try:
        number = float(value.replace(",", ""))
    except Exception:
        return None

    suffix = (suffix or "").lower()
    if suffix.startswith("million") or suffix == "m":
        number *= 1_000_000
    elif suffix.startswith("billion") or suffix == "bn":
        number *= 1_000_000_000
    elif suffix.startswith("thousand") or suffix == "k":
        number *= 1_000
    return number


def _extract_numbers(text: str) -> List[float]:
    text = str(text or "")
    values: List[float] = []
    pattern = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(million|billion|thousand|m|bn|k)?\b", re.I)
    for match in pattern.finditer(text):
        number = _number_from_match(match.group(1), match.group(2) or "")
        if number is not None:
            values.append(number)
    return values


def _has_close_numeric_discrepancy(claim_text: str, verifier_reason: str, tolerance_ratio: float = 0.02) -> bool:
    """Detect small numeric mismatch such as 5,889,000 vs 5.849 million."""
    claim_numbers = _extract_numbers(claim_text)
    reason_numbers = _extract_numbers(verifier_reason)
    if not claim_numbers or not reason_numbers:
        return False

    for a in claim_numbers:
        for b in reason_numbers:
            base = max(abs(a), abs(b), 1.0)
            if abs(a - b) / base <= tolerance_ratio:
                return True
    return False


def _append_explanation(result: Dict[str, Any], addition: str) -> None:
    explanation = _clean(result.get("explanation", ""))
    if addition and addition not in explanation:
        result["explanation"] = (explanation + " " + addition).strip()


def _claim_can_inherit_source_recirculation(claim_text: str, claim_temporal_context: Dict[str, Any]) -> bool:
    """Avoid contaminating unrelated claims in mixed files with source-level media context.

    Example: a text file can contain two Eiffel Tower facts and one viral-video claim.
    Only the viral-video/media claim should inherit recirculation caution. The Eiffel
    Tower claims should stay standard/historical and should not receive old-video terms.
    """
    text = _clean(claim_text).lower()
    if any(term in text for term in MEDIA_CONTEXT_TERMS):
        return True
    if claim_temporal_context.get("recirculation_risk"):
        return True
    if claim_temporal_context.get("relative_time_cues") and any(term in text for term in {"today", "last week", "recently", "now"}):
        return True
    return False


def _adjust_claim_result_for_multiformat(
    claim: Dict[str, Any],
    result: Dict[str, Any],
    input_kind: str,
    limitations: List[str],
) -> Dict[str, Any]:
    """Make extracted-source outputs less overconfident than single-claim mode."""
    adjusted = deepcopy(result)
    claim_text = _clean(claim.get("verification_text") or claim.get("claim_text", ""))
    claim_type = _clean(claim.get("claim_type", "other")).lower()
    importance = _clean(claim.get("importance", "medium")).lower()

    slm = adjusted.get("slm_verification") or {}
    evidence_summary = adjusted.get("evidence_summary") or {}
    stance = _clean(slm.get("stance", "")).upper()
    reason = _clean(slm.get("reason", ""))
    label = adjusted.get("predicted_label", "Uncertain")
    risk = adjusted.get("risk_label", "Medium Risk")
    risk_score = float(adjusted.get("risk_score", 0.5) or 0.5)

    adjustment_reasons: List[str] = []

    if input_kind in SOURCE_TYPES_WITH_EXTRACTION:
        refutes_count = int(evidence_summary.get("refutes_count", 0) or 0)
        strong_refutes = int(evidence_summary.get("strong_refutes_count", 0) or 0)
        has_close_numeric = _has_close_numeric_discrepancy(claim_text, reason)

        if claim_type == "numeric" and label == "Likely Fake" and risk == "High Risk":
            # A verifier may correctly spot a value mismatch, but this is often a
            # precision/version issue rather than high-risk fake content.
            if has_close_numeric or refutes_count == 0 or strong_refutes == 0 or importance in {"medium", "low"}:
                adjusted["predicted_label"] = "Misleading"
                adjusted["risk_label"] = "Medium Risk"
                adjusted["risk_score"] = min(risk_score, 0.55)
                adjustment_reasons.append(
                    "Risk guardrail downgraded a secondary numeric/statistical mismatch from high-risk fake to medium-risk misleading."
                )

        if label == "Likely Fake" and stance in {"NOT_ENOUGH_INFO", ""} and strong_refutes == 0:
            adjusted["predicted_label"] = "Uncertain"
            adjusted["risk_label"] = "Medium Risk"
            adjusted["risk_score"] = min(risk_score, 0.55)
            adjustment_reasons.append(
                "Risk guardrail downgraded a weakly evidenced extracted-claim result to Uncertain."
            )

        if input_kind == "twitter_url":
            limitations.append(
                "X/Twitter analysis is limited to extracted text. Attached images, videos, quote context, and thread context are not yet verified."
            )

        if input_kind in {"image_upload", "image_url"}:
            limitations.append(
                "Image analysis uses OCR text extraction only; it does not verify visual authenticity, deepfakes, or manipulation."
            )

    if adjustment_reasons:
        adjusted["phase2_adjustment"] = {
            "applied": True,
            "reasons": adjustment_reasons,
        }
        for reason_text in adjustment_reasons:
            _append_explanation(adjusted, reason_text)
            limitations.append(reason_text)
    else:
        adjusted["phase2_adjustment"] = {"applied": False, "reasons": []}

    return adjusted


def _aggregate_results(claim_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not claim_results:
        return {
            "overall_label": "Uncertain",
            "overall_risk": "Medium Risk",
            "notes": ["No verifiable claims were available after extraction."],
        }

    # A single extracted claim is already the complete source assessment.
    # Do not reinterpret a claim-level Likely Fake result as Misleading merely
    # because its risk band is below High. Multi-claim aggregation rules apply
    # only when there are actually multiple independently verified claims.
    if len(claim_results) == 1:
        single_result = claim_results[0].get("result", {}) or {}
        return {
            "overall_label": single_result.get("predicted_label", "Uncertain"),
            "overall_risk": single_result.get("risk_label", "Medium Risk"),
            "notes": [],
        }

    labels = [r.get("result", {}).get("predicted_label", "Uncertain") for r in claim_results]
    risks = [r.get("result", {}).get("risk_label", "Medium Risk") for r in claim_results]
    max_risk = max(risks, key=_risk_rank)

    high_claims = [r for r in claim_results if r.get("importance") == "high"]
    medium_or_low_claims = [r for r in claim_results if r.get("importance") != "high"]

    high_bad = [
        r for r in high_claims
        if r.get("result", {}).get("predicted_label") in {"Likely Fake", "Misleading"}
    ]
    high_fake_high_risk = [
        r for r in high_claims
        if r.get("result", {}).get("predicted_label") == "Likely Fake"
        and r.get("result", {}).get("risk_label") == "High Risk"
    ]
    secondary_bad = [
        r for r in medium_or_low_claims
        if r.get("result", {}).get("predicted_label") in {"Likely Fake", "Misleading"}
    ]

    all_real = all(label == "Likely Real" for label in labels)
    any_uncertain = any(label == "Uncertain" for label in labels)
    any_bad = any(label in {"Likely Fake", "Misleading"} for label in labels)

    notes: List[str] = []

    if high_fake_high_risk:
        return {"overall_label": "Likely Fake", "overall_risk": "High Risk", "notes": notes}

    if high_bad:
        risk = max_risk if max_risk != "Very Low Risk" else "Medium Risk"
        return {"overall_label": "Misleading", "overall_risk": risk, "notes": notes}

    if secondary_bad and high_claims and all(
        r.get("result", {}).get("predicted_label") == "Likely Real" for r in high_claims
    ):
        notes.append(
            "Overall source assessment remains mostly supported because high-importance claims were supported; secondary claims showed discrepancies."
        )
        return {"overall_label": "Likely Real", "overall_risk": "Medium Risk", "notes": notes}

    if any_bad:
        return {"overall_label": "Misleading", "overall_risk": max_risk, "notes": notes}

    if all_real:
        low_risk = max_risk if max_risk in {"Low Risk", "Very Low Risk"} else "Low Risk"
        return {"overall_label": "Likely Real", "overall_risk": low_risk, "notes": notes}

    if any_uncertain:
        # Mixed extracted content with one high-risk false claim and one unrelated uncertain claim
        # should retain the high-risk false claim. Otherwise, uncertainty dominates.
        if any_bad:
            return {"overall_label": "Misleading", "overall_risk": max_risk, "notes": notes}
        return {"overall_label": "Uncertain", "overall_risk": "Medium Risk", "notes": notes}

    return {"overall_label": "Uncertain", "overall_risk": max_risk, "notes": notes}


def _build_source_temporal_seed(route_input_kind: str, content_text: str, source_metadata: Dict[str, Any]) -> str:
    """Keep source-level temporal scan compact so long pages do not pollute every claim."""
    if route_input_kind in {"url", "youtube_url", "twitter_url", "twitter_profile_url"}:
        return " ".join(
            part for part in [
                source_metadata.get("title", ""),
                source_metadata.get("published_date", ""),
                content_text[:1600],
            ] if part
        )
    # For uploads/text files, the full text is allowed at source level only for
    # overall limitations. Claim-level retrieval will still use isolated claim context.
    return content_text


def _analyze_extracted_content(
    *,
    original_input: str,
    input_kind: str,
    content_text: str,
    extraction: Dict[str, Any],
    source_metadata: Dict[str, Any],
    inference_mode: str,
    max_claims: int,
    enable_temporal_analysis: bool,
) -> Dict[str, Any]:
    limitations: List[str] = []
    limitations.extend(extraction.get("limitations", []))
    content_text = _clean(content_text)

    if extraction.get("status") == "success" and input_kind != "plain_text" and len(content_text) < 40:
        limitations.append("Extracted content is too short for reliable claim verification.")
        temporal_context = extract_temporal_context(content_text, source_metadata) if enable_temporal_analysis else {}
        return {
            "original_input": original_input,
            "input_kind": input_kind,
            "inference_mode": inference_mode,
            "source_metadata": source_metadata,
            "extraction": extraction,
            "temporal_context": temporal_context,
            "claims_extracted": [],
            "claims_analyzed": [],
            "analysis_status": "not_analyzed",
            "overall_label": "Uncertain",
            "overall_risk": "Medium Risk",
            "limitations": _dedupe(limitations),
            "disclaimer": (
                "The agent extracted too little text to verify this source reliably. "
                "Paste the full content, upload a text-based file, or provide a transcript."
            ),
        }

    claim_extraction = extract_claims(content_text, max_claims=max_claims)
    claims = claim_extraction.get("claims", [])
    limitations.extend(claim_extraction.get("limitations", []))

    if enable_temporal_analysis:
        source_temporal_seed = _build_source_temporal_seed(input_kind, content_text, source_metadata)
        temporal_context = extract_temporal_context(source_temporal_seed, source_metadata)
    else:
        temporal_context = {}

    claim_results: List[Dict[str, Any]] = []

    for claim in claims:
        source_text = _clean(claim.get("source_text") or claim.get("claim_text", ""))
        verification_text = _clean(claim.get("verification_text") or claim.get("claim_text", "") or source_text)
        if not source_text:
            source_text = verification_text
        if not verification_text:
            continue

        if enable_temporal_analysis:
            # Temporal/source framing belongs to the source assertion, not to the
            # normalized proposition used for evidence retrieval. This preserves
            # recirculation and relative-time context without making framing part of
            # the proposition whose truth is being verified.
            claim_temporal_context = extract_temporal_context(source_text, source_metadata)

            # Preserve source-level old-media caution only for claims that are actually
            # about media/video/social sharing. This prevents an unrelated factual claim
            # inside the same uploaded file from receiving viral-video search terms.
            if (
                temporal_context.get("recirculation_risk")
                and not claim_temporal_context.get("recirculation_risk")
                and _claim_can_inherit_source_recirculation(source_text, claim_temporal_context)
            ):
                claim_temporal_context["recirculation_risk"] = True
                claim_temporal_context["time_scope"] = "historical_or_recirculated_media"
                notes = list(claim_temporal_context.get("notes") or [])
                notes.append("Source-level recirculation risk was detected and applied to this media-related claim as a caution.")
                claim_temporal_context["notes"] = notes

            if (
                temporal_context.get("source_age_years") is not None
                and claim_temporal_context.get("source_age_years") is None
            ):
                claim_temporal_context["source_age_years"] = temporal_context.get("source_age_years")
        else:
            claim_temporal_context = {}

        verification_context = build_verification_context(
            claim_text=verification_text,
            source_metadata=source_metadata,
            temporal_context=claim_temporal_context,
            input_kind=input_kind,
        )
        retrieval_query_text = verification_context.get("retrieval_query_text") or verification_text

        result = run_misinformation_agent(
            text=verification_text,
            input_type="claim",
            inference_mode=inference_mode,
            retrieval_query_text=retrieval_query_text,
        )
        result = _adjust_claim_result_for_multiformat(claim, result, input_kind, limitations)

        claim_results.append(
            {
                "claim_id": claim.get("claim_id", len(claim_results) + 1),
                # claim_text remains the actual proposition verified for backward
                # compatibility with existing UI/API consumers. source_text preserves
                # the original framing for auditability.
                "claim_text": verification_text,
                "source_text": source_text,
                "verification_text": verification_text,
                "verification_target": claim.get("verification_target", "direct_proposition"),
                "assertion_status": claim.get("assertion_status", "unclear"),
                "representation_confidence": claim.get("representation_confidence", 0.0),
                "representation_fallback": bool(claim.get("representation_fallback", False)),
                "transformation_reason": claim.get("transformation_reason", ""),
                "claim_type": claim.get("claim_type", "other"),
                "importance": claim.get("importance", "medium"),
                "reason_for_selection": claim.get("reason_for_selection", ""),
                "verification_context": verification_context,
                "claim_temporal_context": claim_temporal_context,
                "result": result,
            }
        )

    aggregate = _aggregate_results(claim_results)
    limitations.extend(aggregate.get("notes", []))

    temporal_notes = list(temporal_context.get("notes", []) if isinstance(temporal_context, dict) else [])
    if isinstance(temporal_context, dict) and temporal_context.get("recirculation_risk"):
        limitations.append(
            "Possible old or recirculated media context detected. Verify whether the source is being shared with accurate date/context."
        )

    limitations = _dedupe(limitations)

    return {
        "original_input": original_input,
        "input_kind": input_kind,
        "inference_mode": inference_mode,
        "source_metadata": source_metadata,
        "extraction": {
            **extraction,
            "text": content_text[:2500] + ("..." if len(content_text) > 2500 else ""),
        },
        "temporal_context": temporal_context,
        "temporal_notes": temporal_notes,
        "claims_extracted": claims,
        "claims_analyzed": claim_results,
        "analysis_status": "completed" if claim_results else "not_analyzed",
        "overall_label": aggregate["overall_label"],
        "overall_risk": aggregate["overall_risk"],
        "limitations": limitations,
        "disclaimer": (
            "Multi-format verification is a misinformation risk assessment, not an absolute truth guarantee. "
            "For URLs, YouTube, social media, files, and OCR, extraction can be blocked, partial, or noisy. "
            "Evidence and SLM output require human judgment."
        ),
    }


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        text = _clean(item)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def analyze_extracted_content(
    *,
    original_input: str,
    input_kind: str,
    content_text: str,
    extraction: Dict[str, Any],
    source_metadata: Dict[str, Any],
    inference_mode: str = "slm_assisted_hybrid",
    max_claims: int = 3,
    enable_temporal_analysis: bool = True,
) -> Dict[str, Any]:
    """Public helper for uploaded files, preserving the real uploaded input kind.

    This avoids treating uploaded documents as generic plain text, so the output
    receives the correct extraction guardrails without polluting claim retrieval.
    """
    return _analyze_extracted_content(
        original_input=original_input,
        input_kind=input_kind,
        content_text=content_text,
        extraction=extraction,
        source_metadata=source_metadata,
        inference_mode=inference_mode,
        max_claims=max_claims,
        enable_temporal_analysis=enable_temporal_analysis,
    )


def _analyze_twitter_profile_extraction(
    *,
    original_input: str,
    extraction: Dict[str, Any],
    source_metadata: Dict[str, Any],
    inference_mode: str,
    max_claims: int,
    enable_temporal_analysis: bool,
) -> Dict[str, Any]:
    """Verify each retrieved profile post independently, then summarize only the sampled posts."""
    metadata = extraction.get("metadata") or {}
    posts = list(metadata.get("profile_posts") or [])
    limitations = list(extraction.get("limitations") or [])
    limitations.append(
        "Profile-level counts summarize only the retrieved public posts in this sample; they are not an account-level truthfulness score."
    )

    per_post_results: List[Dict[str, Any]] = []
    flat_claims_extracted: List[Dict[str, Any]] = []
    flat_claims_analyzed: List[Dict[str, Any]] = []
    global_claim_id = 1

    for index, post in enumerate(posts, start=1):
        post_text = _clean(post.get("text", ""))
        if not post_text:
            continue

        post_id = _clean(post.get("post_id", ""))
        post_url = _clean(post.get("url", ""))
        created_at = _clean(post.get("created_at", ""))
        handle = _clean(metadata.get("resolved_handle") or metadata.get("requested_handle") or "")
        post_extraction = {
            "status": "success",
            "input_kind": "twitter_url",
            "url": post_url,
            "domain": "x.com",
            "title": f"X post {index} from @{handle}" if handle else f"X post {index}",
            "author": handle,
            "published_date": created_at,
            "text": post_text,
            "text_length": len(post_text),
            "limitations": [
                "Post text was retrieved through the official X API as part of a profile sample.",
                "Attached media, quoted-post context, replies, and linked-page contents were not independently verified.",
            ],
            "metadata": {
                "source_type": "twitter_url",
                "twitter_source_kind": "profile_post",
                "provider": "X API v2",
                "post_id": post_id,
                "identity_verified": True,
                "extraction_quality": "reliable",
            },
        }
        post_source_metadata = {
            "source_type": "twitter_url",
            "url": post_url,
            "domain": "x.com",
            "title": post_extraction["title"],
            "author": handle,
            "published_date": created_at,
            "metadata": post_extraction["metadata"],
        }

        post_result = _analyze_extracted_content(
            original_input=post_url or post_text,
            input_kind="twitter_url",
            content_text=post_text,
            extraction=post_extraction,
            source_metadata=post_source_metadata,
            inference_mode=inference_mode,
            max_claims=1,
            enable_temporal_analysis=enable_temporal_analysis,
        )

        analyzed_claims = list(post_result.get("claims_analyzed") or [])
        slm_stance = ""
        if analyzed_claims:
            slm = (analyzed_claims[0].get("result") or {}).get("slm_verification") or {}
            slm_stance = _clean(slm.get("stance", ""))

        per_post_results.append(
            {
                "post_index": index,
                "post_id": post_id,
                "url": post_url,
                "created_at": created_at,
                "text": post_text,
                "label": post_result.get("overall_label", "Uncertain"),
                "risk": post_result.get("overall_risk", "Medium Risk"),
                "slm_stance": slm_stance,
                "claims_extracted": post_result.get("claims_extracted", []),
                "claims_analyzed": analyzed_claims,
                "limitations": post_result.get("limitations", []),
            }
        )

        for claim in list(post_result.get("claims_extracted") or []):
            copied = deepcopy(claim)
            copied["claim_id"] = global_claim_id
            copied["post_index"] = index
            copied["post_id"] = post_id
            copied["post_url"] = post_url
            flat_claims_extracted.append(copied)
            global_claim_id += 1

        for claim in analyzed_claims:
            copied = deepcopy(claim)
            copied["claim_id"] = len(flat_claims_analyzed) + 1
            copied["post_index"] = index
            copied["post_id"] = post_id
            copied["post_url"] = post_url
            flat_claims_analyzed.append(copied)

    if not per_post_results:
        return {
            "original_input": original_input,
            "input_kind": "twitter_profile_url",
            "inference_mode": inference_mode,
            "source_metadata": source_metadata,
            "extraction": extraction,
            "temporal_context": {"time_scope": "profile_sample", "notes": []},
            "claims_extracted": [],
            "claims_analyzed": [],
            "analysis_status": "not_analyzed",
            "overall_label": "Uncertain",
            "overall_risk": "Medium Risk",
            "limitations": _dedupe(limitations),
            "disclaimer": "No public profile posts were available for verification.",
            "social_analysis": {
                "type": "twitter_profile",
                "profile": metadata,
                "posts_analyzed": 0,
                "label_counts": {},
                "risk_counts": {},
                "posts": [],
            },
        }

    label_counts: Dict[str, int] = {}
    risk_counts: Dict[str, int] = {}
    for item in per_post_results:
        label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1
        risk_counts[item["risk"]] = risk_counts.get(item["risk"], 0) + 1

    aggregate = _aggregate_results(flat_claims_analyzed)
    sample_label = aggregate.get("overall_label", "Uncertain")
    sample_risk = aggregate.get("overall_risk", "Medium Risk")

    return {
        "original_input": original_input,
        "input_kind": "twitter_profile_url",
        "inference_mode": inference_mode,
        "source_metadata": source_metadata,
        "extraction": extraction,
        "temporal_context": {
            "time_scope": "profile_sample",
            "event_year_guess": None,
            "years_mentioned": [],
            "relative_time_cues": [],
            "recirculation_risk": False,
            "notes": ["Temporal context is evaluated independently for each retrieved X/Twitter post."],
        },
        "claims_extracted": flat_claims_extracted,
        "claims_analyzed": flat_claims_analyzed,
        "analysis_status": "completed" if flat_claims_analyzed else "not_analyzed",
        "overall_label": sample_label,
        "overall_risk": sample_risk,
        "limitations": _dedupe(limitations),
        "disclaimer": (
            f"This summary applies only to the {len(per_post_results)} recent public posts retrieved and analyzed. "
            "It is not a rating of the X/Twitter account or its owner. Each post is independently treated as a misinformation risk assessment, not absolute truth."
        ),
        "social_analysis": {
            "type": "twitter_profile",
            "profile": {
                "requested_handle": metadata.get("requested_handle", ""),
                "resolved_handle": metadata.get("resolved_handle", ""),
                "profile_name": metadata.get("profile_name", ""),
                "verified": metadata.get("verified", False),
                "posts_requested": metadata.get("posts_requested", len(posts)),
                "posts_retrieved": metadata.get("posts_retrieved", len(posts)),
            },
            "posts_analyzed": len(per_post_results),
            "label_counts": label_counts,
            "risk_counts": risk_counts,
            "posts": per_post_results,
        },
    }

def analyze_multi_format_input(
    raw_input: str,
    source_type: str = "auto",
    inference_mode: str = "slm_assisted_hybrid",
    max_claims: int = 3,
    enable_temporal_analysis: bool = True,
    twitter_manual_text: str = "",
    twitter_profile_post_limit: int = 5,
) -> Dict[str, Any]:
    route = detect_input_kind(raw_input, source_type=source_type)
    extraction = extract_content(
        route,
        twitter_manual_text=twitter_manual_text,
        twitter_profile_post_limit=max(1, min(int(twitter_profile_post_limit or 5), 10)),
    )
    effective_input_kind = extraction.get("input_kind") or route.input_kind
    source_metadata = _source_metadata(extraction, effective_input_kind)

    if effective_input_kind == "twitter_profile_url" and extraction.get("status") == "success":
        return _analyze_twitter_profile_extraction(
            original_input=raw_input,
            extraction=extraction,
            source_metadata=source_metadata,
            inference_mode=inference_mode,
            max_claims=max_claims,
            enable_temporal_analysis=enable_temporal_analysis,
        )

    limitations: List[str] = []
    limitations.extend(extraction.get("limitations", []))

    content_text = _clean(extraction.get("text", ""))

    if extraction.get("status") in {"failed", "unsupported"} and effective_input_kind != "plain_text":
        temporal_context = extract_temporal_context(content_text, source_metadata) if enable_temporal_analysis else {}
        return {
            "original_input": raw_input,
            "input_kind": effective_input_kind,
            "inference_mode": inference_mode,
            "source_metadata": source_metadata,
            "extraction": extraction,
            "temporal_context": temporal_context,
            "claims_extracted": [],
            "claims_analyzed": [],
            "analysis_status": "not_analyzed",
            "overall_label": "Uncertain",
            "overall_risk": "Medium Risk",
            "limitations": _dedupe(limitations),
            "disclaimer": (
                "The agent could not extract enough content from the source. Paste the text, provide a transcript, "
                "or upload supported content for reliable analysis."
            ),
        }

    return _analyze_extracted_content(
        original_input=raw_input,
        input_kind=effective_input_kind,
        content_text=content_text,
        extraction=extraction,
        source_metadata=source_metadata,
        inference_mode=inference_mode,
        max_claims=max_claims,
        enable_temporal_analysis=enable_temporal_analysis,
    )
