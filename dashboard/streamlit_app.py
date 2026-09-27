from typing import Any, Dict, List, Optional

import requests
import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.evidence_presentation_service import (
    build_evidence_coverage_text,
    select_source_diverse_evidence,
)


API_BASE_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_BASE_URL}/api/v1/predict"
ANALYZE_URL = f"{API_BASE_URL}/api/v1/analyze"
ANALYZE_FILE_URL = f"{API_BASE_URL}/api/v1/analyze-file"
HEALTH_URL = f"{API_BASE_URL}/api/v1/health"
DEFAULT_DEMO_MODE = "slm_assisted_hybrid"

INFERENCE_MODE_OPTIONS = {
    "Traditional ML - Logistic Regression": "traditional_lr",
    "Traditional ML - Linear SVM": "traditional_svm",
    "Transformer only - DistilBERT 512": "distilbert",
    "DistilBERT + Local RAG": "distilbert_local_rag",
    "DistilBERT + Hybrid Web Evidence": "distilbert_hybrid",
    "SLM-assisted Local Verification": "slm_assisted_local",
    "SLM-assisted Hybrid Verification": "slm_assisted_hybrid",
}

SOURCE_TYPE_OPTIONS = {
    "Auto detect": "auto",
    "Plain text": "plain_text",
    "Generic URL / article URL": "url",
    "YouTube URL": "youtube_url",
    "X / Twitter URL (post or profile)": "twitter_url",
    "PDF URL": "pdf_url",
    "Image URL": "image_url",
}

STANCE_DISPLAY = {
    "SUPPORTS": "Supports Claim",
    "REFUTES": "Contradicts Claim",
    "NOT_ENOUGH_INFO": "Insufficient Evidence",
    "NOT_USED": "Not Used",
}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _fmt_score(value: Any, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if percent and 0.0 <= number <= 1.0:
        return f"{number:.1%}"
    if 0.0 <= number <= 1.0:
        return f"{number:.3f}"
    return str(value)


def _label_icon(label: str) -> str:
    value = str(label or "").lower()
    if "fake" in value:
        return "🔴"
    if "misleading" in value:
        return "🟠"
    if "real" in value:
        return "🟢"
    if "uncertain" in value:
        return "🟡"
    return "⚪"


def _risk_icon(label: str) -> str:
    value = str(label or "").lower()
    if "high" in value:
        return "🔴"
    if "medium" in value:
        return "🟠"
    if "low" in value:
        return "🟢"
    return "⚪"


def _human_input_kind(kind: str) -> str:
    mapping = {
        "plain_text": "Plain Text",
        "url": "Web Page / Article URL",
        "youtube_url": "YouTube",
        "twitter_url": "X/Twitter Post",
        "twitter_profile_url": "X/Twitter Profile",
        "pdf_url": "PDF URL",
        "image_url": "Image URL",
        "text_file_upload": "Text File",
        "pdf_upload": "PDF Upload",
        "image_upload": "Image / Screenshot",
        "uploaded_file": "Uploaded File",
    }
    return mapping.get(str(kind or ""), str(kind or "Unknown").replace("_", " ").title())


def check_api_health() -> bool:
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_verify_samples() -> Dict[str, str]:
    return {
        "None": "",
        "True claim - Taj Mahal": "The Taj Mahal is located in Agra, Uttar Pradesh.",
        "False claim - Chandrayaan-3": "Chandrayaan-3 landed on Mars in August 2023.",
        "Health misinformation": "COVID-19 vaccines contain microchips used to track people.",
        "Mixed claims": (
            "The Taj Mahal is located in Agra. "
            "Chandrayaan-3 successfully landed on the Moon in 2023. "
            "India won the FIFA World Cup in 2022."
        ),
        "Article URL - Chandrayaan-3": "https://en.wikipedia.org/wiki/Chandrayaan-3",
        "X/Twitter single post - ISRO": "https://x.com/isro/status/1694545322251571687",
        "X/Twitter profile - ISRO": "https://x.com/isro",
        "YouTube URL - sample": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "Recirculated-media style claim": "A viral video shared today claims to show a current protest in Delhi, but the footage is from 2020.",
    }


def get_comparison_samples() -> Dict[str, str]:
    return {
        "None": "",
        "True factual claim": "The Taj Mahal is located in Agra, Uttar Pradesh.",
        "False factual claim": "Chandrayaan-3 landed on Mars in August 2023.",
        "Health misinformation": "COVID-19 vaccines contain microchips used to track people.",
        "Article-style text": (
            "The Indian Space Research Organisation announced updates after the successful Chandrayaan-3 landing. "
            "The mission demonstrated India's ability to perform a soft landing on the lunar surface and support future lunar research."
        ),
    }


def _extract_primary_claim_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    claims = _safe_list(result.get("claims_analyzed"))
    if len(claims) != 1:
        return None
    return _safe_dict(_safe_dict(claims[0]).get("result"))


def _derive_overall_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    primary = _extract_primary_claim_result(result)
    if primary:
        summary = _safe_dict(primary.get("assessment_summary"))
        if summary:
            return summary

    claims = _safe_list(result.get("claims_analyzed"))
    labels = []
    risks = []
    for claim in claims:
        claim_result = _safe_dict(_safe_dict(claim).get("result"))
        labels.append(claim_result.get("predicted_label", "Uncertain"))
        risks.append(claim_result.get("risk_label", "Medium Risk"))

    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    if len(claims) > 1:
        count_text = ", ".join(f"{k}: {v}" for k, v in counts.items())
        why = f"{len(claims)} checkable claims were verified independently. Claim-level results: {count_text}."
    else:
        why = "No claim-level explanation was available. Review extraction status and technical details."

    return {
        "final_assessment": result.get("overall_label", "Uncertain"),
        "misinformation_risk": result.get("overall_risk", "Medium Risk"),
        "evidence_verdict": "See claim-level results" if len(claims) > 1 else "Insufficient Evidence",
        "evidence_quality": "Per-claim" if len(claims) > 1 else "Insufficient",
        "why": why,
    }


def render_not_analyzed(result: Dict[str, Any]) -> None:
    extraction = _safe_dict(result.get("extraction"))
    metadata = _safe_dict(extraction.get("metadata"))
    limitations = _safe_list(result.get("limitations"))
    reason = metadata.get("failure_reason")

    st.subheader("Analysis not run")
    if reason == "x_api_credits_required":
        st.warning("X/Twitter profile timeline retrieval requires X API credits.")
        st.markdown(
            "The profile-analysis capability is configured, but X returned **402 Payment Required / credits depleted**. "
            "You can still demonstrate **single public X/Twitter post verification** through public oEmbed without paid API credits."
        )
    elif reason == "x_api_token_missing":
        st.warning("X/Twitter profile analysis requires an X API Bearer Token.")
        st.markdown("Single public X/Twitter post verification remains available without an API token.")
    elif reason == "x_api_rate_limited":
        st.warning("The X API rate limit was reached. Retry later or verify an individual public post instead.")
    else:
        st.warning("The source could not be extracted reliably, so no misinformation verdict was produced.")

    if limitations:
        with st.expander("Why analysis was not run", expanded=True):
            for item in limitations:
                st.write(f"- {item}")

    with st.expander("Technical extraction details"):
        st.json(extraction)


def render_assessment_header(result: Dict[str, Any], *, profile_sample: bool = False) -> None:
    summary = _derive_overall_summary(result)
    label = result.get("overall_label") or summary.get("final_assessment") or "Uncertain"
    risk = result.get("overall_risk") or summary.get("misinformation_risk") or "Medium Risk"
    verdict = summary.get("evidence_verdict", "Insufficient Evidence")
    quality = summary.get("evidence_quality", "Insufficient")

    st.subheader("Recent-Post Sample Assessment" if profile_sample else "Final Assessment")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample Label" if profile_sample else "Assessment", f"{_label_icon(label)} {label}")
    c2.metric("Misinformation Risk", f"{_risk_icon(risk)} {risk}")
    c3.metric("Evidence Verdict", verdict)
    c4.metric("Evidence Quality", quality)

    why = str(summary.get("why") or "").strip()
    if why:
        st.markdown("#### Why this result?")
        st.info(why)

    if profile_sample:
        st.caption("This assessment summarizes only the retrieved recent-post sample. It is not a rating of the account or its owner.")


def render_what_was_analyzed(result: Dict[str, Any]) -> None:
    extraction = _safe_dict(result.get("extraction"))
    source_metadata = _safe_dict(result.get("source_metadata"))
    metadata = _safe_dict(extraction.get("metadata"))
    claims = _safe_list(result.get("claims_extracted"))

    st.subheader("What was analyzed?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected Input Type", _human_input_kind(result.get("input_kind", "")))
    c2.metric("Content Extraction", str(extraction.get("status", "N/A")).title())
    c3.metric("Checkable Claims", len(claims))

    if source_metadata.get("title"):
        st.write("**Source/title:**", source_metadata.get("title"))
    if source_metadata.get("url"):
        st.write("**Source URL:**", source_metadata.get("url"))

    twitter_kind = metadata.get("twitter_source_kind")
    if twitter_kind == "post":
        x1, x2, x3 = st.columns(3)
        x1.metric("Post Identity", "Verified" if metadata.get("identity_verified") else "Not verified")
        x2.metric("Extraction Quality", metadata.get("extraction_quality", "N/A"))
        x3.metric("Post ID", metadata.get("resolved_status_id") or metadata.get("requested_status_id") or "N/A")
        if metadata.get("manual_text_fallback"):
            st.warning("Manual post text was used. The text-to-URL association was not independently verified.")

    preview = str(extraction.get("text") or "").strip()
    if preview:
        with st.expander("View extracted content", expanded=False):
            st.write(preview)


def _evidence_display_label(label: str) -> str:
    value = str(label or "NOT_ENOUGH_INFO").upper()
    return STANCE_DISPLAY.get(value, value.replace("_", " ").title())


def render_evidence_for_result(claim_result: Dict[str, Any], limit: int = 3) -> None:
    evidence = _safe_list(claim_result.get("evidence"))
    if not evidence:
        st.info("No retrieved evidence was available for this result.")
        return

    summary = _safe_dict(claim_result.get("evidence_summary"))
    min_relevance = summary.get("minimum_relevance_required", 0.5)
    try:
        min_relevance = float(min_relevance)
    except Exception:
        min_relevance = 0.5

    display_evidence = select_source_diverse_evidence(
        [_safe_dict(item) for item in evidence],
        limit=limit,
        min_relevance=min_relevance,
    )

    st.subheader("Top Evidence")
    st.caption(build_evidence_coverage_text(summary, len(evidence)))
    if len(display_evidence) < min(limit, len(evidence)):
        st.caption("Main view prioritizes independent usable sources rather than repeated pages from the same source.")

    for index, raw in enumerate(display_evidence, start=1):
        item = _safe_dict(raw)
        source = item.get("source") or "Unknown source"
        title = item.get("title") or ""
        label = _evidence_display_label(item.get("evidence_label"))
        relevance = _fmt_score(item.get("relevance_score"))
        heading = f"{index}. {source} — {label}"
        st.markdown(f"**{heading}**")
        if title:
            st.caption(title)
        text = str(item.get("text") or "")
        st.write(text[:700] + ("..." if len(text) > 700 else ""))
        if item.get("url"):
            st.caption(f"Source URL: {item.get('url')}")
        st.caption(f"Evidence relevance: {relevance}")
        st.markdown("---")


def render_claim_card(claim: Dict[str, Any], expanded: bool = False) -> None:
    claim_id = claim.get("claim_id", "?")
    claim_text = claim.get("claim_text", "")
    result = _safe_dict(claim.get("result"))
    label = result.get("predicted_label", "Uncertain")
    risk = result.get("risk_label", "Medium Risk")
    summary = _safe_dict(result.get("assessment_summary"))
    title = f"Claim {claim_id}: {_label_icon(label)} {label} | {_risk_icon(risk)} {risk}"

    with st.expander(title, expanded=expanded):
        st.markdown(f"**Claim:** {claim_text}")
        why = summary.get("why")
        if why:
            st.info(str(why))

        c1, c2, c3 = st.columns(3)
        c1.metric("Evidence Verdict", summary.get("evidence_verdict") or "N/A")
        c2.metric("Evidence Quality", summary.get("evidence_quality") or "N/A")
        base_label = summary.get("base_classifier_label") or "N/A"
        base_confidence = summary.get("base_classifier_confidence", result.get("confidence_score"))
        c3.metric(
            "Base Classifier Signal",
            f"{base_label} · {_fmt_score(base_confidence, percent=True)}",
        )

        render_evidence_for_result(result, limit=3)

        with st.expander("Technical details", expanded=False):
            slm = _safe_dict(result.get("slm_verification"))
            evidence_summary = _safe_dict(result.get("evidence_summary"))
            verification_context = _safe_dict(claim.get("verification_context"))
            st.write("**Base classifier label:**", summary.get("base_classifier_label") or "N/A")
            st.write("**Base classifier confidence:**", _fmt_score(summary.get("base_classifier_confidence"), percent=True))
            st.write("**Qwen3.5 evidence stance:**", _evidence_display_label(slm.get("stance")) if slm else "Not used")
            if slm.get("reason"):
                st.write("**Verifier reason:**", slm.get("reason"))
            if evidence_summary:
                st.write("**Evidence summary:**")
                st.json(evidence_summary)
            if verification_context:
                st.write("**Retrieval context:**")
                st.json(verification_context)
            if result.get("retrieval_query_text"):
                st.write("**Evidence retrieval query:**", result.get("retrieval_query_text"))


def render_claim_overview(result: Dict[str, Any]) -> None:
    claims = _safe_list(result.get("claims_analyzed"))
    if not claims:
        return

    if len(claims) == 1:
        render_claim_card(_safe_dict(claims[0]), expanded=True)
        return

    st.subheader("Claim-level Results")
    rows = []
    for claim in claims:
        claim = _safe_dict(claim)
        claim_result = _safe_dict(claim.get("result"))
        summary = _safe_dict(claim_result.get("assessment_summary"))
        rows.append(
            {
                "Claim": claim.get("claim_id"),
                "Text": claim.get("claim_text"),
                "Assessment": claim_result.get("predicted_label"),
                "Misinformation Risk": claim_result.get("risk_label"),
                "Evidence Verdict": summary.get("evidence_verdict") or "N/A",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    for idx, claim in enumerate(claims):
        render_claim_card(_safe_dict(claim), expanded=(idx == 0))


def render_twitter_profile(result: Dict[str, Any]) -> None:
    social = _safe_dict(result.get("social_analysis"))
    if social.get("type") != "twitter_profile":
        return

    profile = _safe_dict(social.get("profile"))
    posts = _safe_list(social.get("posts"))
    st.subheader("X/Twitter Recent-Post Sample")
    p1, p2, p3 = st.columns(3)
    p1.metric("Profile", "@" + str(profile.get("resolved_handle") or profile.get("requested_handle") or "N/A"))
    p2.metric("Posts Retrieved", profile.get("posts_retrieved", len(posts)))
    p3.metric("Posts Analyzed", social.get("posts_analyzed", len(posts)))

    if posts:
        rows = []
        for item in posts:
            text = str(item.get("text") or "")
            rows.append(
                {
                    "Post": item.get("post_index"),
                    "Date": item.get("created_at") or "N/A",
                    "Text": text[:160] + ("..." if len(text) > 160 else ""),
                    "Assessment": item.get("label"),
                    "Risk": item.get("risk"),
                    "Evidence Verdict": _evidence_display_label(item.get("slm_stance")) if item.get("slm_stance") else "N/A",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    counts = _safe_dict(social.get("label_counts"))
    if counts:
        st.caption("Recent-post sample counts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))


def render_limitations(result: Dict[str, Any]) -> None:
    limitations = _safe_list(result.get("limitations"))
    if not limitations:
        return
    st.subheader("Coverage & Limitations")
    with st.expander("What this analysis does not verify", expanded=False):
        for item in limitations:
            st.write(f"- {item}")


def render_multiformat_result(result: Dict[str, Any]) -> None:
    claims = _safe_list(result.get("claims_analyzed"))
    extraction = _safe_dict(result.get("extraction"))
    analysis_status = result.get("analysis_status") or ("completed" if claims else "not_analyzed")

    if analysis_status != "completed" or (extraction.get("status") in {"failed", "unsupported"} and not claims):
        render_not_analyzed(result)
        return

    profile_sample = result.get("input_kind") == "twitter_profile_url"
    render_assessment_header(result, profile_sample=profile_sample)
    render_what_was_analyzed(result)
    if profile_sample:
        render_twitter_profile(result)
    render_claim_overview(result)
    render_limitations(result)

    st.caption(result.get("disclaimer", "This is a misinformation risk assessment, not an absolute truth guarantee."))
    with st.expander("Advanced diagnostics / full response", expanded=False):
        st.json(result)


def render_single_result(result: Dict[str, Any], selected_mode_label: str) -> None:
    summary = _safe_dict(result.get("assessment_summary"))
    label = result.get("predicted_label", "Uncertain")
    risk = result.get("risk_label", "Medium Risk")

    st.subheader("Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assessment", f"{_label_icon(label)} {label}")
    c2.metric("Misinformation Risk", f"{_risk_icon(risk)} {risk}")
    c3.metric("Evidence Verdict", summary.get("evidence_verdict", "Not used"))
    c4.metric("Evidence Quality", summary.get("evidence_quality", "N/A"))

    if summary.get("why"):
        st.markdown("#### Why this result?")
        st.info(summary.get("why"))

    st.markdown("#### What was analyzed?")
    st.write("**Input type:**", result.get("input_type", "N/A"))
    st.write("**Text:**", result.get("input_text", ""))

    render_evidence_for_result(result, limit=3)

    with st.expander("Model & technical details", expanded=False):
        st.write("**Inference mode:**", result.get("inference_mode"))
        st.write("**Selected mode:**", selected_mode_label)
        st.write("**Base model:**", result.get("model_used"))
        st.write("**Base classifier label:**", summary.get("base_classifier_label") or result.get("predicted_label"))
        st.write("**Base classifier confidence:**", _fmt_score(result.get("confidence_score"), percent=True))
        st.write("**Risk score:**", _fmt_score(result.get("risk_score")))
        slm = _safe_dict(result.get("slm_verification"))
        if slm:
            st.write("**SLM verifier:**", slm.get("model"))
            st.write("**SLM stance:**", _evidence_display_label(slm.get("stance")))
            st.write("**SLM confidence:**", _fmt_score(slm.get("confidence"), percent=True))
            if slm.get("reason"):
                st.write("**Verifier reason:**", slm.get("reason"))
        if result.get("evidence_summary"):
            st.write("**Evidence summary:**")
            st.json(result.get("evidence_summary"))

    st.caption(result.get("disclaimer", "This is a misinformation risk assessment, not an absolute truth guarantee."))


def call_multiformat_api(
    *,
    raw_input: str,
    uploaded_file,
    source_type: str,
    inference_mode: str,
    max_claims: int,
    enable_temporal: bool,
    twitter_manual_text: str,
    twitter_profile_post_limit: int,
) -> Dict[str, Any]:
    if uploaded_file is not None:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        }
        data = {
            "source_type": source_type,
            "inference_mode": inference_mode,
            "max_claims": str(max_claims),
            "enable_temporal_analysis": str(bool(enable_temporal)).lower(),
        }
        response = requests.post(ANALYZE_FILE_URL, files=files, data=data, timeout=300)
    else:
        payload = {
            "input": raw_input,
            "source_type": source_type,
            "inference_mode": inference_mode,
            "max_claims": max_claims,
            "enable_temporal_analysis": enable_temporal,
            "twitter_manual_text": twitter_manual_text,
            "twitter_profile_post_limit": twitter_profile_post_limit,
        }
        response = requests.post(ANALYZE_URL, json=payload, timeout=300)

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")
    return response.json()


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Evidence-Grounded Misinformation Verification",
    page_icon="🔎",
    layout="wide",
)

st.title("Evidence-Grounded Misinformation Verification Agent")
st.caption(
    "Verify claims, URLs, social posts and uploaded content using classification, evidence retrieval and Qwen3.5 evidence verification. "
    "The system is a misinformation risk assessment tool, not an absolute truth engine."
)

api_ok = check_api_health()
if not api_ok:
    st.error("FastAPI backend is not running. Start it with `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`.")
    st.stop()

st.sidebar.success("Backend connected")
st.sidebar.header("Demo Settings")
show_advanced = st.sidebar.checkbox(
    "Show advanced settings",
    value=False,
    help="Normal demo mode uses the final SLM-assisted hybrid pipeline. Enable this only for technical/model-comparison discussion.",
)

if show_advanced:
    final_mode_label = st.sidebar.selectbox(
        "Verification pipeline",
        list(INFERENCE_MODE_OPTIONS.keys()),
        index=list(INFERENCE_MODE_OPTIONS.keys()).index("SLM-assisted Hybrid Verification"),
    )
    verify_inference_mode = INFERENCE_MODE_OPTIONS[final_mode_label]
else:
    final_mode_label = "SLM-assisted Hybrid Verification"
    verify_inference_mode = DEFAULT_DEMO_MODE
    st.sidebar.caption("Demo pipeline: SLM-assisted Hybrid Verification")

verify_tab, comparison_tab = st.tabs(["Verify Content", "Model Comparison / Project Evolution"])

with verify_tab:
    st.header("Verify Content")
    st.write("Paste text or a URL, or upload a supported file. The agent detects the input, extracts checkable claims, retrieves evidence and explains the assessment.")

    sample_name = st.selectbox("Demo example", list(get_verify_samples().keys()), key="verify_sample")
    sample_value = get_verify_samples().get(sample_name, "")
    raw_input = st.text_area(
        "Text or URL",
        value=sample_value,
        height=150,
        placeholder="Paste a claim, article, URL, YouTube link or public X/Twitter post URL...",
        key="verify_input",
    )

    uploaded_file = st.file_uploader(
        "Or upload a file",
        type=["pdf", "txt", "md", "csv", "json", "log", "png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"],
        key="verify_file",
    )

    source_type = "auto"
    max_claims = 3
    enable_temporal = True
    twitter_manual_text = ""
    twitter_profile_post_limit = 5

    is_twitter = "x.com/" in raw_input.lower() or "twitter.com/" in raw_input.lower()
    with st.expander("Advanced input options", expanded=False):
        source_type_label = st.selectbox("Source type override", list(SOURCE_TYPE_OPTIONS.keys()), index=0, key="verify_source_type")
        source_type = SOURCE_TYPE_OPTIONS[source_type_label]
        max_claims = st.slider("Maximum checkable claims", 1, 5, 3, key="verify_max_claims")
        enable_temporal = st.checkbox("Time-aware / recirculation analysis", value=True, key="verify_temporal")

        if is_twitter or source_type == "twitter_url":
            st.markdown("**X/Twitter options**")
            twitter_profile_post_limit = st.slider(
                "Recent profile posts to analyze",
                1,
                10,
                5,
                key="verify_twitter_limit",
                help="Profile timelines require official X API access/credits. Single public posts use oEmbed and do not require paid API credits.",
            )
            twitter_manual_text = st.text_area(
                "Optional post-text fallback",
                value="",
                height=80,
                key="verify_twitter_fallback",
                help="Use only if a public single-post URL cannot be reliably extracted. The UI will clearly mark this as manually supplied text.",
            )

    if is_twitter:
        st.caption("Single public X/Twitter posts can be verified without paid API credits. Profile timeline retrieval uses the official X API and may require credits.")

    analyze = st.button("Verify Content", type="primary", key="verify_button")
    if analyze:
        if uploaded_file is None and not raw_input.strip():
            st.warning("Enter text/URL or upload a supported file.")
        else:
            try:
                with st.spinner("Extracting claims, retrieving evidence and verifying... "):
                    result = call_multiformat_api(
                        raw_input=raw_input,
                        uploaded_file=uploaded_file,
                        source_type=source_type,
                        inference_mode=verify_inference_mode,
                        max_claims=max_claims,
                        enable_temporal=enable_temporal,
                        twitter_manual_text=twitter_manual_text,
                        twitter_profile_post_limit=twitter_profile_post_limit,
                    )
                render_multiformat_result(result)
            except Exception as error:
                st.error("The verification request could not be completed.")
                st.exception(error)

with comparison_tab:
    st.header("Model Comparison / Project Evolution")
    st.write(
        "Use this tab to demonstrate the progression from traditional ML to DistilBERT, evidence retrieval and SLM-assisted verification. "
        "For the normal demo, use the **Verify Content** tab."
    )

    c1, c2 = st.columns(2)
    with c1:
        comparison_mode_label = st.selectbox(
            "Inference mode",
            list(INFERENCE_MODE_OPTIONS.keys()),
            index=list(INFERENCE_MODE_OPTIONS.keys()).index("SLM-assisted Hybrid Verification"),
            key="comparison_mode",
        )
        comparison_mode = INFERENCE_MODE_OPTIONS[comparison_mode_label]
    with c2:
        input_type = st.selectbox("Input type", ["claim", "headline", "article"], index=0, key="comparison_input_type")

    comparison_sample = st.selectbox("Sample", list(get_comparison_samples().keys()), key="comparison_sample")
    comparison_text = st.text_area(
        "Claim / headline / article",
        value=get_comparison_samples().get(comparison_sample, ""),
        height=190,
        key="comparison_text",
    )

    if st.button("Run Selected Mode", type="primary", key="comparison_button"):
        if not comparison_text.strip():
            st.warning("Enter content before running the model.")
        else:
            payload = {
                "text": comparison_text,
                "input_type": input_type,
                "inference_mode": comparison_mode,
            }
            try:
                with st.spinner(f"Running {comparison_mode_label}..."):
                    response = requests.post(PREDICT_URL, json=payload, timeout=180)
                if response.status_code != 200:
                    raise RuntimeError(f"API error {response.status_code}: {response.text}")
                render_single_result(response.json(), comparison_mode_label)
            except Exception as error:
                st.error("The comparison request could not be completed.")
                st.exception(error)
