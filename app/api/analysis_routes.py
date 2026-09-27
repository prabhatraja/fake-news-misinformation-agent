from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.analysis_schema import AnalysisRequest, AnalysisResponse
from app.services.file_extraction_service import extract_uploaded_file
from app.services.multi_claim_agent_service import analyze_extracted_content, analyze_multi_format_input

router = APIRouter()


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _failed_file_response(
    *,
    filename: str,
    source_type: str,
    inference_mode: str,
    extraction: Dict[str, Any],
) -> Dict[str, Any]:
    source_metadata = {
        "source_type": extraction.get("input_kind", source_type or "uploaded_file"),
        "url": "",
        "domain": "",
        "title": extraction.get("title") or filename,
        "author": "",
        "published_date": "",
        "metadata": extraction.get("metadata", {}),
    }
    limitations = extraction.get("limitations", []) or []
    return {
        "original_input": filename,
        "input_kind": extraction.get("input_kind", "uploaded_file"),
        "inference_mode": inference_mode,
        "source_metadata": source_metadata,
        "extraction": extraction,
        "temporal_context": {},
        "claims_extracted": [],
        "claims_analyzed": [],
        "analysis_status": "not_analyzed",
        "overall_label": "Uncertain",
        "overall_risk": "Medium Risk",
        "limitations": _dedupe(list(limitations)),
        "disclaimer": (
            "The agent could not extract enough usable text from the uploaded file. "
            "For reliable analysis, upload a text-based PDF, paste the text, or provide a transcript."
        ),
    }


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):
    return analyze_multi_format_input(
        raw_input=request.input,
        source_type=request.source_type or "auto",
        inference_mode=request.inference_mode or "slm_assisted_hybrid",
        max_claims=request.max_claims or 3,
        enable_temporal_analysis=bool(request.enable_temporal_analysis),
        twitter_manual_text=request.twitter_manual_text or "",
        twitter_profile_post_limit=max(1, min(int(request.twitter_profile_post_limit or 5), 10)),
    )


@router.post("/analyze-file", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    source_type: str = Form("auto"),
    inference_mode: str = Form("slm_assisted_hybrid"),
    max_claims: int = Form(3),
    enable_temporal_analysis: bool = Form(True),
):
    """Analyze an uploaded PDF/text/image file.

    This endpoint extracts text first, then reuses the existing multi-format
    claim extraction + evidence + SLM verification pipeline. Image support is
    OCR-only when local OCR dependencies are installed; it does not perform
    deepfake or visual-forensics analysis.
    """
    data = await file.read()
    filename = file.filename or "uploaded_file"
    content_type = file.content_type or "application/octet-stream"

    extraction = extract_uploaded_file(
        data,
        filename=filename,
        content_type=content_type,
        source_type=source_type or "auto",
    )

    if extraction.get("status") in {"failed", "unsupported"} or not extraction.get("text"):
        return _failed_file_response(
            filename=filename,
            source_type=source_type,
            inference_mode=inference_mode,
            extraction=extraction,
        )

    input_kind = extraction.get("input_kind", "uploaded_file")
    source_metadata = {
        "source_type": input_kind,
        "url": extraction.get("url", ""),
        "domain": extraction.get("domain", ""),
        "title": extraction.get("title") or filename,
        "author": extraction.get("author", ""),
        "published_date": extraction.get("published_date", ""),
        "metadata": extraction.get("metadata", {}),
    }

    # Analyze extracted content while preserving the real uploaded-file input kind.
    # This enables file/OCR-specific guardrails and prevents source-level context
    # from being treated as if the user typed one plain-text claim.
    return analyze_extracted_content(
        original_input=filename,
        input_kind=input_kind,
        content_text=extraction.get("text", ""),
        extraction=extraction,
        source_metadata=source_metadata,
        inference_mode=inference_mode or "slm_assisted_hybrid",
        max_claims=max(1, min(int(max_claims or 3), 5)),
        enable_temporal_analysis=bool(enable_temporal_analysis),
    )
