from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    # Request model for multi-format verification.

    input: str = Field(
        ...,
        min_length=5,
        description="Plain text, article URL, YouTube URL, X/Twitter URL, PDF URL, image URL, or another supported input.",
    )
    source_type: Optional[str] = Field(
        default="auto",
        description="auto, text, url, youtube_url, twitter_url, pdf_url, image_url. Uploaded files use /api/v1/analyze-file.",
    )
    inference_mode: Optional[str] = Field(
        default="slm_assisted_hybrid",
        description="Existing inference mode to use for claim-level verification.",
    )
    max_claims: Optional[int] = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of checkable claims to extract and verify.",
    )
    enable_temporal_analysis: Optional[bool] = Field(
        default=True,
        description="Whether to extract event/source time context for historical or time-sensitive claims.",
    )
    twitter_manual_text: Optional[str] = Field(
        default="",
        description="Optional visible post text fallback for a single X/Twitter post when public extraction is blocked or cannot be verified.",
    )
    twitter_profile_post_limit: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="For an X/Twitter profile URL, analyze up to this many recent public original posts.",
    )


class AnalysisResponse(BaseModel):
    # Loose response model for multi-format verification.

    original_input: str
    input_kind: str
    inference_mode: str
    source_metadata: Dict[str, Any]
    extraction: Dict[str, Any]
    temporal_context: Dict[str, Any]
    claims_extracted: List[Dict[str, Any]]
    claims_analyzed: List[Dict[str, Any]]
    analysis_status: Optional[str] = None
    overall_label: str
    overall_risk: str
    limitations: List[str]
    disclaimer: str
    social_analysis: Optional[Dict[str, Any]] = None
