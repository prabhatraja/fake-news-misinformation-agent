from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="News article, headline, or factual claim text.",
    )
    input_type: Optional[str] = Field(
        default="claim",
        description="Input type: claim, headline, or article.",
    )
    inference_mode: Optional[str] = Field(
        default=None,
        description=(
            "Inference mode: traditional_lr, traditional_svm, distilbert, "
            "distilbert_local_rag, distilbert_hybrid, slm_assisted_local, "
            "or slm_assisted_hybrid."
        ),
    )


class EvidenceItem(BaseModel):
    source: str
    text: str
    relevance_score: float
    evidence_label: str
    source_type: Optional[str] = None
    source_name: Optional[str] = None
    provider: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    trust_score: Optional[float] = None
    freshness_score: Optional[float] = None
    final_evidence_score: Optional[float] = None
    heuristic_evidence_label: Optional[str] = None
    slm_evidence_label: Optional[str] = None
    slm_evidence_confidence: Optional[float] = None
    evidence_label_source: Optional[str] = None


class PredictionResponse(BaseModel):
    input_text: str
    input_type: str
    predicted_label: str
    confidence_score: float
    risk_score: float
    risk_label: str
    explanation: str
    influential_terms: List[str]
    evidence: List[EvidenceItem]
    disclaimer: str

    inference_mode: Optional[str] = None
    model_used: Optional[str] = None
    raw_class: Optional[int] = None
    class_probabilities: Optional[Dict[str, float]] = None
    evidence_summary: Optional[Dict] = None
    slm_verification: Optional[Dict] = None
    assessment_summary: Optional[Dict] = None
