# Project Flow and Code Map

## 1. Purpose

This document maps the final codebase for the evidence-grounded misinformation risk-assessment agent.

The project intentionally preserves the progression from classifier-only models to local/hybrid evidence verification and a constrained local SLM.

---

## 2. Entry Points

### `app/main.py`

Creates the FastAPI application and registers:

- prediction routes
- multi-format analysis routes

Root endpoint:

```text
/
```

API documentation:

```text
/docs
```

### `dashboard/streamlit_app.py`

User-facing Streamlit application.

It provides:

- Single Claim / Article verification
- Multi-Format Verification
- all seven inference modes
- Final Assessment
- Misinformation Risk
- Evidence Verdict
- Evidence Quality
- source-diverse Top Evidence
- claim-level results
- coverage and limitations
- technical diagnostics

---

## 3. API Layer

### `app/api/routes.py`

Defines:

```text
GET  /api/v1/health
POST /api/v1/predict
```

`/predict` is the original single-text workflow and accepts:

- text
- input type
- inference mode

### `app/api/analysis_routes.py`

Defines:

```text
POST /api/v1/analyze
POST /api/v1/analyze-file
```

`/analyze` handles plain text and URL/social/video-style inputs.

`/analyze-file` extracts uploaded file content first and then reuses the multi-claim verification workflow.

---

## 4. Schema Layer

### `app/schemas/prediction_schema.py`

Defines the single-text request/response contract.

Supported single-text input types:

```text
claim
headline
article
```

Supported inference modes:

```text
traditional_lr
traditional_svm
distilbert
distilbert_local_rag
distilbert_hybrid
slm_assisted_local
slm_assisted_hybrid
```

### `app/schemas/analysis_schema.py`

Defines the multi-format analysis request/response.

Supported source routing includes:

- plain text
- article/generic URL
- YouTube URL
- X/Twitter URL
- PDF URL
- image URL
- uploaded files through `/api/v1/analyze-file`

The response preserves:

- source metadata
- extraction information
- temporal context
- extracted claims
- per-claim results
- overall label/risk
- limitations
- disclaimer

---

## 5. Configuration Layer

### `app/core/config.py`

Loads runtime configuration from `.env`.

Important settings include:

```text
MODEL_MODE
INFERENCE_MODE
RUNTIME_MODE
TOP_K_EVIDENCE
LOCAL_FEVER_EVIDENCE_ENABLED
LOCAL_LIAR2_EVIDENCE_ENABLED
LIVE_SEARCH_ENABLED
TAVILY_API_KEY
EXA_SEARCH_ENABLED
EXA_API_KEY
SLM_ENABLED
SLM_PROVIDER
SLM_BASE_URL
SLM_MODEL
X_BEARER_TOKEN
```

Final SLM default:

```text
qwen/qwen3.5-9b
```

### `config/evidence_policy.yaml`

Externalizes evidence and risk policy, including:

- source trust
- trusted/official domains
- stance cues
- semantic relation groups
- minimum evidence relevance
- strong-evidence threshold
- aligned-evidence requirements
- suspicious-language cues
- conflict/uncertainty behavior

Important final thresholds:

```text
minimum_evidence_relevance = 0.50
strong_evidence_relevance = 0.65
minimum_aligned_evidence_count = 2
```

---

## 6. Single-Text Orchestration

### `app/services/agent_service.py`

Coordinates the `/predict` workflow:

```text
clean input
-> select inference mode
-> classifier
-> local evidence if enabled by mode
-> live evidence if enabled by mode/config
-> semantic evidence judgments
-> Qwen3.5 verification for SLM modes
-> risk calculation
-> explanation
-> assessment summary
```

Classifier-only modes do not use evidence.

Local-RAG modes use local FEVER/LIAR2 evidence.

Hybrid modes can add current web evidence.

SLM modes use Qwen3.5 only after evidence retrieval.

---

## 7. Classifier Layer

### `app/services/classifier_service.py`

Supports:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Fine-tuned DistilBERT 512

Local model paths:

```text
models/baseline/
models/transformer/distilbert_isot_full_512/
models/transformer/distilbert_isot/
```

The final transformer is the strongest article-level classifier, but classifier confidence is not treated as factual truth for short claims.

---

## 8. Input Routing and Content Extraction

### `app/services/input_router_service.py`

Detects and normalizes:

- plain text
- generic URLs
- YouTube URLs
- X/Twitter URLs
- PDF URLs
- image URLs

### `app/services/content_extraction_service.py`

Coordinates content extraction for supported routed inputs.

### `app/services/youtube_extraction_service.py`

Attempts transcript-based YouTube extraction.

If a transcript is unavailable, the system exposes the limitation instead of claiming full video verification.

### `app/services/twitter_extraction_service.py`

Handles X/Twitter text extraction.

For public individual-post URLs, public oEmbed can be used and the returned post identity is checked against the requested post.

Profile/timeline analysis uses the official API path when configured.

### `app/services/file_extraction_service.py`

Supports:

- text files: TXT, MD, CSV, JSON, LOG
- PDFs with embedded text
- images/screenshots through OCR when dependencies are installed

Image handling is OCR only and does not perform visual authenticity or deepfake detection.

---

## 9. Claim Extraction and Representation

### `app/services/claim_extraction_service.py`

Extracts independently checkable claims and applies the final claim-representation contract.

Important fields:

```text
source_text
verification_text
verification_target
assertion_status
representation_fallback
transformation_reason
```

### Why this matters

Example source statement:

```text
A viral video claims Iran tested a nuclear bomb last week.
```

The system preserves the source wording as context but verifies:

```text
Iran tested a nuclear bomb last week.
```

The representation logic is conservative:

- direct propositions remain direct
- safely embedded propositions can be isolated
- denied/questioned/non-asserted content is not silently converted into a positive fact
- unsafe transformations fall back to the original wording

---

## 10. Temporal Context

### `app/services/temporal_context_service.py`

Extracts and tracks:

- explicit years/dates
- relative-time cues
- historical-event context
- recirculated-media cues
- approximate time windows

Temporal context is used to improve evidence retrieval and interpretation, especially for old or recirculated content.

---

## 11. Evidence Query Construction

### `app/services/evidence_query_builder_service.py`

Builds retrieval queries from the verification proposition and relevant temporal/media context.

The retrieval query uses the factual proposition being verified rather than blindly using carrier/source wording.

---

## 12. Local Evidence Retrieval

### `app/services/retrieval_service.py`

Uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and persistent Chroma indexes:

```text
data/embeddings/chroma_fever/
data/embeddings/chroma_liar2_justifications/
```

FEVER collection:

```text
fever_evidence
```

LIAR2 collection:

```text
liar2_justification_evidence
```

Top-k is normally 5.

Retrieved evidence is not considered correct only because it has high semantic similarity.

---

## 13. Hybrid Web Evidence

### `app/services/web_search_service.py`

Adds current evidence in hybrid runtime modes.

Providers:

- Tavily - primary
- Exa - fallback when configured

The evidence layer uses source trust, relevance, semantic direction and policy rules before evidence is allowed to influence the final result strongly.

---

## 14. Evidence Policy and Presentation

### `app/services/evidence_policy_loader.py`

Loads:

```text
config/evidence_policy.yaml
```

### `app/services/evidence_presentation_service.py`

Builds user-facing evidence presentation.

The final UI prefers:

- usable evidence
- directional evidence
- independent sources
- source/domain diversity

Repeated pages from one source/domain do not count as independent corroboration.

---

## 15. Qwen3.5 Evidence Verification

### `app/services/slm_verifier_service.py`

Final verifier:

```text
qwen/qwen3.5-9b
```

Runtime:

```text
LM Studio
```

Task:

```text
verification proposition
+ retrieved evidence
-> SUPPORTS / REFUTES / NOT_ENOUGH_INFO
```

The verifier is not allowed to act as a new source of facts.

Important safeguards include:

- evidence-only reasoning
- JSON-style constrained output
- semantic evidence judgment
- opposite strong evidence cannot be overridden simply by SLM disagreement
- weak evidence can remain `NOT_ENOUGH_INFO`

LM Studio setting:

```text
Enable Thinking = OFF
```

---

## 16. Risk Layer

### `app/services/risk_service.py`

Combines:

- classifier signal
- evidence relevance
- evidence direction
- evidence quality
- source trust
- independent corroboration
- Qwen3.5 stance
- uncertainty/conflict policy
- suspicious-language cues where applicable

Final labels:

```text
Likely Real
Likely Fake
Misleading
Uncertain
```

Risk labels:

```text
Very Low Risk
Low Risk
Medium Risk
High Risk
```

Weak, missing, unrelated or conflicting evidence can remain `Uncertain`.

---

## 17. Multi-Claim Orchestration

### `app/services/multi_claim_agent_service.py`

Coordinates the multi-format flow:

```text
input routing
-> content extraction
-> claim extraction
-> claim representation
-> temporal context
-> per-claim evidence verification
-> per-claim risk
-> source-level aggregation
```

Each claim keeps its own evidence and temporal context so one claim does not contaminate another.

For one extracted claim, the source-level result mirrors that claim.

For multiple claims, the source-level result is aggregated after independent verification.

---

## 18. Explanation Layer

### `app/services/explanation_service.py`

Builds the human-readable explanation while keeping:

- classifier signal
- evidence summary
- SLM stance
- final assessment
- risk
- limitations

separate and visible.

The final dashboard intentionally distinguishes classifier confidence from evidence quality.

---

## 19. Final Decision Principle

The final result is not:

```text
classifier confidence = truth
```

It is an evidence-aware decision over:

```text
classifier signal
+ verification proposition
+ local/hybrid evidence
+ semantic evidence direction
+ source trust
+ independent corroboration
+ Qwen3.5 evidence stance
+ uncertainty/risk rules
```

This layered design is the central technical contribution of the final project.
