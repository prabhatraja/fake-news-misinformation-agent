# Fake News and Misinformation Detection Agent - Final Runbook

## 1. Project

**Title:** Explainable Real-Time Fake News and Misinformation Detection Agent with Evidence Retrieval Using Transformers and RAG

This project implements an evidence-grounded misinformation risk-assessment agent. It combines traditional machine-learning baselines, a fine-tuned DistilBERT classifier, local and hybrid evidence retrieval, constrained Qwen3.5 evidence verification, temporal/context handling, multi-claim analysis, risk scoring and an explainable Streamlit dashboard.

The system is a **decision-support and misinformation-risk assessment tool**, not an absolute authority on truth.

---

## 2. Final Architecture

The final flow is:

```text
User input
  -> Streamlit dashboard
  -> FastAPI
  -> input routing / extraction
  -> claim extraction + claim representation + temporal context
  -> classifier
  -> local/hybrid evidence retrieval
  -> evidence policy + semantic quality controls
  -> Qwen3.5 evidence verification
  -> risk scoring + multi-claim aggregation
  -> final assessment + evidence + explanation + limitations
```

### Main components

- **Traditional ML:** TF-IDF + Logistic Regression, TF-IDF + Linear SVM
- **Transformer:** Fine-tuned DistilBERT 512
- **Local evidence:** FEVER and LIAR2 Enhanced Chroma indexes
- **Hybrid web evidence:** Tavily with Exa fallback
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Local SLM verifier:** `qwen/qwen3.5-9b` through LM Studio
- **Backend:** FastAPI
- **Dashboard:** Streamlit
- **Evidence policy:** `config/evidence_policy.yaml`

---

## 3. Final Inference Modes

All seven modes are retained because they show the evolution of the project:

1. `traditional_lr`
2. `traditional_svm`
3. `distilbert`
4. `distilbert_local_rag`
5. `distilbert_hybrid`
6. `slm_assisted_local`
7. `slm_assisted_hybrid`

The default final demonstration mode is:

```text
slm_assisted_hybrid
```

---

## 4. Supported Inputs

### Single Claim / Article flow

`/api/v1/predict` accepts:

- claim
- headline
- article

### Multi-Format flow

`/api/v1/analyze` and `/api/v1/analyze-file` support:

- plain text
- web/article URLs
- YouTube URLs when a transcript is available
- public X/Twitter post URLs
- X/Twitter profile analysis when official API access is configured
- PDF URLs
- image URLs
- uploaded TXT / MD / CSV / JSON / LOG files
- uploaded PDFs
- uploaded images/screenshots through OCR when OCR dependencies are installed

Important limitations:

- YouTube verification is transcript-based.
- X/Twitter verification is mainly text-based; attached media, replies, quote-post context and full thread context are not fully verified.
- Image analysis is OCR/text extraction, not deepfake or visual-forensics detection.
- Live web evidence can change over time.

---

## 5. Environment Configuration

Copy `.env.example` to `.env` and add only the credentials required for the selected runtime mode.

**Never submit or commit the real `.env` file.**

Final SLM configuration:

```env
INFERENCE_MODE=slm_assisted_hybrid
RUNTIME_MODE=hybrid

SLM_ENABLED=true
SLM_PROVIDER=lmstudio
SLM_BASE_URL=http://127.0.0.1:1234/v1
SLM_MODEL=qwen/qwen3.5-9b
SLM_TEMPERATURE=0.0
SLM_TIMEOUT_SECONDS=180
```

For hybrid evidence, configure Tavily and optionally Exa:

```env
LIVE_SEARCH_ENABLED=true
TAVILY_API_KEY=<your-key>

EXA_SEARCH_ENABLED=true
EXA_API_KEY=<your-key>
```

For a fully local/offline demonstration, live search can remain disabled.

---

## 6. Local Runtime Assets

The runtime expects local model/evidence assets under the project tree.

### Baseline models

```text
models/baseline/tfidf_vectorizer.joblib
models/baseline/logistic_regression_model.joblib
models/baseline/linear_svm_model.joblib
```

### DistilBERT

The classifier prefers:

```text
models/transformer/distilbert_isot_full_512/
```

and falls back to:

```text
models/transformer/distilbert_isot/
```

### Local embedding model

Preferred offline path:

```text
models/embedding/all-MiniLM-L6-v2/
```

If this directory is not present, the retrieval service can fall back to the Hugging Face model name during development.

### Chroma evidence indexes

```text
data/embeddings/chroma_fever/
data/embeddings/chroma_liar2_justifications/
```

If a source-code-only submission excludes large datasets/models/indexes, retain the README files in `data/` and `models/` explaining the expected locations.

---

## 7. Start LM Studio

1. Open LM Studio.
2. Load:

```text
qwen/qwen3.5-9b
```

3. Set:
   - Temperature: `0`
   - **Enable Thinking: OFF**
   - GPU offload: Max / Auto
4. Start the local OpenAI-compatible server.
5. Confirm:

```text
http://127.0.0.1:1234/v1
```

PowerShell check:

```powershell
Invoke-RestMethod http://127.0.0.1:1234/v1/models
```

The verifier is deliberately constrained to the retrieved evidence and returns:

```text
SUPPORTS
REFUTES
NOT_ENOUGH_INFO
```

It is not used as an independent source of truth.

---

## 8. Start FastAPI

From the project root:

```powershell
conda activate fake-news-agent
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Main endpoints:

```text
GET  /api/v1/health
POST /api/v1/predict
POST /api/v1/analyze
POST /api/v1/analyze-file
```

---

## 9. Start Streamlit

Open a second PowerShell window:

```powershell
conda activate fake-news-agent
streamlit run dashboard\streamlit_app.py
```

Default URL:

```text
http://localhost:8501
```

For the final demonstration select:

```text
SLM-assisted Hybrid Verification
```

---

## 10. Recommended Final Demo Story

### Demo A - Supported factual claim

```text
Chandrayaan-3 successfully soft-landed on the Moon in August 2023.
```

Validated final behavior:

```text
Likely Real
Evidence Verdict: Supports Claim
```

### Demo B - Materially false claim

```text
Chandrayaan-3 landed on Mars in August 2023.
```

Validated final behavior:

```text
Likely Fake
High Risk
Evidence Verdict: Contradicts Claim
```

### Demo C - Multi-claim isolation

Use:

```text
The Eiffel Tower is located in Paris.
The Eiffel Tower was constructed between 1887 and 1889.
A viral video claims Iran tested a nuclear bomb last week.
```

The system should extract and verify the claims independently. The carrier wording is preserved as `source_text`, while the embedded proposition is verified separately as:

```text
Iran tested a nuclear bomb last week.
```

### Demo D - Project evolution

For:

```text
The Eiffel Tower is located in Paris.
```

the final E2E validation demonstrated the value of the retained modes:

| Mode | Validated output |
|---|---|
| `traditional_lr` | Likely Fake |
| `traditional_svm` | Likely Fake |
| `distilbert` | Likely Fake |
| `distilbert_local_rag` | Uncertain |
| `distilbert_hybrid` | Uncertain |
| `slm_assisted_local` | Likely Real |
| `slm_assisted_hybrid` | Likely Real |

This is the clearest example of why classifier confidence alone is not sufficient for short factual claims.

---

## 11. Final Validation Commands

Start FastAPI first.

### Full E2E validation

```powershell
python scripts\run_e2e_validation_suite.py
```

Final recorded result:

```text
8 / 8 cases passed
0 critical failures
0 warnings
```

### Offline/local E2E validation

```powershell
python scripts\run_e2e_validation_suite.py --skip-external
```

Final recorded result:

```text
8 / 8 cases passed
0 critical failures
0 warnings
```

### Golden factual regression

For strongest results, use LM Studio + hybrid evidence:

```powershell
python scripts\run_golden_claims_regression.py
```

Final recorded result:

```text
11 / 12 expected directional outcomes
12 / 12 internal coherence
```

The single directional miss remained `Uncertain` because available evidence was not strong enough. Thresholds were not weakened only to force a 12/12 score.

### Optional latency benchmark

```powershell
python scripts\benchmark_slm_latency.py
```

Recorded final benchmark:

```text
15 requests
Average: 2.227 s
Minimum: 1.902 s
Maximum: 2.587 s
```

Validation reports are stored under:

```text
reports/e2e/
reports/quality/
reports/metrics/
```

---

## 12. Final Reported Metrics

### ISOT article classifier

Fine-tuned DistilBERT 512:

```text
Train records: 31,280
Test records: 7,820
Accuracy: 99.99%
Precision: 99.97%
Recall: 100.00%
F1-score: 99.99%
Confusion matrix: [[4238, 1], [0, 3581]]
```

These are article-level ISOT results and are not interpreted as universal truth-detection accuracy.

### LIAR2 offline evaluation

```text
Evaluated: 200
Safe behavior: 199/200 = 99.5%
Usable evidence coverage: 170/200 = 85.0%
Uncertain: 181/200 = 90.5%
Correct non-Uncertain binary cases: 17/17
```

The 17/17 figure is **not** reported as general claim-level accuracy.

### Evidence indexes

FEVER:

```text
20,000 records
SUPPORTS: 12,105
REFUTES: 4,621
NOT_ENOUGH_INFO: 3,274
```

LIAR2 Enhanced:

```text
20,665 train/validation records
Test split excluded
SUPPORTS: 5,412
REFUTES: 8,673
NOT_ENOUGH_INFO: 6,580
```

### Evidence thresholds

```text
Minimum usable relevance: 0.50
Strong evidence relevance: 0.65
Minimum aligned evidence for stronger direction: 2
```

---

## 13. Environment Check

To confirm the local Python/GPU setup:

```powershell
python environment_check.py
```

This checks the main Python libraries and whether PyTorch can see CUDA/GPU support.

---

## 14. Final Submission Hygiene

Before creating the final code ZIP:

- remove the real `.env`
- keep `.env.example`
- remove `.vscode/`
- remove `archive/`
- remove all `__pycache__/` directories
- remove `.pyc` and `.pyo` files
- remove temporary patch scripts and one-off development outputs
- remove obsolete `reports/megabatch*.txt` files
- retain the final application, config, dashboard, retained scripts and useful final metrics/reports
- if large model/data/index assets are excluded, keep `data/README.md` and `models/README.md`

The submission package must not contain API keys, tokens or other secrets.

---

## 15. Final Viva Message

The project should be explained as an evolution:

```text
Traditional ML
-> DistilBERT 512
-> local evidence retrieval
-> hybrid current evidence
-> constrained Qwen3.5 evidence verification
-> claim representation + temporal context
-> multi-claim aggregation
-> explainable risk-aware dashboard
```

The key contribution is not a single accuracy number. It is the complete evidence-grounded workflow that can preserve source context, verify the factual proposition, distinguish support from contradiction or insufficient evidence, return uncertainty when appropriate, and expose the reason and limitations to the user.
