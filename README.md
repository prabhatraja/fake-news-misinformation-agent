# Explainable Real-Time Fake News and Misinformation Detection Agent

An evidence-grounded misinformation risk-assessment system combining traditional machine learning, a fine-tuned DistilBERT classifier, local and hybrid evidence retrieval, and a constrained Qwen3.5 verifier.

This repository contains the final implementation of an M.Tech Data Science & Artificial Intelligence individual project.

## What the system does

The system goes beyond simple fake/real classification. It:

- accepts plain text, articles, URLs, YouTube links, X/Twitter links and uploaded files
- extracts independently checkable claims from longer or mixed content
- preserves source and temporal context
- separates original `source_text` from the factual `verification_text` when needed
- retrieves evidence from local FEVER and LIAR2 Chroma indexes
- optionally enriches evidence with Tavily and Exa web search
- uses Qwen3.5 9B through LM Studio to classify retrieved evidence as:
  - `SUPPORTS`
  - `REFUTES`
  - `NOT_ENOUGH_INFO`
- applies evidence-quality, source-independence and corroboration controls
- returns:
  - `Likely Real`
  - `Likely Fake`
  - `Misleading`
  - `Uncertain`
- provides misinformation risk, evidence quality, explanations and visible limitations

The SLM is used only as an **evidence verifier**. It is not treated as an independent source of truth.

---

## Project evolution

The repository intentionally retains all major stages of the project:

```text
TF-IDF + Logistic Regression / Linear SVM
        ↓
Fine-tuned DistilBERT 512
        ↓
Local FEVER + LIAR2 evidence retrieval
        ↓
Hybrid Tavily + Exa evidence
        ↓
Qwen3.5 evidence-stance verification
        ↓
Claim representation + temporal context
        ↓
Multi-claim aggregation
        ↓
Explainable risk-aware dashboard
```

## Final inference modes

1. `traditional_lr`
2. `traditional_svm`
3. `distilbert`
4. `distilbert_local_rag`
5. `distilbert_hybrid`
6. `slm_assisted_local`
7. `slm_assisted_hybrid`

The strongest final demonstration mode is:

```text
slm_assisted_hybrid
```

---

## Architecture

Main layers:

```text
User Input
  ↓
Streamlit Dashboard
  ↓
FastAPI
  ↓
Input Routing and Content Extraction
  ↓
Claim Extraction / Representation / Temporal Context
  ↓
Classifier + Local/Hybrid Evidence Retrieval
  ↓
Evidence Policy and Semantic Quality Controls
  ↓
Qwen3.5 Evidence Verification
  ↓
Risk Scoring and Multi-Claim Aggregation
  ↓
Final Assessment + Evidence + Explanation + Limitations
```

---

## Final results

### ISOT article-level classification

| Model | Accuracy | F1-score |
|---|---:|---:|
| TF-IDF + Logistic Regression | 98.63% | 98.50% |
| TF-IDF + Linear SVM | 99.37% | 99.31% |
| DistilBERT 256 | 99.95% | 99.94% |
| **DistilBERT 512** | **99.99%** | **99.99%** |

Final DistilBERT 512 result:

```text
Train records: 31,280
Test records: 7,820
Accuracy: 99.99%
Precision: 99.97%
Recall: 100.00%
F1-score: 99.99%
Errors: 1 / 7,820
```

These are article-level ISOT results and are **not** interpreted as universal truth-detection accuracy.

### LIAR2 offline evaluation

```text
Evaluated samples: 200
Safe behavior: 99.5%
Usable evidence coverage: 85.0%
Uncertain rate: 90.5%
Correct non-Uncertain binary cases: 17 / 17
```

The 17/17 figure is not reported as general claim-level accuracy.

### Final hardening and validation

```text
Deterministic claim representation: 25 / 25
Live Qwen claim representation: 11 / 11
Evidence / response semantics: 14 / 14
Final label-risk consistency: 4 / 4
X/Twitter hardening: 5 / 5
Golden expected direction: 11 / 12
Golden internal coherence: 12 / 12
Offline E2E: 8 / 8, 0 critical failures, 0 warnings
Full E2E: 8 / 8, 0 critical failures, 0 warnings
```

### SLM-assisted latency

```text
15 requests
Average: 2.227 s
Minimum: 1.902 s
Maximum: 2.587 s
```

---

## Evidence indexes

### FEVER

```text
20,000 records
SUPPORTS: 12,105
REFUTES: 4,621
NOT_ENOUGH_INFO: 3,274
```

### LIAR2 Enhanced

```text
20,665 train/validation records
Test split excluded
SUPPORTS: 5,412
REFUTES: 8,673
NOT_ENOUGH_INFO: 6,580
```

Final evidence thresholds:

```text
Minimum usable relevance: 0.50
Strong evidence relevance: 0.65
Minimum aligned evidence count: 2
```

---

## Repository structure

```text
fake-news-misinformation-agent/
├── app/
│   ├── api/
│   ├── core/
│   ├── schemas/
│   └── services/
├── config/
│   └── evidence_policy.yaml
├── dashboard/
│   └── streamlit_app.py
├── data/
│   └── README.md
├── models/
│   └── README.md
├── reports/
├── scripts/
│   └── README.md
├── .env.example
├── .gitignore
├── environment_check.py
├── FINAL_VALIDATION_SUMMARY.md
├── PHASE2_PLAN.md
├── PROJECT_CLEANUP_SUMMARY.md
├── PROJECT_FLOW.md
├── README.md
├── README_RUNBOOK.md
└── requirements.txt
```

Large datasets, trained models and persistent vector indexes are intentionally excluded from the Git repository. Their expected local paths are documented in `data/README.md` and `models/README.md`.

---

## Setup

### 1. Create the Python environment

Python 3.10 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The original project was developed in a Conda environment named `fake-news-agent`.

### 2. Configure environment variables

Copy:

```text
.env.example
```

to:

```text
.env
```

Add only the credentials required for the selected mode.

**Never commit `.env`.**

### 3. Local SLM

Open LM Studio and load:

```text
qwen/qwen3.5-9b
```

Recommended settings:

```text
Temperature: 0
Enable Thinking: OFF
Server: http://127.0.0.1:1234/v1
```

### 4. Start FastAPI

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health:

```text
http://127.0.0.1:8000/api/v1/health
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### 5. Start Streamlit

In another terminal:

```powershell
streamlit run dashboard\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

For the strongest final demo, select:

```text
SLM-assisted Hybrid Verification
```

---

## Main API endpoints

```text
GET  /api/v1/health
POST /api/v1/predict
POST /api/v1/analyze
POST /api/v1/analyze-file
```

---

## Validation

Start FastAPI first.

### Full E2E

```powershell
python scripts\run_e2e_validation_suite.py
```

### Offline E2E

```powershell
python scripts\run_e2e_validation_suite.py --skip-external
```

### Golden regression

```powershell
python scripts\run_golden_claims_regression.py
```

### SLM latency benchmark

```powershell
python scripts\benchmark_slm_latency.py
```

---

## Example claims

Supported factual claim:

```text
Chandrayaan-3 successfully soft-landed on the Moon in August 2023.
```

Materially false claim:

```text
Chandrayaan-3 landed on Mars in August 2023.
```

Multi-claim example:

```text
The Eiffel Tower is located in Paris.
The Eiffel Tower was constructed between 1887 and 1889.
A viral video claims Iran tested a nuclear bomb last week.
```

The third source statement is preserved for context, while the substantive proposition is verified separately.

---

## Important limitations

- This is a misinformation-risk assessment system, not an absolute truth engine.
- ISOT performance is article-level and may reflect source/topic/style patterns.
- Static FEVER/LIAR2 indexes cannot cover every recent event.
- Hybrid evidence depends on external search availability and can change over time.
- Qwen3.5 can still be affected by incomplete or ambiguous evidence snippets.
- YouTube verification depends on transcript availability.
- X/Twitter verification is mainly text-based and does not fully verify attached media, replies or thread context.
- Image support is OCR/text-oriented, not deepfake or visual-forensics detection.
- Human review remains important for high-impact decisions.

---

## Detailed documentation

- `README_RUNBOOK.md` - detailed runtime and demo guide
- `PROJECT_FLOW.md` - implementation/code flow
- `FINAL_VALIDATION_SUMMARY.md` - final measured results
- `PHASE2_PLAN.md` - completed Phase 2 implementation record
- `PROJECT_CLEANUP_SUMMARY.md` - final packaging guidance
- `scripts/README.md` - retained scripts and validation commands
- `data/README.md` - expected data/index locations
- `models/README.md` - expected model locations

---

## Author

**Peri Prabhat Raja**  
M.Tech Data Science & Artificial Intelligence  
PES University  
Student ID: PES2PGE24DS192

---

## Academic use

This repository contains the implementation of an academic final project. Dataset, model, API and library ownership remains with their respective authors and providers. See the project report and documentation for citations and references.
