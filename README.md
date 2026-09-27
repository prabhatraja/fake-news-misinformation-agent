# Explainable Real-Time Fake News and Misinformation Detection Agent

An evidence-grounded misinformation risk-assessment system combining traditional machine learning, a fine-tuned DistilBERT classifier, local and hybrid evidence retrieval, and a constrained Qwen3.5 verifier.

This repository contains the final implementation of an **M.Tech Data Science & Artificial Intelligence individual project at PES University**.

---

## What the System Does

The system goes beyond simple fake/real classification. It:

- accepts plain text, articles, URLs, YouTube links, X/Twitter links, and uploaded files
- extracts independently checkable claims from longer or mixed content
- preserves source and temporal context
- separates original `source_text` from factual `verification_text` when required
- retrieves evidence from local FEVER and LIAR2 Chroma indexes
- optionally enriches evidence using Tavily and Exa web search
- uses Qwen3.5 9B through LM Studio to classify retrieved evidence as:
  - `SUPPORTS`
  - `REFUTES`
  - `NOT_ENOUGH_INFO`
- applies evidence-quality, semantic-alignment, source-independence, and corroboration controls
- produces one of:
  - `Likely Real`
  - `Likely Fake`
  - `Misleading`
  - `Uncertain`
- provides misinformation risk, evidence quality, explanations, and visible limitations

The SLM is used only as an **evidence verifier**. It is not treated as an independent source of truth.

---

## Project Evolution

The repository intentionally retains all major stages of the project:

```text
TF-IDF + Logistic Regression / Linear SVM
        ↓
Fine-tuned DistilBERT 512
        ↓
Local FEVER + LIAR2 Evidence Retrieval
        ↓
Hybrid Tavily + Exa Evidence
        ↓
Qwen3.5 Evidence-Stance Verification
        ↓
Claim Representation + Temporal Context
        ↓
Multi-Claim Aggregation
        ↓
Explainable Risk-Aware Dashboard
```

This allows the project to demonstrate the progression from traditional machine learning to transformer classification and finally to evidence-grounded SLM-assisted verification.

---

## Final Inference Modes

The application retains seven inference modes:

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
Classifier + Local / Hybrid Evidence Retrieval
    ↓
Evidence Policy and Semantic Quality Controls
    ↓
Qwen3.5 Evidence Verification
    ↓
Risk Scoring and Multi-Claim Aggregation
    ↓
Final Assessment + Evidence + Explanation + Limitations
```

The architecture deliberately separates:

- classification
- retrieval
- evidence interpretation
- decision logic
- presentation

so that no individual model is treated as the sole source of factual truth.

---

## Final Results

### ISOT Article-Level Classification

| Model | Accuracy | F1-score |
|---|---:|---:|
| TF-IDF + Logistic Regression | 98.63% | 98.50% |
| TF-IDF + Linear SVM | 99.37% | 99.31% |
| DistilBERT 256 | 99.95% | 99.94% |
| **DistilBERT 512** | **99.99%** | **99.99%** |

Final DistilBERT 512 result:

```text
Train records: 31,280
Test records:  7,820

Accuracy:  99.99%
Precision: 99.97%
Recall:    100.00%
F1-score:  99.99%

Errors: 1 / 7,820
```

These are **article-level ISOT classification results** and are not interpreted as universal truth-detection accuracy.

---

### LIAR2 Offline Evaluation

```text
Evaluated samples: 200
Safe behavior: 99.5%
Usable evidence coverage: 85.0%
Uncertain rate: 90.5%
Correct non-Uncertain binary cases: 17 / 17
```

The `17 / 17` result is not reported as general claim-level accuracy.

The high `Uncertain` rate reflects the deliberately conservative behavior of the evidence-grounded system when sufficiently strong evidence is unavailable.

---

### Final Hardening and Validation

```text
Deterministic claim representation: 25 / 25
Live Qwen claim representation:     11 / 11
Evidence / response semantics:      14 / 14
Final label-risk consistency:        4 / 4
X/Twitter hardening:                 5 / 5

Golden expected direction:          11 / 12
Golden internal coherence:          12 / 12

Offline E2E:
8 / 8
0 critical failures
0 warnings

Full E2E:
8 / 8
0 critical failures
0 warnings
```

The remaining golden directional case safely remained `Uncertain` rather than forcing a stronger conclusion without sufficient evidence.

---

### SLM-Assisted Latency

```text
Requests: 15

Average: 2.227 s
Minimum: 1.902 s
Maximum: 2.587 s
```

This demonstrates that local Qwen3.5 evidence verification is practical for an interactive application and academic demonstration.

---

## Final DistilBERT 512 Model

The large final DistilBERT 512 model weights are hosted separately on Hugging Face rather than committed to this GitHub repository.

### Hugging Face Model

**Repository:**

https://huggingface.co/prabhatraja445/fake-news-misinformation-distilbert-512

The application expects the model under:

```text
models/transformer/distilbert_isot_full_512/
```

Download the Hugging Face repository and place the inference-ready model files there.

Expected structure:

```text
models/
└── transformer/
    └── distilbert_isot_full_512/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── training_args.bin
```

The Hugging Face repository contains the final inference-ready model. Training checkpoints and optimizer state are intentionally excluded because they are not required for normal inference.

The classifier service prefers:

```text
models/transformer/distilbert_isot_full_512/
```

and can fall back to:

```text
models/transformer/distilbert_isot/
```

---

## Model Assets

### Traditional ML Baselines

The repository includes the baseline model artifacts:

```text
models/baseline/
├── tfidf_vectorizer.joblib
├── logistic_regression_model.joblib
└── linear_svm_model.joblib
```

They can also be regenerated using:

```powershell
python scripts\train_baseline_model.py
```

---

### Embedding Model

Evidence retrieval uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

For a fully local runtime, the preferred location is:

```text
models/embedding/all-MiniLM-L6-v2/
```

The public embedding model itself is not redistributed through this repository.

During development, the retrieval service can use the public model identifier if the local directory is unavailable.

For a completely offline demonstration, download the embedding model in advance and keep it under the expected local path.

---

### Qwen3.5 9B

Qwen3.5 is served separately through **LM Studio** rather than stored inside this repository.

Final model identifier:

```text
qwen/qwen3.5-9b
```

Recommended LM Studio settings:

```text
Temperature: 0
Enable Thinking: OFF
```

Typical local API endpoint:

```text
http://127.0.0.1:1234/v1
```

Qwen3.5 is used only to evaluate the stance of retrieved evidence:

```text
SUPPORTS
REFUTES
NOT_ENOUGH_INFO
```

It is not used as an independent source of truth.

---

## Evidence Indexes

### FEVER

```text
20,000 records

SUPPORTS:         12,105
REFUTES:           4,621
NOT_ENOUGH_INFO:   3,274
```

### LIAR2 Enhanced

```text
20,665 train / validation records
Test split excluded

SUPPORTS:          5,412
REFUTES:           8,673
NOT_ENOUGH_INFO:   6,580
```

### Evidence Thresholds

```text
Minimum usable relevance:       0.50
Strong evidence relevance:      0.65
Minimum aligned evidence count: 2
```

The system also distinguishes simple threshold strength from independent corroboration and source diversity.

---

## Repository Structure

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
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   └── README.md
├── models/
│   ├── baseline/
│   ├── transformer/
│   │   ├── distilbert_isot/
│   │   └── distilbert_isot_full_512/
│   └── README.md
├── reports/
├── scripts/
│   └── README.md
├── .env
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

The repository includes the project source code, baseline model artifacts, documented data assets, configuration, validation scripts, and project documentation.

The large final DistilBERT model weights are hosted on Hugging Face.

The public `all-MiniLM-L6-v2` embedding model is obtained separately when a fully local runtime is required.

---

## Setup

### 1. Create the Python Environment

Python **3.10** is recommended.

Using a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The original project was developed in a Conda environment named:

```text
fake-news-agent
```

---

### 2. Download the Final DistilBERT Model

Download:

https://huggingface.co/prabhatraja445/fake-news-misinformation-distilbert-512

Place the model files under:

```text
models/transformer/distilbert_isot_full_512/
```

Expected files:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
training_args.bin
```

---

### 3. Configure Environment Variables

The repository contains environment configuration examples for local execution.

Review:

```text
.env
.env.example
```

The public `.env` contains only placeholders/default configuration and must not contain real credentials.

Add your own credentials locally only when required by the selected inference mode.

Typical optional credentials include:

```text
TAVILY_API_KEY
EXA_API_KEY
X_BEARER_TOKEN
```

**Never commit real API keys, bearer tokens, passwords, or other credentials.**

For fully local modes, Tavily and Exa credentials are not required.

---

### 4. Start the Local SLM

Open LM Studio and load:

```text
qwen/qwen3.5-9b
```

Recommended settings:

```text
Temperature: 0
Enable Thinking: OFF
```

Start the LM Studio local server.

Typical endpoint:

```text
http://127.0.0.1:1234/v1
```

Ensure `SLM_BASE_URL` points to the active LM Studio endpoint.

---

### 5. Start FastAPI

From the project root:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health endpoint:

```text
http://127.0.0.1:8000/api/v1/health
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

### 6. Start Streamlit

Open another terminal and run:

```powershell
streamlit run dashboard\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

For the strongest final demonstration, select:

```text
SLM-assisted Hybrid Verification
```

---

## Main API Endpoints

```text
GET  /api/v1/health
POST /api/v1/predict
POST /api/v1/analyze
POST /api/v1/analyze-file
```

---

## Validation

Start FastAPI before running the end-to-end suites.

### Full E2E

Includes external/hybrid integrations:

```powershell
python scripts\run_e2e_validation_suite.py
```

### Offline E2E

Skips external services:

```powershell
python scripts\run_e2e_validation_suite.py --skip-external
```

### Golden Regression

```powershell
python scripts\run_golden_claims_regression.py
```

### SLM Latency Benchmark

```powershell
python scripts\benchmark_slm_latency.py
```

Additional validation and hardening scripts are documented in:

```text
scripts/README.md
```

---

## Example Claims

### Supported Factual Claim

```text
Chandrayaan-3 successfully soft-landed on the Moon in August 2023.
```

### Materially False Claim

```text
Chandrayaan-3 landed on Mars in August 2023.
```

### Multi-Claim Example

```text
The Eiffel Tower is located in Paris.

The Eiffel Tower was constructed between 1887 and 1889.

A viral video claims Iran tested a nuclear bomb last week.
```

For the third statement, the source wording is retained for context while the substantive proposition:

```text
Iran tested a nuclear bomb last week.
```

is verified independently.

---

## Claim Representation

The system distinguishes between what a source says and what factual proposition should actually be checked.

Example:

```text
source_text:
A viral video claims Iran tested a nuclear bomb last week.

verification_text:
Iran tested a nuclear bomb last week.
```

Relevant representation fields include:

```text
source_text
verification_text
verification_target
assertion_status
representation_fallback
```

This prevents the system from treating the existence of a claim as evidence that the underlying proposition is true.

---

## Important Limitations

- This is a misinformation-risk assessment system, not an absolute truth engine.
- ISOT performance is article-level and may reflect source, topic, vocabulary, or writing-style patterns.
- Short factual claims can behave differently from full news articles.
- Static FEVER and LIAR2 indexes cannot cover every real-world or recent event.
- Hybrid evidence depends on external search availability and can change over time.
- Qwen3.5 can still be affected by incomplete, conflicting, or ambiguous evidence.
- YouTube verification depends on transcript availability.
- X/Twitter verification is mainly text-based and does not fully verify attached media, replies, quote-post context, or discussion threads.
- Image support is OCR/text-oriented rather than deepfake or visual-forensics detection.
- High classifier confidence does not establish factual truth.
- Human review remains important for high-impact decisions.

---

## Detailed Documentation

- `README_RUNBOOK.md` — detailed runtime and demonstration guide
- `PROJECT_FLOW.md` — implementation and code flow
- `FINAL_VALIDATION_SUMMARY.md` — final measured validation results
- `PHASE2_PLAN.md` — completed Phase 2 implementation record
- `PROJECT_CLEANUP_SUMMARY.md` — final packaging and cleanup record
- `scripts/README.md` — retained scripts and validation commands
- `data/README.md` — data and evidence-index information
- `models/README.md` — local model assets and expected paths

---

## Author

**Peri Prabhat Raja**  
M.Tech Data Science & Artificial Intelligence  
PES University  
---

## Academic Use

This repository contains the implementation of an academic final project.

Dataset, model, API, library, and external-service ownership remains with their respective authors and providers.

The project report and accompanying documentation contain the detailed academic references and citations.

The system is intended for research, education, experimentation, and misinformation-risk assessment. It should not be treated as an absolute authority on whether a statement, article, or event is true or false.