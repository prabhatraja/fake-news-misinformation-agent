# Project Cleanup Summary

## Purpose

This file defines the final submission-cleanup state for the project.

The active submission should contain only the code, configuration, documentation, retained preparation/evaluation scripts, and useful final validation outputs required to understand or reproduce the project.

---

## 1. Keep in the Final Submission

Recommended active structure:

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
│   ├── e2e/
│   ├── figures/
│   ├── metrics/
│   └── quality/
├── scripts/
│   ├── README.md
│   ├── benchmark_slm_latency.py
│   ├── build_fever_evidence_index.py
│   ├── build_liar2_evidence_index.py
│   ├── evaluate_liar2_claims.py
│   ├── prepare_isot_data.py
│   ├── run_e2e_validation_suite.py
│   ├── run_golden_claims_regression.py
│   ├── train_baseline_model.py
│   └── train_distilbert_model.py
├── .env.example
├── environment_check.py
├── FINAL_VALIDATION_SUMMARY.md
├── PHASE2_PLAN.md
├── PROJECT_CLEANUP_SUMMARY.md
├── PROJECT_FLOW.md
└── README_RUNBOOK.md
```

If the final package includes the large trained models, raw datasets and Chroma indexes, keep them in the expected `models/` and `data/` locations.

If the submission is source-code-only because the assets are too large, keep the `data/README.md` and `models/README.md` files so the expected paths are explicit.

---

## 2. Remove Before Submission

The final ZIP should **not** contain:

```text
.env
.vscode/
archive/
__pycache__/
*.pyc
*.pyo
*.bak
temporary patch scripts
one-off hardening scripts no longer used
reports/megabatch*.txt
local editor metadata
API keys or access tokens
```

The real `.env` must never be submitted.

Only `.env.example` should remain.

---

## 3. Secret Safety

Before creating the final ZIP, verify:

```text
TAVILY_API_KEY=
EXA_API_KEY=
X_BEARER_TOKEN=
```

are empty in `.env.example`.

Do not place real credentials in any README, screenshot, source file or validation output.

If a real `.env` was ever shared outside the local machine, rotate those credentials.

---

## 4. Active Validation Surface

The final project intentionally keeps a small validation surface.

### Main E2E suite

```powershell
python scripts\run_e2e_validation_suite.py
```

Offline/local-only:

```powershell
python scripts\run_e2e_validation_suite.py --skip-external
```

Final recorded result for both full and offline runs:

```text
8 / 8 passed
0 critical failures
0 warnings
```

### Golden regression

```powershell
python scripts\run_golden_claims_regression.py
```

Final recorded result:

```text
11 / 12 expected directional outcomes
12 / 12 internal coherence
```

### Optional latency benchmark

```powershell
python scripts\benchmark_slm_latency.py
```

Recorded result:

```text
15 requests
Average: 2.227 s
Minimum: 1.902 s
Maximum: 2.587 s
```

---

## 5. Runtime Assumptions

- Conda environment: `fake-news-agent`
- FastAPI base URL: `http://127.0.0.1:8000`
- Health endpoint: `http://127.0.0.1:8000/api/v1/health`
- Swagger: `http://127.0.0.1:8000/docs`
- Streamlit: `http://localhost:8501`
- LM Studio: `http://127.0.0.1:1234/v1`
- SLM model: `qwen/qwen3.5-9b`
- LM Studio setting: `Enable Thinking = OFF`

---

## 6. Final Documentation State

The final documentation should consistently describe:

- Qwen3.5 9B as the final SLM verifier
- `/api/v1/health`, `/api/v1/predict`, `/api/v1/analyze`, `/api/v1/analyze-file`
- seven retained inference modes
- multi-format input support
- `source_text` / `verification_text` claim representation
- temporal and recirculated-media context
- independent-source corroboration
- final validation results
- conservative `Uncertain` behavior
- the SLM as evidence verifier, not source of truth

Historical model experiments and old one-off scripts should not be described as current runtime components.

---

## 7. Final Packaging Check

Before submission:

1. Start from the clean project directory.
2. Remove the excluded files/directories listed above.
3. Confirm `.env.example` contains no secrets.
4. Confirm the active Markdown files match the final code.
5. Confirm `python environment_check.py` succeeds in the final environment.
6. Start LM Studio with Qwen3.5 if demonstrating SLM modes.
7. Start FastAPI and check `/api/v1/health`.
8. Run the final E2E suite.
9. Open Streamlit and verify the final dashboard.
10. Create the final ZIP only after these checks.

The final code package should be clean enough that an examiner can understand the active system without encountering stale patches, old model instructions or secret credentials.
