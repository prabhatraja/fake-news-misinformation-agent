# Scripts Inventory

This folder contains the retained scripts required for final validation, data/index preparation, model training and evaluation.

One-off patch scripts and granular development-only hardening scripts are not required in the active final submission package.

---

## 1. Runtime Validation

### `run_e2e_validation_suite.py`

Main API-level end-to-end suite.

Requires FastAPI to be running.

Run:

```powershell
python scripts\run_e2e_validation_suite.py
```

Offline/local-only run:

```powershell
python scripts\run_e2e_validation_suite.py --skip-external
```

Optional arguments:

```text
--base-url
--skip-external
--timeout
```

Coverage includes:

- `/api/v1/health`
- `/api/v1/predict`
- all seven inference modes
- plain-text `/api/v1/analyze`
- URL analysis
- YouTube best-effort transcript analysis
- X/Twitter best-effort text extraction and identity handling
- recirculated-media temporal handling
- mixed uploaded-file claim/context isolation

Final recorded result:

```text
Full run:    8 / 8 passed, 0 critical failures, 0 warnings
Offline run: 8 / 8 passed, 0 critical failures, 0 warnings
```

Reports are written to:

```text
reports/e2e/
```

---

### `run_golden_claims_regression.py`

Runs a small factual regression set against:

```text
POST /api/v1/analyze
```

Run:

```powershell
python scripts\run_golden_claims_regression.py
```

Optional:

```powershell
python scripts\run_golden_claims_regression.py --mode slm_assisted_hybrid
```

Final recorded result:

```text
11 / 12 expected directional outcomes
12 / 12 internal coherence
```

Reports are written to:

```text
reports/quality/
```

Live/hybrid evidence can change over time, so directional differences should be reviewed rather than treated as training accuracy.

---

## 2. Latency Benchmark

### `benchmark_slm_latency.py`

Benchmarks the final SLM-assisted hybrid `/predict` path.

Run with FastAPI and LM Studio active:

```powershell
python scripts\benchmark_slm_latency.py
```

Recorded final benchmark:

```text
15 total requests
Average: 2.227 s
Minimum: 1.902 s
Maximum: 2.587 s
```

Output:

```text
reports/metrics/slm_latency_benchmark_outputs.json
```

---

## 3. Dataset Preparation

### `prepare_isot_data.py`

Prepares the ISOT article dataset used for baseline and DistilBERT training.

Final cleaned dataset:

```text
39,100 records
31,280 train
7,820 test
```

---

## 4. Baseline Training

### `train_baseline_model.py`

Trains and persists:

- TF-IDF vectorizer
- Logistic Regression
- Linear SVM

Expected runtime artifacts:

```text
models/baseline/tfidf_vectorizer.joblib
models/baseline/logistic_regression_model.joblib
models/baseline/linear_svm_model.joblib
```

---

## 5. DistilBERT Training

### `train_distilbert_model.py`

Fine-tunes DistilBERT for article-level ISOT classification.

The current script writes its trained artifact under:

```text
models/transformer/distilbert_isot/
```

The classifier service also prefers the final artifact if present at:

```text
models/transformer/distilbert_isot_full_512/
```

Final reported 512-token metrics:

```text
Accuracy: 99.99%
Precision: 99.97%
Recall: 100.00%
F1-score: 99.99%
```

---

## 6. Evidence Index Building

### `build_fever_evidence_index.py`

Builds:

```text
data/embeddings/chroma_fever/
```

Final index summary:

```text
20,000 records
SUPPORTS: 12,105
REFUTES: 4,621
NOT_ENOUGH_INFO: 3,274
```

### `build_liar2_evidence_index.py`

Builds:

```text
data/embeddings/chroma_liar2_justifications/
```

Final index summary:

```text
20,665 train/validation records
Test split excluded
SUPPORTS: 5,412
REFUTES: 8,673
NOT_ENOUGH_INFO: 6,580
```

---

## 7. LIAR2 Evaluation

### `evaluate_liar2_claims.py`

Runs the offline LIAR2 claim-level safety/evidence-coverage evaluation.

Final reported result on 200 samples:

```text
Safe behavior: 99.5%
Usable evidence coverage: 85.0%
Uncertain rate: 90.5%
Correct non-Uncertain binary cases: 17 / 17
```

The 17/17 result is not general claim-level accuracy.

---

## 8. Final Script Policy

The active final script folder should remain intentionally small.

Keep scripts that:

- build datasets/indexes
- train retained models
- evaluate the final system
- reproduce final metrics
- validate the active API

Do not keep:

- one-off patch scripts
- obsolete smoke-test variants
- duplicate temporary scripts
- archived development copies

The final source submission should also exclude `archive/`, `__pycache__/`, `.pyc` files and the real `.env`.
