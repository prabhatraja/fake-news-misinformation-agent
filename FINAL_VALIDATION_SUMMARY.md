# Final Validation Summary

## Project

**Explainable Real-Time Fake News and Misinformation Detection Agent with Evidence Retrieval Using Transformers and RAG**

This file records the final validated state of the project. The system is an evidence-grounded misinformation risk-assessment agent, not an absolute truth engine.

---

## 1. Final Runtime Configuration

- Traditional baselines: TF-IDF + Logistic Regression, TF-IDF + Linear SVM
- Transformer classifier: Fine-tuned DistilBERT 512
- Local evidence: FEVER + LIAR2 Enhanced Chroma indexes
- Hybrid current evidence: Tavily + Exa
- Final local SLM verifier: Qwen3.5 9B via LM Studio
- Default final demo mode: `slm_assisted_hybrid`
- Evidence policy: `config/evidence_policy.yaml`
- SLM role: retrieved-evidence stance verification only

The final SLM is:

```text
qwen/qwen3.5-9b
```

LM Studio must run with:

```text
Enable Thinking = OFF
Temperature = 0
```

---

## 2. Final Model Results - ISOT

Dataset:

```text
Cleaned records: 39,100
Train: 31,280
Test: 7,820
```

Model comparison:

| Model | Accuracy | F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 98.63% | 98.50% |
| TF-IDF + Linear SVM | 99.37% | 99.31% |
| DistilBERT 256 | 99.95% | 99.94% |
| DistilBERT 512 | 99.99% | 99.99% |

Final DistilBERT 512:

```text
Accuracy: 99.99%
Precision: 99.97%
Recall: 100.00%
F1-score: 99.99%
Errors: 1 / 7,820
Confusion matrix: [[4238, 1], [0, 3581]]
```

These are **article-level ISOT metrics** and are not interpreted as universal factual-verification accuracy.

---

## 3. Evidence Indexes

### FEVER

```text
Records: 20,000
SUPPORTS: 12,105
REFUTES: 4,621
NOT_ENOUGH_INFO: 3,274
Embedding: sentence-transformers/all-MiniLM-L6-v2
Vector store: Chroma
```

### LIAR2 Enhanced

```text
Records: 20,665
Train indexed: 18,368
Validation indexed: 2,297
Test split indexed: No
SUPPORTS: 5,412
REFUTES: 8,673
NOT_ENOUGH_INFO: 6,580
Embedding: sentence-transformers/all-MiniLM-L6-v2
Vector store: Chroma
```

The LIAR2 test split is intentionally excluded from the evidence index to reduce evaluation leakage.

---

## 4. LIAR2 Offline Evaluation

Final recorded result on 200 test samples:

```text
Processed: 200
Failed: 0
Safe behavior: 199 / 200 = 99.5%
Usable evidence coverage: 170 / 200 = 85.0%
Uncertain: 181 / 200 = 90.5%
Binary non-Uncertain decisions: 17
Binary correct non-Uncertain decisions: 17
```

The 17/17 result is **not** treated as general claim-level accuracy.

The important interpretation is:

- the system remains conservative when evidence is weak
- uncertainty is a valid safety outcome
- usable evidence coverage and safe behavior are more meaningful for this offline claim-level test

---

## 5. Final Hardening Checks

Recorded final hardening results:

| Quality area | Result |
|---|---:|
| Deterministic claim representation | 25 / 25 passed |
| Live Qwen claim representation | 11 / 11 passed |
| Response / evidence semantics | 14 / 14 passed |
| Final label-risk consistency | 4 / 4 passed |
| X/Twitter hardening | 5 / 5 passed |

These checks validate the final semantic contract and quality controls, including:

- direct claims remain unchanged
- embedded propositions can be isolated safely
- denied/questioned content is not promoted into positive facts
- retrieval uses `verification_text`
- temporal analysis preserves `source_text`
- independent-source corroboration is distinguished from repeated pages
- the SLM cannot override strong opposite evidence simply by disagreeing

The granular development test scripts are not required in the active final submission package after final consolidated validation.

---

## 6. Golden Regression

Final recorded golden regression:

```text
Total cases: 12
Expected directional outcomes: 11 / 12
Internal coherence: 12 / 12
Directional rate: 91.67%
Coherence rate: 100%
```

The single directional miss was:

```text
Sydney is the capital of Australia.
```

The system returned:

```text
Uncertain / Medium Risk
```

rather than forcing `Likely Fake` because the retrieved evidence was not strong enough under the final semantic policy.

This case remained internally coherent and was accepted as safer behavior instead of weakening thresholds simply to obtain 12/12 directional agreement.

---

## 7. Consolidated E2E Validation

### Full external/hybrid run

Final recorded run:

```text
Total cases: 8
Passed: 8
Critical failures: 0
Warnings: 0
```

Coverage:

1. backend health
2. all seven `/predict` inference modes
3. plain-text `/analyze`
4. Wikipedia URL analysis
5. YouTube transcript best-effort analysis
6. X/Twitter text extraction and identity handling
7. recirculated-media temporal handling
8. mixed uploaded-file claim/context isolation

### Offline run

Final recorded result:

```text
8 / 8 passed
0 critical failures
0 warnings
```

The offline mode allows reproducible validation without relying on changing external URL/social results.

---

## 8. Seven Inference Modes

All seven modes remain functional:

```text
traditional_lr
traditional_svm
distilbert
distilbert_local_rag
distilbert_hybrid
slm_assisted_local
slm_assisted_hybrid
```

### Project-evolution example

For:

```text
The Eiffel Tower is located in Paris.
```

the final E2E validation produced:

| Mode | Final label | Risk |
|---|---|---|
| `traditional_lr` | Likely Fake | Medium Risk |
| `traditional_svm` | Likely Fake | Medium Risk |
| `distilbert` | Likely Fake | Medium Risk |
| `distilbert_local_rag` | Uncertain | Medium Risk |
| `distilbert_hybrid` | Uncertain | Medium Risk |
| `slm_assisted_local` | Likely Real | Low Risk |
| `slm_assisted_hybrid` | Likely Real | Low Risk |

This demonstrates the reason the project evolved beyond classifier-only prediction.

---

## 9. Multi-Format Validation

The final `/analyze` and `/analyze-file` flows validate more than single typed claims.

Supported/validated paths include:

- plain text
- web URLs
- YouTube transcript analysis
- X/Twitter extracted text
- uploaded text/PDF/image path
- temporal/recirculated-media context
- mixed multi-claim content

### Mixed-file validation

Final recorded mixed-file result:

```text
Overall: Misleading / Medium Risk
```

Claims:

1. `The Eiffel Tower is located in Paris.`
   - Likely Real
   - Low Risk
   - SLM: SUPPORTS

2. `The Eiffel Tower was constructed between 1887 and 1889.`
   - Likely Real
   - Very Low Risk
   - SLM: SUPPORTS

3. Source:
   `A viral video claims Iran tested a nuclear bomb last week.`

   Verification proposition:
   `Iran tested a nuclear bomb last week.`

   - Likely Fake
   - Medium Risk
   - SLM: REFUTES
   - recirculated-media focus retained

This validates both claim isolation and the `source_text` / `verification_text` contract.

---

## 10. X/Twitter Validation

The final X/Twitter hardening validates:

- post URL detection
- public oEmbed extraction where available
- requested/resolved post-ID matching
- extraction-quality reporting
- explicit limitations for media/thread context

The final E2E X/Twitter case passed with:

```text
extraction: success
identity_verified: true
extraction_quality: reliable
```

The system does **not** claim to verify attached images, video, quoted-post context, replies or the full thread.

---

## 11. Evidence Policy

Final thresholds:

```text
Minimum usable relevance: 0.50
Strong evidence relevance: 0.65
Minimum aligned evidence count: 2
```

Important final controls:

- semantic SUPPORTS must establish the same proposition
- semantic REFUTES must materially contradict the proposition
- relevant but non-decisive evidence remains `NOT_ENOUGH_INFO`
- repeated pages from one source/domain count once for independent corroboration
- Top Evidence prefers source-diverse usable evidence
- weak, missing, unrelated or conflicting evidence can remain `Uncertain`

---

## 12. SLM-Assisted Latency

Recorded final benchmark:

```text
Total requests: 15
Average: 2.227 seconds
Minimum: 1.902 seconds
Maximum: 2.587 seconds
Mode: slm_assisted_hybrid
Model: qwen/qwen3.5-9b
```

This shows that local Qwen3.5 evidence verification is practical for the interactive final demonstration.

---

## 13. Final System Principle

The final system does not treat:

```text
classifier confidence = truth
```

Instead, the final decision combines:

```text
classifier signal
+ claim representation
+ temporal context
+ local/hybrid evidence
+ evidence quality
+ semantic evidence direction
+ source trust
+ independent corroboration
+ Qwen3.5 evidence verification
+ uncertainty/risk policy
```

The SLM is an **evidence interpreter**, not a source of truth.

The project therefore demonstrates an end-to-end progression from fake-news classification to evidence-grounded, explainable, multi-format misinformation risk assessment.
