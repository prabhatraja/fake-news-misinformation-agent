# Local Model Assets

The final runtime is designed to use local model artifacts.

Large trained-model directories may be excluded from a source-code-only submission because of file size.

## Expected structure

```text
models/
├── baseline/
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_regression_model.joblib
│   └── linear_svm_model.joblib
├── transformer/
│   ├── distilbert_isot_full_512/
│   └── distilbert_isot/
└── embedding/
    └── all-MiniLM-L6-v2/
```

## Baseline artifacts

The classifier service expects:

```text
models/baseline/tfidf_vectorizer.joblib
models/baseline/logistic_regression_model.joblib
models/baseline/linear_svm_model.joblib
```

These can be regenerated with:

```powershell
python scripts\train_baseline_model.py
```

## DistilBERT

The classifier service prefers:

```text
models/transformer/distilbert_isot_full_512/
```

and falls back to:

```text
models/transformer/distilbert_isot/
```

Final reported 512-token result:

```text
Accuracy: 99.99%
Precision: 99.97%
Recall: 100.00%
F1-score: 99.99%
1 error / 7,820 test records
```

The retained training script writes to:

```text
models/transformer/distilbert_isot/
```
## Pretrained DistilBERT 512 Model

The final trained DistilBERT 512 classifier is hosted separately on Hugging Face because the model weights are too large for the main GitHub repository.

**Hugging Face model:**

https://huggingface.co/prabhatraja445/fake-news-misinformation-distilbert-512

The project expects the model under:

```text
models/transformer/distilbert_isot_full_512/

## Embedding model

Preferred fully local path:

```text
models/embedding/all-MiniLM-L6-v2/
```

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The retrieval service can fall back to the model name during development if the local directory is unavailable. For an offline demonstration, keep the embedding model locally.

## Qwen3.5

Qwen3.5 is served separately through LM Studio rather than stored under this project directory.

Final model identifier:

```text
qwen/qwen3.5-9b
```

LM Studio setting:

```text
Enable Thinking = OFF
Temperature = 0
```

Qwen3.5 is used only to verify the stance of retrieved evidence as:

```text
SUPPORTS
REFUTES
NOT_ENOUGH_INFO
```

It is not used as an independent source of truth.
