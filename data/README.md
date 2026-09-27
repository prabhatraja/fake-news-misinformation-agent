# Data Assets

The final runtime uses local dataset and evidence-index assets under this directory.

Large raw datasets and persistent Chroma indexes may be excluded from a source-code-only submission because of file size.

## Expected structure

```text
data/
├── raw/
├── processed/
└── embeddings/
    ├── chroma_fever/
    └── chroma_liar2_justifications/
```

## FEVER evidence index

Expected path:

```text
data/embeddings/chroma_fever/
```

Final recorded index:

```text
Collection: fever_evidence
Records: 20,000
SUPPORTS: 12,105
REFUTES: 4,621
NOT_ENOUGH_INFO: 3,274
Embedding: sentence-transformers/all-MiniLM-L6-v2
```

## LIAR2 Enhanced evidence index

Expected path:

```text
data/embeddings/chroma_liar2_justifications/
```

Final recorded index:

```text
Collection: liar2_justification_evidence
Records: 20,665
Train: 18,368
Validation: 2,297
Test split indexed: No
SUPPORTS: 5,412
REFUTES: 8,673
NOT_ENOUGH_INFO: 6,580
Embedding: sentence-transformers/all-MiniLM-L6-v2
```

The LIAR2 test split is intentionally excluded from the evidence index to reduce evaluation leakage.

## Rebuilding

Retained scripts:

```powershell
python scripts\build_fever_evidence_index.py
python scripts\build_liar2_evidence_index.py
```

Dataset preparation/training assets should be kept local for the reproducible offline project environment.

If this source package does not contain the large data/index files, this README documents the locations expected by the runtime.
