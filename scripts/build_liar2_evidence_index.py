import json
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIAR2_DIR = PROJECT_ROOT / "data" / "raw" / "liar2"
EVIDENCE_DIR = PROJECT_ROOT / "data" / "embeddings" / "chroma_liar2_justifications"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
SUMMARY_FILE = METRICS_DIR / "liar2_evidence_index_summary.json"

COLLECTION_NAME = "liar2_justification_evidence"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 128

INPUT_FILES = {
    "train": LIAR2_DIR / "liar2_train.csv",
    "validation": LIAR2_DIR / "liar2_validation.csv",
}

NUMERIC_LABEL_TO_NAME = {
    "0": "pants-fire",
    "0.0": "pants-fire",
    "1": "false",
    "1.0": "false",
    "2": "barely-true",
    "2.0": "barely-true",
    "3": "half-true",
    "3.0": "half-true",
    "4": "mostly-true",
    "4.0": "mostly-true",
    "5": "true",
    "5.0": "true",
}

LABEL_TO_EVIDENCE_LABEL = {
    "true": "SUPPORTS",
    "mostly-true": "SUPPORTS",
    "half-true": "NOT_ENOUGH_INFO",
    "barely-true": "NOT_ENOUGH_INFO",
    "false": "REFUTES",
    "pants-fire": "REFUTES",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value)
    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def normalize_label(label: Any) -> str:
    raw_label = clean_text(label).lower()

    if raw_label in NUMERIC_LABEL_TO_NAME:
        return NUMERIC_LABEL_TO_NAME[raw_label]

    return raw_label

def map_to_evidence_label(liar2_label: str) -> str:
    return LABEL_TO_EVIDENCE_LABEL.get(liar2_label, "UNKNOWN")


def create_evidence_text(row: Dict[str, Any]) -> str:
    statement = clean_text(row.get("statement", ""))
    justification = clean_text(row.get("justification", ""))
    liar2_label = normalize_label(row.get("label", ""))
    speaker = clean_text(row.get("speaker", ""))
    subject = clean_text(row.get("subject", ""))
    date = clean_text(row.get("date", ""))
    context = clean_text(row.get("context", ""))

    parts = [
        f"Claim: {statement}.",
        f"Fact-check rating: {liar2_label}.",
    ]

    if speaker:
        parts.append(f"Speaker or source: {speaker}.")

    if subject:
        parts.append(f"Subject: {subject}.")

    if date:
        parts.append(f"Date: {date}.")

    if context:
        parts.append(f"Context: {context}.")

    if justification:
        parts.append(f"Justification: {justification}")

    return " ".join(parts)


def load_liar2_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    print("Loading LIAR2 train + validation files from local project folder...")

    for split_name, file_path in INPUT_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing LIAR2 file: {file_path}")

        df = pd.read_csv(file_path)
        print(f"{split_name}: {len(df)} rows")

        for row_index, row in df.iterrows():
            row_dict = row.to_dict()

            statement = clean_text(row_dict.get("statement", ""))
            justification = clean_text(row_dict.get("justification", ""))
            liar2_label = normalize_label(row_dict.get("label", ""))

            if not statement:
                continue

            if liar2_label not in LABEL_TO_EVIDENCE_LABEL:
                continue

            # The justification is the useful fact-check evidence.
            # Keep statement-only records out of the evidence index.
            if len(justification) < 30:
                continue

            evidence_label = map_to_evidence_label(liar2_label)
            evidence_text = create_evidence_text(row_dict)

            liar2_id = clean_text(row_dict.get("id", ""))
            record_id = f"{split_name}_{liar2_id}_{row_index}"

            records.append(
                {
                    "id": record_id,
                    "split": split_name,
                    "liar2_id": liar2_id,
                    "statement": statement,
                    "justification": justification,
                    "liar2_label": liar2_label,
                    "evidence_label": evidence_label,
                    "speaker": clean_text(row_dict.get("speaker", "")),
                    "subject": clean_text(row_dict.get("subject", "")),
                    "date": clean_text(row_dict.get("date", "")),
                    "context": clean_text(row_dict.get("context", "")),
                    "evidence_text": evidence_text,
                }
            )

    print(f"Prepared LIAR2 evidence records: {len(records)}")
    return records


def build_chroma_index(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("Loading sentence-transformer model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Creating Chroma persistent client...")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(EVIDENCE_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing LIAR2 evidence collection.")
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    print("Building LIAR2 Chroma evidence index...")

    total_records = len(records)

    for start in range(0, total_records, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_records)
        batch = records[start:end]

        ids = [item["id"] for item in batch]
        documents = [item["evidence_text"] for item in batch]

        metadatas = [
            {
                "source_type": "local_liar2",
                "source_name": "LIAR2 Enhanced",
                "split": item["split"],
                "liar2_id": item["liar2_id"],
                "statement": item["statement"],
                "liar2_label": item["liar2_label"],
                "label": item["evidence_label"],
                "speaker": item["speaker"],
                "subject": item["subject"],
                "date": item["date"],
                "context": item["context"],
            }
            for item in batch
        ]

        embeddings = embedding_model.encode(
            documents,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(f"Indexed records {start + 1} to {end} of {total_records}")

    summary: Dict[str, Any] = {
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "records_indexed": total_records,
        "chroma_path": str(EVIDENCE_DIR),
        "source_files": {key: str(value) for key, value in INPUT_FILES.items()},
        "splits_indexed": {},
        "liar2_labels": {},
        "evidence_labels": {},
        "note": (
            "Only LIAR2 train and validation splits are indexed. "
            "The test split is intentionally excluded to avoid evaluation leakage."
        ),
    }

    for record in records:
        split = record["split"]
        liar2_label = record["liar2_label"]
        evidence_label = record["evidence_label"]

        summary["splits_indexed"][split] = summary["splits_indexed"].get(split, 0) + 1
        summary["liar2_labels"][liar2_label] = summary["liar2_labels"].get(liar2_label, 0) + 1
        summary["evidence_labels"][evidence_label] = summary["evidence_labels"].get(evidence_label, 0) + 1

    return summary


def save_summary(summary: Dict[str, Any]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Saved summary: {SUMMARY_FILE}")
    print(json.dumps(summary, indent=4))


def main() -> None:
    records = load_liar2_records()

    if not records:
        raise RuntimeError("No LIAR2 evidence records were prepared. Check input CSV files.")

    summary = build_chroma_index(records)
    save_summary(summary)

    print("\nLIAR2 justification evidence index build completed successfully.")


if __name__ == "__main__":
    main()