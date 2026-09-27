import json
from pathlib import Path
from typing import Any

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
EVIDENCE_DIR = DATA_DIR / "embeddings" / "chroma_fever"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

EVIDENCE_SUMMARY_FILE = METRICS_DIR / "fever_evidence_index_summary.json"

COLLECTION_NAME = "fever_evidence"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_RECORDS = 20000
BATCH_SIZE = 128


def normalize_label(label: str) -> str:
    label = str(label).strip().upper()

    if label == "SUPPORTS":
        return "SUPPORTS"

    if label == "REFUTES":
        return "REFUTES"

    if label in {"NOT ENOUGH INFO", "NOTENOUGHINFO"}:
        return "NOT_ENOUGH_INFO"

    return label


def clean_text(text: Any) -> str:
    if text is None:
        return ""

    return str(text).replace("\n", " ").replace("\t", " ").strip()


def create_evidence_text(record: dict) -> str:
    claim = clean_text(record.get("claim", ""))
    label = normalize_label(record.get("label", ""))
    wiki_url = clean_text(record.get("evidence_wiki_url", ""))

    if wiki_url:
        return f"Claim: {claim}. Evidence source: {wiki_url}. Label: {label}."

    return f"Claim: {claim}. Label: {label}."


def load_fever_records() -> list[dict]:
    print("Loading FEVER dataset from Hugging Face...")

    dataset = load_dataset(
        "fever/fever",
        "v1.0",
        split="train",
        trust_remote_code=True,
    )

    print(f"Original FEVER train records: {len(dataset)}")

    records = []

    for row in dataset:
        claim = clean_text(row.get("claim", ""))
        label = normalize_label(row.get("label", ""))
        wiki_url = clean_text(row.get("evidence_wiki_url", ""))

        if not claim:
            continue

        if label not in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}:
            continue

        evidence_text = create_evidence_text(row)

        records.append(
            {
                "id": str(row.get("id", len(records))),
                "claim": claim,
                "evidence_text": evidence_text,
                "label": label,
                "evidence_wiki_url": wiki_url,
                "evidence_sentence_id": str(row.get("evidence_sentence_id", "")),
            }
        )

        if len(records) >= MAX_RECORDS:
            break

    print(f"Prepared records for indexing: {len(records)}")

    return records


def build_chroma_index(records: list[dict]) -> dict:
    print("Loading sentence-transformer model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Creating Chroma persistent client...")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(EVIDENCE_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    print("Building Chroma index...")

    total_records = len(records)

    for start in range(0, total_records, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_records)
        batch = records[start:end]

        ids = [item["id"] + f"_{start + index}" for index, item in enumerate(batch)]
        documents = [item["evidence_text"] for item in batch]
        metadatas = [
            {
                "claim": item["claim"],
                "label": item["label"],
                "evidence_wiki_url": item["evidence_wiki_url"],
                "evidence_sentence_id": item["evidence_sentence_id"],
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

    summary = {
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "records_indexed": total_records,
        "chroma_path": str(EVIDENCE_DIR),
        "labels": {},
    }

    for record in records:
        label = record["label"]
        summary["labels"][label] = summary["labels"].get(label, 0) + 1

    return summary


def save_summary(summary: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVIDENCE_SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Saved summary: {EVIDENCE_SUMMARY_FILE}")
    print(json.dumps(summary, indent=4))


def main() -> None:
    records = load_fever_records()
    summary = build_chroma_index(records)
    save_summary(summary)

    print("\nFEVER evidence index build completed successfully.")


if __name__ == "__main__":
    main()