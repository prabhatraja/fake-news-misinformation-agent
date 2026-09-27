import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_API_URL = "http://127.0.0.1:8000/api/v1/predict"

LIAR2_DIR = PROJECT_ROOT / "data" / "raw" / "liar2"

DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "reports" / "metrics" / "liar2_claim_evaluation.json"
DEFAULT_CSV_OUTPUT = PROJECT_ROOT / "reports" / "metrics" / "liar2_claim_evaluation_details.csv"


NUMERIC_LABEL_MAP = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true",
}


FALSE_LABELS = {"false", "pants-fire"}
TRUE_LABELS = {"true", "mostly-true"}
PARTIAL_LABELS = {"barely-true", "half-true"}


REQUIRED_COLUMNS = {
    "label",
    "statement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the local misinformation agent on local LIAR2 enhanced claim data."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "valid", "test"],
        help="Local LIAR2 split to evaluate.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum number of LIAR2 claims to evaluate.",
    )

    parser.add_argument(
        "--api_url",
        type=str,
        default=DEFAULT_API_URL,
        help="FastAPI prediction endpoint.",
    )

    parser.add_argument(
        "--json_output",
        type=str,
        default=str(DEFAULT_JSON_OUTPUT),
        help="Path to save summary JSON output.",
    )

    parser.add_argument(
        "--csv_output",
        type=str,
        default=str(DEFAULT_CSV_OUTPUT),
        help="Path to save detailed CSV output.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="API timeout in seconds.",
    )

    return parser.parse_args()


def resolve_liar2_split_path(split: str) -> Path:
    normalized_split = split.lower().strip()

    if normalized_split in {"validation", "valid"}:
        filename = "liar2_validation.csv"
    elif normalized_split == "train":
        filename = "liar2_train.csv"
    elif normalized_split == "test":
        filename = "liar2_test.csv"
    else:
        raise ValueError(f"Unsupported split: {split}")

    file_path = LIAR2_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Local LIAR2 file not found: {file_path}\n"
            "This project is configured for offline/runtime-local execution.\n"
            "Place LIAR2 CSV files under data/raw/liar2/."
        )

    return file_path


def load_local_liar2_split(split: str) -> pd.DataFrame:
    file_path = resolve_liar2_split_path(split)

    print("Loading LIAR2 enhanced dataset from local project folder...")
    print(f"File: {file_path}")

    df = pd.read_csv(
        file_path,
        encoding="utf-8",
        on_bad_lines="skip",
    )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required LIAR2 columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["label", "statement"]).copy()
    df["statement"] = df["statement"].astype(str).str.strip()
    df = df[df["statement"].str.len() >= 5].copy()

    if df.empty:
        raise ValueError(f"LIAR2 split is empty after cleaning: {file_path}")

    return df


def normalize_liar2_label(label_value) -> str:
    if isinstance(label_value, str):
        value = label_value.strip()

        if value.isdigit():
            numeric_label = int(value)
            return NUMERIC_LABEL_MAP.get(numeric_label, f"unknown-{numeric_label}")

        return value.lower().replace("_", "-").strip()

    numeric_label = int(label_value)
    return NUMERIC_LABEL_MAP.get(numeric_label, f"unknown-{numeric_label}")


def map_liar_label_to_expected_group(label: str) -> str:
    normalized = label.strip().lower().replace("_", "-")

    if normalized in FALSE_LABELS:
        return "fake_or_misleading"

    if normalized in TRUE_LABELS:
        return "likely_real"

    if normalized in PARTIAL_LABELS:
        return "partial_or_ambiguous"

    return "unknown"


def map_agent_label_to_group(final_label: str) -> str:
    label = str(final_label).strip().lower()

    if label == "likely real":
        return "likely_real"

    if label in {"likely fake", "misleading"}:
        return "fake_or_misleading"

    if label == "uncertain":
        return "uncertain"

    return "unknown"


def is_expected_behavior(expected_group: str, predicted_group: str) -> bool:
    """
    This is not simple binary accuracy.

    LIAR2 has six truthfulness labels. Barely-true and half-true are ambiguous
    for a four-label misinformation agent. For partial labels, Uncertain or
    Misleading is treated as acceptable behavior.
    """

    if expected_group == "likely_real":
        return predicted_group == "likely_real"

    if expected_group == "fake_or_misleading":
        return predicted_group == "fake_or_misleading"

    if expected_group == "partial_or_ambiguous":
        return predicted_group in {"uncertain", "fake_or_misleading"}

    return False


def is_safe_behavior(expected_group: str, predicted_group: str) -> bool:
    """
    Safe behavior means the system avoids a clearly harmful overclaim.

    Uncertain is treated as safe because the system explicitly refuses to force
    a real/fake decision.
    """

    if predicted_group == "uncertain":
        return True

    if expected_group == "likely_real":
        return predicted_group == "likely_real"

    if expected_group == "fake_or_misleading":
        return predicted_group == "fake_or_misleading"

    if expected_group == "partial_or_ambiguous":
        return predicted_group in {"uncertain", "fake_or_misleading"}

    return False


def call_prediction_api(api_url: str, text: str, timeout: int) -> Dict:
    payload = {
        "text": text,
        "input_type": "claim",
    }

    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()

    return response.json()


def evaluate_liar2_claims(args: argparse.Namespace) -> Tuple[Dict, List[Dict]]:
    df = load_local_liar2_split(args.split)

    total_available = len(df)
    max_samples = min(args.max_samples, total_available)

    print(f"LIAR2 split          : {args.split}")
    print(f"Available records    : {total_available}")
    print(f"Evaluating records   : {max_samples}")
    print(f"API URL              : {args.api_url}")
    print()

    details: List[Dict] = []

    sample_df = df.head(max_samples).reset_index(drop=True)

    for index, row in sample_df.iterrows():
        statement = str(row["statement"]).strip()
        liar_label = normalize_liar2_label(row["label"])
        expected_group = map_liar_label_to_expected_group(liar_label)

        print(f"[{index + 1}/{max_samples}] {liar_label} | {statement[:90]}")

        try:
            api_result = call_prediction_api(
                api_url=args.api_url,
                text=statement,
                timeout=args.timeout,
            )

            final_label = api_result.get("predicted_label")
            predicted_group = map_agent_label_to_group(final_label)

            expected_behavior = is_expected_behavior(expected_group, predicted_group)
            safe_behavior = is_safe_behavior(expected_group, predicted_group)

            evidence_summary = api_result.get("evidence_summary") or {}

            detail = {
                "index": int(index),
                "id": row.get("id"),
                "statement": statement,
                "liar2_label": liar_label,
                "expected_group": expected_group,
                "final_label": final_label,
                "predicted_group": predicted_group,
                "expected_behavior": expected_behavior,
                "safe_behavior": safe_behavior,
                "risk_label": api_result.get("risk_label"),
                "risk_score": api_result.get("risk_score"),
                "model_used": api_result.get("model_used"),
                "classifier_confidence": api_result.get("confidence_score"),
                "classifier_probabilities": api_result.get("class_probabilities"),
                "raw_class": api_result.get("raw_class"),
                "usable_evidence_count": evidence_summary.get("usable_evidence_count"),
                "weak_evidence_count": evidence_summary.get("weak_evidence_count"),
                "supports_count": evidence_summary.get("supports_count"),
                "refutes_count": evidence_summary.get("refutes_count"),
                "top_relevance_score": evidence_summary.get("top_relevance_score"),
                "subject": row.get("subject"),
                "speaker": row.get("speaker"),
                "date": row.get("date"),
                "context": row.get("context"),
                "justification": row.get("justification"),
                "content_terms": api_result.get("influential_terms"),
                "top_evidence": api_result.get("evidence", [])[:3],
                "explanation": api_result.get("explanation"),
                "error": None,
            }

        except Exception as exc:
            detail = {
                "index": int(index),
                "id": row.get("id"),
                "statement": statement,
                "liar2_label": liar_label,
                "expected_group": expected_group,
                "final_label": None,
                "predicted_group": "error",
                "expected_behavior": False,
                "safe_behavior": False,
                "risk_label": None,
                "risk_score": None,
                "model_used": None,
                "classifier_confidence": None,
                "classifier_probabilities": None,
                "raw_class": None,
                "usable_evidence_count": None,
                "weak_evidence_count": None,
                "supports_count": None,
                "refutes_count": None,
                "top_relevance_score": None,
                "subject": row.get("subject"),
                "speaker": row.get("speaker"),
                "date": row.get("date"),
                "context": row.get("context"),
                "justification": row.get("justification"),
                "content_terms": None,
                "top_evidence": None,
                "explanation": None,
                "error": str(exc),
            }

        details.append(detail)

    summary = summarize_results(
        details=details,
        split=args.split,
        total_available=total_available,
        evaluated=max_samples,
        api_url=args.api_url,
    )

    return summary, details


def summarize_results(
    details: List[Dict],
    split: str,
    total_available: int,
    evaluated: int,
    api_url: str,
) -> Dict:
    processed = [item for item in details if item.get("error") is None]
    failed = [item for item in details if item.get("error") is not None]

    expected_behavior_count = sum(1 for item in processed if item["expected_behavior"])
    safe_behavior_count = sum(1 for item in processed if item["safe_behavior"])
    uncertain_count = sum(1 for item in processed if item["predicted_group"] == "uncertain")

    label_counts = Counter(item["liar2_label"] for item in processed)
    expected_group_counts = Counter(item["expected_group"] for item in processed)
    predicted_group_counts = Counter(item["predicted_group"] for item in processed)
    final_label_counts = Counter(item["final_label"] for item in processed)
    risk_label_counts = Counter(item["risk_label"] for item in processed)

    per_label = defaultdict(lambda: {"total": 0, "expected_behavior": 0, "safe_behavior": 0})

    for item in processed:
        label = item["liar2_label"]
        per_label[label]["total"] += 1

        if item["expected_behavior"]:
            per_label[label]["expected_behavior"] += 1

        if item["safe_behavior"]:
            per_label[label]["safe_behavior"] += 1

    per_label_summary = {}

    for label, values in per_label.items():
        total = values["total"]
        per_label_summary[label] = {
            "total": total,
            "expected_behavior": values["expected_behavior"],
            "safe_behavior": values["safe_behavior"],
            "expected_behavior_rate": round(values["expected_behavior"] / total, 4) if total else 0.0,
            "safe_behavior_rate": round(values["safe_behavior"] / total, 4) if total else 0.0,
        }

    binary_candidates = [
        item
        for item in processed
        if item["expected_group"] in {"likely_real", "fake_or_misleading"}
    ]

    binary_non_uncertain = [
        item
        for item in binary_candidates
        if item["predicted_group"] != "uncertain"
    ]

    binary_correct_non_uncertain = [
        item
        for item in binary_non_uncertain
        if item["expected_group"] == item["predicted_group"]
    ]

    evidence_usable_counts = [
        int(item.get("usable_evidence_count") or 0)
        for item in processed
    ]

    records_with_usable_evidence = sum(1 for count in evidence_usable_counts if count > 0)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "dataset": "LIAR2 Enhanced",
        "source": "local_csv",
        "split": split,
        "api_url": api_url,
        "model_mode": "transformer",
        "expected_model": "Fine-tuned DistilBERT 512",
        "total_available": total_available,
        "evaluated": evaluated,
        "processed": len(processed),
        "failed": len(failed),
        "expected_behavior_count": expected_behavior_count,
        "expected_behavior_rate": round(expected_behavior_count / len(processed), 4) if processed else 0.0,
        "safe_behavior_count": safe_behavior_count,
        "safe_behavior_rate": round(safe_behavior_count / len(processed), 4) if processed else 0.0,
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_count / len(processed), 4) if processed else 0.0,
        "records_with_usable_evidence": records_with_usable_evidence,
        "usable_evidence_rate": round(records_with_usable_evidence / len(processed), 4) if processed else 0.0,
        "binary_candidates": len(binary_candidates),
        "binary_non_uncertain": len(binary_non_uncertain),
        "binary_correct_non_uncertain": len(binary_correct_non_uncertain),
        "binary_accuracy_excluding_uncertain": (
            round(len(binary_correct_non_uncertain) / len(binary_non_uncertain), 4)
            if binary_non_uncertain
            else 0.0
        ),
        "liar2_label_counts": dict(label_counts),
        "expected_group_counts": dict(expected_group_counts),
        "predicted_group_counts": dict(predicted_group_counts),
        "final_label_counts": dict(final_label_counts),
        "risk_label_counts": dict(risk_label_counts),
        "per_label_summary": per_label_summary,
    }

    return summary


def save_outputs(summary: Dict, details: List[Dict], json_output: Path, csv_output: Path) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "summary": summary,
        "details": details,
    }

    with json_output.open("w", encoding="utf-8") as file:
        json.dump(json_payload, file, indent=4)

    csv_fields = [
        "index",
        "id",
        "statement",
        "liar2_label",
        "expected_group",
        "final_label",
        "predicted_group",
        "expected_behavior",
        "safe_behavior",
        "risk_label",
        "risk_score",
        "model_used",
        "classifier_confidence",
        "raw_class",
        "usable_evidence_count",
        "weak_evidence_count",
        "supports_count",
        "refutes_count",
        "top_relevance_score",
        "subject",
        "speaker",
        "date",
        "error",
    ]

    with csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()

        for item in details:
            writer.writerow({field: item.get(field) for field in csv_fields})


def main() -> None:
    args = parse_args()

    json_output = Path(args.json_output)
    csv_output = Path(args.csv_output)

    summary, details = evaluate_liar2_claims(args)

    save_outputs(
        summary=summary,
        details=details,
        json_output=json_output,
        csv_output=csv_output,
    )

    print()
    print("LIAR2 claim-level evaluation completed.")
    print(f"Dataset source                     : {summary['source']}")
    print(f"Evaluated records                  : {summary['evaluated']}")
    print(f"Processed records                  : {summary['processed']}")
    print(f"Failed records                     : {summary['failed']}")
    print(f"Expected behavior rate             : {summary['expected_behavior_rate']}")
    print(f"Safe behavior rate                 : {summary['safe_behavior_rate']}")
    print(f"Uncertain rate                     : {summary['uncertain_rate']}")
    print(f"Usable evidence rate               : {summary['usable_evidence_rate']}")
    print(f"Binary accuracy excluding uncertain: {summary['binary_accuracy_excluding_uncertain']}")
    print(f"JSON output                        : {json_output}")
    print(f"CSV output                         : {csv_output}")


if __name__ == "__main__":
    main()