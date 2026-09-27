import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "isot"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

TRUE_FILE = RAW_DATA_DIR / "True.csv"
FAKE_FILE = RAW_DATA_DIR / "Fake.csv"

CLEAN_FILE = PROCESSED_DATA_DIR / "isot_clean.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "isot_train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "isot_test.csv"
SUMMARY_FILE = METRICS_DIR / "isot_dataset_summary.json"


def clean_text(value: str) -> str:
    if pd.isna(value):
        return ""

    value = str(value)
    value = value.strip()
    value = re.sub(r"\s+", " ", value)

    return value


def validate_input_files() -> None:
    if not TRUE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TRUE_FILE}")

    if not FAKE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {FAKE_FILE}")


def load_isot_dataset() -> pd.DataFrame:
    print("Loading ISOT dataset...")

    true_df = pd.read_csv(TRUE_FILE)
    fake_df = pd.read_csv(FAKE_FILE)

    true_df["label"] = 0
    true_df["label_name"] = "real"

    fake_df["label"] = 1
    fake_df["label_name"] = "fake"

    df = pd.concat([true_df, fake_df], ignore_index=True)

    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("Preparing dataset...")

    required_columns = {"title", "text", "subject", "date", "label", "label_name"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)
    df["subject"] = df["subject"].apply(clean_text)
    df["date"] = df["date"].apply(clean_text)

    df["full_text"] = (df["title"] + ". " + df["text"]).apply(clean_text)

    before_rows = len(df)

    df = df[df["full_text"].str.len() > 20].copy()
    df = df.drop_duplicates(subset=["full_text"]).copy()

    after_rows = len(df)

    df["text_length_chars"] = df["full_text"].str.len()
    df["text_length_words"] = df["full_text"].str.split().str.len()

    df = df[
        [
            "title",
            "text",
            "full_text",
            "subject",
            "date",
            "label",
            "label_name",
            "text_length_chars",
            "text_length_words",
        ]
    ].copy()

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Rows before cleaning: {before_rows}")
    print(f"Rows after cleaning : {after_rows}")
    print(f"Removed rows        : {before_rows - after_rows}")

    return df


def create_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Creating train/test split...")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def create_summary(
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    summary = {
        "dataset_name": "ISOT Fake News Dataset",
        "total_records": int(len(clean_df)),
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "label_distribution_total": clean_df["label_name"].value_counts().to_dict(),
        "label_distribution_train": train_df["label_name"].value_counts().to_dict(),
        "label_distribution_test": test_df["label_name"].value_counts().to_dict(),
        "subject_distribution_total": clean_df["subject"].value_counts().to_dict(),
        "average_words_per_article": float(round(clean_df["text_length_words"].mean(), 2)),
        "median_words_per_article": float(round(clean_df["text_length_words"].median(), 2)),
        "min_words_per_article": int(clean_df["text_length_words"].min()),
        "max_words_per_article": int(clean_df["text_length_words"].max()),
        "label_mapping": {
            "0": "real",
            "1": "fake",
        },
    }

    return summary


def save_outputs(
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary: dict,
) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(CLEAN_FILE, index=False)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nSaved files:")
    print(f"- {CLEAN_FILE}")
    print(f"- {TRAIN_FILE}")
    print(f"- {TEST_FILE}")
    print(f"- {SUMMARY_FILE}")


def main() -> None:
    validate_input_files()

    raw_df = load_isot_dataset()
    clean_df = prepare_dataset(raw_df)
    train_df, test_df = create_train_test_split(clean_df)
    summary = create_summary(clean_df, train_df, test_df)

    save_outputs(clean_df, train_df, test_df, summary)

    print("\nDataset preparation completed successfully.")
    print("\nSummary:")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()