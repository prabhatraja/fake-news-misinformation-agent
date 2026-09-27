import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "isot_train.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "isot_test.csv"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "transformer" / "distilbert_isot"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

METRICS_FILE = METRICS_DIR / "distilbert_isot_metrics.json"
CONFUSION_MATRIX_FILE = FIGURES_DIR / "distilbert_confusion_matrix.png"

MODEL_NAME = "distilbert-base-uncased"

LABEL_MAPPING = {
    0: "real",
    1: "fake",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quick_run",
        action="store_true",
        help="Run on a small subset first to verify pipeline and GPU setup.",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional maximum number of training samples.",
    )

    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional maximum number of evaluation samples.",
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum token length.",
    )

    return parser.parse_args()


def print_gpu_status():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory: {total_memory_gb:.2f} GB")


def load_data(args):
    print("Loading ISOT train/test data...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    required_columns = {"full_text", "label"}

    if not required_columns.issubset(train_df.columns):
        raise ValueError(f"Train file must contain columns: {required_columns}")

    if not required_columns.issubset(test_df.columns):
        raise ValueError(f"Test file must contain columns: {required_columns}")

    train_df = train_df[["full_text", "label"]].dropna()
    test_df = test_df[["full_text", "label"]].dropna()

    train_df["labels"] = train_df["label"].astype(int)
    test_df["labels"] = test_df["label"].astype(int)

    train_df = train_df.drop(columns=["label"])
    test_df = test_df.drop(columns=["label"])

    if args.quick_run:
        print("Quick run enabled. Using small subset.")
        train_df = train_df.sample(n=min(1000, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(300, len(test_df)), random_state=42)

    if args.max_train_samples:
        train_df = train_df.sample(
            n=min(args.max_train_samples, len(train_df)),
            random_state=42,
        )

    if args.max_eval_samples:
        test_df = test_df.sample(
            n=min(args.max_eval_samples, len(test_df)),
            random_state=42,
        )

    print(f"Train records: {len(train_df)}")
    print(f"Eval records : {len(test_df)}")

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    eval_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    return train_dataset, eval_dataset


def tokenize_dataset(train_dataset, eval_dataset, tokenizer, max_length):
    def tokenize_function(batch):
        return tokenizer(
            batch["full_text"],
            truncation=True,
            max_length=max_length,
        )

    print("Tokenizing datasets...")

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["full_text"],
    )

    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["full_text"],
    )

    return tokenized_train, tokenized_eval


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def save_final_metrics(trainer, eval_dataset):
    print("Running final evaluation...")

    predictions_output = trainer.predict(eval_dataset)

    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    predictions = np.argmax(logits, axis=-1)

    metrics = {
        "model_name": MODEL_NAME,
        "trained_model_path": str(MODEL_OUTPUT_DIR),
        "generated_at": datetime.now().isoformat(),
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=["real", "fake"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "label_mapping": {
            "0": "real",
            "1": "fake",
        },
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Saved metrics to: {METRICS_FILE}")
    print(json.dumps(metrics, indent=4))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["real", "fake"],
    )

    display.plot(values_format="d")
    plt.title("DistilBERT Confusion Matrix - ISOT Test Set")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=200)
    plt.close()

    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_FILE}")


def main():
    args = parse_args()

    print_gpu_status()

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, eval_dataset = load_data(args)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_train, tokenized_eval = tokenize_dataset(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    print("Loading DistilBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "real",
            1: "fake",
        },
        label2id={
            "real": 0,
            "fake": 1,
        },
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    fp16_enabled = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16_enabled,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting DistilBERT training...")
    trainer.train()

    print("Saving best model and tokenizer...")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    save_final_metrics(trainer, tokenized_eval)

    print()
    print("DistilBERT training completed successfully.")


if __name__ == "__main__":
    main()