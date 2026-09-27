import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "baseline"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TRAIN_FILE = PROCESSED_DATA_DIR / "isot_train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "isot_test.csv"

VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.joblib"
LOGISTIC_MODEL_FILE = MODELS_DIR / "logistic_regression_model.joblib"
SVM_MODEL_FILE = MODELS_DIR / "linear_svm_model.joblib"

METRICS_FILE = METRICS_DIR / "baseline_metrics.json"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing train file: {TRAIN_FILE}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_FILE}")

    print("Loading processed ISOT train/test data...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_df = train_df.dropna(subset=["full_text", "label"]).copy()
    test_df = test_df.dropna(subset=["full_text", "label"]).copy()

    return train_df, test_df


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )


def train_logistic_regression(x_train, y_train) -> LogisticRegression:
    print("Training Logistic Regression model...")

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    model.fit(x_train, y_train)

    return model


def train_linear_svm(x_train, y_train) -> LinearSVC:
    print("Training Linear SVM model...")

    model = LinearSVC(
        class_weight="balanced",
        random_state=42,
    )

    model.fit(x_train, y_train)

    return model


def evaluate_model(model, x_test, y_test, model_name: str) -> dict:
    print(f"Evaluating {model_name}...")

    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["real", "fake"],
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return metrics


def save_confusion_matrix(model_name: str, confusion_matrix_values: list[list[int]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(confusion_matrix_values)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticklabels(["Real", "Fake"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                confusion_matrix_values[i][j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    output_file = FIGURES_DIR / f"{safe_name}_confusion_matrix.png"

    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved confusion matrix: {output_file}")


def save_outputs(
    vectorizer: TfidfVectorizer,
    logistic_model: LogisticRegression,
    svm_model: LinearSVC,
    metrics: dict,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(logistic_model, LOGISTIC_MODEL_FILE)
    joblib.dump(svm_model, SVM_MODEL_FILE)

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nSaved model files:")
    print(f"- {VECTORIZER_FILE}")
    print(f"- {LOGISTIC_MODEL_FILE}")
    print(f"- {SVM_MODEL_FILE}")
    print(f"- {METRICS_FILE}")


def main() -> None:
    train_df, test_df = load_data()

    x_train_text = train_df["full_text"]
    y_train = train_df["label"]

    x_test_text = test_df["full_text"]
    y_test = test_df["label"]

    print("\nVectorizing text using TF-IDF...")
    vectorizer = build_tfidf_vectorizer()

    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    print(f"TF-IDF train shape: {x_train.shape}")
    print(f"TF-IDF test shape : {x_test.shape}")

    logistic_model = train_logistic_regression(x_train, y_train)
    svm_model = train_linear_svm(x_train, y_train)

    logistic_metrics = evaluate_model(
        logistic_model,
        x_test,
        y_test,
        "Logistic Regression",
    )

    svm_metrics = evaluate_model(
        svm_model,
        x_test,
        y_test,
        "Linear SVM",
    )

    all_metrics = {
        "dataset": "ISOT Fake News Dataset",
        "label_mapping": {
            "0": "real",
            "1": "fake",
        },
        "models": {
            "logistic_regression": logistic_metrics,
            "linear_svm": svm_metrics,
        },
    }

    save_confusion_matrix(
        "Logistic Regression",
        logistic_metrics["confusion_matrix"],
    )

    save_confusion_matrix(
        "Linear SVM",
        svm_metrics["confusion_matrix"],
    )

    save_outputs(
        vectorizer,
        logistic_model,
        svm_model,
        all_metrics,
    )

    print("\nBaseline model training completed successfully.")

    print("\nLogistic Regression metrics:")
    print(json.dumps(logistic_metrics, indent=4))

    print("\nLinear SVM metrics:")
    print(json.dumps(svm_metrics, indent=4))


if __name__ == "__main__":
    main()