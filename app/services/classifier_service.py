from pathlib import Path
from typing import Dict, Optional

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
TRANSFORMER_MODEL_DIR_FINAL = (
    PROJECT_ROOT / "models" / "transformer" / "distilbert_isot_full_512"
)
TRANSFORMER_MODEL_DIR_FALLBACK = (
    PROJECT_ROOT / "models" / "transformer" / "distilbert_isot"
)

VECTORIZER_FILE = BASELINE_MODEL_DIR / "tfidf_vectorizer.joblib"
LOGISTIC_MODEL_FILE = BASELINE_MODEL_DIR / "logistic_regression_model.joblib"
LINEAR_SVM_MODEL_FILE = BASELINE_MODEL_DIR / "linear_svm_model.joblib"

MAX_TRANSFORMER_LENGTH = 512

_baseline_vectorizer = None
_logistic_model = None
_linear_svm_model = None

_transformer_tokenizer = None
_transformer_model = None
_transformer_device = None


def normalize_mode(inference_mode: Optional[str] = None) -> str:
    mode = (inference_mode or settings.inference_mode or settings.model_mode or "transformer")
    return mode.lower().strip()


def get_transformer_model_dir() -> Path:
    if TRANSFORMER_MODEL_DIR_FINAL.exists():
        return TRANSFORMER_MODEL_DIR_FINAL

    return TRANSFORMER_MODEL_DIR_FALLBACK


def load_vectorizer():
    global _baseline_vectorizer

    if _baseline_vectorizer is None:
        if not VECTORIZER_FILE.exists():
            raise FileNotFoundError(f"Missing vectorizer file: {VECTORIZER_FILE}")

        _baseline_vectorizer = joblib.load(VECTORIZER_FILE)

    return _baseline_vectorizer


def load_logistic_model():
    global _logistic_model

    if _logistic_model is None:
        if not LOGISTIC_MODEL_FILE.exists():
            raise FileNotFoundError(f"Missing model file: {LOGISTIC_MODEL_FILE}")

        _logistic_model = joblib.load(LOGISTIC_MODEL_FILE)

    return _logistic_model


def load_linear_svm_model():
    global _linear_svm_model

    if _linear_svm_model is None:
        if not LINEAR_SVM_MODEL_FILE.exists():
            raise FileNotFoundError(f"Missing model file: {LINEAR_SVM_MODEL_FILE}")

        _linear_svm_model = joblib.load(LINEAR_SVM_MODEL_FILE)

    return _linear_svm_model


def load_transformer_model():
    global _transformer_tokenizer, _transformer_model, _transformer_device

    transformer_model_dir = get_transformer_model_dir()

    if not transformer_model_dir.exists():
        raise FileNotFoundError(
            f"Missing transformer model directory: {transformer_model_dir}. "
            "Run DistilBERT training first."
        )

    if _transformer_device is None:
        _transformer_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if _transformer_tokenizer is None:
        _transformer_tokenizer = AutoTokenizer.from_pretrained(
            str(transformer_model_dir),
            local_files_only=True,
        )

    if _transformer_model is None:
        _transformer_model = AutoModelForSequenceClassification.from_pretrained(
            str(transformer_model_dir),
            local_files_only=True,
        )
        _transformer_model.to(_transformer_device)
        _transformer_model.eval()

    return _transformer_tokenizer, _transformer_model, _transformer_device


def label_from_class(predicted_class: int) -> str:
    return "Likely Fake" if predicted_class == 1 else "Likely Real"


def classify_with_logistic_regression(text: str) -> Dict:
    vectorizer = load_vectorizer()
    model = load_logistic_model()

    text_vector = vectorizer.transform([text])
    predicted_class = int(model.predict(text_vector)[0])
    probabilities = model.predict_proba(text_vector)[0]
    confidence_score = float(max(probabilities))

    return {
        "predicted_label": label_from_class(predicted_class),
        "confidence_score": round(confidence_score, 4),
        "model_used": "TF-IDF + Logistic Regression",
        "raw_class": predicted_class,
        "class_probabilities": {
            "real": round(float(probabilities[0]), 4),
            "fake": round(float(probabilities[1]), 4),
        },
    }


def classify_with_linear_svm(text: str) -> Dict:
    vectorizer = load_vectorizer()
    model = load_linear_svm_model()

    text_vector = vectorizer.transform([text])
    predicted_class = int(model.predict(text_vector)[0])

    # Linear SVM does not normally expose probabilities. Use a transparent
    # sigmoid transform of the decision score as an approximate confidence for
    # dashboard comparison only. Formal metrics still come from held-out test data.
    decision_score = float(model.decision_function(text_vector)[0])
    pseudo_fake_probability = 1.0 / (1.0 + torch.exp(torch.tensor(-decision_score)).item())

    if predicted_class == 1:
        fake_probability = pseudo_fake_probability
        real_probability = 1.0 - pseudo_fake_probability
    else:
        real_probability = 1.0 - pseudo_fake_probability
        fake_probability = pseudo_fake_probability

    confidence_score = max(real_probability, fake_probability)

    return {
        "predicted_label": label_from_class(predicted_class),
        "confidence_score": round(float(confidence_score), 4),
        "model_used": "TF-IDF + Linear SVM",
        "raw_class": predicted_class,
        "class_probabilities": {
            "real": round(float(real_probability), 4),
            "fake": round(float(fake_probability), 4),
        },
    }


def classify_with_transformer(text: str) -> Dict:
    tokenizer, model, device = load_transformer_model()

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_TRANSFORMER_LENGTH,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

    predicted_class = int(probabilities.argmax())
    confidence_score = float(probabilities[predicted_class])

    return {
        "predicted_label": label_from_class(predicted_class),
        "confidence_score": round(confidence_score, 4),
        "model_used": "Fine-tuned DistilBERT 512",
        "raw_class": predicted_class,
        "class_probabilities": {
            "real": round(float(probabilities[0]), 4),
            "fake": round(float(probabilities[1]), 4),
        },
    }


def classify_text(text: str, inference_mode: Optional[str] = None) -> Dict:
    mode = normalize_mode(inference_mode)

    if mode in {"traditional_lr", "baseline", "logistic_regression"}:
        return classify_with_logistic_regression(text)

    if mode in {"traditional_svm", "linear_svm"}:
        return classify_with_linear_svm(text)

    if mode == "transformer" and settings.model_mode.lower().strip() != "transformer":
        return classify_with_logistic_regression(text)

    return classify_with_transformer(text)

def load_baseline_model():
    """
    Backward-compatible helper used by explanation_service/LIME.
    Returns the TF-IDF vectorizer and Logistic Regression baseline model.
    """
    vectorizer = load_vectorizer()
    model = load_logistic_model()
    return vectorizer, model
