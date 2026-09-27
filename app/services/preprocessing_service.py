import re


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def detect_input_length_type(text: str) -> str:
    """
    Simple runtime fallback when caller does not explicitly provide input_type.

    Final project input types:
    - claim: short factual assertion or social claim
    - headline: short headline-like text
    - article: longer article-style text
    """
    word_count = len(text.split())

    if word_count <= 25:
        return "claim"

    if word_count <= 80:
        return "headline"

    return "article"
