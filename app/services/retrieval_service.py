from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings

from app.services.web_search_service import search_live_web


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEVER_CHROMA_PATH = PROJECT_ROOT / "data" / "embeddings" / "chroma_fever"
FEVER_COLLECTION_NAME = "fever_evidence"

LIAR2_CHROMA_PATH = PROJECT_ROOT / "data" / "embeddings" / "chroma_liar2_justifications"
LIAR2_COLLECTION_NAME = "liar2_justification_evidence"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBEDDING_MODEL_DIR = PROJECT_ROOT / "models" / "embedding" / "all-MiniLM-L6-v2"

# Lower than 0.35 because LIAR2 claims are short and paraphrased.
# The negation-direction guard below protects cases such as:
# user claim: "Obama was born in Hawaii"
# retrieved claim: "Hawaii governor can't find proof Obama was born in Hawaii"
MIN_LIAR2_STANCE_CLAIM_SIMILARITY = 0.20

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "from", "with", "without", "by", "as",
    "at", "it", "its", "his", "her", "their", "our", "your", "you", "we",
    "they", "he", "she", "them", "him", "who", "what", "when", "where",
    "why", "how", "says", "said", "say", "has", "have", "had", "will",
    "would", "could", "should", "about", "into", "over", "under", "after",
    "before", "during", "more", "most", "less", "least", "all", "any",
}

NEGATION_CUES = {
    "not", "no", "never", "none", "cannot", "can't", "cant", "doesn't",
    "doesnt", "didn't", "didnt", "isn't", "isnt", "aren't", "arent",
    "won't", "wont", "false", "fake", "hoax", "bogus", "fabricated",
    "doctored", "misleading",
}

_embedding_model: Optional[SentenceTransformer] = None
_fever_collection = None
_liar2_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model

    if _embedding_model is None:
        if LOCAL_EMBEDDING_MODEL_DIR.exists():
            _embedding_model = SentenceTransformer(str(LOCAL_EMBEDDING_MODEL_DIR))
        else:
            # Keep a fallback for development, but use scripts/save_embedding_model.py
            # before offline submission/demo to avoid Hugging Face Hub calls.
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model


def get_chroma_collection(chroma_path: Path, collection_name: str):
    if not chroma_path.exists():
        print(f"Evidence index path not found: {chroma_path}")
        return None

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        return client.get_collection(collection_name)
    except Exception as error:
        print(
            f"Could not load Chroma collection '{collection_name}' "
            f"from {chroma_path}: {error}"
        )
        return None


def get_fever_collection():
    global _fever_collection

    if _fever_collection is None:
        _fever_collection = get_chroma_collection(
            FEVER_CHROMA_PATH,
            FEVER_COLLECTION_NAME,
        )

    return _fever_collection


def get_liar2_collection():
    global _liar2_collection

    if _liar2_collection is None:
        _liar2_collection = get_chroma_collection(
            LIAR2_CHROMA_PATH,
            LIAR2_COLLECTION_NAME,
        )

    return _liar2_collection


def calculate_relevance_score(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0

    distance = max(float(distance), 0.0)
    relevance_score = 1.0 / (1.0 + distance)

    return round(relevance_score, 4)


def normalize_evidence_label(label: Optional[str]) -> str:
    if not label:
        return "UNKNOWN"

    label = str(label).upper().strip()

    if label in {"SUPPORTS", "SUPPORT"}:
        return "SUPPORTS"

    if label in {"REFUTES", "REFUTE"}:
        return "REFUTES"

    if label in {"NOT_ENOUGH_INFO", "NOT ENOUGH INFO", "NEI"}:
        return "NOT_ENOUGH_INFO"

    return label


def clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""

    text = str(value)

    if text.lower() == "nan":
        return ""

    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())

    return text.strip()


def normalize_token(token: str) -> str:
    token = token.lower().strip()

    if token.endswith("'s"):
        token = token[:-2]

    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("s"):
        token = token[:-1]

    return token


def extract_tokens(text: str) -> List[str]:
    cleaned = clean_text(text).lower()
    raw_tokens: List[str] = []

    current: List[str] = []

    for char in cleaned:
        if char.isalnum() or char in {"-", "'"}:
            current.append(char)
        else:
            if current:
                raw_tokens.append("".join(current))
                current = []

    if current:
        raw_tokens.append("".join(current))

    tokens = [
        normalize_token(token)
        for token in raw_tokens
        if normalize_token(token)
    ]

    return tokens


def get_content_tokens(text: str) -> set:
    tokens = {
        token
        for token in extract_tokens(text)
        if len(token) >= 3 and token not in STOPWORDS
    }

    return tokens


def calculate_token_jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = get_content_tokens(text_a)
    tokens_b = get_content_tokens(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection_count = len(tokens_a.intersection(tokens_b))
    union_count = len(tokens_a.union(tokens_b))

    if union_count == 0:
        return 0.0

    return round(intersection_count / union_count, 4)


def has_negation_cue(text: str) -> bool:
    # Use all tokens, not only content tokens, so short cue "no" is preserved.
    tokens = set(extract_tokens(text))
    return bool(tokens.intersection(NEGATION_CUES))


def should_trust_liar2_stance(
    query_text: str,
    source_statement: str,
    evidence_label: str,
) -> Tuple[bool, float, str]:
    claim_similarity = calculate_token_jaccard_similarity(
        query_text,
        source_statement,
    )

    if evidence_label not in {"SUPPORTS", "REFUTES"}:
        return True, claim_similarity, "label_is_not_directional"

    if not source_statement:
        return False, claim_similarity, "missing_source_statement"

    if claim_similarity < MIN_LIAR2_STANCE_CLAIM_SIMILARITY:
        return False, claim_similarity, "source_claim_not_close_enough"

    query_has_negation = has_negation_cue(query_text)
    source_has_negation = has_negation_cue(source_statement)

    # If LIAR2 refutes a negated/skeptical claim, that refutation may support
    # the user's positive claim. Without an NLI stance model, downgrade to NEI.
    #
    # Example:
    # user: "Barack Obama was born in Hawaii."
    # source: "Hawaii governor can't find proof Obama was born in Hawaii."
    # LIAR2 rating: false -> REFUTES
    # Directly treating REFUTES as user-claim REFUTES would be wrong.
    if evidence_label == "REFUTES" and source_has_negation and not query_has_negation:
        return False, claim_similarity, "negation_direction_mismatch"

    return True, claim_similarity, "trusted"


def build_source_name(source_type: str, metadata: Dict) -> str:
    if source_type == "local_fever":
        source = (
            metadata.get("source")
            or metadata.get("source_name")
            or metadata.get("evidence_source")
            or "FEVER"
        )
        return clean_text(source) or "FEVER"

    if source_type == "local_liar2":
        speaker = clean_text(metadata.get("speaker"))
        subject = clean_text(metadata.get("subject"))

        if speaker and subject:
            return f"LIAR2 Enhanced | {speaker} | {subject}"

        if speaker:
            return f"LIAR2 Enhanced | {speaker}"

        return "LIAR2 Enhanced"

    return "Unknown Evidence Source"


def query_collection(
    collection,
    query_text: str,
    top_k: int,
    source_type: str,
) -> List[Dict]:
    if collection is None:
        return []

    if not query_text.strip():
        return []

    embedding_model = get_embedding_model()

    query_embedding = embedding_model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].tolist()

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as error:
        print(f"Evidence retrieval failed for {source_type}: {error}")
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    evidence_items: List[Dict] = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}

        evidence_label = normalize_evidence_label(
            metadata.get("label") or metadata.get("evidence_label")
        )

        relevance_score = calculate_relevance_score(distance)
        source_name = build_source_name(source_type, metadata)
        document_text = clean_text(document)

        if not document_text:
            continue

        source_statement = clean_text(metadata.get("statement"))

        liar2_stance_trusted = True
        liar2_claim_similarity = None
        liar2_stance_reason = ""

        if source_type == "local_liar2":
            (
                liar2_stance_trusted,
                liar2_claim_similarity,
                liar2_stance_reason,
            ) = should_trust_liar2_stance(
                query_text=query_text,
                source_statement=source_statement,
                evidence_label=evidence_label,
            )

            if not liar2_stance_trusted:
                evidence_label = "NOT_ENOUGH_INFO"

        evidence_items.append(
            {
                "source": source_name,
                "text": document_text,
                "relevance_score": relevance_score,
                "evidence_label": evidence_label,
                "source_type": source_type,
                "source_statement": source_statement,
                "liar2_claim_similarity": liar2_claim_similarity,
                "liar2_stance_trusted": liar2_stance_trusted,
                "liar2_stance_reason": liar2_stance_reason,
            }
        )

    return evidence_items


def deduplicate_evidence(evidence: List[Dict]) -> List[Dict]:
    best_by_text: Dict[str, Dict] = {}

    for item in evidence:
        text = clean_text(item.get("text", ""))
        key = text.lower()

        if not key:
            continue

        current_best = best_by_text.get(key)

        if current_best is None:
            best_by_text[key] = item
            continue

        if float(item.get("relevance_score", 0.0)) > float(
            current_best.get("relevance_score", 0.0)
        ):
            best_by_text[key] = item

    return list(best_by_text.values())


def retrieve_evidence(
    query_text: str,
    top_k: int = 5,
    evidence_mode: str = "local",
) -> List[Dict]:
    all_evidence: List[Dict] = []

    # Retrieve more than final top_k from each source so the merge has enough candidates.
    per_source_top_k = max(top_k, 5)

    if getattr(settings, "local_fever_evidence_enabled", True):
        fever_evidence = query_collection(
            collection=get_fever_collection(),
            query_text=query_text,
            top_k=per_source_top_k,
            source_type="local_fever",
        )
        all_evidence.extend(fever_evidence)

    if getattr(settings, "local_liar2_evidence_enabled", True):
        liar2_evidence = query_collection(
            collection=get_liar2_collection(),
            query_text=query_text,
            top_k=per_source_top_k,
            source_type="local_liar2",
        )
        all_evidence.extend(liar2_evidence)

    normalized_evidence_mode = str(evidence_mode or "local").lower().strip()

    should_use_web = (
        normalized_evidence_mode == "hybrid"
        or (
            getattr(settings, "runtime_mode", "offline_eval").lower() == "hybrid"
            and getattr(settings, "live_search_enabled", False)
        )
    )

    if should_use_web:
        web_evidence = search_live_web(query_text=query_text)
        all_evidence.extend(web_evidence)

    deduplicated_evidence = deduplicate_evidence(all_evidence)

    ranked_evidence = sorted(
        deduplicated_evidence,
        key=lambda item: float(
            item.get("final_evidence_score")
            or item.get("relevance_score")
            or 0.0
        ),
        reverse=True,
    )

    return ranked_evidence[:top_k]