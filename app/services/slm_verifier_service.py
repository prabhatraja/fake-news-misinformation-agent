import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import requests

from app.core.config import settings


VALID_STANCES = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}
SLM_MODES = {"slm_assisted_local", "slm_assisted_hybrid"}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def truncate_text(text: str, max_chars: int = 650) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_dotenv_values() -> Dict[str, str]:
    env_path = project_root() / ".env"
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_runtime_slm_config() -> Dict[str, Any]:
    """Read final-demo SLM configuration at call time.

    Reading .env at call time prevents stale shell variables from silently
    redirecting the verifier to an older model/provider configuration.
    """
    dotenv = read_dotenv_values()

    def pick(key: str, attr: str, default: str) -> str:
        value = dotenv.get(key)
        if value is None or str(value).strip() == "":
            value = os.environ.get(key)
        if value is None or str(value).strip() == "":
            value = getattr(settings, attr, default)
        return clean_text(value)

    def pick_bool(key: str, attr: str, default: bool) -> bool:
        return pick(key, attr, str(default)).strip().lower() in {"1", "true", "yes", "y", "on"}

    def pick_float(key: str, attr: str, default: float) -> float:
        try:
            return float(pick(key, attr, str(default)))
        except Exception:
            return float(default)

    def pick_int(key: str, attr: str, default: int) -> int:
        try:
            return int(float(pick(key, attr, str(default))))
        except Exception:
            return int(default)

    return {
        "enabled": pick_bool("SLM_ENABLED", "slm_enabled", True),
        "provider": "lmstudio",
        "base_url": pick("SLM_BASE_URL", "slm_base_url", "http://127.0.0.1:1234/v1"),
        "model": pick("SLM_MODEL", "slm_model", "qwen/qwen3.5-9b"),
        "timeout_seconds": pick_int("SLM_TIMEOUT_SECONDS", "slm_timeout_seconds", 180),
        "temperature": pick_float("SLM_TEMPERATURE", "slm_temperature", 0.0),
        "max_evidence_items": pick_int("SLM_MAX_EVIDENCE_ITEMS", "slm_max_evidence_items", 3),
        "semantic_evidence_items": pick_int(
            "SLM_SEMANTIC_EVIDENCE_ITEMS", "slm_semantic_evidence_items", 5
        ),
        "max_tokens": pick_int("SLM_MAX_TOKENS", "slm_max_tokens", 1200),
        "min_confidence_to_override": pick_float(
            "SLM_MIN_CONFIDENCE_TO_OVERRIDE", "slm_min_confidence_to_override", 0.60
        ),
    }


def build_evidence_context(evidence: List[Dict]) -> str:
    cfg = get_runtime_slm_config()
    # The overall verifier still uses one model call, but the semantic audit covers
    # all evidence cards normally displayed by the application (five by default).
    # This avoids leaving later live-web items on weaker lexical labels that can
    # create artificial mixed evidence.
    semantic_limit = max(1, int(cfg.get("semantic_evidence_items", 5)))
    selected = evidence[:semantic_limit]
    lines: List[str] = []

    for index, item in enumerate(selected, start=1):
        source = clean_text(item.get("source", "Unknown"))
        title = clean_text(item.get("title", ""))
        existing_label = clean_text(item.get("evidence_label", "UNKNOWN"))
        relevance = item.get("relevance_score", 0.0)
        trust = item.get("trust_score")
        published_date = clean_text(item.get("published_date", ""))
        provider = clean_text(item.get("provider", ""))
        text = truncate_text(item.get("text", ""), max_chars=750)

        header = [
            f"Evidence {index}",
            f"Source: {source}",
        ]
        if title:
            header.append(f"Title: {title}")
        if provider:
            header.append(f"Provider: {provider}")
        if published_date:
            header.append(f"Published date: {published_date}")
        header.extend(
            [
                f"Existing heuristic/dataset label: {existing_label}",
                f"Relevance score: {relevance}",
            ]
        )
        if trust is not None:
            header.append(f"Source trust score: {trust}")
        header.append(f"Text: {text}")
        lines.append("\n".join(header))

    return "\n\n".join(lines)


def build_prompt(claim: str, evidence: List[Dict]) -> str:
    evidence_context = build_evidence_context(evidence)
    semantic_limit = max(1, int(get_runtime_slm_config().get("semantic_evidence_items", 5)))
    shown_count = min(len(evidence), semantic_limit)

    return f"""/no_think
You are a strict claim-to-evidence relation verifier.

Evaluate the user's EXACT claim against ONLY the retrieved evidence snippets below.
Your job is NOT to decide whether an evidence snippet is generally true. Your job is to
decide whether that snippet SUPPORTS, REFUTES, or does not establish THIS EXACT CLAIM.

Critical rules:
1. Use only the supplied evidence. Do not use outside knowledge.
2. The existing heuristic/dataset label is only a hint. Independently judge the actual text.
3. SUPPORTS requires the evidence to establish the same important factual proposition as the claim.
   For event-occurrence claims, the evidence must state or clearly entail that the event actually happened/completed.
   A plan, objective, capability, intended destination, scheduled date, expected action, or future-tense description
   does NOT establish that the event occurred and must be NOT_ENOUGH_INFO unless occurrence is also stated.
   Wording such as "consistent with" or "supports the context" is not enough for SUPPORTS.
4. REFUTES applies when the evidence contradicts ANY material part of the claim, including:
   subject/entity, action/relation, location/destination, date/time, quantity, identity, causality, or negation.
5. NOT_ENOUGH_INFO applies when the evidence is merely about the same topic, is ambiguous,
   contains only background/context, or does not establish the disputed proposition.
6. A true statement about the same subject does NOT automatically SUPPORT the claim.
7. Entity overlap alone is never sufficient for SUPPORTS.
8. If a material slot differs, treat that as a contradiction even without words such as
   'false', 'fake', 'incorrect', or 'not'.
9. Temporal context matters. A source written before an event that says the event has not yet
   happened does NOT support a different location/outcome claim. Judge what it actually establishes.
10. If an evidence item contains multiple statements, use only statements relevant to the claim.
11. Judge EVERY evidence item shown below exactly once. Return {shown_count} evidence_judgments.
12. If you are unsure about an individual item, use NOT_ENOUGH_INFO rather than guessing SUPPORTS.
13. The overall stance must summarize the evidence set. If reliable items materially conflict, use
   NOT_ENOUGH_INFO unless one direction is clearly stronger and independently corroborated.
14. Return valid JSON only. No markdown or prose outside JSON.
15. Overall reason and item reasons must be concise and evidence-grounded.
16. Confidence values must be numbers between 0.0 and 1.0.

Important relation examples:
- Claim: "Mission X landed on Mars."
  Evidence: "Mission X successfully landed on the Moon."
  => REFUTES. The destination materially conflicts (Mars vs Moon).
- Claim: "The Taj Mahal is in Jaipur."
  Evidence: "The Taj Mahal is in Agra."
  => REFUTES. Same subject, conflicting location.
- Claim: "Country A won the 2022 tournament."
  Evidence: "Country A participated in the 2022 tournament."
  => NOT_ENOUGH_INFO. Participation does not establish winning.
- Claim: "Mission X successfully soft-landed on the Moon in August 2023."
  Evidence: "Mission X has the objective and capability to demonstrate a safe soft landing on the lunar surface."
  => NOT_ENOUGH_INFO. An objective/capability does not establish that the landing actually occurred.
- Claim: "Mission X landed on Mars in August."
  Evidence: "Before the scheduled lunar landing, Mission X had not yet landed on the Moon."
  => NOT_ENOUGH_INFO or REFUTES if the snippet clearly establishes a lunar destination; never SUPPORTS.

Claim:
{clean_text(claim)}

Retrieved evidence:
{evidence_context}

Return JSON exactly in this shape:
{{
  "stance": "SUPPORTS",
  "confidence": 0.0,
  "reason": "short claim-relative evidence reason",
  "used_evidence_indices": [1],
  "evidence_judgments": [
    {{
      "evidence_index": 1,
      "stance": "SUPPORTS",
      "confidence": 0.0,
      "material_conflict": false,
      "conflict_fields": [],
      "reason": "short claim-relative reason"
    }}
  ]
}}
""".strip()


def parse_json_from_text(text: str) -> Dict:
    """Parse a local-model JSON response conservatively."""
    text = clean_text(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Conservative fallback for a truncated response. Evidence-level judgments are
    # deliberately omitted unless valid JSON was returned.
    stance_match = re.search(
        r'"stance"\s*:\s*"?(SUPPORTS|REFUTES|NOT_ENOUGH_INFO|NOT ENOUGH INFO|NEI)"?',
        text,
        flags=re.IGNORECASE,
    )
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text, flags=re.IGNORECASE)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', text, flags=re.IGNORECASE)
    indices_match = re.search(r'"used_evidence_indices"\s*:\s*\[([^\]]*)\]', text, flags=re.IGNORECASE)

    if not stance_match:
        raise ValueError(f"SLM response did not contain JSON. Raw response: {text[:300]}")

    stance = stance_match.group(1).upper().replace(" ", "_")
    confidence = 0.0
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
        except Exception:
            confidence = 0.0
    if confidence > 1.0:
        confidence /= 100.0
    confidence = max(0.0, min(1.0, confidence))

    used_indices: List[int] = []
    if indices_match:
        for value in indices_match.group(1).split(","):
            value = value.strip()
            if value.isdigit():
                used_indices.append(int(value))

    return {
        "stance": stance,
        "confidence": confidence,
        "reason": reason_match.group(1).strip() if reason_match else "",
        "used_evidence_indices": used_indices,
        "evidence_judgments": [],
    }


def _normalize_stance(value: Any) -> str:
    stance = clean_text(value or "NOT_ENOUGH_INFO").upper().replace(" ", "_")
    stance_map = {
        "SUPPORT": "SUPPORTS",
        "SUPPORTED": "SUPPORTS",
        "SUPPORTS": "SUPPORTS",
        "ENTAILS": "SUPPORTS",
        "ENTAILMENT": "SUPPORTS",
        "REFUTE": "REFUTES",
        "REFUTED": "REFUTES",
        "REFUTES": "REFUTES",
        "CONTRADICTS": "REFUTES",
        "CONTRADICTION": "REFUTES",
        "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
        "NOT_ENOUGH_INFORMATION": "NOT_ENOUGH_INFO",
        "NOT_ENOUGH": "NOT_ENOUGH_INFO",
        "NEI": "NOT_ENOUGH_INFO",
        "UNKNOWN": "NOT_ENOUGH_INFO",
        "UNCERTAIN": "NOT_ENOUGH_INFO",
    }
    return stance_map.get(stance, "NOT_ENOUGH_INFO")


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value or 0.0)
    except Exception:
        confidence = 0.0
    if confidence > 1.0:
        confidence /= 100.0
    return round(max(0.0, min(1.0, confidence)), 4)


def normalize_slm_output(raw_output: Dict, model: str) -> Dict:
    used_indices = raw_output.get("used_evidence_indices", [])
    if not isinstance(used_indices, list):
        used_indices = []

    clean_indices: List[int] = []
    for item in used_indices:
        try:
            index = int(item)
            if index > 0 and index not in clean_indices:
                clean_indices.append(index)
        except Exception:
            continue

    judgments: List[Dict[str, Any]] = []
    raw_judgments = raw_output.get("evidence_judgments", [])
    if isinstance(raw_judgments, list):
        for item in raw_judgments:
            if not isinstance(item, dict):
                continue
            try:
                evidence_index = int(item.get("evidence_index"))
            except Exception:
                continue
            if evidence_index <= 0:
                continue
            conflict_fields = item.get("conflict_fields", [])
            if not isinstance(conflict_fields, list):
                conflict_fields = []
            clean_conflict_fields = [clean_text(value) for value in conflict_fields if clean_text(value)]
            judgments.append(
                {
                    "evidence_index": evidence_index,
                    "stance": _normalize_stance(item.get("stance")),
                    "confidence": _normalize_confidence(item.get("confidence")),
                    "material_conflict": bool(item.get("material_conflict", False)),
                    "conflict_fields": clean_conflict_fields[:8],
                    "reason": clean_text(item.get("reason", "")),
                }
            )

    return {
        "enabled": True,
        "used": True,
        "provider": "lmstudio",
        "model": model,
        "stance": _normalize_stance(raw_output.get("stance")),
        "confidence": _normalize_confidence(raw_output.get("confidence")),
        "reason": clean_text(raw_output.get("reason", "")),
        "used_evidence_indices": clean_indices,
        "evidence_judgments": judgments,
        "error": None,
    }


def apply_semantic_evidence_judgments(
    evidence: List[Dict],
    slm_verification: Dict,
    *,
    min_confidence: float = 0.75,
    min_relevance: float = 0.50,
    demote_unverified_live_web: bool = True,
) -> List[Dict]:
    """Apply claim-relative semantic judgments to live-web evidence.

    The previous heuristic label is always retained for auditability. In SLM-assisted
    modes, directional live-web labels that the semantic verifier did not confidently
    confirm are conservatively demoted to NOT_ENOUGH_INFO rather than being allowed to
    create false SUPPORTS/REFUTES consensus. FEVER/LIAR2 labels are untouched.
    """
    refined = deepcopy(evidence)
    if not slm_verification or not slm_verification.get("used"):
        return refined

    judgments = slm_verification.get("evidence_judgments") or []
    if not isinstance(judgments, list):
        return refined

    judgment_by_index: Dict[int, Dict[str, Any]] = {}
    for judgment in judgments:
        if not isinstance(judgment, dict):
            continue
        try:
            idx = int(judgment.get("evidence_index")) - 1
        except Exception:
            continue
        if 0 <= idx < len(refined) and idx not in judgment_by_index:
            judgment_by_index[idx] = judgment

    for index, item in enumerate(refined):
        if str(item.get("source_type") or "").lower() != "live_web":
            continue

        try:
            relevance = float(item.get("relevance_score", 0.0) or 0.0)
        except Exception:
            relevance = 0.0

        previous = str(item.get("evidence_label") or "UNKNOWN").upper().strip().replace(" ", "_")
        judgment = judgment_by_index.get(index)

        # Record provenance even when the original heuristic label is retained/demoted.
        item["heuristic_evidence_label"] = previous

        if judgment is None:
            if demote_unverified_live_web and relevance >= float(min_relevance) and previous in {"SUPPORTS", "REFUTES"}:
                item["evidence_label"] = "NOT_ENOUGH_INFO"
                item["evidence_label_source"] = "semantic_unverified_demoted"
                item["semantic_evidence_note"] = (
                    "Directional live-web heuristic was not confirmed by the grounded semantic verifier."
                )
            continue

        stance = _normalize_stance(judgment.get("stance"))
        confidence = _normalize_confidence(judgment.get("confidence"))
        material_conflict = bool(judgment.get("material_conflict", False))
        conflict_fields = judgment.get("conflict_fields", [])
        if not isinstance(conflict_fields, list):
            conflict_fields = []
        conflict_fields = [clean_text(value) for value in conflict_fields if clean_text(value)][:8]
        reason = clean_text(judgment.get("reason", ""))

        item["slm_evidence_label"] = stance
        item["slm_evidence_confidence"] = confidence
        item["slm_material_conflict"] = material_conflict
        item["slm_conflict_fields"] = conflict_fields
        item["slm_evidence_reason"] = reason

        # A model response saying SUPPORTS while simultaneously identifying a material
        # conflict is internally inconsistent. Material conflict wins by definition.
        if material_conflict and stance == "SUPPORTS":
            stance = "REFUTES"
            item["slm_evidence_label"] = stance
            item["semantic_consistency_correction"] = "material_conflict_forced_refutes"

        if relevance < float(min_relevance):
            continue

        if confidence < float(min_confidence):
            # A weak semantic judgment must not leave a strong-looking lexical direction
            # in place. Be conservative for live web evidence.
            if demote_unverified_live_web and previous in {"SUPPORTS", "REFUTES"}:
                item["evidence_label"] = "NOT_ENOUGH_INFO"
                item["evidence_label_source"] = "semantic_low_confidence_demoted"
            continue

        item["evidence_label"] = stance
        item["evidence_label_source"] = "slm_semantic_verification"

    return refined


def normalize_lmstudio_base_url(base_url: str) -> str:
    base_url = clean_text(base_url).rstrip("/")
    if not base_url:
        base_url = "http://127.0.0.1:1234/v1"
    if base_url.endswith("/api/v1"):
        base_url = base_url[: -len("/api/v1")] + "/v1"
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def call_lmstudio(prompt: str, cfg: Dict[str, Any]) -> str:
    base_url = normalize_lmstudio_base_url(cfg["base_url"])
    url = f"{base_url}/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "system",
                "content": "/no_think You are a strict evidence-grounded verifier. Return JSON only. Do not use outside knowledge.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(cfg["temperature"]),
        "max_tokens": int(cfg["max_tokens"]),
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=int(cfg["timeout_seconds"]))
    response.raise_for_status()
    data = response.json()
    return clean_text(data["choices"][0]["message"]["content"])


def disabled_result(reason: str = "SLM verifier disabled or not applicable.") -> Dict:
    cfg = get_runtime_slm_config()
    return {
        "enabled": bool(cfg["enabled"]),
        "used": False,
        "provider": "lmstudio",
        "model": cfg["model"],
        "stance": "NOT_ENOUGH_INFO",
        "confidence": 0.0,
        "reason": reason,
        "used_evidence_indices": [],
        "evidence_judgments": [],
        "error": None,
    }


def failed_result(error: Exception) -> Dict:
    cfg = get_runtime_slm_config()
    return {
        "enabled": bool(cfg["enabled"]),
        "used": False,
        "provider": "lmstudio",
        "model": cfg["model"],
        "stance": "NOT_ENOUGH_INFO",
        "confidence": 0.0,
        "reason": "SLM verification failed; falling back to the deterministic evidence/risk layer.",
        "used_evidence_indices": [],
        "evidence_judgments": [],
        "error": str(error),
    }


def verify_claim_with_slm(
    claim: str,
    evidence: List[Dict],
    inference_mode: str = "slm_assisted_hybrid",
) -> Dict:
    cfg = get_runtime_slm_config()
    mode = clean_text(inference_mode).lower()

    if mode not in SLM_MODES:
        return disabled_result("Selected inference mode does not use SLM verification.")
    if not bool(cfg["enabled"]):
        return disabled_result("SLM verifier is disabled.")
    if not evidence:
        return disabled_result("No evidence available for SLM verification.")

    try:
        prompt = build_prompt(claim, evidence)
        raw_text = call_lmstudio(prompt, cfg)
        parsed = parse_json_from_text(raw_text)
        return normalize_slm_output(parsed, model=cfg["model"])
    except Exception as error:
        return failed_result(error)
