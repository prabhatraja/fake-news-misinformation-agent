from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "evidence_policy.yaml"


class EvidencePolicyError(RuntimeError):
    pass


def _as_dict(value: Any, field_name: str) -> Dict:
    if not isinstance(value, dict):
        raise EvidencePolicyError(f"Expected '{field_name}' to be a dictionary.")
    return value


def _as_list(value: Any, field_name: str) -> List:
    if not isinstance(value, list):
        raise EvidencePolicyError(f"Expected '{field_name}' to be a list.")
    return value


def _to_string_set(value: Any, field_name: str) -> Set[str]:
    items = _as_list(value, field_name)
    return {str(item).strip().lower() for item in items if str(item).strip()}


def _to_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except Exception as error:
        raise EvidencePolicyError(f"Expected '{field_name}' to be numeric.") from error


@lru_cache(maxsize=1)
def load_evidence_policy(policy_path: str = str(DEFAULT_POLICY_PATH)) -> Dict[str, Any]:
    path = Path(policy_path)

    if not path.exists():
        raise EvidencePolicyError(f"Evidence policy file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        policy = yaml.safe_load(file)

    policy = _as_dict(policy, "policy")

    source_trust = _as_dict(policy.get("source_trust"), "source_trust")
    stance_policy = _as_dict(policy.get("stance_policy"), "stance_policy")

    normalized_policy: Dict[str, Any] = {
        "version": str(policy.get("version", "unknown")),
        "source_trust": {
            "default_trust": _to_float(
                source_trust.get("default_trust", 0.60),
                "source_trust.default_trust",
            ),
            "unknown_domain_trust": _to_float(
                source_trust.get("unknown_domain_trust", 0.35),
                "source_trust.unknown_domain_trust",
            ),
            "domain_suffix_scores": {
                str(suffix).lower(): _to_float(score, f"source_trust.domain_suffix_scores.{suffix}")
                for suffix, score in _as_dict(
                    source_trust.get("domain_suffix_scores", {}),
                    "source_trust.domain_suffix_scores",
                ).items()
            },
            "trusted_fact_check_domains": {
                "trust_score": _to_float(
                    _as_dict(
                        source_trust.get("trusted_fact_check_domains", {}),
                        "source_trust.trusted_fact_check_domains",
                    ).get("trust_score", 0.85),
                    "source_trust.trusted_fact_check_domains.trust_score",
                ),
                "domains": _to_string_set(
                    _as_dict(
                        source_trust.get("trusted_fact_check_domains", {}),
                        "source_trust.trusted_fact_check_domains",
                    ).get("domains", []),
                    "source_trust.trusted_fact_check_domains.domains",
                ),
            },
            "official_domains": {
                "trust_score": _to_float(
                    _as_dict(
                        source_trust.get("official_domains", {}),
                        "source_trust.official_domains",
                    ).get("trust_score", 0.90),
                    "source_trust.official_domains.trust_score",
                ),
                "domains": _to_string_set(
                    _as_dict(
                        source_trust.get("official_domains", {}),
                        "source_trust.official_domains",
                    ).get("domains", []),
                    "source_trust.official_domains.domains",
                ),
            },
            "authority_domains": {
                "trust_score": _to_float(
                    _as_dict(
                        source_trust.get("authority_domains", {}),
                        "source_trust.authority_domains",
                    ).get("trust_score", 0.78),
                    "source_trust.authority_domains.trust_score",
                ),
                "domains": _to_string_set(
                    _as_dict(
                        source_trust.get("authority_domains", {}),
                        "source_trust.authority_domains",
                    ).get("domains", []),
                    "source_trust.authority_domains.domains",
                ),
            },
            "low_trust_domains": {
                "trust_score": _to_float(
                    _as_dict(
                        source_trust.get("low_trust_domains", {}),
                        "source_trust.low_trust_domains",
                    ).get("trust_score", 0.25),
                    "source_trust.low_trust_domains.trust_score",
                ),
                "domains": _to_string_set(
                    _as_dict(
                        source_trust.get("low_trust_domains", {}),
                        "source_trust.low_trust_domains",
                    ).get("domains", []),
                    "source_trust.low_trust_domains.domains",
                ),
            },
        },
        "stance_policy": {
            "minimum_overlap_count": int(
                stance_policy.get("minimum_overlap_count", 2)
            ),
            "minimum_overlap_ratio": _to_float(
                stance_policy.get("minimum_overlap_ratio", 0.40),
                "stance_policy.minimum_overlap_ratio",
            ),
            "assertion_overlap_count": int(
                stance_policy.get("assertion_overlap_count", 2)
            ),
            "assertion_overlap_ratio": _to_float(
                stance_policy.get("assertion_overlap_ratio", 0.60),
                "stance_policy.assertion_overlap_ratio",
            ),
            "stopwords": _to_string_set(
                stance_policy.get("stopwords", []),
                "stance_policy.stopwords",
            ),
            "negation_cues": _to_string_set(
                stance_policy.get("negation_cues", []),
                "stance_policy.negation_cues",
            ),
            "neutral_verdict_cues": _to_string_set(
                stance_policy.get("neutral_verdict_cues", []),
                "stance_policy.neutral_verdict_cues",
            ),
            "refute_verdict_cues": _to_string_set(
                stance_policy.get("refute_verdict_cues", []),
                "stance_policy.refute_verdict_cues",
            ),
            "support_verdict_cues": _to_string_set(
                stance_policy.get("support_verdict_cues", []),
                "stance_policy.support_verdict_cues",
            ),
            "explicit_refute_patterns": _to_string_set(
                stance_policy.get("explicit_refute_patterns", []),
                "stance_policy.explicit_refute_patterns",
            ),
            "explicit_support_patterns": _to_string_set(
                stance_policy.get("explicit_support_patterns", []),
                "stance_policy.explicit_support_patterns",
            ),
            "assertion_cues": _to_string_set(
                stance_policy.get("assertion_cues", []),
                "stance_policy.assertion_cues",
            ),
            "relation_groups": {
                str(group_name): {
                    "query": _to_string_set(
                        _as_dict(group_value, f"stance_policy.relation_groups.{group_name}").get("query", []),
                        f"stance_policy.relation_groups.{group_name}.query",
                    ),
                    "evidence": _to_string_set(
                        _as_dict(group_value, f"stance_policy.relation_groups.{group_name}").get("evidence", []),
                        f"stance_policy.relation_groups.{group_name}.evidence",
                    ),
                }
                for group_name, group_value in _as_dict(
                    stance_policy.get("relation_groups", {}),
                    "stance_policy.relation_groups",
                ).items()
            },
        },
    }

    return normalized_policy


def reload_evidence_policy() -> Dict[str, Any]:
    load_evidence_policy.cache_clear()
    return load_evidence_policy()