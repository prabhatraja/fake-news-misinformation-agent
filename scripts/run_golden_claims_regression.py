"""Run a small live factual regression suite against /api/v1/analyze.

Requires FastAPI to be running. For the strongest check, run with LM Studio and
hybrid web evidence enabled. Live web availability can change, so failures are
reported for review rather than treated as a training metric.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import requests


CASES = [
    ("taj_true", "The Taj Mahal is located in Agra, Uttar Pradesh.", {"Likely Real"}),
    ("taj_false", "The Taj Mahal is located in Jaipur, Rajasthan.", {"Likely Fake"}),
    ("chandrayaan_true", "Chandrayaan-3 successfully soft-landed on the Moon in August 2023.", {"Likely Real"}),
    ("chandrayaan_false", "Chandrayaan-3 landed on Mars in August 2023.", {"Likely Fake"}),
    ("australia_true", "Canberra is the capital of Australia.", {"Likely Real"}),
    ("australia_false", "Sydney is the capital of Australia.", {"Likely Fake"}),
    ("apollo_true", "Apollo 11 landed humans on the Moon in 1969.", {"Likely Real"}),
    ("apollo_false", "Apollo 11 landed humans on Mars in 1969.", {"Likely Fake"}),
    ("health_false", "COVID-19 vaccines contain microchips used to track people.", {"Likely Fake"}),
    ("football_false", "India won the FIFA World Cup in 2022.", {"Likely Fake"}),
    ("everest_true", "Mount Everest is the highest mountain above sea level.", {"Likely Real"}),
    ("weak_claim", "Breaking shocking secret exposed by anonymous doctors and hidden media sources.", {"Uncertain", "Misleading"}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", default="slm_assisted_hybrid")
    args = parser.parse_args()

    results = []
    passed = 0
    coherent = 0
    for name, claim, expected in CASES:
        payload = {
            "input": claim,
            "source_type": "auto",
            "inference_mode": args.mode,
            "max_claims": 1,
            "enable_temporal_analysis": True,
            "twitter_manual_text": "",
            "twitter_profile_post_limit": 5,
        }
        try:
            response = requests.post(f"{args.base_url}/api/v1/analyze", json=payload, timeout=240)
            response.raise_for_status()
            body = response.json()
            label = body.get("overall_label", "")
            ok = label in expected
            if ok:
                passed += 1
            claims = body.get("claims_analyzed") or []
            claim_result = (claims[0].get("result") or {}) if claims else {}
            summary = claim_result.get("assessment_summary") or {}
            evidence_verdict = summary.get("evidence_verdict")
            slm_stance = (claim_result.get("slm_verification") or {}).get("stance")
            coherence_ok = True
            coherence_reasons = []
            if label == "Likely Real" and evidence_verdict == "Contradicts Claim":
                coherence_ok = False
                coherence_reasons.append("final_real_but_evidence_refutes")
            if label == "Likely Fake" and evidence_verdict == "Supports Claim":
                coherence_ok = False
                coherence_reasons.append("final_fake_but_evidence_supports")
            if label == "Likely Real" and slm_stance == "REFUTES":
                coherence_ok = False
                coherence_reasons.append("final_real_but_slm_refutes")
            if label == "Likely Fake" and slm_stance == "SUPPORTS":
                coherence_ok = False
                coherence_reasons.append("final_fake_but_slm_supports")
            if coherence_ok:
                coherent += 1

            row = {
                "name": name,
                "claim": claim,
                "expected": sorted(expected),
                "actual": label,
                "passed": ok,
                "risk": body.get("overall_risk"),
                "evidence_verdict": evidence_verdict,
                "evidence_quality": summary.get("evidence_quality"),
                "slm_stance": slm_stance,
                "coherence_ok": coherence_ok,
                "coherence_reasons": coherence_reasons,
                "evidence_labels": [item.get("evidence_label") for item in (claim_result.get("evidence") or [])],
                "why": summary.get("why"),
            }
        except Exception as error:
            row = {
                "name": name,
                "claim": claim,
                "expected": sorted(expected),
                "actual": "ERROR",
                "passed": False,
                "error": str(error),
            }
        results.append(row)
        status = "PASS" if row.get("passed") and row.get("coherence_ok", True) else "REVIEW"
        print(
            f"[{status}] {name}: {row.get('actual')} | expected {', '.join(row['expected'])} "
            f"| evidence={row.get('evidence_verdict')} | slm={row.get('slm_stance')} "
            f"| coherent={row.get('coherence_ok', True)}"
        )

    total = len(CASES)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "mode": args.mode,
        "total": total,
        "passed": passed,
        "review": total - passed,
        "directional_accuracy": round(passed / total, 4),
        "coherent": coherent,
        "coherence_rate": round(coherent / total, 4),
        "results": results,
    }
    out_dir = Path("reports") / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"golden_claims_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGolden claims: {passed}/{total} matched expected direction")
    print(f"Internal coherence: {coherent}/{total}")
    print(f"Report written: {out_path}")


if __name__ == "__main__":
    main()
