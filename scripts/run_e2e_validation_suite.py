"""Consolidated end-to-end API validation suite.

Prerequisites:
1. Start LM Studio with Qwen loaded and Enable Thinking OFF.
2. Start backend:
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
3. Run:
   python scripts/run_e2e_validation_suite.py

Use --skip-external to skip Wikipedia/YouTube/Twitter network-dependent cases.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "e2e"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

PREDICT_MODES = [
    "traditional_lr",
    "traditional_svm",
    "distilbert",
    "distilbert_local_rag",
    "distilbert_hybrid",
    "slm_assisted_local",
    "slm_assisted_hybrid",
]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


class E2ERunner:
    def __init__(self, base_url: str, skip_external: bool = False, timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.skip_external = skip_external
        self.timeout = timeout
        self.results: List[Dict[str, Any]] = []

    def record(
        self,
        name: str,
        passed: bool,
        details: Optional[Dict[str, Any]] = None,
        critical: bool = True,
    ) -> None:
        item = {
            "name": name,
            "passed": bool(passed),
            "critical": bool(critical),
            "details": json_safe(details or {}),
        }
        self.results.append(item)
        status = "PASS" if passed else ("WARN" if not critical else "FAIL")
        print(f"[{status}] {name}")
        if details:
            print(json.dumps(json_safe(details), indent=2, ensure_ascii=False)[:3000])

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def test_health(self) -> None:
        name = "health_check"
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=30)
            data = response.json()
            self.record(name, response.status_code == 200 and data.get("status") == "ok", data)
        except Exception as exc:
            self.record(name, False, {"error": str(exc)})

    def test_predict_all_modes(self) -> None:
        name = "predict_all_7_inference_modes"
        outputs: List[Dict[str, Any]] = []
        passed = True
        try:
            for mode in PREDICT_MODES:
                payload = {
                    "text": "The Eiffel Tower is located in Paris.",
                    "input_type": "claim",
                    "inference_mode": mode,
                }
                data = self.post_json("/api/v1/predict", payload)
                slm = data.get("slm_verification") or {}
                outputs.append(
                    {
                        "mode": mode,
                        "label": data.get("predicted_label"),
                        "risk": data.get("risk_label"),
                        "model_used": data.get("model_used"),
                        "slm_used": slm.get("used"),
                        "slm_stance": slm.get("stance"),
                        "slm_error": slm.get("error"),
                    }
                )
                if mode.startswith("slm_assisted"):
                    if not slm.get("used") or slm.get("stance") != "SUPPORTS" or slm.get("error"):
                        passed = False
            self.record(name, passed and len(outputs) == len(PREDICT_MODES), {"outputs": outputs})
        except Exception as exc:
            self.record(name, False, {"error": str(exc), "outputs": outputs})

    def test_analyze_plain_text(self) -> None:
        name = "analyze_plain_text_claim"
        try:
            data = self.post_json(
                "/api/v1/analyze",
                {
                    "input": "The Eiffel Tower is located in Paris.",
                    "source_type": "auto",
                    "inference_mode": "slm_assisted_hybrid",
                    "max_claims": 3,
                    "enable_temporal_analysis": True,
                },
            )
            passed = (
                data.get("input_kind") == "plain_text"
                and data.get("overall_label") == "Likely Real"
                and data.get("overall_risk") in {"Very Low Risk", "Low Risk", "Medium Risk"}
                and len(data.get("claims_analyzed", [])) >= 1
            )
            self.record(
                name,
                passed,
                {
                    "input_kind": data.get("input_kind"),
                    "overall_label": data.get("overall_label"),
                    "overall_risk": data.get("overall_risk"),
                    "claims": len(data.get("claims_analyzed", [])),
                },
            )
        except Exception as exc:
            self.record(name, False, {"error": str(exc)})

    def test_wikipedia_url(self) -> None:
        name = "analyze_url_wikipedia_regression"
        if self.skip_external:
            self.record(name, True, {"skipped": "--skip-external used"}, critical=False)
            return
        try:
            data = self.post_json(
                "/api/v1/analyze",
                {
                    "input": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                    "source_type": "auto",
                    "inference_mode": "slm_assisted_hybrid",
                    "max_claims": 3,
                    "enable_temporal_analysis": True,
                },
            )
            bad_overall = data.get("overall_label") == "Likely Fake" and data.get("overall_risk") == "High Risk"
            passed = (
                data.get("input_kind") == "url"
                and data.get("extraction", {}).get("status") == "success"
                and len(data.get("claims_analyzed", [])) >= 2
                and not bad_overall
            )
            claims = [
                {
                    "claim": c.get("claim_text"),
                    "label": (c.get("result") or {}).get("predicted_label"),
                    "risk": (c.get("result") or {}).get("risk_label"),
                    "focus": (c.get("verification_context") or {}).get("query_focus"),
                }
                for c in data.get("claims_analyzed", [])
            ]
            self.record(
                name,
                passed,
                {
                    "overall_label": data.get("overall_label"),
                    "overall_risk": data.get("overall_risk"),
                    "limitations": data.get("limitations"),
                    "claims": claims,
                },
            )
        except Exception as exc:
            self.record(name, False, {"error": str(exc)})

    def test_youtube_url(self) -> None:
        name = "analyze_youtube_url_best_effort"
        if self.skip_external:
            self.record(name, True, {"skipped": "--skip-external used"}, critical=False)
            return
        try:
            data = self.post_json(
                "/api/v1/analyze",
                {
                    "input": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "source_type": "auto",
                    "inference_mode": "slm_assisted_hybrid",
                    "max_claims": 3,
                    "enable_temporal_analysis": True,
                },
            )
            metadata = data.get("source_metadata", {}).get("metadata", {})
            passed = data.get("input_kind") == "youtube_url" and data.get("extraction", {}).get("status") in {
                "success",
                "partial",
                "failed",
            }
            # YouTube transcript access can be blocked; for E2E, no crash + clear metadata/limitations is acceptable.
            if data.get("extraction", {}).get("status") == "failed" and not data.get("limitations"):
                passed = False
            self.record(
                name,
                passed,
                {
                    "extraction": data.get("extraction", {}).get("status"),
                    "title": data.get("source_metadata", {}).get("title"),
                    "transcript_available": metadata.get("transcript_available"),
                    "claims": len(data.get("claims_analyzed", [])),
                    "limitations": data.get("limitations"),
                },
                critical=False,
            )
        except Exception as exc:
            self.record(name, False, {"error": str(exc)}, critical=False)

    def test_twitter_detection(self) -> None:
        name = "analyze_twitter_x_detection_best_effort"
        if self.skip_external:
            self.record(name, True, {"skipped": "--skip-external used"}, critical=False)
            return
        try:
            data = self.post_json(
                "/api/v1/analyze",
                {
                    "input": "https://x.com/isro/status/1694545322251571687",
                    "source_type": "auto",
                    "inference_mode": "slm_assisted_hybrid",
                    "max_claims": 3,
                    "enable_temporal_analysis": True,
                },
            )
            extraction = data.get("extraction", {}) or {}
            metadata = extraction.get("metadata", {}) or {}
            status = extraction.get("status")
            passed = (
                data.get("input_kind") == "twitter_url"
                and status in {"success", "partial", "failed"}
                and (status != "success" or metadata.get("identity_verified") is True)
            )
            self.record(
                name,
                passed,
                {
                    "input_kind": data.get("input_kind"),
                    "extraction": extraction.get("status"),
                    "identity_verified": metadata.get("identity_verified"),
                    "extraction_quality": metadata.get("extraction_quality"),
                    "requested_status_id": metadata.get("requested_status_id"),
                    "resolved_status_id": metadata.get("resolved_status_id"),
                    "limitations": data.get("limitations"),
                },
                critical=False,
            )
        except Exception as exc:
            self.record(name, False, {"error": str(exc)}, critical=False)

    def test_recirculated_media_claim(self) -> None:
        name = "analyze_recirculated_media_claim"
        try:
            data = self.post_json(
                "/api/v1/analyze",
                {
                    "input": "A video shared today claims to show CAA protests in Delhi in 2020.",
                    "source_type": "auto",
                    "inference_mode": "slm_assisted_hybrid",
                    "max_claims": 3,
                    "enable_temporal_analysis": True,
                },
            )
            temporal = data.get("temporal_context") or {}
            contexts = [c.get("verification_context") or {} for c in data.get("claims_analyzed", [])]
            focus_values = [c.get("query_focus") for c in contexts]
            passed = (
                temporal.get("recirculation_risk") is True
                and any(f in {"recirculated_media", "historical_event"} for f in focus_values)
                and data.get("overall_label") != "Likely Real"
            )
            self.record(
                name,
                passed,
                {
                    "overall_label": data.get("overall_label"),
                    "overall_risk": data.get("overall_risk"),
                    "temporal": temporal,
                    "focus_values": focus_values,
                    "limitations": data.get("limitations"),
                },
            )
        except Exception as exc:
            self.record(name, False, {"error": str(exc)})

    def test_mixed_file_upload(self) -> None:
        name = "analyze_file_mixed_claim_context_isolation"
        text = (
            "The Eiffel Tower is located in Paris. "
            "The Eiffel Tower was constructed between 1887 and 1889. "
            "A viral video claims Iran tested a nuclear bomb last week."
        )
        try:
            files = {"file": ("e2e_mixed_claims.txt", text.encode("utf-8"), "text/plain")}
            form = {
                "source_type": "auto",
                "inference_mode": "slm_assisted_hybrid",
                "max_claims": "5",
                "enable_temporal_analysis": "true",
            }
            response = requests.post(
                f"{self.base_url}/api/v1/analyze-file",
                files=files,
                data=form,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            claims = data.get("claims_analyzed", [])
            summaries = []
            for claim in claims:
                result = claim.get("result") or {}
                slm = result.get("slm_verification") or {}
                context = claim.get("verification_context") or {}
                summaries.append(
                    {
                        "claim": claim.get("claim_text"),
                        "source_text": claim.get("source_text"),
                        "verification_text": claim.get("verification_text"),
                        "verification_target": claim.get("verification_target"),
                        "representation_fallback": claim.get("representation_fallback"),
                        "label": result.get("predicted_label"),
                        "risk": result.get("risk_label"),
                        "slm_stance": slm.get("stance"),
                        "focus": context.get("query_focus"),
                        "retrieval_query": context.get("retrieval_query_text"),
                    }
                )
            eiffel_location = next((s for s in summaries if "located in Paris" in str(s.get("claim"))), {})
            iran_video = next((s for s in summaries if "Iran" in str(s.get("claim"))), {})
            if self.skip_external:
                # Local-only mode intentionally has no fresh web evidence. For a time-sensitive
                # extracted proposition, safe uncertainty is valid; a grounded refutation is also valid
                # if the local corpus happens to contain enough evidence. Unsupported Likely Real is not.
                iran_outcome_ok = (
                    (
                        iran_video.get("label") == "Uncertain"
                        and iran_video.get("slm_stance") in {None, "NOT_ENOUGH_INFO"}
                    )
                    or (
                        iran_video.get("label") == "Likely Fake"
                        and iran_video.get("slm_stance") == "REFUTES"
                    )
                )
            else:
                # Full hybrid validation requires current external evidence to refute this fixture.
                iran_outcome_ok = (
                    iran_video.get("label") == "Likely Fake"
                    and iran_video.get("slm_stance") == "REFUTES"
                )

            passed = (
                data.get("input_kind") == "text_file_upload"
                and len(claims) >= 3
                and eiffel_location.get("label") == "Likely Real"
                and iran_video.get("verification_target") == "embedded_proposition"
                and iran_video.get("representation_fallback") is False
                and iran_outcome_ok
                and iran_video.get("focus") == "recirculated_media"
                and str(iran_video.get("verification_text") or "") in str(iran_video.get("retrieval_query") or "")
            )
            self.record(
                name,
                passed,
                {
                    "overall_label": data.get("overall_label"),
                    "overall_risk": data.get("overall_risk"),
                    "claims": summaries,
                    "limitations": data.get("limitations"),
                },
            )
        except Exception as exc:
            self.record(name, False, {"error": str(exc)})

    def run(self) -> int:
        print("\n=== E2E Validation Suite ===")
        print(f"Base URL: {self.base_url}")
        print(f"Skip external: {self.skip_external}")
        print("============================\n")

        self.test_health()
        self.test_predict_all_modes()
        self.test_analyze_plain_text()
        self.test_wikipedia_url()
        self.test_youtube_url()
        self.test_twitter_detection()
        self.test_recirculated_media_claim()
        self.test_mixed_file_upload()

        critical_failures = [r for r in self.results if r.get("critical") and not r.get("passed")]
        warnings = [r for r in self.results if not r.get("critical") and not r.get("passed")]
        passed_count = sum(1 for r in self.results if r.get("passed"))

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "base_url": self.base_url,
            "skip_external": self.skip_external,
            "total_cases": len(self.results),
            "passed_cases": passed_count,
            "critical_failures": len(critical_failures),
            "warnings": len(warnings),
            "results": self.results,
        }

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORT_DIR / f"e2e_validation_{now_stamp()}.json"
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n=== E2E Summary ===")
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
        print(f"Report written: {output_path}")

        if critical_failures:
            print("\nCritical failures:")
            for failure in critical_failures:
                print("-", failure.get("name"))
            return 1
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    runner = E2ERunner(base_url=args.base_url, skip_external=args.skip_external, timeout=args.timeout)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
