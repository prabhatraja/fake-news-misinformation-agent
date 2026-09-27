import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests


API_URL = "http://127.0.0.1:8000/api/v1/predict"
OUTPUT_PATH = Path("reports/metrics/slm_latency_benchmark_outputs.json")

TEST_CASES = [
    "The Eiffel Tower is located in Paris.",
    "The Eiffel Tower is located in Berlin.",
    "New Delhi is the capital of India.",
    "Mumbai is the capital of India.",
    "COVID-19 vaccines contain microchips used to track people.",
]

RUNS_PER_CASE = 3


def run_once(text: str):
    payload = {
        "text": text,
        "input_type": "claim",
        "inference_mode": "slm_assisted_hybrid",
    }

    start = time.perf_counter()
    response = requests.post(API_URL, json=payload, timeout=240)
    elapsed = time.perf_counter() - start

    response.raise_for_status()
    result = response.json()

    return {
        "latency_seconds": round(elapsed, 3),
        "label": result.get("predicted_label"),
        "risk_label": result.get("risk_label"),
        "model_used": result.get("model_used"),
        "slm_model": (result.get("slm_verification") or {}).get("model"),
        "slm_used": (result.get("slm_verification") or {}).get("used"),
        "slm_stance": (result.get("slm_verification") or {}).get("stance"),
        "slm_confidence": (result.get("slm_verification") or {}).get("confidence"),
        "slm_error": (result.get("slm_verification") or {}).get("error"),
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    outputs = []

    for text in TEST_CASES:
        print(f"Benchmarking: {text}")

        case_runs = []

        for run_id in range(1, RUNS_PER_CASE + 1):
            output = run_once(text)
            output["run_id"] = run_id
            case_runs.append(output)

            print(
                f"  run={run_id} "
                f"latency={output['latency_seconds']}s "
                f"label={output['label']} "
                f"risk={output['risk_label']} "
                f"stance={output['slm_stance']}"
            )

        latencies = [item["latency_seconds"] for item in case_runs]

        outputs.append(
            {
                "input_text": text,
                "runs": case_runs,
                "latency_summary": {
                    "avg_seconds": round(statistics.mean(latencies), 3),
                    "min_seconds": round(min(latencies), 3),
                    "max_seconds": round(max(latencies), 3),
                },
            }
        )

        print()

    all_latencies = [
        run["latency_seconds"]
        for case in outputs
        for run in case["runs"]
    ]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "api_url": API_URL,
        "test_type": "slm_assisted_hybrid_latency_benchmark",
        "runs_per_case": RUNS_PER_CASE,
        "total_requests": len(all_latencies),
        "overall_latency": {
            "avg_seconds": round(statistics.mean(all_latencies), 3),
            "min_seconds": round(min(all_latencies), 3),
            "max_seconds": round(max(all_latencies), 3),
        },
        "cases": outputs,
    }

    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"Total requests: {len(all_latencies)}")
    print(f"Average latency: {summary['overall_latency']['avg_seconds']}s")
    print(f"Min latency: {summary['overall_latency']['min_seconds']}s")
    print(f"Max latency: {summary['overall_latency']['max_seconds']}s")
    print(f"Saved to: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
