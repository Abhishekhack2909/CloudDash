"""CLI entry point for running CloudDash evaluation suites.

Usage:
    python -m evals.run --suite retrieval
    python -m evals.run --suite e2e [--limit 10]
    python -m evals.run --suite all
"""

import argparse
import sys
from pathlib import Path

from evals.retrieval_eval import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser(description="Run CloudDash evaluation suites.")
    parser.add_argument(
        "--suite",
        choices=["retrieval", "e2e", "all"],
        default="all",
        help="Which evaluation suite to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases to evaluate (useful for rapid checks)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for results (default: evals/results)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge during e2e suite to reduce API tokens",
    )
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Compare retrieval results against evals/baseline_retrieval.json and fail if degraded",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current retrieval results as evals/baseline_retrieval.json",
    )

    args = parser.parse_args()

    results = {}

    if args.suite in ["retrieval", "all"]:
        print("\n" + "=" * 60)
        print("RUNNING RETRIEVAL EVALUATION SUITE")
        print("=" * 60)
        retrieval_summary = evaluate_retrieval(
            output_dir=args.output_dir,
            limit=args.limit,
        )
        results["retrieval"] = retrieval_summary

        print("\n--- Retrieval Results Table ---")
        metrics = retrieval_summary["metrics"]
        print(f"{'Configuration':<30} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR':<10}")
        print("-" * 78)
        for cfg, m in metrics.items():
            print(f"{cfg:<30} | {m['recall@1']:<10.3f} | {m['recall@3']:<10.3f} | {m['recall@5']:<10.3f} | {m['mrr']:<10.3f}")

        # Baseline management
        baseline_file = Path(__file__).parent / "baseline_retrieval.json"
        if args.save_baseline or not baseline_file.exists():
            import json
            with open(baseline_file, "w", encoding="utf-8") as f:
                json.dump(retrieval_summary["metrics"], f, indent=2)
            print(f"\n[INFO] Saved baseline metrics to {baseline_file}")

        if args.check_baseline:
            import json
            if not baseline_file.exists():
                print(f"[ERROR] Baseline file {baseline_file} does not exist.")
                sys.exit(1)

            with open(baseline_file, "r", encoding="utf-8") as f:
                baseline = json.load(f)

            # Compare hybrid_rerank (or best config)
            target_cfg = "hybrid_rerank"
            curr_r5 = metrics[target_cfg]["recall@5"]
            curr_mrr = metrics[target_cfg]["mrr"]
            base_r5 = baseline[target_cfg]["recall@5"]
            base_mrr = baseline[target_cfg]["mrr"]

            threshold = 0.05
            print(f"\n[BASELINE CHECK] {target_cfg}:")
            print(f"Recall@5: current={curr_r5:.3f}, baseline={base_r5:.3f} (Δ={curr_r5 - base_r5:+.3f})")
            print(f"MRR:      current={curr_mrr:.3f}, baseline={base_mrr:.3f} (Δ={curr_mrr - base_mrr:+.3f})")

            if (base_r5 - curr_r5) > threshold:
                print(f"[FAIL] Recall@5 dropped by more than {threshold:.2f} threshold!")
                sys.exit(1)
            if (base_mrr - curr_mrr) > threshold:
                print(f"[FAIL] MRR dropped by more than {threshold:.2f} threshold!")
                sys.exit(1)
            print("[PASS] Retrieval metrics meet or exceed baseline tolerance.")

    if args.suite in ["e2e", "all"]:
        print("\n" + "=" * 60)
        print("RUNNING END-TO-END MULTI-AGENT EVALUATION SUITE")
        print("=" * 60)
        from evals.e2e_eval import evaluate_e2e
        e2e_summary = evaluate_e2e(
            output_dir=args.output_dir,
            limit=args.limit,
            enable_judge=not args.no_judge,
        )
        results["e2e"] = e2e_summary

    print("\n[SUCCESS] Evaluation run finished.")


if __name__ == "__main__":
    main()
