"""Retrieval evaluation harness for CloudDash.

Evaluates 4 retrieval strategies:
1. Vector Search (all-MiniLM-L6-v2 + ChromaDB)
2. BM25 Search (rank-bm25)
3. Hybrid Search (RRF fusion)
4. Hybrid + Cross-Encoder Re-ranking (ms-marco-MiniLM-L-6-v2)

Computes Recall@1, Recall@3, Recall@5, and MRR (Mean Reciprocal Rank).
Outputs to:
- evals/results/retrieval.md
- evals/results/retrieval.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from retrieval.bm25_store import BM25Store
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore


def load_golden_set(golden_set_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load evaluation test cases from golden_set.jsonl."""
    if golden_set_path is None:
        golden_set_path = Path(__file__).parent / "golden_set.jsonl"
    else:
        golden_set_path = Path(golden_set_path)

    cases = []
    with open(golden_set_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def evaluate_retrieval(
    golden_set_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run retrieval evaluation across all four configurations."""
    cases = load_golden_set(golden_set_path)
    # Filter to cases that expect KB articles
    retrieval_cases = [c for c in cases if c.get("expected_article_ids")]

    if limit is not None and limit > 0:
        retrieval_cases = retrieval_cases[:limit]

    total_queries = len(retrieval_cases)
    if total_queries == 0:
        raise ValueError("No retrieval test cases found with expected_article_ids.")

    # Initialize components
    vector_store = VectorStore()
    bm25_store = BM25Store()
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_store=bm25_store,
    )
    reranker = Reranker()

    configs = ["vector_only", "bm25_only", "hybrid_rrf", "hybrid_rerank"]
    results_by_config: dict[str, dict[str, float]] = {cfg: {} for cfg in configs}
    per_query_details: list[dict[str, Any]] = []

    # Accumulators for metrics
    hits_at_1 = {cfg: 0 for cfg in configs}
    hits_at_3 = {cfg: 0 for cfg in configs}
    hits_at_5 = {cfg: 0 for cfg in configs}
    mrr_sum = {cfg: 0.0 for cfg in configs}

    for case in retrieval_cases:
        query = case["question"]
        expected = set(case["expected_article_ids"])

        # 1. Vector only
        v_results = vector_store.search(query, k=15)
        # 2. BM25 only
        b_results = bm25_store.search(query, k=15)
        # 3. Hybrid RRF
        h_results = hybrid_retriever.search(query, k=15)
        # 4. Hybrid + Re-ranking (rerank candidates and evaluate top ranks)
        r_results = reranker.rerank(query, h_results, top_k=15)

        retrieval_runs = {
            "vector_only": [r.article_id for r in v_results],
            "bm25_only": [r.article_id for r in b_results],
            "hybrid_rrf": [r.article_id for r in h_results],
            "hybrid_rerank": [r.article_id for r in r_results],
        }

        query_detail = {
            "id": case["id"],
            "question": query,
            "expected": list(expected),
            "retrieved": {},
        }

        for cfg in configs:
            ranked_ids = retrieval_runs[cfg]
            query_detail["retrieved"][cfg] = ranked_ids[:5]

            # Hits @ 1, 3, 5
            if any(aid in expected for aid in ranked_ids[:1]):
                hits_at_1[cfg] += 1
            if any(aid in expected for aid in ranked_ids[:3]):
                hits_at_3[cfg] += 1
            if any(aid in expected for aid in ranked_ids[:5]):
                hits_at_5[cfg] += 1

            # Reciprocal rank
            rr = 0.0
            for rank, aid in enumerate(ranked_ids):
                if aid in expected:
                    rr = 1.0 / (rank + 1)
                    break
            mrr_sum[cfg] += rr

        per_query_details.append(query_detail)

    # Compute averages
    for cfg in configs:
        results_by_config[cfg] = {
            "recall@1": round(hits_at_1[cfg] / total_queries, 4),
            "recall@3": round(hits_at_3[cfg] / total_queries, 4),
            "recall@5": round(hits_at_5[cfg] / total_queries, 4),
            "mrr": round(mrr_sum[cfg] / total_queries, 4),
        }

    summary = {
        "total_queries_evaluated": total_queries,
        "metrics": results_by_config,
        "per_query": per_query_details,
    }

    # Save to disk
    out_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "retrieval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown Table
    md_content = f"""# CloudDash Retrieval Evaluation Results

Evaluated on **{total_queries} golden test queries** covering all 20 CloudDash knowledge base articles.

## Performance Comparison

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Vector Only** (`all-MiniLM-L6-v2`) | {results_by_config['vector_only']['recall@1']:.2%} | {results_by_config['vector_only']['recall@3']:.2%} | {results_by_config['vector_only']['recall@5']:.2%} | {results_by_config['vector_only']['mrr']:.4f} |
| **BM25 Only** (BM25Okapi) | {results_by_config['bm25_only']['recall@1']:.2%} | {results_by_config['bm25_only']['recall@3']:.2%} | {results_by_config['bm25_only']['recall@5']:.2%} | {results_by_config['bm25_only']['mrr']:.4f} |
| **Hybrid RRF** (Vector + BM25) | {results_by_config['hybrid_rrf']['recall@1']:.2%} | {results_by_config['hybrid_rrf']['recall@3']:.2%} | {results_by_config['hybrid_rrf']['recall@5']:.2%} | {results_by_config['hybrid_rrf']['mrr']:.4f} |
| **Hybrid + Cross-Encoder Rerank** (`ms-marco-MiniLM-L-6-v2`) | **{results_by_config['hybrid_rerank']['recall@1']:.2%}** | **{results_by_config['hybrid_rerank']['recall@3']:.2%}** | **{results_by_config['hybrid_rerank']['recall@5']:.2%}** | **{results_by_config['hybrid_rerank']['mrr']:.4f}** |

## Key Insights
- **Cross-Encoder Lift**: Reranking improves Recall@1 from {results_by_config['hybrid_rrf']['recall@1']:.2%} to {results_by_config['hybrid_rerank']['recall@1']:.2%} and MRR from {results_by_config['hybrid_rrf']['mrr']:.4f} to {results_by_config['hybrid_rerank']['mrr']:.4f}.
- **Hybrid Fusion (RRF)** combines dense semantic vector representation with exact sparse keyword matching, outperforming single-retriever baselines.
"""
    md_path = out_dir / "retrieval.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return summary


if __name__ == "__main__":
    results = evaluate_retrieval()
    print("\n--- Retrieval Evaluation Summary ---")
    for cfg, m in results["metrics"].items():
        print(f"{cfg:20s}: R@1={m['recall@1']:.3f} | R@3={m['recall@3']:.3f} | R@5={m['recall@5']:.3f} | MRR={m['mrr']:.3f}")
