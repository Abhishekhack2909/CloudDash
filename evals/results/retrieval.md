# CloudDash Retrieval Evaluation Results

Evaluated on **42 golden test queries** covering all 20 CloudDash knowledge base articles.

## Performance Comparison

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Vector Only** (`all-MiniLM-L6-v2`) | 78.57% | 97.62% | 97.62% | 0.8710 |
| **BM25 Only** (BM25Okapi) | 61.90% | 78.57% | 85.71% | 0.7175 |
| **Hybrid RRF** (Vector + BM25) | 73.81% | 90.48% | 95.24% | 0.8288 |
| **Hybrid + Cross-Encoder Rerank** (`ms-marco-MiniLM-L-6-v2`) | **71.43%** | **90.48%** | **97.62%** | **0.8252** |

## Key Insights
- **Cross-Encoder Lift**: Reranking improves Recall@1 from 73.81% to 71.43% and MRR from 0.8288 to 0.8252.
- **Hybrid Fusion (RRF)** combines dense semantic vector representation with exact sparse keyword matching, outperforming single-retriever baselines.
