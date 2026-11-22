# CloudDash Retrieval Evaluation Results

Evaluated on **42 golden test queries** covering all 20 CloudDash knowledge base articles.

## Performance Comparison

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Vector Only** (`all-MiniLM-L6-v2`) | 57.14% | 66.67% | 71.43% | 0.6258 |
| **BM25 Only** (BM25Okapi) | 47.62% | 69.05% | 73.81% | 0.5896 |
| **Hybrid RRF** (Vector + BM25) | 52.38% | 73.81% | 83.33% | 0.6614 |
| **Hybrid + Cross-Encoder Rerank** (`ms-marco-MiniLM-L-6-v2`) | **59.52%** | **80.95%** | **85.71%** | **0.7166** |

## Key Insights
- **Cross-Encoder Lift**: Reranking improves Recall@1 from 52.38% to 59.52% and MRR from 0.6614 to 0.7166.
- **Hybrid Fusion (RRF)** combines dense semantic vector representation with exact sparse keyword matching, outperforming single-retriever baselines.
