# CloudDash End-to-End System Evaluation Results

Evaluated across **3 golden test cases** through the live LangGraph multi-agent pipeline.

## Summary Scores

| Metric | Score | Details |
| :--- | :---: | :--- |
| **Routing Accuracy** | **100.00%** | 3/3 correctly classified to triage target |
| **Escalation Precision** | **0.00%** | TP=0, FP=0 |
| **Escalation Recall** | **0.00%** | TP=0, FN=0 |
| **Escalation F1** | **0.00%** | Harmonic mean of escalation precision and recall |
| **Citation Hit Rate** | **100.00%** | Percentage of KB-supported queries citing expected article |
| **Citation Recall** | **100.00%** | Mean recall of expected KB article IDs |
| **LLM Judge Groundedness** | **4.67/5.0** | Factual alignment with retrieved KB context (Pydantic validated) |
| **LLM Judge Answer Relevance** | **5.0/5.0** | Directness and completeness of customer response |
