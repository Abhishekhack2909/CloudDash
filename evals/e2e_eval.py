"""End-to-end evaluation harness for the full CloudDash multi-agent system.

Scores:
(i) Routing accuracy (triage routing decision vs expected agent)
(ii) Escalation precision, recall, and F1
(iii) Citation correctness (overlap of cited KB IDs with expected article IDs)
(iv) LLM-as-judge groundedness (1-5) and answer relevance (1-5) using strict Pydantic JSON validation.

Outputs:
- evals/results/e2e.md
- evals/results/e2e.json
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from agents.orchestrator import Orchestrator
from config.llm import (
    DEFAULT_GROQ_MODEL,
    FallbackChatGroq,
    get_fallback_models,
    groq_model_options,
    load_agent_config,
)


class JudgeEvaluation(BaseModel):
    """Pydantic schema for LLM-as-judge evaluation."""

    groundedness_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="1-5 score: Is the response fully grounded in the provided KB context? 5=Fully grounded/transparent when no info, 1=Completely hallucinated.",
    )
    groundedness_reasoning: str = Field(..., description="Brief explanation of the groundedness score.")
    relevance_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="1-5 score: How directly and accurately does the response address the user's inquiry? 5=Direct, helpful, 1=Completely unhelpful.",
    )
    relevance_reasoning: str = Field(..., description="Brief explanation of the relevance score.")


def create_judge_llm() -> FallbackChatGroq:
    """Create a FallbackChatGroq client for the LLM judge."""
    configured_model = load_agent_config().get("llm", {}).get("model", DEFAULT_GROQ_MODEL)
    primary_opts = groq_model_options(configured_model)
    primary_client = ChatGroq(
        **primary_opts,
        temperature=0.0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    return FallbackChatGroq(
        primary_client=primary_client,
        primary_model=primary_opts.get("model", DEFAULT_GROQ_MODEL),
        fallback_models=get_fallback_models(),
        temperature=0.0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        chat_groq_cls=ChatGroq,
    )


def run_llm_judge(
    judge_llm: FallbackChatGroq,
    question: str,
    response: str,
    kb_context: str,
    max_retries: int = 3,
) -> JudgeEvaluation:
    """Call LLM judge with rubric and parse JSON output via Pydantic."""
    prompt = f"""You are an expert AI evaluator assessing an enterprise customer support agent.
Evaluate the agent's response to the user's question based on the provided Knowledge Base context.

RUBRIC:
1. Groundedness (1-5):
   5 - Completely grounded in the KB context. No hallucinated claims. If the KB had no info, clearly and transparently acknowledged it.
   4 - Mostly grounded; minor minor stylistic elaboration without contradictory facts.
   3 - Partially grounded, but introduces 1-2 unverified technical assumptions.
   2 - Significant unsupported claims or false procedures not found in the KB.
   1 - Completely fabricated or contradictory to the KB facts.

2. Answer Relevance (1-5):
   5 - Directly answers the user's inquiry with precision and practical instructions.
   4 - Helpful and mostly relevant, but includes minor unnecessary details.
   3 - Tangentially relevant or answers only half of the question.
   2 - Mostly misses the core user question or misunderstands intent.
   1 - Completely irrelevant, unhelpful, or garbled response.

USER QUESTION:
{question}

RETRIEVED KB CONTEXT:
{kb_context if kb_context else "None provided (unsupported/negative inquiry)"}

AGENT RESPONSE:
{response}

Respond ONLY with valid JSON in this exact structure:
{{
  "groundedness_score": <1-5 integer>,
  "groundedness_reasoning": "<1-2 sentences>",
  "relevance_score": <1-5 integer>,
  "relevance_reasoning": "<1-2 sentences>"
}}"""

    for attempt in range(max_retries):
        try:
            resp = judge_llm.invoke(prompt)
            raw = resp.content.strip()

            # Clean JSON markdown fences
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            parsed = json.loads(raw)
            return JudgeEvaluation(**parsed)
        except Exception as e:
            if attempt == max_retries - 1:
                # Return neutral fallback on repeated error
                return JudgeEvaluation(
                    groundedness_score=3,
                    groundedness_reasoning=f"Evaluation failed after {max_retries} attempts: {e}",
                    relevance_score=3,
                    relevance_reasoning="Evaluation failed.",
                )
            time.sleep(2.0 * (attempt + 1))


def evaluate_e2e(
    golden_set_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    enable_judge: bool = True,
) -> dict[str, Any]:
    """Execute end-to-end evaluation of the multi-agent graph."""
    if golden_set_path is None:
        golden_set_path = Path(__file__).parent / "golden_set.jsonl"
    else:
        golden_set_path = Path(golden_set_path)

    cases = []
    with open(golden_set_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    if limit is not None and limit > 0:
        cases = cases[:limit]

    total_cases = len(cases)
    orchestrator = Orchestrator()
    judge_llm = create_judge_llm() if enable_judge else None

    # Tracking counters
    correct_routing_count = 0
    tp_escalate = 0
    fp_escalate = 0
    fn_escalate = 0
    tn_escalate = 0

    citation_eval_count = 0
    citation_hits = 0
    citation_precision_sum = 0.0
    citation_recall_sum = 0.0

    groundedness_scores: list[int] = []
    relevance_scores: list[int] = []

    case_eval_logs: list[dict[str, Any]] = []

    print(f"Starting E2E evaluation across {total_cases} test cases...")

    for idx, case in enumerate(cases, 1):
        q = case["question"]
        expected_agent = case["expected_agent"]
        expected_articles = case.get("expected_article_ids", [])
        should_escalate = case.get("should_escalate", False)

        print(f"[{idx}/{total_cases}] Eval {case['id']}: '{q[:60]}...'")

        # Process message through LangGraph orchestrator
        start_t = time.perf_counter()
        result = orchestrator.process_message(message=q)
        duration = round(time.perf_counter() - start_t, 3)

        # 1. Routing extraction
        # Check initial route from triage handover, or current agent
        routed_agent = None
        for h in result.get("handover_history", []):
            if h.get("source_agent") == "triage":
                routed_agent = h.get("target_agent")
                break
        if not routed_agent:
            routed_agent = result.get("current_agent", "unknown")

        # Normalize routed agent
        is_routing_correct = (routed_agent == expected_agent)
        if is_routing_correct:
            correct_routing_count += 1

        # 2. Escalation extraction
        pred_escalate = bool(
            result.get("is_escalated")
            or result.get("requires_human")
            or result.get("current_agent") == "escalation"
            or any(h.get("target_agent") == "escalation" for h in result.get("handover_history", []))
        )

        if should_escalate and pred_escalate:
            tp_escalate += 1
        elif not should_escalate and pred_escalate:
            fp_escalate += 1
        elif should_escalate and not pred_escalate:
            fn_escalate += 1
        else:
            tn_escalate += 1

        # 3. Citation correctness
        messages = result.get("messages", [])
        assistant_msgs = [m for m in messages if (isinstance(m, dict) and m.get("role") == "assistant") or getattr(m, "role", "") == "assistant"]
        if assistant_msgs:
            last_msg = assistant_msgs[-1]
            content = last_msg.get("content", "") if isinstance(last_msg, dict) else last_msg.content
            meta = last_msg.get("metadata", {}) if isinstance(last_msg, dict) else getattr(last_msg, "metadata", {})
        else:
            content = ""
            meta = {}

        cited_ids = set(re.findall(r"KB-\d{3}", content))
        for cit in meta.get("citations", []):
            if isinstance(cit, dict) and "article_id" in cit:
                cited_ids.add(cit["article_id"])

        hit = False
        prec = 0.0
        rec = 0.0
        if expected_articles:
            citation_eval_count += 1
            exp_set = set(expected_articles)
            overlap = cited_ids & exp_set
            hit = len(overlap) > 0
            if hit:
                citation_hits += 1
            rec = len(overlap) / len(exp_set) if exp_set else 0.0
            prec = len(overlap) / len(cited_ids) if cited_ids else 0.0
            citation_precision_sum += prec
            citation_recall_sum += rec

        # 4. LLM Judge Evaluation
        judge_res = None
        if enable_judge and judge_llm:
            kb_context_str = "\n".join(
                f"[{c.get('article_id')}]: {c.get('chunk_text', '')}"
                for c in result.get("retrieved_contexts", [])
            )
            judge_res = run_llm_judge(judge_llm, q, content, kb_context_str)
            groundedness_scores.append(judge_res.groundedness_score)
            relevance_scores.append(judge_res.relevance_score)

        case_eval_logs.append({
            "id": case["id"],
            "question": q,
            "duration_sec": duration,
            "routing": {
                "expected": expected_agent,
                "predicted": routed_agent,
                "correct": is_routing_correct,
            },
            "escalation": {
                "expected": should_escalate,
                "predicted": pred_escalate,
            },
            "citations": {
                "expected": expected_articles,
                "cited": list(cited_ids),
                "hit": hit,
                "precision": round(prec, 3),
                "recall": round(rec, 3),
            },
            "judge": judge_res.model_dump() if judge_res else None,
            "response_preview": content[:150] + "...",
        })

    # Compute overall metrics
    routing_accuracy = round(correct_routing_count / total_cases, 4) if total_cases else 0.0

    esc_p = tp_escalate / (tp_escalate + fp_escalate) if (tp_escalate + fp_escalate) > 0 else 0.0
    esc_r = tp_escalate / (tp_escalate + fn_escalate) if (tp_escalate + fn_escalate) > 0 else 0.0
    esc_f1 = (2 * esc_p * esc_r) / (esc_p + esc_r) if (esc_p + esc_r) > 0 else 0.0

    cit_hit_rate = round(citation_hits / citation_eval_count, 4) if citation_eval_count else 0.0
    cit_avg_p = round(citation_precision_sum / citation_eval_count, 4) if citation_eval_count else 0.0
    cit_avg_r = round(citation_recall_sum / citation_eval_count, 4) if citation_eval_count else 0.0

    avg_groundedness = round(sum(groundedness_scores) / len(groundedness_scores), 2) if groundedness_scores else 0.0
    avg_relevance = round(sum(relevance_scores) / len(relevance_scores), 2) if relevance_scores else 0.0

    summary = {
        "total_cases": total_cases,
        "routing_accuracy": routing_accuracy,
        "escalation": {
            "precision": round(esc_p, 4),
            "recall": round(esc_r, 4),
            "f1": round(esc_f1, 4),
            "tp": tp_escalate,
            "fp": fp_escalate,
            "fn": fn_escalate,
            "tn": tn_escalate,
        },
        "citations": {
            "evaluated_cases": citation_eval_count,
            "hit_rate": cit_hit_rate,
            "precision": cit_avg_p,
            "recall": cit_avg_r,
        },
        "llm_judge": {
            "average_groundedness": avg_groundedness,
            "average_relevance": avg_relevance,
            "scale": "1-5",
        },
        "per_case_logs": case_eval_logs,
    }

    # Save outputs
    out_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "e2e.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_content = f"""# CloudDash End-to-End System Evaluation Results

Evaluated across **{total_cases} golden test cases** through the live LangGraph multi-agent pipeline.

## Summary Scores

| Metric | Score | Details |
| :--- | :---: | :--- |
| **Routing Accuracy** | **{routing_accuracy:.2%}** | {correct_routing_count}/{total_cases} correctly classified to triage target |
| **Escalation Precision** | **{esc_p:.2%}** | TP={tp_escalate}, FP={fp_escalate} |
| **Escalation Recall** | **{esc_r:.2%}** | TP={tp_escalate}, FN={fn_escalate} |
| **Escalation F1** | **{esc_f1:.2%}** | Harmonic mean of escalation precision and recall |
| **Citation Hit Rate** | **{cit_hit_rate:.2%}** | Percentage of KB-supported queries citing expected article |
| **Citation Recall** | **{cit_avg_r:.2%}** | Mean recall of expected KB article IDs |
| **LLM Judge Groundedness** | **{avg_groundedness}/5.0** | Factual alignment with retrieved KB context (Pydantic validated) |
| **LLM Judge Answer Relevance** | **{avg_relevance}/5.0** | Directness and completeness of customer response |
"""
    md_path = out_dir / "e2e.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n--- E2E Evaluation Completed ---")
    print(f"Routing Accuracy:       {routing_accuracy:.2%}")
    print(f"Escalation F1:          {esc_f1:.2%} (P={esc_p:.2%}, R={esc_r:.2%})")
    print(f"Citation Hit Rate:      {cit_hit_rate:.2%}")
    print(f"Judge Groundedness:     {avg_groundedness}/5.0")
    print(f"Judge Answer Relevance: {avg_relevance}/5.0")

    return summary


if __name__ == "__main__":
    evaluate_e2e()
