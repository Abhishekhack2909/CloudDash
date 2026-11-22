"""Tests for personalized retrieval: plan tier boosting, integration ranking, and down-weighting."""

import pytest
from retrieval.pipeline import RAGPipeline
from retrieval.reranker import Reranker
from models.knowledge import RetrievalResult


def test_datadog_vs_slack_alert_query_changes_top_result():
    """Assert that the exact same query 'How do I set up alerts?' returns different top-1 results for Datadog vs Slack."""
    pipeline = RAGPipeline()
    query = "How do I set up alerts?"

    results_datadog = pipeline.retrieve(
        query=query,
        top_k=5,
        active_integrations=["Datadog"],
    )

    results_slack = pipeline.retrieve(
        query=query,
        top_k=5,
        active_integrations=["Slack"],
    )

    top_datadog = results_datadog[0].article_id
    top_slack = results_slack[0].article_id

    print(f"Top result for Datadog: {top_datadog} ({results_datadog[0].article_title})")
    print(f"Top result for Slack: {top_slack} ({results_slack[0].article_title})")

    assert top_datadog != top_slack
    assert top_datadog == "KB-095"  # Migrating Dashboards and Alerts from Datadog
    assert top_slack in ["KB-001", "KB-109"]  # Slack Alert configuration


def test_integration_boosting_changes_ranking():
    """Assert that the exact same query produces different ranking when integration context changes."""
    pipeline = RAGPipeline()
    query = "How do I configure infrastructure monitoring integration?"

    results_aws = pipeline.retrieve(
        query=query,
        top_k=5,
        active_integrations=["AWS", "CloudWatch"],
    )

    results_k8s = pipeline.retrieve(
        query=query,
        top_k=5,
        active_integrations=["Kubernetes", "daemonset"],
    )

    ids_aws = [r.article_id for r in results_aws]
    ids_k8s = [r.article_id for r in results_k8s]

    assert ids_aws != ids_k8s
    assert "KB-002" in ids_aws[:2]


def test_plan_tier_boosting_changes_ranking():
    """Assert that Enterprise tier users see Enterprise-applicable articles ranked higher."""
    reranker = Reranker()
    query = "What data retention and auditing options are available?"

    starter_candidate = RetrievalResult(
        article_id="KB-013",
        article_title="Data Retention Policy (Starter 7 days)",
        category="faq",
        chunk_text="Starter plan provides 7 days data retention without cold archive.",
        relevance_score=0.5,
        metadata={"applies_to": "Starter"},
    )

    enterprise_candidate = RetrievalResult(
        article_id="KB-070",
        article_title="Audit Log Streaming to S3 and Long-Term Cold Archive",
        category="account",
        chunk_text="Enterprise plan provides 365 days retention and S3 cold storage.",
        relevance_score=0.5,
        metadata={"applies_to": "Enterprise"},
    )

    candidates = [starter_candidate, enterprise_candidate]

    reranked_starter = reranker.rerank(
        query=query,
        results=candidates,
        top_k=2,
        plan_tier="Starter",
    )

    reranked_enterprise = reranker.rerank(
        query=query,
        results=candidates,
        top_k=2,
        plan_tier="Enterprise",
    )

    starter_top = reranked_starter[0].article_id
    enterprise_top = reranked_enterprise[0].article_id

    assert starter_top == "KB-013"
    assert enterprise_top == "KB-070"
    assert [r.article_id for r in reranked_starter] != [r.article_id for r in reranked_enterprise]


def test_free_tier_downweights_enterprise_only():
    """Assert that enterprise-only articles are down-weighted for free/starter users."""
    reranker = Reranker()
    query = "How do I configure data export and backups?"

    starter_doc = RetrievalResult(
        article_id="KB-013",
        article_title="Configuring Data Export and Backup Schedules",
        category="account",
        chunk_text="How to configure daily data exports and CSV downloads for backup.",
        relevance_score=0.5,
        metadata={"applies_to": "Starter,Pro"},
    )

    enterprise_only_doc = RetrievalResult(
        article_id="KB-070",
        article_title="Enterprise S3 Cold Storage and Glacier Data Backups",
        category="account",
        chunk_text="How to configure enterprise S3 cold storage archives and backups.",
        relevance_score=0.5,
        metadata={"applies_to": "Enterprise"},
    )

    results = reranker.rerank(
        query=query,
        results=[enterprise_only_doc, starter_doc],
        top_k=2,
        plan_tier="Starter",
    )

    assert results[0].article_id == "KB-013"
