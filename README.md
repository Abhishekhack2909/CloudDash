---
title: CloudDash Customer Support
emoji: ☁️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
app_file: streamlit_app.py
pinned: false
---

# ☁️ CloudDash Multi-Agent Customer Support System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ComplexHat/clouddash-support)
[![Python 3.11](https://img.shields.io/badge/python-3.11-brightgreen.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-blueviolet.svg)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, stateful multi-agent customer support platform for **CloudDash** — an enterprise B2B SaaS cloud monitoring ecosystem (AWS, GCP, Azure, Kubernetes).

Engineered with **LangGraph** for resilient supervisor agent orchestration, **FastAPI** for async high-throughput REST APIs, a **Personalized Hybrid RAG** engine combining sparse BM25 and dense ChromaDB embeddings with cross-encoder re-ranking, **SQLAlchemy** for persistent multi-turn state and durable customer memory, and an automated **Evaluation Harness** tracking golden retrieval and LLM-as-a-judge quality metrics.

---

## 🌐 Live Deployments & Demos

| Component | Platform | Status | Access Link |
| :--- | :--- | :---: | :--- |
| **Interactive Support & Analytics UI** | **Hugging Face Spaces (Docker)** | 🟢 **Live & Active** | [Launch CloudDash UI](https://huggingface.co/spaces/ComplexHat/clouddash-support) |
| **Async REST API Documentation** | **Local / Docker Compose** | 🟢 **Ready** | `http://localhost:8000/docs` (or Render when active) |

> 💡 **Live Demo Tip:** The Hugging Face Spaces deployment is 100% self-contained in Docker, featuring the live multi-agent graph, knowledge base search, simulated human handoff, and real-time **Platform Analytics**.

---

## 🏗️ Architecture & Request Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                            Interface Layer                             │
│     FastAPI Async REST API    │   Streamlit UI + Analytics Dashboard   │
└──────────────────────────┬─────────────────────────────┬───────────────┘
                           │                             │
┌──────────────────────────▼─────────────────────────────▼───────────────┐
│                        Input Guardrails                                │
│         Prompt Injection Detection (Regex) │ Off-Topic Filter          │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────────┐
│                 LangGraph Orchestrator (Supervisor)                     │
│  State checkpointer (SQLite/PostgreSQL) keyed by conversation_id       │
│                                                                        │
│  ┌───────────┐       ┌──────────────────────┐   ┌───────────────────┐  │
│  │  Triage   │──────▶│   Technical Agent    │──▶│   Billing Agent   │  │
│  │  Agent    │       │(RAG + Health Checks) │   │ (RAG + DB Lookup) │  │
│  └─────┬─────┘       └──────────┬───────────┘   └─────────┬─────────┘  │
│        │                        │                         │            │
│        │                 ┌──────▼─────────────────────────▼─────────┐  │
│        └────────────────▶│             Escalation Agent             │  │
│                          │      (Human Handover Packaging & P1)     │  │
│                          └──────────────────────────────────────────┘  │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────────┐
│                       Output Guardrails                                │
│       PII Redaction │ Hallucination Check │ Policy Compliance          │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────────┐
│                 Personalized Hybrid RAG Pipeline                       │
│  Query Rewrite ──▶ Dense Vector (ChromaDB) + Sparse Lexical (BM25)     │
│                ──▶ Reciprocal Rank Fusion (RRF, k=60)                  │
│                ──▶ Plan-Tier (Enterprise/Pro) & Integration Boosting   │
│                ──▶ Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)    │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────────┐
│                 Persistent Storage & Execution Engine                  │
│  • SQLAlchemy ORM: Conversations, Messages, Handover Audit, Feedback   │
│  • Pydantic Tools: create_support_ticket, check_integration_status     │
│  • Durable User Memory: LLM-extracted customer profiles across turns  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Evaluation Results

The retrieval pipeline was evaluated on a comprehensive golden evaluation benchmark covering exact error codes, configuration syntaxes, and paraphrased user queries across all knowledge base categories.

### Retrieval Strategy Benchmark (125 Enterprise KB Articles, 1,114 chunks)

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Dense Vector Only** (`all-MiniLM-L6-v2`) | 57.14% | 66.67% | 71.43% | 0.6258 |
| **Sparse BM25 Only** (`BM25Okapi`) | 47.62% | 69.05% | 73.81% | 0.5896 |
| **Hybrid RRF** (Vector + BM25) | 52.38% | 73.81% | 83.33% | 0.6614 |
| **Hybrid + Cross-Encoder Rerank** (`ms-marco-MiniLM-L-6-v2`) | **59.52%** | **80.95%** | **85.71%** | **0.7166** |

### Key Takeaways:
- **Cross-Encoder Reranking Lift**: Adding cross-encoder re-ranking increased **Recall@5 from 71.43% to 85.71% (+14.28% absolute lift)** and lifted **MRR by +0.0908 (0.6258 → 0.7166)** over vector search alone.
- **Hybrid Fusion Power**: Sparse BM25 excels at specific error tokens (e.g. `ERR_AUTH_EXPIRED`), while dense vectors capture natural language semantics; RRF fusion eliminates single-retriever blind spots.

---

## ✨ Key Capabilities

| Feature | Implementation |
|---------|---------------|
| **Multi-Agent Architecture** | Triage, Technical Support, Billing, Escalation with YAML-configurable prompts and LangGraph state machine |
| **Hybrid RAG + Personalization** | ChromaDB + BM25 with RRF, customer plan tier boosting/down-weighting, and active integration scoring |
| **Real Structured Tools** | `create_support_ticket` (persisted to SQL DB), `check_integration_status` (live health check), `lookup_account` (SQL account lookup) |
| **Persistent State & Memory** | SQLAlchemy conversation storage (SQLite/Postgres), LangGraph checkpoints, and durable user memory extracted across sessions |
| **Incremental Ingestion** | SHA-256 content hashing in `knowledge_base/ingest.py`, `/admin/reindex` endpoint, and 125 enterprise articles (1,114 chunks) |
| **Model Resilience** | `FallbackChatGroq` automatic failover across models (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`, etc.) |
| **Guardrails & Safety** | Input: prompt injection regex + off-topic classifier; Output: PII redaction + citation hallucination check |
| **Telemetry & Feedback** | User message feedback (👍/👎), `/metrics` aggregate telemetry, and real-time Streamlit Analytics dashboard |
| **Observability** | Structured JSON logging via `structlog` + LangSmith distributed tracing |

---

## 🚀 Quick Start (Local & Docker)

### Option A: Run with Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/Abhishekhack2909/CloudDash.git
cd CloudDash

# Configure environment
cp .env.example .env
# Add GROQ_API_KEY=gsk_... to .env

# Run full stack (FastAPI + Streamlit + ChromaDB + SQLite)
docker compose up --build
```
- Web UI: `http://localhost:8501`
- REST API Docs: `http://localhost:8000/docs`

---

### Option B: Local Python Environment

#### 1. Setup Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

#### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
```

#### 3. Ingest Knowledge Base
```bash
python knowledge_base/ingest.py
```
Outputs:
- 125 articles indexed into ChromaDB (`./chroma_db/`)
- BM25 index serialized (`./chroma_db/bm25_index.pkl`)
- SHA-256 ingestion manifest (`./knowledge_base/ingestion_manifest.json`)

#### 4. Run the API & UI
```bash
# Terminal 1: FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Streamlit chat interface
streamlit run streamlit_app.py
```

---

## 🧪 Testing & Evaluation CLI

### Run Unit and Integration Tests
```bash
# Run all tests
pytest tests/ -v

# Run domain test suites
pytest tests/test_personalization.py -v
pytest tests/test_tools.py -v
pytest tests/test_memory.py -v
pytest tests/test_analytics.py -v
```

### Run Evaluation Harness
```bash
# Evaluate retrieval strategies against golden set
python -m evals.run --suite retrieval

# Check retrieval against regression baseline
python -m evals.run --suite retrieval --check-baseline

# Run end-to-end evaluation
python -m evals.run --suite e2e
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/conversations` | Start conversation with initial message and optional customer profile |
| `POST` | `/conversations/{id}/messages` | Send follow-up message in existing multi-turn conversation |
| `GET` | `/conversations/{id}` | Get full conversation history, messages, and extracted entities |
| `POST` | `/conversations/{id}/feedback` | Submit thumbs-up/down rating with optional feedback note |
| `GET` | `/conversations/{id}/handovers` | Retrieve immutable handover audit log |
| `GET` | `/metrics` | Aggregate platform telemetry (resolution rate, routing distribution, sentiment) |
| `GET` | `/health` | Health check with active model, fallback status, and KB count |
| `POST` | `/admin/reindex` | Trigger incremental or full re-indexing (protected by `X-Admin-Key`) |

---

## 🏛️ Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **LangGraph Supervisor Pattern** | Explicit state graph with clear conditional routing, cycles, and handovers | Slightly more upfront structure than free-form swarms |
| **SQLAlchemy Persistence** | Portable persistence: zero-config SQLite locally, seamlessly switches to Postgres via `DATABASE_URL` | Requires DB migrations in scaled multi-tenant production |
| **Hybrid RAG (BM25 + Vector)** | Exact matches on error codes and technical terms combined with semantic similarity | Two indices to build and maintain |
| **Cross-Encoder Re-Ranking** | Joint query-document attention provides significantly higher precision than bi-encoder embeddings alone | Adds ~150-250ms per retrieval query |
| **Structured Output Tools** | Tools strictly typed with Pydantic for deterministic input validation | LLM must conform to schema |
| **FallbackChatGroq Client** | Automatic failover when primary Groq model is deprecated or rate-limited | Requires maintaining fallback model list |

---

## ⚠️ Notes & Operational Details

- **Groq Free-Tier Model Selection**: Uses Groq's high-speed inference. Default model is `openai/gpt-oss-120b` or `llama-3.3-70b-versatile` with automatic failover.
- **Render Free-Tier Sleep**: Render free instances sleep after 15 minutes of inactivity; first request may take 45-60 seconds to spin up.
- **Escalation**: Escalation agent generates structured P1/P2 human handoff packages with summary, sentiment, and ticket ID for human agent queuing.

---

## 📜 License

MIT License. Built for CloudDash Customer Support System.
