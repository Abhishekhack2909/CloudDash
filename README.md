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

A production-style multi-agent customer support system for **CloudDash** — a B2B SaaS cloud infrastructure monitoring platform (AWS, GCP, Azure, Kubernetes).

Built with **LangGraph** for multi-agent supervisor orchestration, **FastAPI** for async REST endpoints, **ChromaDB + BM25** with cross-encoder re-ranking for hybrid RAG, **SQLAlchemy** (SQLite/PostgreSQL) for persistent conversation state and durable user memory, and **LangSmith** for LLM observability.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Interface Layer                           │
│     FastAPI Async REST API  │  Streamlit Web UI + Customer Profile│
└──────────────────────┬───────────────────┬───────────────────────┘
                       │                   │
┌──────────────────────▼───────────────────▼───────────────────────┐
│                    Input Guardrails                               │
│         Prompt Injection Detection │ Off-Topic Filter             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│              LangGraph Orchestrator (Supervisor)                  │
│  State checkpointer (SQLite/PostgreSQL) keyed by conversation_id  │
│                                                                   │
│  ┌───────────┐   ┌─────────────────┐   ┌──────────────┐         │
│  │  Triage   │──▶│ Technical Agent │──▶│   Billing    │         │
│  │  Agent    │   │  (RAG + Tools)  │   │(RAG + DB Acc)│         │
│  └───────────┘   └────────┬────────┘   └──────┬───────┘         │
│       │                   │                    │                  │
│       │            ┌──────▼────────────────────▼──────┐          │
│       └───────────▶│       Escalation Agent           │          │
│                    │  (Human Handover Packaging)       │          │
│                    └──────────────────────────────────┘          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│                   Output Guardrails                               │
│      PII Redaction │ Hallucination Check │ Policy Filter          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│               Personalized Hybrid RAG Pipeline                    │
│  Query Rewrite → Vector Search (ChromaDB) + BM25 Keyword Search  │
│              → Reciprocal Rank Fusion (RRF)                      │
│              → Plan Tier & Active Integration Boosting           │
│              → Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2) │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│              Persistent Storage & Tool Execution                  │
│  - SQLite (default) / PostgreSQL (via DATABASE_URL)              │
│  - Real Tools: create_support_ticket, check_integration_status,  │
│    lookup_account (database-backed)                             │
│  - Long-Term User Memory: extracted durable customer facts       │
└──────────────────────────────────────────────────────────────────┘
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
| **Incremental Ingestion** | SHA-256 content hashing in `knowledge_base/ingest.py`, `/admin/reindex` endpoint, and 125 enterprise articles |
| **Model Resilience** | `FallbackChatGroq` automatic failover across models (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`, etc.) |
| **Guardrails & Safety** | Input: prompt injection regex + off-topic classifier; Output: PII redaction + citation hallucination check |
| **Observability** | Structured JSON logging via `structlog` + LangSmith distributed tracing |

---

## 🌐 Live Deployments

| Service | URL |
|---------|-----|
| **Streamlit UI** | https://huggingface.co/spaces/ComplexHat/clouddash-support |
| **REST API (Swagger)** | https://clouddash-r5xg.onrender.com/docs |

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

# Run personalization tests
pytest tests/test_personalization.py -v

# Run tools tests
pytest tests/test_tools.py -v

# Run memory tests
pytest tests/test_memory.py -v
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
| `GET` | `/conversations/{id}/handovers` | Retrieve immutable handover audit log |
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
