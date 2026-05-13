---
title: CloudDash Customer Support
emoji: ☁️
colorFrom: blue
colorTo: purple
sdk: docker
app_file: streamlit_app.py
pinned: false
---

# ☁️ CloudDash Multi-Agent Customer Support System

A production-quality prototype multi-agent customer support system for **CloudDash** — a B2B SaaS cloud infrastructure monitoring platform (AWS, GCP, Azure).

Built with **LangGraph** for agent orchestration, **FastAPI** for the REST API, **ChromaDB + BM25** for hybrid RAG retrieval, and **LangSmith** for LLM observability.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Interface Layer                           │
│              FastAPI REST API  │  Streamlit Web UI               │
└──────────────────────┬───────────────────┬───────────────────────┘
                       │                   │
┌──────────────────────▼───────────────────▼───────────────────────┐
│                    Input Guardrails                               │
│         Prompt Injection Detection │ Off-Topic Filter             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│              LangGraph Orchestrator (Supervisor)                  │
│                                                                   │
│  ┌───────────┐   ┌─────────────────┐   ┌──────────────┐         │
│  │  Triage   │──▶│ Technical Agent  │──▶│   Billing    │         │
│  │  Agent    │   │  (RAG-powered)   │   │   Agent      │         │
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
└──────────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│                    RAG Pipeline                                   │
│  Query Rewrite → Vector Search (ChromaDB) + BM25 Keyword Search  │
│              → Reciprocal Rank Fusion → Cross-Encoder Reranking  │
└──────────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

| Feature | Implementation |
|---------|---------------|
| **4 Specialized Agents** | Triage, Technical Support, Billing, Escalation — each with configurable YAML prompts |
| **Hybrid RAG** | ChromaDB (vector) + BM25 (keyword) with RRF fusion and cross-encoder re-ranking |
| **Agent Handovers** | Context-preserving handovers with fallback chains and full audit logging |
| **Guardrails** | Input: prompt injection detection + off-topic filter. Output: PII redaction + hallucination check |
| **Observability** | Structured JSON logging (structlog) + LangSmith tracing for full LLM observability |
| **Knowledge Base** | 20 articles across 5 categories with source citations in every response |
| **REST API** | FastAPI with Swagger docs — start conversations, send messages, view history |
| **Web UI** | Streamlit chat interface with agent badges, citation display, and test scenarios |
| **Configurable** | Agent prompts, routing rules, and guardrails are all YAML-configurable |

---

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| **Streamlit UI** | https://huggingface.co/spaces/ComplexHat/clouddash-support |
| **REST API (Swagger)** | https://clouddash-r5xg.onrender.com/docs |

---

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com)) — primary LLM provider
- LangSmith API key (optional, for tracing)

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/CloudDash.git
cd CloudDash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys:
#   GROQ_API_KEY=gsk_...           (required — primary LLM)
#   LANGSMITH_API_KEY=lsv2_...     (optional, for LangSmith tracing)
#   LANGCHAIN_TRACING_V2=true      (optional, enable tracing)
```

### 3. Ingest Knowledge Base

```bash
python knowledge_base/ingest.py
```

This will:
- Load 20 KB articles from `knowledge_base/articles/`
- Chunk them by section headers (~500 char chunks)
- Embed using `sentence-transformers/all-MiniLM-L6-v2` (local, no API key needed)
- Index into ChromaDB (persisted in `./chroma_db/`)
- Build a BM25 index (pickled in `./chroma_db/bm25_index.pkl`)

### 4. Run the API

```bash
python -m api.main
# or
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

### 5. Run the Web UI (Optional)

```bash
streamlit run streamlit_app.py
```

### 6. Run Tests

```bash
# Unit tests (no API key needed)
pytest tests/test_guardrails.py tests/test_handover.py tests/test_retrieval.py -v

# Integration tests (requires OPENAI_API_KEY + ingested KB)
pytest tests/test_scenarios.py -v
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/conversations` | Start a new conversation |
| `POST` | `/conversations/{id}/messages` | Send a follow-up message |
| `GET` | `/conversations/{id}` | Get conversation history |
| `GET` | `/conversations/{id}/handovers` | Get handover audit log |
| `GET` | `/health` | System health check |

### Example: Start a Conversation

```bash
curl -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{"message": "My CloudDash alerts stopped firing after I updated my AWS credentials."}'
```

### Example: Send Follow-up Message

```bash
curl -X POST http://localhost:8000/conversations/{conversation_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"message": "I am on the Pro plan. Can you also check my billing?"}'
```

---

## 🧪 Test Scenarios

The system handles all 4 assessment scenarios:

### Scenario 1 — Single-Agent Resolution
> "My CloudDash alerts stopped firing after I updated my AWS integration credentials yesterday. I'm on the Pro plan."

**Flow:** Triage → Technical Support → KB-grounded resolution with citations `[KB-001]`, `[KB-002]`, `[KB-020]`

### Scenario 2 — Cross-Agent Handover
> "I want to upgrade from Pro to Enterprise, but first can you check if the SSO integration issue I reported last week has been resolved?"

**Flow:** Triage → Technical Support (SSO) → handover to Billing (upgrade) with full context

### Scenario 3 — Escalation to Human
> "I've been charged twice for April. I need an immediate refund and I want to speak to a manager."

**Flow:** Triage → Billing/Escalation → human handover package with priority P1/P2, sentiment analysis, ticket ID

### Scenario 4 — KB Retrieval Failure
> "Does CloudDash support integration with Datadog for cross-platform alerting?"

**Flow:** Agent searches KB → no relevant article → transparently acknowledges → offers escalation/feature request

---

## 🏛️ Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **LangGraph (supervisor pattern)** | Fine-grained control over state transitions, conditional routing, and cyclic workflows. Central orchestrator is clear to debug and extend | Less autonomous than swarm pattern |
| **ChromaDB** | File-persistent, zero-config, no external service. Self-contained prototype | Less scalable than Pinecone/Qdrant |
| **Hybrid Retrieval (BM25 + Vector)** | Technical KB has exact terms (API endpoints, error codes) that BM25 catches better than vector alone | Additional complexity + BM25 index memory |
| **Cross-encoder Re-ranking** | Significantly improves final retrieval precision at ~200ms latency cost | Adds sentence-transformers dependency |
| **Custom Guardrails** | Lightweight, transparent, zero external dependencies. Adequate for prototype | Less comprehensive than LLM Guard |
| **LangSmith** | Native LangGraph auto-instrumentation. Zero-config tracing | Requires LangSmith account |
| **YAML Configuration** | Agent prompts, routing, guardrails are all YAML-configurable. Adding a new agent = adding a YAML block + a Python class | Slightly more boilerplate vs hardcoded |

---

## 📁 Repository Structure

```
CloudDash/
├── README.md                    # This file
├── ARCHITECTURE.md              # Detailed architecture document
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project config
├── .env.example                 # Environment variable template
├── .gitignore
│
├── agents/                      # Agent implementations
│   ├── base_agent.py            # Abstract base class
│   ├── triage_agent.py          # Intent classification + routing
│   ├── technical_agent.py       # Technical support (RAG-powered)
│   ├── billing_agent.py         # Billing + account (mock data)
│   ├── escalation_agent.py      # Human handover packaging
│   └── orchestrator.py          # LangGraph supervisor orchestrator
│
├── knowledge_base/              # KB articles + ingestion
│   ├── articles/                # 20 JSON articles (KB-001 to KB-020)
│   └── ingest.py                # Chunking, embedding, indexing script
│
├── retrieval/                   # RAG pipeline
│   ├── embeddings.py            # OpenAI embedding wrapper
│   ├── vector_store.py          # ChromaDB vector search
│   ├── bm25_store.py            # BM25 keyword search
│   ├── hybrid_retriever.py      # RRF fusion of vector + BM25
│   ├── reranker.py              # Cross-encoder re-ranking
│   └── pipeline.py              # Full RAG pipeline orchestration
│
├── handover/                    # Handover protocol
│   └── protocol.py              # Context transfer + audit logging
│
├── guardrails/                  # Safety layer
│   ├── input_guardrails.py      # Injection detection + off-topic
│   └── output_guardrails.py     # PII redaction + hallucination check
│
├── observability/               # Logging & tracing
│   ├── logger.py                # Structured JSON logging (structlog)
│   └── tracing.py               # LangSmith tracing config
│
├── api/                         # REST API
│   ├── main.py                  # FastAPI application
│   └── schemas.py               # Request/response Pydantic models
│
├── config/                      # YAML configuration
│   ├── agents.yaml              # Agent prompts, capabilities, routing
│   ├── routing.yaml             # Intent-to-agent mapping
│   └── guardrails.yaml          # Guardrail patterns and rules
│
├── tests/                       # Test suite
│   ├── test_triage.py           # Triage classification tests
│   ├── test_retrieval.py        # RAG pipeline tests
│   ├── test_handover.py         # Handover protocol tests
│   ├── test_guardrails.py       # Guardrails safety tests
│   └── test_scenarios.py        # Integration tests (4 scenarios)
│
└── streamlit_app.py             # Web UI
```

---

## 🔧 Extending the System

### Adding a New Agent (e.g., Onboarding Agent)

1. **Add config** in `config/agents.yaml`:
```yaml
agents:
  onboarding:
    name: "Onboarding Agent"
    system_prompt: |
      You are the Onboarding Agent for CloudDash...
    capabilities: [guided_setup, tutorial_generation]
    routes_to: [technical_support, escalation]
    tools: [search_knowledge_base]
```

2. **Create agent** in `agents/onboarding_agent.py`:
```python
from agents.base_agent import BaseAgent

class OnboardingAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_id="onboarding")

    def process(self, state):
        # Your agent logic here
        ...
```

3. **Register** in `agents/orchestrator.py` — add the node and edges.

4. **Update routing** in `config/routing.yaml` — add intent mapping.

No changes needed to the core orchestration logic, guardrails, or RAG pipeline.

---

## ⚠️ Known Limitations

- **In-memory conversation store**: Conversations are stored in memory and lost on restart. A production system would use Redis or PostgreSQL.
- **Mock billing data**: Account lookup returns static mock data. A real system would integrate with Stripe/billing APIs.
- **Single-threaded**: The API processes requests sequentially. Production would need async workers or a task queue.
- **Cold start on Render**: Free tier instances spin down after 15 minutes of inactivity; first request takes ~60s to wake up.
- **Cross-encoder latency**: Re-ranking adds ~200ms per query. Could be disabled for latency-sensitive deployments.
- **No persistent checkpointing**: LangGraph checkpointing is not enabled; multi-turn conversations rely on in-memory state passed between calls.

---

## 📜 License

This project was built as part of a technical assessment. All code is original.
