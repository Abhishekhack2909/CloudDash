# CloudDash Multi-Agent System — Architecture Document

## 1. System Overview

The CloudDash Customer Support System is a **multi-agent AI system** that handles end-to-end customer interactions through specialized agents coordinated by a central supervisor.

### Core Design Principles

1. **Separation of Concerns** — Each agent handles a specific domain (technical, billing, escalation). The orchestrator handles routing logic. RAG handles retrieval. Guardrails handle safety.
2. **Configuration over Code** — Agent prompts, routing rules, and guardrail patterns are YAML-configurable. Adding a new agent requires minimal code changes.
3. **Grounded Responses** — Every agent response is grounded in KB articles via RAG, with explicit source citations. The system never fabricates information.
4. **Graceful Degradation** — If an agent fails, the handover protocol falls back to the Triage or Escalation agent. If KB has no answer, the system transparently acknowledges the gap.
5. **Observable by Default** — Every agent invocation, KB retrieval, handover, and guardrail event is logged with structured JSON and traced via LangSmith.

---

## 2. Agent Architecture

### 2.1 Orchestration Pattern: Supervisor

We use LangGraph's **Supervisor pattern** where a central orchestrator manages the flow:

```
User Message
    │
    ▼
┌─────────────┐
│ Input Guard  │ ─── Blocked? → Return safety message
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Triage    │ ─── Classifies intent, extracts entities
└──────┬──────┘
       │
       ├── technical → Technical Support Agent
       ├── billing   → Billing Agent
       └── escalation → Escalation Agent
              │
              ▼
       ┌─────────────┐
       │ Output Guard │ ─── PII redaction, hallucination check
       └──────┬──────┘
              │
              ▼
         Response
```

**Why Supervisor over Swarm?**
- Easier to reason about, debug, and explain in a live discussion
- Clear audit trail of routing decisions
- Prevents agent loops (max-turn protection)
- Matches the hierarchical nature of customer support (triage → specialist → escalation)

### 2.2 Agent Descriptions

| Agent | Responsibility | Key Design Choice |
|-------|---------------|-------------------|
| **Triage** | Intent classification + entity extraction via LLM. Returns structured JSON with routing decision | Uses structured JSON output for reliable parsing with fallback on malformed responses |
| **Technical Support** | RAG-powered troubleshooting with KB citations and code snippets | Signals handovers via `[HANDOVER: agent \| reason]` markers in response |
| **Billing** | Mock account lookup + KB-grounded policy citation | Includes simulated account data in LLM context for realistic responses |
| **Escalation** | Creates structured handover package for human operators | Generates both internal package (priority, sentiment) and customer-facing message |

### 2.3 State Management

All agents share a `ConversationState` Pydantic model that flows through the graph:

```python
ConversationState:
  conversation_id: str          # Unique trace ID
  messages: list[Message]       # Full conversation history
  current_agent: str            # Currently active agent
  extracted_entities: Entities   # Customer ID, plan, urgency, sentiment
  handover_history: list[dict]  # Audit trail of all handovers
  retrieved_contexts: list      # Latest RAG results
  next_agent: str | None        # Routing decision for next step
  is_resolved / is_escalated    # Terminal state flags
```

---

## 3. RAG Pipeline

### 3.1 Architecture

```
User Query + Conversation Context
         │
         ▼
┌─────────────────┐
│  Query Rewrite  │  LLM rewrites query using conversation context
└────────┬────────┘  to resolve coreferences and add relevant terms
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────┐
│ChromaDB│ │ BM25 │   Parallel retrieval
│(Vector)│ │(KW)  │   Vector: semantic similarity
└───┬────┘ └──┬───┘   BM25: exact keyword matching
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│   RRF Fusion    │   Reciprocal Rank Fusion
│                 │   Score = Σ weight/(k + rank)
└────────┬────────┘   No score normalization needed
         │
         ▼
┌─────────────────┐
│  Cross-Encoder  │   Re-rank top candidates
│  Re-ranker      │   ms-marco-MiniLM-L-6-v2
└────────┬────────┘
         │
         ▼
   Top-K Results
   (with citations)
```

### 3.2 Key Decisions

**Why Hybrid Retrieval?**
- Technical KB articles contain exact terms like `CloudWatchReadOnlyAccess`, `429 Too Many Requests`, `ACS URL` that vector search may miss
- BM25 provides precision for these exact matches while vector search provides semantic recall
- RRF elegantly combines both without needing score normalization

**Why Cross-Encoder Re-ranking?**
- Bi-encoders (used in initial retrieval) encode query and document separately — fast but less precise
- Cross-encoders process query-document pairs jointly — slower but much more accurate
- We use it as a final precision step on the top ~15 candidates from hybrid retrieval

**Chunking Strategy:**
- Split by markdown headers (`##`, `###`) to preserve semantic boundaries
- Fallback to paragraph-based splitting for large sections
- Target ~500 characters per chunk with no overlap (section-based splitting makes overlap unnecessary)
- Each chunk retains full metadata (article ID, title, category, tags, plan applicability)

---

## 4. Handover Protocol

### 4.1 Handover Flow

```
Source Agent detects cross-domain need
         │
         ▼
┌─────────────────────┐
│ Handover Protocol   │
│ 1. Package context  │ ← Full history OR LLM summary
│ 2. Transfer entities│ ← Customer ID, plan, urgency
│ 3. Validate target  │ ← Is target agent available?
│ 4. Log audit event  │ ← timestamp, source, target, reason
└────────┬────────────┘
         │
    ┌────┴────┐
    │ Success │──▶ Target agent receives full context
    │         │    Customer doesn't repeat information
    └─────────┘
         │
    ┌────┴────┐
    │ Failure │──▶ Fallback chain: Triage → Escalation
    │         │    Error logged with context
    └─────────┘
```

### 4.2 Audit Log Schema

Every handover produces an audit record:

```json
{
  "id": "uuid",
  "conversation_id": "uuid",
  "timestamp": "2026-05-08T10:30:00Z",
  "source_agent": "technical_support",
  "target_agent": "billing",
  "reason": "Customer asked about plan upgrade",
  "status": "success",
  "context_snapshot": {
    "message_count": 5,
    "entities": {"plan_type": "Pro", "urgency": "medium"},
    "turn_count": 3
  }
}
```

---

## 5. Guardrails

### 5.1 Input Guardrails (Pre-LLM)

| Check | Method | Action |
|-------|--------|--------|
| **Prompt Injection** | Pattern matching against known phrases | Block + return safety message |
| **Off-Topic** | Keyword matching against CloudDash topics | Block only clearly irrelevant short messages |

**Design Choice:** We use pattern matching (not an LLM classifier) for injection detection because:
- Zero latency cost (sub-millisecond)
- No additional API calls
- Deterministic behavior (same input → same result)
- Easy to extend via YAML config

### 5.2 Output Guardrails (Post-LLM)

| Check | Method | Action |
|-------|--------|--------|
| **PII Redaction** | Regex patterns for email, phone, SSN, credit card, IP | Replace with `[REDACTED]` |
| **Hallucination** | Check if grounded topics (pricing, policies) appear in KB context | Append disclaimer if ungrounded |
| **Policy Violations** | Check for blocked claims ("unlimited", "guaranteed uptime") | Flag and log |

---

## 6. Observability

### 6.1 Structured Logging

Every event is logged as JSON via `structlog`:

```json
{
  "event": "agent_invoked",
  "timestamp": "2026-05-08T10:30:00Z",
  "level": "info",
  "agent_id": "technical_support",
  "conversation_id": "conv-abc123",
  "turn_count": 2
}
```

Log categories: `agent_invoked`, `kb_retrieval_start`, `kb_retrieval_complete`, `agent_handover`, `input_blocked`, `output_pii_redacted`, `escalation_complete`

### 6.2 LangSmith Tracing

LangSmith auto-instruments all LangChain/LangGraph calls when enabled:
- Every LLM call: model, prompt, tokens, latency
- Every agent node execution in the graph
- Full conversation flow visualization
- Token usage and cost tracking

Enable via: `LANGCHAIN_TRACING_V2=true` + `LANGSMITH_API_KEY`

---

## 7. Configuration & Extensibility

### 7.1 YAML-Driven Configuration

```
config/
├── agents.yaml       # Agent definitions (prompts, capabilities, routes)
├── routing.yaml      # Intent → agent mapping + fallback chains
└── guardrails.yaml   # Injection patterns, PII rules, hallucination topics
```

**Adding a new agent type does NOT require modifying:**
- The RAG pipeline
- The guardrails
- The handover protocol
- The API schemas

**It only requires:**
1. A YAML config block in `agents.yaml`
2. A Python class extending `BaseAgent`
3. Registering the node in the orchestrator graph
4. Adding routing rules in `routing.yaml`

### 7.2 Future Production Considerations

| Concern | Current (Prototype) | Production Evolution |
|---------|-------------------|---------------------|
| **State Persistence** | In-memory dict | Redis/PostgreSQL + LangGraph checkpointing |
| **Scaling** | Single process | Async workers + task queue (Celery/RQ) |
| **Multi-tenancy** | N/A | Tenant-scoped KB collections + API key auth |
| **Rate Limiting** | None | FastAPI middleware + Redis token bucket |
| **Cost Optimization** | GPT-4o-mini for all | GPT-4o-mini for triage, GPT-4o for complex |
| **KB Updates** | Manual re-ingestion | Webhook-triggered incremental updates |
| **Monitoring** | LangSmith + structlog | Prometheus + Grafana + PagerDuty alerting |
