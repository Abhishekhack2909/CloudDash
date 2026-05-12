"""CloudDash Support — Streamlit Web UI.

A minimal chat interface for demonstrating the multi-agent system.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from agents.orchestrator import Orchestrator
from observability.tracing import configure_tracing, get_tracing_config

# Page config
st.set_page_config(
    page_title="CloudDash Support",
    page_icon="☁️",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .agent-triage { background: #e3f2fd; color: #1565c0; }
    .agent-technical { background: #e8f5e9; color: #2e7d32; }
    .agent-billing { background: #fff3e0; color: #e65100; }
    .agent-escalation { background: #fce4ec; color: #c62828; }
    .agent-guardrails { background: #f3e5f5; color: #6a1b9a; }
    .citation-box {
        background: rgba(25, 118, 210, 0.12);
        border-left: 3px solid #1976d2;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85rem;
        border-radius: 0 4px 4px 0;
        color: inherit;
    }
    .citation-box strong {
        color: #42a5f5;
    }
    .handover-event {
        background: rgba(103, 58, 183, 0.25);
        border: 1px solid #7c4dff;
        border-left: 4px solid #7c4dff;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 0.88rem;
        margin: 10px 0;
        color: inherit;
    }
    .handover-event strong {
        color: #b39ddb;
    }
</style>
""", unsafe_allow_html=True)


def get_agent_badge(agent_id: str) -> str:
    """Get an HTML badge for an agent."""
    agent_names = {
        "triage": "🔀 Triage",
        "technical_support": "🔧 Technical Support",
        "billing": "💳 Billing",
        "escalation": "🚨 Escalation",
        "guardrails": "🛡️ Guardrails",
    }
    css_class = f"agent-{agent_id.split('_')[0] if '_' in agent_id else agent_id}"
    name = agent_names.get(agent_id, agent_id)
    return f'<span class="agent-badge {css_class}">{name}</span>'


def _get_cite_field(cite, field: str, default: str = "") -> str:
    """Safely get a field from a citation — handles both dict and Pydantic objects."""
    if isinstance(cite, dict):
        return str(cite.get(field, default))
    return str(getattr(cite, field, default))


def render_citations(citations: list):
    """Render KB citation boxes."""
    if not citations:
        return
    with st.expander(f"📚 Sources ({len(citations)} KB articles)"):
        for cite in citations:
            article_id = _get_cite_field(cite, "article_id", "KB-???")
            article_title = _get_cite_field(cite, "article_title", "Unknown Article")
            category = _get_cite_field(cite, "category", "")
            score_raw = _get_cite_field(cite, "relevance_score", "")
            score_str = f" — score: {float(score_raw):.2f}" if score_raw else ""
            cat_str = f" [{category}]" if category else ""
            st.markdown(
                f'<div class="citation-box">'
                f'<strong>{article_id}</strong>: {article_title}{cat_str}{score_str}'
                f'</div>',
                unsafe_allow_html=True,
            )


@st.cache_resource
def get_orchestrator():
    """Get cached orchestrator instance."""
    configure_tracing()
    return Orchestrator()


def main():
    """Main Streamlit app."""
    # Sidebar
    with st.sidebar:
        st.title("☁️ CloudDash Support")
        st.caption("Multi-Agent Customer Support System")
        st.divider()

        # Conversation info
        if "conversation_state" in st.session_state and st.session_state.conversation_state:
            state = st.session_state.conversation_state
            st.subheader("📋 Conversation Info")
            st.text(f"ID: {state.get('conversation_id', 'N/A')[:8]}...")
            st.text(f"Agent: {state.get('current_agent', 'N/A')}")
            st.text(f"Turns: {state.get('turn_count', 0)}")

            entities = state.get("extracted_entities", {})
            if entities:
                st.subheader("🏷️ Extracted Entities")
                if entities.get("plan_type"):
                    st.text(f"Plan: {entities['plan_type']}")
                if entities.get("urgency"):
                    st.text(f"Urgency: {entities['urgency']}")
                if entities.get("sentiment"):
                    st.text(f"Sentiment: {entities['sentiment']}")
                if entities.get("issue_type"):
                    st.text(f"Issue: {entities['issue_type']}")

            # Handover history
            handovers = state.get("handover_history", [])
            if handovers:
                st.subheader("🔄 Handovers")
                for h in handovers:
                    st.text(f"{h.get('source_agent', '?')} → {h.get('target_agent', '?')}")
                    st.caption(h.get("reason", ""))

            # Status indicators
            st.subheader("📊 Status")
            if state.get("is_resolved"):
                st.success("✅ Resolved")
            elif state.get("is_escalated"):
                st.error("🚨 Escalated to Human")
            else:
                st.info("💬 In Progress")

            st.divider()

        # Tracing info
        tracing = get_tracing_config()
        st.subheader("🔍 Observability")
        if tracing["enabled"]:
            st.success(f"LangSmith: Enabled ({tracing['project']})")
        else:
            st.warning("LangSmith: Disabled")

        st.divider()

        # New conversation button
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_state = None
            st.rerun()

        st.divider()

        # Test scenarios
        st.subheader("🧪 Test Scenarios")
        scenarios = {
            "Scenario 1 - Single Agent": "My CloudDash alerts stopped firing after I updated my AWS integration credentials yesterday. I'm on the Pro plan.",
            "Scenario 2 - Cross-Agent": "I want to upgrade from Pro to Enterprise, but first can you check if the SSO integration issue I reported last week has been resolved?",
            "Scenario 3 - Escalation": "I've been charged twice for April. I need an immediate refund and I want to speak to a manager.",
            "Scenario 4 - KB Failure": "Does CloudDash support integration with Datadog for cross-platform alerting?",
        }
        for name, query in scenarios.items():
            if st.button(name, use_container_width=True, key=f"scenario_{name}"):
                st.session_state.messages = []
                st.session_state.conversation_state = None
                st.session_state.pending_message = query
                st.rerun()

    # Main chat area
    st.title("☁️ CloudDash Customer Support")
    st.caption("Powered by Multi-Agent AI System with RAG")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = None

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                # Show agent badge
                if msg.get("agent_id"):
                    st.markdown(get_agent_badge(msg["agent_id"]), unsafe_allow_html=True)

                # Handover events stored as special type
                if msg.get("type") == "handover":
                    st.markdown(
                        f'<div class="handover-event">'
                        f'🔄 <strong>Handover:</strong> '
                        f'{msg["content"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.write(msg["content"])
                    render_citations(msg.get("citations", []))

    # Handle pending message from scenario buttons
    pending = st.session_state.pop("pending_message", None)

    # Chat input
    user_input = pending or st.chat_input("How can we help you today?")

    if user_input:
        # Add user message to display
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Process through orchestrator
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    orch = get_orchestrator()
                    result = orch.process_message(
                        message=user_input,
                        existing_state=st.session_state.conversation_state,
                    )
                    st.session_state.conversation_state = result

                    # Display new assistant messages
                    for msg_data in result.get("messages", []):
                        role = msg_data.get("role", "")
                        if hasattr(role, "value"):
                            role = role.value
                        if role == "assistant":
                            agent_id = msg_data.get("agent_id", "")
                            content = msg_data.get("content", "")
                            metadata = msg_data.get("metadata", {})
                            citations = metadata.get("citations", [])

                            # Check if this message is already displayed
                            already_shown = any(
                                m.get("content") == content and m.get("role") == "assistant"
                                for m in st.session_state.messages
                            )

                            if not already_shown:
                                st.markdown(get_agent_badge(agent_id), unsafe_allow_html=True)
                                st.write(content)

                                render_citations(citations)

                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": content,
                                    "agent_id": agent_id,
                                    "citations": citations,
                                })

                    # Show handover events — save to session state so they persist
                    handovers = result.get("handover_history", [])
                    if handovers:
                        latest = handovers[-1]
                        src = latest.get('source_agent', '?')
                        tgt = latest.get('target_agent', '?')
                        reason = latest.get('reason', '')
                        handover_text = f"{src} → {tgt} | {reason}"

                        # Only add if not already in messages
                        already_shown = any(
                            m.get("type") == "handover" and m.get("content") == handover_text
                            for m in st.session_state.messages
                        )
                        if not already_shown:
                            st.markdown(
                                f'<div class="handover-event">'
                                f'🔄 <strong>Handover:</strong> {handover_text}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            st.session_state.messages.append({
                                "role": "assistant",
                                "type": "handover",
                                "content": handover_text,
                                "agent_id": "",
                            })

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"I apologize, but I encountered an error. Please try again. Error: {str(e)}",
                        "agent_id": "system",
                    })


if __name__ == "__main__":
    main()
