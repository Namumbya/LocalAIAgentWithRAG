"""Streamlit chat UI for the local restaurant RAG agent."""

from __future__ import annotations

import streamlit as st

from agent import (
    CHAT_MODEL,
    OllamaNotAvailableError,
    build_agent,
    check_ollama,
    run_turn,
)

st.set_page_config(
    page_title="Local Restaurant Agent",
    page_icon=":pizza:",
    layout="centered",
)

st.title("Local Restaurant Agent")
st.caption(
    f"LangChain v1 + Ollama (`{CHAT_MODEL}`) · RAG tools over pizza reviews"
)

EXAMPLE_PROMPTS = [
    "Should I order the BBQ chicken pizza?",
    "What are the worst complaints?",
    "Summarize the rating distribution",
    "Draft an owner reply about late delivery",
]


@st.cache_resource(show_spinner="Starting local agent...")
def get_agent():
    check_ollama()
    return build_agent()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []


with st.sidebar:
    st.header("About")
    st.write(
        "This agent decides when to search reviews, filter by rating, "
        "or pull stats before answering."
    )
    st.caption(
        "Clear chat resets the UI conversation only. "
        "The terminal keeps older lines (terminals never erase history)."
    )
    st.divider()
    st.subheader("Try asking")
    for prompt in EXAMPLE_PROMPTS:
        if st.button(prompt, use_container_width=True):
            st.session_state.pending_prompt = prompt
            st.rerun()
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_history = []
        st.session_state.pop("pending_prompt", None)
        print("\n======== CHAT CLEARED (UI session reset) ========\n")
        st.rerun()

    st.caption(f"Messages in this chat: {len(st.session_state.messages)}")


try:
    agent = get_agent()
except OllamaNotAvailableError as exc:
    st.error(str(exc))
    st.stop()


prompt = st.session_state.pop("pending_prompt", None) or st.chat_input(
    "Ask about the restaurant reviews..."
)

# Handle the turn first, then always render from session_state (stable history).
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking with local tools..."):
        try:
            history, answer, trace = run_turn(
                agent, st.session_state.agent_history, prompt
            )
            st.session_state.agent_history = history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "trace": trace}
            )
        except OllamaNotAvailableError as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": str(exc), "trace": []}
            )
    st.rerun()


if not st.session_state.messages:
    st.info("Ask a question or use a sidebar example to start a new chat.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            trace = message.get("trace") or []
            if trace:
                with st.expander("Tool activity"):
                    for line in trace:
                        st.markdown(f"- {line}")
            else:
                st.caption("No tools used for this reply.")
