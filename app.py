"""Streamlit chat UI for the local agentic RAG demo."""

from __future__ import annotations

import streamlit as st

from agent import (
    OllamaNotAvailableError,
    build_agent,
    check_ollama,
    run_turn,
)
from constants import (
    CHAT_MODEL,
    CSV_PATH,
    DEFAULT_RETRIEVER_K,
    DEFAULT_TEMPERATURE,
    EMBEDDING_MODEL,
)
from logging_setup import get_logger, setup_logging
from vector import reset_vector_store, set_retriever_k

setup_logging()
logger = get_logger(__name__)

st.set_page_config(
    page_title="Local AI Agent with RAG",
    page_icon=":robot_face:",
    layout="centered",
)

st.title("Local AI Agent with RAG")
st.caption(
    f"LangChain v1 + Ollama (`{CHAT_MODEL}`) · agent tools over a sample "
    f"review dataset (`{CSV_PATH.name}`)"
)

EXAMPLE_PROMPTS = [
    "Should I order the BBQ chicken pizza?",
    "What are the worst complaints?",
    "Summarize the rating distribution",
    "Draft an owner reply about late delivery",
]


@st.cache_resource(show_spinner="Starting local agent...")
def get_agent(temperature: float):
    check_ollama()
    return build_agent(temperature=temperature)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []


with st.sidebar:
    st.caption(
        "Local agent with review tools. Models/paths: `constants.py`."
    )

    st.divider()
    st.subheader("Controls")
    temperature = st.slider(
        "Response temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULT_TEMPERATURE),
        step=0.1,
        help="0 is more deterministic; higher values are more varied.",
    )
    retriever_k = st.slider(
        "Reviews retrieved (k)",
        min_value=1,
        max_value=10,
        value=int(DEFAULT_RETRIEVER_K),
        step=1,
        help="How many similar reviews search_reviews returns.",
    )
    set_retriever_k(retriever_k)

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
        logger.info("Chat cleared (UI session reset)")
        st.rerun()

    if st.button("Reindex reviews", use_container_width=True):
        reset_vector_store()
        get_agent.clear()
        st.session_state.messages = []
        st.session_state.agent_history = []
        st.session_state.pop("pending_prompt", None)
        st.success(
            f"Cleared the local vector DB. Next question will re-index "
            f"`{CSV_PATH.name}` with `{EMBEDDING_MODEL}`."
        )
        st.rerun()

    st.caption(f"Messages in this chat: {len(st.session_state.messages)}")
    st.caption("Tune models/paths in `constants.py`.")


try:
    agent = get_agent(temperature)
except OllamaNotAvailableError as exc:
    st.error(str(exc))
    st.stop()


prompt = st.session_state.pop("pending_prompt", None) or st.chat_input(
    "Ask about the sample reviews..."
)

# Handle the turn first, then always render from session_state (stable history).
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking with local tools..."):
        try:
            history, answer, trace = run_turn(
                agent,
                st.session_state.agent_history,
                prompt,
                model_name=CHAT_MODEL,
                temperature=temperature,
            )
            st.session_state.agent_history = history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "trace": trace}
            )
        except (OllamaNotAvailableError, RuntimeError) as exc:
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
