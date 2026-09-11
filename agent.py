"""Shared agent helpers for CLI and Streamlit UI."""

from __future__ import annotations

from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from tools import filter_reviews_by_rating, get_rating_stats, search_reviews

# Smaller defaults to save disk (~2–3 GB total). Use qwen2.5 / mxbai-embed-large if you have space.
CHAT_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

SYSTEM_PROMPT = """You are a local restaurant operations assistant for a pizza place.

Tools:
- search_reviews(query): find reviews about a topic (delivery, crust, vegan, BBQ, noise, etc.)
- filter_reviews_by_rating(min_rating, max_rating): list complaints (1-2), mixed (3), or praise (4-5)
- get_rating_stats(): ONLY for overall counts/average/distribution questions

Tool choice:
- worst complaints / bad ratings -> filter_reviews_by_rating(1, 2)
- a topic (late delivery, BBQ pizza, vegan) -> search_reviews with that topic
- rating distribution / average -> get_rating_stats
- Never use get_rating_stats for drafting replies or topical questions

When asked to draft an owner reply:
1. Call search_reviews (or filter low ratings) for the issue first
2. Write a complete email-style reply in this format:

Dear Customer,

Thank you for sharing your experience regarding [issue]. We apologize for [specific problem from the reviews]. [One sentence about what you will improve.]

We value your feedback and hope to serve you better next time.

Best regards,
[Restaurant Owner]

Do not ask the user to write the reply. Do not answer as if you are apologizing in chat — output the letter itself.

Other rules:
- Ground answers in tool results. Do not invent reviews.
- For complaint summaries, mention 2-3 themes or reviews, not only one.
- Greetings with no review need -> answer directly, no tools
- Be clear and practical
"""


class OllamaNotAvailableError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


def check_ollama(base_url: str = OLLAMA_BASE_URL) -> None:
    """Fail fast with a clear message if Ollama is not running."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        response.raise_for_status()
    except Exception as exc:
        raise OllamaNotAvailableError(
            "Cannot reach Ollama at "
            f"{base_url}. Install it from https://ollama.com/download, "
            "start the app, then run:\n"
            "  ollama pull qwen2.5:3b\n"
            "  ollama pull nomic-embed-text"
        ) from exc


def build_agent(model_name: str = CHAT_MODEL):
    model = ChatOllama(model=model_name, temperature=0)
    return create_agent(
        model=model,
        tools=[search_reviews, filter_reviews_by_rating, get_rating_stats],
        system_prompt=SYSTEM_PROMPT,
    )


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def tool_trace(history: list, start_index: int = 0) -> list[str]:
    """Summarize tool calls from new messages in the agent history."""
    lines: list[str] = []
    for message in history[start_index:]:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name", "unknown")
            args = call.get("args", {})
            lines.append(f"Called `{name}` with {args}")

        msg_type = getattr(message, "type", None)
        if msg_type == "tool":
            name = getattr(message, "name", "tool")
            preview = message_text(message.content).replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            lines.append(f"`{name}` returned: {preview}")
    return lines


def run_turn(agent, history: list, question: str) -> tuple[list, str, list[str]]:
    """Append a user question, invoke the agent, return history + answer + tool trace."""
    history = list(history)
    start_index = len(history)
    history.append({"role": "user", "content": question})
    try:
        result = agent.invoke({"messages": history})
    except httpx.ConnectError as exc:
        raise OllamaNotAvailableError(
            "Lost connection to Ollama while generating a reply. "
            "Make sure the Ollama app is running."
        ) from exc

    history = result["messages"]
    answer = message_text(history[-1].content)
    trace = tool_trace(history, start_index=start_index)

    print(f"\n[agent] Q: {question}")
    if trace:
        for line in trace:
            print(f"[agent] {line}")
    else:
        print("[agent] No tools used (answered directly)")
    print(f"[agent] A: {answer[:200]}{'...' if len(answer) > 200 else ''}\n")

    return history, answer, trace
