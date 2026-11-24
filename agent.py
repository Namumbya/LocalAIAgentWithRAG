"""Shared agent helpers for CLI and Streamlit UI."""

from __future__ import annotations

from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from constants import (
    CHAT_MODEL,
    DEFAULT_TEMPERATURE,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)
from logging_setup import get_logger
from tools import filter_reviews_by_rating, get_rating_stats, search_reviews

logger = get_logger(__name__)

_GREETINGS = {"hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"}
_REVIEW_HINTS = (
    "order",
    "review",
    "complaint",
    "rating",
    "delivery",
    "pizza",
    "draft",
    "reply",
    "owner",
    "vegan",
    "bbq",
    "worst",
    "stats",
    "distribution",
    "crust",
    "late",
)
_PREVIEW_LIMIT = 120

SYSTEM_PROMPT = """You are a local customer-review operations assistant.
The sample dataset is pizza restaurant reviews, but answer from whatever
reviews are indexed.

Tools (use exact argument names only):
- search_reviews(query): semantic search for a topic (BBQ, vegan, delivery, crust, etc.)
- filter_reviews_by_rating(min_rating, max_rating): filter by stars only. No topic argument.
- get_rating_stats(): overall counts/average/distribution only

Tool choice (follow strictly):
- "Should I order X?" / opinions about a dish or topic -> MUST call search_reviews(query="X") before answering. Do not answer from memory.
- worst complaints / 1-2 star issues -> filter_reviews_by_rating(1, 2)
- rating distribution / average -> get_rating_stats()
- Never call filter_reviews_by_rating for topical dish questions
- Never invent extra tool arguments (no topic= on filter_reviews_by_rating)
- If you have not called a tool yet for a review question, call one now.

Answer style:
- Normal questions: answer helpfully in plain chat. Cite 1-3 review points from tool results.
- "Should I order...?" -> after search_reviews, give a short yes/no/mixed recommendation using only those results.
- ONLY when the user asks to draft/write an owner reply/letter/email: output this format
  (after searching first):

Dear Customer,

Thank you for sharing your experience regarding [issue]. We apologize for [specific problem from the reviews]. [One sentence about what you will improve.]

We value your feedback and hope to serve you better next time.

Best regards,
[Restaurant Owner]

Do NOT use the Dear Customer letter for ordinary questions like menu advice.

Other rules:
- Ground answers in tool results. Do not invent reviews.
- For complaint summaries, mention 2-3 themes or reviews, not only one.
- Greetings with no review need -> answer directly, no tools
- Be clear and practical
"""


class OllamaNotAvailableError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "")
    return str(getattr(message, "type", "") or getattr(message, "role", ""))


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def _needs_review_tools(text: str) -> bool:
    cleaned = " ".join(text.lower().strip().split())
    if not cleaned or cleaned in _GREETINGS:
        return False
    return any(hint in cleaned for hint in _REVIEW_HINTS)


def _is_owner_reply_request(question: str) -> bool:
    lowered = question.lower()
    return any(
        token in lowered
        for token in ("draft", "owner reply", "write a reply", "letter", "email")
    )


def _history_has_tool_evidence(messages: list[Any]) -> bool:
    """True if earlier turns already retrieved tool evidence."""
    for message in messages:
        role = _message_role(message)
        if role == "tool":
            return True
        if getattr(message, "tool_calls", None):
            return True
        content = message_text(_message_content(message))
        if content.startswith("Tool result ("):
            return True
    return False


def _preview_text(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    compact = text.replace("\n", " ")
    if len(compact) > limit:
        return compact[: limit - 3] + "..."
    return compact


def _grounded_fallback_prompt(question: str, tool_name: str, tool_result: str) -> tuple[str, str]:
    """Return (system_content, user_prompt) for the grounded fallback answer."""
    if _is_owner_reply_request(question):
        system_content = (
            "You write finished customer apology letters. "
            "Never leave placeholders like [issue] or [Your Name]."
        )
        user_prompt = (
            "Write one finished customer apology letter from this evidence.\n"
            f"Evidence:\n{tool_result}\n\n"
            "Rules:\n"
            "- Start with Dear Customer,\n"
            "- Mention the real issue from the evidence (e.g. long wait / late delivery)\n"
            "- Apologize for a concrete detail from the evidence\n"
            "- Promise one specific improvement\n"
            "- End with Best regards, then Restaurant Owner\n"
            "- No brackets, no placeholders, no [issue] text\n\n"
            "Example of a finished letter:\n"
            "Dear Customer,\n\n"
            "Thank you for sharing your experience regarding late delivery. "
            "We apologize for the 90-minute wait on a busy night. "
            "We are adding more drivers at peak hours to improve delivery times.\n\n"
            "We value your feedback and hope to serve you better next time.\n\n"
            "Best regards,\n"
            "Restaurant Owner"
        )
        return system_content, user_prompt

    system_content = (
        "You answer from provided review evidence only. Be concise and practical."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Evidence from `{tool_name}`:\n{tool_result}\n\n"
        "Answer using ONLY this evidence. Do not invent reviews. "
        "Give a short practical recommendation."
    )
    return system_content, user_prompt


def check_ollama(base_url: str = OLLAMA_BASE_URL) -> None:
    """Fail fast with a clear message if Ollama is not running."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        response.raise_for_status()
    except Exception as exc:
        logger.error("Ollama health check failed at %s: %s", base_url, exc)
        raise OllamaNotAvailableError(
            "Cannot reach Ollama at "
            f"{base_url}. Install it from https://ollama.com/download, "
            "start the app, then run:\n"
            f"  ollama pull {CHAT_MODEL}\n"
            f"  ollama pull {EMBEDDING_MODEL}"
        ) from exc
    logger.info("Ollama reachable at %s", base_url)


def build_agent(
    model_name: str = CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    base_url: str = OLLAMA_BASE_URL,
) -> Any:
    """Build a LangChain agent with review tools."""
    logger.info(
        "Building agent model=%s temperature=%s", model_name, temperature
    )
    model = ChatOllama(
        model=model_name, temperature=temperature, base_url=base_url
    )
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


def tool_trace(history: list[Any], start_index: int = 0) -> list[str]:
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
            preview = _preview_text(message_text(message.content))
            lines.append(f"`{name}` returned: {preview}")
    return lines


def _fallback_tool_for_question(question: str) -> tuple[str, dict[str, Any], str]:
    """Pick a deterministic tool when the model skips tool calling."""
    args: dict[str, Any]
    lowered = question.lower()
    if any(token in lowered for token in ("worst", "complaint", "1-2", "1 star")):
        args = {"min_rating": 1, "max_rating": 2}
        return "filter_reviews_by_rating", args, filter_reviews_by_rating.invoke(args)
    if any(
        token in lowered
        for token in ("distribution", "average rating", "rating stats", "how many stars")
    ):
        args = {}
        return "get_rating_stats", args, get_rating_stats.invoke(args)

    query = question
    if "late delivery" in lowered:
        query = "late delivery"
    elif "bbq" in lowered:
        query = "BBQ chicken pizza"
    elif "vegan" in lowered:
        query = "vegan pizza"
    args = {"query": query}
    return "search_reviews", args, search_reviews.invoke(args)


def run_turn(
    agent: Any,
    history: list[Any],
    question: str,
    *,
    model_name: str = CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    base_url: str = OLLAMA_BASE_URL,
) -> tuple[list[Any], str, list[str]]:
    """Append a user question, invoke the agent, return history + answer + tool trace."""
    history = list(history)
    start_index = len(history)
    prior_history = list(history)
    history.append({"role": "user", "content": question})
    try:
        result = agent.invoke({"messages": history})
    except httpx.ConnectError as exc:
        logger.error("Lost Ollama connection during turn: %s", exc)
        raise OllamaNotAvailableError(
            "Lost connection to Ollama while generating a reply. "
            "Make sure the Ollama app is running."
        ) from exc

    history = result["messages"]
    answer = message_text(history[-1].content)
    trace = tool_trace(history, start_index=start_index)

    # Small local models sometimes skip tools and invent answers. Recover once,
    # but only when earlier turns have no tool evidence to reuse.
    if (
        not trace
        and _needs_review_tools(question)
        and not _history_has_tool_evidence(prior_history)
    ):
        try:
            tool_name, tool_args, tool_result = _fallback_tool_for_question(question)
        except Exception as exc:
            logger.exception("Fallback tool failed")
            raise RuntimeError(
                f"Could not retrieve reviews for fallback answering: {exc}"
            ) from exc

        logger.warning(
            "Model skipped tools; fallback %s with %s", tool_name, tool_args
        )
        system_content, grounded_prompt = _grounded_fallback_prompt(
            question, tool_name, tool_result
        )
        try:
            answerer = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
            )
            grounded = answerer.invoke(
                [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": grounded_prompt},
                ]
            )
        except httpx.ConnectError as exc:
            logger.error("Lost Ollama connection during fallback turn: %s", exc)
            raise OllamaNotAvailableError(
                "Lost connection to Ollama while generating a reply. "
                "Make sure the Ollama app is running."
            ) from exc

        answer = message_text(grounded.content)
        # Drop the ungrounded assistant reply and keep tool evidence for follow-ups.
        history = list(history[:-1])
        history.append(
            {
                "role": "assistant",
                "content": f"Tool result ({tool_name} {tool_args}):\n{tool_result}",
            }
        )
        history.append({"role": "assistant", "content": answer})
        trace = [
            f"Called `{tool_name}` with {tool_args} (fallback)",
            f"`{tool_name}` returned: {_preview_text(tool_result)}",
        ]

    logger.info("Q: %s", question)
    if trace:
        for line in trace:
            logger.info("%s", line)
    else:
        logger.info("No tools used (answered directly)")
    logger.info("A: %s", answer[:200] + ("..." if len(answer) > 200 else ""))

    return history, answer, trace
