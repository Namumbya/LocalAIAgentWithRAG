"""Shared agent helpers for CLI and Streamlit UI."""

from __future__ import annotations

import re
import uuid
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

import guardrails as g
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
    "worst",
    "stats",
    "distribution",
    "crust",
    "late",
)
_PREVIEW_LIMIT = 120
_FOLLOW_UP_PATTERN = re.compile(
    r"\b(that|this|those|these|it|them|above|same|previous)\b", re.IGNORECASE
)

SYSTEM_PROMPT = """You are a local customer-review operations assistant.
The sample dataset is pizza restaurant reviews, but answer from whatever
reviews are indexed.

Tools (use exact argument names only):
- search_reviews(query): semantic search for a topic (pepperoni, vegan, delivery, crust, etc.)
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
- If the search results are not actually about the item asked about, say the reviews do not cover it. Never recommend an item no review mentions.
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

Evidence = tuple[str, dict[str, Any], str]  # (tool_name, args, result)


class OllamaNotAvailableError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "")
    return str(getattr(message, "type", "") or getattr(message, "role", ""))


def _needs_review_tools(text: str) -> bool:
    cleaned = " ".join(text.lower().strip().split())
    if not cleaned or cleaned in _GREETINGS:
        return False
    return any(hint in cleaned for hint in _REVIEW_HINTS)


def _is_follow_up(question: str) -> bool:
    """True if the question refers back to the previous answer ("draft a reply about that")."""
    return _FOLLOW_UP_PATTERN.search(question) is not None


def _last_turn_has_tool_evidence(messages: list[Any]) -> bool:
    """True if the most recent completed turn retrieved tool evidence."""
    last_user = -1
    for index, message in enumerate(messages):
        if _message_role(message) in {"human", "user"}:
            last_user = index
    if last_user < 0:
        return False
    return any(
        _message_role(message) == "tool" or getattr(message, "tool_calls", None)
        for message in messages[last_user + 1 :]
    )


def _preview_text(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    compact = text.replace("\n", " ")
    if len(compact) > limit:
        return compact[: limit - 3] + "..."
    return compact


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


def _fallback_tool_for_question(question: str) -> Evidence:
    """Run the right tool with the right arguments for a question, deterministically."""
    args: dict[str, Any]
    intent = g.classify_intent(question)
    lowered = question.lower()
    if intent == g.COMPLAINTS:
        args = {"min_rating": 1, "max_rating": 2}
        return "filter_reviews_by_rating", args, filter_reviews_by_rating.invoke(args)
    if intent == g.STATS:
        args = {}
        return "get_rating_stats", args, get_rating_stats.invoke(args)

    if intent == g.OWNER_REPLY:
        query = g.issue_from_question(question)
    elif intent == g.RECOMMEND:
        query = g.dish_topic(question)
    elif "late delivery" in lowered:
        query = "late delivery"
    elif "vegan" in lowered:
        query = "vegan pizza"
    else:
        query = question
    args = {"query": query}
    return "search_reviews", args, search_reviews.invoke(args)


def _model_evidence(intent: str, new_messages: list[Any]) -> Evidence | None:
    """The model's own tool call for this turn, if it is valid evidence for `intent`."""
    results = {
        getattr(message, "tool_call_id", ""): message_text(message.content)
        for message in new_messages
        if _message_role(message) == "tool"
    }
    for message in new_messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = str(call.get("name", ""))
            args = dict(call.get("args") or {})
            call_id = call.get("id", "")
            if call_id in results and g.evidence_fits_intent(intent, name, args):
                return name, args, results[call_id]
    return None


def _ask_model(
    prompt: tuple[str, str], *, model_name: str, temperature: float, base_url: str
) -> str:
    system_content, user_content = prompt
    try:
        answerer = ChatOllama(model=model_name, temperature=temperature, base_url=base_url)
        reply = answerer.invoke(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
        )
    except httpx.ConnectError as exc:
        logger.error("Lost Ollama connection while writing an answer: %s", exc)
        raise OllamaNotAvailableError(
            "Lost connection to Ollama while generating a reply. "
            "Make sure the Ollama app is running."
        ) from exc
    return message_text(reply.content)


def _compose_answer(
    intent: str | None,
    question: str,
    evidence: Evidence,
    *,
    model_name: str,
    temperature: float,
    base_url: str,
) -> str:
    """Build the answer from evidence; the model only fills short free-text parts."""
    tool_name, tool_args, tool_result = evidence

    def ask(prompt: tuple[str, str]) -> str:
        return _ask_model(
            prompt, model_name=model_name, temperature=temperature, base_url=base_url
        )

    if intent == g.STATS:
        return g.stats_answer(tool_result)
    if intent == g.COMPLAINTS:
        return g.complaints_answer(ask(g.complaints_prompt(tool_result)), tool_result)
    if intent == g.OWNER_REPLY:
        issue = g.issue_from_question(question)
        return g.letter_answer(issue, ask(g.letter_prompt(issue, tool_result)))
    if intent == g.RECOMMEND:
        topic = g.dish_topic(question) or str(tool_args.get("query", ""))
        missing = g.missing_topic_answer(topic, tool_result)
        if missing is not None:
            logger.info("Search results do not mention %r; not asking the model", topic)
            return missing
        reviews = g.relevant_reviews(topic, tool_result)
        summary = ask(g.recommendation_prompt(topic, reviews)) if reviews else ""
        if not reviews:
            # Unparseable evidence: fall back to a plain grounded answer.
            summary = ask(g.general_prompt(question, tool_name, tool_result))
        return g.recommendation_answer(topic, reviews, summary)
    return ask(g.general_prompt(question, tool_name, tool_result))


def _with_evidence(history: list[Any], user_index: int, evidence: Evidence, answer: str) -> list[Any]:
    """History up to the user message, then the evidence as a real tool call, then the answer."""
    tool_name, tool_args, tool_result = evidence
    call_id = f"call-{uuid.uuid4().hex[:12]}"
    return [
        *history[: user_index + 1],
        AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": tool_args, "id": call_id, "type": "tool_call"}],
        ),
        ToolMessage(content=tool_result, tool_call_id=call_id, name=tool_name),
        AIMessage(content=answer),
    ]


def run_turn(
    agent: Any,
    history: list[Any],
    question: str,
    *,
    model_name: str = CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    base_url: str = OLLAMA_BASE_URL,
) -> tuple[list[Any], str, list[str]]:
    """Append a user question, invoke the agent, return history + answer + tool trace.

    The agent runs first. For the question types in `guardrails` the code then
    checks that the model used valid evidence (re-running the right tool if
    not) and builds the answer around it. Other review questions where the
    model skipped tools get one grounded retry.
    """
    history = list(history)
    user_index = len(history)
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
    trace = tool_trace(history, start_index=user_index)
    new_messages = history[user_index + 1 :]

    # A follow-up ("draft a reply about that") can rely on last turn's evidence.
    reuses_evidence = _is_follow_up(question) and _last_turn_has_tool_evidence(prior_history)
    intent = None if reuses_evidence else g.classify_intent(question)
    needs_retry = not trace and _needs_review_tools(question) and not reuses_evidence

    if intent is not None or needs_retry:
        evidence = _model_evidence(intent, new_messages) if intent is not None else None
        if evidence is None:
            try:
                evidence = _fallback_tool_for_question(question)
            except Exception as exc:
                logger.exception("Guardrail tool call failed")
                raise RuntimeError(f"Could not retrieve reviews: {exc}") from exc
            tool_name, tool_args, tool_result = evidence
            reason = "model skipped tools" if not trace else "model's tool call did not fit"
            logger.warning("Guardrail (%s): %s with %s", reason, tool_name, tool_args)
            trace = [
                *trace,
                f"Called `{tool_name}` with {tool_args} (guardrail: {reason})",
                f"`{tool_name}` returned: {_preview_text(tool_result)}",
            ]
        answer = _compose_answer(
            intent,
            question,
            evidence,
            model_name=model_name,
            temperature=temperature,
            base_url=base_url,
        )
        history = _with_evidence(history, user_index, evidence, answer)

    logger.info("Q: %s", question)
    if trace:
        for line in trace:
            logger.info("%s", line)
    else:
        logger.info("No tools used (answered directly)")
    logger.info("A: %s", answer[:200] + ("..." if len(answer) > 200 else ""))

    return history, answer, trace
