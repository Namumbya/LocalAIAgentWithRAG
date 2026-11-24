"""CLI entrypoint for the local agentic RAG demo."""

from __future__ import annotations

from agent import OllamaNotAvailableError, build_agent, check_ollama, run_turn
from constants import CHAT_MODEL
from logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    try:
        check_ollama()
    except OllamaNotAvailableError as exc:
        print(exc)
        return

    agent = build_agent()
    messages: list[object] = []

    print("Local AI Agent with RAG (tools + reviews)")
    print(f"Model: {CHAT_MODEL}")
    print(
        "Ask about the sample reviews, complaints, or draft owner replies. "
        "Type q to quit.\n"
    )
    logger.info("CLI agent ready")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue

        try:
            messages, answer, _trace = run_turn(agent, messages, question)
        except OllamaNotAvailableError as exc:
            print(f"\n{exc}\n")
            continue

        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
