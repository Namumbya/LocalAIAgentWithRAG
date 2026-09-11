"""CLI entrypoint for the local restaurant RAG agent."""

from __future__ import annotations

from agent import CHAT_MODEL, OllamaNotAvailableError, build_agent, check_ollama, run_turn


def main() -> None:
    try:
        check_ollama()
    except OllamaNotAvailableError as exc:
        print(exc)
        return

    agent = build_agent()
    messages: list = []

    print("Local Restaurant Agent (RAG + tools)")
    print(f"Model: {CHAT_MODEL}")
    print("Ask about reviews, complaints, or draft owner replies. Type q to quit.\n")

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
