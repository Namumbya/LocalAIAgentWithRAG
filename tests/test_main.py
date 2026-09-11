from __future__ import annotations

from unittest.mock import MagicMock

import agent
import main


def test_build_agent_wires_tools(monkeypatch):
    fake_model = object()
    captured = {}

    def fake_chat_ollama(*, model, temperature):
        captured["model"] = model
        captured["temperature"] = temperature
        return fake_model

    def fake_create_agent(*, model, tools, system_prompt):
        captured["agent_model"] = model
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        return MagicMock(name="agent")

    monkeypatch.setattr(agent, "ChatOllama", fake_chat_ollama)
    monkeypatch.setattr(agent, "create_agent", fake_create_agent)

    built = agent.build_agent(model_name="qwen2.5:3b")

    assert built is not None
    assert captured["model"] == "qwen2.5:3b"
    assert captured["temperature"] == 0
    assert captured["agent_model"] is fake_model
    assert len(captured["tools"]) == 3
    assert "search_reviews" in captured["system_prompt"]
    assert "Dear Customer" in captured["system_prompt"]


def test_message_text_handles_string_and_blocks():
    assert agent.message_text("hello") == "hello"
    assert agent.message_text([{"type": "text", "text": "hi"}, " there"]) == "hi\n there"


def test_run_turn_updates_history(monkeypatch):
    class FakeMessage:
        def __init__(self, content):
            self.content = content

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "messages": [
            {"role": "user", "content": "hi"},
            FakeMessage("hello back"),
        ]
    }

    history, answer, trace = agent.run_turn(fake_agent, [], "hi")

    assert answer == "hello back"
    assert trace == []
    assert len(history) == 2
    fake_agent.invoke.assert_called_once()


def test_tool_trace_from_tool_calls():
    class FakeAI:
        type = "ai"
        tool_calls = [{"name": "search_reviews", "args": {"query": "late"}}]
        content = ""

    class FakeTool:
        type = "tool"
        name = "search_reviews"
        content = "Found a late delivery review"

    lines = agent.tool_trace([FakeAI(), FakeTool()])
    assert lines[0].startswith("Called `search_reviews`")
    assert "returned:" in lines[1]


def test_check_ollama_raises_clear_error(monkeypatch):
    def boom(*args, **kwargs):
        raise agent.httpx.ConnectError("refused")

    monkeypatch.setattr(agent.httpx, "get", boom)

    try:
        agent.check_ollama()
        assert False, "expected OllamaNotAvailableError"
    except agent.OllamaNotAvailableError as exc:
        assert "Cannot reach Ollama" in str(exc)


def test_cli_exits_cleanly_when_ollama_down(monkeypatch, capsys):
    monkeypatch.setattr(
        main,
        "check_ollama",
        MagicMock(side_effect=agent.OllamaNotAvailableError("down")),
    )
    main.main()
    assert "down" in capsys.readouterr().out
