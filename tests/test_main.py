from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage

import agent
import main


def test_build_agent_wires_tools(monkeypatch):
    fake_model = object()
    captured = {}

    def fake_chat_ollama(*, model, temperature, base_url=None):
        captured["model"] = model
        captured["temperature"] = temperature
        captured["base_url"] = base_url
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
    assert captured["temperature"] == 0.0
    assert captured["base_url"] == agent.OLLAMA_BASE_URL
    assert captured["agent_model"] is fake_model
    assert len(captured["tools"]) == 3
    assert "search_reviews" in captured["system_prompt"]
    assert "Dear Customer" in captured["system_prompt"]


def test_message_text_handles_string_and_blocks():
    assert agent.message_text("hello") == "hello"
    assert agent.message_text([{"type": "text", "text": "hi"}, " there"]) == "hi\n there"


def test_needs_review_tools():
    assert agent._needs_review_tools("Should I order the BBQ chicken pizza?")
    assert agent._needs_review_tools("What are the worst complaints?")
    assert not agent._needs_review_tools("Hi")
    assert not agent._needs_review_tools("thanks")


def test_fallback_tool_for_question_routes(monkeypatch):
    monkeypatch.setattr(
        agent,
        "filter_reviews_by_rating",
        MagicMock(invoke=MagicMock(return_value="low ratings")),
    )
    monkeypatch.setattr(
        agent,
        "get_rating_stats",
        MagicMock(invoke=MagicMock(return_value="stats")),
    )
    monkeypatch.setattr(
        agent,
        "search_reviews",
        MagicMock(invoke=MagicMock(return_value="search hits")),
    )

    name, args, result = agent._fallback_tool_for_question("What are the worst complaints?")
    assert name == "filter_reviews_by_rating"
    assert args == {"min_rating": 1, "max_rating": 2}
    assert result == "low ratings"

    name, args, result = agent._fallback_tool_for_question("Summarize the rating distribution")
    assert name == "get_rating_stats"
    assert args == {}
    assert result == "stats"

    name, args, result = agent._fallback_tool_for_question(
        "Should I order the BBQ chicken pizza?"
    )
    assert name == "search_reviews"
    assert args == {"query": "BBQ chicken pizza"}
    assert result == "search hits"


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


def test_run_turn_fallback_replaces_ungrounded_answer(monkeypatch):
    class FakeMessage:
        def __init__(self, content):
            self.content = content

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "messages": [
            {"role": "user", "content": "Should I order the BBQ chicken pizza?"},
            FakeMessage("Invented praise with no tools."),
        ]
    }

    monkeypatch.setattr(
        agent,
        "_fallback_tool_for_question",
        lambda question: (
            "search_reviews",
            {"query": "BBQ chicken pizza"},
            "[1] Rating: 5/5 BBQ is great",
        ),
    )

    fake_answerer = MagicMock()
    fake_answerer.invoke.return_value = FakeMessage("Yes, order it.")

    def fake_chat_ollama(*, model, temperature, base_url=None):
        assert model == "test-model"
        assert temperature == 0.3
        assert base_url == agent.OLLAMA_BASE_URL
        return fake_answerer

    monkeypatch.setattr(agent, "ChatOllama", fake_chat_ollama)

    history, answer, trace = agent.run_turn(
        fake_agent,
        [],
        "Should I order the BBQ chicken pizza?",
        model_name="test-model",
        temperature=0.3,
    )

    assert answer == "Yes, order it."
    assert "fallback" in trace[0]
    tool_call, tool_msg, final = history[-3:]
    assert isinstance(tool_call, AIMessage)
    assert tool_call.tool_calls[0]["name"] == "search_reviews"
    assert tool_call.tool_calls[0]["args"] == {"query": "BBQ chicken pizza"}
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == tool_call.tool_calls[0]["id"]
    assert tool_msg.content == "[1] Rating: 5/5 BBQ is great"
    assert isinstance(final, AIMessage)
    assert final.content == "Yes, order it."
    assert "Invented praise" not in str(history)


def _prior_turn_with_tool_evidence():
    call = {
        "name": "filter_reviews_by_rating",
        "args": {"min_rating": 1, "max_rating": 2},
        "id": "call-1",
        "type": "tool_call",
    }
    return [
        {"role": "user", "content": "What are the worst complaints?"},
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content="late delivery", tool_call_id="call-1", name=call["name"]),
        AIMessage(content="Main complaints are late delivery."),
    ]


def test_is_follow_up():
    assert agent._is_follow_up("Draft a reply about that")
    assert agent._is_follow_up("Write an owner reply to those complaints")
    assert not agent._is_follow_up("Is the vegan pizza any good?")
    assert not agent._is_follow_up("Should I order the BBQ chicken pizza?")


def test_run_turn_fallback_runs_for_new_topic_after_tool_turn(monkeypatch):
    class FakeMessage:
        def __init__(self, content):
            self.content = content

    prior = _prior_turn_with_tool_evidence()
    question = "Is the vegan pizza any good?"
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "messages": prior
        + [{"role": "user", "content": question}, FakeMessage("Invented vegan praise.")]
    }
    fallback = MagicMock(
        return_value=("search_reviews", {"query": "vegan pizza"}, "vegan hits")
    )
    monkeypatch.setattr(agent, "_fallback_tool_for_question", fallback)
    fake_answerer = MagicMock()
    fake_answerer.invoke.return_value = FakeMessage("Mixed reviews.")
    monkeypatch.setattr(agent, "ChatOllama", lambda **kwargs: fake_answerer)

    _history, answer, trace = agent.run_turn(fake_agent, prior, question)

    fallback.assert_called_once_with(question)
    assert answer == "Mixed reviews."
    assert "fallback" in trace[0]


def test_run_turn_skips_fallback_when_prior_tool_evidence_exists(monkeypatch):
    class FakeMessage:
        def __init__(self, content):
            self.content = content

    prior = _prior_turn_with_tool_evidence()
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "messages": prior
        + [
            {"role": "user", "content": "Draft a reply about that"},
            FakeMessage("Dear Customer, sorry about late delivery."),
        ]
    }
    fallback = MagicMock(
        side_effect=AssertionError("fallback should not run with prior evidence")
    )
    monkeypatch.setattr(agent, "_fallback_tool_for_question", fallback)

    history, answer, trace = agent.run_turn(
        fake_agent, prior, "Draft a reply about that"
    )

    assert answer == "Dear Customer, sorry about late delivery."
    assert trace == []
    fallback.assert_not_called()
    assert history[-1].content == "Dear Customer, sorry about late delivery."


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
        raise AssertionError("expected OllamaNotAvailableError")
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


def test_cli_exits_cleanly_on_eof_at_prompt(monkeypatch, capsys):
    monkeypatch.setattr(main, "check_ollama", lambda: None)
    monkeypatch.setattr(main, "build_agent", lambda: MagicMock(name="agent"))

    def eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    main.main()
    assert "Bye." in capsys.readouterr().out
