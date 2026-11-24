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


# --------------------------------------------------------------------------- #
# Guardrailed run_turn behaviour
# --------------------------------------------------------------------------- #

PEPPERONI_HITS = (
    "[1] Rating: 5/5 | Date: 2024-03-15\nBest pizza in town\n"
    "The pepperoni curled up into little cups of deliciousness.\n\n"
    "[2] Rating: 4/5 | Date: 2024-02-01\nSolid pepperoni\nGood pepperoni, a bit greasy."
)
LOW_RATED = (
    "Found 3 reviews rated 1-2 (1 stars: 1, 2 stars: 2). Showing 3 spread across all "
    "matches, lowest rating first:\n\n"
    "Title: Pizza arrived upside-down\nDate: 2024-03-06\nRating: 1/5\nReview: Box was flipped.\n\n"
    "Title: Rude staff\nDate: 2024-03-11\nRating: 2/5\nReview: Cashier ignored us.\n\n"
    "Title: Late again\nDate: 2024-02-20\nRating: 2/5\nReview: Waited over an hour."
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


def _agent_returning(question, *messages, prior=()):
    fake = MagicMock()
    fake.invoke.return_value = {
        "messages": [*prior, {"role": "user", "content": question}, *messages]
    }
    return fake


def _model_tool_turn(name, args, result, final="model answer"):
    call = {"name": name, "args": args, "id": "c1", "type": "tool_call"}
    return (
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content=result, tool_call_id="c1", name=name),
        AIMessage(content=final),
    )


def _fake_llm(monkeypatch, reply):
    answerer = MagicMock()
    answerer.invoke.return_value = FakeMessage(reply)
    calls = []

    def fake_chat_ollama(**kwargs):
        calls.append(kwargs)
        return answerer

    monkeypatch.setattr(agent, "ChatOllama", fake_chat_ollama)
    return calls


def _no_llm(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(agent, "ChatOllama", boom)


def test_recommendation_when_model_skipped_tools(monkeypatch):
    question = "Should I order the pepperoni pizza?"
    fake_agent = _agent_returning(question, FakeMessage("Invented praise with no tools."))
    monkeypatch.setattr(
        agent,
        "_fallback_tool_for_question",
        lambda q: ("search_reviews", {"query": "pepperoni pizza"}, PEPPERONI_HITS),
    )
    calls = _fake_llm(monkeypatch, "Reviewers love the crispy pepperoni cups.")

    history, answer, trace = agent.run_turn(
        fake_agent, [], question, model_name="test-model", temperature=0.3
    )

    assert answer.startswith("**Yes.** 2 retrieved reviews mention the pepperoni pizza")
    assert "Reviewers love the crispy pepperoni cups." in answer
    assert '"Best pizza in town" (5/5)' in answer
    assert "guardrail: model skipped tools" in trace[0]
    assert calls[0]["model"] == "test-model"
    assert calls[0]["temperature"] == 0.3
    assert calls[0]["base_url"] == agent.OLLAMA_BASE_URL
    tool_call, tool_msg, final = history[-3:]
    assert isinstance(tool_call, AIMessage)
    assert tool_call.tool_calls[0]["args"] == {"query": "pepperoni pizza"}
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == tool_call.tool_calls[0]["id"]
    assert final.content == answer
    assert "Invented praise" not in str(history)


def test_recommendation_uses_model_search_when_valid(monkeypatch):
    question = "Should I order the pepperoni pizza?"
    fake_agent = _agent_returning(
        question, *_model_tool_turn("search_reviews", {"query": "pepperoni"}, PEPPERONI_HITS, "Yes")
    )
    fallback = MagicMock(side_effect=AssertionError("model evidence was valid"))
    monkeypatch.setattr(agent, "_fallback_tool_for_question", fallback)
    _fake_llm(monkeypatch, "Crispy, well-loved pepperoni.")

    history, answer, trace = agent.run_turn(fake_agent, [], question)

    assert answer.startswith("**Yes.**")
    assert "What reviewers said:" in answer
    assert all("guardrail" not in line for line in trace)
    assert history[-1].content == answer
    # The model's bare "Yes" is replaced, not kept alongside.
    assert sum(isinstance(m, AIMessage) and m.content == "Yes" for m in history) == 0


def test_recommendation_refuses_unreviewed_dish_from_model_search(monkeypatch):
    question = "Should I order the BBQ chicken pizza?"
    hits = "[1] Rating: 5/5 | Date: 2024-02-15\nBrilliant buffalo chicken pizza\nSpicy and creamy."
    fake_agent = _agent_returning(
        question,
        *_model_tool_turn("search_reviews", {"query": "bbq"}, hits, "Yes, it's great!"),
    )
    _no_llm(monkeypatch)

    history, answer, _trace = agent.run_turn(fake_agent, [], question)

    assert answer.startswith("None of the reviews mention BBQ chicken pizza")
    assert history[-1].content == answer


def test_recommendation_refuses_unreviewed_dish_from_fallback(monkeypatch):
    question = "Should I order the BBQ chicken pizza?"
    fake_agent = _agent_returning(question, FakeMessage("Yes, order it!"))
    monkeypatch.setattr(
        agent,
        "_fallback_tool_for_question",
        lambda q: ("search_reviews", {"query": "BBQ chicken pizza"}, "buffalo chicken hits"),
    )
    _no_llm(monkeypatch)

    _history, answer, _trace = agent.run_turn(fake_agent, [], question)

    assert answer.startswith("None of the reviews mention BBQ chicken pizza")


def test_complaints_rerun_when_model_used_wrong_ratings(monkeypatch):
    question = "What are the worst complaints?"
    fake_agent = _agent_returning(
        question,
        *_model_tool_turn(
            "filter_reviews_by_rating", {"min_rating": 1, "max_rating": 5}, "all reviews incl. praise"
        ),
    )
    low = MagicMock(invoke=MagicMock(return_value=LOW_RATED))
    monkeypatch.setattr(agent, "filter_reviews_by_rating", low)
    _fake_llm(monkeypatch, "- Delivery problems: box arrived flipped\n- Rude staff: ignored at the counter")

    _history, answer, trace = agent.run_turn(fake_agent, [], question)

    low.invoke.assert_called_once_with({"min_rating": 1, "max_rating": 2})
    assert "guardrail: model's tool call did not fit" in trace[-2]
    assert answer.startswith("**3 reviews are rated 1-2 stars** (1 stars: 1, 2 stars: 2).")
    assert "Rude staff: ignored at the counter" in answer
    assert '"Pizza arrived upside-down" (1/5)' in answer
    assert "praise" not in answer


def test_complaints_drop_model_refusal(monkeypatch):
    question = "What are the worst complaints?"
    fake_agent = _agent_returning(
        question,
        *_model_tool_turn("filter_reviews_by_rating", {"min_rating": 1, "max_rating": 2}, LOW_RATED),
    )
    _fake_llm(monkeypatch, "I'm sorry, but I can't help with that.")

    _history, answer, _trace = agent.run_turn(fake_agent, [], question)

    assert "sorry" not in answer.lower()
    assert "Examples:" in answer


def test_stats_answer_is_deterministic(monkeypatch):
    question = "Summarize the rating distribution"
    stats = "1 stars: 6\n2 stars: 34\n3 stars: 20\n4 stars: 7\n5 stars: 56\nAverage rating: 3.59\nTotal reviews: 123"
    fake_agent = _agent_returning(
        question,
        *_model_tool_turn("get_rating_stats", {"min_rating": 1, "max_rating": 5}, stats),
    )
    _no_llm(monkeypatch)

    _history, answer, _trace = agent.run_turn(fake_agent, [], question)

    assert answer.startswith("**Rating distribution across 123 reviews** (average 3.59/5):")
    assert "- 5 stars: 56 (46%)" in answer
    assert "- 2 stars: 34 (28%)" in answer


def test_owner_reply_frame_is_fixed_and_model_frame_removed(monkeypatch):
    question = "Draft an owner reply about late delivery"
    fake_agent = _agent_returning(question, FakeMessage("Sure! Here's a letter about a reservation."))
    monkeypatch.setattr(
        agent,
        "_fallback_tool_for_question",
        lambda q: ("search_reviews", {"query": "late delivery"}, "Waited over an hour for delivery."),
    )
    _fake_llm(
        monkeypatch,
        "Dear Customer,\n1. We are sorry you waited over an hour for your delivery.\n"
        "2. We are adding a driver on weekend evenings.\nBest regards,\n[Restaurant Owner]",
    )

    _history, answer, _trace = agent.run_turn(fake_agent, [], question)

    assert answer.count("Dear Customer") == 1
    assert answer.count("Best regards") == 1
    assert "regarding late delivery. We are sorry you waited over an hour" in answer
    assert "adding a driver on weekend evenings" in answer
    assert "[" not in answer
    assert "reservation" not in answer


def test_new_topic_after_tool_turn_is_still_guarded(monkeypatch):
    prior = _prior_turn_with_tool_evidence()
    question = "Is the vegan pizza any good?"
    fake_agent = _agent_returning(question, FakeMessage("Invented vegan praise."), prior=prior)
    vegan_hits = "[1] Rating: 5/5 | Date: 2024-01-04\nVegan win\nCashew cheese on the vegan pizza melts well."
    fallback = MagicMock(return_value=("search_reviews", {"query": "vegan pizza"}, vegan_hits))
    monkeypatch.setattr(agent, "_fallback_tool_for_question", fallback)
    _fake_llm(monkeypatch, "The cashew cheese gets praise.")

    _history, answer, trace = agent.run_turn(fake_agent, prior, question)

    fallback.assert_called_once_with(question)
    assert answer.startswith("**Yes.** 1 retrieved review mentions the vegan pizza")
    assert "guardrail" in trace[0]
