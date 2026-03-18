from src import rag_processor


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def invoke(self, _prompt):
        idx = min(self.calls, len(self.outputs) - 1)
        self.calls += 1
        return _FakeResponse(self.outputs[idx])


class _FakeRetriever:
    def invoke(self, _query):
        class _Doc:
            page_content = "Local context about documents."
        return [_Doc()]


def test_parse_react_step_final_answer():
    parsed = rag_processor._parse_react_step("Final Answer: hello")
    assert parsed["final"] == "hello"
    assert parsed["action"] == ""


def test_run_react_agent_uses_calculator_then_final(monkeypatch):
    monkeypatch.setattr(rag_processor, "_math_tool_context", lambda q: "Expression: 2+2\nResult: 4")

    llm = _FakeLLM([
        "Action: calculator\nAction Input: 2+2",
        "Final Answer: The result is 4.",
    ])
    retriever = _FakeRetriever()

    answer = rag_processor._run_react_agent(
        question="What is 2+2?",
        llm=llm,
        retriever=retriever,
        use_web_search=False,
        use_math_tool=True,
    )

    assert "4" in answer


def test_is_current_events_query_detects_sports_updates():
    assert rag_processor._is_current_events_query(
        "What is happening in round of 16 in the UEFA Champions League this year?"
    )


def test_run_react_agent_falls_back_to_web_when_no_action(monkeypatch):
    monkeypatch.setattr(
        rag_processor,
        "_serper_web_search",
        lambda q: "[1] UEFA update\nRound of 16 fixtures announced.",
    )

    llm = _FakeLLM([
        "I should check latest updates first.",
        "Final Answer: The Champions League round of 16 fixtures were announced.",
    ])
    retriever = _FakeRetriever()

    answer = rag_processor._run_react_agent(
        question="What is happening in round of 16 in the UEFA Champions League this year?",
        llm=llm,
        retriever=retriever,
        use_web_search=True,
        use_math_tool=True,
    )

    assert "round of 16" in answer.lower()


def test_run_react_agent_ignores_premature_weak_final_and_uses_web(monkeypatch):
    monkeypatch.setattr(
        rag_processor,
        "_serper_web_search",
        lambda q: "[1] UEFA latest\nArsenal and Real Madrid are among in-form sides.",
    )

    llm = _FakeLLM([
        "Final Answer: I'm unable to retrieve the latest information at the moment.",
        "Final Answer: Based on current reports, Arsenal and Real Madrid are among the strongest teams.",
    ])
    retriever = _FakeRetriever()

    answer = rag_processor._run_react_agent(
        question="What about the football uefa champions league round of 16 this year? Which teams are doing the best?",
        llm=llm,
        retriever=retriever,
        use_web_search=True,
        use_math_tool=True,
    )

    assert "arsenal" in answer.lower() or "real madrid" in answer.lower()


def test_run_react_agent_does_not_accept_weak_final_after_tool(monkeypatch):
    monkeypatch.setattr(
        rag_processor,
        "_serper_web_search",
        lambda q: "[1] Ranking\nArsenal, Bayern, and Real Madrid are top performers.",
    )

    llm = _FakeLLM([
        "Action: web_search\nAction Input: UEFA Champions League round of 16 top teams",
        "Final Answer: I don't know.",
        "Final Answer: Arsenal, Bayern, and Real Madrid are among the strongest performers this round.",
    ])
    retriever = _FakeRetriever()

    answer = rag_processor._run_react_agent(
        question="Which teams are performing the best in uefa football champions league round of 16 this year?",
        llm=llm,
        retriever=retriever,
        use_web_search=True,
        use_math_tool=True,
    )

    assert "arsenal" in answer.lower()


def test_is_math_query_not_triggered_by_round_of_16_phrase():
    assert not rag_processor._is_math_query(
        "Which teams are performing the best in uefa football champions league round of 16 this year?"
    )


def test_heuristic_action_prefers_web_for_round_of_16_news():
    action, action_input = rag_processor._heuristic_action(
        "What is happening in uefa champions league round of 16 this year?",
        use_web_search=True,
        use_math_tool=True,
    )
    assert action == "web_search"
    assert "round of 16" in action_input.lower()
