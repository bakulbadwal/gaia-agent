"""Tests for gaia_agent.trace.

Two layers:
* hand-built memories made from smolagents' own step classes (no model at all), which
  pin the schema, truncation, redaction, tool-call recovery and totals exactly;
* a scripted end-to-end run: a real CodeAgent + managed agent driven by a fake Model
  that returns canned code. No network, no API. This proves the SubAgentRecorder
  captures *every* sub-agent invocation even though smolagents resets the managed
  agent's memory on each call.
"""

from __future__ import annotations

import json
import time

import pytest
from smolagents import CodeAgent, tool
from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep, TaskStep, ToolCall
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import Timing, TokenUsage
from smolagents.utils import AgentExecutionError, AgentMaxStepsError

from gaia_agent import trace as T


class _NoCallModel(Model):
    """A model that must never be called: proves the hand-built tests make no model call."""

    def __init__(self):
        super().__init__(model_id="test-model")

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kw):
        raise AssertionError("model must not be called in hand-built tests")


@tool
def web_search(query: str) -> str:
    """Search the web.

    Args:
        query: the search query
    """
    return f"results for {query}"


@tool
def visit_webpage(url: str) -> str:
    """Visit a page.

    Args:
        url: the url
    """
    return f"page {url}"


class _Log:
    """Minimal logger stand-in: AgentError.__init__ calls logger.log_error."""

    def log_error(self, *_a, **_k):
        pass


def _timing(start: float, dur: float) -> Timing:
    return Timing(start_time=start, end_time=start + dur)


def _action(n, start, dur, code, obs=None, err=None, tokens=(100, 20), final=False, out=None):
    return ActionStep(
        step_number=n,
        timing=_timing(start, dur),
        model_output=f"Thought: step {n}\n<code>\n{code}\n</code>",
        code_action=code,
        tool_calls=[ToolCall(name="python_interpreter", arguments=code, id=f"call_{n}")],
        observations=obs,
        error=err,
        token_usage=TokenUsage(*tokens) if tokens else None,
        is_final_answer=final,
        action_output=out,
    )


def _single_agent():
    agent = CodeAgent(tools=[web_search, visit_webpage], model=_NoCallModel(), max_steps=5)
    t0 = 1_760_000_000.0
    agent.memory.steps = [
        TaskStep(task="How many studio albums did X release between 2000 and 2009?"),
        PlanningStep(
            model_input_messages=[],
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="plan"),
            plan="1. search wikipedia\n2. count albums",
            timing=_timing(t0, 1.5),
            token_usage=TokenUsage(50, 30),
        ),
        _action(
            1,
            t0 + 2,
            3.0,
            'r = web_search(query="X discography")\nprint(r)\npage = visit_webpage(url=r.split()[0])',
            obs="Execution logs:\n" + "a" * 5000,
        ),
        _action(
            2,
            t0 + 6,
            1.0,
            "n = len(albums)",
            err=AgentExecutionError("NameError: name 'albums' is not defined", _Log()),
        ),
        _action(3, t0 + 8, 0.5, 'final_answer("3")', obs="Last output: 3", final=True, out="3"),
    ]
    return agent


# ------------------------------------------------------------------ schema


def test_schema_top_level_and_step_fields():
    agent = _single_agent()
    tr = T.trace_from_agent(
        agent,
        task_id="t1",
        question="How many?",
        run_id="r1",
        final_answer="3",
        error=None,
        seconds=9.25,
        provider_info={"provider": "gemini", "model_id": "gemini-2.5-flash", "key_env": "GEMINI_API_KEY",
                       "key_present": True, "api_key": "AIza" + "SyD-should-never-be-written-1234567890"},
    )
    for key in ("task_id", "question", "run_id", "model", "provider", "arch", "started_at", "seconds",
                "final_answer", "error", "steps", "totals"):
        assert key in tr
    assert tr["model"] == "gemini-2.5-flash" and tr["provider"] == "gemini" and tr["arch"] == "single"
    assert tr["started_at"].startswith("2025-10-09T")  # earliest step start, ISO UTC
    assert "api_key" not in tr["provider_info"] and tr["provider_info"]["key_env"] == "GEMINI_API_KEY"
    assert [s["type"] for s in tr["steps"]] == ["task", "planning", "action", "action", "action", "final"]
    assert [s["index"] for s in tr["steps"]] == list(range(6))
    for s in tr["steps"]:
        for key in ("index", "type", "model_output", "code", "tool_calls", "observation", "error",
                    "input_tokens", "output_tokens", "duration_s", "agent"):
            assert key in s, (key, s["type"])
    plan = tr["steps"][1]
    assert plan["input_tokens"] == 50 and plan["output_tokens"] == 30 and plan["duration_s"] == 1.5
    assert tr["steps"][-1]["model_output"] == "3"
    json.dumps(tr)  # fully serialisable


def test_model_falls_back_to_agent_model_id():
    tr = T.trace_from_agent(_single_agent(), task_id="t", question="q", run_id="r",
                            final_answer="3", error=None, seconds=1.0)
    assert tr["model"] == "test-model" and tr["provider"] is None and tr["provider_info"] is None


def test_code_agent_tool_calls_recovered_from_code():
    tr = T.trace_from_agent(_single_agent(), task_id="t", question="q", run_id="r",
                            final_answer="3", error=None, seconds=1.0)
    s1 = tr["steps"][2]
    assert [c["name"] for c in s1["tool_calls"]] == ["web_search", "visit_webpage"]
    assert s1["tool_calls"][0]["arguments"] == {"query": "X discography"}
    assert s1["tool_calls"][1]["arguments"]["url"].startswith("<expr> r.split()")
    assert tr["steps"][3]["tool_calls"] == []  # `len` is not a tool
    assert tr["steps"][4]["tool_calls"] == [{"name": "final_answer", "arguments": {"args": ["3"]}}]
    # python_interpreter is a smolagents wrapper, never a counted tool
    assert "python_interpreter" not in tr["totals"]["tool_calls_by_name"]


def test_tools_called_in_code_edge_cases():
    names = ["web_search", "final_answer"]
    assert T.tools_called_in_code(None, names) == []
    assert T.tools_called_in_code("x = 1", []) == []
    # attribute calls and lookalike names are not tool calls
    assert T.tools_called_in_code("obj.web_search('a')\nmy_web_search('b')", names) == []
    # nested calls in source order
    got = T.tools_called_in_code("final_answer(web_search('a'))", names)
    assert [c["name"] for c in got] == ["final_answer", "web_search"]
    # unparseable code: regex fallback still records intent
    got = T.tools_called_in_code("web_search('a'\nfinal_answer(", names)
    assert [c["name"] for c in got] == ["web_search", "final_answer"]
    assert all(c["arguments"] == {"unparsed": True} for c in got)


def test_tool_calling_agent_style_calls_used_verbatim():
    agent = CodeAgent(tools=[web_search], model=_NoCallModel())
    step = ActionStep(step_number=1, timing=_timing(1.0, 1.0),
                      tool_calls=[ToolCall(name="web_search", arguments={"query": "q"}, id="c1")],
                      token_usage=TokenUsage(1, 1))
    agent.memory.steps = [TaskStep(task="q"), step]
    tr = T.trace_from_agent(agent, task_id="t", question="q", run_id="r", final_answer=None,
                            error=None, seconds=1.0)
    assert tr["steps"][1]["tool_calls"] == [{"name": "web_search", "arguments": {"query": "q"}}]


# -------------------------------------------------------------- truncation


def test_truncation_marker_and_limit():
    tr = T.trace_from_agent(_single_agent(), task_id="t", question="q", run_id="r",
                            final_answer="3", error=None, seconds=1.0)
    obs = tr["steps"][2]["observation"]
    original_len = len("Execution logs:\n") + 5000
    assert obs.endswith(f"…[truncated {original_len - T.MAX_TEXT} chars]")
    assert len(obs) == T.MAX_TEXT + len(f"…[truncated {original_len - T.MAX_TEXT} chars]")
    assert T.clip("x" * 4000) == "x" * 4000  # exactly at the limit: untouched
    assert T.clip("x" * 4001).endswith("…[truncated 1 chars]")
    assert T.clip(None) is None
    assert T.clip({"a": [1, 2]}) == '{"a": [1, 2]}'


def test_secrets_are_redacted_in_text():
    # Fake, key-shaped strings, assembled at runtime so secret scanners do not flag the test file.
    text = "key=" + "AIza" + "SyA1234567890abcdefghijklmnopqrstuv and " + "gsk_" + "abcdefghijklmnopqrstuvwx hf_" + "a" * 34
    out = T.clip(text)
    assert "AIza" not in out and "gsk_" not in out and "hf_a" not in out
    assert out.count("[REDACTED]") == 3


# ----------------------------------------------------------------- errors


def test_step_errors_and_totals():
    tr = T.trace_from_agent(_single_agent(), task_id="t", question="q", run_id="r",
                            final_answer="3", error=None, seconds=1.0)
    err_step = tr["steps"][3]
    assert err_step["error_type"] == "AgentExecutionError"
    assert "NameError" in err_step["error"]
    tot = tr["totals"]
    assert tot["steps"] == 3 and tot["planning_steps"] == 1
    assert tot["model_calls"] == 4  # 1 planning + 3 actions
    assert tot["input_tokens"] == 50 + 3 * 100 and tot["output_tokens"] == 30 + 3 * 20
    assert tot["total_tokens"] == tot["input_tokens"] + tot["output_tokens"]
    assert tot["tool_calls_by_name"] == {"web_search": 1, "visit_webpage": 1, "final_answer": 1}
    assert tot["tool_calls"] == 3
    assert tot["step_errors"] == 1 and tot["step_error_types"] == {"AgentExecutionError": 1}
    assert tot["token_usage_complete"] is True
    assert tot["by_agent"]["agent"]["steps"] == 3


def test_run_level_error_and_missing_token_usage():
    agent = CodeAgent(tools=[], model=_NoCallModel())
    agent.memory.steps = [
        TaskStep(task="q"),
        ActionStep(step_number=1, timing=_timing(10.0, 2.0), model_output="...", token_usage=None,
                   error=AgentMaxStepsError("Reached max steps.", _Log())),
    ]
    tr = T.trace_from_agent(agent, task_id="t", question="q", run_id="r", final_answer=None,
                            error=TimeoutError("question timed out after 300s"), seconds=300.0)
    assert tr["error"] == "question timed out after 300s" and tr["error_type"] == "TimeoutError"
    assert tr["steps"][-1]["type"] == "final" and tr["steps"][-1]["error_type"] == "TimeoutError"
    assert tr["totals"]["token_usage_complete"] is False
    assert tr["totals"]["step_error_types"] == {"AgentMaxStepsError": 1, "TimeoutError": 1}


def test_trace_without_agent_still_records_question():
    tr = T.trace_from_agent(None, task_id="t", question="what?", run_id="r", final_answer=None,
                            error="model unreachable", seconds=0.2)
    assert [s["type"] for s in tr["steps"]] == ["task", "final"]
    assert tr["steps"][0]["model_output"] == "what?" and tr["error"] == "model unreachable"
    assert tr["totals"]["steps"] == 0 and tr["totals"]["model_calls"] == 0


# --------------------------------------------------------------- sub-agents


def _multi_agent_handbuilt():
    web = CodeAgent(tools=[web_search], model=_NoCallModel(), name="web_agent", description="searches the web")
    manager = CodeAgent(tools=[], model=_NoCallModel(), managed_agents=[web])
    t0 = 2_000.0
    manager.memory.steps = [
        TaskStep(task="Q"),
        _action(1, t0, 10.0, 'r = web_agent(task="find X")\nprint(r)', obs="report", tokens=(300, 40)),
        _action(2, t0 + 11, 1.0, "final_answer(r)", final=True, out="X", tokens=(320, 10)),
    ]
    # managed agent memory: only its LAST invocation survives in smolagents
    web.memory.steps = [
        TaskStep(task="find X"),
        _action(1, t0 + 1, 2.0, 'web_search(query="X")', obs="hits", tokens=(80, 15)),
        _action(2, t0 + 4, 1.0, 'final_answer("X")', final=True, out="X", tokens=(90, 5)),
    ]
    return manager


def test_sub_agent_fallback_nests_last_invocation_as_partial():
    tr = T.trace_from_agent(_multi_agent_handbuilt(), task_id="t", question="Q", run_id="r",
                            final_answer="X", error=None, seconds=12.0, arch="multi")
    s1 = tr["steps"][1]
    assert [c["name"] for c in s1["tool_calls"]] == ["web_agent"]
    assert s1["tool_calls"][0]["arguments"] == {"task": "find X"}
    assert len(s1["sub_agents"]) == 1 and tr["steps"][2]["sub_agents"] == []
    run = s1["sub_agents"][0]
    assert run["agent"] == "web_agent" and run["partial"] is True and run["task"] == "find X"
    assert [s["agent"] for s in run["steps"]] == ["web_agent", "web_agent"]
    tot = tr["totals"]
    assert tot["steps"] == 2  # top-level only
    assert tot["model_calls"] == 4  # 2 manager + 2 sub-agent
    assert tot["input_tokens"] == 300 + 320 + 80 + 90
    assert tot["sub_agent_calls"] == 1
    assert tot["tool_calls_by_name"] == {"web_agent": 1, "final_answer": 2, "web_search": 1}
    assert tot["by_agent"]["manager"]["steps"] == 2 and tot["by_agent"]["web_agent"]["steps"] == 2
    assert all(not k.startswith("_") for run in s1["sub_agents"] for k in run)


class _ScriptedModel(Model):
    """Returns canned replies in order. Deterministic and offline: not a model call."""

    def __init__(self, replies, model_id="scripted"):
        super().__init__(model_id=model_id)
        self.replies = list(replies)
        self.calls = 0

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kw):
        self.calls += 1
        time.sleep(0.002)  # keep step windows strictly ordered on coarse clocks
        return ChatMessage(role=MessageRole.ASSISTANT, content=self.replies.pop(0),
                           token_usage=TokenUsage(input_tokens=100, output_tokens=10))


def _scripted_multi():
    web = CodeAgent(
        tools=[web_search],
        model=_ScriptedModel([
            'Thought: search\n<code>\nr = web_search(query="alpha")\nfinal_answer("A")\n</code>',
            'Thought: search\n<code>\nr = web_search(query="beta")\n</code>',
            'Thought: done\n<code>\nfinal_answer("B")\n</code>',
        ]),
        name="web_agent",
        description="Searches the web and reports back.",
        max_steps=4,
        verbosity_level=0,
    )
    manager = CodeAgent(
        tools=[],
        model=_ScriptedModel([
            'Thought: delegate twice\n<code>\na = web_agent(task="find alpha")\nb = web_agent(task="find beta")\nprint(len(a), len(b))\n</code>',
            'Thought: answer\n<code>\nfinal_answer("AB")\n</code>',
        ], model_id="scripted-manager"),
        managed_agents=[web],
        max_steps=4,
        verbosity_level=0,
    )
    return manager, web


def test_recorder_captures_every_sub_agent_invocation():
    manager, web = _scripted_multi()
    recorder = T.SubAgentRecorder.attach(manager)
    assert recorder.attached_agents == ["web_agent"]
    t0 = time.time()
    out = manager.run("Combine alpha and beta.")
    secs = time.time() - t0
    assert out == "AB"
    # smolagents itself kept only the second invocation:
    assert any(isinstance(s, TaskStep) and "find beta" in s.task for s in web.memory.steps)
    assert not any(isinstance(s, TaskStep) and "find alpha" in s.task for s in web.memory.steps)

    tr = T.trace_from_agent(manager, task_id="t", question="Combine", run_id="r", final_answer=out,
                            error=None, seconds=secs, arch="multi", recorder=recorder)
    action1 = tr["steps"][1]
    assert action1["type"] == "action"
    assert [c["name"] for c in action1["tool_calls"]] == ["web_agent", "web_agent"]
    runs = action1["sub_agents"]
    assert len(runs) == 2, "both invocations must be captured"
    assert "find alpha" in runs[0]["task"] and "find beta" in runs[1]["task"]
    assert runs[0]["partial"] is False and runs[0]["final_answer"] == "A" and runs[1]["final_answer"] == "B"
    assert [len(r["steps"]) for r in runs] == [1, 2]
    assert runs[0]["steps"][0]["tool_calls"][0] == {"name": "web_search", "arguments": {"query": "alpha"}}
    assert tr["sub_agent_runs_unattached"] == []
    tot = tr["totals"]
    assert tot["steps"] == 2 and tot["model_calls"] == 2 + 3
    assert tot["input_tokens"] == 500 and tot["output_tokens"] == 50
    assert tot["sub_agent_calls"] == 2
    assert tot["tool_calls_by_name"]["web_search"] == 2 and tot["tool_calls_by_name"]["web_agent"] == 2
    assert tot["by_agent"]["web_agent"] == {"steps": 3, "model_calls": 3, "input_tokens": 300,
                                            "output_tokens": 30, "tool_calls": 4}
    assert tr["model"] == "scripted-manager"

    # reset() clears history but the recorder keeps recording the next question;
    # a second, independent recorder on the same agent does not disturb it
    recorder.reset()
    assert recorder.invocations == []
    T.SubAgentRecorder.attach(manager)
    manager.model.replies = ['<code>\nx = web_agent(task="again")\nfinal_answer(x)\n</code>']
    web.model.replies = ['<code>\nfinal_answer("C")\n</code>']
    manager.run("again")
    assert len(recorder.invocations) == 1


def test_single_agent_recorder_is_a_noop():
    agent = _single_agent()
    rec = T.SubAgentRecorder.attach(agent)
    assert rec.attached_agents == [] and rec.invocations == []
    tr = T.trace_from_agent(agent, task_id="t", question="q", run_id="r", final_answer="3",
                            error=None, seconds=1.0, recorder=rec)
    assert all(s.get("sub_agents", []) == [] for s in tr["steps"])


# ---------------------------------------------------------------- io + summary


def test_write_and_load_roundtrip(tmp_path):
    tr = T.trace_from_agent(_single_agent(), task_id="abc-123", question="q", run_id="run 1/../x",
                            final_answer="3", error=None, seconds=4.0)
    path = T.write_trace(tr, root=tmp_path)
    assert path.parent.parent == tmp_path / "traces"  # traversal neutralised
    assert path.name == "abc-123.json"
    assert not list(path.parent.glob(".tmp-*"))
    (path.parent / "broken.json").write_text("{not json")
    loaded = T.load_traces("run 1/../x", root=tmp_path)
    assert len(loaded) == 1 and loaded[0] == json.loads(json.dumps(tr))
    assert T.load_traces("nope", root=tmp_path) == []
    with pytest.raises(ValueError):
        T.trace_path("..", "t", tmp_path)


def test_write_trace_overwrites_atomically(tmp_path):
    tr = T.trace_from_agent(None, task_id="t", question="q", run_id="r", final_answer="1",
                            error=None, seconds=1.0)
    T.write_trace(tr, root=tmp_path)
    tr["final_answer"] = "2"
    T.write_trace(tr, root=tmp_path)
    assert T.load_traces("r", root=tmp_path)[0]["final_answer"] == "2"


def test_summarize_traces_percentiles_and_totals():
    base = T.trace_from_agent(_single_agent(), task_id="t", question="q", run_id="r",
                              final_answer="3", error=None, seconds=1.0)
    traces = []
    for i, secs in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        t = json.loads(json.dumps(base))
        t["task_id"], t["seconds"] = f"t{i}", float(secs)
        traces.append(t)
    traces[0]["error"], traces[0]["error_type"] = "boom", "TimeoutError"
    s = T.summarize_traces(traces)
    assert s["n"] == 10 and s["errors"] == 1 and s["error_types"] == {"TimeoutError": 1}
    assert s["p50_seconds"] == 5.5 and s["p90_seconds"] == 9.1  # numpy 'linear' convention
    assert s["mean_seconds"] == 5.5 and s["max_seconds"] == 10.0 and s["total_seconds"] == 55.0
    assert s["mean_steps"] == 3.0 and s["total_steps"] == 30
    assert s["input_tokens"] == 10 * 350 and s["total_tokens"] == 10 * (350 + 90)
    assert s["model_calls"] == 40
    assert s["tool_calls_by_name"] == {"web_search": 10, "visit_webpage": 10, "final_answer": 10}
    assert s["step_errors"] == 10 and s["by_agent"]["agent"]["steps"] == 30


def test_summarize_empty_and_single():
    s = T.summarize_traces([])
    assert s["n"] == 0 and s["p50_seconds"] is None and s["mean_steps"] is None
    one = T.trace_from_agent(None, task_id="t", question="q", run_id="r", final_answer="1",
                             error=None, seconds=2.0)
    s = T.summarize_traces([one])
    assert s["p50_seconds"] == 2.0 and s["p90_seconds"] == 2.0


def test_format_trace_and_cli(tmp_path, capsys):
    tr = T.trace_from_agent(_multi_agent_handbuilt(), task_id="task-xyz", question="Q", run_id="r",
                            final_answer="X", error=None, seconds=12.0, arch="multi")
    text = T.format_trace(tr)
    assert "↳ web_agent (partial)" in text and "tools: web_agent" in text and "totals:" in text
    T.write_trace(tr, root=tmp_path)
    assert T._main(["r", "--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 1
    assert T._main(["r", "--root", str(tmp_path), "--task", "task-x"]) == 0
    assert "task-xyz" in capsys.readouterr().out
    assert T._main(["missing", "--root", str(tmp_path)]) == 1
