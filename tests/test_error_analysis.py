"""Tests for analysis/error_analysis.py on hand-built trace dicts (no gold answers anywhere)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import error_analysis as EA  # noqa: E402


def step(index, code=None, tools=(), observation=None, error=None, error_type=None, final=False, type_="action", **kw):
    d = {
        "index": index,
        "type": type_,
        "agent": "agent",
        "model_output": None,
        "code": code,
        "tool_calls": [{"name": n, "arguments": a} for n, a in tools],
        "observation": observation,
        "error": error,
        "error_type": error_type,
        "input_tokens": 100,
        "output_tokens": 20,
        "duration_s": 1.0,
        "is_final_answer": final,
    }
    d.update(kw)
    return d


def trace(steps, final_answer="42", error=None, error_type=None, task_id="t1"):
    return {
        "task_id": task_id,
        "question": "q",
        "run_id": "r1",
        "final_answer": final_answer,
        "error": error,
        "error_type": error_type,
        "seconds": 12.0,
        "steps": [step(0, type_="task")] + list(steps) + [step(len(steps) + 1, type_="final")],
        "totals": {"tool_calls_by_name": {}},
    }


def tags_of(tr, **kw):
    return EA.analyze_trace(tr, **kw)["tags"]


# --------------------------------------------------------------------------- clean baseline


def test_clean_trace_has_no_failure_tags():
    tr = trace([step(1, code='print(web_search("x"))', tools=[("web_search", {"query": "x"})], observation="results")])
    assert tags_of(tr) == []


# --------------------------------------------------------------------------- each tag


def test_hit_max_steps_from_error_type_and_from_step_count():
    tr = trace([step(1, error="Reached max steps.", error_type="AgentMaxStepsError")], final_answer="7")
    assert "hit_max_steps" in tags_of(tr)
    many = trace([step(i, code="x = 1") for i in range(1, 13)], final_answer="7")
    assert "hit_max_steps" in tags_of(many, max_steps=12)
    assert "hit_max_steps" not in tags_of(many, max_steps=20)


def test_tool_error_from_returned_error_string_is_attributed():
    tr = trace(
        [
            step(
                1,
                code='print(visit_webpage("https://a.test"))',
                tools=[("visit_webpage", {"url": "https://a.test"})],
                observation="ERROR: HTTP 403 fetching https://a.test. The page may block bots",
            )
        ]
    )
    item = EA.analyze_trace(tr)
    assert "tool_error" in item["tags"] and "tool_error:visit_webpage" in item["tags"]
    assert item["details"]["tool_error"] == ["visit_webpage"]
    assert "403" in item["evidence"]


def test_tool_error_from_smolagents_tool_exception():
    err = "Error executing tool 'transcribe_audio' with arguments {'path': 'x.mp3'}: RuntimeError: ffmpeg missing"
    tr = trace([step(1, error=err, error_type="AgentToolExecutionError")])
    assert "tool_error:transcribe_audio" in tags_of(tr)


def test_exception_raised_inside_tool_call_is_tool_error_not_code_error():
    err = "Code execution failed at line 'r = read_file(\"data.xlsx\")' due to: FileNotFoundError: no such file"
    tr = trace([step(1, code='r = read_file("data.xlsx")', error=err, error_type="AgentExecutionError")])
    tags = tags_of(tr)
    assert "tool_error:read_file" in tags and "code_execution_error" not in tags


def test_code_execution_error():
    err = "Code execution failed at line 'x = 1/0' due to: ZeroDivisionError: division by zero"
    tr = trace([step(1, code="x = 1/0", error=err, error_type="AgentExecutionError")])
    assert "code_execution_error" in tags_of(tr)


def test_parse_error():
    err = "Error in code parsing:\nYour code snippet is invalid, because the regex pattern <code>(.*?)</code> was not found in it."
    tr = trace([step(1, error=err, error_type="AgentParsingError")])
    assert tags_of(tr) == ["parse_error"]


def test_rate_limited_from_model_generation_error():
    err = "Error in generating model output:\nError code: 429 - {'error': {'message': 'Rate limit exceeded'}}"
    tr = trace([step(1, error=err, error_type="AgentGenerationError")], final_answer=None, error="AllModelsFailed: rate_limit 429")
    tags = tags_of(tr)
    assert "rate_limited" in tags and "empty_or_refusal_answer" in tags


def test_rate_limited_from_budget_exceeded_run_error():
    tr = trace([], final_answer=None, error="Daily request cap reached for 'openrouter': 50/50", error_type="BudgetExceeded")
    assert "rate_limited" in tags_of(tr)


def test_429_inside_ids_or_page_text_is_not_rate_limited():
    # A UUID containing 429 in the run error, and a web page that talks about rate limits.
    tr = trace(
        [step(1, code='print(web_search("api"))', tools=[("web_search", {"query": "api"})],
              observation="The API returns HTTP 429 when the rate limit is exceeded.")],
        error="task 8e867cd7-cff9-4429-a4ab-b5aa04d2c429 failed: ValueError",
    )
    assert "rate_limited" not in tags_of(tr)


def test_tool_side_429_is_a_tool_error_not_model_rate_limit():
    tr = trace([step(1, code='print(web_search("x"))', tools=[("web_search", {"query": "x"})],
                     observation="ERROR: search failed (HTTP 429 from DuckDuckGo); retry later")])
    tags = tags_of(tr)
    assert "tool_error:web_search" in tags and "rate_limited" not in tags


def test_timeout_from_runner_error_and_answers_row():
    tr = trace([step(1, code="x=1")], final_answer=None, error="Timeout after 420s")
    assert "timeout" in tags_of(tr)


def test_empty_and_refusal_answers():
    assert "empty_or_refusal_answer" in tags_of(trace([], final_answer=""))
    assert "empty_or_refusal_answer" in tags_of(trace([], final_answer=None))
    for refusal in ["I cannot determine this", "Unknown", "N/A", "unable to determine the answer", "I'm sorry, but"]:
        assert "empty_or_refusal_answer" in tags_of(trace([], final_answer=refusal)), refusal
    for fine in ["Unknown Pleasures", "Nonesuch"]:  # real answers that merely start like a refusal word
        assert "empty_or_refusal_answer" not in tags_of(trace([], final_answer=fine)), fine


@pytest.mark.parametrize(
    "answer, reason",
    [
        ("FINAL ANSWER: 42", "FINAL ANSWER"),
        ("42 km", "units"),
        ("$1,200", "units"),
        ("45%", "units"),
        ("1,234,567", "thousands separator"),
        ("The answer is that the city was Paris in 1920.", "sentence"),
        ('"Paris"', "quotes"),
        ("x " * 40, "very long"),
        ("line one\nline two", "multi-line"),
    ],
)
def test_answer_format_suspicious(answer, reason):
    item = EA.analyze_trace(trace([], final_answer=answer))
    assert "answer_format_suspicious" in item["tags"], answer
    assert reason.lower() in " ".join(item["details"]["answer_format_suspicious"]).lower()


@pytest.mark.parametrize("answer", ["42", "Paris", "b, e, f", "Rd5", "3.14", "St. Petersburg", "broccoli, celery, lettuce, sweet potatoes", "1984"])
def test_well_formed_answers_not_flagged(answer):
    assert "answer_format_suspicious" not in tags_of(trace([], final_answer=answer))


def test_blocked_by_guard():
    obs = "ERROR: query refused. BLOCKED by the answer-leak guard: this agent must solve each question itself"
    tr = trace([step(1, code='web_search("gaia benchmark answer")', tools=[("web_search", {"query": "gaia benchmark answer"})], observation=obs)])
    tags = tags_of(tr)
    assert "blocked_by_guard" in tags
    assert not any(t.startswith("tool_error") for t in tags)  # a guard refusal is not a tool failure


def test_no_tool_used():
    tr = trace([step(1, code="print(sorted(['b','a']))", observation="['a', 'b']")])
    assert "no_tool_used" in tags_of(tr)


def test_tools_detected_from_code_when_tool_calls_are_python_interpreter():
    tr = trace([step(1, code='print(wikipedia_page("Mercedes Sosa"))', tools=[("python_interpreter", 'print(wikipedia_page("Mercedes Sosa"))')])])
    assert "no_tool_used" not in tags_of(tr)


def test_repeated_identical_tool_call_three_times():
    calls = [step(i, code=f'print(web_search("same"))  # {i}', tools=[("web_search", {"query": "same"})]) for i in range(1, 4)]
    item = EA.analyze_trace(trace(calls))
    assert "repeated_identical_tool_call" in item["tags"]
    assert "x3" in item["evidence"]


def test_identical_code_in_consecutive_steps_is_a_loop():
    code = 'print(visit_webpage("https://a.test"))'
    tr = trace([step(1, code=code, tools=[("visit_webpage", {"url": "https://a.test"})]),
                step(2, code=code, tools=[("visit_webpage", {"url": "https://a.test"})])])
    assert "repeated_identical_tool_call" in tags_of(tr)


def test_different_arguments_are_not_a_loop_but_paging_is_tagged():
    calls = [
        step(i, code=f'print(visit_webpage("https://a.test", start={i * 15000}))', tools=[("python_interpreter", "")])
        for i in range(1, 6)
    ]
    tags = tags_of(trace(calls))
    assert "repeated_identical_tool_call" not in tags
    assert "paging_loop" in tags


def test_vision_used_uses_token_match_not_substring():
    tr = trace([step(1, code='image_question("a.png", "what?")', tools=[("image_question", {"path": "a.png"})])])
    assert "vision_used" in tags_of(tr)
    # "wikipedia_revision" contains the substring "vision": it must NOT count.
    tr2 = trace([step(1, code='wikipedia_revision("X", "2022-01-01")', tools=[("wikipedia_revision", {"title": "X"})])])
    assert "vision_used" not in tags_of(tr2)


def test_sub_agent_steps_are_analysed():
    sub = {"agent": "web_agent", "steps": [step(0, code='web_search("x")', tools=[("web_search", {"query": "x"})],
                                                observation="ERROR: search backend unavailable")]}
    tr = trace([step(1, code='web_agent(task="find x")', tools=[("web_agent", {"task": "find x"})], sub_agents=[sub])])
    assert "tool_error:web_search" in tags_of(tr)


def test_unattributable_tool_error_with_multiple_tools():
    tr = trace([step(1, code='a = web_search("x")\nb = wikipedia_search("y")\nprint(a, b)',
                     tools=[("web_search", {"query": "x"}), ("wikipedia_search", {"query": "y"})],
                     observation="ERROR: connection reset")])
    assert "tool_error:unattributed" in tags_of(tr)


# --------------------------------------------------------------------------- run level + file output


def _write_run(root: Path, run_id: str, traces: list[dict], answers: list[dict]):
    d = root / "traces" / run_id
    d.mkdir(parents=True)
    for t in traces:
        t["run_id"] = run_id
        (d / f"{t['task_id']}.json").write_text(json.dumps(t))
    (root / "answers.jsonl").write_text("\n".join(json.dumps(a) for a in answers) + "\n")


def test_analyze_run_writes_contract_json_and_counts(tmp_path, capsys):
    ok = trace([step(1, code='web_search("x")', tools=[("web_search", {"query": "x"})])], task_id="a")
    bad = trace([step(1, error="Error in code parsing: nope", error_type="AgentParsingError")], final_answer="", task_id="b")
    answers = [
        {"task_id": "a", "answer": "42", "run_id": "r1"},
        {"task_id": "b", "answer": "", "run_id": "r1"},
        {"task_id": "c", "answer": "", "error": "Timeout after 420s", "steps": None, "run_id": "r1"},  # no trace
        {"task_id": "z", "answer": "x", "run_id": "other-run"},  # ignored
    ]
    _write_run(tmp_path, "r1", [ok, bad], answers)
    assert EA.main(["--run-id", "r1", "--runs-dir", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "error_analysis.json").read_text())
    assert out["run_id"] == "r1" and out["n_tasks"] == 3
    assert {"task_id", "tags", "evidence"} <= set(out["items"][0])
    by_id = {it["task_id"]: it["tags"] for it in out["items"]}
    assert by_id["a"] == []
    assert "parse_error" in by_id["b"] and "empty_or_refusal_answer" in by_id["b"]
    assert by_id["c"][:2] == ["no_trace", "timeout"]
    assert out["counts"]["empty_or_refusal_answer"] == 2 and out["counts"]["parse_error"] == 1
    assert out["clean_tasks"] == ["a"]
    printed = capsys.readouterr().out
    assert "parse_error" in printed and "wrote" in printed


def test_latest_run_id_and_missing_run(tmp_path):
    assert EA.latest_run_id(tmp_path) is None
    _write_run(tmp_path, "r9", [trace([], task_id="a")], [])
    assert EA.latest_run_id(tmp_path) == "r9"
    assert EA.main(["--run-id", "nope", "--runs-dir", str(tmp_path)]) == 2


def test_corrupt_trace_is_skipped(tmp_path):
    _write_run(tmp_path, "r1", [trace([], task_id="a")], [])
    (tmp_path / "traces" / "r1" / "broken.json").write_text("{not json")
    report = EA.analyze_run("r1", tmp_path)
    assert [it["task_id"] for it in report["items"]] == ["a"]


def test_format_table_shortens_uuids():
    report = {
        "run_id": "r",
        "n_tasks": 1,
        "items": [{"task_id": "8e867cd7-cff9-4e6c-867a-ff5ddc2550be", "tags": ["timeout"], "steps": 3}],
        "counts": {"timeout": 1},
        "clean_tasks": [],
    }
    table = EA.format_table(report)
    assert "8e867cd7 " in table and "cff9" not in table
