"""Tests for analysis/reliability.py: the case-study numbers must match the committed run files.

Everything here reads committed data (evals/results/, docs/traces/) or synthetic dicts; no network.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import reliability as rel  # noqa: E402

PASSK = json.loads((ROOT / "evals" / "results" / "passk-practice-baseline.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def repeat():
    return [rel.load_run(r) for r in rel.REPEAT_RUNS]


@pytest.fixture(scope="module")
def head(repeat):
    return rel.headline(repeat)


# ----------------------------------------------------------------------------- headline numbers


def test_repeat_runs_are_the_pooled_runs_and_share_one_config(repeat, head):
    assert list(rel.REPEAT_RUNS) == PASSK["runs"]
    assert head["configs_identical"]
    assert head["tasks"] == 16 and head["n_runs"] == 4


def test_recomputed_numbers_match_the_committed_passk_report(head):
    last = head["table"][-1]
    assert last["k"] == 4
    assert round(last["pass@k"], 4) == PASSK["table"][0]["pass@k"] == 1.0
    assert round(last["pass^k"], 4) == PASSK["table"][0]["pass^k"] == 0.9375
    assert round(head["mean"], 4) == PASSK["run_accuracy"]["mean"]
    assert round(head["stdev"], 4) == PASSK["run_accuracy"]["stdev"]
    assert {t: {"n": n, "c": c} for t, (n, c) in head["per_task"].items()} == {
        t: {"n": v["n"], "c": v["c"]} for t, v in PASSK["per_task"].items()}
    assert list(head["flaky"]) == PASSK["flaky"] and head["never_solved"] == PASSK["never_solved"] == []


def test_pass_at_k_rises_and_pass_hat_k_falls_with_k(head):
    at = [r["pass@k"] for r in head["table"]]
    hat = [r["pass^k"] for r in head["table"]]
    assert at == sorted(at) and hat == sorted(hat, reverse=True)
    assert at[0] == pytest.approx(hat[0])  # pass@1 == pass^1 == mean per-task success


def test_readme_quotes_the_recomputed_numbers(head):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    line = next(ln for ln in readme.splitlines() if "pass^4" in ln and "mean accuracy" in ln)
    assert f"{head['mean']:.1%}" in line
    assert f"pass@4 {head['table'][-1]['pass@k']:.0%}" in line
    assert f"pass^4 {head['table'][-1]['pass^k']:.1%}" in line


# ----------------------------------------------------------------------------- failure causes


@pytest.mark.parametrize(
    "item, expected",
    [
        ({"correct": True, "steps": 3}, "pass"),
        ({"correct": False, "timed_out": True, "steps": 6}, "timeout"),
        ({"correct": False, "error": "AgentError: Agent interrupted.", "steps": 1}, "timeout"),
        ({"correct": False, "steps": 13, "answer": "<code> x </code>"}, "max_steps"),
        ({"correct": False, "steps": 4, "answer": "Qd8#"}, "wrong"),
    ],
)
def test_outcome(item, expected):
    assert rel.outcome(item, max_steps=12) == expected


@pytest.mark.parametrize(
    "out, sig, skills, expected",
    [
        ("pass", {}, [], "-"),
        ("timeout", {"tool_errors": 3}, [], "timeout"),
        ("max_steps", {}, [], "max_steps"),
        ("wrong", {"tool_errors": 1}, [], "tool_error"),
        ("wrong", {"degenerate_outputs": 1}, [], "tool_error"),
        ("wrong", {"tags": ["no_tool_used"]}, ["web", "wikipedia"], "retrieval_miss"),
        ("wrong", {"tags": ["no_tool_used"]}, ["file:csv"], "reasoning"),
        ("wrong", {"tags": ["vision_used"]}, ["vision"], "reasoning"),
    ],
)
def test_cause(out, sig, skills, expected):
    assert rel.cause(out, sig, skills) == expected


def test_degenerate_output_detection():
    assert rel._degenerate("White King g1 " + "<unk>" * 20)
    assert rel._degenerate("A un in de " + "ns " * 80)
    board = "\n".join([". . . . . . . k", ". . . . . p p p"] * 8) + " Legal moves: " + "Move.from_uci('d5d8') " * 5
    assert not rel._degenerate(board)
    assert not rel._degenerate("White: King g1, Queen d7, Rook a5. Black: King g8, Queen g6.")


def test_repeat_run_failures_are_all_the_chess_task(repeat):
    rows = rel.failures(repeat)
    assert {r["task"] for r in rows} == {"p09-image-chess-mate"}
    assert sorted((r["run"], r["outcome"], r["cause"]) for r in rows) == [
        ("abl-2026-10-10-baseline", "wrong", "reasoning"),
        ("practice-baseline-r2", "timeout", "timeout"),
        ("practice-baseline-r3", "wrong", "reasoning"),
    ]
    r2 = next(r for r in rows if r["run"] == "practice-baseline-r2")
    assert r2["tool_errors"] >= 1 and r2["degenerate_outputs"] >= 1 and r2["vision_seconds"] > 300


def test_markdown_lists_every_failure(repeat, head):
    md = rel.to_markdown(head, rel.failures(repeat), rel.counts(repeat))
    assert "| 4 | 100.0% | 93.8% |" in md
    assert "p09-image-chess-mate (1/4)" in md
    assert len(re.findall(r"^\| Run \d \| p09", md, re.M)) == 3


def test_chart_renders(tmp_path, repeat):
    pytest.importorskip("matplotlib")
    out = rel.draw_grid(repeat, tmp_path / "grid.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
