"""Tests for report/build_report.py, the static results site.

All data here is the FAKE fixture in tests/fixtures/sample_run/ or synthetic dicts built
inline. Nothing touches GAIA questions, answers or the network.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "sample_run"
sys.path.insert(0, str(ROOT / "report"))

import build_report as br  # noqa: E402

SECTIONS = ["overview", "practice", "trajectory", "evals", "questions", "policy", "analytics", "usage", "failures", "how"]


def _data(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', html, re.S)
    assert m, "inlined data block missing"
    return json.loads(m.group(1))


@pytest.fixture(scope="module")
def fixture_site(tmp_path_factory):
    """PRIVATE build (full GAIA text): what a local --private build shows."""
    out = tmp_path_factory.mktemp("site")
    index, data = br.build(FIXTURE, out, public=False)
    return out, index.read_text(encoding="utf-8"), data


@pytest.fixture(scope="module")
def public_site(tmp_path_factory):
    """PUBLIC build (the default): GAIA items redacted, practice shown in full."""
    out = tmp_path_factory.mktemp("public")
    index, data = br.build(FIXTURE, out)
    return out, index.read_text(encoding="utf-8"), data


# --------------------------------------------------------------------------- full builds


def test_fixture_build_writes_a_complete_site(fixture_site):
    out, html, _ = fixture_site
    assert (out / "assets" / "app.js").is_file()
    assert (out / "assets" / "style.css").is_file()
    assert (out / ".nojekyll").exists()
    for sec in SECTIONS:
        assert f'id="{sec}"' in html, sec
    # Assets are cache-busted so a republished page never pairs new HTML with stale JS.
    assert re.search(r'assets/app\.js\?v=[0-9a-f]{10}"', html)
    assert re.search(r'assets/style\.css\?v=[0-9a-f]{10}"', html)
    assert "{{" not in html  # every template placeholder was filled


def test_fixture_is_labelled_as_fixture(fixture_site):
    _, html, data = fixture_site
    assert data["meta"]["fixture"] is True
    assert "<title>[FIXTURE]" in html
    assert 'class="is-fixture is-private"' in html  # fixture_site is the private build
    assert "FIXTURE DATA" in html


def test_fixture_questions_and_run_selection(fixture_site):
    _, _, data = fixture_site
    qs = data["questions"]
    assert data["meta"]["run_id"] == "fixture-run-002"  # the last submitted run
    assert len(qs) == 10
    q1 = qs[0]
    # answers.jsonl has an older run's "5" for q01 before the current run's "4": latest run wins.
    assert q1["task_id"] == "fixture-q01-library" and q1["answer"] == "4"
    assert [q["n"] for q in qs] == list(range(1, 11))
    statuses = {q["task_id"]: q["status"] for q in qs}
    assert statuses["fixture-q06-rate-limit"] == "error"
    assert statuses["fixture-q07-timeout"] == "timeout"
    assert statuses["fixture-q10-no-answer"] == "empty"
    assert sum(s == "answered" for s in statuses.values()) == 7
    assert data["analytics"]["status"] == {"answered": 7, "error": 1, "timeout": 1, "empty": 1}


def test_fixture_tools_tokens_and_traces(fixture_site):
    _, _, data = fixture_site
    q1 = data["questions"][0]
    # final_answer / python_interpreter are not tools; the sub-agent's own call is counted.
    assert q1["tools"] == {"web_search": 1, "web_agent": 1, "visit_webpage": 1}
    tr = data["traces"]["fixture-q01-library"]
    assert [s["type"] for s in tr["steps"]] == ["task", "planning", "action", "action", "action", "action", "final"]
    assert tr["totals"]["steps"] == 4 and tr["totals"]["sub_agent_calls"] == 1
    deleg = tr["steps"][3]
    assert deleg["step_number"] == 2 and deleg["sub_agents"][0]["agent"] == "web_agent"
    assert deleg["sub_agents"][0]["steps"][1]["tools"] == ["visit_webpage"]
    assert tr["steps"][5]["is_final_answer"] is True
    assert q1["tokens"] == q1["input_tokens"] + q1["output_tokens"] > 0
    usage = {r["name"]: r for r in data["analytics"]["tool_usage"]}
    assert usage["visit_webpage"]["calls"] == 11 and usage["visit_webpage"]["questions"] == 2
    assert len(data["traces"]) == 10


def test_fixture_submission(fixture_site):
    _, _, data = fixture_site
    sub = data["submission"]
    assert sub["latest"]["score"] == 30.0 and sub["latest"]["correct_count"] == 6
    assert [h["score"] for h in sub["history"]] == [20.0, 30.0]
    assert sub["pass_mark"] == 30.0


def test_fixture_evals(fixture_site):
    _, _, data = fixture_site
    ev = data["evals"]
    # 9 run docs; the ablation_*.json and passk_*.json reports in the same folder are not runs.
    assert len(ev["runs"]) == 9
    groups = {g["label"]: g for g in ev["groups"]}
    assert set(groups) == {"baseline", "multi", "no web", "no wiki", "verify"}
    base = groups["baseline"]
    assert base["is_baseline"] and base["runs"] == 3 and ev["baseline"] == "baseline"
    assert base["passk"]["k"] == [1, 2, 3]
    pa, ph = base["passk"]["pass_at_k"], base["passk"]["pass_hat_k"]
    assert pa[0] == pytest.approx(ph[0])  # k=1: both equal mean accuracy
    assert pa == sorted(pa) and ph == sorted(ph, reverse=True)
    assert pa[0] == pytest.approx(base["accuracy"])
    # Skipped (budget-stopped) items do not count toward n.
    assert groups["no wiki"]["items_per_run"] == 11 and groups["no wiki"]["skipped"] == 1
    assert groups["no web"]["mcnemar_p"] == pytest.approx(0.008)
    assert groups["no web"]["delta_pp"] == pytest.approx((groups["no web"]["accuracy"] - base["accuracy"]) * 100)
    assert base["by_skill"] and all(0 <= s["accuracy"] <= 1 for s in base["by_skill"])
    lo, hi = base["ci95"]
    assert lo < base["accuracy"] < hi


def test_fixture_errors_use_analysis_descriptions(fixture_site):
    _, _, data = fixture_site
    err = data["errors"]
    assert err["descriptions"]["rate_limited"].startswith("A model call hit")
    assert err["n_tasks"] == 10 and err["clean"] == 5
    assert all(it["tags"] for it in err["items"])  # clean tasks are not listed as failures
    q7 = next(it for it in err["items"] if it["task_id"] == "fixture-q07-timeout")
    assert q7["n"] == 7 and q7["tags"] == ["timeout", "paging_loop"]


def test_untrusted_text_cannot_break_out_of_the_data_block(fixture_site):
    _, html, data = fixture_site
    # q01's observation contains a literal <script> tag from a web page.
    obs = json.dumps(data["traces"]["fixture-q01-library"])
    assert "<script>alert" in obs
    assert "<script>alert" not in html
    assert html.count("</script>") == 3  # theme bootstrap, data block, app.js


def test_empty_root_builds_graceful_empty_states(tmp_path):
    out = tmp_path / "site"
    index, data = br.build(tmp_path / "nothing-here", out)
    html = index.read_text(encoding="utf-8")
    for sec in SECTIONS:
        assert f'id="{sec}"' in html
    assert data["questions"] == [] and data["traces"] == {}
    assert data["submission"]["latest"] is None
    assert data["evals"] is None and data["errors"] is None
    assert data["practice"] is None and data["usage"] is None
    assert data["meta"]["public"] is True
    assert data["meta"]["fixture"] is False and data["meta"]["run_id"] is None
    assert not any(data["meta"]["sources"].values())
    assert "[FIXTURE]" not in html
    assert data["analytics"]["tool_usage"] == [] and data["analytics"]["seconds"]["median"] is None


def test_partial_inputs_answers_without_traces(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "answers.jsonl").write_text(
        json.dumps({"task_id": "t1", "question": "Q?", "answer": "42", "steps": 3, "seconds": 5.0, "run_id": "r1"})
        + "\nnot json\n"
        + json.dumps({"task_id": "t2", "question": "Q2?", "answer": "", "error": "Timeout after 600s", "run_id": "r1"})
        + "\n",
        encoding="utf-8",
    )
    _, data = br.build(tmp_path, tmp_path / "site")
    qs = data["questions"]
    assert [q["status"] for q in qs] == ["answered", "timeout"]
    assert not any(q["has_trace"] for q in qs)
    assert data["meta"]["run_id"] == "r1"


def test_cli_refuses_to_publish_fixture_into_docs():
    assert br.main(["--root", str(FIXTURE), "--out", str(ROOT / "docs")]) == 2


def test_cli_refuses_a_private_build_into_docs(tmp_path):
    assert br.main(["--private", "--root", str(tmp_path), "--out", str(ROOT / "docs")]) == 2


def test_cli_writes_elsewhere(tmp_path, capsys):
    assert br.main(["--root", str(FIXTURE), "--out", str(tmp_path / "s")]) == 0
    out = capsys.readouterr().out
    assert "FIXTURE" in out and "PUBLIC" in out
    assert br.main(["--private", "--root", str(FIXTURE), "--out", str(tmp_path / "p")]) == 0
    assert "PRIVATE" in capsys.readouterr().out


# --------------------------------------------------------------------------- units


def test_pass_at_k_and_pass_hat_k_closed_forms():
    assert br.pass_at_k(3, 1, 2) == pytest.approx(2 / 3)  # 1 - C(2,2)/C(3,2)
    assert br.pass_hat_k(3, 1, 2) == 0.0
    assert br.pass_hat_k(3, 2, 2) == pytest.approx(1 / 3)
    assert br.pass_at_k(5, 0, 3) == 0.0
    assert br.pass_at_k(5, 5, 3) == br.pass_hat_k(5, 5, 3) == 1.0
    for n, c in [(4, 2), (6, 1), (10, 7)]:
        assert br.pass_at_k(n, c, 1) == pytest.approx(c / n) == pytest.approx(br.pass_hat_k(n, c, 1))
    with pytest.raises(ValueError):
        br.pass_at_k(2, 1, 3)


def test_wilson_interval():
    assert br.wilson(0, 0) is None
    lo, hi = br.wilson(10, 20)
    assert lo == pytest.approx(0.2993, abs=1e-3) and hi == pytest.approx(0.7007, abs=1e-3)
    lo, hi = br.wilson(0, 12)
    assert lo == 0.0 and 0 < hi < 0.3


def test_passk_from_repeated_runs():
    runs = [
        {"items": [{"id": "a", "correct": True}, {"id": "b", "correct": False}]},
        {"items": [{"id": "a", "correct": True}, {"id": "b", "correct": True}]},
    ]
    pk = br._passk(runs)
    assert pk["k"] == [1, 2]
    assert pk["pass_at_k"] == pytest.approx([0.75, 1.0])
    assert pk["pass_hat_k"] == pytest.approx([0.75, 0.5])
    assert br._passk(runs[:1]) is None


def test_status_of():
    assert br.status_of("42", None) == "answered"
    assert br.status_of("", "Timeout after 600s") == "timeout"
    assert br.status_of("", "TimeoutError: timed out") == "timeout"
    assert br.status_of("", "RateLimitError: 429") == "error"
    assert br.status_of(None, None) == "empty"
    assert br.status_of("   ", "") == "empty"


def test_tools_in_code_ignores_methods_and_unknown_names():
    code = 'x = web_search("a")\nprint(visit_webpage(url=x))\nobj.web_search(1)\nfoo(2)\nweb_search ("b")'
    assert br.tools_in_code(code, set(br.KNOWN_TOOLS)) == ["web_search", "visit_webpage", "web_search"]
    assert br.tools_in_code(None, set(br.KNOWN_TOOLS)) == []


def test_legacy_trace_schema_is_normalized():
    raw = {
        "task_id": "t",
        "steps": [
            {"step": 1, "code": 'r = wikipedia_page("X")\nprint(r)', "observations": "obs", "error": None, "final": False},
            {"step": 2, "code": 'final_answer("y")', "observations": None, "error": None, "final": True},
        ],
    }
    tr = br.normalize_trace(raw, set(br.KNOWN_TOOLS))
    assert [s["type"] for s in tr["steps"]] == ["action", "final"]
    assert tr["steps"][0]["observation"] == "obs" and tr["steps"][0]["tools"] == ["wikipedia_page"]
    assert tr["totals"]["steps"] == 2  # a legacy final step with code was an action
    assert tr["totals"]["tool_calls_by_name"] == {"wikipedia_page": 1}


def test_python_interpreter_only_calls_fall_back_to_code_parsing():
    raw = {"steps": [{"type": "action", "code": 'print(read_file("f.xlsx"))',
                      "tool_calls": [{"name": "python_interpreter", "arguments": "..."}]}],
           "totals": {"tool_calls_by_name": {"python_interpreter": 1}}}
    tr = br.normalize_trace(raw, set(br.KNOWN_TOOLS))
    assert tr["totals"]["tool_calls_by_name"] == {"read_file": 1}


def test_clip_marker_and_idempotence():
    s = "x" * 4100
    c = br.clip(s)
    assert c.endswith("…[truncated 100 chars]") and len(c) == 4000 + len("…[truncated 100 chars]")
    assert br.clip("short") == "short" and br.clip(None) is None


def test_inline_json_escapes_html_breakers():
    s = br.inline_json({"a": "</script><b>& "})
    assert "<" not in s and ">" not in s and "&" not in s and " " not in s
    assert json.loads(s) == {"a": "</script><b>& "}


def test_pick_run_prefers_the_last_submitted_run():
    answers = [{"run_id": "a"}, {"run_id": "b"}]
    assert br.pick_run(answers, {}, [{"run_id": "a"}]) == "a"
    assert br.pick_run(answers, {}, []) == "b"
    assert br.pick_run([], {}, []) is None


def test_is_eval_run():
    assert br.is_eval_run({"run_id": "r", "items": []})
    assert not br.is_eval_run({"tag": "x", "configs": []})
    assert not br.is_eval_run({"runs": ["r"]})


def test_percentile():
    assert br.percentile([], 0.5) is None
    assert br.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert math.isclose(br.percentile([10.0], 0.9), 10.0)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_parses():
    subprocess.run(["node", "--check", str(ROOT / "report" / "assets" / "app.js")], check=True)


# --------------------------------------------------------------------------- public build: contamination policy


GAIA_RUN = FIXTURE / "runs" / "traces" / "fixture-run-002"
TEXT_FIELDS = ("model_output", "code", "observation", "error", "action_output", "task", "final_answer", "question")


def _gaia_strings() -> tuple[set[str], set[str]]:
    """(distinctive text lines, task ids) from the fixture's GAIA side: questions, answers,
    every step's reasoning, code, observations and errors, sub-agent tasks, error evidence."""
    texts: list[str] = []
    ids: set[str] = set()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in TEXT_FIELDS and isinstance(v, str):
                    texts.append(v)
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    for f in GAIA_RUN.glob("*.json"):
        tr = json.loads(f.read_text(encoding="utf-8"))
        ids.add(tr["task_id"])
        walk(tr)
    for line in (FIXTURE / "runs" / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        texts += [row.get("question") or "", row.get("answer") or "", row.get("raw") or "", row.get("error") or ""]
    ea = json.loads((FIXTURE / "runs" / "error_analysis.json").read_text(encoding="utf-8"))
    texts += [it.get("evidence") or "" for it in ea["items"]]
    lines = set()
    for t in texts:
        for ln in t.splitlines():
            ln = ln.strip()
            if len(ln) >= 12 and not ln.startswith(("Execution logs:", "Last output from code snippet:")):
                lines.add(ln)
    lines |= {"1284.50", "Qh5", "olive oil"}  # short but distinctive submitted answers
    # Generic code lines the practice fixture also contains ("import pandas as pd") are not GAIA text.
    practice = (FIXTURE / "evals" / "practice" / "tasks.jsonl").read_text(encoding="utf-8")
    for f in (FIXTURE / "runs" / "traces").glob("fixture-eval-*/*.json"):
        practice += f.read_text(encoding="utf-8")
    return {ln for ln in lines if ln not in practice}, ids


def _site_text(out: Path) -> str:
    """Everything the public site serves, with the inlined JSON decoded (so escaping can't hide text)."""
    parts = []
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.suffix in {".html", ".json", ".js", ".css"}:
            raw = f.read_text(encoding="utf-8")
            parts.append(raw)
            if f.name == "index.html":
                parts.append(json.dumps(_data(raw), ensure_ascii=False))
    return "\n".join(parts)


def test_public_build_never_emits_gaia_question_answer_or_output_text(public_site):
    out, html, data = public_site
    assert data["meta"]["public"] is True and "is-private" not in html
    lines, ids = _gaia_strings()
    assert len(lines) > 40  # the check is not vacuous
    site = _site_text(out)
    leaked = sorted(ln for ln in lines if ln in site)
    assert not leaked, f"GAIA text in the public site: {leaked[:5]}"
    for tid in ids:  # only the short id is published, never the full task id
        assert tid not in site
    # Positive control: the same check finds the text in a private build.
    priv = br.collect(FIXTURE, public=False)
    priv_text = json.dumps(priv, ensure_ascii=False, default=str)
    assert sum(ln in priv_text for ln in lines) > 20


def test_public_gaia_rows_and_skeletons_have_only_allowed_fields(public_site):
    _, _, data = public_site
    allowed = {"n", "task_id", "attachment", "status", "error_type", "steps", "model_calls", "seconds",
               "input_tokens", "output_tokens", "tokens", "tools", "tags", "has_trace", "run_id"}
    qs = data["questions"]
    assert len(qs) == 10
    for q in qs:
        assert set(q) <= allowed, set(q) - allowed
        assert len(q["task_id"]) >= 8
    by_n = {q["n"]: q for q in qs}
    assert by_n[2]["attachment"] == "xlsx" and by_n[8]["attachment"] == "png" and by_n[1]["attachment"] is None
    assert by_n[7]["tags"] == ["timeout", "paging_loop"] and by_n[7]["status"] == "timeout"
    assert by_n[1]["tools"] == {"web_search": 1, "web_agent": 1, "visit_webpage": 1}
    forbidden = {"model_output", "code", "observation", "action_output", "question", "final_answer", "task", "arguments"}

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for x in obj:
                yield from keys(x)

    for tid, tr in data["traces"].items():
        assert tr["redacted"] is True and tr["task_id"] == tid
        assert not forbidden & set(keys(tr)), tid
        assert all(isinstance(st["error"], bool) for st in tr["steps"])
    q1 = data["traces"][by_n[1]["task_id"]]
    assert [s["type"] for s in q1["steps"]] == ["task", "planning", "action", "action", "action", "action", "final"]
    assert q1["steps"][3]["sub_agents"][0]["agent"] == "web_agent"
    assert q1["steps"][3]["sub_agents"][0]["steps"][1]["tools"] == ["visit_webpage"]
    assert data["errors"]["redacted"] and all(set(it) == {"task_id", "n", "tags"} for it in data["errors"]["items"])


def test_public_build_shows_practice_in_full(public_site):
    out, html, data = public_site
    site = _site_text(out)
    tasks = [json.loads(x) for x in (FIXTURE / "evals" / "practice" / "tasks.jsonl").read_text().splitlines() if x]
    for t in tasks:
        assert t["question"] in site and t["gold"] in site
    pr = data["practice"]
    assert pr["default_run"] == "fixture-eval-baseline-1"  # the baseline run with the most traces
    assert len(pr["runs"]) == 9 and pr["tasks"]["practice-02"]["attachment"] == "csv"
    assert sorted(pr["trace_files"]) == ["fixture-eval-baseline-1", "fixture-eval-multi-1"]
    assert len(pr["trace_files"]["fixture-eval-baseline-1"]) == 12
    # Practice traces are complete (reasoning, code, observations) and written as files.
    tr = pr["traces"]["fixture-eval-baseline-1"]["practice-02"]
    codes = [s["code"] for s in tr["steps"] if s.get("code")]
    assert any("read_file" in c for c in codes) and any(s.get("observation") for s in tr["steps"])
    assert (out / "traces" / "fixture-eval-baseline-1" / "practice-02.json").is_file()
    items = {it["id"]: it for it in pr["items"]["fixture-eval-baseline-1"]}
    assert all(it["has_trace"] for it in items.values())
    assert all(it["gold"] for it in items.values())
    assert sum(it["has_trace"] for it in pr["items"]["fixture-eval-multi-1"]) == 3
    # Practice traces never leak into the GAIA scoreboard.
    assert not any(q["task_id"].startswith("practice") for q in data["questions"])


def test_public_build_writes_no_gaia_trace_files(public_site):
    out, _, _ = public_site
    runs = {p.name for p in (out / "traces").iterdir() if p.is_dir()}
    assert runs == {"fixture-eval-baseline-1", "fixture-eval-multi-1"}
    assert (out / "traces" / br.TRACE_MARKER).exists()


def test_rebuild_replaces_its_own_trace_files_only(tmp_path):
    out = tmp_path / "s"
    br.build(FIXTURE, out)
    stale = out / "traces" / "old-run" / "x.json"
    stale.parent.mkdir()
    stale.write_text("{}")
    br.build(FIXTURE, out)
    assert not stale.exists()  # the marker says the folder is ours
    foreign = tmp_path / "f"
    (foreign / "traces").mkdir(parents=True)
    (foreign / "traces" / "keep.json").write_text("{}")
    br.build(FIXTURE, foreign)
    assert (foreign / "traces" / "keep.json").exists()  # no marker: never deleted


def test_practice_traces_beyond_the_inline_budget_are_fetched(tmp_path):
    data = br.collect(FIXTURE)
    prac = data.pop("_practice_traces")
    br.inline_practice_traces(data["practice"], prac, budget=1)
    inlined = sum(len(v) for v in data["practice"]["traces"].values())
    assert inlined == 1 and data["practice"]["traces"].get("fixture-eval-baseline-1")


def test_usage_aggregates_requests_models_and_scopes(public_site):
    _, _, data = public_site
    u = data["usage"]
    n_rows = len((FIXTURE / "runs" / "usage.jsonl").read_text().splitlines())
    assert u["totals"]["requests"] == n_rows
    assert sum(m["requests"] for m in u["models"]) == n_rows
    assert set(u["scopes"]) == {"gaia", "practice"}
    assert u["all_free"] is True and u["non_free_models"] == []
    by = {m["model"]: m for m in u["models"]}
    assert by["fixture/fake-fallback-4b:free"]["fallback_served"] == by["fixture/fake-fallback-4b:free"]["requests"] > 0
    assert any(k.startswith("rate_limit") for k in by["fixture/fake-model-8b:free"]["attempt_errors"])
    assert u["totals"]["retries"] > 0 and u["totals"]["failed"] >= 1


def test_usage_flags_a_non_free_model():
    rows = [{"run_id": "r", "status": "ok", "model": "vendor/paid-model", "input_tokens": 5, "output_tokens": 1},
            {"run_id": "r", "status": "ok", "model": "vendor/x:free"}]
    u = br.build_usage(rows, "r", set())
    assert u["all_free"] is False and u["non_free_models"] == ["vendor/paid-model"]
    assert br.build_usage([], None, set()) is None


def test_usage_never_copies_error_messages():
    rows = [{"run_id": "r", "status": "error", "requested_model": "m:free",
             "errors": [{"model": "m:free", "kind": "bad_request", "status": 400, "message": "SECRET PROMPT ECHO"}]}]
    assert "SECRET PROMPT ECHO" not in json.dumps(br.build_usage(rows, None, set()))


def test_short_ids_lengthen_only_on_collision():
    ids = ["abcdef1234", "abcdef1299", "0123456789abcdef"]
    out = br.short_ids(ids)
    assert out["0123456789abcdef"] == "01234567"
    assert out["abcdef1234"] == "abcdef123" and out["abcdef1299"] == "abcdef129"
    assert len(set(out.values())) == 3


def test_attachment_type_is_an_extension_only():
    assert br.attachment_type("1f975693-876d-457b-a649-393859e79bf3.MP3") == "mp3"
    assert br.attachment_type(None, {"steps": [{"type": "task", "model_output": "Q?\n\nAttached file: data/files/x.xlsx"}]}) == "xlsx"
    assert br.attachment_type("") is None and br.attachment_type("weird.name with space") is None


def test_submission_reader_accepts_flat_and_nested_rows():
    sub = br.build_submission([
        {"submitted_at": "t1", "score": 25.0, "correct_count": 5, "total_attempted": 20, "message": "ok"},
        {"submitted_at": "t2", "error": "HTTP 500"},
        {"submitted_at": "t3", "run_id": "r", "response": {"score": 35.0, "correct_count": 7}},
    ])
    assert [h["score"] for h in sub["history"]] == [25.0, 35.0]
    assert sub["latest"]["run_id"] == "r"


def test_legacy_wrapped_trace_files_are_unwrapped():
    raw = {"task_id": "t", "question": "Q", "trace": [{"step": 1, "code": 'web_search("x")', "observations": "o"}]}
    tr = br.normalize_trace(br.unwrap_trace(raw), set(br.KNOWN_TOOLS))
    assert tr["task_id"] == "t" and tr["totals"]["tool_calls_by_name"] == {"web_search": 1}


def test_practice_task_traces_outside_an_eval_run_stay_off_the_gaia_board(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "root")
    root = tmp_path / "root"
    (root / "FIXTURE").unlink()
    stray = root / "runs" / "traces" / "practice-smoke"
    stray.mkdir()
    (stray / "practice-01.json").write_text(json.dumps({"task_id": "practice-01", "question": "Q", "steps": []}))
    data = br.collect(root)
    assert "practice-smoke" not in {q.get("run_id") for q in data["questions"]}
    assert len(data["questions"]) == 10


def test_error_analysis_of_another_run_is_not_shown_as_gaia_failures(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "root")
    root = tmp_path / "root"
    (root / "FIXTURE").unlink()
    ea = json.loads((root / "runs" / "error_analysis.json").read_text())
    ea["run_id"] = "fixture-eval-baseline-1"
    (root / "runs" / "error_analysis.json").write_text(json.dumps(ea))
    data = br.collect(root)
    assert data["errors"] is None
    assert all(q["tags"] == [] for q in data["questions"])
