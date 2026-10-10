"""Offline tests for the CORE pieces: normalize_answer, submit payload, provider_info, run resume."""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaia_agent import config
from gaia_agent.agent import INSTRUCTIONS, build_task, normalize_answer
from gaia_agent import run as runmod
from gaia_agent import submit as submod


# ----------------------------------------------------------------------------- normalize_answer

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  42  ", "42"),
        ("FINAL ANSWER: 42", "42"),
        ("final answer: Paris", "Paris"),
        ("Final Answer - Paris", "Paris"),
        ("**FINAL ANSWER:** Paris", "Paris"),
        ('"Paris"', "Paris"),
        ("'Paris'", "Paris"),
        ("“Paris”", "Paris"),
        ("`e4`", "e4"),
        ("**Paris**", "Paris"),
        ("Paris.", "Paris"),
        ('"Paris."', "Paris"),
        ("FINAL ANSWER: \"Paris.\"", "Paris"),
        ("3.14", "3.14"),
        ("3.14.", "3.14"),
        ("17.", "17"),
        ("Washington, D.C.", "Washington, D.C."),
        ("U.S.", "U.S."),
        ("Sammy Davis Jr.", "Sammy Davis Jr."),
        ("Acme Inc.", "Acme Inc."),
        ("New   York\n City", "New York City"),
        ("a,b,c", "a, b, c"),
        ("a ,b ,  c", "a, b, c"),
        ("apple, banana, cherry.", "apple, banana, cherry"),
        ("a, b, c,", "a, b, c"),
        ("1,234", "1234"),
        ("1,234,567.5", "1234567.5"),
        ("-12,000", "-12000"),
        ("1, 2, 3", "1, 2, 3"),
        ('"a", "b"', '"a", "b"'),  # separate quoted items are not one wrapped string
        ("", ""),
        ("   ", ""),
        (None, ""),
        (42, "42"),
        (3.0, "3"),
        (2.5, "2.5"),
        (float("nan"), ""),
        (True, "True"),
        (["b", "a"], "b, a"),
        (("x", 1), "x, 1"),
        ("Rd5", "Rd5"),
        ("St. Petersburg", "St. Petersburg"),
        ("Dr.", "Dr."),
        ("...", "..."),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_normalize_is_idempotent():
    for s in ["FINAL ANSWER: a , b", "'x.'", "1,000", "Washington, D.C.", "Jr.", "  q  "]:
        once = normalize_answer(s)
        assert normalize_answer(once) == once


def test_instructions_cover_format_rules():
    text = INSTRUCTIONS.lower()
    for needle in ["exact match", "comma-separated", "thousands", "articles", "final answer", "attached file"]:
        assert needle in text


def test_build_task_appends_file():
    assert build_task("Q?", "/tmp/x.xlsx", "x.xlsx").endswith("Attached file: /tmp/x.xlsx")
    assert "could not be downloaded" in build_task("Q?", None, "x.xlsx")
    assert build_task("  Q?  ", None, None) == "Q?"


def test_missing_attachment_fails_fast():
    # Regression for run gaia-v1: a missing file made the agent loop to max_steps hunting for it.
    task = build_task("Q?", None, "x.xlsx")
    assert "do not search for it" in task and "final_answer('unknown')" in task


# ----------------------------------------------------------------------------- submit payload

def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_build_payload_latest_answer_per_task(tmp_path):
    p = tmp_path / "answers.jsonl"
    _write_jsonl(
        p,
        [
            {"task_id": "t1", "answer": "old", "error": None},
            {"task_id": "t2", "answer": "FINAL ANSWER: 7", "error": None},
            {"task_id": "t1", "answer": "new", "error": None},
            {"task_id": "t3", "answer": "keep", "error": None},
            {"task_id": "t3", "answer": "", "error": "Timeout after 420s"},  # must not hide 'keep'
            {"task_id": "t4", "answer": "", "error": "boom"},  # only an errored attempt
        ],
    )
    with open(p, "a") as fh:
        fh.write("{not json\n")
    payload = submod.build_payload(p, username="bacool", space="gaia-agent")
    assert payload["username"] == "bacool"
    assert payload["agent_code"] == "https://huggingface.co/spaces/bacool/gaia-agent/tree/main"
    got = {a["task_id"]: a["submitted_answer"] for a in payload["answers"]}
    assert got == {"t1": "new", "t2": "7", "t3": "keep", "t4": ""}
    assert all(set(a) == {"task_id", "submitted_answer"} for a in payload["answers"])


def test_build_payload_order_and_space(tmp_path):
    p = tmp_path / "answers.jsonl"
    _write_jsonl(p, [{"task_id": "b", "answer": "2"}, {"task_id": "a", "answer": "1"}])
    payload = submod.build_payload(p, space="someone/other-space", task_ids=["a", "b", "c"])
    assert [a["task_id"] for a in payload["answers"]] == ["a", "b"]
    assert payload["agent_code"] == "https://huggingface.co/spaces/someone/other-space/tree/main"


def test_submit_main_is_dry_run_by_default(tmp_path, monkeypatch, capsys):
    p = tmp_path / "answers.jsonl"
    _write_jsonl(p, [{"task_id": "abc", "answer": "x"}])

    def boom(*a, **k):
        raise AssertionError("network POST attempted during dry run")

    monkeypatch.setattr(submod.requests, "post", boom)
    monkeypatch.setattr(submod, "_question_order", lambda: None)
    assert submod.main(["--path", str(p)]) is None
    assert "DRY RUN" in capsys.readouterr().out


def test_submit_missing_file(tmp_path):
    payload = submod.build_payload(tmp_path / "nope.jsonl")
    assert payload["answers"] == []


# ----------------------------------------------------------------------------- config

def test_provider_info_has_no_secrets(monkeypatch):
    secret = "sk-test-SECRET-VALUE-123"
    monkeypatch.setattr(config, "_env_loaded", True)
    monkeypatch.setenv("GAIA_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.delenv("GAIA_MODEL_ID", raising=False)
    monkeypatch.delenv("GAIA_VISION_MODEL_ID", raising=False)
    info = config.provider_info()
    assert secret not in json.dumps(info)
    assert info["provider"] == "gemini"
    assert info["key_env"] == "GEMINI_API_KEY"
    assert info["key_present"] is True
    assert info["base_url"].startswith("https://generativelanguage.googleapis.com")


def test_provider_presets_and_overrides(monkeypatch):
    monkeypatch.setattr(config, "_env_loaded", True)
    monkeypatch.setenv("GAIA_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GAIA_MODEL_ID", "my-model")
    monkeypatch.delenv("GAIA_VISION_MODEL_ID", raising=False)
    info = config.provider_info()
    assert info["base_url"] == "https://api.groq.com/openai/v1"
    assert info["key_present"] is False
    # GAIA_MODEL_ID alone must not move vision calls onto a (possibly text-only) model
    assert info["model_id"] == "my-model"
    assert info["vision_model_id"] == "meta-llama/llama-4-scout-17b-16e-instruct"
    monkeypatch.setenv("GAIA_VISION_MODEL_ID", "vis")
    assert config.provider_info()["vision_model_id"] == "vis"
    with pytest.raises(RuntimeError):
        config.get_model()  # no key -> clear error, no network

    monkeypatch.setenv("GAIA_PROVIDER", "lmstudio")
    info = config.provider_info()
    assert info["base_url"] == "http://localhost:1234/v1" and info["key_env"] is None

    monkeypatch.setenv("GAIA_PROVIDER", "nope")
    with pytest.raises(ValueError):
        config.provider_info()


def test_get_model_builds_without_network(monkeypatch):
    monkeypatch.setattr(config, "_env_loaded", True)
    monkeypatch.setenv("GAIA_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("GAIA_MODEL_ID", "gemini-test")
    m = config.get_model()
    assert m.model_id == "gemini-test"
    assert "generativelanguage" in str(m.client_kwargs["base_url"])


def test_load_env_parses_and_skips_blocked(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nexport GAIA_T_A=one\nGAIA_T_B=\"two words\"\nXAI_API_KEY=paid\nGAIA_T_C='3'\nnot a line\n"
    )
    for k in ["GAIA_T_A", "GAIA_T_B", "GAIA_T_C", "XAI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GAIA_T_C", "existing")
    names = config.load_env(env)
    import os

    assert os.environ["GAIA_T_A"] == "one"
    assert os.environ["GAIA_T_B"] == "two words"
    assert os.environ["GAIA_T_C"] == "existing"  # env wins without override
    assert "XAI_API_KEY" not in os.environ
    assert set(names) == {"GAIA_T_A", "GAIA_T_B"}


# ----------------------------------------------------------------------------- run.py resume logic

QUESTIONS = [
    {"task_id": "aaa111", "question": "Q1", "file_name": ""},
    {"task_id": "bbb222", "question": "Q2", "file_name": "f.xlsx"},
    {"task_id": "ccc333", "question": "Q3", "file_name": ""},
]


def test_answered_ok_uses_latest_record():
    recs = [
        {"task_id": "aaa111", "answer": "x", "error": None},
        {"task_id": "bbb222", "answer": "y", "error": None},
        {"task_id": "bbb222", "answer": "", "error": "Timeout"},
        {"task_id": "ccc333", "answer": "", "error": None},
    ]
    assert runmod.answered_ok(recs) == {"aaa111"}


def test_select_questions_flags():
    recs = [{"task_id": "aaa111", "answer": "x", "error": None}]
    ids = lambda qs: [q["task_id"] for q in qs]  # noqa: E731
    assert ids(runmod.select_questions(QUESTIONS, recs)) == ["bbb222", "ccc333"]
    assert ids(runmod.select_questions(QUESTIONS, recs, redo=True)) == ["aaa111", "bbb222", "ccc333"]
    assert ids(runmod.select_questions(QUESTIONS, recs, only=["aaa", "ccc"])) == ["ccc333"]
    assert ids(runmod.select_questions(QUESTIONS, recs, only=["aaa,ccc"], redo=True)) == ["aaa111", "ccc333"]
    assert ids(runmod.select_questions(QUESTIONS, recs, limit=1)) == ["bbb222"]


def test_main_resumes_and_redoes(tmp_path):
    out = tmp_path / "answers.jsonl"
    calls = []
    attempts = {}

    def fake_solve(question, task_id=None, file_name=None):
        calls.append((task_id, file_name))
        attempts[task_id] = attempts.get(task_id, 0) + 1
        if task_id == "ccc333" and attempts[task_id] == 1:
            return {"answer": "", "raw": None, "steps": 1, "error": "RuntimeError: flaky"}
        return {"answer": f"FINAL ANSWER: ans-{question}", "raw": "r", "steps": 2, "error": None}

    kw = dict(solve_fn=fake_solve, questions=QUESTIONS, out_path=out, traces_dir=None)

    runmod.main([], **kw)
    assert [c[0] for c in calls] == ["aaa111", "bbb222", "ccc333"]
    assert calls[1] == ("bbb222", "f.xlsx") and calls[0] == ("aaa111", None)
    recs = runmod.load_records(out)
    assert len(recs) == 3
    assert {"task_id", "question", "answer", "raw", "steps", "error", "seconds", "provider_info"} <= set(recs[0])
    assert runmod.answered_ok(recs) == {"aaa111", "bbb222"}
    assert recs[0]["answer"] == "ans-Q1"  # runner normalizes what the solver returns

    # second run: only the errored task is retried
    calls.clear()
    runmod.main([], **kw)
    assert [c[0] for c in calls] == ["ccc333"]
    assert runmod.answered_ok(runmod.load_records(out)) == {"aaa111", "bbb222", "ccc333"}

    # nothing left to do
    calls.clear()
    runmod.main([], **kw)
    assert calls == []

    # --redo with --only re-runs just the selected prefix
    runmod.main(["--redo", "--only", "bbb"], **kw)
    assert [c[0] for c in calls] == ["bbb222"]
    assert len(runmod.load_records(out)) == 5


def test_run_with_timeout_does_not_block():
    started = threading.Event()
    seen = {}

    def hanging_solve(question, task_id=None, file_name=None, stop_event=None):
        seen["stop_event"] = stop_event
        started.set()
        time.sleep(5)
        return {"answer": "late", "error": None}

    t0 = time.time()
    res = runmod.run_with_timeout(hanging_solve, {"task_id": "x", "question": "q"}, timeout=0.3)
    assert time.time() - t0 < 2
    assert res["error"].startswith("Timeout") and res["answer"] == ""
    assert started.is_set() and seen["stop_event"].is_set()


def test_run_with_timeout_catches_exceptions():
    def bad_solve(question, task_id=None, file_name=None):
        raise ValueError("nope")

    res = runmod.run_with_timeout(bad_solve, {"task_id": "x", "question": "q"}, timeout=5)
    assert res["error"] == "ValueError: nope" and res["answer"] == ""


def test_submit_reads_run_output(tmp_path):
    out = tmp_path / "answers.jsonl"
    runmod.main(
        [],
        solve_fn=lambda q, task_id=None, file_name=None: {"answer": normalize_answer(f"'{q}.'"), "error": None},
        questions=QUESTIONS,
        out_path=out,
        traces_dir=None,
    )
    payload = submod.build_payload(out, task_ids=[q["task_id"] for q in QUESTIONS])
    assert [a["submitted_answer"] for a in payload["answers"]] == ["Q1", "Q2", "Q3"]


# ----------------------------------------------------------------------------- solve() end to end (fake model)

class _ScriptedModel:
    """Minimal smolagents Model stand-in: replays canned code actions, no network."""

    def __init__(self, replies):
        from smolagents.models import Model

        self._base = Model(model_id="fake")
        self.replies = list(replies)
        self.model_id = "fake"
        self.seen_system = None

    def __getattr__(self, name):
        return getattr(self._base, name)

    def generate(self, messages, stop_sequences=None, **kwargs):
        from smolagents.models import ChatMessage, MessageRole
        from smolagents.monitoring import TokenUsage

        if self.seen_system is None:
            first = messages[0]
            content = first.content if hasattr(first, "content") else first["content"]
            self.seen_system = content if isinstance(content, str) else " ".join(c.get("text", "") for c in content)
        reply = self.replies.pop(0) if self.replies else "<code>\nfinal_answer('fallback')\n</code>"
        return ChatMessage(role=MessageRole.ASSISTANT, content=reply, token_usage=TokenUsage(input_tokens=1, output_tokens=1))

    def __call__(self, *a, **k):
        return self.generate(*a, **k)


def test_solve_end_to_end_with_fake_model():
    from gaia_agent.agent import build_agent, solve

    model = _ScriptedModel([
        "Thought: compute\n<code>\nx = 6 * 7\nprint(x)\n</code>",
        "Thought: an empty answer must be rejected by the check\n<code>\nfinal_answer('')\n</code>",
        "Thought: done\n<code>\nfinal_answer('FINAL ANSWER: 42.')\n</code>",
    ])
    agent = build_agent(model=model, tools=[])
    res = solve("What is six times seven?", agent=agent)
    assert res["error"] is None, res["error"]
    assert res["answer"] == "42"
    assert res["steps"] == 3
    assert res["trace"][0]["observations"] and "42" in res["trace"][0]["observations"]
    assert res["trace"][1]["error"] and "empty" in res["trace"][1]["error"]
    assert "EXACT MATCH" in model.seen_system  # instructions reached the system prompt
    assert agent.max_steps == 12


def test_solve_reports_errors_instead_of_raising():
    from gaia_agent.agent import solve

    class Broken:
        memory = None

        def run(self, *a, **k):
            raise RuntimeError("model down")

    res = solve("q", agent=Broken())
    assert res["answer"] == "" and "model down" in res["error"]


# ----------------------------------------------------------------------------- network (free course API)

@pytest.mark.network
def test_fetch_questions_and_download_file(tmp_path):
    from gaia_agent.files import download_task_file

    qs = runmod.fetch_questions(refresh=True, path=tmp_path / "questions.json")
    assert len(qs) >= 1 and all("task_id" in q and "question" in q for q in qs)
    assert (tmp_path / "questions.json").is_file()
    with_file = [q for q in qs if q.get("file_name")]
    if with_file:
        q = with_file[0]
        path = download_task_file(q["task_id"], q["file_name"], files_dir=tmp_path / "files")
        if path is None:  # the scoring API answered 404 ("No file path associated with task_id")
            pytest.skip("scoring API /files returned 404 for this task; download_task_file returned None cleanly")
        assert Path(path).stat().st_size > 0
        assert Path(path).name == f"{q['task_id']}_{q['file_name']}"
        # cached: second call returns the same path
        assert download_task_file(q["task_id"], q["file_name"], files_dir=tmp_path / "files") == path
