"""Integration wiring (wave 3): provider delegation, the env contract, tracing, the verifier,
the optional HF Hub attachment path and the runners' run ids.

Offline: every model is a scripted stand-in or is built without being called; no request is
sent anywhere. conftest.py points GAIA_RUNS_DIR at a temp dir, so traces land there.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

from gaia_agent import agent as agent_mod
from gaia_agent import config
from gaia_agent import files as files_mod
from gaia_agent import providers as P

WEB_TOOLS = {"web_search", "visit_webpage", "wikipedia_search", "wikipedia_page", "wikipedia_revision",
             "wikipedia_page_info"}
MEDIA_TOOLS = {"read_file", "run_python_file", "transcribe_audio", "image_question", "youtube_transcript",
               "youtube_audio_transcript", "youtube_frames_question"}


@pytest.fixture
def offline_providers(tmp_path, monkeypatch):
    """Provider module pointed at tmp files, no metadata/key network calls, fresh limiters."""
    monkeypatch.setattr(P, "USAGE_JSON", tmp_path / "usage.json")
    monkeypatch.setattr(P, "USAGE_JSONL", tmp_path / "usage.jsonl")
    monkeypatch.setattr(P, "MODELS_CACHE", tmp_path / "models.json")
    monkeypatch.setattr(P, "_LIMITERS", {})
    monkeypatch.setattr(P, "_supported_params_for", lambda name, chain: {})
    monkeypatch.setattr(config, "_env_loaded", True)
    for var in ("GAIA_PROVIDER", "GAIA_MODEL_ID", "GAIA_VISION_MODEL_ID", "GAIA_FALLBACKS", "GAIA_VISION_FALLBACKS",
                "GAIA_RPM", "GAIA_BASE_URL", "GAIA_ALLOW_PAID", "GAIA_TEMPERATURE", "GAIA_FLATTEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GAIA_DAILY_CAP", "5")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    return tmp_path


# =============================================================================== provider wiring


def test_default_provider_is_openrouter_when_its_key_is_present(monkeypatch):
    monkeypatch.setattr(config, "_env_loaded", True)
    monkeypatch.delenv("GAIA_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    assert config.provider_name() == "openrouter"
    assert P.resolve_preset()[0] == "openrouter"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert config.provider_name() == "gemini"
    monkeypatch.setenv("GAIA_PROVIDER", "groq")
    assert config.provider_name() == "groq"


def test_config_and_providers_share_one_preset_table():
    assert config.PROVIDERS is P.PRESETS


def test_get_model_delegates_to_resilient_model(offline_providers):
    m = config.get_model()
    assert isinstance(m, P.ResilientModel)
    assert m.chain == [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
    ]
    assert m.limiter is P.get_limiter("openrouter") and m.limiter.daily_cap == 5
    info = config.provider_info()
    assert info["provider"] == "openrouter" and info["allow_paid"] is False
    assert info["vision_model_id"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    assert "sk-test-not-real" not in json.dumps(info)


def test_non_free_openrouter_model_is_refused_through_config(offline_providers):
    assert os.environ.get("GAIA_ALLOW_PAID") is None  # the spend guard is on by default
    os.environ["GAIA_MODEL_ID"] = "nvidia/nemotron-3-ultra-550b-a55b"  # paid variant: no ":free"
    try:
        with pytest.raises(ValueError, match="paid"):
            config.get_model()
    finally:
        del os.environ["GAIA_MODEL_ID"]


def test_non_free_vision_model_is_refused_before_any_request(offline_providers, monkeypatch, tmp_path):
    monkeypatch.setenv("GAIA_VISION_MODEL_ID", "openai/gpt-5o")
    img = tmp_path / "x.png"
    img.write_bytes(b"not really a png")
    with pytest.raises(ValueError, match="paid"):
        config.vision_complete("what is this?", [str(img)])
    assert not (offline_providers / "usage.json").exists()  # nothing was counted, nothing was sent


def test_vision_complete_delegates_with_preset_vision_model(offline_providers, monkeypatch):
    seen = {}

    def fake(prompt, image_paths, preset=None, **kw):
        seen.update(prompt=prompt, paths=image_paths, preset=preset, **kw)
        return "a cat"

    monkeypatch.setattr(P, "vision_complete", fake)
    monkeypatch.setenv("GAIA_MODEL_ID", "nvidia/nemotron-3-super-120b-a12b:free")  # text-only swap
    assert config.vision_complete("q?", ["/x.png"], max_tokens=99) == "a cat"
    assert seen["preset"] == "openrouter" and seen["max_tokens"] == 99
    assert seen["model_id"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"


# =============================================================================== scripted model


class Scripted(Model):
    """Answers by role: picks a script from the system prompt of whichever agent is calling."""

    def __init__(self, scripts: dict[str, list[str]]):
        super().__init__(model_id="scripted")
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.calls: list[str] = []

    def _role(self, messages) -> str:
        first = messages[0]
        content = first.content if hasattr(first, "content") else first["content"]
        text = content if isinstance(content, str) else " ".join(c.get("text", "") for c in content)
        if "You check the FORMAT" in text:
            return "verifier"
        if "You are web_researcher" in text:
            return "web"
        if "You are media_analyst" in text:
            return "media"
        if "MANAGER" in text:
            return "manager"
        return "single"

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kw):
        role = self._role(messages)
        self.calls.append(role)
        queue = self.scripts.get(role) or ["<code>\nfinal_answer('fallback')\n</code>"]
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        return ChatMessage(role=MessageRole.ASSISTANT, content=reply,
                           token_usage=TokenUsage(input_tokens=10, output_tokens=2))


def _code(src: str) -> str:
    return f"Thought: go.\n<code>\n{src}\n</code>"


# =============================================================================== env contract


def test_disable_tools_removes_tools_and_tells_the_model(monkeypatch):
    monkeypatch.setenv("GAIA_DISABLE_TOOLS", "web_search, visit_webpage")
    a = agent_mod.build_agent(model=Scripted({}))
    assert set(a.tools) == (WEB_TOOLS | MEDIA_TOOLS | {"final_answer"}) - {"web_search", "visit_webpage"}
    assert "TOOLS DISABLED IN THIS CONFIGURATION" in a.system_prompt
    assert "visit_webpage, web_search" in a.system_prompt


def test_unknown_disabled_tool_warns(monkeypatch, capsys):
    monkeypatch.setenv("GAIA_DISABLE_TOOLS", "no_such_tool")
    a = agent_mod.build_agent(model=Scripted({}))
    assert set(a.tools) == WEB_TOOLS | MEDIA_TOOLS | {"final_answer"}
    assert "no_such_tool" in capsys.readouterr().err


def test_multi_arch_builds_manager_with_two_named_specialists(monkeypatch):
    monkeypatch.setenv("GAIA_ARCH", "multi")
    m = agent_mod.build_agent(model=Scripted({}))
    assert set(m.tools) == {"final_answer"}  # python is the CodeAgent itself
    assert set(m.managed_agents) == {"web_researcher", "media_analyst"}
    web, media = m.managed_agents["web_researcher"], m.managed_agents["media_analyst"]
    assert set(web.tools) == WEB_TOOLS | {"final_answer"}
    assert set(media.tools) == MEDIA_TOOLS | {"final_answer"}
    assert web.description and media.description and "Wikipedia" in web.description
    assert "web_researcher" in m.system_prompt and "media_analyst" in m.system_prompt
    assert web.model is m.model is media.model  # one model object -> one limiter and budget
    assert m.max_steps == agent_mod.MAX_STEPS and web.max_steps == agent_mod.SUB_AGENT_MAX_STEPS


def test_multi_arch_drops_a_specialist_whose_tools_are_all_disabled(monkeypatch):
    monkeypatch.setenv("GAIA_ARCH", "multi")
    monkeypatch.setenv("GAIA_DISABLE_TOOLS", ",".join(sorted(WEB_TOOLS)))
    m = agent_mod.build_agent(model=Scripted({}))
    assert set(m.managed_agents) == {"media_analyst"}
    assert "UNAVAILABLE IN THIS CONFIGURATION" in m.system_prompt


def test_code_steps_get_a_realistic_timeout(monkeypatch):
    """smolagents' default 30 s per code step discarded finished sub-agent runs (seen live)."""
    single = agent_mod.build_agent(model=Scripted({}))
    assert single.python_executor.timeout_seconds == agent_mod.CODE_TIMEOUT_S
    monkeypatch.setenv("GAIA_ARCH", "multi")
    m = agent_mod.build_agent(model=Scripted({}))
    assert m.python_executor.timeout_seconds == agent_mod.MANAGER_CODE_TIMEOUT_S
    assert m.managed_agents["web_researcher"].python_executor.timeout_seconds == agent_mod.CODE_TIMEOUT_S
    monkeypatch.setenv("GAIA_CODE_TIMEOUT", "0")
    monkeypatch.setenv("GAIA_ARCH", "single")
    assert agent_mod.build_agent(model=Scripted({})).python_executor.timeout_seconds is None


def test_bad_arch_is_rejected(monkeypatch):
    monkeypatch.setenv("GAIA_ARCH", "swarm")
    with pytest.raises(ValueError):
        agent_mod.build_agent(model=Scripted({}))


# =============================================================================== tracing


def _runs() -> Path:
    return config.runs_dir()


def test_solve_writes_a_full_trace_under_the_run_id(monkeypatch):
    monkeypatch.setenv("GAIA_RUN_ID", "wiring-single")
    model = Scripted({"single": [_code("x = 6 * 7\nprint(x)"), _code("final_answer(str(x))")]})
    agent = agent_mod.build_agent(model=model, tools=[])
    res = agent_mod.solve("What is six times seven?", task_id="t-single", agent=agent)
    assert res["answer"] == "42" and res["error"] is None and res["run_id"] == "wiring-single"
    path = Path(res["trace_path"])
    assert path == _runs() / "traces" / "wiring-single" / "t-single.json"
    tr = json.loads(path.read_text())
    assert tr["run_id"] == "wiring-single" and tr["arch"] == "single" and tr["final_answer"] == "42"
    assert tr["totals"]["steps"] == 2 and tr["totals"]["model_calls"] == 2
    assert tr["provider_info"]["model_id"] == "scripted" and tr["provider_info"]["verify"] is False
    assert res["model_calls"] == 2 and res["input_tokens"] == 20


def test_solve_without_task_id_gets_a_stable_adhoc_id():
    model = Scripted({"single": [_code("final_answer('ok')")]})
    r1 = agent_mod.solve("same question", agent=agent_mod.build_agent(model=model, tools=[]))
    r2 = agent_mod.solve("same question", agent=agent_mod.build_agent(model=model, tools=[]))
    assert r1["task_id"] == r2["task_id"] and r1["task_id"].startswith("adhoc-")


def test_solve_traces_a_failed_build(monkeypatch):
    def broken_model():
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    monkeypatch.setattr(agent_mod, "get_model", broken_model)
    res = agent_mod.solve("q?", task_id="t-broken")
    assert res["answer"] == "" and "OPENROUTER_API_KEY" in res["error"]
    tr = json.loads(Path(res["trace_path"]).read_text())
    assert tr["error_type"] == "RuntimeError" and tr["steps"][0]["type"] == "task"


def test_multi_solve_traces_sub_agent_runs(monkeypatch, tmp_path, scripted_get_model):
    monkeypatch.setenv("GAIA_ARCH", "multi")
    monkeypatch.setenv("GAIA_RUN_ID", "wiring-multi")
    data = tmp_path / "n.txt"
    data.write_text("value: 41\n")
    scripted_get_model({
        "manager": [
            _code(f"r = media_analyst(task='Read the attached file {data} and report the value.')\nprint(r)"),
            _code("final_answer('42')"),
        ],
        "media": [_code(f"t = read_file({str(data)!r})\nprint(t)"), _code("final_answer('the value is 41')")],
    })
    res = agent_mod.solve("What is the value plus one?", task_id="t-multi", file_path=str(data))
    assert res["error"] is None, res["error"]
    assert res["answer"] == "42" and res["arch"] == "multi"
    tr = json.loads(Path(res["trace_path"]).read_text())
    assert tr["arch"] == "multi"
    subs = [run for s in tr["steps"] for run in (s.get("sub_agents") or [])]
    assert [r["agent"] for r in subs] == ["media_analyst"] and not subs[0]["partial"]
    assert any(tc["name"] == "read_file" for st in subs[0]["steps"] for tc in st.get("tool_calls") or [])
    assert tr["totals"]["sub_agent_calls"] == 1
    assert tr["totals"]["by_agent"]["media_analyst"]["model_calls"] == 2
    assert tr["totals"]["tool_calls_by_name"].get("media_analyst") == 1


@pytest.fixture
def scripted_get_model(monkeypatch):
    holder = {}

    def make(scripts):
        holder["model"] = Scripted(scripts)
        monkeypatch.setattr(agent_mod, "get_model", lambda *a, **k: holder["model"])
        return holder["model"]

    return make


# =============================================================================== verifier


@pytest.mark.parametrize(
    "candidate, revised, ok",
    [
        ("42 km", "42", True),
        ("The Beatles", "Beatles", True),
        ("banana, apple, cherry", "apple, banana, cherry", True),
        ("three", "3", True),
        ("3.14159", "3.14", True),
        ("apples", "apple", True),
        ("42", "43", False),
        ("Paris", "Lyon", False),
        ("17", "17 meters", False),
        ("x", "", False),
    ],
)
def test_is_reformat(candidate, revised, ok):
    assert agent_mod.is_reformat(candidate, revised)[0] is ok


def test_verifier_reformats_and_is_recorded_in_the_trace(monkeypatch, scripted_get_model):
    monkeypatch.setenv("GAIA_VERIFY", "1")
    monkeypatch.setenv("GAIA_RUN_ID", "wiring-verify")
    model = scripted_get_model({
        "single": [_code("final_answer('42 km')")],
        "verifier": ["FINAL ANSWER: 42"],
    })
    res = agent_mod.solve("How far is it, in km?", task_id="t-verify")
    assert res["answer"] == "42" and res["verifier"]["accepted"] and res["verifier"]["changed"]
    assert model.calls == ["single", "verifier"]
    tr = json.loads(Path(res["trace_path"]).read_text())
    types = [s["type"] for s in tr["steps"]]
    assert types[-2:] == ["verifier", "final"]
    assert tr["verifier"]["candidate"] == "42 km" and tr["final_answer"] == "42"
    assert tr["totals"]["model_calls"] == 2 and tr["totals"]["by_agent"]["verifier"]["model_calls"] == 1


def test_verifier_may_not_change_the_substance(monkeypatch, scripted_get_model):
    monkeypatch.setenv("GAIA_VERIFY", "1")
    scripted_get_model({"single": [_code("final_answer('42')")], "verifier": ["FINAL ANSWER: 43"]})
    res = agent_mod.solve("What is six times seven?", task_id="t-verify-reject")
    assert res["answer"] == "42"
    assert res["verifier"]["accepted"] is False and "rejected" in res["verifier"]["reason"]


def test_verifier_failure_keeps_the_answer(monkeypatch, scripted_get_model):
    monkeypatch.setenv("GAIA_VERIFY", "1")
    model = scripted_get_model({"single": [_code("final_answer('42')")]})
    orig = model.generate

    def flaky(messages, **kw):
        if model._role(messages) == "verifier":
            raise RuntimeError("429 rate limit")
        return orig(messages, **kw)

    model.generate = flaky
    res = agent_mod.solve("q?", task_id="t-verify-fail")
    assert res["answer"] == "42" and "429" in res["verifier"]["error"]


def test_verifier_off_by_default(scripted_get_model):
    model = scripted_get_model({"single": [_code("final_answer('42 km')")]})
    res = agent_mod.solve("q?", task_id="t-noverify")
    assert res["answer"] == "42 km" and res["verifier"] is None and model.calls == ["single"]


# =============================================================================== ablation configs


def test_every_ablation_config_changes_what_gets_built(monkeypatch):
    from evals import ablate

    cfgs = ablate.load_configs()
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("GAIA_")}
    base_env.update({"GAIA_ARCH": "multi", "GAIA_VERIFY": "1", "GAIA_MODEL_ID": "caller-model"})

    def built(name):
        env = ablate.config_env(cfgs[name], base_env, run_id=f"abl-x-{name}")
        assert env["GAIA_RUN_ID"] == f"abl-x-{name}" and "GAIA_ALLOW_PAID" not in env
        for var in ("GAIA_ARCH", "GAIA_VERIFY", "GAIA_DISABLE_TOOLS", "GAIA_PROVIDER", "GAIA_MODEL_ID"):
            monkeypatch.delenv(var, raising=False)
        for k, v in env.items():
            if k.startswith("GAIA_"):
                monkeypatch.setenv(k, v)
        a = agent_mod.build_agent(model=Scripted({}))
        tools = set(a.tools) | {t for sub in a.managed_agents.values() for t in sub.tools}
        return a, tools, config.verify_enabled(), env

    base, base_tools, base_verify, base_env_out = built("baseline")
    assert not base.managed_agents and not base_verify  # the caller's multi/verify did not leak in
    assert base_env_out["GAIA_MODEL_ID"] == "caller-model"  # baseline keeps the caller's model
    _, t, _, _ = built("no_web")
    assert not (t & WEB_TOOLS) and MEDIA_TOOLS <= t
    _, t, _, _ = built("no_media")
    assert {"transcribe_audio", "image_question"}.isdisjoint(t) and "read_file" in t
    _, t, _, _ = built("no_wikipedia")
    assert "web_search" in t and "wikipedia_page" not in t
    multi, _, _, _ = built("arch_multi")
    assert set(multi.managed_agents) == {"web_researcher", "media_analyst"}
    _, _, verify, _ = built("verify")
    assert verify is True
    _, _, _, env = built("model_gemini_flash")
    assert env["GAIA_PROVIDER"] == "gemini" and "GAIA_MODEL_ID" not in env  # caller's model id dropped


# =============================================================================== runners


def test_run_main_rows_carry_run_id_and_analysis_runs(monkeypatch):
    from gaia_agent import run as runmod

    monkeypatch.setenv("GAIA_RUN_ID", "wiring-run")
    out = _runs() / "answers.jsonl"
    qs = [{"task_id": "aaa111", "question": "Q1", "file_name": ""}]

    def fake_solve(question, task_id=None, file_name=None):
        return {"answer": "", "error": "Timeout after 1s", "steps": None}

    runmod.main([], solve_fn=fake_solve, questions=qs, out_path=out)
    row = runmod.load_records(out)[-1]
    assert row["run_id"] == "wiring-run" and "trace_path" in row
    report = json.loads((_runs() / "error_analysis" / "wiring-run.json").read_text())
    assert report["run_id"] == "wiring-run" and "no_trace" in report["items"][0]["tags"]


def test_run_main_run_id_flag(monkeypatch):
    from gaia_agent import run as runmod

    out = _runs() / "answers-flag.jsonl"
    runmod.main(["--run-id", "wiring-flag", "--no-analysis"], questions=[{"task_id": "b1", "question": "Q"}],
                solve_fn=lambda q, task_id=None, file_name=None: {"answer": "x"}, out_path=out)
    assert runmod.load_records(out)[-1]["run_id"] == "wiring-flag"


def test_practice_run_id_defaults_to_env(monkeypatch, tmp_path):
    from evals import run_practice as rp

    seen = {}

    def fake_run(tasks, *, run_id, **kw):
        seen["run_id"] = run_id
        return {"run_id": run_id, "config": {}, "items": [], "summary": {"n": 0, "skipped": 0},
                "budget": {"stopped": None, "calls_this_invocation": 0}}

    monkeypatch.setattr(rp, "run", fake_run)
    monkeypatch.setattr(rp, "_error_analysis", lambda run_id, max_steps: None)
    monkeypatch.setenv("GAIA_RUN_ID", "smoke-x")
    assert rp.main(["--only", "p01", "--results-dir", str(tmp_path)]) == 0
    assert seen["run_id"] == "smoke-x"


# =============================================================================== HF Hub attachments


class _Resp:
    def __init__(self, status):
        self.status_code = status


class _HubError(Exception):
    def __init__(self, status):
        super().__init__(f"{status} Client Error")
        self.response = _Resp(status)


def test_hub_download_requests_only_the_attachment(tmp_path):
    src = tmp_path / "cache" / "sheet.xlsx"
    src.parent.mkdir()
    src.write_bytes(b"xlsx-bytes")
    calls = []

    def fake(repo_id, repo_type, filename):
        calls.append((repo_id, repo_type, filename))
        return str(src)

    dest = tmp_path / "files" / "tid_sheet.xlsx"
    out = files_mod.download_from_hub("sheet.xlsx", dest, hf_hub_download=fake)
    assert out == str(dest) and dest.read_bytes() == b"xlsx-bytes"
    assert calls == [("gaia-benchmark/GAIA", "dataset", "2023/validation/sheet.xlsx")]


@pytest.mark.parametrize("status", [401, 403])
def test_hub_download_denied_returns_none_with_message(tmp_path, status, capsys):
    def denied(**kw):
        raise _HubError(status)

    assert files_mod.download_from_hub("a.mp3", tmp_path / "a.mp3", hf_hub_download=denied) is None
    assert "access denied" in capsys.readouterr().err and "gated" in files_mod.last_message


@pytest.mark.parametrize("name", ["metadata.jsonl", "2023/validation/metadata.parquet", "answers.jsonl", ""])
def test_hub_download_refuses_non_attachments(tmp_path, name):
    def boom(**kw):
        raise AssertionError("must not be called")

    assert files_mod.download_from_hub(name, tmp_path / "x", hf_hub_download=boom) is None


def test_hub_path_is_off_by_default_and_used_only_after_a_course_404(tmp_path, monkeypatch):
    class R404:
        status_code = 404

    monkeypatch.setattr(files_mod.requests, "get", lambda *a, **k: R404())
    used = []
    monkeypatch.setattr(files_mod, "download_from_hub", lambda name, dest: used.append(name) or None)
    assert files_mod.download_task_file("tid", "a.png", files_dir=tmp_path) is None
    assert used == []
    monkeypatch.setenv("GAIA_ATTACHMENTS_FROM_HUB", "1")
    assert files_mod.download_task_file("tid", "a.png", files_dir=tmp_path) is None
    assert used == ["a.png"]
