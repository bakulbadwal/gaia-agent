"""Integration checks: the package wires together, and the answer-leak guard covers agent code.

Offline except where marked. No model is called.
"""

from __future__ import annotations

import pytest

from gaia_agent import agent as agent_mod
from gaia_agent.tools_media import TOOLS_MEDIA
from gaia_agent.tools_web import TOOLS_WEB

EXPECTED_TOOLS = {
    "web_search", "visit_webpage", "wikipedia_search", "wikipedia_page", "wikipedia_revision",
    "wikipedia_page_info", "read_file", "run_python_file", "transcribe_audio", "image_question",
    "youtube_transcript", "youtube_audio_transcript", "youtube_frames_question",
}


def test_tool_lists_are_disjoint_and_complete():
    names = [t.name for t in TOOLS_WEB] + [t.name for t in TOOLS_MEDIA]
    assert len(names) == len(set(names))
    assert set(names) == EXPECTED_TOOLS


def test_instructions_only_name_real_tools():
    for name in ("read_file", "transcribe_audio", "image_question", "run_python_file",
                 "youtube_transcript", "youtube_audio_transcript", "youtube_frames_question"):
        assert name in agent_mod.INSTRUCTIONS
    assert "requests" not in agent_mod.AUTHORIZED_IMPORTS
    assert "urllib" not in agent_mod.AUTHORIZED_IMPORTS


def test_build_agent_with_lmstudio_preset(monkeypatch):
    import gaia_agent.config as config

    monkeypatch.setenv("GAIA_PROVIDER", "lmstudio")
    monkeypatch.setattr(config, "_first_local_model", lambda base_url: "local-test-model")
    a = agent_mod.build_agent()
    assert type(a).__name__ == "CodeAgent"
    assert set(a.tools) == EXPECTED_TOOLS | {"final_answer"}
    assert a.max_steps == agent_mod.MAX_STEPS
    assert a.model.model_id == "local-test-model"
    assert "EXACT MATCH" in a.system_prompt
    for name in EXPECTED_TOOLS:
        assert name in a.system_prompt


def _run_in_sandbox(code: str):
    from smolagents.local_python_executor import LocalPythonExecutor

    agent_mod.install_code_guard()
    ex = LocalPythonExecutor(additional_authorized_imports=agent_mod.AUTHORIZED_IMPORTS)
    ex.send_tools({})
    return ex(code)


@pytest.mark.parametrize(
    "code",
    [
        'import pandas as pd\npd.read_csv("https://huggingface.co/datasets/someone/x/resolve/main/a.csv")',
        'import pandas as pd\npd.read_csv("https://example.com/GAIA/answers.csv")',
        'import pandas as pd\npd.read_csv("hf://datasets/someone/x/a.csv")',
        'import pandas as pd\npd.read_html("https://someone-unit4.hf.space/")',
        'import numpy as np\nnp.genfromtxt("https://x.hf.space/answers.csv")',
    ],
)
def test_code_guard_blocks_leak_urls_from_agent_code(code):
    with pytest.raises(Exception) as exc:
        _run_in_sandbox(code)
    assert "BLOCKED by the answer-leak guard" in str(exc.value)


def test_code_guard_leaves_local_paths_alone(tmp_path):
    # The project directory itself is called gaia-agent; local files must still open.
    d = tmp_path / "gaia-agent"
    d.mkdir()
    f = d / "t.csv"
    f.write_text("a,b\n1,2\n")
    out = _run_in_sandbox(f'import pandas as pd\nprint(pd.read_csv("{f}").shape)')
    assert "(1, 2)" in out.logs


def test_code_guard_is_idempotent():
    import urllib.request

    agent_mod.install_code_guard()
    first = urllib.request.urlopen
    agent_mod.install_code_guard()
    assert urllib.request.urlopen is first


def test_requests_import_is_refused_in_sandbox():
    with pytest.raises(Exception) as exc:
        _run_in_sandbox("import requests")
    assert "not allowed" in str(exc.value)


@pytest.mark.network
def test_code_guard_allows_ordinary_urls():
    out = _run_in_sandbox(
        'import pandas as pd\n'
        'print(pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv").shape)'
    )
    assert "(150, 5)" in out.logs
