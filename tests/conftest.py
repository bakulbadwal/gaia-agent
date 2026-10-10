"""Shared test isolation: tests never write into the real runs/ folder.

solve() writes a trace for every question and the runners run error analysis at the end, both
under config.runs_dir() (GAIA_RUNS_DIR). Point that at a per-session temp dir, and make sure a
GAIA_* switch exported in the developer's shell (arch, verify, disabled tools, run id) cannot
change what an offline test builds.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_runs_dir(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("GAIA_RUNS_DIR", str(tmp_path_factory.getbasetemp() / "runs"))
    for var in ("GAIA_ARCH", "GAIA_VERIFY", "GAIA_DISABLE_TOOLS", "GAIA_ATTACHMENTS_FROM_HUB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GAIA_RUN_ID", "pytest")
