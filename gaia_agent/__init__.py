"""GAIA Level-1 agent for Unit 4 of the Hugging Face Agents Course.

Public entry points (imported lazily so `import gaia_agent.config` stays cheap):
    solve(question, task_id=None, file_name=None) -> dict
    build_agent() -> smolagents.CodeAgent
    normalize_answer(s) -> str
"""

__all__ = ["solve", "build_agent", "normalize_answer"]


def __getattr__(name):
    if name in __all__:
        from . import agent

        return getattr(agent, name)
    raise AttributeError(f"module 'gaia_agent' has no attribute {name!r}")
