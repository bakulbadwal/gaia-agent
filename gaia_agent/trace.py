"""Structured, per-question traces built from a smolagents agent's memory.

Why this exists: `agent.memory` is an in-process object graph (dataclasses holding
ChatMessages, PIL images, exceptions). It is lost when the process exits, it is not
JSON-serialisable as-is, and it does not answer the questions you ask after a run:
*which tools did the agent call, where did it burn tokens, which step failed and how*.
This module turns that memory into one plain JSON document per question, written to
``runs/traces/<run_id>/<task_id>.json``, and aggregates a run's traces into totals.

Trace schema (one file per question)::

    {task_id, question, run_id, model, provider, arch, started_at, seconds,
     final_answer, error, provider_info,
     steps: [{index, type: "task"|"planning"|"action"|"final", agent,
              model_output, code, tool_calls: [{name, arguments}], observation,
              error, error_type, input_tokens, output_tokens, duration_s,
              start_time, step_number, is_final_answer, images,
              sub_agents: [<sub-agent run>, ...]}],          # action steps only
     sub_agent_runs_unattached: [...],                         # rarely non-empty
     totals: {steps, planning_steps, input_tokens, output_tokens, total_tokens,
              model_calls, tool_calls, tool_calls_by_name, step_errors,
              step_error_types, sub_agent_calls, token_usage_complete, by_agent}}

Design decisions (documented because they are judgment calls):

* **Tool calls in a CodeAgent are recovered from the code.** smolagents records every
  CodeAgent action as a single ``ToolCall(name="python_interpreter", arguments=<code>)``,
  so counting ``tool_calls`` verbatim would report "python_interpreter: 1 per step" and
  nothing else. Instead the step's code is parsed with :mod:`ast` and every call to a
  name the agent actually exposes (its tools plus its managed agents) becomes one
  ``{name, arguments}`` entry. Literal arguments are decoded with ``ast.literal_eval``;
  anything computed at runtime is kept as source text (``"<expr> query"``). If the code
  does not parse, a regex fallback still finds ``tool_name(`` occurrences. For a
  ToolCallingAgent the recorded tool calls are used directly.
* **Sub-agents are nested, not flattened.** A managed agent runs *inside* one action
  step of its parent (the parent's code calls it like a function), so its run is
  attached to that step as ``steps[i].sub_agents[j] = {agent, task, final_answer,
  seconds, steps:[...], partial}``. Nesting keeps causality ("step 3 delegated to
  web_agent, which made 6 searches") that a flat list with an ``agent`` field loses.
  Every step still carries an ``agent`` field so a consumer can flatten trivially, and
  ``totals.by_agent`` gives the per-agent roll-up.
* **Capturing every sub-agent call needs a recorder.** ``ManagedAgent.__call__`` runs
  ``self.run(task)`` with ``reset=True``, so after the parent finishes, a managed
  agent's memory only holds its *last* invocation. :class:`SubAgentRecorder` hooks the
  managed agents' step callbacks and keeps every invocation. Without a recorder, the
  last invocation is still traced and flagged ``partial: true``.
* **Sub-agent runs are matched to parent steps by time window** (the invocation starts
  inside the parent step's ``[start_time, end_time]``), because smolagents records no
  explicit parent link.
* **Counting conventions.** ``totals.steps`` counts the top-level agent's action steps,
  the same number ``runs/answers.jsonl`` stores. Each ActionStep and PlanningStep is one
  model call (smolagents 1.26 issues one ``generate`` per step, including the forced
  final answer after max steps). Tokens are summed across all agents; a step with no
  ``token_usage`` contributes 0 and flips ``token_usage_complete`` to false, so a
  provider that does not report usage is visible instead of silently under-counted.
* **The FinalAnswerStep is synthesised.** smolagents yields it but never appends it to
  ``memory.steps``, so the trace ends with a ``type: "final"`` step built from the
  ``final_answer``/``error`` the runner passes in.
* **Long text is truncated to 4000 chars** with a ``…[truncated N chars]`` marker, and
  strings that look like API keys are redacted before anything touches disk.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep, TaskStep

from .config import RUNS_DIR

__all__ = [
    "MAX_TEXT",
    "SubAgentRecorder",
    "clip",
    "format_trace",
    "load_traces",
    "summarize_traces",
    "trace_from_agent",
    "trace_path",
    "tools_called_in_code",
    "write_trace",
]

logger = logging.getLogger(__name__)

MAX_TEXT = 4000
SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- text


_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API key
    re.compile(r"gsk_[0-9A-Za-z]{20,}"),  # Groq
    re.compile(r"\bsk-[0-9A-Za-z_\-]{20,}"),  # OpenAI-style
    re.compile(r"\bhf_[0-9A-Za-z]{30,}"),  # Hugging Face token
    re.compile(r"\bxai-[0-9A-Za-z]{20,}"),  # xAI
]


def _redact(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def _to_text(value: Any) -> str | None:
    """Render any memory field as text: str stays str, structures become JSON."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def clip(value: Any, limit: int = MAX_TEXT) -> str | None:
    """Text form of ``value``, secrets redacted, cut to ``limit`` chars with a marker.

    The marker reports how many characters were dropped, so a reader can tell a short
    observation from a truncated one: ``"abc…[truncated 1234 chars]"``.
    """
    text = _to_text(value)
    if text is None:
        return None
    text = _redact(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…[truncated {len(text) - limit} chars]"


def _jsonable(value: Any, limit: int = MAX_TEXT) -> Any:
    """Make tool-call arguments JSON-safe, clipping long strings inside them."""
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, str):
        return clip(value, limit)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, limit) for v in value]
    return clip(repr(value), limit)


# --------------------------------------------------------------- tool extraction


def tools_called_in_code(code: str | None, tool_names: Iterable[str]) -> list[dict]:
    """Every call to a known tool in a CodeAgent action, in source order.

    Returns ``[{"name": str, "arguments": {...}}]``. Positional arguments land under
    ``"args"``; keyword arguments keep their names. A literal argument is decoded
    (``"query": "moon landing"``); a computed one is kept as source text prefixed
    with ``<expr>`` (``"query": "<expr> f'{name} albums'"``) so it is never mistaken
    for the runtime value.
    """
    names = set(tool_names)
    if not code or not names:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Unparseable code still ran nothing useful, but record intent if visible.
        pattern = re.compile(r"(?<![\w.])(" + "|".join(map(re.escape, sorted(names))) + r")\s*\(")
        return [{"name": m.group(1), "arguments": {"unparsed": True}} for m in pattern.finditer(code)]

    def arg_value(node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            try:
                return "<expr> " + ast.unparse(node)
            except Exception:  # pragma: no cover - ast.unparse is total on parsed trees
                return "<expr>"

    calls: list[tuple[int, int, dict]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
            arguments: dict[str, Any] = {}
            positional = [arg_value(a) for a in node.args if not isinstance(a, ast.Starred)]
            if positional:
                arguments["args"] = positional
            for kw in node.keywords:
                arguments[kw.arg if kw.arg is not None else "**"] = arg_value(kw.value)
            calls.append((node.lineno, node.col_offset, {"name": node.func.id, "arguments": _jsonable(arguments)}))
    calls.sort(key=lambda c: (c[0], c[1]))
    return [c[2] for c in calls]


def _callable_names(agent: Any) -> list[str]:
    names: list[str] = []
    for attr in ("tools", "managed_agents"):
        mapping = getattr(agent, attr, None) or {}
        if isinstance(mapping, dict):
            names.extend(str(k) for k in mapping)
    return names


def _step_tool_calls(step: ActionStep, tool_names: list[str]) -> list[dict]:
    raw = step.tool_calls or []
    out: list[dict] = []
    for tc in raw:
        name = getattr(tc, "name", None)
        if name == "python_interpreter":
            code = step.code_action if step.code_action is not None else _to_text(getattr(tc, "arguments", None))
            out.extend(tools_called_in_code(code, tool_names))
        else:
            out.append({"name": str(name), "arguments": _jsonable(getattr(tc, "arguments", None))})
    if not raw and step.code_action:
        out.extend(tools_called_in_code(step.code_action, tool_names))
    return out


# ------------------------------------------------------------------ sub-agents


class SubAgentRecorder:
    """Keeps every invocation of every managed (sub-)agent for the trace.

    smolagents resets a managed agent's memory on each call, so only the last call
    survives in ``agent.memory``. Attach a recorder before ``agent.run`` and pass it to
    :func:`trace_from_agent`::

        recorder = SubAgentRecorder.attach(agent)   # no-op for a single agent
        agent.run(task)
        trace = trace_from_agent(agent, ..., recorder=recorder)
        recorder.reset()                           # before reusing the agent

    The recorder registers ActionStep / PlanningStep / FinalAnswerStep callbacks on
    every managed agent, recursively. A new invocation starts when the agent's task
    text changes, when a FinalAnswerStep closed the previous one, or when step
    numbering restarts at 1 (an invocation that raised leaves no FinalAnswerStep).
    """

    def __init__(self) -> None:
        self.invocations: list[dict] = []
        self._current: dict[int, dict] = {}
        self._attached: set[int] = set()
        self._parents: dict[int, int] = {}
        self._names: dict[int, str] = {}

    @classmethod
    def attach(cls, agent: Any) -> "SubAgentRecorder":
        recorder = cls()
        recorder._walk(agent, parent=None)
        return recorder

    def reset(self) -> None:
        self.invocations.clear()
        self._current.clear()

    @property
    def attached_agents(self) -> list[str]:
        return [self._names[k] for k in self._attached]

    def _walk(self, agent: Any, parent: Any | None) -> None:
        if parent is not None and id(agent) not in self._attached:
            self._attached.add(id(agent))
            self._parents[id(agent)] = id(parent)
            self._names[id(agent)] = getattr(agent, "name", None) or type(agent).__name__
            registry = getattr(agent, "step_callbacks", None)
            if registry is not None and hasattr(registry, "register"):
                cb = self._make_callback(agent)
                for cls_ in (ActionStep, PlanningStep, FinalAnswerStep):
                    registry.register(cls_, cb)
        for child in (getattr(agent, "managed_agents", None) or {}).values():
            self._walk(child, parent=agent)

    def _make_callback(self, sub_agent: Any):
        key = id(sub_agent)

        def _record(memory_step, agent=None):  # two params -> smolagents passes agent=
            self._on_step(key, memory_step, agent if agent is not None else sub_agent)

        return _record

    def _on_step(self, key: int, step: Any, agent: Any) -> None:
        task = getattr(agent, "task", None)
        cur = self._current.get(key)
        restart = (
            isinstance(step, ActionStep)
            and step.step_number == 1
            and cur is not None
            and any(isinstance(s, ActionStep) for s in cur["steps"])
        )
        if cur is None or cur["closed"] or restart or (task is not None and task != cur["task"]):
            timing = getattr(step, "timing", None)
            cur = {
                "agent_key": key,
                "parent_key": self._parents.get(key),
                "agent": self._names.get(key, "sub_agent"),
                "task": task,
                "steps": [],
                "final_answer": None,
                "start_time": getattr(timing, "start_time", None) or time.time(),
                "end_time": None,
                "closed": False,
            }
            self._current[key] = cur
            self.invocations.append(cur)
        if isinstance(step, FinalAnswerStep):
            cur["final_answer"] = step.output
            cur["closed"] = True
            cur["end_time"] = time.time()
        else:
            cur["steps"].append(step)
            timing = getattr(step, "timing", None)
            if timing is not None and timing.end_time is not None:
                cur["end_time"] = timing.end_time


# ------------------------------------------------------------- step conversion


def _tokens(step: Any) -> tuple[int | None, int | None]:
    usage = getattr(step, "token_usage", None)
    if usage is None:
        return None, None
    return int(usage.input_tokens), int(usage.output_tokens)


def _duration(step: Any) -> float | None:
    timing = getattr(step, "timing", None)
    if timing is None or timing.duration is None:
        return None
    return round(float(timing.duration), 3)


def _start(step: Any) -> float | None:
    timing = getattr(step, "timing", None)
    return getattr(timing, "start_time", None)


def _error_fields(err: Any) -> tuple[str | None, str | None]:
    if err is None:
        return None, None
    if isinstance(err, BaseException):
        return clip(getattr(err, "message", None) or str(err)), type(err).__name__
    if isinstance(err, dict):
        return clip(err.get("message") or err), err.get("type")
    return clip(err), None


def _blank(index: int, type_: str, agent_name: str) -> dict:
    return {
        "index": index,
        "type": type_,
        "agent": agent_name,
        "model_output": None,
        "code": None,
        "tool_calls": [],
        "observation": None,
        "error": None,
        "error_type": None,
        "input_tokens": None,
        "output_tokens": None,
        "duration_s": None,
    }


def _convert_steps(
    steps: list[Any],
    *,
    agent_name: str,
    tool_names: list[str],
    sub_runs: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Convert memory steps; attach ``sub_runs`` (already converted) to action steps.

    Returns ``(steps, unattached_sub_runs)``.
    """
    out: list[dict] = []
    action_windows: list[tuple[float, float, dict]] = []
    for step in steps:
        idx = len(out)
        if isinstance(step, TaskStep):
            d = _blank(idx, "task", agent_name)
            d["model_output"] = clip(step.task)
            d["images"] = len(step.task_images or [])
        elif isinstance(step, PlanningStep):
            d = _blank(idx, "planning", agent_name)
            d["model_output"] = clip(step.plan)
            d["input_tokens"], d["output_tokens"] = _tokens(step)
            d["duration_s"] = _duration(step)
            d["start_time"] = _start(step)
        elif isinstance(step, ActionStep):
            d = _blank(idx, "action", agent_name)
            d["step_number"] = step.step_number
            d["model_output"] = clip(step.model_output)
            d["code"] = clip(step.code_action)
            d["tool_calls"] = _step_tool_calls(step, tool_names)
            d["observation"] = clip(step.observations)
            d["error"], d["error_type"] = _error_fields(step.error)
            d["input_tokens"], d["output_tokens"] = _tokens(step)
            d["duration_s"] = _duration(step)
            d["start_time"] = _start(step)
            d["is_final_answer"] = bool(step.is_final_answer)
            d["images"] = len(step.observations_images or [])
            if step.is_final_answer or step.action_output is not None:
                d["action_output"] = clip(step.action_output)
            d["sub_agents"] = []
            timing = step.timing
            if timing is not None and timing.start_time is not None:
                end = timing.end_time if timing.end_time is not None else math.inf
                action_windows.append((timing.start_time, end, d))
        elif isinstance(step, FinalAnswerStep):
            d = _blank(idx, "final", agent_name)
            d["model_output"] = clip(step.output)
        else:  # unknown step type from a future smolagents: keep it visible
            d = _blank(idx, type(step).__name__, agent_name)
            d["model_output"] = clip(getattr(step, "__dict__", step))
        out.append(d)

    unattached: list[dict] = []
    for run in sorted(sub_runs, key=lambda r: r.get("_start") or 0.0):
        start = run.get("_start")
        target = None
        if start is not None:
            for lo, hi, d in action_windows:
                if lo <= start <= hi:
                    target = d
                    break
            if target is None:  # fall back to the last action step that began before it
                earlier = [d for lo, _hi, d in action_windows if lo <= start]
                target = earlier[-1] if earlier else None
        if target is None:
            unattached.append(run)
        else:
            target["sub_agents"].append(run)
    return out, unattached


def _convert_invocation(inv: dict, recorder: SubAgentRecorder | None, agent_obj: Any | None) -> dict:
    children = []
    if recorder is not None:
        children = [
            _convert_invocation(child, recorder, None)
            for child in recorder.invocations
            if child["parent_key"] == inv["agent_key"]
            and child["start_time"] is not None
            and inv["start_time"] is not None
            and child["start_time"] >= inv["start_time"]
            and (inv["end_time"] is None or child["start_time"] <= inv["end_time"])
        ]
    tool_names = inv.get("tool_names") or []
    steps, unattached = _convert_steps(inv["steps"], agent_name=inv["agent"], tool_names=tool_names, sub_runs=children)
    start, end = inv.get("start_time"), inv.get("end_time")
    run = {
        "agent": inv["agent"],
        "task": clip(inv.get("task")),
        "final_answer": clip(inv.get("final_answer")),
        "seconds": round(end - start, 3) if (start is not None and end is not None) else None,
        "partial": bool(inv.get("partial", False)),
        "steps": steps,
        "_start": start,
    }
    if unattached:
        run["sub_agent_runs_unattached"] = unattached
    return run


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def _agents_by_key(agent: Any) -> dict[int, Any]:
    found: dict[int, Any] = {}

    def walk(a: Any) -> None:
        for child in (getattr(a, "managed_agents", None) or {}).values():
            if id(child) not in found:
                found[id(child)] = child
                walk(child)

    walk(agent)
    return found


def _fallback_invocations(agent: Any) -> list[dict]:
    """Last invocation of each managed agent, read from its (reset-on-call) memory."""
    invs: list[dict] = []
    for key, sub in _agents_by_key(agent).items():
        steps = list(getattr(getattr(sub, "memory", None), "steps", None) or [])
        if not steps:
            continue
        task = next((s.task for s in steps if isinstance(s, TaskStep)), getattr(sub, "task", None))
        starts = [_start(s) for s in steps if _start(s) is not None]
        ends = [s.timing.end_time for s in steps if getattr(getattr(s, "timing", None), "end_time", None)]
        invs.append(
            {
                "agent_key": key,
                "parent_key": None,
                "agent": getattr(sub, "name", None) or type(sub).__name__,
                "task": task,
                "steps": [s for s in steps if not isinstance(s, TaskStep)],
                "final_answer": None,
                "start_time": min(starts) if starts else None,
                "end_time": max(ends) if ends else None,
                "closed": True,
                "partial": True,
                "tool_names": _callable_names(sub),
            }
        )
    return invs


# --------------------------------------------------------------------- totals


def _walk_steps(steps: list[dict]) -> Iterable[dict]:
    for s in steps:
        yield s
        for run in s.get("sub_agents") or []:
            yield from _walk_steps(run.get("steps") or [])


def _totals(steps: list[dict], unattached: list[dict], top_agent: str) -> dict:
    all_steps = list(_walk_steps(steps))
    for run in unattached:
        all_steps.extend(_walk_steps(run.get("steps") or []))
    tool_counter: Counter = Counter()
    err_counter: Counter = Counter()
    by_agent: dict[str, dict] = {}
    totals = {
        "steps": sum(1 for s in steps if s["type"] == "action"),
        "planning_steps": sum(1 for s in steps if s["type"] == "planning"),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model_calls": 0,
        "tool_calls": 0,
        "tool_calls_by_name": {},
        "step_errors": 0,
        "step_error_types": {},
        "sub_agent_calls": sum(len(s.get("sub_agents") or []) for s in all_steps) + len(unattached),
        "token_usage_complete": True,
        "by_agent": by_agent,
    }
    for s in all_steps:
        agent = s.get("agent") or top_agent
        a = by_agent.setdefault(
            agent, {"steps": 0, "model_calls": 0, "input_tokens": 0, "output_tokens": 0, "tool_calls": 0}
        )
        if s["type"] in ("action", "planning"):
            totals["model_calls"] += 1
            a["model_calls"] += 1
            if s["type"] == "action":
                a["steps"] += 1
            if s.get("input_tokens") is None:
                totals["token_usage_complete"] = False
            totals["input_tokens"] += s.get("input_tokens") or 0
            totals["output_tokens"] += s.get("output_tokens") or 0
            a["input_tokens"] += s.get("input_tokens") or 0
            a["output_tokens"] += s.get("output_tokens") or 0
        for tc in s.get("tool_calls") or []:
            tool_counter[tc["name"]] += 1
            a["tool_calls"] += 1
        if s.get("error"):
            totals["step_errors"] += 1
            err_counter[s.get("error_type") or "unknown"] += 1
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    totals["tool_calls"] = sum(tool_counter.values())
    totals["tool_calls_by_name"] = dict(tool_counter.most_common())
    totals["step_error_types"] = dict(err_counter.most_common())
    return totals


# ----------------------------------------------------------------- public API


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


_SECRET_KEY_RE = re.compile(r"(api[_-]?key|secret|token|password|authorization)", re.I)


def _safe_provider_info(info: dict | None) -> dict | None:
    if not info:
        return None
    safe = {}
    for k, v in info.items():
        if _SECRET_KEY_RE.search(str(k)):
            continue  # never persist anything shaped like a credential
        safe[str(k)] = _jsonable(v, 500)
    return safe


def trace_from_agent(
    agent: Any,
    *,
    task_id: str,
    question: str,
    run_id: str,
    final_answer: Any,
    error: Any,
    seconds: float | None,
    arch: str = "single",
    provider_info: dict | None = None,
    recorder: SubAgentRecorder | None = None,
) -> dict:
    """Build the trace document for one question from ``agent.memory.steps``.

    ``agent`` may be ``None`` (e.g. it failed to build); the trace then holds only the
    task and final steps so a failed question still leaves a record.
    """
    memory_steps = list(getattr(getattr(agent, "memory", None), "steps", None) or [])
    top_name = getattr(agent, "name", None) or ("manager" if getattr(agent, "managed_agents", None) else "agent")

    # Sub-agent invocations: full history from the recorder, else last-call fallback.
    if recorder is not None and recorder.invocations:
        by_key = _agents_by_key(agent)
        for inv in recorder.invocations:
            inv.setdefault("tool_names", _callable_names(by_key.get(inv["agent_key"])))
        top_level = [inv for inv in recorder.invocations if inv["parent_key"] == id(agent)]
        sub_runs = [_convert_invocation(inv, recorder, None) for inv in top_level]
    else:
        sub_runs = [_convert_invocation(inv, None, None) for inv in _fallback_invocations(agent)]

    steps, unattached = _convert_steps(
        memory_steps, agent_name=top_name, tool_names=_callable_names(agent), sub_runs=sub_runs
    )
    if not any(s["type"] == "task" for s in steps):
        task_step = _blank(0, "task", top_name)
        task_step["model_output"] = clip(question)
        steps.insert(0, task_step)
    final = _blank(len(steps), "final", top_name)
    final["model_output"] = clip(final_answer)
    final["error"], final["error_type"] = _error_fields(error)
    steps.append(final)
    for i, s in enumerate(steps):
        s["index"] = i

    starts = [s["start_time"] for s in _walk_steps(steps) if s.get("start_time") is not None]
    if starts:
        started = min(starts)
    else:
        started = time.time() - (seconds or 0.0)

    model_obj = getattr(agent, "model", None)
    info = provider_info or {}
    error_text, error_type = _error_fields(error)
    trace = {
        "schema_version": SCHEMA_VERSION,
        "task_id": str(task_id),
        "question": clip(question),
        "run_id": str(run_id),
        "model": info.get("model_id") or info.get("model") or getattr(model_obj, "model_id", None),
        "provider": info.get("provider"),
        "arch": arch,
        "started_at": _iso(started),
        "seconds": round(float(seconds), 3) if seconds is not None else None,
        "final_answer": clip(final_answer),
        "error": error_text,
        "error_type": error_type,
        "provider_info": _safe_provider_info(provider_info),
        "steps": _strip_private(steps),
        "sub_agent_runs_unattached": _strip_private(unattached),
    }
    trace["totals"] = _totals(trace["steps"], trace["sub_agent_runs_unattached"], top_name)
    return trace


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(value: str) -> str:
    cleaned = _UNSAFE.sub("_", str(value)).strip("._")
    if not cleaned:
        raise ValueError(f"unusable path component: {value!r}")
    return cleaned[:200]


def trace_path(run_id: str, task_id: str, root: str | os.PathLike = RUNS_DIR) -> Path:
    """``<root>/traces/<run_id>/<task_id>.json`` with path-traversal-safe components."""
    return Path(root) / "traces" / _safe_component(run_id) / f"{_safe_component(task_id)}.json"


def write_trace(trace: dict, root: str | os.PathLike = RUNS_DIR) -> Path:
    """Write one trace atomically (temp file + rename): a crash never leaves half a JSON."""
    path = trace_path(trace["run_id"], trace["task_id"], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(trace, fh, ensure_ascii=False, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_traces(run_id: str, root: str | os.PathLike = RUNS_DIR) -> list[dict]:
    """All traces of a run, sorted by start time. Corrupt files are skipped with a warning."""
    folder = Path(root) / "traces" / _safe_component(run_id)
    if not folder.is_dir():
        return []
    traces = []
    for p in sorted(folder.glob("*.json")):
        if p.name.startswith(".tmp-"):
            continue
        try:
            traces.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skipping unreadable trace %s: %s", p, exc)
    traces.sort(key=lambda t: (t.get("started_at") or "", t.get("task_id") or ""))
    return traces


def _percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default), q in [0, 100]."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def summarize_traces(traces: list[dict]) -> dict:
    """Run-level totals: tokens, model calls, tool calls by name, steps, latency."""
    tool_counter: Counter = Counter()
    err_types: Counter = Counter()
    step_err_types: Counter = Counter()
    by_agent: dict[str, dict] = {}
    seconds = [float(t["seconds"]) for t in traces if t.get("seconds") is not None]
    steps = [int((t.get("totals") or {}).get("steps") or 0) for t in traces]
    out = {
        "n": len(traces),
        "errors": 0,
        "error_types": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model_calls": 0,
        "tool_calls": 0,
        "tool_calls_by_name": {},
        "sub_agent_calls": 0,
        "step_errors": 0,
        "step_error_types": {},
        "token_usage_complete": True,
        "total_steps": sum(steps),
        "mean_steps": round(sum(steps) / len(steps), 3) if steps else None,
        "max_steps": max(steps) if steps else None,
        "mean_seconds": round(sum(seconds) / len(seconds), 3) if seconds else None,
        "p50_seconds": _round(_percentile(seconds, 50)),
        "p90_seconds": _round(_percentile(seconds, 90)),
        "max_seconds": _round(max(seconds)) if seconds else None,
        "total_seconds": _round(sum(seconds)) if seconds else None,
        "mean_tokens": None,
        "by_agent": by_agent,
    }
    for t in traces:
        tot = t.get("totals") or {}
        if t.get("error"):
            out["errors"] += 1
            err_types[t.get("error_type") or "unknown"] += 1
        for key in ("input_tokens", "output_tokens", "model_calls", "sub_agent_calls", "step_errors"):
            out[key] += int(tot.get(key) or 0)
        if tot.get("token_usage_complete") is False:
            out["token_usage_complete"] = False
        tool_counter.update(tot.get("tool_calls_by_name") or {})
        step_err_types.update(tot.get("step_error_types") or {})
        for name, a in (tot.get("by_agent") or {}).items():
            agg = by_agent.setdefault(name, {})
            for k, v in a.items():
                agg[k] = agg.get(k, 0) + (v or 0)
    out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    out["mean_tokens"] = round(out["total_tokens"] / len(traces), 1) if traces else None
    out["tool_calls"] = sum(tool_counter.values())
    out["tool_calls_by_name"] = dict(tool_counter.most_common())
    out["error_types"] = dict(err_types.most_common())
    out["step_error_types"] = dict(step_err_types.most_common())
    return out


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


def format_trace(trace: dict, width: int = 160) -> str:
    """Compact human-readable timeline of one trace, for terminals and READMEs."""

    def one_line(s: Any) -> str:
        text = " ".join(str(s or "").split())
        return text if len(text) <= width else text[: width - 1] + "…"

    lines = [
        f"{trace.get('task_id')}  model={trace.get('model')}  arch={trace.get('arch')}  "
        f"{trace.get('seconds')}s  answer={trace.get('final_answer')!r}"
        + (f"  ERROR={trace.get('error_type')}" if trace.get("error") else ""),
    ]

    def render(steps: list[dict], depth: int) -> None:
        pad = "  " * depth
        for s in steps:
            tok = ""
            if s.get("input_tokens") is not None:
                tok = f" [{s['input_tokens']}→{s['output_tokens']} tok]"
            dur = f" {s['duration_s']}s" if s.get("duration_s") is not None else ""
            head = f"{pad}#{s['index']} {s['type']}<{s.get('agent')}>{dur}{tok}"
            tools = ", ".join(tc["name"] for tc in s.get("tool_calls") or [])
            if tools:
                head += f" tools: {tools}"
            lines.append(head)
            if s["type"] in ("task", "planning", "final") and s.get("model_output"):
                lines.append(f"{pad}   {one_line(s['model_output'])}")
            if s.get("observation"):
                lines.append(f"{pad}   obs: {one_line(s['observation'])}")
            if s.get("error"):
                lines.append(f"{pad}   ERR {s.get('error_type')}: {one_line(s['error'])}")
            for run in s.get("sub_agents") or []:
                lines.append(f"{pad}   ↳ {run['agent']}{' (partial)' if run.get('partial') else ''}: {one_line(run.get('task'))}")
                render(run.get("steps") or [], depth + 2)

    render(trace.get("steps") or [], 0)
    tot = trace.get("totals") or {}
    lines.append(
        f"totals: steps={tot.get('steps')} model_calls={tot.get('model_calls')} "
        f"tokens={tot.get('total_tokens')} tools={tot.get('tool_calls_by_name')}"
    )
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect GAIA run traces.")
    ap.add_argument("run_id")
    ap.add_argument("--task", help="print one task's timeline instead of the run summary")
    ap.add_argument("--root", default=str(RUNS_DIR))
    args = ap.parse_args(argv)
    traces = load_traces(args.run_id, args.root)
    if not traces:
        print(f"no traces for run {args.run_id!r} under {args.root}")
        return 1
    if args.task:
        match = [t for t in traces if str(t.get("task_id", "")).startswith(args.task)]
        if not match:
            print(f"no trace for task {args.task!r}")
            return 1
        print(format_trace(match[0]))
    else:
        print(json.dumps(summarize_traces(traces), indent=2))
    return 0


if __name__ == "__main__":  # python -m gaia_agent.trace <run_id> [--task <id-prefix>]
    raise SystemExit(_main())
