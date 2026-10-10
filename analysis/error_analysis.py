"""Heuristic failure-mode tagging for a run's traces (no gold answers needed).

Reads ``runs/traces/<run_id>/*.json`` (schema: gaia_agent/trace.py) plus the run's rows in
``runs/answers.jsonl`` (for questions that timed out before a trace was written), tags each
task with observable failure modes, writes ``runs/error_analysis.json`` and prints a table.

The tags describe *how the agent behaved*, not whether the answer was right: this runs on
the 20 course questions, whose answers we never see. It is the open-coding pass of an error
analysis; the counts tell you which tool or prompt fix to try first.

Tags
----
hit_max_steps              ran out of steps (AgentMaxStepsError / "Reached max steps")
tool_error                 a tool returned or raised an error; also ``tool_error:<tool>``
code_execution_error       the agent's own Python failed (not inside a tool call)
parse_error                model output had no parsable code block
rate_limited               the *model* call hit a 429 / quota (tool-side 429s are tool errors)
timeout                    the question hit the runner's wall-clock limit, or a model timeout
empty_or_refusal_answer    no answer, or a refusal / "unknown" style answer
answer_format_suspicious   answer has units, a sentence, "FINAL ANSWER", quotes, or is long
blocked_by_guard           the answer-leak guard refused a query or URL
no_tool_used               finished without calling a single tool
repeated_identical_tool_call  same tool + same arguments 3+ times, or identical code twice in a row
paging_loop                same tool on the same first argument 4+ times with other args varying
vision_used                an image / video-frame question went to the vision model
no_trace                   only an answers.jsonl row exists (usually a timeout)

Usage: ``python -m analysis.error_analysis [--run-id ID] [--runs-dir runs] [--max-steps 12]``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"

TAG_DESCRIPTIONS: dict[str, str] = {
    "hit_max_steps": "Ran out of steps before a final answer",
    "tool_error": "A tool returned or raised an error",
    "code_execution_error": "The agent's own Python code failed",
    "parse_error": "Model output had no parsable code block",
    "rate_limited": "A model call hit a 429 / quota",
    "timeout": "Wall-clock or model timeout",
    "empty_or_refusal_answer": "Empty, refusal or 'unknown' answer",
    "answer_format_suspicious": "Answer shape likely to fail exact match",
    "blocked_by_guard": "Answer-leak guard refused a query/URL",
    "no_tool_used": "Answered without calling any tool",
    "repeated_identical_tool_call": "Looped on an identical tool call",
    "paging_loop": "Same tool on the same target 4+ times (paging/re-reading)",
    "vision_used": "Used the vision model",
    "no_trace": "No trace file (answers.jsonl row only)",
}

# Tools the agent may expose (gaia_agent/tools_web.py + tools_media.py). Names seen in a run's
# own traces are added at runtime, so a renamed tool is still recognised.
DEFAULT_TOOL_NAMES = {
    "web_search",
    "visit_webpage",
    "wikipedia_search",
    "wikipedia_page",
    "wikipedia_revision",
    "wikipedia_page_info",
    "read_file",
    "run_python_file",
    "transcribe_audio",
    "image_question",
    "youtube_transcript",
    "youtube_audio_transcript",
    "youtube_frames_question",
}
VISION_TOOLS = {"image_question", "youtube_frames_question"}
NON_TOOLS = {"final_answer", "python_interpreter"}

# --------------------------------------------------------------------------- patterns

GUARD_RE = re.compile(r"BLOCKED by the answer-leak guard|\b(query|URL) refused\b", re.I)
# "429" only as a standalone token: a bare substring match also hits UUIDs and byte counts.
HTTP_429_RE = re.compile(r"(?<![0-9A-Za-z_-])429(?![0-9A-Za-z_-])")
RATE_RE = re.compile(
    r"rate[ _-]?limit|too many requests|RemoteQuotaExhausted|BudgetExceeded|quota (?:exceeded|exhausted)"
    r"|daily (?:request )?cap|resource[_ ]exhausted",
    re.I,
)
TIMEOUT_RE = re.compile(r"\btime(?:d)?[ _-]?out\b|timeouterror|APITimeoutError|deadline exceeded", re.I)
MAX_STEPS_RE = re.compile(r"reached max steps|AgentMaxStepsError|max(?:imum)? (?:number of )?steps", re.I)
PARSE_RE = re.compile(r"Error in code parsing|regex pattern .* was not found|AgentParsingError", re.I | re.S)
EXEC_RE = re.compile(r"Code execution failed|InterpreterError|AgentExecutionError", re.I)
FAILED_LINE_RE = re.compile(r"Code execution failed at line '(?P<line>.*?)' due to", re.S)
TOOL_EXC_RE = re.compile(r"Error (?:when )?executing tool '?(?P<tool>[A-Za-z_][A-Za-z0-9_]*)'?")
TOOL_ERROR_LINE_RE = re.compile(r"^\s*(?:ERROR|Error)\b[:\s]", re.M)
MODEL_ERROR_TYPES = {"AgentGenerationError"}

# Bare "unknown"/"none" style answers must match the whole answer ("Unknown Pleasures" is a
# legitimate album title); refusal phrases only need to open the answer.
REFUSAL_FULL_RE = re.compile(r"^(?:unknown|n/?a|none|null|nil|not (?:found|available|known|applicable)|no answer|\?+)$", re.I)
REFUSAL_PREFIX_RE = re.compile(
    r"^(?:cannot (?:be )?determined?|unable to (?:determine|find|answer)"
    r"|i(?: can ?not| cannot| can't| am unable|'m unable| do not know| don't know| could not| couldn't)"
    r"|i'?m sorry|sorry\b|insufficient (?:information|data)|it is not possible|there is no (?:way|information))",
    re.I,
)
UNIT_RE = re.compile(
    r"(?:^[$€£¥])|(?:\d\s*%$)|(?:\b\d[\d.,]*\s*(?:km|kg|mg|g|m|cm|mm|mi|miles?|lbs?|pounds?|ft|feet|inch(?:es)?|"
    r"meters?|metres?|seconds?|sec|s|minutes?|mins?|hours?|hrs?|h|days?|years?|yrs?|usd|eur|dollars?|euros?|"
    r"percent|°c|°f|k?w|mph|kph)\b)",
    re.I,
)
THOUSANDS_RE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
SENTENCE_VERB_RE = re.compile(r"\b(?:is|are|was|were|has|have|had|the answer|answer is|would be|equals)\b", re.I)


# --------------------------------------------------------------------------- helpers


def _walk(steps: Iterable[dict]) -> Iterable[dict]:
    """Every step, including steps of sub-agents nested under an action step."""
    for s in steps or []:
        yield s
        for run in s.get("sub_agents") or []:
            yield from _walk(run.get("steps") or [])


def _short(text: Any, n: int = 140) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _tool_names_in_code(code: str | None, known: set[str]) -> list[str]:
    if not code:
        return []
    return [m.group(1) for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code) if m.group(1) in known]


def _calls_in_code(code: str | None, known: set[str]) -> list[dict]:
    """Calls to known tools in a code block, with their argument source text as the key.

    Uses ``ast`` so ``visit_webpage(url_a)`` and ``visit_webpage(url_b)`` stay distinct; if the
    code does not parse, falls back to bare names with ``arguments=None`` (never counted as
    repeats, since identical names alone prove nothing).
    """
    if not code:
        return []
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [{"name": n, "arguments": None} for n in _tool_names_in_code(code, known)]
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in known:
            args = [ast.unparse(a) for a in node.args] + [f"{k.arg}={ast.unparse(k.value)}" for k in node.keywords]
            first = args[0].split("=", 1)[-1] if args else None
            calls.append({"name": node.func.id, "arguments": ", ".join(args), "first": first, "_lineno": node.lineno})
    calls.sort(key=lambda c: c.pop("_lineno"))
    return calls


def _step_tools(step: dict, known: set[str]) -> list[dict]:
    """Tool calls of one step: the trace's recovered calls, else calls parsed from the code."""
    calls = [c for c in step.get("tool_calls") or [] if c.get("name") and c["name"] not in NON_TOOLS]
    if calls:
        return calls
    return _calls_in_code(step.get("code"), known)


def _is_vision_tool(name: str) -> bool:
    # Token match, not substring: "wikipedia_revision" contains "vision".
    return name in VISION_TOOLS or bool(re.search(r"(?:^|_)(?:vision|vlm)(?:_|$)", name))


def _first_arg(call: dict) -> str | None:
    if call.get("first") is not None:
        return str(call["first"])
    args = call.get("arguments")
    if isinstance(args, dict) and args:
        return str(next(iter(args.values())))
    if isinstance(args, (list, tuple)) and args:
        return str(args[0])
    return None


def _args_key(args: Any) -> str:
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except Exception:
        return str(args)


def format_issues(answer: Any) -> list[str]:
    """Reasons an answer string looks wrong for GAIA's exact-match grader (empty list = fine)."""
    if answer is None:
        return []
    a = str(answer).strip()
    if not a:
        return []
    reasons = []
    if re.search(r"final\s*answer", a, re.I):
        reasons.append('contains "FINAL ANSWER"')
    if "\n" in a:
        reasons.append("multi-line")
    items = [x.strip() for x in a.split(",")] if "," in a and not THOUSANDS_RE.match(a) else [a]
    longest_item_words = max(len(x.split()) for x in items)
    if len(a) > 200 or longest_item_words > 12:
        reasons.append(f"very long ({len(a)} chars)")
    if any(UNIT_RE.search(x) for x in items):
        reasons.append("has units/currency/percent")
    if THOUSANDS_RE.match(a):
        reasons.append("thousands separator")
    words = a.split()
    if (a.endswith(".") and len(words) >= 4) or (len(words) >= 6 and SENTENCE_VERB_RE.search(a) and "," not in a):
        reasons.append("reads like a sentence")
    if len(a) >= 2 and a[0] in "\"'“‘`" and a[-1] in "\"'”’`":
        reasons.append("wrapped in quotes")
    if a.startswith(("**", "__")) or a.startswith("```"):
        reasons.append("markdown formatting")
    return reasons


def is_empty_or_refusal(answer: Any) -> str | None:
    if answer is None:
        return "no final answer"
    a = str(answer).strip().strip(".").strip()
    if not a:
        return "empty answer"
    if REFUSAL_FULL_RE.match(a) or REFUSAL_PREFIX_RE.match(a):
        return f"refusal-like answer {_short(a, 60)!r}"
    return None


# --------------------------------------------------------------------------- core


def analyze_trace(
    trace: dict,
    *,
    max_steps: int | None = None,
    answer_row: dict | None = None,
    known_tools: set[str] | None = None,
) -> dict:
    """Tag one trace. Returns {task_id, tags, evidence, details}."""
    known = set(DEFAULT_TOOL_NAMES) | set(known_tools or ())
    for name in (trace.get("totals") or {}).get("tool_calls_by_name", {}) or {}:
        if name not in NON_TOOLS:
            known.add(name)

    details: dict[str, list[str]] = {}

    def tag(name: str, ev: str) -> None:
        lst = details.setdefault(name, [])
        if ev not in lst and len(lst) < 4:
            lst.append(ev)

    steps = list(_walk(trace.get("steps") or []))
    top_actions = [s for s in (trace.get("steps") or []) if s.get("type") == "action"]
    run_error = trace.get("error") or (answer_row or {}).get("error")
    run_error_type = trace.get("error_type")

    # ---- run-level errors
    if run_error:
        if TIMEOUT_RE.search(str(run_error)) or (run_error_type or "").lower().endswith("timeouterror"):
            tag("timeout", f"run error: {_short(run_error, 100)}")
        if HTTP_429_RE.search(str(run_error)) or RATE_RE.search(str(run_error)) or (
            run_error_type in ("BudgetExceeded", "RemoteQuotaExhausted")
        ):
            tag("rate_limited", f"run error: {_short(run_error, 100)}")
        if MAX_STEPS_RE.search(str(run_error)) or run_error_type == "AgentMaxStepsError":
            tag("hit_max_steps", f"run error: {_short(run_error, 80)}")

    # ---- per-step
    call_counter: Counter = Counter()
    target_counter: Counter = Counter()
    call_first_step: dict[str, Any] = {}
    prev_code = None
    vision_steps: list[str] = []
    any_tool = False
    for s in steps:
        idx = s.get("index")
        where = f"step {idx}" + (f" ({s['agent']})" if s.get("agent") else "")
        err = s.get("error") or ""
        etype = s.get("error_type") or ""
        obs = s.get("observation") or ""
        tools = _step_tools(s, known) if s.get("type") == "action" else []
        tool_names = [t["name"] for t in tools]
        if tools:
            any_tool = True

        if etype == "AgentMaxStepsError" or (err and MAX_STEPS_RE.search(err)):
            tag("hit_max_steps", f"{where}: {_short(err, 80)}")

        if err:
            model_side = etype in MODEL_ERROR_TYPES or err.startswith("Error in generating model output")
            if model_side and (HTTP_429_RE.search(err) or RATE_RE.search(err)):
                tag("rate_limited", f"{where}: {_short(err, 100)}")
            if model_side and TIMEOUT_RE.search(err):
                tag("timeout", f"{where}: model call timed out")
            if etype == "AgentParsingError" or PARSE_RE.search(err):
                tag("parse_error", f"{where}: {_short(err, 90)}")
            elif etype in ("AgentToolExecutionError", "AgentToolCallError") or TOOL_EXC_RE.search(err):
                m = TOOL_EXC_RE.search(err)
                name = m.group("tool") if m else (tool_names[0] if len(set(tool_names)) == 1 else "unattributed")
                tag(f"tool_error:{name}", f"{where}: {_short(err, 110)}")
            elif etype == "AgentExecutionError" or EXEC_RE.search(err):
                failing = FAILED_LINE_RE.search(err)
                in_tool = [n for n in _tool_names_in_code(failing.group("line"), known)] if failing else []
                if GUARD_RE.search(err):
                    tag("blocked_by_guard", f"{where}: {_short(err, 90)}")
                elif in_tool:
                    tag(f"tool_error:{in_tool[0]}", f"{where}: raised in {in_tool[0]}(): {_short(err, 90)}")
                elif not model_side:
                    tag("code_execution_error", f"{where}: {_short(err, 110)}")

        if GUARD_RE.search(obs) or GUARD_RE.search(err):
            tag("blocked_by_guard", f"{where}: guard refused a {'query' if 'query refused' in (obs + err) else 'URL/result'}")

        # Tools in this codebase report failures as returned strings starting with "ERROR".
        for m in TOOL_ERROR_LINE_RE.finditer(obs):
            line = obs[m.start() : obs.find("\n", m.start()) if obs.find("\n", m.start()) != -1 else len(obs)]
            if GUARD_RE.search(line):
                continue
            named = [n for n in set(tool_names) if n in line]
            if named:
                name = named[0]
            elif len(set(tool_names)) == 1:
                name = tool_names[0]
            else:
                name = "unattributed"
            tag(f"tool_error:{name}", f"{where}: {_short(line, 110)}")

        for t in tools:
            first = _first_arg(t)
            if first is not None:
                target_counter[f"{t['name']}|{first}"] += 1
            if t.get("arguments") is not None:
                key = f"{t['name']}({_args_key(t.get('arguments'))})"
                call_counter[key] += 1
                call_first_step.setdefault(key, idx)
            if _is_vision_tool(t["name"]):
                vision_steps.append(f"{where}: {t['name']}")

        code = (s.get("code") or "").strip()
        if s.get("type") == "action" and code and prev_code is not None and code == prev_code and tools:
            tag("repeated_identical_tool_call", f"{where}: identical code to the previous step")
        if s.get("type") == "action":
            prev_code = code or None

    for key, n in call_counter.items():
        if n >= 3:
            tag("repeated_identical_tool_call", f"{_short(key, 100)} x{n}")
    for key, n in target_counter.items():
        if n >= 4 and not any(k.startswith(key.split("|", 1)[0] + "(") and c >= 3 for k, c in call_counter.items()):
            name, target = key.split("|", 1)
            tag("paging_loop", f"{name} on {_short(target, 80)} x{n}")

    for v in vision_steps:
        tag("vision_used", v)

    ran_code = [s for s in top_actions if (s.get("code") or "").strip()]
    # Reading an attachment directly in code (pandas, open()) is real work, not "answered from memory".
    read_direct = any(re.search(r"\b(read_csv|read_excel|read_json|open\()", s.get("code") or "") for s in ran_code)
    if ran_code and not any_tool and not read_direct:  # parse failures that never ran code are not "answered without tools"
        tag("no_tool_used", f"{len(ran_code)} code step(s), no tool calls")

    if max_steps is not None and len(top_actions) >= max_steps and "hit_max_steps" not in details:
        if not any(s.get("is_final_answer") for s in top_actions):
            tag("hit_max_steps", f"{len(top_actions)} action steps >= max_steps {max_steps} without final_answer")

    # ---- the answer
    final = trace.get("final_answer")
    if final is None and answer_row is not None:
        final = answer_row.get("answer")
    empty = is_empty_or_refusal(final)
    if empty:
        tag("empty_or_refusal_answer", empty)
    else:
        issues = format_issues(final)
        raw = (answer_row or {}).get("raw")
        if not issues and raw is not None and raw != final:
            raw_issues = format_issues(raw)
            if raw_issues:  # the normalizer cleaned it, but the model's habit is worth knowing
                details.setdefault("_notes", []).append(f"raw answer had: {', '.join(raw_issues)}")
        if issues:
            tag("answer_format_suspicious", f"{', '.join(issues)}: {_short(final, 70)!r}")

    # Collapse tool_error:<name> into the generic tag as well.
    tool_tags = sorted(k for k in details if k.startswith("tool_error:"))
    if tool_tags:
        details["tool_error"] = [k.split(":", 1)[1] for k in tool_tags]

    ordered = [t for t in TAG_DESCRIPTIONS if t in details] + tool_tags
    evidence = "; ".join(f"{t}: {details[t][0]}" for t in ordered if t != "tool_error")
    return {
        "task_id": trace.get("task_id"),
        "tags": ordered,
        "evidence": evidence,
        "details": {k: v for k, v in details.items()},
        "final_answer": None if final is None else _short(final, 120),
        "steps": len(top_actions),
        "seconds": trace.get("seconds"),
    }


def analyze_answer_row(row: dict) -> dict:
    """Tag a task that has no trace file, from its answers.jsonl row alone."""
    pseudo = {
        "task_id": row.get("task_id"),
        "steps": [],
        "final_answer": row.get("answer"),
        "error": row.get("error"),
        "seconds": row.get("seconds"),
    }
    item = analyze_trace(pseudo, answer_row=row)
    item["tags"] = ["no_trace"] + item["tags"]
    item["details"]["no_trace"] = ["only an answers.jsonl row"]
    item["evidence"] = ("no_trace: answers.jsonl row only; " + item["evidence"]).rstrip("; ")
    item["steps"] = row.get("steps")
    return item


# --------------------------------------------------------------------------- run level


def load_run_traces(run_id: str, runs_dir: Path) -> list[dict]:
    folder = Path(runs_dir) / "traces" / run_id
    traces = []
    if not folder.is_dir():
        return traces
    for p in sorted(folder.glob("*.json")):
        if p.name.startswith("."):
            continue
        try:
            traces.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            print(f"warning: skipping unreadable trace {p.name}: {e}", file=sys.stderr)
    return traces


def load_answer_rows(run_id: str, runs_dir: Path) -> dict[str, dict]:
    """Latest answers.jsonl row per task for this run (rows without run_id are ignored)."""
    path = Path(runs_dir) / "answers.jsonl"
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("run_id") == run_id and r.get("task_id"):
            rows[r["task_id"]] = r
    return rows


def latest_run_id(runs_dir: Path) -> str | None:
    folder = Path(runs_dir) / "traces"
    if not folder.is_dir():
        return None
    dirs = [d for d in folder.iterdir() if d.is_dir() and any(d.glob("*.json"))]
    return max(dirs, key=lambda d: d.stat().st_mtime).name if dirs else None


def analyze_run(run_id: str, runs_dir: Path | str = DEFAULT_RUNS_DIR, max_steps: int | None = None) -> dict:
    runs_dir = Path(runs_dir)
    traces = load_run_traces(run_id, runs_dir)
    rows = load_answer_rows(run_id, runs_dir)
    known: set[str] = set()
    for t in traces:
        known.update(k for k in ((t.get("totals") or {}).get("tool_calls_by_name") or {}) if k not in NON_TOOLS)
    items = []
    seen = set()
    for t in traces:
        tid = t.get("task_id")
        seen.add(tid)
        items.append(analyze_trace(t, max_steps=max_steps, answer_row=rows.get(tid), known_tools=known))
    for tid, row in rows.items():
        if tid not in seen:
            items.append(analyze_answer_row(row))
    counts: Counter = Counter()
    for it in items:
        counts.update(it["tags"])
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tasks": len(items),
        "max_steps": max_steps,
        "items": items,
        "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "clean_tasks": [it["task_id"] for it in items if not [t for t in it["tags"] if t != "vision_used"]],
        "tag_descriptions": TAG_DESCRIPTIONS,
        "note": "Heuristic behaviour tags from traces; no gold answers are used.",
    }


def write_report(report: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return out


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-", re.I)


def _short_id(tid: Any) -> str:
    s = str(tid)
    return s[:8] if _UUID_RE.match(s) else s[:24]


def format_table(report: dict) -> str:
    n = report["n_tasks"] or 1
    lines = [f"Error analysis: run {report['run_id']} ({report['n_tasks']} tasks)", ""]
    header = f"{'tag':38s} {'n':>3s} {'%':>5s}  examples"
    lines += [header, "-" * len(header)]
    examples: dict[str, list[str]] = {}
    for it in report["items"]:
        for t in it["tags"]:
            examples.setdefault(t, []).append(_short_id(it["task_id"]))
    for tag, c in report["counts"].items():
        ex = ", ".join(examples.get(tag, [])[:3]) + (" …" if len(examples.get(tag, [])) > 3 else "")
        lines.append(f"{tag:38s} {c:3d} {100 * c / n:4.0f}%  {ex}")
    if not report["counts"]:
        lines.append("(no failure tags)")
    lines += ["", f"clean (no failure tag): {len(report['clean_tasks'])}/{report['n_tasks']}", ""]
    lines.append(f"{'task':24s} {'steps':>5s}  tags")
    for it in report["items"]:
        tags = ", ".join(it["tags"]) or "-"
        lines.append(f"{_short_id(it['task_id']):24s} {str(it.get('steps') or '-'):>5s}  {tags}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tag failure modes in a run's traces (no gold needed).")
    ap.add_argument("--run-id", default=os.environ.get("GAIA_RUN_ID"))
    ap.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    ap.add_argument("--max-steps", type=int, default=None, help="agent max_steps, to infer hit_max_steps")
    ap.add_argument("--out", default=None, help="output path (default <runs-dir>/error_analysis.json)")
    ap.add_argument("--json", action="store_true", help="print the JSON instead of the table")
    args = ap.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    run_id = args.run_id or latest_run_id(runs_dir)
    if not run_id:
        print(f"No traces found under {runs_dir / 'traces'}; pass --run-id.", file=sys.stderr)
        return 2
    report = analyze_run(run_id, runs_dir, args.max_steps)
    if not report["items"]:
        print(f"No traces or answers for run {run_id!r} under {runs_dir}.", file=sys.stderr)
        return 2
    out = write_report(report, Path(args.out) if args.out else runs_dir / "error_analysis.json")
    print(json.dumps(report, indent=2) if args.json else format_table(report))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
