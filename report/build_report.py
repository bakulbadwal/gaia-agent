#!/usr/bin/env python3
"""Build the static results site (docs/) from the agent's run artifacts.

Reads, all optional:
    runs/answers.jsonl            one answer per question (latest line per task wins)
    runs/traces/<run_id>/*.json   step-by-step trajectories
    runs/submissions.jsonl        course-API responses (official score)
    runs/error_analysis.json      failure taxonomy
    runs/usage.jsonl              one row per model request (provider layer)
    evals/results/*.json          practice-set evals and ablations
    evals/practice/tasks.jsonl    the practice questions (ours, shown in full)

Writes a self-contained static site with no build step and no framework:
    <out>/index.html              page shell + data inlined as JSON
    <out>/assets/style.css
    <out>/assets/app.js
    <out>/traces/<run>/<task>.json  practice trajectories (fetched on demand)

Usage:
    python report/build_report.py                 # root=., out=docs, PUBLIC build
    python report/build_report.py --private --out /tmp/site   # local, full GAIA text
    python report/build_report.py --root tests/fixtures/sample_run --out /tmp/site

Contamination policy (public build, the default): GAIA asks that its items not be
reshared in crawlable form. GAIA run items are reduced to a short task id, attachment
type, tools, counts, timing, status and failure tags; their trajectories to a step
skeleton. No question text, submitted answer, model output, code or observation is
written for a GAIA item. Practice-set items are our own and are shown in full.

A root containing a file named FIXTURE is treated as fake development data; the
site then carries a banner on every view saying so. Fixture data is never results.

Only the standard library is used, so the report builds anywhere.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template.html"
ASSETS = HERE / "assets"

PROJECT = "GAIA Agent — HF Agents Course Unit 4"
PITCH = (
    "A smolagents CodeAgent on free, open-weight models that answers GAIA Level-1 questions "
    "with web, Wikipedia, file, audio, image and video tools, and records every "
    "step it takes so each answer can be audited."
)
PASS_MARK = 30.0  # percent needed for the course certificate

# Tools the agent registers (gaia_agent.tools_web.TOOLS_WEB + tools_media.TOOLS_MEDIA).
# Kept static so the report never imports the agent's heavy dependencies; any
# extra names found in trace totals (e.g. managed agents in multi mode) are added.
KNOWN_TOOLS = [
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
]
NON_TOOLS = {"python_interpreter", "final_answer"}
TEXT_LIMIT = 4000  # same cap as the trace contract
INLINE_TRACE_BUDGET = 1_500_000  # bytes of practice-trace JSON inlined; the rest is fetched
TRACE_MARKER = ".generated-by-build_report"
POLICY = (
    "GAIA asks that its questions and answers not be reshared in a form crawlers can index, "
    "so this page never shows a GAIA question, submitted answer, model output, code or "
    "observation. Each GAIA item appears only as a short task id, its attachment type, the "
    "tools it used, step, call, token and time counts, its status and its failure tags; its "
    "trajectory is reduced to a step skeleton. The practice set is our own and is shown in full."
)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,63}$")
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_:.\-]{0,63}$")
_EXT_RE = re.compile(r"^[a-z0-9]{1,6}$")
_ATTACHED_RE = re.compile(r"Attached file:\s*(\S+)")
_TRUNC_RE = re.compile(r"…\[truncated (\d+) chars\]")


def warn(msg: str) -> None:
    print(f"[build_report] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- readers


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        warn(f"skipping unreadable {path}: {e}")
        return None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            warn(f"{path.name}:{n}: skipping malformed line ({e})")
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def clip(x: Any, n: int = TEXT_LIMIT) -> str | None:
    """Clip a long text field to the trace contract's limit (idempotent on clipped text)."""
    if x is None:
        return None
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    if len(s) <= n:
        return s
    return s[:n] + f"…[truncated {len(s) - n} chars]"


def num(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None or isinstance(x, bool):
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def as_int(x: Any) -> int:
    v = num(x, 0.0)
    return int(v or 0)


# --------------------------------------------------------------------------- statistics


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion, as (low, high) in [0, 1]."""
    if n <= 0:
        return None
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021): P(at least one of k samples is correct)."""
    if k > n:
        raise ValueError("k must be <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k (Yao et al., tau-bench 2024): P(all k samples are correct)."""
    if k > n:
        raise ValueError("k must be <= n")
    return math.comb(c, k) / math.comb(n, k)


def percentile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


# --------------------------------------------------------------------------- traces


def status_of(answer: Any, error: Any) -> str:
    err = (error or "").strip() if isinstance(error, str) else (str(error) if error else "")
    if err:
        if re.search(r"time(?:d)?\s*-?\s*out|timeout|deadline", err, re.I):
            return "timeout"
        return "error"
    if answer is None or str(answer).strip() == "":
        return "empty"
    return "answered"


def tools_in_code(code: str | None, known: set[str]) -> list[str]:
    """Tool names called in a code action, in call order (repeats kept)."""
    if not code:
        return []
    found = []
    for m in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\s*\(", code):
        if m.group(1) in known:
            found.append(m.group(1))
    return found


def _step_type(raw: dict) -> str:
    t = str(raw.get("type") or "").lower()
    if t in {"planning", "action", "final", "task"}:
        return t
    if raw.get("final"):  # legacy agent.py trace: the final_answer action step
        return "final"
    if raw.get("plan") is not None:
        return "planning"
    return "action"


MAX_SUB_DEPTH = 3  # managed agents can nest; deeper levels are summarised, not drawn


def normalize_step(raw: dict, i: int, known: set[str], depth: int = 0) -> dict:
    code = raw.get("code") if raw.get("code") is not None else raw.get("code_action")
    obs = raw.get("observation") if raw.get("observation") is not None else raw.get("observations")
    out = raw.get("model_output") if raw.get("model_output") is not None else raw.get("plan")
    calls = []
    for tc in raw.get("tool_calls") or []:
        if isinstance(tc, dict):
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            args = tc.get("arguments", (tc.get("function") or {}).get("arguments"))
            calls.append({"name": str(name or "?"), "arguments": clip(args, 1200)})
    direct = [c["name"] for c in calls if c["name"] not in NON_TOOLS]
    # gaia_agent.trace already parses real tool calls out of the code; older or raw
    # smolagents traces only say "python_interpreter", so read the code ourselves.
    tools = direct or tools_in_code(code if isinstance(code, str) else None, known)
    step_type = _step_type(raw)
    subs = []
    for run in raw.get("sub_agents") or []:
        if isinstance(run, dict):
            subs.append(normalize_subrun(run, known, depth + 1))
    tin, tout = raw.get("input_tokens"), raw.get("output_tokens")
    return {
        "index": as_int(raw.get("index", raw.get("step", i))),
        "step_number": as_int(raw.get("step_number")) or None,
        "type": step_type,
        "agent": raw.get("agent"),
        "is_final_answer": bool(raw.get("is_final_answer")) or (step_type == "final" and bool(code)),
        "model_output": clip(out),
        "code": clip(code),
        "tool_calls": calls,
        "tools": tools,
        "observation": clip(obs),
        "action_output": clip(raw.get("action_output"), 1200),
        "error": clip(raw.get("error")),
        "error_type": raw.get("error_type"),
        "images": as_int(raw.get("images")) or None,
        "input_tokens": as_int(tin),
        "output_tokens": as_int(tout),
        "tokens_known": tin is not None or tout is not None,
        "duration_s": num(raw.get("duration_s")),
        "sub_agents": subs,
    }


def normalize_subrun(run: dict, known: set[str], depth: int) -> dict:
    steps = run.get("steps") or []
    if depth > MAX_SUB_DEPTH:
        kept: list[dict] = []
    else:
        kept = [normalize_step(s, i, known, depth) for i, s in enumerate(steps) if isinstance(s, dict)]
    return {
        "agent": str(run.get("agent") or "sub-agent"),
        "task": clip(run.get("task"), 1500),
        "final_answer": clip(run.get("final_answer"), 1200),
        "seconds": num(run.get("seconds")),
        "partial": bool(run.get("partial")),
        "steps": kept,
        "elided_steps": len(steps) - len(kept),
    }


def _is_action(s: dict) -> bool:
    # A legacy "final" step that carries code was an action step too.
    return s["type"] == "action" or (s["type"] == "final" and bool(s["code"]))


def normalize_trace(raw: dict, known: set[str]) -> dict:
    totals = raw.get("totals") or {}
    extra = {k for k in (totals.get("tool_calls_by_name") or {}) if k not in NON_TOOLS}
    known = known | extra
    steps = [normalize_step(s, i, known) for i, s in enumerate(raw.get("steps") or []) if isinstance(s, dict)]

    def walk(ss):
        for s in ss:
            yield s
            for run in s["sub_agents"]:
                yield from walk(run["steps"])

    by_name: Counter = Counter()
    for s in walk(steps):
        by_name.update(s["tools"])
    reported = {k: as_int(v) for k, v in (totals.get("tool_calls_by_name") or {}).items() if k not in NON_TOOLS}
    # Prefer the runner's own count when it saw real tool names; else what the code shows.
    tool_counts = reported if sum(reported.values()) else dict(by_name)
    all_steps = list(walk(steps))
    tin = as_int(totals.get("input_tokens")) or sum(s["input_tokens"] for s in all_steps)
    tout = as_int(totals.get("output_tokens")) or sum(s["output_tokens"] for s in all_steps)
    n_steps = as_int(totals.get("steps")) or sum(1 for s in steps if _is_action(s))
    model_calls = as_int(totals.get("model_calls")) or sum(
        1 for s in all_steps if _is_action(s) or s["type"] == "planning"
    )
    agents = sorted({s["agent"] for s in all_steps if s.get("agent")})
    return {
        "task_id": str(raw.get("task_id") or ""),
        "question": raw.get("question") or "",
        "run_id": raw.get("run_id"),
        "model": raw.get("model"),
        "provider": raw.get("provider"),
        "arch": raw.get("arch"),
        "started_at": raw.get("started_at"),
        "seconds": num(raw.get("seconds")),
        "final_answer": raw.get("final_answer"),
        "error": clip(raw.get("error")),
        "error_type": raw.get("error_type"),
        "steps": steps,
        "agents": agents,
        "totals": {
            "steps": n_steps,
            "planning_steps": as_int(totals.get("planning_steps")) or sum(1 for s in steps if s["type"] == "planning"),
            "input_tokens": tin,
            "output_tokens": tout,
            "model_calls": model_calls,
            "tool_calls_by_name": tool_counts,
            "step_errors": as_int(totals.get("step_errors")) or sum(1 for s in all_steps if s["error"]),
            "sub_agent_calls": as_int(totals.get("sub_agent_calls")) or sum(len(s["sub_agents"]) for s in all_steps),
            "token_usage_complete": totals.get("token_usage_complete", True) is not False,
        },
    }


def unwrap_trace(raw: Any) -> Any:
    """run.py's legacy files wrap the trace: {task_id, question, trace: {...} | [steps]}."""
    if not isinstance(raw, dict) or "steps" in raw or "trace" not in raw:
        return raw
    inner = raw["trace"]
    if isinstance(inner, list):
        return {**{k: v for k, v in raw.items() if k != "trace"}, "steps": inner}
    if isinstance(inner, dict):
        return {**{k: v for k, v in raw.items() if k != "trace"}, **inner}
    return raw


def attachment_type(file_name: Any, raw_trace: dict | None = None) -> str | None:
    """Lower-case file extension of a task's attachment, or None. Never the name itself."""
    name = file_name if isinstance(file_name, str) and file_name.strip() else None
    if name is None and isinstance(raw_trace, dict):
        for st in raw_trace.get("steps") or []:
            if isinstance(st, dict) and str(st.get("type") or "") == "task":
                m = _ATTACHED_RE.search(str(st.get("model_output") or ""))
                if m:
                    name = m.group(1)
                break
    if not name:
        return None
    ext = Path(name.strip().rstrip(".,;)")).suffix.lstrip(".").lower()
    return ext if _EXT_RE.match(ext) else None


def load_traces(runs: Path, known: set[str]) -> dict[str, dict[str, dict]]:
    """{run_id: {task_id: trace}} from runs/traces/<run_id>/<task_id>.json."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    base = runs / "traces"
    if not base.is_dir():
        return {}
    for path in sorted(base.glob("**/*.json")):
        raw = unwrap_trace(read_json(path))
        if not isinstance(raw, dict):
            continue
        tr = normalize_trace(raw, known)
        tr["attachment"] = attachment_type(None, raw)
        tr["task_id"] = tr["task_id"] or path.stem
        run_id = str(tr["run_id"] or (path.parent.name if path.parent != base else "default"))
        tr["run_id"] = run_id
        out[run_id][tr["task_id"]] = tr
    return dict(out)


# --------------------------------------------------------------------------- run selection


def _provider_fields(info: Any) -> dict:
    if isinstance(info, dict):
        return {
            "provider": info.get("provider") or info.get("name"),
            "model": info.get("model") or info.get("model_id"),
            "arch": info.get("arch"),
            "vision_model": info.get("vision_model") or info.get("vision_model_id"),
            "verify": info.get("verify"),
        }
    if isinstance(info, str) and info.strip():
        return {"provider": info.strip()}
    return {}


def pick_run(answers: list[dict], traces: dict, submissions: list[dict]) -> str | None:
    """The run the site describes: the last submitted run, else the latest answered run."""
    answered = {a.get("run_id") for a in answers if a.get("run_id")}
    for sub in reversed(submissions):
        rid = sub.get("run_id")
        if rid and (rid in answered or rid in traces):
            return rid
    for a in reversed(answers):
        if a.get("run_id"):
            return a["run_id"]
    if traces:
        return max(traces, key=lambda r: max((t.get("started_at") or "" for t in traces[r].values()), default=""))
    return None


def build_questions(answers: list[dict], traces: dict, run_id: str | None) -> tuple[list[dict], dict]:
    in_run = [a for a in answers if run_id is None or a.get("run_id") in (run_id, None)]
    if run_id and not any(a.get("run_id") == run_id for a in in_run):
        in_run = answers
    latest: dict[str, dict] = {}
    order: list[str] = []
    for a in in_run:
        tid = str(a.get("task_id") or "")
        if not tid:
            continue
        if tid not in latest:
            order.append(tid)
        latest[tid] = a

    run_traces = dict(traces.get(run_id or "", {}))
    # Traces with no answer line still deserve a row, but only the selected run's (or, with
    # no answers at all, every run's): a smoke run's traces are not part of the submission.
    extra = run_traces if (latest or run_id) else {t: tr for r in sorted(traces) for t, tr in traces[r].items()}
    # Fall back to any run's trace for an answered task the selected run has none for.
    for rid in sorted(traces):
        for tid, tr in traces[rid].items():
            if tid in latest:
                run_traces.setdefault(tid, tr)
    for tid, tr in extra.items():
        run_traces.setdefault(tid, tr)
    for tid in extra:
        if tid not in latest:
            order.append(tid)
            tr = run_traces[tid]
            latest[tid] = {
                "task_id": tid,
                "question": tr.get("question"),
                "answer": tr.get("final_answer"),
                "error": tr.get("error"),
                "seconds": tr.get("seconds"),
                "run_id": tr.get("run_id"),
            }

    questions, used = [], {}
    for n, tid in enumerate(order, 1):
        a = latest[tid]
        tr = run_traces.get(tid)
        totals = (tr or {}).get("totals") or {}
        answer = a.get("answer")
        q = {
            "n": n,
            "task_id": tid,
            "question": a.get("question") or (tr or {}).get("question") or "",
            "answer": "" if answer is None else str(answer),
            "raw": clip(a.get("raw"), 600),
            "error": clip(a.get("error") or (tr or {}).get("error"), 1200),
            "status": status_of(answer, a.get("error") or (tr or {}).get("error")),
            "steps": as_int(a.get("steps")) or as_int(totals.get("steps")),
            "seconds": num(a.get("seconds"), num((tr or {}).get("seconds"))),
            "input_tokens": as_int(totals.get("input_tokens")) or as_int(a.get("input_tokens")),
            "output_tokens": as_int(totals.get("output_tokens")) or as_int(a.get("output_tokens")),
            "tools": dict(sorted((totals.get("tool_calls_by_name") or {}).items(), key=lambda kv: -kv[1])),
            "model_calls": as_int(totals.get("model_calls")) or as_int(a.get("model_calls")) or None,
            "attachment": attachment_type(a.get("file_name")) or (tr or {}).get("attachment"),
            "has_trace": tr is not None,
            "run_id": a.get("run_id") or (tr or {}).get("run_id"),
        }
        q["tokens"] = q["input_tokens"] + q["output_tokens"]
        questions.append(q)
        if tr is not None:
            used[tid] = tr
    return questions, used


def build_analytics(questions: list[dict]) -> dict:
    calls: Counter = Counter()
    users: Counter = Counter()
    for q in questions:
        for name, c in q["tools"].items():
            calls[name] += c
            users[name] += 1
    secs = [q["seconds"] for q in questions if q["seconds"] is not None]
    steps = [q["steps"] for q in questions if q["steps"]]
    tokens = [q["tokens"] for q in questions if q["tokens"]]
    status = Counter(q["status"] for q in questions)
    return {
        "tool_usage": [
            {"name": n, "calls": c, "questions": users[n]} for n, c in sorted(calls.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "status": {k: status.get(k, 0) for k in ("answered", "error", "timeout", "empty")},
        "seconds": {
            "median": percentile(secs, 0.5),
            "p90": percentile(secs, 0.9),
            "total": round(sum(secs), 1) if secs else None,
        },
        "steps": {"median": percentile(steps, 0.5), "max": max(steps) if steps else None},
        "tokens": {"total": sum(tokens) if tokens else 0, "median": percentile(tokens, 0.5)},
        "tool_calls_total": sum(calls.values()),
    }


# --------------------------------------------------------------------------- evals


def _config_key(res: dict) -> tuple:
    cfg = res.get("config") or {}
    disabled = cfg.get("disabled_tools") or []
    if isinstance(disabled, str):
        disabled = [t for t in disabled.split(",") if t.strip()]
    verify = cfg.get("verify")
    return (
        str(res.get("set") or "practice"),
        str(cfg.get("model") or "?"),
        str(cfg.get("arch") or "single"),
        tuple(sorted(str(t).strip() for t in disabled)),
        bool(verify) and str(verify).lower() not in {"0", "false", "no"},
    )


def _config_label(key: tuple, multi_model: bool, name: str | None = None) -> str:
    _set, model, arch, disabled, verify = key
    if name:
        label = name.replace("_", " ")
    else:
        parts = []
        if arch == "multi":
            parts.append("multi-agent")
        if verify:
            parts.append("+ verifier")
        if disabled:
            parts.append("− " + ", ".join(disabled))
        label = " ".join(parts) or "baseline"
    if multi_model and not name:
        label = f"{model.split('/')[-1]} · {label}"
    return label


def _is_correct(item: dict, field: str = "correct") -> bool | None:
    c = item.get(field)
    return None if c is None else bool(c)


def is_eval_run(doc: Any) -> bool:
    """evals/results/ also holds pass@k and ablation reports; only per-run docs have items."""
    return isinstance(doc, dict) and isinstance(doc.get("items"), list) and bool(doc.get("run_id"))


def load_ablation_names(paths: list[Path]) -> dict:
    """run_id -> (config name, McNemar p vs baseline) from evals/ablate.py reports, newest wins."""
    out: dict[str, dict] = {}
    docs = [d for d in (read_json(p) for p in paths) if isinstance(d, dict) and isinstance(d.get("configs"), list)]
    for doc in sorted(docs, key=lambda d: str(d.get("created") or "")):
        for row in doc["configs"]:
            if not isinstance(row, dict) or not row.get("name"):
                continue
            vs = row.get("vs_baseline") or {}
            for rid in row.get("run_ids") or []:
                out[str(rid)] = {
                    "name": str(row["name"]),
                    "mcnemar_p": num(vs.get("mcnemar_p")),
                    "shared_tasks": vs.get("shared_tasks"),
                }
    return out


def build_evals(results: list[dict], ablation: dict | None = None) -> dict:
    ablation = ablation or {}
    runs = []
    for r in results:
        all_items = [i for i in r.get("items") or [] if isinstance(i, dict)]
        items = [i for i in all_items if not i.get("skipped")]
        judged = [i for i in items if _is_correct(i) is not None]
        correct = sum(1 for i in judged if _is_correct(i))
        strict_j = [i for i in items if _is_correct(i, "correct_strict") is not None]
        summ = r.get("summary") or {}
        acc = (correct / len(judged)) if judged else num(summ.get("accuracy"))
        if acc is not None and acc > 1:  # tolerate a percentage
            acc /= 100.0
        steps = [num(i.get("steps")) for i in items if num(i.get("steps")) is not None]
        secs = [num(i.get("seconds")) for i in items if num(i.get("seconds")) is not None]
        toks = [as_int(i.get("input_tokens")) + as_int(i.get("output_tokens")) for i in items]
        rid = str(r.get("run_id") or "?")
        runs.append(
            {
                "run_id": rid,
                "set": str(r.get("set") or "practice"),
                "config": r.get("config") or {},
                "key": _config_key(r),
                "name": (ablation.get(rid) or {}).get("name"),
                "n": len(items) or as_int(summ.get("n")),
                "skipped": len(all_items) - len(items),
                "judged": len(judged),
                "correct": correct if judged else (as_int(summ.get("correct")) if summ.get("correct") is not None else None),
                "accuracy": acc,
                "accuracy_strict": (sum(1 for i in strict_j if _is_correct(i, "correct_strict")) / len(strict_j))
                if strict_j
                else None,
                "mean_steps": (sum(steps) / len(steps)) if steps else num(summ.get("mean_steps")),
                "mean_seconds": (sum(secs) / len(secs)) if secs else num(summ.get("mean_seconds")),
                "total_tokens": sum(toks) or as_int(summ.get("total_tokens")),
                "errors": sum(1 for i in items if i.get("error")),
                "items": items,
            }
        )
    runs.sort(key=lambda r: r["run_id"])
    multi_model = len({r["key"][1] for r in runs}) > 1

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        # Group by the ablation's config name when it has one, else by the config itself.
        # A run named "baseline" by the ablation runner is the same config as the plain repeat runs
        # (practice-baseline-r1..rK), so they pool: that is what makes pass^k computable.
        groups[(r["key"], None if r["name"] == "baseline" else r["name"])].append(r)

    rows = []
    for (key, name), grs in groups.items():
        judged = sum(r["judged"] for r in grs)
        correct = sum(r["correct"] or 0 for r in grs if r["judged"])
        accs = [r["accuracy"] for r in grs if r["accuracy"] is not None]
        stricts = [r["accuracy_strict"] for r in grs if r["accuracy_strict"] is not None]
        ci = wilson(correct, judged)
        n_items = sum(r["n"] for r in grs)
        ps = [ablation[r["run_id"]].get("mcnemar_p") for r in grs if r["run_id"] in ablation]
        row = {
            "label": _config_label(key, multi_model, name),
            "name": name,
            "set": key[0],
            "model": key[1],
            "arch": key[2],
            "disabled_tools": list(key[3]),
            "verify": key[4],
            "runs": len(grs),
            "run_ids": [r["run_id"] for r in grs],
            "items_per_run": round(n_items / len(grs), 1) if grs else 0,
            "skipped": sum(r["skipped"] for r in grs),
            "accuracy": (sum(accs) / len(accs)) if accs else None,
            "accuracy_strict": (sum(stricts) / len(stricts)) if stricts else None,
            "accuracy_min": min(accs) if accs else None,
            "accuracy_max": max(accs) if accs else None,
            "ci95": list(ci) if ci else None,
            "mean_steps": _mean([r["mean_steps"] for r in grs]),
            "mean_seconds": _mean([r["mean_seconds"] for r in grs]),
            "tokens_per_item": (sum(r["total_tokens"] for r in grs) / n_items) if n_items else None,
            "mcnemar_p": next((p for p in ps if p is not None), None),
            "passk": _passk(grs),
            "by_skill": _by_skill(grs),
        }
        rows.append(row)

    practice = [r for r in rows if r["set"] != "gaia"]
    baseline = _pick_baseline(practice)
    for r in rows:
        same = bool(baseline) and r is not baseline and r["set"] == baseline["set"]
        r["is_baseline"] = r is baseline
        r["delta_pp"] = (
            (r["accuracy"] - baseline["accuracy"]) * 100
            if same and r["accuracy"] is not None and baseline["accuracy"] is not None
            else None
        )
        r["separable"] = (
            (r["ci95"][0] > baseline["ci95"][1] or r["ci95"][1] < baseline["ci95"][0])
            if same and r["ci95"] and baseline["ci95"]
            else None
        )
        r["delta_tokens_pct"] = (
            (r["tokens_per_item"] / baseline["tokens_per_item"] - 1) * 100
            if same and r["tokens_per_item"] and baseline["tokens_per_item"]
            else None
        )
    rows.sort(key=lambda r: (r["set"] == "gaia", not r["is_baseline"], r["label"]))

    for r in runs:
        r["label"] = _config_label(r.pop("key"), multi_model, r["name"])
        r.pop("items")
    return {"runs": runs, "groups": rows, "baseline": baseline["label"] if baseline else None}


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pick_baseline(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    named = [r for r in rows if r.get("name") == "baseline"]
    if named:
        return named[0]
    plain = [r for r in rows if r["arch"] == "single" and not r["disabled_tools"] and not r["verify"]]
    pool = plain or rows
    return max(pool, key=lambda r: (r["runs"], r["accuracy"] is not None))


def _by_skill(grs: list[dict]) -> list[dict]:
    """Accuracy per practice-set skill tag, pooled over the group's runs."""
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in grs:
        for it in r["items"]:
            c = _is_correct(it)
            if c is None:
                continue
            for sk in it.get("skills") or []:
                acc[str(sk)][0] += int(c)
                acc[str(sk)][1] += 1
    return [
        {"skill": k, "correct": v[0], "n": v[1], "accuracy": v[0] / v[1]}
        for k, v in sorted(acc.items(), key=lambda kv: (-kv[1][1], kv[0]))
    ]


def _passk(grs: list[dict]) -> dict | None:
    """pass@k and pass^k across repeated runs of one config, per item, averaged."""
    if len(grs) < 2:
        return None
    per_item: dict[str, list[bool]] = defaultdict(list)
    for r in grs:
        for it in r["items"]:
            c = _is_correct(it)
            if c is not None and it.get("id") is not None:
                per_item[str(it["id"])].append(c)
    if not per_item:
        return None
    n_runs = max(len(v) for v in per_item.values())
    ks, at, hat = [], [], []
    for k in range(1, n_runs + 1):
        elig = [(len(v), sum(v)) for v in per_item.values() if len(v) >= k]
        if not elig:
            break
        ks.append(k)
        at.append(sum(pass_at_k(n, c, k) for n, c in elig) / len(elig))
        hat.append(sum(pass_hat_k(n, c, k) for n, c in elig) / len(elig))
    return {"k": ks, "pass_at_k": at, "pass_hat_k": hat, "items": len(per_item), "runs": n_runs}


# --------------------------------------------------------------------------- errors / submissions


def build_errors(raw: Any, questions: list[dict]) -> dict | None:
    if not isinstance(raw, dict):
        return None
    qmap = {q["task_id"]: q for q in questions}
    items = []
    for it in raw.get("items") or []:
        if not isinstance(it, dict):
            continue
        tags = it.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tid = str(it.get("task_id") or "")
        items.append(
            {
                "task_id": tid,
                "n": qmap.get(tid, {}).get("n"),
                "question": qmap.get(tid, {}).get("question", ""),
                "tags": [str(t) for t in tags],
                "evidence": clip(it.get("evidence"), 1500),
            }
        )
    counts = raw.get("counts")
    if not isinstance(counts, dict) or not counts:
        counts = Counter(t for i in items for t in i["tags"])
    counts = dict(sorted(((str(k), as_int(v)) for k, v in counts.items()), key=lambda kv: (-kv[1], kv[0])))
    descs = raw.get("tag_descriptions")
    clean = raw.get("clean_tasks")
    # Items with no tags are clean; keep the list to the ones with something to say.
    items.sort(key=lambda i: (not i["tags"], i["n"] if i["n"] is not None else 10**6))
    return {
        "run_id": raw.get("run_id"),
        "items": [i for i in items if i["tags"]],
        "counts": counts,
        "descriptions": {str(k): str(v) for k, v in descs.items()} if isinstance(descs, dict) else {},
        "note": raw.get("note"),
        "n_tasks": as_int(raw.get("n_tasks")) or len(items) or None,
        "clean": len(clean) if isinstance(clean, list) else sum(1 for i in items if not i["tags"]) or None,
    }


def build_submission(subs: list[dict]) -> dict:
    hist = []
    for s in subs:
        # submit.py may log {submitted_at, run_id, response: {...}} or the API response flat.
        resp = s.get("response") if isinstance(s.get("response"), dict) else s
        score = num(resp.get("score"))
        if score is None and resp.get("error"):
            continue  # a failed POST is not a score
        hist.append(
            {
                "submitted_at": s.get("submitted_at") or s.get("ts") or s.get("timestamp"),
                "run_id": s.get("run_id"),
                "score": score,
                "correct_count": resp.get("correct_count"),
                "total_attempted": resp.get("total_attempted"),
                "message": clip(resp.get("message"), 600),
            }
        )
    return {"latest": hist[-1] if hist else None, "history": hist, "pass_mark": PASS_MARK}


# --------------------------------------------------------------------------- public redaction (GAIA)


def _ident(x: Any) -> str | None:
    """An identifier-shaped string (tool, agent or exception name), else None."""
    return x if isinstance(x, str) and _IDENT_RE.match(x) else None


def short_ids(task_ids: list[str], n: int = 8) -> dict[str, str]:
    """task_id -> its first n characters, lengthened only where two ids would collide."""
    ids = list(dict.fromkeys(str(t) for t in task_ids))
    out: dict[str, str] = {}
    for tid in ids:
        k = min(n, len(tid))
        while k < len(tid) and any(o != tid and o[:k] == tid[:k] for o in ids):
            k += 1
        out[tid] = tid[:k]
    return out


def skeleton_step(st: dict, depth: int = 0) -> dict:
    """A step with every free-text field removed: type, tools, tokens, timing, error type."""
    subs = []
    for run in st.get("sub_agents") or []:
        subs.append(
            {
                "agent": _ident(run.get("agent")) or "sub-agent",
                "seconds": run.get("seconds"),
                "partial": bool(run.get("partial")),
                "steps": [skeleton_step(x, depth + 1) for x in run.get("steps") or []],
                "elided_steps": as_int(run.get("elided_steps")),
            }
        )
    err_type = _ident(st.get("error_type"))
    return {
        "index": as_int(st.get("index")),
        "step_number": st.get("step_number"),
        "type": st.get("type") if st.get("type") in {"task", "planning", "action", "final"} else "action",
        "agent": _ident(st.get("agent")),
        "is_final_answer": bool(st.get("is_final_answer")),
        "tools": [t for t in (st.get("tools") or []) if _ident(t)],
        "error": bool(st.get("error")),  # a flag only; the message can quote the task
        "error_type": err_type,
        "images": st.get("images"),
        "input_tokens": as_int(st.get("input_tokens")),
        "output_tokens": as_int(st.get("output_tokens")),
        "tokens_known": bool(st.get("tokens_known")),
        "duration_s": num(st.get("duration_s")),
        "sub_agents": subs,
    }


def skeleton_trace(tr: dict, short: str) -> dict:
    totals = dict(tr.get("totals") or {})
    totals["tool_calls_by_name"] = {k: v for k, v in (totals.get("tool_calls_by_name") or {}).items() if _ident(k)}
    return {
        "task_id": short,
        "redacted": True,
        "run_id": tr.get("run_id"),
        "model": tr.get("model"),
        "provider": tr.get("provider"),
        "arch": tr.get("arch"),
        "started_at": tr.get("started_at"),
        "seconds": num(tr.get("seconds")),
        "attachment": tr.get("attachment"),
        "error_type": _ident(tr.get("error_type")),
        "agents": [a for a in (tr.get("agents") or []) if _ident(a)],
        "steps": [skeleton_step(st) for st in tr.get("steps") or []],
        "totals": totals,
    }


def redact_gaia(questions: list[dict], traces: dict, errors: dict | None) -> tuple[list[dict], dict, dict | None]:
    """The public view of the GAIA run: no question, answer, output, code or observation text."""
    sid = short_ids([q["task_id"] for q in questions] + list(traces))
    tags_by: dict[str, list[str]] = {}
    for it in (errors or {}).get("items") or []:
        tags_by[it["task_id"]] = [t for t in it.get("tags") or [] if _TAG_RE.match(t)]
    pub_q = []
    for q in questions:
        tid = q["task_id"]
        tr = traces.get(tid) or {}
        pub_q.append(
            {
                "n": q["n"],
                "task_id": sid[tid],
                "attachment": q.get("attachment"),
                "status": q["status"],
                "error_type": _ident((tr.get("error_type") or "")) if q["status"] in ("error", "timeout") else None,
                "steps": q["steps"],
                "model_calls": q.get("model_calls"),
                "seconds": q["seconds"],
                "input_tokens": q["input_tokens"],
                "output_tokens": q["output_tokens"],
                "tokens": q["tokens"],
                "tools": {k: v for k, v in q["tools"].items() if _ident(k)},
                "tags": tags_by.get(tid, []),
                "has_trace": q["has_trace"],
                "run_id": q.get("run_id"),
            }
        )
    pub_t = {sid[t]: skeleton_trace(tr, sid[t]) for t, tr in traces.items()}
    pub_e = None
    if errors is not None:
        counts = {k: v for k, v in (errors.get("counts") or {}).items() if _TAG_RE.match(k)}
        pub_e = {
            "run_id": errors.get("run_id"),
            "items": [
                {"task_id": sid.get(it["task_id"], it["task_id"][:8]), "n": it.get("n"), "tags": tags_by.get(it["task_id"], [])}
                for it in errors.get("items") or []
            ],
            "counts": counts,
            "descriptions": {k: v for k, v in (errors.get("descriptions") or {}).items() if _TAG_RE.match(k)},
            "note": errors.get("note"),
            "n_tasks": errors.get("n_tasks"),
            "clean": errors.get("clean"),
            "redacted": True,
        }
    return pub_q, pub_t, pub_e


# --------------------------------------------------------------------------- practice set (ours, shown in full)


def load_practice_tasks(root: Path) -> dict[str, dict]:
    out = {}
    for t in read_jsonl(root / "evals" / "practice" / "tasks.jsonl"):
        if t.get("id"):
            out[str(t["id"])] = t
    return out


def split_traces(traces: dict, practice_runs: set[str], practice_tasks: set[str]) -> tuple[dict, dict]:
    """(GAIA-side traces, practice traces). A trace belongs to the practice set when its run is
    a practice eval run, or its task id is a practice task id (an eval run whose results doc is
    missing). Practice traces never reach the GAIA scoreboard."""
    gaia: dict = {}
    prac: dict = {}
    for rid, by_task in traces.items():
        for tid, tr in by_task.items():
            if rid in practice_runs:
                prac.setdefault(rid, {})[tid] = tr
            elif tid in practice_tasks:
                continue  # orphan practice trace with no scored result: shown nowhere
            else:
                gaia.setdefault(rid, {})[tid] = tr
    return gaia, prac


def build_practice(eval_raw: list[dict], evals: dict | None, prac_traces: dict, tasks: dict[str, dict]) -> dict | None:
    if not eval_raw:
        return None
    run_meta = {r["run_id"]: r for r in (evals or {}).get("runs") or []}
    group_of: dict[str, str] = {}
    for g in (evals or {}).get("groups") or []:
        for rid in g.get("run_ids") or []:
            group_of[rid] = g["label"]
    runs, items = [], {}
    for doc in sorted(eval_raw, key=lambda d: str(d.get("run_id"))):
        rid = str(doc["run_id"])
        trs = prac_traces.get(rid, {})
        rows = []
        for it in doc.get("items") or []:
            if not isinstance(it, dict) or it.get("id") is None:
                continue
            tid = str(it["id"])
            tr = trs.get(tid)
            totals = (tr or {}).get("totals") or {}
            rows.append(
                {
                    "id": tid,
                    "answer": None if it.get("answer") is None else str(it.get("answer")),
                    "gold": None if it.get("gold") is None else str(it.get("gold")),
                    "correct": _is_correct(it),
                    "correct_strict": _is_correct(it, "correct_strict"),
                    "skipped": bool(it.get("skipped")),
                    "format": it.get("format"),
                    "skills": [str(x) for x in it.get("skills") or []],
                    "steps": as_int(it.get("steps")) or as_int(totals.get("steps")) or None,
                    "model_calls": as_int(it.get("model_calls")) or as_int(totals.get("model_calls")) or None,
                    "seconds": num(it.get("seconds"), num((tr or {}).get("seconds"))),
                    "tokens": (as_int(it.get("input_tokens")) + as_int(it.get("output_tokens")))
                    or (as_int(totals.get("input_tokens")) + as_int(totals.get("output_tokens")))
                    or None,
                    "tools": dict(sorted((totals.get("tool_calls_by_name") or {}).items(), key=lambda kv: -kv[1])),
                    "error": clip(it.get("error"), 600),
                    "timed_out": bool(it.get("timed_out")),
                    "has_trace": tr is not None,
                }
            )
        items[rid] = rows
        m = run_meta.get(rid, {})
        runs.append(
            {
                "run_id": rid,
                "label": m.get("label") or rid,
                "group": group_of.get(rid),
                "config": doc.get("config") or {},
                "accuracy": m.get("accuracy"),
                "accuracy_strict": m.get("accuracy_strict"),
                "correct": m.get("correct"),
                "judged": m.get("judged"),
                "n": m.get("n"),
                "skipped": m.get("skipped"),
                "started_at": doc.get("started_at"),
                "traces": sorted(trs),
            }
        )
    baseline = (evals or {}).get("baseline")
    pool = [r for r in runs if r["group"] == baseline and r["traces"]] or [r for r in runs if r["traces"]] or runs
    default = max(pool, key=lambda r: (len(r["traces"]), r["run_id"]))["run_id"] if pool else None
    used = {it["id"] for rows in items.values() for it in rows}
    task_info = {}
    for tid in sorted(used):
        t = tasks.get(tid, {})
        tr = next((prac_traces[r][tid] for r in sorted(prac_traces) if tid in prac_traces[r]), None)
        task_info[tid] = {
            "question": t.get("question") or (tr or {}).get("question") or "",
            "attachment": attachment_type(t.get("file")) or (tr or {}).get("attachment"),
            "format": t.get("format"),
            "skills": [str(x) for x in t.get("skills") or []],
            "trap": (t.get("source") or {}).get("derivation") if isinstance(t.get("source"), dict) else None,
            "source_type": (t.get("source") or {}).get("type") if isinstance(t.get("source"), dict) else None,
        }
    return {"runs": runs, "items": items, "tasks": task_info, "default_run": default, "traces": {}, "trace_base": "traces/"}


def inline_practice_traces(practice: dict | None, prac_traces: dict, budget: int = INLINE_TRACE_BUDGET) -> None:
    """Inline practice traces into the page, default run first, until the byte budget is spent.
    Everything else is still written to <out>/traces/ and fetched by the page on demand."""
    if not practice:
        return
    order = [practice["default_run"]] + [r["run_id"] for r in practice["runs"] if r["run_id"] != practice["default_run"]]
    used = 0
    for rid in order:
        for tid, tr in sorted(prac_traces.get(rid, {}).items()):
            size = len(json.dumps(tr, ensure_ascii=False, default=str))
            if used + size > budget and used:
                continue
            practice["traces"].setdefault(rid, {})[tid] = tr
            used += size
    practice["inlined_bytes"] = used


# --------------------------------------------------------------------------- usage (runs/usage.jsonl)


def build_usage(rows: list[dict], gaia_run: str | None, practice_runs: set[str]) -> dict | None:
    """Aggregate the provider layer's per-request log. Only counts, models and error kinds are
    read: the log holds no prompts, and its error messages are never copied into the page."""
    if not rows:
        return None

    def bucket() -> dict:
        return {"requests": 0, "ok": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "retries": 0,
                "fallback_served": 0, "latencies": [], "waited_s": 0.0}

    tot = bucket()
    by_model: dict[str, dict] = defaultdict(bucket)
    by_scope: dict[str, dict] = defaultdict(bucket)
    by_kind: Counter = Counter()
    attempt_errors: dict[str, Counter] = defaultdict(Counter)
    days: Counter = Counter()
    first = last = None
    for r in rows:
        rid = str(r.get("run_id") or "")
        scope = "gaia" if gaia_run and rid == gaia_run else "practice" if rid in practice_runs else "other"
        served = str(r.get("served_model") or r.get("model") or r.get("requested_model") or "?")
        ok = r.get("status") == "ok"
        lat = num(r.get("latency_s"))
        for b in (tot, by_model[served], by_scope[scope]):
            b["requests"] += 1
            b["ok"] += ok
            b["failed"] += not ok
            b["input_tokens"] += as_int(r.get("input_tokens"))
            b["output_tokens"] += as_int(r.get("output_tokens"))
            b["retries"] += as_int(r.get("retries"))
            b["fallback_served"] += bool(as_int(r.get("fallback_index"))) and ok
            b["waited_s"] += (num(r.get("waited_limiter_s"), 0) or 0) + (num(r.get("waited_backoff_s"), 0) or 0)
            if lat is not None:
                b["latencies"].append(lat)
        by_kind[str(r.get("kind") or "text")] += 1
        for e in r.get("errors") or []:
            if isinstance(e, dict):
                kind = e.get("kind") if isinstance(e.get("kind"), str) and _TAG_RE.match(e["kind"]) else "error"
                status = e.get("status") if isinstance(e.get("status"), int) else None
                attempt_errors[str(e.get("model") or served)][kind + (f" {status}" if status else "")] += 1
        ts = str(r.get("ts") or "")
        if ts:
            days[ts[:10]] += 1
            first = ts if first is None or ts < first else first
            last = ts if last is None or ts > last else last

    def finish(b: dict) -> dict:
        lats = b.pop("latencies")
        b["latency_median"] = percentile(lats, 0.5)
        b["latency_p90"] = percentile(lats, 0.9)
        b["waited_s"] = round(b["waited_s"], 1)
        return b

    models = []
    names = set(by_model) | set(attempt_errors)
    for name in names:
        b = finish(by_model[name]) if name in by_model else finish(bucket())
        b["model"] = name
        b["free"] = name.endswith(":free")
        b["attempt_errors"] = dict(attempt_errors.get(name, Counter()).most_common())
        models.append(b)
    models.sort(key=lambda m: (-m["requests"], -sum(m["attempt_errors"].values()), m["model"]))
    served_names = [m["model"] for m in models if m["requests"]]
    return {
        "totals": finish(tot),
        "models": models,
        "scopes": {k: finish(v) for k, v in by_scope.items()},
        "kinds": dict(by_kind.most_common()),
        "days": dict(sorted(days.items())),
        "first": first,
        "last": last,
        "all_free": all(n.endswith(":free") for n in served_names) if served_names else None,
        "non_free_models": [n for n in served_names if not n.endswith(":free")],
    }


# --------------------------------------------------------------------------- assemble


def collect(root: Path, public: bool = True) -> dict:
    """All the page's data. public=True strips every GAIA free-text field (see POLICY)."""
    root = Path(root)
    runs = root / "runs"
    answers = read_jsonl(runs / "answers.jsonl")
    subs = read_jsonl(runs / "submissions.jsonl")
    all_traces = load_traces(runs, set(KNOWN_TOOLS))

    res_dir = root / "evals" / "results"
    eval_files = sorted(res_dir.glob("*.json")) if res_dir.is_dir() else []
    eval_raw = [d for d in (read_json(p) for p in eval_files if not p.name.startswith("ablation_")) if is_eval_run(d)]
    ablation = load_ablation_names(sorted(res_dir.glob("ablation_*.json"))) if res_dir.is_dir() else {}
    errors_raw = read_json(runs / "error_analysis.json")
    tasks = load_practice_tasks(root)

    practice_runs = {str(d["run_id"]) for d in eval_raw if str(d.get("set") or "practice") != "gaia"}
    traces, prac_traces = split_traces(all_traces, practice_runs, set(tasks))
    # Answer lines from the practice harness (if any) are not GAIA questions either.
    answers = [a for a in answers if str(a.get("run_id") or "") not in practice_runs]
    run_id = pick_run(answers, traces, subs)
    questions, used_traces = build_questions(answers, traces, run_id)

    # Model / arch: the traces know best; answers' provider_info is the fallback.
    meta_src: dict = {}
    for tr in used_traces.values():
        for k in ("model", "provider", "arch"):
            if tr.get(k) and not meta_src.get(k):
                meta_src[k] = tr[k]
    for a in reversed(answers):
        if a.get("run_id") in (run_id, None):
            for k, v in _provider_fields(a.get("provider_info")).items():
                if v not in (None, "") and meta_src.get(k) in (None, ""):
                    meta_src[k] = v

    # The failure taxonomy describes the GAIA run on this page; an analysis of another run
    # (a practice or smoke run) would mislabel those tasks as course questions.
    if isinstance(errors_raw, dict) and errors_raw.get("run_id") and errors_raw["run_id"] != run_id:
        warn(f"error_analysis.json is for run {errors_raw['run_id']!r}, not the GAIA run {run_id!r}; not shown")
        errors_raw = None
    errors = build_errors(errors_raw, questions)
    analytics = build_analytics(questions)
    if public:
        questions, used_traces, errors = redact_gaia(questions, used_traces, errors)

    evals = build_evals(eval_raw, ablation) if eval_raw else None
    practice = build_practice(eval_raw, evals, prac_traces, tasks)
    usage = build_usage(read_jsonl(runs / "usage.jsonl"), run_id, practice_runs)

    fixture = (root / "FIXTURE").exists()
    return {
        "meta": {
            "project": PROJECT,
            "pitch": PITCH,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fixture": fixture,
            "public": bool(public),
            "policy": POLICY,
            "run_id": run_id,
            "model": meta_src.get("model"),
            "provider": meta_src.get("provider"),
            "arch": meta_src.get("arch"),
            "vision_model": meta_src.get("vision_model"),
            "verify": meta_src.get("verify"),
            "known_tools": KNOWN_TOOLS,
            "sources": {
                "answers": bool(answers),
                "traces": bool(traces),
                "submissions": bool(subs),
                "error_analysis": errors_raw is not None,
                "evals": bool(eval_raw),
                "practice_traces": bool(prac_traces),
                "usage": usage is not None,
            },
        },
        "submission": build_submission(subs),
        "questions": questions,
        "traces": used_traces,
        "analytics": analytics,
        "errors": errors,
        "evals": evals,
        "practice": practice,
        "usage": usage,
        "_practice_traces": prac_traces,  # written to <out>/traces/, popped before inlining
    }


def _scrub_home(s: str) -> str:
    """Published pages show "~" instead of the builder's home directory (no local usernames/paths)."""
    home = str(Path.home())
    return s.replace(home, "~") if home and home != "/" else s


def inline_json(data: Any) -> str:
    """JSON safe to embed inside <script type="application/json">."""
    s = _scrub_home(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str))
    return (
        s.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def render(data: dict) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    meta = data["meta"]
    sub = data["submission"]["latest"]
    desc = PITCH
    if sub and sub.get("score") is not None and not meta["fixture"]:
        desc = f"Official score {sub['score']:g}% on the course's 20 GAIA Level-1 questions. " + PITCH
    replacements = {
        "{{TITLE}}": html.escape(("[FIXTURE] " if meta["fixture"] else "") + PROJECT),
        "{{DESCRIPTION}}": html.escape(desc),
        "{{PROJECT}}": html.escape(PROJECT),
        "{{PITCH}}": html.escape(PITCH),
        "{{GENERATED_AT}}": html.escape(meta["generated_at"]),
        "{{DATA}}": inline_json(data),
        "{{BODY_CLASS}}": " ".join(
            c for c in ("is-fixture" if meta["fixture"] else "", "" if meta["public"] else "is-private") if c
        ),
    }
    for k, v in replacements.items():
        tpl = tpl.replace(k, v)
    # Cache-bust the assets so a republished site never pairs new HTML with stale JS/CSS.
    for name in ("style.css", "app.js"):
        digest = hashlib.sha1((ASSETS / name).read_bytes()).hexdigest()[:10]
        tpl = tpl.replace(f'"assets/{name}"', f'"assets/{name}?v={digest}"')
    return tpl


def _safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x)).strip("._") or "_"


def write_practice_traces(out: Path, prac_traces: dict, practice: dict | None) -> int:
    """Write every practice trace to <out>/traces/<run>/<task>.json for on-demand loading.
    Only the practice set is ever written here; GAIA traces never become files."""
    tdir = out / "traces"
    if tdir.exists():
        if (tdir / TRACE_MARKER).exists():
            shutil.rmtree(tdir)  # our own previous output; regenerate from scratch
        else:
            warn(f"{tdir} exists and was not written by this script; leaving it and writing alongside")
    n = 0
    files: dict[str, dict[str, str]] = {}
    for rid, by_task in sorted(prac_traces.items()):
        for tid, tr in sorted(by_task.items()):
            rel = f"{_safe_name(rid)}/{_safe_name(tid)}.json"
            path = tdir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_scrub_home(json.dumps(tr, ensure_ascii=False, separators=(",", ":"), default=str)), encoding="utf-8")
            files.setdefault(rid, {})[tid] = rel
            n += 1
    if n:
        (tdir / TRACE_MARKER).write_text("Regenerated by report/build_report.py on every build.\n", encoding="utf-8")
    if practice is not None:
        practice["trace_files"] = files
    return n


def build(root: Path | str = ".", out: Path | str = "docs", public: bool = True) -> tuple[Path, dict]:
    """Write the site; returns (index.html path, the data inlined into it)."""
    root, out = Path(root), Path(out)
    data = collect(root, public=public)
    prac_traces = data.pop("_practice_traces")
    out.mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    for f in ASSETS.iterdir():
        if f.is_file():
            shutil.copy2(f, out / "assets" / f.name)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    write_practice_traces(out, prac_traces, data["practice"])
    inline_practice_traces(data["practice"], prac_traces)
    index = out / "index.html"
    index.write_text(render(data), encoding="utf-8")
    return index, data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".", help="project root containing runs/ and evals/ (default: .)")
    ap.add_argument("--out", default="docs", help="output directory (default: docs)")
    ap.add_argument(
        "--private",
        action="store_true",
        help="include GAIA question/answer/trace text (local use only; refused for docs/)",
    )
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    docs = (HERE.parent / "docs").resolve()
    if (root / "FIXTURE").exists() and out.resolve() == docs:
        print("refusing to publish FIXTURE data into docs/; pass a different --out", file=sys.stderr)
        return 2
    if args.private and out.resolve() == docs:
        print("refusing a --private build into docs/ (it would publish GAIA text); pass --out elsewhere",
              file=sys.stderr)
        return 2
    index, data = build(root, out, public=not args.private)
    q = data["questions"]
    pr = data["practice"] or {}
    print(
        f"wrote {index}  ({'PUBLIC' if data['meta']['public'] else 'PRIVATE'}; {len(q)} GAIA questions, "
        f"{len(data['traces'])} GAIA traces, "
        f"{sum(len(v) for v in (pr.get('trace_files') or {}).values())} practice traces, "
        f"submission={'yes' if data['submission']['latest'] else 'no'}, "
        f"evals={len((data['evals'] or {}).get('runs', []))} runs"
        f"{', FIXTURE' if data['meta']['fixture'] else ''})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
