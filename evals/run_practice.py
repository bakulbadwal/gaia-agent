"""Run the agent over the practice set, score it two ways, and write evals/results/<run_id>.json.

    .venv/bin/python -m evals.run_practice                       # whole set, current env config
    .venv/bin/python -m evals.run_practice --only p01,w03         # a subset (ids or prefixes)
    .venv/bin/python -m evals.run_practice --limit 4 --max-requests 30
    .venv/bin/python -m evals.run_practice --repeat 3             # 3 runs -> feed to evals.passk
    .venv/bin/python -m evals.run_practice --run-id X --resume    # continue an interrupted run

Configuration comes from the same env contract the agent honours (GAIA_PROVIDER, GAIA_MODEL_ID,
GAIA_VISION_MODEL_ID, GAIA_ARCH, GAIA_DISABLE_TOOLS, GAIA_VERIFY); the CLI flags below just set
those variables for this process. GAIA_RUN_ID is set to the run id, so the agent's traces land in
runs/traces/<run_id>/ and its model calls in runs/usage.jsonl under the same id.

Budget guard (--max-requests): free tiers cap requests per day (OpenRouter free: 50/day without
credits), and one practice task costs ~3-12 model calls. Before each task the runner counts the
calls this run has made so far (runs/usage.jsonl rows tagged with the run id; falling back to trace
totals, then to step counts), stops when the budget is spent, and caps the next task's max_steps
at what is left. A provider-side quota error also stops the run instead of being scored as a
wrong answer; the untried tasks are recorded as skipped and --resume picks them up later.

Isolation: the attachment is copied to runs/practice_files/<task_id>/ before the agent sees it,
so the agent is never pointed inside evals/ (where the gold answers live).
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.scorer import exact_match, gaia_match  # noqa: E402

EVALS_DIR = ROOT / "evals"
TASKS_PATH = EVALS_DIR / "practice" / "tasks.jsonl"
RESULTS_DIR = EVALS_DIR / "results"
RUNS_DIR = ROOT / "runs"
STAGING_DIR = RUNS_DIR / "practice_files"

DEFAULT_MAX_STEPS = 12
MIN_TASK_BUDGET = 3  # do not start a task with fewer model calls than this left
_QUOTA_RE = re.compile(r"BudgetExceeded|RemoteQuotaExhausted|request cap reached|daily allowance exhausted", re.I)

SolveFn = Callable[..., dict]


# --------------------------------------------------------------------------- tasks


def load_tasks(path: Path = TASKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def select_tasks(tasks: list[dict], only: str | list[str] | None = None, limit: int | None = None) -> list[dict]:
    """Filter by id or id prefix (comma list), then truncate to `limit`."""
    if only:
        wanted = [w.strip() for w in (only.split(",") if isinstance(only, str) else only) if w.strip()]
        tasks = [t for t in tasks if any(t["id"] == w or t["id"].startswith(w) for w in wanted)]
    if limit is not None:
        tasks = tasks[: max(0, limit)]
    return tasks


def stage_file(task: dict, root: Path = ROOT, staging: Path = STAGING_DIR) -> str | None:
    """Copy a task's attachment outside evals/ and return its absolute path."""
    if not task.get("file"):
        return None
    src = (root / task["file"]).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"practice file missing for {task['id']}: {src}")
    dest = staging / task["id"] / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dest)
    return str(dest.resolve())


# --------------------------------------------------------------------------- config / usage


def current_config() -> dict:
    """The config block of the results schema, read from the env contract (no secrets)."""
    info: dict = {}
    try:
        from gaia_agent import config as gcfg  # lazy: keeps this module importable without deps

        info = gcfg.provider_info()
    except Exception:
        info = {}
    disabled = [t.strip() for t in os.environ.get("GAIA_DISABLE_TOOLS", "").split(",") if t.strip()]
    return {
        "provider": os.environ.get("GAIA_PROVIDER") or info.get("provider"),
        "model": os.environ.get("GAIA_MODEL_ID") or info.get("model_id"),
        "vision_model": os.environ.get("GAIA_VISION_MODEL_ID") or info.get("vision_model_id"),
        "arch": os.environ.get("GAIA_ARCH", "single") or "single",
        "disabled_tools": sorted(disabled),
        "verify": os.environ.get("GAIA_VERIFY", "0") in ("1", "true", "yes"),
    }


def _slug(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s or "default")).strip("-").lower()[:40] or "default"


def default_run_id(cfg: dict) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    model = str(cfg.get("model") or "model").split("/")[-1]
    return f"practice-{_slug(model)}-{cfg.get('arch') or 'single'}-{stamp}"


def _usage_log_calls(run_id: str) -> int | None:
    """Model calls logged in runs/usage.jsonl for this run id (None if that log is unavailable)."""
    try:
        from gaia_agent.providers import summarize_usage

        return int(summarize_usage(run_id=run_id).get("calls", 0))
    except Exception:
        return None


def model_calls_so_far(run_id: str, items: list[dict], baseline: int = 0) -> int:
    """Model calls made by THIS invocation: usage log minus its value at start, else per-item counts."""
    logged = _usage_log_calls(run_id)
    if logged:
        return max(0, logged - baseline)
    return sum(int(it.get("model_calls") or it.get("steps") or 0) for it in items if not it.get("resumed"))


def _runs_root() -> Path:
    try:
        from gaia_agent.config import runs_dir

        return runs_dir()
    except Exception:
        return RUNS_DIR


def _trace_totals(run_id: str, task_id: str) -> dict:
    root = _runs_root()
    try:
        from gaia_agent.trace import trace_path

        p = trace_path(run_id, task_id, root)
    except Exception:
        p = root / "traces" / run_id / f"{task_id}.json"
    try:
        return json.loads(Path(p).read_text(encoding="utf-8")).get("totals") or {}
    except Exception:
        return {}


def _usage(result: dict, run_id: str, task_id: str) -> dict:
    """Tokens and model calls for one task, from the result, else from its trace file."""
    tu = result.get("token_usage") or {}
    inp = result.get("input_tokens", tu.get("input_tokens"))
    out = result.get("output_tokens", tu.get("output_tokens"))
    calls = result.get("model_calls")
    if inp is None or out is None or calls is None:
        totals = _trace_totals(run_id, task_id)
        inp = inp if inp is not None else totals.get("input_tokens")
        out = out if out is not None else totals.get("output_tokens")
        calls = calls if calls is not None else totals.get("model_calls")
    return {"input_tokens": inp, "output_tokens": out, "model_calls": calls}


# --------------------------------------------------------------------------- solving


def default_solve(question: str, task_id: str, file_path: str | None, max_steps: int, stop_event) -> dict:
    """Adapter onto gaia_agent.agent.solve that attaches a LOCAL file (not a course task file)."""
    from gaia_agent.agent import solve

    params = inspect.signature(solve).parameters
    kwargs: dict[str, Any] = {}
    if "max_steps" in params:
        kwargs["max_steps"] = max_steps
    if "stop_event" in params:
        kwargs["stop_event"] = stop_event
    if "run_id" in params:
        kwargs["run_id"] = os.environ.get("GAIA_RUN_ID") or None
    if "file_path" in params:
        kwargs["file_path"] = file_path
        q = question
    else:
        # Same wording gaia_agent.agent.build_task uses for a downloaded course file.
        q = question + (f"\n\nAttached file: {file_path}" if file_path else "")
    # file_name=None: the agent must not try to download a practice file from the course API.
    return solve(q, task_id, None, **kwargs)


def _verifier_summary(v: dict | None) -> dict | None:
    if not v:
        return None
    return {k: v.get(k) for k in ("candidate", "proposed", "accepted", "changed", "reason", "error")}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion (better than normal approx at small n)."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def summarize(items: list[dict]) -> dict:
    done = [it for it in items if not it.get("skipped")]
    n = len(done)

    def mean(key: str) -> float | None:
        vals = [it[key] for it in done if isinstance(it.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    correct = sum(1 for it in done if it.get("correct"))
    strict = sum(1 for it in done if it.get("correct_strict"))
    toks = [(it.get("input_tokens") or 0) + (it.get("output_tokens") or 0) for it in done]
    tok_known = any(it.get("input_tokens") is not None for it in done)
    by: dict[str, dict] = {"skill": {}, "format": {}}
    for it in done:
        for kind, keys in (("skill", it.get("skills") or []), ("format", [it.get("format")])):
            for k in keys:
                if k is None:
                    continue
                b = by[kind].setdefault(k, {"n": 0, "correct": 0})
                b["n"] += 1
                b["correct"] += bool(it.get("correct"))
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else None,
        "accuracy_ci95": wilson_ci(correct, n),
        "correct_strict": strict,
        "accuracy_strict": round(strict / n, 4) if n else None,
        "format_gap": correct - strict,
        "errors": sum(1 for it in done if it.get("error")),
        "skipped": len(items) - n,
        "mean_seconds": mean("seconds"),
        "mean_steps": mean("steps"),
        "total_tokens": sum(toks) if tok_known else None,
        "model_calls": sum(int(it.get("model_calls") or 0) for it in done) or None,
        "by_skill": by["skill"],
        "by_format": by["format"],
    }


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(
    tasks: list[dict],
    *,
    run_id: str,
    solve_fn: SolveFn | None = None,
    max_requests: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout: float | None = None,
    resume: bool = False,
    results_dir: Path = RESULTS_DIR,
    config: dict | None = None,
    root: Path = ROOT,
    staging: Path = STAGING_DIR,
    log: Callable[[str], None] | None = print,
) -> dict:
    """Solve each task, score it, and keep evals/results/<run_id>.json current after every task."""
    solve_fn = solve_fn or default_solve
    log = log or (lambda _msg: None)
    os.environ["GAIA_RUN_ID"] = run_id
    out_path = Path(results_dir) / f"{run_id}.json"
    cfg = config if config is not None else current_config()

    prior: dict[str, dict] = {}
    started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if resume and out_path.is_file():
        old = json.loads(out_path.read_text(encoding="utf-8"))
        started_at = old.get("started_at", started_at)
        prior = {it["id"]: it for it in old.get("items", []) if not it.get("skipped")}

    doc: dict = {
        "run_id": run_id,
        "config": cfg,
        "set": "practice",
        "started_at": started_at,
        "finished_at": None,
        "budget": {"max_requests": max_requests, "max_steps": max_steps, "stopped": None},
        "items": [],
        "summary": {},
    }
    stop_reason: str | None = None
    baseline = _usage_log_calls(run_id) or 0  # resumed runs: the budget covers new calls only
    for task in tasks:
        tid = task["id"]
        base = {"id": tid, "gold": task["gold"], "format": task.get("format"), "skills": task.get("skills", [])}
        if tid in prior:
            doc["items"].append({**prior[tid], "resumed": True})
            continue
        if stop_reason is None and max_requests is not None:
            left = max_requests - model_calls_so_far(run_id, doc["items"], baseline)
            if left < MIN_TASK_BUDGET:
                stop_reason = f"request budget spent ({max_requests - left}/{max_requests} calls)"
        if stop_reason:
            doc["items"].append({**base, "answer": None, "correct": None, "correct_strict": None,
                                 "skipped": True, "error": stop_reason})
            continue

        steps_cap = max_steps
        if max_requests is not None:
            steps_cap = max(1, min(max_steps, max_requests - model_calls_so_far(run_id, doc["items"], baseline) - 1))
        stop_event = threading.Event()
        timer = threading.Timer(timeout, stop_event.set) if timeout else None
        t0 = time.time()
        try:
            file_path = stage_file(task, root, staging)
            if timer:
                timer.start()
            res = solve_fn(task["question"], tid, file_path, steps_cap, stop_event) or {}
        except Exception as e:  # solve() should never raise; a stub or staging error might
            res = {"answer": "", "error": f"{type(e).__name__}: {e}"}
        finally:
            if timer:
                timer.cancel()
        seconds = res.get("seconds", round(time.time() - t0, 1))
        err = res.get("error")
        if err and _QUOTA_RE.search(str(err)) and not res.get("answer"):
            stop_reason = f"provider quota: {str(err)[:200]}"
            doc["items"].append({**base, "answer": None, "correct": None, "correct_strict": None,
                                 "skipped": True, "error": stop_reason, "seconds": seconds})
            log(f"  {tid:<28} STOP  {stop_reason}")
            continue
        answer = res.get("answer")
        answer = "" if answer is None else str(answer)
        usage = _usage(res, run_id, tid)
        item = {
            **base,
            "answer": answer,
            "correct": gaia_match(answer, task["gold"]),
            "correct_strict": exact_match(answer, task["gold"]),
            "seconds": seconds,
            "steps": res.get("steps"),
            **usage,
            "error": err,
            "timed_out": bool(stop_event.is_set()),
            "arch": res.get("arch"),
            "verifier": _verifier_summary(res.get("verifier")),
            "trace_path": res.get("trace_path"),
        }
        doc["items"].append(item)
        mark = "OK  " if item["correct"] else "MISS"
        log(f"  {tid:<28} {mark} answer={answer!r:<32} gold={task['gold']!r} ({seconds}s, {item['steps']} steps)")
        doc["summary"] = summarize(doc["items"])
        _atomic_write(out_path, doc)

    doc["budget"]["stopped"] = stop_reason
    doc["budget"]["calls_this_invocation"] = model_calls_so_far(run_id, doc["items"], baseline)
    doc["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    doc["summary"] = summarize(doc["items"])
    _atomic_write(out_path, doc)
    return doc


def format_summary(doc: dict) -> str:
    s = doc["summary"]
    if not s.get("n"):
        return f"{doc['run_id']}: nothing scored ({s.get('skipped', 0)} skipped)"
    ci = s.get("accuracy_ci95") or (None, None)
    lines = [
        f"run {doc['run_id']}  model={doc['config'].get('model')}  arch={doc['config'].get('arch')}",
        f"  GAIA quasi-exact : {s['correct']}/{s['n']} = {s['accuracy']:.1%}  (95% Wilson CI {ci[0]:.0%}-{ci[1]:.0%})",
        f"  course strict    : {s['correct_strict']}/{s['n']} = {s['accuracy_strict']:.1%}  (format gap {s['format_gap']})",
        f"  mean steps {s['mean_steps']}  mean seconds {s['mean_seconds']}  tokens {s['total_tokens']}  "
        f"errors {s['errors']}  skipped {s['skipped']}",
    ]
    if doc.get("budget", {}).get("stopped"):
        lines.append(f"  stopped early: {doc['budget']['stopped']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def _error_analysis(run_id: str, max_steps: int | None) -> Path | None:
    """Tag failure modes over this run's traces (gaia_agent.run.run_error_analysis; never raises)."""
    try:
        from gaia_agent.run import run_error_analysis
    except Exception as e:
        print(f"error analysis skipped: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    return run_error_analysis(run_id, max_steps=max_steps)


def _apply_env(args: argparse.Namespace) -> None:
    mapping = {
        "provider": "GAIA_PROVIDER",
        "model": "GAIA_MODEL_ID",
        "vision_model": "GAIA_VISION_MODEL_ID",
        "arch": "GAIA_ARCH",
        "disable_tools": "GAIA_DISABLE_TOOLS",
        "verify": "GAIA_VERIFY",
    }
    for attr, var in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            os.environ[var] = str(val)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the GAIA agent on the practice set and score it.")
    ap.add_argument("--only", help="comma list of task ids or id prefixes (e.g. p01,w)")
    ap.add_argument("--limit", type=int, help="run only the first N selected tasks")
    ap.add_argument("--max-requests", type=int, help="model-call budget for the whole run")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--timeout", type=float, default=600, help="per-task watchdog seconds (0 = off)")
    ap.add_argument("--run-id", help="results file name (default: practice-<model>-<arch>-<timestamp>)")
    ap.add_argument("--resume", action="store_true", help="keep scored items of an existing --run-id")
    ap.add_argument("--repeat", type=int, default=1, help="run the set K times (ids <run-id>-r1..rK)")
    ap.add_argument("--provider")
    ap.add_argument("--model")
    ap.add_argument("--vision-model")
    ap.add_argument("--arch", choices=["single", "multi"])
    ap.add_argument("--disable-tools", help="comma list of tool names to remove")
    ap.add_argument("--verify", choices=["0", "1"])
    ap.add_argument("--tasks", default=str(TASKS_PATH), help="tasks.jsonl to run")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    _apply_env(args)
    tasks = select_tasks(load_tasks(Path(args.tasks)), args.only, args.limit)
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2
    cfg = current_config()
    base_id = args.run_id or os.environ.get("GAIA_RUN_ID") or default_run_id(cfg)
    budget = args.max_requests
    for r in range(1, max(1, args.repeat) + 1):
        run_id = base_id if args.repeat == 1 else f"{base_id}-r{r}"
        print(f"== {run_id}: {len(tasks)} task(s); budget={budget if budget is not None else 'none'}")
        doc = run(tasks, run_id=run_id, max_requests=budget, max_steps=args.max_steps,
                  timeout=args.timeout or None, resume=args.resume, results_dir=Path(args.results_dir), config=cfg)
        print(format_summary(doc))
        out = Path(args.results_dir) / (run_id + ".json")
        analysis_path = _error_analysis(run_id, args.max_steps)
        if analysis_path is not None:
            doc["error_analysis"] = str(analysis_path)
            _atomic_write(out, doc)
        print(f"  -> {out}")
        if budget is not None:
            budget = max(0, budget - (doc["budget"].get("calls_this_invocation") or 0))
        if doc["budget"]["stopped"]:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
