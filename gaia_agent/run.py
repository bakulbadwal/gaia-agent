"""Batch runner: fetch the Unit 4 questions, solve each with a timeout, append to runs/answers.jsonl.

Usage:
    python -m gaia_agent.run                      # all unanswered questions
    python -m gaia_agent.run --only 8e86 a1e9     # task_id prefixes
    python -m gaia_agent.run --redo --limit 3
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from .config import API_URL, DATA_DIR, RUNS_DIR, get_run_id, provider_info, runs_dir


def normalize_answer(s) -> str:
    from .agent import normalize_answer as _norm

    return _norm(s)

DEFAULT_TIMEOUT = 420
ANSWERS_PATH = RUNS_DIR / "answers.jsonl"
TRACES_DIR = RUNS_DIR / "traces"
QUESTIONS_PATH = DATA_DIR / "questions.json"


def fetch_questions(refresh: bool = False, path: Path = QUESTIONS_PATH, api_url: str = API_URL) -> list[dict]:
    """GET {API_URL}/questions, cached at data/questions.json."""
    path = Path(path)
    if path.is_file() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    resp = requests.get(f"{api_url}/questions", timeout=30)
    resp.raise_for_status()
    questions = resp.json()
    if not isinstance(questions, list) or not questions:
        raise RuntimeError("questions endpoint returned an empty or invalid payload")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    return questions


def load_records(path: Path = ANSWERS_PATH) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written line from a killed run
    return records


def answered_ok(records: list[dict]) -> set[str]:
    """task_ids whose LATEST record has a non-empty answer and no error."""
    latest: dict[str, dict] = {}
    for r in records:
        if r.get("task_id"):
            latest[r["task_id"]] = r
    return {tid for tid, r in latest.items() if not r.get("error") and str(r.get("answer") or "").strip()}


def select_questions(
    questions: list[dict],
    records: list[dict],
    only: list[str] | None = None,
    redo: bool = False,
    limit: int | None = None,
) -> list[dict]:
    done = set() if redo else answered_ok(records)
    prefixes = [p.strip() for chunk in (only or []) for p in chunk.split(",") if p.strip()]
    picked = []
    for q in questions:
        tid = q.get("task_id")
        if not tid:
            continue
        if prefixes and not any(tid.startswith(p) for p in prefixes):
            continue
        if tid in done:
            continue
        picked.append(q)
    return picked[:limit] if limit else picked


def run_with_timeout(solve_fn: Callable, question: dict, timeout: float) -> dict:
    """Run solve_fn in a daemon thread; on timeout, signal it to stop and return an error result."""
    stop_event = threading.Event()
    box: dict = {}
    kwargs = {"task_id": question.get("task_id"), "file_name": question.get("file_name") or None}
    try:
        if "stop_event" in inspect.signature(solve_fn).parameters:
            kwargs["stop_event"] = stop_event
    except (TypeError, ValueError):
        pass

    def target():
        try:
            box["result"] = solve_fn(question.get("question", ""), **kwargs)
        except BaseException as e:  # solve() should not raise, but a fake/other solver might
            box["result"] = {"answer": "", "raw": None, "steps": 0, "error": f"{type(e).__name__}: {e}"}

    th = threading.Thread(target=target, name=f"solve-{question.get('task_id', '')[:8]}", daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        stop_event.set()  # the agent stops at its next step; the daemon thread can't block exit
        return {"answer": "", "raw": None, "steps": None, "error": f"Timeout after {timeout:.0f}s"}
    return box.get("result") or {"answer": "", "raw": None, "steps": 0, "error": "solver returned nothing"}


def append_record(record: dict, path: Path = ANSWERS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fh.flush()


def _safe_provider_info() -> dict:
    try:
        return provider_info()
    except Exception as e:
        return {"error": repr(e)}


def _verifier_summary(v: dict | None) -> dict | None:
    if not v:
        return None
    return {k: v.get(k) for k in ("candidate", "proposed", "accepted", "changed", "reason", "error")}


def run_batch(
    questions: list[dict],
    solve_fn: Callable,
    out_path: Path = ANSWERS_PATH,
    traces_dir: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    info: dict | None = None,
    run_id: str | None = None,
) -> list[dict]:
    """Solve each question and append one answers.jsonl row per question.

    The full per-question trace is written by solve() itself to
    runs/traces/<run_id>/<task_id>.json (its path lands in the row as trace_path).
    ``traces_dir`` is accepted for backward compatibility and ignored.
    """
    info = info if info is not None else _safe_provider_info()
    run_id = run_id or get_run_id()
    results = []
    for i, q in enumerate(questions, 1):
        tid = q.get("task_id", "")
        print(f"[{i}/{len(questions)}] {tid[:8]}  {q.get('question', '')[:90]!r}", flush=True)
        t0 = time.time()
        res = run_with_timeout(solve_fn, q, timeout)
        seconds = round(time.time() - t0, 1)
        record = {
            "run_id": res.get("run_id") or run_id,
            "task_id": tid,
            "question": q.get("question"),
            "file_name": q.get("file_name") or None,
            "answer": normalize_answer(res.get("answer", "")),
            "raw": res.get("raw"),
            "steps": res.get("steps"),
            "error": res.get("error"),
            "state": res.get("state"),
            "seconds": seconds,
            "arch": res.get("arch"),
            "input_tokens": res.get("input_tokens"),
            "output_tokens": res.get("output_tokens"),
            "model_calls": res.get("model_calls"),
            "verifier": _verifier_summary(res.get("verifier")),
            "trace_path": res.get("trace_path"),
            "provider_info": info,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        append_record(record, out_path)
        results.append(record)
        status = f"ERROR {record['error'][:60]}" if record["error"] else repr(record["answer"])[:60]
        print(f"    -> {status}  ({seconds}s, {record['steps']} steps)", flush=True)
        if record["trace_path"]:
            print(f"       trace: {record['trace_path']}", flush=True)
    return results


def run_error_analysis(run_id: str, max_steps: int | None = None, quiet: bool = False) -> Path | None:
    """Tag failure modes for one run (analysis/error_analysis.py). Never raises.

    Writes <runs>/error_analysis/<run_id>.json and refreshes <runs>/error_analysis.json (latest).
    """
    try:
        try:
            from analysis.error_analysis import analyze_run, format_table, write_report
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from analysis.error_analysis import analyze_run, format_table, write_report

        root = runs_dir()
        report = analyze_run(run_id, root, max_steps)
        if not report.get("items"):
            return None
        path = write_report(report, root / "error_analysis" / f"{run_id}.json")
        write_report(report, root / "error_analysis.json")
        if not quiet:
            print("\n" + format_table(report))
            print(f"\nerror analysis: {path}")
        return path
    except Exception as e:  # analysis is a report, never a reason to fail the run
        print(f"error analysis skipped: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def print_summary(results: list[dict], all_records: list[dict] | None = None, total: int | None = None) -> None:
    if results:
        print("\n%-10s %7s %5s  %-42s %s" % ("task_id", "secs", "steps", "answer", "error"))
        print("-" * 96)
        for r in results:
            print(
                "%-10s %7s %5s  %-42s %s"
                % (
                    r["task_id"][:8],
                    r.get("seconds", ""),
                    "" if r.get("steps") is None else r["steps"],
                    (r.get("answer") or "")[:42],
                    (r.get("error") or "")[:40],
                )
            )
        ok = sum(1 for r in results if not r.get("error") and r.get("answer"))
        print(f"\nThis run: {ok}/{len(results)} answered without error.")
    if all_records is not None:
        done = answered_ok(all_records)
        suffix = f" of {total}" if total else ""
        print(f"Cache: {len(done)}{suffix} tasks have a clean answer in {ANSWERS_PATH.name}.")


def main(argv: list[str] | None = None, solve_fn: Callable | None = None, questions: list[dict] | None = None,
         out_path: Path | None = None, traces_dir: Path | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description="Run the GAIA agent over the Unit 4 questions.")
    ap.add_argument("--only", nargs="+", help="task_id prefixes (space or comma separated)")
    ap.add_argument("--redo", action="store_true", help="re-run tasks that already have a clean answer")
    ap.add_argument("--limit", type=int, default=None, help="run at most N questions")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per-question wall-clock seconds")
    ap.add_argument("--refresh-questions", action="store_true", help="re-download questions.json")
    ap.add_argument("--run-id", help="trace/usage grouping id (default: $GAIA_RUN_ID, else a timestamp)")
    ap.add_argument("--no-analysis", action="store_true", help="skip the error analysis at the end")
    args = ap.parse_args(argv)
    if args.run_id:
        import os

        os.environ["GAIA_RUN_ID"] = args.run_id
    run_id = get_run_id()

    out_path = Path(out_path or ANSWERS_PATH)
    if questions is None:
        questions = fetch_questions(refresh=args.refresh_questions)
    if solve_fn is None:
        from .agent import solve as solve_fn  # noqa: N813

    records = load_records(out_path)
    todo = select_questions(questions, records, only=args.only, redo=args.redo, limit=args.limit)
    print(f"{len(questions)} questions; {len(todo)} to run (provider: {_safe_provider_info().get('provider')}, "
          f"run_id: {run_id}).")
    results = run_batch(todo, solve_fn, out_path=out_path, timeout=args.timeout, run_id=run_id)
    print_summary(results, load_records(out_path), total=len(questions))
    if results and not args.no_analysis:
        from .agent import MAX_STEPS

        run_error_analysis(run_id, max_steps=MAX_STEPS)
    return results


if __name__ == "__main__":
    main()
    sys.stdout.flush()
