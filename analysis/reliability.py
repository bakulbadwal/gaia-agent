"""Reliability case study: pass@k / pass^k and why answers flip, recomputed from committed files.

    python -m analysis.reliability                 # tables to stdout + docs/media/reliability_grid.png
    python -m analysis.reliability --no-chart      # tables only (no matplotlib needed)

Inputs, all committed to the repo:

- ``evals/results/<run_id>.json``: one file per run of the practice set (schema: evals/run_practice.py).
  The repeat runs are the four same-config baseline runs that evals/results/passk-practice-baseline.json
  pools; the ablation runs are shown next to them for comparison but never pooled into pass^k.
- ``docs/traces/<run_id>/<task_id>.json``: the per-step trace of every practice attempt (written by
  report/build_report.py). Used to attribute each failure to a cause.

pass@k and pass^k use the unbiased estimators in evals/passk.py. Failure causes reuse the trace tags
of analysis/error_analysis.py and add three trace signals it does not compute: seconds spent inside
vision-tool calls, degenerate tool output, and runaway generation (a step that hit the output cap).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.error_analysis import VISION_TOOLS, _walk, analyze_trace  # noqa: E402
from evals.passk import pass_at_k, pass_hat_k  # noqa: E402
from evals.run_practice import wilson_ci  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results"
TRACES_DIR = ROOT / "docs" / "traces"
CHART_PATH = ROOT / "docs" / "media" / "reliability_grid.png"

# The same config, run 4 times on 2026-10-10. The 4th is the ablation's baseline (identical config).
REPEAT_RUNS = ("practice-baseline-r1", "practice-baseline-r2", "practice-baseline-r3", "abl-2026-10-10-baseline")
# Different configs on the same 16 tasks: context only, never pooled into pass^k.
ABLATION_RUNS = ("abl-2026-10-10-no_web", "abl-2026-10-10-verify", "abl-2026-10-10-arch_multi", "abl-2026-10-10-no_media")
RUN_LABELS = {
    "practice-baseline-r1": "Run 1", "practice-baseline-r2": "Run 2", "practice-baseline-r3": "Run 3",
    "abl-2026-10-10-baseline": "Run 4", "abl-2026-10-10-no_web": "no web", "abl-2026-10-10-verify": "verifier",
    "abl-2026-10-10-arch_multi": "multi-agent", "abl-2026-10-10-no_media": "no media",
}

# A step whose output reached this many tokens hit the provider's generation cap (65,536 here).
RUNAWAY_OUTPUT_TOKENS = 32_000
_UNK_RE = re.compile(r"<unk>")


# --------------------------------------------------------------------------- loading


def load_run(run_id: str, results_dir: Path = RESULTS_DIR) -> dict:
    return json.loads((results_dir / f"{run_id}.json").read_text(encoding="utf-8"))


def load_trace(run_id: str, task_id: str, traces_dir: Path = TRACES_DIR) -> dict | None:
    p = traces_dir / run_id / f"{task_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------------------- metrics


def counts(docs: list[dict]) -> dict[str, tuple[int, int]]:
    """{task_id: (attempts, correct)} under the GAIA scorer; skipped items are not attempts."""
    out: dict[str, list[int]] = {}
    for d in docs:
        for it in d["items"]:
            if it.get("skipped") or it.get("correct") is None:
                continue
            nc = out.setdefault(it["id"], [0, 0])
            nc[0] += 1
            nc[1] += bool(it["correct"])
    return {t: (n, c) for t, (n, c) in sorted(out.items())}


def headline(docs: list[dict]) -> dict:
    """Per-run accuracy, pass@k and pass^k for k = 1..n (task-averaged), and the flaky tasks."""
    nc = counts(docs)
    n_runs = min(n for n, _ in nc.values())
    run_acc = [d["summary"]["correct"] / d["summary"]["n"] for d in docs]
    table = []
    for k in range(1, n_runs + 1):
        table.append({
            "k": k,
            "pass@k": sum(pass_at_k(n, c, k) for n, c in nc.values()) / len(nc),
            "pass^k": sum(pass_hat_k(n, c, k) for n, c in nc.values()) / len(nc),
        })
    always = [t for t, (n, c) in nc.items() if c == n]
    return {
        "runs": [d["run_id"] for d in docs],
        "configs_identical": len({json.dumps(d["config"], sort_keys=True) for d in docs}) == 1,
        "tasks": len(nc),
        "n_runs": n_runs,
        "run_accuracy": run_acc,
        "mean": statistics.mean(run_acc),
        "stdev": statistics.stdev(run_acc) if len(run_acc) > 1 else 0.0,
        "table": table,
        "always_solved": always,
        "always_solved_ci95": wilson_ci(len(always), len(nc)),
        "flaky": {t: (n, c) for t, (n, c) in nc.items() if 0 < c < n},
        "never_solved": [t for t, (n, c) in nc.items() if c == 0],
        "format_gap": sum(d["summary"].get("format_gap") or 0 for d in docs),
        "per_task": nc,
    }


# --------------------------------------------------------------------------- failure causes


def outcome(item: dict, max_steps: int) -> str:
    """What the grader saw: pass, timeout, max_steps (ran out of steps) or wrong."""
    if item.get("correct"):
        return "pass"
    if item.get("timed_out") or "interrupted" in str(item.get("error") or "").lower():
        return "timeout"
    if (item.get("steps") or 0) > max_steps:
        return "max_steps"
    return "wrong"


def _degenerate(text: str) -> bool:
    """Garbage tool output: <unk> runs, or one token making up most of a long observation."""
    if len(_UNK_RE.findall(text)) >= 5:
        return True
    toks = [t for t in text.split() if re.search(r"[A-Za-z]", t)]  # a printed chess board is mostly "."
    if len(toks) >= 60:
        top = max(toks.count(t) for t in set(toks))
        return top / len(toks) > 0.4
    return False


def trace_signals(trace: dict | None, max_steps: int) -> dict:
    """Evidence from one attempt's trace (empty when the trace is missing)."""
    if not trace:
        return {"has_trace": False}
    steps = list(_walk(trace.get("steps") or []))
    vision = [s for s in steps if VISION_TOOLS & set(s.get("tools") or [])]
    tags = analyze_trace(trace, max_steps=max_steps)["tags"]
    obs = [str(s.get("observation") or "") for s in steps if s.get("tools")]  # tool output, not the agent's own prints
    return {
        "has_trace": True,
        "tags": tags,
        "vision_calls": len(vision),
        "vision_seconds": round(sum(s.get("duration_s") or 0 for s in vision), 1),
        "tool_errors": sum(1 for o in obs if re.search(r"^\s*ERROR\b", o.replace("Execution logs:", ""), re.M)),
        "degenerate_outputs": sum(1 for o in obs if _degenerate(o)),
        "code_errors": sum(1 for s in steps if "Code execution failed" in str(s.get("error") or "")),
        "runaway_steps": sum(1 for s in steps if (s.get("output_tokens") or 0) >= RUNAWAY_OUTPUT_TOKENS),
    }


def cause(out: str, sig: dict, skills: list[str]) -> str:
    """Primary cause, in the order a fix would address it.

    timeout and max_steps come straight from the result row. A wrong answer is a tool error when a
    tool returned an error or garbage, a retrieval miss when a web task finished without using a
    web tool, and reasoning otherwise (the inputs were there; the model drew the wrong conclusion).
    Formatting failures (right in substance, wrong in form) are counted separately via format_gap.
    """
    if out == "pass":
        return "-"
    if out in ("timeout", "max_steps"):
        return out
    if sig.get("tool_errors") or sig.get("degenerate_outputs"):
        return "tool_error"
    if "web" in skills and "no_tool_used" in sig.get("tags", []):
        return "retrieval_miss"
    return "reasoning"


def failures(docs: list[dict], traces_dir: Path = TRACES_DIR) -> list[dict]:
    rows = []
    for d in docs:
        ms = (d.get("budget") or {}).get("max_steps") or 12
        for it in d["items"]:
            out = outcome(it, ms)
            if out == "pass" or it.get("skipped"):
                continue
            sig = trace_signals(load_trace(d["run_id"], it["id"], traces_dir), ms)
            rows.append({
                "run": d["run_id"], "task": it["id"], "outcome": out, "cause": cause(out, sig, it.get("skills") or []),
                "answer": it.get("answer"), "gold": it.get("gold"), "seconds": it.get("seconds"), **sig,
            })
    return rows


def flaky_task_times(docs: list[dict], task: str) -> tuple[list[float], list[float]]:
    """Wall-clock seconds of every attempt at `task`, and of every attempt at the other tasks."""
    mine, rest = [], []
    for d in docs:
        for it in d["items"]:
            (mine if it["id"] == task else rest).append(it.get("seconds") or 0.0)
    return mine, rest


# --------------------------------------------------------------------------- output


def _pct(x: float) -> str:
    return f"{x:.1%}"


def to_markdown(h: dict, fails: list[dict], compare: dict[str, tuple[int, int]]) -> str:
    acc = " · ".join(_pct(a) for a in h["run_accuracy"])
    lo, hi = h["always_solved_ci95"]
    lines = [
        f"Runs: {', '.join(h['runs'])} (configs identical: {h['configs_identical']})",
        f"Per-run accuracy: {acc} (mean {_pct(h['mean'])}, sd {_pct(h['stdev'])})",
        f"Tasks solved in every run: {len(h['always_solved'])}/{h['tasks']} (Wilson 95% CI {_pct(lo)}-{_pct(hi)})",
        f"Format gap (right in substance, wrong in form) across all runs: {h['format_gap']}",
        "",
        "| k | pass@k | pass^k |",
        "|---|---|---|",
    ]
    lines += [f"| {r['k']} | {_pct(r['pass@k'])} | {_pct(r['pass^k'])} |" for r in h["table"]]
    lines += ["", "Flaky: " + (", ".join(f"{t} ({c}/{n})" for t, (n, c) in h["flaky"].items()) or "-"),
              "Never solved: " + (", ".join(h["never_solved"]) or "-"), "",
              "Every attempt that had the vision tool (repeat runs + no web, verifier, multi-agent):", ""]
    lines += [f"- {t}: {c}/{n}" for t, (n, c) in compare.items() if c < n] or ["- (none failed)"]
    others = [(n, c) for t, (n, c) in compare.items() if c == n]
    lines += [f"- the other {len(others)} tasks: {sum(c for _, c in others)}/{sum(n for n, _ in others)}", "",
              "| run | task | outcome | cause | answer | s | vision calls | vision s | tool errors | garbage outputs | code errors | runaway steps |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in fails:
        ans = f["answer"] if f["answer"] and not str(f["answer"]).startswith("<code>") else ("(raw code)" if f["answer"] else "(none)")
        lines.append(
            f"| {RUN_LABELS.get(f['run'], f['run'])} | {f['task']} | {f['outcome']} | {f['cause']} | {ans} | "
            f"{f['seconds']:.0f} | {f.get('vision_calls', '?')} | {f.get('vision_seconds', '?')} | {f.get('tool_errors', '?')} | "
            f"{f.get('degenerate_outputs', '?')} | {f.get('code_errors', '?')} | {f.get('runaway_steps', '?')} |")
    return "\n".join(lines) + "\n"


def draw_grid(docs: list[dict], path: Path = CHART_PATH, n_repeat: int = len(REPEAT_RUNS)) -> Path:
    """Task x run grid: pale green = solved, solid = failed (labelled), so failures carry text too."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    colors = {"pass": "#cfeccf", "wrong": "#d03b3b", "timeout": "#fab219", "max_steps": "#ec835a"}
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    tasks = [it["id"] for it in docs[0]["items"]]
    n_cols = len(docs)
    gap = 0.5  # space between the repeat runs and the ablation configs
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    xs = []
    for j, d in enumerate(docs):
        x = j + (gap if j >= n_repeat else 0)
        xs.append(x)
        ms = (d.get("budget") or {}).get("max_steps") or 12
        by_id = {it["id"]: it for it in d["items"]}
        for i, t in enumerate(tasks):
            it = by_id[t]
            out = outcome(it, ms)
            ax.add_patch(Rectangle((x + 0.04, i + 0.06), 0.92, 0.88, facecolor=colors[out], edgecolor=surface, linewidth=2))
            if out == "pass":
                label, color = "✓", "#2b6b2b"
            elif out == "wrong":
                label, color = f"✗ {it.get('answer') if not str(it.get('answer')).startswith('<code>') else 'wrong'}", "white"
            elif out == "timeout":
                label, color = f"timeout {it.get('seconds', 0):.0f}s", ink
            else:
                label, color = "out of steps", ink
            ax.text(x + 0.5, i + 0.5, label, ha="center", va="center", fontsize=7.5, color=color)
    ax.set_xlim(-0.05, xs[-1] + 1.05)
    ax.set_ylim(len(tasks), 0)
    ax.set_xticks([x + 0.5 for x in xs])
    ax.set_xticklabels([RUN_LABELS.get(d["run_id"], d["run_id"]) for d in docs], fontsize=8.5, color=ink)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(tasks))])
    ax.set_yticklabels(tasks, fontsize=8.5, color=ink, family="monospace")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    top = -0.95
    ax.text(n_repeat / 2, top, "same config, 4 repeat runs", ha="center", fontsize=8.5, color=muted)
    if n_cols > n_repeat:
        ax.text(n_repeat + gap + (n_cols - n_repeat) / 2, top, "ablation configs (context only)", ha="center", fontsize=8.5, color=muted)
    fig.legend(handles=[Patch(color=colors["pass"], label="solved"), Patch(color=colors["wrong"], label="wrong answer"),
                        Patch(color=colors["timeout"], label="timed out (600 s watchdog)"),
                        Patch(color=colors["max_steps"], label="ran out of steps")],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=ink)
    fig.suptitle("Practice set: every task in every run (2026-10-10)", fontsize=11, color=ink, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reliability case study numbers from the committed run files.")
    ap.add_argument("--no-chart", action="store_true", help="skip the PNG (no matplotlib needed)")
    ap.add_argument("--chart", default=str(CHART_PATH), help="where to write the grid PNG")
    args = ap.parse_args(argv)

    repeat = [load_run(r) for r in REPEAT_RUNS]
    ablation = [load_run(r) for r in ABLATION_RUNS]
    h = headline(repeat)
    with_vision = repeat + [d for d in ablation if not VISION_TOOLS & set(d["config"].get("disabled_tools") or [])]
    print(to_markdown(h, failures(repeat + ablation), counts(with_vision)))
    mine, rest = flaky_task_times(repeat, next(iter(h["flaky"]), ""))
    if mine:
        print(f"Median seconds per attempt: flaky task {statistics.median(mine):.0f}, other tasks {statistics.median(rest):.0f}")
    if not args.no_chart:
        print(f"-> {draw_grid(repeat + ablation, Path(args.chart))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
