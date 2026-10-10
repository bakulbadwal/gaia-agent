"""pass@k and pass^k from k repeated runs of the same configuration.

    .venv/bin/python -m evals.passk evals/results/practice-X-r1.json evals/results/practice-X-r2.json ...
    .venv/bin/python -m evals.passk --glob 'evals/results/practice-X-r*.json' --metric strict

One run of 16 tasks says little: a model that solves a task 60% of the time looks like a hit or
a miss depending on the draw. Repeating the same config n times gives, per task, c successes out
of n attempts, and two complementary numbers:

pass@k  (Chen et al. 2021, "Evaluating Large Language Models Trained on Code", eq. 1)
    The probability that AT LEAST ONE of k attempts, drawn without replacement from the n
    observed, is correct. Unbiased estimator:

        pass@k = 1 - C(n - c, k) / C(n, k)          (= 1 when n - c < k)

    It answers "if I may retry k times and keep the best, how often do I succeed?" -- the
    ceiling a verifier/selector could reach.

pass^k  (Yao et al. 2024, "tau-bench", Sec. 5)
    The probability that ALL k attempts are correct. Unbiased estimator:

        pass^k = C(c, k) / C(n, k)                  (= 0 when c < k)

    It answers "if this runs k times in production, how often is it right every time?" -- the
    reliability number. pass^1 == pass@1 == c / n; as k grows pass@k rises and pass^k falls,
    and the gap between them is the flakiness.

Both are averaged over tasks (each task weighted equally) and only for k <= n. Using the
combinatorial forms instead of 1 - (1 - c/n)^k matters at small n: with n = 5, c = 2, k = 2 the
naive plug-in gives 0.64, the unbiased estimator 0.70.

The script also reports the per-run accuracy mean and standard deviation, and lists the
flaky tasks (0 < c < n), which are where prompt or tool work pays off fastest.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import statistics
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evals" / "results"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    _check(n, c, k)
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased pass^k (tau-bench): C(c, k) / C(n, k)."""
    _check(n, c, k)
    return comb(c, k) / comb(n, k)


def naive_pass_at_k(p: float, k: int) -> float:
    """Plug-in estimate 1 - (1 - p)^k, shown only for comparison (biased at small n)."""
    return 1.0 - (1.0 - p) ** k


def _check(n: int, c: int, k: int) -> None:
    if not (0 <= c <= n) or n <= 0:
        raise ValueError(f"need 0 <= c <= n and n > 0, got n={n} c={c}")
    if not (1 <= k <= n):
        raise ValueError(f"need 1 <= k <= n, got k={k} n={n}")


# --------------------------------------------------------------------------- aggregation


def load_results(paths: list[str | Path]) -> list[dict]:
    docs = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["_path"] = str(p)
        docs.append(doc)
    return docs


def config_key(doc: dict) -> str:
    cfg = dict(doc.get("config") or {})
    return json.dumps({k: cfg.get(k) for k in ("provider", "model", "arch", "disabled_tools", "verify")}, sort_keys=True)


def counts_by_task(docs: list[dict], metric: str = "gaia") -> dict[str, tuple[int, int]]:
    """{task_id: (n attempts, c correct)} over runs; skipped items are not attempts."""
    field = "correct" if metric == "gaia" else "correct_strict"
    out: dict[str, list[int]] = {}
    for doc in docs:
        for it in doc.get("items", []):
            if it.get("skipped") or it.get(field) is None:
                continue
            nc = out.setdefault(it["id"], [0, 0])
            nc[0] += 1
            nc[1] += bool(it[field])
    return {k: (v[0], v[1]) for k, v in out.items()}


def analyze(docs: list[dict], metric: str = "gaia", ks: list[int] | None = None) -> dict:
    if not docs:
        raise ValueError("no result files")
    keys = {config_key(d) for d in docs}
    counts = counts_by_task(docs, metric)
    if not counts:
        raise ValueError("no scored items in the result files")
    n_max = max(n for n, _ in counts.values())
    ks = ks or list(range(1, n_max + 1))
    table = []
    for k in ks:
        eligible = {t: nc for t, nc in counts.items() if nc[0] >= k}
        if not eligible:
            continue
        pa = [pass_at_k(n, c, k) for n, c in eligible.values()]
        ph = [pass_hat_k(n, c, k) for n, c in eligible.values()]
        table.append({"k": k, "tasks": len(eligible), "pass@k": round(sum(pa) / len(pa), 4),
                      "pass^k": round(sum(ph) / len(ph), 4)})
    field = "correct" if metric == "gaia" else "correct_strict"
    run_acc = []
    for d in docs:
        scored = [it for it in d.get("items", []) if not it.get("skipped") and it.get(field) is not None]
        if scored:
            run_acc.append(sum(bool(it[field]) for it in scored) / len(scored))
    per_task = {t: {"n": n, "c": c, "p": round(c / n, 4)} for t, (n, c) in sorted(counts.items())}
    return {
        "metric": metric,
        "runs": [d.get("run_id") or d["_path"] for d in docs],
        "same_config": len(keys) == 1,
        "config": json.loads(next(iter(keys))) if len(keys) == 1 else [json.loads(k) for k in sorted(keys)],
        "n_runs": len(docs),
        "run_accuracy": {
            "mean": round(statistics.mean(run_acc), 4) if run_acc else None,
            "stdev": round(statistics.stdev(run_acc), 4) if len(run_acc) > 1 else None,
            "min": round(min(run_acc), 4) if run_acc else None,
            "max": round(max(run_acc), 4) if run_acc else None,
        },
        "table": table,
        "per_task": per_task,
        "always_solved": sorted(t for t, (n, c) in counts.items() if c == n),
        "never_solved": sorted(t for t, (n, c) in counts.items() if c == 0),
        "flaky": sorted(t for t, (n, c) in counts.items() if 0 < c < n),
    }


def to_markdown(rep: dict) -> str:
    lines = [f"### pass@k / pass^k ({rep['metric']} scorer, {rep['n_runs']} runs)", ""]
    if not rep["same_config"]:
        lines += ["> WARNING: the result files do not share one config; pass@k assumes i.i.d. repeats.", ""]
    ra = rep["run_accuracy"]
    if ra["mean"] is not None:
        sd = f" ± {ra['stdev']:.1%}" if ra["stdev"] is not None else ""
        lines += [f"Per-run accuracy: {ra['mean']:.1%}{sd} (min {ra['min']:.1%}, max {ra['max']:.1%})", ""]
    lines += ["| k | tasks | pass@k | pass^k |", "|---|---|---|---|"]
    for row in rep["table"]:
        lines.append(f"| {row['k']} | {row['tasks']} | {row['pass@k']:.1%} | {row['pass^k']:.1%} |")
    pt = rep["per_task"]
    flaky = ", ".join("%s (%d/%d)" % (t, pt[t]["c"], pt[t]["n"]) for t in rep["flaky"]) or "-"
    lines += ["",
              "Always solved (%d): %s" % (len(rep["always_solved"]), ", ".join(rep["always_solved"]) or "-"),
              "Flaky (%d): %s" % (len(rep["flaky"]), flaky),
              "Never solved (%d): %s" % (len(rep["never_solved"]), ", ".join(rep["never_solved"]) or "-")]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pass@k and pass^k from repeated result files.")
    ap.add_argument("paths", nargs="*", help="result JSON files of repeated runs of ONE config")
    ap.add_argument("--glob", help="glob for result files, e.g. 'evals/results/practice-X-r*.json'")
    ap.add_argument("--metric", choices=["gaia", "strict"], default="gaia")
    ap.add_argument("--k", help="comma list of k values (default 1..n)")
    ap.add_argument("--out", help="write the JSON report here (default evals/results/passk_<first run>.json)")
    args = ap.parse_args(argv)

    paths = list(args.paths)
    if args.glob:
        paths += sorted(globlib.glob(args.glob))
    if not paths:
        ap.error("give result files or --glob")
    rep = analyze(load_results(paths), args.metric, [int(k) for k in args.k.split(",")] if args.k else None)
    print(to_markdown(rep))
    out = Path(args.out) if args.out else RESULTS_DIR / f"passk_{rep['runs'][0]}_{args.metric}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
