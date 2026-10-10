"""Ablations: run several agent configurations over the practice set and compare them.

    .venv/bin/python -m evals.ablate --plan                         # what would run, and its cost
    .venv/bin/python -m evals.ablate --max-requests 300             # default configs, shared budget
    .venv/bin/python -m evals.ablate --configs baseline,no_web --only p,w01 --max-requests 120
    .venv/bin/python -m evals.ablate --group model --max-requests 400
    .venv/bin/python -m evals.ablate --aggregate-only --tag 20261010   # rebuild the table only

Each config in evals/configs/*.json (or .yaml) is ``{name, group, default, description, env,
repeats, requires}``. ``env`` is layered on top of the caller's environment, so tool and
architecture ablations run on the same model as the baseline, and model configs swap only the
model. Every config runs in its own subprocess (``python -m evals.run_practice``), so no module
state, cached client or env var leaks from one config into the next.

Budget: the total --max-requests is split evenly over the config-runs still to do, each
subprocess enforces its share, and the leftover rolls forward. Run ids are deterministic
(``abl-<tag>-<config>[-rN]``) and every subprocess runs with --resume, so when a free-tier daily
cap stops the ablation, re-running the same command the next day continues where it stopped.

Outputs: evals/results/ablation_<tag>.json and .md -- accuracy under both scorers, the change
vs baseline, an exact McNemar test on the paired per-task outcomes, mean steps, tokens and
seconds, and a per-skill matrix. With 16 tasks one task is 6.25 points, so the table reports the
paired test rather than letting a one-task wobble read as an effect.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.run_practice import RESULTS_DIR, TASKS_PATH, load_tasks, select_tasks  # noqa: E402

CONFIGS_DIR = ROOT / "evals" / "configs"
DEFAULT_EST_CALLS = 7  # model calls per task; refine with --est-calls from a real run's summary


# --------------------------------------------------------------------------- configs


def load_configs(configs_dir: Path = CONFIGS_DIR) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(Path(configs_dir).iterdir()):
        if p.suffix == ".json":
            cfg = json.loads(p.read_text(encoding="utf-8"))
        elif p.suffix in (".yaml", ".yml"):
            import yaml  # optional dependency, only needed for YAML configs

            cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        else:
            continue
        cfg.setdefault("name", p.stem)
        cfg.setdefault("env", {})
        cfg.setdefault("repeats", 1)
        cfg.setdefault("default", False)
        cfg.setdefault("group", "other")
        cfg["_path"] = str(p)
        if cfg["name"] in out:
            raise ValueError(f"duplicate config name {cfg['name']!r} ({p})")
        out[cfg["name"]] = cfg
    return out


def known_tool_names() -> set[str] | None:
    """Tool names the agent exposes, or None when the tool modules cannot be imported."""
    names: set[str] = set()
    try:
        from gaia_agent.tools_web import TOOLS_WEB

        names |= {t.name for t in TOOLS_WEB}
        from gaia_agent.tools_media import TOOLS_MEDIA

        names |= {t.name for t in TOOLS_MEDIA}
    except Exception:
        return None
    return names


def validate_config(cfg: dict, tool_names: set[str] | None = None) -> list[str]:
    """Problems with a config (empty list = fine)."""
    problems = []
    for key in ("name", "env"):
        if key not in cfg:
            problems.append(f"missing {key!r}")
    if not isinstance(cfg.get("env"), dict) or not all(isinstance(v, str) for v in cfg.get("env", {}).values()):
        problems.append("env must map names to strings")
    disabled = [t.strip() for t in cfg.get("env", {}).get("GAIA_DISABLE_TOOLS", "").split(",") if t.strip()]
    if tool_names is not None:
        unknown = sorted(set(disabled) - tool_names)
        if unknown:
            problems.append(f"GAIA_DISABLE_TOOLS names unknown tools: {unknown}")
    arch = cfg.get("env", {}).get("GAIA_ARCH")
    if arch is not None and arch not in ("single", "multi"):
        problems.append(f"GAIA_ARCH must be single|multi, got {arch!r}")
    if cfg.get("env", {}).get("GAIA_VERIFY") not in (None, "0", "1"):
        problems.append("GAIA_VERIFY must be '0' or '1'")
    if not isinstance(cfg.get("repeats", 1), int) or cfg.get("repeats", 1) < 1:
        problems.append("repeats must be a positive int")
    return problems


def requirement_gap(cfg: dict) -> str | None:
    """Why a config cannot run here (missing key, local server down), or None."""
    req = cfg.get("requires") or {}
    if req.get("env"):
        try:
            from gaia_agent.config import load_env

            load_env()  # ~/.claude/.env -> os.environ; never prints values
        except Exception:
            pass
        missing = [k for k in req["env"] if not os.environ.get(k)]
        if missing:
            return f"missing env {missing}"
    if req.get("url"):
        try:
            import requests

            requests.get(req["url"], timeout=3).raise_for_status()
        except Exception as e:
            return f"{req['url']} unreachable ({type(e).__name__})"
    return None


def select_configs(all_cfgs: dict[str, dict], names: str | None = None, group: str | None = None) -> list[dict]:
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        missing = [n for n in wanted if n not in all_cfgs]
        if missing:
            raise SystemExit(f"unknown config(s): {missing}; have {sorted(all_cfgs)}")
        chosen = [all_cfgs[n] for n in wanted]
    elif group:
        chosen = [c for c in all_cfgs.values() if c.get("group") == group]
    else:
        chosen = [c for c in all_cfgs.values() if c.get("default")]
    if "baseline" in all_cfgs:  # deltas need a reference, and it runs first
        chosen = [all_cfgs["baseline"]] + [c for c in chosen if c["name"] != "baseline"]
    return chosen


def run_ids_for(cfg: dict, tag: str) -> list[str]:
    base = f"abl-{tag}-{cfg['name']}"
    reps = int(cfg.get("repeats", 1))
    return [base] if reps == 1 else [f"{base}-r{i}" for i in range(1, reps + 1)]


# --------------------------------------------------------------------------- stats


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A right, B wrong) and c (vice versa)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _per_task(docs: list[dict], field: str = "correct") -> dict[str, float]:
    """Mean success per task across a config's repeats (skipped items excluded)."""
    acc: dict[str, list[int]] = {}
    for d in docs:
        for it in d.get("items", []):
            if it.get("skipped") or it.get(field) is None:
                continue
            acc.setdefault(it["id"], []).append(int(bool(it[field])))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None


def aggregate(configs: list[dict], tag: str, results_dir: Path = RESULTS_DIR,
              tasks: list[dict] | None = None) -> dict:
    rows = []
    per_task_matrix: dict[str, dict[str, float]] = {}
    skills_of = {t["id"]: t.get("skills", []) for t in (tasks or [])}
    for cfg in configs:
        docs = []
        for rid in run_ids_for(cfg, tag):
            p = Path(results_dir) / f"{rid}.json"
            if p.is_file():
                docs.append(json.loads(p.read_text(encoding="utf-8")))
        row = {"name": cfg["name"], "group": cfg.get("group"), "env": cfg.get("env", {}),
               "run_ids": [d["run_id"] for d in docs], "status": "ok" if docs else "not run"}
        if cfg.get("_skip_reason"):
            row["status"] = f"skipped: {cfg['_skip_reason']}"
        if docs:
            items = [it for d in docs for it in d.get("items", []) if not it.get("skipped")]
            pt, pts = _per_task(docs, "correct"), _per_task(docs, "correct_strict")
            per_task_matrix[cfg["name"]] = pt
            n = len(pt)
            row.update({
                "model": docs[0].get("config", {}).get("model"),
                "n_tasks": n,
                "skipped": sum(1 for d in docs for it in d.get("items", []) if it.get("skipped")),
                "accuracy": round(sum(pt.values()) / n, 4) if n else None,
                "accuracy_strict": round(sum(pts.values()) / len(pts), 4) if pts else None,
                "mean_steps": _mean([it.get("steps") for it in items]),
                "mean_seconds": _mean([it.get("seconds") for it in items]),
                "mean_tokens": _mean([(it.get("input_tokens") or 0) + (it.get("output_tokens") or 0)
                                      for it in items if it.get("input_tokens") is not None]),
                "errors": sum(1 for it in items if it.get("error")),
            })
        rows.append(row)

    base = per_task_matrix.get("baseline")
    for row in rows:
        mine = per_task_matrix.get(row["name"])
        if base is None or mine is None or row["name"] == "baseline":
            continue
        shared = sorted(set(base) & set(mine))
        if not shared:
            continue
        b_acc = sum(base[t] for t in shared) / len(shared)
        m_acc = sum(mine[t] for t in shared) / len(shared)
        # Paired test on majority outcomes (a task "passes" for a config if it passed >= half its repeats).
        b_only = sum(1 for t in shared if base[t] >= 0.5 > mine[t])
        m_only = sum(1 for t in shared if mine[t] >= 0.5 > base[t])
        row["vs_baseline"] = {
            "shared_tasks": len(shared),
            "delta": round(m_acc - b_acc, 4),
            "baseline_only": b_only,
            "config_only": m_only,
            "mcnemar_p": round(mcnemar_exact(b_only, m_only), 4),
        }

    skill_matrix: dict[str, dict[str, str]] = {}
    for name, pt in per_task_matrix.items():
        for tid, p in pt.items():
            for sk in skills_of.get(tid, []):
                cell = skill_matrix.setdefault(sk, {}).setdefault(name, [0.0, 0])
                cell[0] += p
                cell[1] += 1
    skill_matrix_fmt = {sk: {name: f"{v[0]:g}/{v[1]}" for name, v in cols.items()}
                        for sk, cols in sorted(skill_matrix.items())}
    return {
        "tag": tag,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "set": "practice",
        "tasks": [t["id"] for t in tasks] if tasks else None,
        "configs": rows,
        "per_task": per_task_matrix,
        "per_skill": skill_matrix_fmt,
    }


def _pct(x) -> str:
    return "-" if x is None else f"{x:.1%}"


def _v(x) -> str:
    return "-" if x is None else str(x)


def to_markdown(rep: dict) -> str:
    lines = [f"## Ablation {rep['tag']} (practice set)", "",
             "| config | model | n | GAIA acc | strict acc | Δ vs baseline | McNemar p | mean steps | mean tokens | mean s | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["configs"]:
        vb = r.get("vs_baseline") or {}
        delta = "-" if not vb else f"{vb['delta']:+.1%} ({vb['config_only']}+/{vb['baseline_only']}-)"
        lines.append(
            f"| {r['name']} | {_v(r.get('model'))} | {_v(r.get('n_tasks'))} | {_pct(r.get('accuracy'))} | "
            f"{_pct(r.get('accuracy_strict'))} | {delta} | {_v(vb.get('mcnemar_p'))} | {_v(r.get('mean_steps'))} | "
            f"{_v(r.get('mean_tokens'))} | {_v(r.get('mean_seconds'))} | {r['status']} |"
        )
    names = [r["name"] for r in rep["configs"] if r["name"] in rep["per_task"]]
    if rep["per_skill"] and names:
        lines += ["", "### Per-skill (solved / tasks with that skill)", "",
                  "| skill | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for sk, cols in rep["per_skill"].items():
            lines.append(f"| {sk} | " + " | ".join(cols.get(n, "-") for n in names) + " |")
    lines += ["", "Δ is paired on the tasks both configs scored; (a+/b-) = tasks only this config / only the "
              "baseline solved. With 16 tasks one task is 6.25 points: treat p > 0.05 as no evidence of an effect."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- running

# Behaviour switches every config must start from scratch: a GAIA_ARCH=multi exported in the
# caller's shell must not silently turn the "baseline" into a multi-agent run.
BEHAVIOUR_VARS = ("GAIA_ARCH", "GAIA_VERIFY", "GAIA_DISABLE_TOOLS")
# Model-selection vars that only make sense for the provider they were written for.
MODEL_VARS = ("GAIA_MODEL_ID", "GAIA_VISION_MODEL_ID", "GAIA_FALLBACKS", "GAIA_VISION_FALLBACKS", "GAIA_BASE_URL")


def config_env(cfg: dict, base: dict | None = None, run_id: str | None = None) -> dict:
    """The exact environment one config runs under.

    Starts from ``base`` (default os.environ), clears the behaviour switches (arch, verifier,
    disabled tools) so only the config sets them, clears the caller's model overrides when the
    config switches provider, then layers the config's env on top. The agent reads these vars
    in gaia_agent.config (arch(), disabled_tools(), verify_enabled(), model_id()).
    """
    env = dict(os.environ if base is None else base)
    cfg_env = cfg.get("env", {}) or {}
    for var in BEHAVIOUR_VARS:
        env.pop(var, None)
    if "GAIA_PROVIDER" in cfg_env:
        for var in MODEL_VARS:
            env.pop(var, None)
    env.update(cfg_env)
    env.pop("GAIA_ALLOW_PAID", None)  # ablations never enable paid models
    if run_id:
        env["GAIA_RUN_ID"] = run_id
    return env


def run_config(cfg: dict, run_id: str, *, budget: int | None, only: str | None, limit: int | None,
               timeout: float | None, dry_run: bool = False) -> dict | None:
    cmd = [sys.executable, "-m", "evals.run_practice", "--run-id", run_id, "--resume"]
    if budget is not None:
        cmd += ["--max-requests", str(budget)]
    if only:
        cmd += ["--only", only]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if timeout:
        cmd += ["--timeout", str(timeout)]
    env = config_env(cfg, run_id=run_id)
    if dry_run:
        print("   $", " ".join(f"{k}={v!r}" for k, v in cfg.get("env", {}).items()), " ".join(cmd[1:]))
        return None
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    p = RESULTS_DIR / f"{run_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run and compare agent configurations on the practice set.")
    ap.add_argument("--configs", help="comma list of config names (default: configs with default=true)")
    ap.add_argument("--group", help="run every config in this group (tools|architecture|model)")
    ap.add_argument("--tag", default=dt.date.today().strftime("%Y%m%d"), help="ablation id (default: today)")
    ap.add_argument("--only", help="task ids or prefixes, passed to run_practice")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-requests", type=int, help="total model-call budget across all configs")
    ap.add_argument("--est-calls", type=float, default=DEFAULT_EST_CALLS, help="planning estimate per task")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--plan", action="store_true", help="print the plan and cost estimate, run nothing")
    ap.add_argument("--aggregate-only", action="store_true", help="rebuild the table from existing results")
    ap.add_argument("--configs-dir", default=str(CONFIGS_DIR))
    args = ap.parse_args(argv)

    all_cfgs = load_configs(Path(args.configs_dir))
    chosen = select_configs(all_cfgs, args.configs, args.group)
    tools = known_tool_names()
    for cfg in chosen:
        bad = validate_config(cfg, tools)
        if bad:
            raise SystemExit(f"config {cfg['name']}: {'; '.join(bad)}")
    tasks = select_tasks(load_tasks(TASKS_PATH), args.only, args.limit)

    runnable = []
    for cfg in chosen:
        gap = requirement_gap(cfg)
        if gap:
            cfg["_skip_reason"] = gap
        else:
            runnable.append(cfg)
    plan = [(cfg, rid) for cfg in runnable for rid in run_ids_for(cfg, args.tag)]
    est = round(len(plan) * len(tasks) * args.est_calls)
    print(f"ablation {args.tag}: {len(tasks)} task(s) x {len(plan)} config-run(s) "
          f"~= {est} model calls at {args.est_calls}/task; budget {args.max_requests or 'none'}")
    for cfg in chosen:
        status = f"SKIP ({cfg['_skip_reason']})" if cfg.get("_skip_reason") else f"{cfg.get('repeats', 1)} run(s)"
        print(f"  - {cfg['name']:<26} [{cfg.get('group')}] {status}: {cfg.get('description', '')[:90]}")
    if args.max_requests is not None and est > args.max_requests:
        print(f"  note: the estimate exceeds the budget; later configs will be cut short (re-run to resume).")

    if not args.aggregate_only:
        remaining = args.max_requests
        for i, (cfg, rid) in enumerate(plan):
            share = None
            if remaining is not None:
                share = remaining // (len(plan) - i)
                if share < 3:
                    print(f"  budget exhausted before {rid}; re-run the same command to resume")
                    break
            print(f"\n>> {rid}  (budget share {share if share is not None else 'none'})")
            doc = run_config(cfg, rid, budget=share, only=args.only, limit=args.limit, timeout=args.timeout,
                             dry_run=args.plan)
            if doc is None:
                if remaining is not None:
                    remaining -= share  # plan mode: assume each config spends its full share
                continue
            if remaining is not None:
                remaining -= int(doc.get("budget", {}).get("calls_this_invocation") or 0)
            stopped = doc.get("budget", {}).get("stopped") or ""
            if stopped.startswith("provider quota"):
                print(f"  provider quota hit ({stopped[:120]}); stopping. Re-run after the reset to resume.")
                break
        if args.plan:
            return 0

    rep = aggregate(chosen, args.tag, RESULTS_DIR, tasks)
    md = to_markdown(rep)
    out = RESULTS_DIR / f"ablation_{args.tag}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"-> {out.with_suffix('.json')}\n-> {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
