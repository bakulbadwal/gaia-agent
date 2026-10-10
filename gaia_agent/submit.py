"""Submit cached answers from runs/answers.jsonl to the Unit 4 scoring API.

DRY RUN by default: prints the payload summary. Only POSTs when --yes is passed.

    python -m gaia_agent.submit                     # dry run
    python -m gaia_agent.submit --space gaia-agent --yes
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from .agent import normalize_answer
from .config import API_URL, DATA_DIR, RUNS_DIR

USERNAME = "bacool"
DEFAULT_SPACE = "gaia-agent"
ANSWERS_PATH = RUNS_DIR / "answers.jsonl"


def agent_code_url(space: str = DEFAULT_SPACE, username: str = USERNAME) -> str:
    space_id = space if "/" in space else f"{username}/{space}"
    return f"https://huggingface.co/spaces/{space_id}/tree/main"


def latest_answers(path: Path = ANSWERS_PATH) -> dict[str, dict]:
    """Latest record per task_id, preferring the latest one that has a non-empty answer.

    A later errored/empty attempt never hides an earlier usable answer.
    """
    path = Path(path)
    latest_any: dict[str, dict] = {}
    latest_good: dict[str, dict] = {}
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = rec.get("task_id")
        if not tid:
            continue
        latest_any[tid] = rec
        if normalize_answer(rec.get("answer")):
            latest_good[tid] = rec
    return {tid: latest_good.get(tid, rec) for tid, rec in latest_any.items()}


def build_payload(
    path: Path = ANSWERS_PATH,
    username: str = USERNAME,
    space: str = DEFAULT_SPACE,
    agent_code: str | None = None,
    task_ids: list[str] | None = None,
) -> dict:
    """{username, agent_code, answers: [{task_id, submitted_answer}]} from the answers cache."""
    recs = latest_answers(path)
    order = task_ids or list(recs)
    answers = [
        {"task_id": tid, "submitted_answer": normalize_answer(recs[tid].get("answer"))}
        for tid in order
        if tid in recs
    ]
    return {
        "username": username.strip(),
        "agent_code": agent_code or agent_code_url(space, username),
        "answers": answers,
    }


def post_payload(payload: dict, api_url: str = API_URL, timeout: float = 120) -> dict:
    resp = requests.post(f"{api_url}/submit", json=payload, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        body = {"text": resp.text[:1000]}
    if resp.status_code >= 400:
        raise RuntimeError(f"submit failed with HTTP {resp.status_code}: {body}")
    return body


def _question_order() -> list[str] | None:
    qpath = DATA_DIR / "questions.json"
    if not qpath.is_file():
        return None
    try:
        return [q["task_id"] for q in json.loads(qpath.read_text(encoding="utf-8")) if q.get("task_id")]
    except Exception:
        return None


def main(argv: list[str] | None = None) -> dict | None:
    ap = argparse.ArgumentParser(description="Submit cached GAIA answers (dry run unless --yes).")
    ap.add_argument("--space", default=DEFAULT_SPACE, help="HF Space name (or owner/name)")
    ap.add_argument("--username", default=USERNAME)
    ap.add_argument("--path", default=str(ANSWERS_PATH), help="answers.jsonl path")
    ap.add_argument("--yes", action="store_true", help="actually POST to the scoring API")
    args = ap.parse_args(argv)

    order = _question_order()
    payload = build_payload(Path(args.path), username=args.username, space=args.space, task_ids=order)

    print(f"username:   {payload['username']}")
    print(f"agent_code: {payload['agent_code']}")
    print(f"answers:    {len(payload['answers'])}" + (f" of {len(order)} questions" if order else ""))
    for a in payload["answers"]:
        print(f"  {a['task_id'][:8]}  {a['submitted_answer'][:70]!r}")
    if order:
        missing = [t for t in order if t not in {a['task_id'] for a in payload['answers']}]
        if missing:
            print(f"missing ({len(missing)}): " + ", ".join(t[:8] for t in missing))
    empty = [a["task_id"][:8] for a in payload["answers"] if not a["submitted_answer"]]
    if empty:
        print(f"empty answers ({len(empty)}): " + ", ".join(empty))

    if not args.yes:
        print("\nDRY RUN: nothing sent. Re-run with --yes to submit.")
        return None
    if not payload["answers"]:
        print("No answers to submit.")
        return None
    print(f"\nPOST {API_URL}/submit ...")
    result = post_payload(payload)
    print(json.dumps(result, indent=2))
    run_ids = sorted({r.get("run_id") for r in latest_answers(Path(args.path)).values() if r.get("run_id")})
    record = {
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_ids[-1] if len(run_ids) == 1 else ",".join(run_ids),
        "agent_code": payload["agent_code"],
        "response": {k: result.get(k) for k in ("score", "correct_count", "total_attempted", "message") if k in result},
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "submissions.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return result


if __name__ == "__main__":
    main()
