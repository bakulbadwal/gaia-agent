# Reliability case study: one flaky task, and why

Accuracy says how often the agent is right on one try. Someone who relies on it needs to know if it's right **every** time. This page measures that on the 16-task practice set, finds the one task that flips between runs, and traces each flip to its cause. Every number on this page is recomputed from committed files by [`analysis/reliability.py`](../analysis/reliability.py).

| | |
|---|---|
| **Attempts** | 4 identical runs × 16 tasks = 64, on 2026-10-10 |
| **pass@1** (mean accuracy) | **95.3%**: the runs scored 100%, 93.8%, 93.8% and 93.8% |
| **pass@4** (solved at least once) | **100%** |
| **pass^4** (solved in all 4 runs) | **93.8%**: 15 of 16 tasks (Wilson 95% interval 71.7% to 98.9%) |
| **Flaky tasks** | 1: reading a chess diagram, solved 1 time in 4 |
| **Format gap** (right in substance, wrong in form) | 0 in all 8 runs, ablations included |

**The short version:** every failure in the repeat runs is the chess task. Its vision calls are about 10× slower than those of the other image task, and its one success finished 10 s inside the 600 s limit. When the agent did get a correct reading of the board, it twice turned that reading into the wrong move. The answer formatter is not the problem.

## What was measured

The same configuration (single CodeAgent, Nemotron 3 Ultra for reasoning, Nemotron 3 Nano Omni for vision, all tools, no verifier) ran the practice set 4 times. Run 4 is the ablation baseline, an identical config. For each task, *c* is the number of correct runs out of *n* = 4, scored by the GAIA quasi-exact-match port in [`evals/scorer.py`](../evals/scorer.py). Both metrics are the unbiased estimators from [`evals/passk.py`](../evals/passk.py), averaged over tasks:

- **pass@k = 1 − C(n−c, k) / C(n, k)**: the chance that at least one of k tries is right ([Chen et al., 2021](https://arxiv.org/abs/2107.03374)). This is the ceiling if something could pick the right answer.
- **pass^k = C(c, k) / C(n, k)**: the chance that all k tries are right ([Yao et al., 2024](https://arxiv.org/abs/2406.12045), tau-bench). This is the reliability number.

| k | pass@k | pass^k |
|---|---|---|
| 1 | 95.3% | 95.3% |
| 2 | 96.9% | 93.8% |
| 3 | 98.4% | 93.8% |
| 4 | 100% | 93.8% |

The gap between the two columns is all one task: the chess task has c = 1 of 4, and every other task has c = 4 of 4.

**Where the data lives:**

| Data | Path |
|---|---|
| One result file per run (answer, gold, correct, seconds, steps, error, timed_out per task) | `evals/results/practice-baseline-r1.json`, `-r2`, `-r3`, `abl-2026-10-10-baseline.json` |
| The pooled pass^k report the README quotes | `evals/results/passk-practice-baseline.json` |
| Per-step traces (the agent's code, every tool call and what it returned, tokens, timing) | `docs/traces/<run_id>/<task_id>.json` |

```bash
python -m analysis.reliability             # prints every table on this page, redraws the chart
python -m pytest -q tests/test_reliability.py
```

The test fails if the recomputed numbers drift from `passk-practice-baseline.json` or from the README.

## Every task, every run

![Grid of 16 practice tasks against 8 runs. In the 4 repeat runs every cell is solved except the chess task, which timed out at 610 s in Run 2 and gave wrong moves Qd8# and Nxg6# in Runs 3 and 4. In the ablation configs the chess task timed out in the no-web, verifier and multi-agent runs, and the no-media run lost both audio tasks, the chart and the chess task to running out of steps, plus a timeout on the cipher task.](media/reliability_grid.png)

The four columns on the right are the ablation configs. They're a different setup, so they're never pooled into pass^k, but they show the same thing: whenever the vision tool was available (the repeat runs plus no web, verifier and multi-agent), the other 15 tasks were solved **105 of 105 times** and the chess task **1 of 7**.

## Why the chess task flips

The result files say *what* happened; the traces say why. The script assigns each failed attempt one cause from the result row and the trace: a timeout; a tool error (a tool returned an error or garbage); a retrieval miss (a web task answered without a web tool); a formatting failure (the format gap); or reasoning (the inputs were there, but the conclusion was wrong). The right-hand column comes from reading each trace, and it agrees with the automatic cause.

| Run | Result | Time | Cause | What the trace shows |
|---|---|---|---|---|
| 1 | **Qe8#** (correct) | 590 s | n/a | Two vision calls took 225 s and 315 s, and the first reading was wrong (no black bishop, king on h8). Python-chess raised an error before the agent found the mate. It finished **10 s before the watchdog**. |
| 2 | no answer | 610 s | **timeout** (tool errors behind it) | Of four vision calls, two failed when all 4 free vision models in the fallback chain failed (the last with a 429). One returned a wrong board (the black queen misplaced, plus a rook that isn't there), and one returned `<unk>` tokens. Two hand-written FENs raised errors in python-chess. 532 s of the 610 s were vision calls. |
| 3 | Qd8# (wrong) | 221 s | **reasoning** | The vision model listed all 15 pieces on the right squares. The agent's own code comment repeated that list, but it then wrote a FEN that matched neither (for example, the white queen on d5 and the black king on h8) and searched for mates on that board. |
| 4 | Nxg6# (wrong) | 250 s | **reasoning** | The vision reading was correct again, but the vision model's reply also speculated about knight checks. The agent answered with no python-chess check. |

**By cause, across the 3 failed repeat attempts:** 1 timeout (with tool errors and garbage output behind it), 2 reasoning, and none from retrieval or formatting. The 3 extra chess failures in the ablation configs were all timeouts: 864 s, 1,022 s and 898 s, spent mostly in the vision model.

Four findings carry over beyond this one task:

1. **The vision model is the latency outlier.** In the repeat runs, the median chess vision call took 163 s (8 calls), and the median bar-chart call (p10, same model) took 17 s. The median chess attempt took 420 s, and the median attempt at the other 15 tasks took 33 s. The usage summary embedded in the results site ([docs/index.html](index.html), built from `runs/usage.jsonl`, which isn't committed, so the script doesn't recompute it) shows the same thing across all of that day's runs: the vision model's median latency was 54 s, with a p90 of 397 s, 16 empty replies and 23 retries in 35 requests. The reasoning model's median was 5.8 s.
2. **Perception was not the deciding failure.** In Runs 3 and 4 the board was read correctly. The agent turned a correct observation into a wrong board, or skipped the check. A better vision model alone wouldn't have fixed those two runs.
3. **The watchdog doesn't preempt.** The 600 s per-task limit is checked in a step callback (`gaia_agent/agent.py`), so it can't stop a step that's still running. Every timeout overran it: by 10 s in Run 2, and by 264 to 422 s for the three ablation chess timeouts. The worst case was the no-media run's cipher task, which ran 1,360 s on a single step that generated 65,536 output tokens.
4. **The runs weren't fully independent.** Runs 1 and 2 overlapped the official GAIA run, and the chess attempts in Runs 3 and 4 overlapped each other. They all shared one free-tier quota, and Run 2's chess failure began with two rate-limited vision calls.

## How much to trust these numbers

- **The sample is small.** 15 of 16 tasks never failed, but the 95% interval on that rate runs from 71.7% to 98.9%. Four clean runs is weak evidence that a task is reliable: a task that fails 10% of the time still passes 4 of 4 with probability 0.9⁴ ≈ 66%.
- **One flaky task means one data point.** With a single flaky task, a fix can't be told apart from noise until it's re-run many times.
- **It's a practice set, not GAIA.** The tasks were written by the agent's builder, the inputs are clean, and the 4 Wikipedia answers are known to the model (see the no-web ablation). [evals/README.md](../evals/README.md#3-why-practice-numbers-are-not-gaia-numbers) covers the limits.
- **Free endpoints drift.** A re-run will not reproduce these exact numbers, but the traces record what happened on each attempt.

## What changed, and what the history can show

No agent code changed between the four repeat runs. The only commit that landed during them ([`7bed801`](https://github.com/bakulbadwal/gaia-agent/commit/7bed801)) changes the prompt for a *missing* attachment, and the practice runner never takes that path (it passes `file_name=None`). So the history has **no controlled before/after on pass^k**: these four runs are the baseline that the next fixes should be measured against. Here's what the history does show:

| Commit | Change | Effect on reliability | Evidence |
|---|---|---|---|
| `0728da5` | Code-step timeout raised from smolagents' 30 s to 300 s (1,200 s for a manager) | Slow but successful tool calls are no longer thrown away. The chess steps that called the vision model took 59 to 315 s each, far above the old limit. | Comment and constant in `gaia_agent/agent.py`, `test_code_steps_get_a_realistic_timeout`. No run under the old limit is committed. |
| `7bed801` | Fail fast when an attachment is missing: answer `unknown` instead of searching | Before the fix, each of the 5 GAIA attachment questions used all 12 steps hunting for its file. On the official run, once the files were fetched, the 4 non-image ones finished in 2–4 steps. That gain came from having the files; the new prompt stops the waste when a file is missing. | Commit message, code comment, `test_missing_attachment_fails_fast`, and the per-question rows in the site data. The pre-fix answers aren't committed. |
| `6dfb71c` → `99176b7` → `73f6077` | The 4th identical run was pooled into pass^k, and the README count was fixed from 3 to 4 | A measurement change, not an agent change: pass^3 93.8% became pass^4 93.8%, and mean accuracy went from 95.8% to 95.3% (sd 3.6 to 3.1 pts) | `git log -p evals/results/passk-practice-baseline.json` |
| (ablation) | Format verifier; manager + 2 specialists | Neither one helped. The chess task timed out under both, and both made tasks slower (98 s and 130 s per task, against 54 s). The verifier had nothing to fix, since the format gap was already 0. | `evals/results/ablation_2026-10-10.md` |

## What I'd do next

1. **A hard deadline on every vision call and every attempt.** Give `image_question` a wall-clock timeout, and make the watchdog preempt a step that's still running. Cap output tokens per step, too: one step ran to 65,536 tokens. The target is that no attempt runs past 600 s, so a slow model costs one task, not the run.
2. **Check transcriptions before reasoning on them.** When an answer depends on turning an image into structured data, the agent should check its own transcription. For the chess task: rebuild the piece list from its FEN, diff it against the vision output, and call `board.is_valid()` before searching for moves. That rule is general, and it would have caught Run 3.
3. **Re-measure with more runs, run one after another.** Run k = 8–10 repeats, never at the same time as other runs, and add more vision tasks so there's more than one flaky item to learn from. `analysis/reliability.py` turns each re-run into a before/after on pass^k.
