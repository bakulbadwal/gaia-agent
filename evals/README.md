# Evals

The course grades this agent once, on 20 questions, by exact match. That number cannot tell
you whether a change helped, which tool matters, or whether a correct answer was luck. This
directory is the instrument that can:

| File | What it does |
|---|---|
| `scorer.py` | A faithful port of the official GAIA quasi-exact-match scorer, plus the course API's strict rule. Every result is scored both ways. |
| `practice/` | 16 original GAIA-Level-1-style tasks with verifiable gold answers: 12 on generated files, 4 on live Wikipedia. |
| `run_practice.py` | Runs `gaia_agent.agent.solve` over the practice set with a budget guard, then writes `results/<run_id>.json`. |
| `passk.py` | pass@k and pass^k from repeated runs: is a correct answer reliable or lucky? |
| `ablate.py` + `configs/` | Runs configurations side by side (tools removed, architecture, verifier, other free models), with a paired significance test. |

```bash
PY=.venv/bin/python
$PY -m evals.run_practice --only p01,w03 --max-requests 20       # a smoke run
$PY -m evals.run_practice --max-requests 120                     # the whole set
$PY -m evals.run_practice --repeat 3 --run-id base               # base-r1..r3
$PY -m evals.passk --glob 'evals/results/base-r*.json'           # pass@k / pass^k
$PY -m evals.ablate --plan --max-requests 300                    # cost of an ablation, runs nothing
$PY -m evals.ablate --max-requests 300                           # default ablation set
$PY -m pytest tests/test_evals.py -m "not network"               # offline tests
```

## 1. Two scorers, because the graders disagree

**GAIA quasi-exact match** (`gaia_match`, `question_scorer`). The code is ported line for line
from the public leaderboard Space
([`scorer.py`](https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py),
last changed in commit `1349a17` on 2025-03-07, fetched 2026-10-10). Only the scoring code was
read; no dataset or answer file was opened. The rules:

- If the gold answer is a number, the answer has `$`, `%` and `,` removed and is compared as a
  float. `"$1,234.50"` matches `1234.5`, but `"17 km"` does not match `17`.
- If the gold answer contains `,` or `;`, it is a list. Both sides are split, the lengths must
  match, order matters, and each element is compared as a number or as a lowercased string with
  the spaces removed. Punctuation is kept inside list elements.
- Otherwise it is a string. Spaces and ASCII punctuation are removed and case is ignored, so
  `"Sea gull."` matches `seagull`.

The port keeps the reference's quirks on purpose. A port that "fixes" its reference is no longer
a port.

**Course strict match** (`exact_match`). The Unit 4 API decides the certificate. Its source
([`main.py`](https://huggingface.co/spaces/agents-course/Unit4_scoring/blob/main/main.py)) compares
`submitted.strip().lower() == ground_truth.strip().lower()`. Under that rule `"1,000"` ≠ `1000`,
`"Paris."` ≠ `Paris`, and `"a,b"` ≠ `"a, b"`.

Every result file reports both, plus `format_gap`: the number of answers that are right in
substance but wrong in form. A large gap means the next fix is to the answer normalizer, not to
the model.

## 2. The practice set

| id | input | answer format | the trap it sets |
|---|---|---|---|
| p01-xlsx-late-loans | xlsx | number | date window plus a strict `> 21` bound, with rows at exactly 21 days and on Feb 28 and Mar 31 |
| p02-xlsx-multisheet-sku | xlsx, 3 sheets | string | the biggest single row is a different SKU from the biggest total across sheets |
| p03-csv-evening-trips | csv | number | departures at exactly 18:00, trips of exactly 90 min, arrivals after midnight |
| p04-csv-pm25-ranking | csv | ordered list | treating blanks as zero changes the top three |
| p05-py-queue-sim | .py | number | the script's printed value is not the answer; you have to call it with other arguments |
| p06-py-invert-cipher | .py | string | you have to invert the function, not run it |
| p07-audio-platform | mp3 (TTS) | number | four platform numbers are spoken, and the first Preston train is cancelled |
| p08-audio-mean-reading | mp3 (TTS) | number | transcribe, then compute and round |
| p09-image-chess-mate | png | SAN | read the board from the image; four other checking moves are not mate |
| p10-image-chart-mean | png | string | read 12 bar labels, then compute; there is no single value to look up |
| p11-pdf-action-items | pdf | number | two people called Priya, and a due date that falls exactly on the cutoff |
| p12-json-top-customer | json | string | a large cancelled order makes the overall leader the wrong answer |
| w01-wiki-element-city | Wikipedia | number | 2 hops: the Latin name of Stockholm, then holmium |
| w02-wiki-architect-prize | Wikipedia | year | 2 hops: the Opera House, then Utzon |
| w03-wiki-bolivia-borders | Wikipedia | ordered list | a 5-item list in alphabetical order |
| w04-wiki-mozart-river | Wikipedia | string | 2 hops: Mozart's birthplace, then its river |

**How each gold answer is known to be right.**

- *Generated files.* `practice/build_practice.py` writes each file with a fixed seed, then
  derives the gold answer by reading the file back from disk. It asserts that the derived answer
  matches the intended one, and that every trap really changes the answer (for example, that
  filling blanks with zero produces a different top three).
- *Independent re-derivation.* `tests/test_evals.py` derives every gold answer again with
  different code: openpyxl, csv, pypdf and json directly, where the generator uses pandas.
- *Images.* The test re-renders both PNGs from their stored FEN and bar values and compares them
  pixel by pixel with the committed files. This ties each image to its gold answer.
- *Audio.* The test transcribes both audio files with faster-whisper. `manifest.json` stores a
  sha256 for every file, so a rebuilt file that has not been re-verified fails the test.
- *Reproducibility.* Rebuilding the set with the same generator produces byte-identical files
  (checked by rebuilding twice and diffing `manifest.json`). Audio is the exception: it is only
  re-synthesized on request with `--audio`, on a Mac.
- *Wikipedia.* All four tasks were checked against the live pages on 2026-10-10. Each stores the
  page URL, the revision id that was read, and the supporting sentence.

**Contamination guard.** None of the 16 tasks is taken from GAIA or reworded from the course's
20 questions. A network test fetches the course *questions* (never answers) and requires every
practice question to have token Jaccard similarity below 0.35 with each of them. The highest
measured value is 0.20.

**Isolation.** The runner copies each attachment to `runs/practice_files/<task_id>/` before the
agent sees it. The agent is never pointed inside `evals/`, where `tasks.jsonl` holds the gold
answers.

## 3. Why practice numbers are not GAIA numbers

Read any accuracy from this directory as an **engineering signal, not a benchmark result**.

1. **The sample is tiny.** With n = 16, one task is worth 6.25 points. The 95% Wilson interval
   around 50% runs from 28% to 72%. Every result file reports this interval, and `ablate.py`
   reports an exact McNemar test so that a one-task change does not look like an effect. The
   course's own 20 questions have the same problem: 30% means 6 answers right.
2. **The same person wrote the tasks and built the agent.** The tasks test the failure modes
   the builder already thought of. GAIA's questions were written by people who had never seen
   this agent.
3. **The inputs are cleaner than real GAIA files.** The generated files have tidy layouts, the
   audio is synthetic text-to-speech, and both images were rendered from data. GAIA's files are
   messier and its web questions are more ambiguous. In the other direction, GAIA has nothing
   like the deliberate traps here. Neither set is a calibrated proxy for the other.
4. **The Wikipedia facts are stable and famous.** A large model may already know them, so a
   correct answer may not involve any tool. The `no_web` ablation measures how much this
   happens.
5. **It is a development set.** If prompts are tuned until this set passes, it stops measuring
   anything. The held-out test is still the course's 20 questions, and no prompt may be tuned
   on any gold answer.

The direction of a change is what transfers. If removing the Wikipedia tools costs three
Wikipedia tasks here, they matter. If the format gap is 3, the normalizer needs work.

## 4. Metrics

Each `results/<run_id>.json` follows the shared schema
(`{run_id, config, set, items[], summary}`). It also records the start and finish times, the
budget state, and a breakdown by skill and by answer format. These are the metrics:

- **accuracy / accuracy_strict**: GAIA quasi-exact match and course strict match, each with a
  Wilson 95% interval.
- **format_gap**: answers counted correct by the GAIA scorer but not by the strict one.
- **mean_steps, mean_seconds, total_tokens, model_calls**: cost, taken from the result or the
  trace (`runs/traces/<run_id>/`). The budget guard counts calls from the usage log
  (`runs/usage.jsonl`).
- **pass@k** (Chen et al. 2021): `1 - C(n-c, k) / C(n, k)`, the chance that at least one of k
  tries is correct. This is the ceiling if a verifier or a selection step can pick the right
  answer.
- **pass^k** (Yao et al. 2024, tau-bench): `C(c, k) / C(n, k)`, the chance that all k tries are
  correct. This is the reliability figure.

  Both are unbiased estimates from c successes in n runs. At n = 5, c = 2, k = 2 the naive
  `1 - (1 - p)^k` gives 0.64 and the unbiased form gives 0.70. `passk.py` also lists the flaky
  tasks (0 < c < n), which is where work pays off fastest.

## 5. Budget

Free tiers are the binding constraint. OpenRouter's free models allow 50 requests a day on an
account with fewer than 10 credits, and that allowance is shared by every free model. One
practice task is estimated (not yet measured) to cost about 3–12 model calls, so the full set
would cost about 100. The tools handle
this as follows:

- `--max-requests` sets a budget. The runner counts the calls the run has actually made (from
  the usage log, then the traces, then the step counts). It does not start a task with fewer
  than 3 calls left, and it caps each task's `max_steps` at what remains.
- A quota error from the provider stops the run. The task is marked `skipped`, not wrong, so
  quota exhaustion never shows up as low accuracy. `--resume` continues later.
- `ablate.py --plan` prints the call estimate before anything runs. The budget is split evenly
  across configs, and unused calls carry over to the next config. Run ids are deterministic
  (`abl-<tag>-<config>`), so rerunning the same command after the daily reset picks up where it
  stopped.
- Each config runs in its own subprocess. Comparisons use only the tasks that both configs
  actually scored.

## 6. Status

`results/` holds the redacted, aggregate results of the committed runs (raw run data stays in
the gitignored `results/raw/`). Any accuracy quoted in the project README must name its run id,
so the claim can be traced to a file. [`docs/RELIABILITY.md`](../docs/RELIABILITY.md) recomputes
the pass^k numbers from these files.
