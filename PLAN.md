# GAIA agent — Unit 4 of the Hugging Face Agents Course

*Plan written 2026-10-10. Approved by Bakul the same day:*
- *build it with multi-agent help;*
- *no paid API spend;*
- *a public HF Space plus a leaderboard submission;*
- *a public GitHub repo.*

## Goal

An agent that scores **≥ 30%** (6 of 20) on the course's Unit 4 set, which is 20 GAIA Level-1 validation questions scored by exact match. That earns the Certificate of Completion, which Bakul claims himself at `agents-course/Unit4-Final-Certificate`.

## Rules

1. **The agent must solve every question itself.** No hard-coded answers. No looking up GAIA ground truth: the agent's web tools block the GAIA dataset and known answer-dump pages. Prompts may not be tuned to a specific known answer.
2. **No paid API spend.** The model is a free tier, or a local model in LM Studio.
3. Submit only through the course API: `POST /submit` with username `bacool` and the public Space code link.

## Design (smolagents 1.26.0, CodeAgent)

| Piece | Choice |
|---|---|
| **Model** | An OpenAI-compatible endpoint, chosen by env var. Presets: `gemini` (Google AI Studio free tier), `groq` (free tier) and `lmstudio` (local, free). |
| **Web tools** | DuckDuckGo search, a page visitor (HTML → markdown, truncated), and Wikipedia search, page and revision tools through the MediaWiki API. A domain blocklist covers GAIA answer sources. |
| **Media tools** | YouTube captions (yt-dlp), video frames (ffmpeg) sent to the vision model, audio transcription with local faster-whisper, an image question to the vision model, and file reading for xlsx, csv, py, txt and json. |
| **Task files** | `GET /files/{task_id}`, cached under `data/files/`. |
| **Answer format** | A final normalizer that strips "FINAL ANSWER:", trailing periods and quotes, and keeps comma lists tidy. The prompt follows GAIA's format rules (numbers without units or commas unless asked; minimal strings). |
| **Runner** | `run.py` caches each answer and its trace in `runs/answers.jsonl`, resumes, applies a per-question timeout, and accepts `--only <ids>`. |
| **Submit** | `submit.py` does a dry run by default, and `--yes` sends. |
| **Space** | `app.py` is the course template wired to `gaia_agent`, with cached answers so the Space doesn't re-run everything. |

## Build order

1. The project venv (done) and this plan.
2. **Parallel builders** (a workflow):
   - core (`config`, `agent`, `run`, `submit`, `app`);
   - web tools;
   - media and file tools.

   Each piece gets offline tests.
3. Integration: one agent wires everything together, runs the tests, and dry-runs one question per tool type once a model is reachable.
4. A full run, then a review of the failures and fixes to tools or prompts. These are general fixes, never answer-specific.
5. Create the public Space, submit, and record the score.
6. The public GitHub repo, with a README that has the score, the design and a walkthrough Bakul can explain in an interview.

## v2 scope: a portfolio piece, not just another submission (added 2026-10-10 12:50)

Bakul asked for a project that clearly shows production agent engineering. On top of the base agent:

1. **Observability.** Every run saves a full trace: each step's reasoning, code, tool calls, observations, tokens and timing (`gaia_agent/trace.py`).
2. **A results site.** GitHub Pages hosts the scoreboard, a step-by-step trajectory viewer for every question, tool-usage and latency charts, the failure taxonomy, evals and the architecture (`report/` builds `docs/`).
3. **An eval suite** (`evals/`):
   - an exact port of GAIA's official quasi-exact scorer, plus strict exact match;
   - a 16-task original practice set with verifiable gold answers;
   - pass@k and pass^k (reliability across repeat runs);
   - an ablation runner (tools off, single vs multi-agent, verifier on/off, model swaps).
4. **Reliability engineering** (`gaia_agent/providers.py`):
   - free-tier OpenRouter rate limiting and a daily budget guard;
   - backoff, then fallback to the next model;
   - per-call usage logging;
   - routing text questions and vision questions to different models.
5. **Error analysis.** Each trace gets automatic failure tags (`analysis/error_analysis.py`).
6. **Architecture options.** A single CodeAgent, or a manager agent with web-researcher and media-analyst specialists; an optional verifier pass; a GAIA answer-leak guard.

Waves:
- **wave 1:** the base agent;
- **wave 2:** the modules above, written to new files in parallel;
- **wave 3:** an integrator wires them in, then real runs on the practice set, then the GAIA run and submission, then the README and site.

## Model access (blocker)

*Resolved 12:46: Bakul chose OpenRouter free models. He is adding `OPENROUTER_API_KEY` to `~/.claude/.env`.*

The free paths that need Bakul to act:
- a Google AI Studio key (no card; Flash-Lite allows ~500 requests a day);
- a Groq key (no card).

The free path that needs nothing from him is a local model in LM Studio (a ~18 GB download; slower and weaker).
