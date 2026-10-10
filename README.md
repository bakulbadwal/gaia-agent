# gaia-agent

An agent for the final assignment of the Hugging Face Agents Course, Unit 4: 20 GAIA Level-1
validation questions, scored by exact match. The pass mark for the certificate is 30% (6 of 20).

It is a single [smolagents](https://github.com/huggingface/smolagents) 1.26.0 `CodeAgent`. The
model writes Python, calls tools from that Python, and hands back a short final answer, which is
then normalized for exact-match scoring. Every model it uses is free: a free API tier or a local
model.

**Score:** TBD (no full run yet).

> Draft README, 2026-10-10. The base agent is integrated and tested offline. No real model call
> has been made yet, so answer quality is still unmeasured.

## How it works

```
question (+ attached file) ──► CodeAgent (max 12 steps, temperature 0)
                                  │  writes Python that calls tools
                                  ├─ web:   web_search, visit_webpage, wikipedia_*
                                  ├─ media: read_file, run_python_file, transcribe_audio,
                                  │         image_question, youtube_*
                                  └─ final_answer ─► shape check ─► normalize_answer ─► runs/answers.jsonl
```

- **The system prompt** is GAIA's format rules: numbers with no units or thousands separators,
  minimal strings, comma-separated lists, and the question's own formatting instructions
  followed exactly. It also includes working habits: open the attached file first, prefer the
  Wikipedia tools for encyclopedic facts, count by printing the items, do arithmetic in Python.
- **The answer check** (`final_answer_checks`) refuses an empty answer or an essay-length one, so
  the agent has to retry.
- **`normalize_answer`** strips "FINAL ANSWER:", wrapping quotes or bold, a trailing period
  (abbreviations keep theirs) and thousands separators, and tidies comma lists.

## Tools

| Tool | What it does |
|---|---|
| `web_search(query, max_results=8)` | Searches DuckDuckGo through `ddgs`, falling back to other engines. Results that hit the leak guard are removed. |
| `visit_webpage(url, start=0, max_chars=15000)` | Converts HTML to markdown with real tables, and reads PDFs page by page. Wikipedia URLs go through the MediaWiki API. Long pages are paged. |
| `wikipedia_search(query)` | Searches titles through the MediaWiki API. |
| `wikipedia_page(title, section="", ...)` | Renders a page or one section as markdown with expanded tables; also lists the sections. |
| `wikipedia_revision(title, as_of, ...)` | Gets the latest revision on or before a date, for questions that ask "as of <date>". |
| `wikipedia_page_info(title)` | Gets the creation date and creator, the latest revision, the talk page, Featured or Good article status, and article-history templates. |
| `read_file(path)` | Reads xlsx, xls or csv (as tables, with sums, category values and cell fill colours), py, txt or json, pdf, docx, pptx, and zip. |
| `run_python_file(path, timeout_s=30)` | Runs an attached `.py` in a subprocess and returns its exit code, stdout and stderr. |
| `transcribe_audio(path, timestamps=False)` | Transcribes with local faster-whisper (`small` model, CPU, cached). |
| `image_question(path, question)` | Asks the vision model about a local image. |
| `youtube_transcript(url)` | Fetches YouTube captions through yt-dlp, manual before automatic, de-duplicated and timestamped. |
| `youtube_audio_transcript(url)` | Downloads only the audio and transcribes it with Whisper, for videos without captions. |
| `youtube_frames_question(url, question, num_frames=16)` | Samples frames with ffmpeg and sends them to the vision model in batches of 8. |

Every tool returns a string and never raises. A failure comes back as `ERROR: ...`, so the
agent can read it and choose another tool.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
brew install ffmpeg            # Whisper decoding and video frames; the Space uses packages.txt
```

Keys are read from the environment, and from `KEY=VALUE` lines in `~/.claude/.env` if that file
exists. Key values are never printed.

### Model presets (`GAIA_PROVIDER`)

| Preset | Endpoint | Key env | Default model |
|---|---|---|---|
| `gemini` (default) | Google AI Studio, OpenAI-compatible | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| `groq` | Groq | `GROQ_API_KEY` | `openai/gpt-oss-120b` for text, `meta-llama/llama-4-scout-17b-16e-instruct` for vision |
| `lmstudio` | `http://localhost:1234/v1` | none | the first model the server lists |

Other environment variables:
- `GAIA_MODEL_ID` and `GAIA_VISION_MODEL_ID` override the models.
- `GAIA_BASE_URL` overrides the endpoint.
- `GAIA_RPM` caps requests per minute on the client side. The defaults are 8 for gemini and 25 for groq.
- `GAIA_TEMPERATURE` sets the temperature (default 0).
- `GAIA_WHISPER_MODEL` picks the Whisper model (default `small`).

To check which model ids a provider actually serves, run
`python -c 'from gaia_agent.config import list_models; print(list_models())'`. It is a free
metadata call.

An OpenRouter free-tier layer is written in `gaia_agent/providers.py`: rate limiting, a daily
budget guard, fallback chains and usage logging. It is **not wired in yet**.

## Running

```bash
python run.py                       # every question without a clean cached answer
python run.py --only 8e86 a1e9      # task_id prefixes, space- or comma-separated
python run.py --redo --limit 3      # re-run, at most 3 questions
python run.py --timeout 600         # per-question wall clock in seconds (default 420)
```

`run.py` caches the questions in `data/questions.json` and appends one JSON line per attempt to
`runs/answers.jsonl`, with the answer, raw output, steps, error, seconds and provider info. It also
writes each trace to `runs/traces/`. On a rerun it skips any task whose latest record has a clean
answer.

```bash
python submit.py                    # DRY RUN: prints the payload, sends nothing
python submit.py --yes              # POSTs to the scoring API
```

`submit.py` sends the latest non-empty answer for each task, in question order, as user `bacool`.
The `agent_code` is the Space's `/tree/main` URL. `app.py` is the course's Gradio template wired
to the same code. Its second button submits cached answers without re-running the agent.

```bash
.venv/bin/python -m pytest -q tests                 # all tests, network included
.venv/bin/python -m pytest -q tests -m "not network"
```

## Honesty rules

1. **The agent solves every question itself.** No answers are hard-coded anywhere: not in code,
   not in prompts, not in fixtures. Prompts are not tuned toward any known answer, and fixes after
   a run must be general (tools, format rules), never tied to one question.
2. **No answer lookups.** An answer-leak guard refuses any URL, search query or search result
   that mentions "gaia", every Hugging Face dataset or Space URL (`huggingface.co/datasets/`,
   `/spaces/`, `hf.co/...`, `*.hf.space`), and anything containing `agents-course`. The guard
   covers the web tools, and it also covers URLs opened from the agent's own Python:
   `pandas.read_*` and `numpy.genfromtxt` go through a guarded `urllib.request.urlopen` and
   `fsspec.open`, which also refuse every `hf://` path. `requests` and `urllib` cannot be imported
   in the agent's sandbox.
3. **No paid API calls.** Only free tiers and local models are used.
4. **Submissions go only through the course API, and only by hand**, with `python submit.py --yes`.

## Known limitations

- **The course API's `GET /files/{task_id}` returns 404 for all five questions with an
  attachment** (png, two mp3s, py, xlsx; checked 2026-10-10). The agent is told the file could
  not be downloaded. Until the API serves those files, the realistic ceiling is 15 of 20.
- The leak guard blocks by keyword. A copied answer list on a page that never says "gaia" and
  isn't on a Hugging Face URL would get through.
- `transcribe_audio` uses Whisper `small` on CPU. Use `GAIA_WHISPER_MODEL=medium` if a transcript
  looks wrong.

## Layout

```
gaia_agent/
  config.py       provider presets, get_model(), vision_complete(), load_env()
  files.py        task attachment download + cache
  agent.py        build_agent(), solve(), normalize_answer(), the code-level leak guard
  tools_web.py    TOOLS_WEB
  tools_media.py  TOOLS_MEDIA
  run.py submit.py
  providers.py trace.py   v2 (reliability + observability), being wired in
app.py            HF Space (Gradio + OAuth)
tests/            pytest; `network` marker for live tests
```
