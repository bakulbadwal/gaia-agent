"""The GAIA agent: build_agent(), solve(), verify_answer(), normalize_answer().

Architectures (GAIA_ARCH):
    single  one CodeAgent with every web, media and file tool.
    multi   a manager CodeAgent (python + final_answer only) that delegates to two managed
            CodeAgents: ``web_researcher`` (web + Wikipedia tools) and ``media_analyst``
            (file, audio, image and YouTube tools).

Other env switches: GAIA_DISABLE_TOOLS (comma list of tool names removed from every agent),
GAIA_VERIFY=1 (one extra model call that may only REFORMAT the answer, never re-research it).

Every solve() writes a full trace to runs/traces/<run_id>/<task_id>.json (gaia_agent.trace).
"""

from __future__ import annotations

import hashlib
import re
import sys
import threading
import time
from typing import Any

from . import config
from .config import get_model
from .files import download_task_file

MAX_STEPS = 12
SUB_AGENT_MAX_STEPS = 8
# smolagents wraps every code step in a 30 s timeout, but the wrapper's thread pool waits for the
# code to finish anyway, so a slow step (audio transcription, video frames, a sub-agent run) is
# executed in full and then reported as a failure, its result thrown away. Observed live on
# 2026-10-10: a manager step calling web_researcher took 40 s and was marked failed. Steps get a
# realistic budget instead; the per-question watchdog still bounds the whole run.
CODE_TIMEOUT_S = 300  # one code step of a single agent or a specialist (GAIA_CODE_TIMEOUT)
MANAGER_CODE_TIMEOUT_S = 1200  # a manager step may run a whole specialist (GAIA_MANAGER_CODE_TIMEOUT)
WEB_AGENT_NAME = "web_researcher"
MEDIA_AGENT_NAME = "media_analyst"

AUTHORIZED_IMPORTS = [
    "pandas",
    "numpy",
    "math",
    "re",
    "json",
    "datetime",
    "collections",
    "itertools",
    "statistics",
    "chess",
    "openpyxl",
    "fractions",
    "unicodedata",
    "csv",
    "string",
]

_PREAMBLE = """\
You are answering a GAIA benchmark question. The grader compares your final answer to a short
ground-truth string by EXACT MATCH, so final_answer must contain ONLY the answer itself.

"""

FORMAT_RULES = """\
ANSWER FORMAT (follow strictly):
- Call final_answer with a plain string that is just the answer: no explanation, no sentence,
  no "FINAL ANSWER" prefix, no surrounding quotes, no trailing period.
- A number: digits only, no thousands separators, no units, currency or percent signs unless the
  question asks for them. Round only when the question says to, and exactly as it says.
- A string: as few words as possible. No articles (a, an, the) and no abbreviations unless the
  question asks for them. Write numbers as digits unless told to spell them out.
- A list: comma-separated ("x, y, z"), each element following the number/string rules above.
- Obey every formatting instruction in the question exactly: ordering (e.g. alphabetical), singular
  vs plural, first name / surname only, a code standard (e.g. IOC country code), a notation (e.g.
  algebraic chess notation), a number of decimal places, or a unit the question names.
- If you cannot fully verify the answer, still give your single best guess in this format.
"""

WORK_RULES = """\
HOW TO WORK:
- If the task includes "Attached file: <path>", inspect that file before anything else:
  read_file for spreadsheets, csv, code, text, json or pdf; transcribe_audio for audio;
  image_question for images. For an attached Python file, read it and use run_python_file to get
  its real output.
- For a YouTube URL, call youtube_transcript first (youtube_audio_transcript if there are no
  usable captions); use youtube_frames_question only when the question is about what can be seen
  in the video.
- For facts that live on Wikipedia, prefer the Wikipedia tools (search, page content, and the
  revision tools for "as of <date>" questions) over generic web search. For everything else use
  web search, then visit the most authoritative pages and read them; do not trust snippets alone.
- When the answer is a count, print every matching item explicitly with its evidence, then count
  the printed list in code. Apply date ranges and inclusive/exclusive bounds exactly as stated.
- Do arithmetic in Python, not in your head. Print short excerpts, not whole pages.
- Re-read the question before answering: check you answer what is asked, about the right entity,
  in the requested unit and format.
- Never look up this question in GAIA datasets, benchmark answer keys or leaderboards.
"""

INSTRUCTIONS = _PREAMBLE + FORMAT_RULES + "\n" + WORK_RULES

MANAGER_RULES = """\
HOW TO WORK (you are the MANAGER of a small team):
- You have no web, file or media tools yourself. You plan, delegate, check the evidence your team
  returns, do arithmetic / counting / sorting / formatting in Python, and call final_answer.
- Your team members are called like functions with one argument, task: a complete,
  self-contained instruction. They cannot see the original question, so include every name,
  date, constraint and unit that matters, and copy any attached file path or URL verbatim
  (for example "Attached file: /abs/path/file.xlsx").
  * web_researcher: web search, web pages (HTML and PDF), Wikipedia pages, search and
    historical revisions ("as of <date>").
  * media_analyst: attached files (spreadsheets, csv, pdf, docx, pptx, json, text, code; it can
    run Python files), audio transcription, image questions (vision model), YouTube captions,
    audio and frames.
- Ask for the raw facts and evidence you need (with sources), not for a formatted final answer.
  Delegate one focused sub-question at a time; if a report is incomplete or contradictory,
  send a sharper follow-up instead of guessing.
- When the answer is a count, have the evidence listed item by item, then count it in code.
  Apply date ranges and inclusive/exclusive bounds exactly as stated.
- Re-read the question before answering: check you answer what is asked, about the right entity,
  in the requested unit and format.
- Never look up this question in GAIA datasets, benchmark answer keys or leaderboards.
"""

MANAGER_INSTRUCTIONS = _PREAMBLE + FORMAT_RULES + "\n" + MANAGER_RULES

WEB_RESEARCHER_INSTRUCTIONS = """\
You are web_researcher, a research specialist working for a manager agent that answers a
benchmark question. Find the facts your task asks for.
- For facts that live on Wikipedia, prefer the Wikipedia tools (search, page content, and the
  revision tools for "as of <date>" questions) over generic web search. For everything else use
  web search, then visit the most authoritative pages and read them; do not trust snippets alone.
- When asked to count or list, print every matching item with its evidence.
- Print short excerpts, not whole pages. Do arithmetic in Python.
- Call final_answer with a concise report: the facts found (exact values copied verbatim), the
  source URL for each, and anything you could not verify. Do not pad it.
- Never look up the question in GAIA datasets, benchmark answer keys or leaderboards.
"""

MEDIA_ANALYST_INSTRUCTIONS = """\
You are media_analyst, a file and media specialist working for a manager agent that answers a
benchmark question. Extract exactly what your task asks for from the file, audio, image or video.
- Files: read_file for spreadsheets, csv, code, text, json, pdf, docx, pptx; for a Python file
  read it and use run_python_file to get its real output. Then compute in Python (pandas is
  available) and print the evidence (the matching rows / values), not only the result.
- Audio: transcribe_audio. Images: image_question (a vision model; ask it precise questions).
- YouTube: youtube_transcript first (youtube_audio_transcript if there are no usable captions);
  youtube_frames_question only for what can be seen in the video.
- Call final_answer with a concise report: the extracted facts / values (verbatim) and how you got
  them. Do not pad it.
- Never look up the question in GAIA datasets, benchmark answer keys or leaderboards.
"""

WEB_RESEARCHER_DESCRIPTION = (
    "Researches facts on the web. Has web search, web page reading (HTML and PDF) and Wikipedia "
    "tools, including Wikipedia page revisions as of a past date. Give it ONE self-contained "
    "research question with every name, date and constraint it needs; it returns the facts it "
    "found with source URLs."
)
MEDIA_ANALYST_DESCRIPTION = (
    "Analyses attached files and media. Reads spreadsheets, CSV, PDF, Word, PowerPoint, JSON, "
    "text and code files (and runs Python files), transcribes audio, answers questions about "
    "images with a vision model, and reads YouTube captions, audio and frames. Give it the "
    "absolute file path or URL verbatim plus exactly what to extract; it returns the facts."
)


# --------------------------------------------------------------------------- answer normalizing

_FINAL_PREFIX = re.compile(r"^(?:\*\*|__)?\s*final\s*answer\s*(?:\*\*|__)?\s*[:：\-–]?\s*(?:\*\*|__)?\s*", re.I)
_WRAPPERS = [("**", "**"), ("__", "__"), ("`", "`"), ('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("«", "»")]
_ABBREVIATIONS = {"jr", "sr", "inc", "ltd", "corp", "co", "etc", "vs", "st", "dr", "mr", "mrs", "ms", "bros"}
_INITIALISM = re.compile(r"(?:^|[\s,(])(?:[A-Za-z]\.){2,}$")  # D.C.  U.S.A.
_THOUSANDS_NUMBER = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")


def _strip_trailing_period(s: str) -> str:
    if not s.endswith(".") or s.endswith(".."):
        return s
    if _INITIALISM.search(s):
        return s
    last = re.split(r"[\s,(]+", s[:-1])[-1].lower() if s[:-1] else ""
    if last in _ABBREVIATIONS:
        return s
    return s[:-1].rstrip()


def _unwrap(s: str) -> str:
    for open_, close in _WRAPPERS:
        if len(s) >= len(open_) + len(close) + 1 and s.startswith(open_) and s.endswith(close):
            inner = s[len(open_) : len(s) - len(close)]
            if close not in inner and open_ not in inner:
                return inner.strip()
    return s


def normalize_answer(s: Any) -> str:
    """Clean a model answer for exact-match submission.

    Strips whitespace, a leading "FINAL ANSWER:" (any case), surrounding quotes/backticks/bold,
    a trailing period that is not part of an abbreviation, collapses whitespace, removes
    thousands separators from a lone number, and normalizes comma lists to ", ".
    """
    if s is None:
        return ""
    if isinstance(s, bool):
        return str(s)
    if isinstance(s, float):
        if s != s:  # NaN
            return ""
        if s.is_integer() and abs(s) < 1e15:
            return str(int(s))
        return repr(s)
    if isinstance(s, (list, tuple)):
        return ", ".join(x for x in (normalize_answer(v) for v in s) if x)
    s = str(s)
    for _ in range(6):
        before = s
        s = s.strip()
        s = _FINAL_PREFIX.sub("", s, count=1)
        s = _unwrap(s)
        s = _strip_trailing_period(s)
        if s == before:
            break
    s = re.sub(r"\s+", " ", s).strip()
    if _THOUSANDS_NUMBER.fullmatch(s):
        return s.replace(",", "")
    if "," in s:
        s = re.sub(r"\s*,\s*", ", ", s).strip().rstrip(",").strip()
    return s


# --------------------------------------------------------------------------- agent


def _check_answer_shape(final_answer, memory, agent=None) -> bool:
    """final_answer_checks hook: reject empty or essay-length answers so the agent retries."""
    text = normalize_answer(final_answer)
    if not text:
        raise ValueError("final_answer was empty. Provide the answer itself.")
    if len(text) > 300:
        raise ValueError(
            "final_answer is too long. Give ONLY the answer (a number, a few words, or a comma-separated list)."
        )
    return True


_guard_lock = threading.Lock()
_guard_installed = False


def _url_of(obj: Any) -> str | None:
    """Best-effort URL string from a path/URL/urllib Request argument."""
    url = getattr(obj, "full_url", obj)
    if isinstance(url, bytes):
        url = url.decode("utf-8", "replace")
    try:
        import os as _os

        url = _os.fspath(url) if not isinstance(url, str) else url
    except TypeError:
        return None
    return url if isinstance(url, str) else None


def install_code_guard() -> None:
    """Extend the answer-leak guard to URLs opened from the agent's own Python code.

    The web tools refuse GAIA / HF dataset / HF Space URLs, but the CodeAgent may import pandas and
    numpy, whose readers fetch URLs themselves (pd.read_csv("https://..."), np.genfromtxt(url)).
    Both go through urllib.request.urlopen, and pandas sends other remote schemes (hf://, s3://, ...)
    through fsspec.open. This wraps those choke points so a blocked URL raises PermissionError.
    Every other URL still works. Idempotent and process-wide.
    """
    global _guard_installed
    with _guard_lock:
        if _guard_installed:
            return
        import urllib.request

        from .tools_web import BLOCK_NOTE, is_blocked

        def _check(target: Any) -> None:
            url = _url_of(target)
            if not url or "://" not in url:
                return  # local paths (the project dir itself is named gaia-agent) are never blocked
            if is_blocked(url) or url.strip().lower().startswith("hf://"):
                raise PermissionError(BLOCK_NOTE)

        orig_urlopen = urllib.request.urlopen

        def guarded_urlopen(url, *args, **kwargs):
            _check(url)
            return orig_urlopen(url, *args, **kwargs)

        guarded_urlopen.__wrapped__ = orig_urlopen  # type: ignore[attr-defined]
        urllib.request.urlopen = guarded_urlopen

        try:
            import fsspec

            orig_open, orig_open_files = fsspec.open, fsspec.open_files

            def guarded_open(urlpath, *args, **kwargs):
                _check(urlpath)
                return orig_open(urlpath, *args, **kwargs)

            def guarded_open_files(urlpath, *args, **kwargs):
                for u in urlpath if isinstance(urlpath, (list, tuple)) else [urlpath]:
                    _check(u)
                return orig_open_files(urlpath, *args, **kwargs)

            fsspec.open, fsspec.open_files = guarded_open, guarded_open_files
        except ImportError:
            pass
        _guard_installed = True


def _tool_groups() -> tuple[list, list]:
    """(web tools, media + file tools) as the builder modules export them."""
    try:
        from .tools_web import TOOLS_WEB
    except Exception as e:  # pragma: no cover - depends on the web builder's module
        raise ImportError(f"Could not import TOOLS_WEB from gaia_agent.tools_web: {e!r}") from e
    try:
        from .tools_media import TOOLS_MEDIA
    except Exception as e:  # pragma: no cover - depends on the media builder's module
        raise ImportError(f"Could not import TOOLS_MEDIA from gaia_agent.tools_media: {e!r}") from e
    return list(TOOLS_WEB), list(TOOLS_MEDIA)


def _drop_disabled(tools: list, disabled: set[str]) -> list:
    return [t for t in tools if getattr(t, "name", None) not in disabled]


def _disabled_note(disabled: set[str], available: set[str]) -> str:
    """Tell the model which tools this configuration removed, so it does not call ghosts."""
    gone = sorted(disabled & available)
    if not gone:
        return ""
    return (
        "\nTOOLS DISABLED IN THIS CONFIGURATION (they do not exist here; ignore any advice above that "
        f"mentions them and work with what you have): {', '.join(gone)}.\n"
    )


def _code_timeout(env: str, default: int) -> int | None:
    import os

    raw = (os.environ.get(env) or "").strip()
    if not raw:
        return default
    val = int(float(raw))
    return val if val > 0 else None  # 0 = no per-step limit


def _warn_unknown(disabled: set[str], known: set[str]) -> None:
    unknown = sorted(disabled - known)
    if unknown:
        print(f"warning: GAIA_DISABLE_TOOLS names unknown tools {unknown}; known: {sorted(known)}", file=sys.stderr)


def build_agent(
    model=None,
    tools: list | None = None,
    max_steps: int = MAX_STEPS,
    step_callbacks=None,
    *,
    arch: str | None = None,
    disabled: list[str] | set[str] | None = None,
):
    """Return the configured agent (GAIA_ARCH single|multi; GAIA_DISABLE_TOOLS applied).

    ``tools`` overrides the single agent's tool list (tests pass []); it is ignored in multi mode,
    where the tool groups are fixed by role.
    """
    from smolagents import CodeAgent
    from smolagents.monitoring import LogLevel

    install_code_guard()
    arch = (arch or config.arch()).lower()
    disabled_set = set(disabled if disabled is not None else config.disabled_tools())
    model = model or get_model()
    if arch == "multi":
        return build_multi_agent(model, max_steps=max_steps, step_callbacks=step_callbacks, disabled=disabled_set)
    if arch != "single":
        raise ValueError(f"arch must be 'single' or 'multi', got {arch!r}")

    if tools is None:
        web, media = _tool_groups()
        tools = web + media
    known = {t.name for t in tools}
    if tools:
        _warn_unknown(disabled_set, known)
    return CodeAgent(
        tools=_drop_disabled(tools, disabled_set),
        model=model,
        instructions=INSTRUCTIONS + _disabled_note(disabled_set, known),
        max_steps=max_steps,
        verbosity_level=LogLevel.ERROR,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        final_answer_checks=[_check_answer_shape],
        step_callbacks=step_callbacks,
        executor_kwargs={"timeout_seconds": _code_timeout("GAIA_CODE_TIMEOUT", CODE_TIMEOUT_S)},
    )


def build_multi_agent(model, *, max_steps: int = MAX_STEPS, step_callbacks=None, disabled: set[str] | None = None,
                      sub_max_steps: int = SUB_AGENT_MAX_STEPS):
    """Manager CodeAgent (python + final_answer) with ``web_researcher`` and ``media_analyst``.

    All three share one model object, so they share one rate limiter, daily budget and usage
    log. A specialist whose every tool is disabled is left out of the team.
    """
    from smolagents import CodeAgent
    from smolagents.monitoring import LogLevel

    disabled = set(disabled or ())
    web, media = _tool_groups()
    _warn_unknown(disabled, {t.name for t in web + media})
    team = []
    common = dict(
        model=model,
        max_steps=max(1, min(sub_max_steps, max_steps)),
        verbosity_level=LogLevel.ERROR,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        provide_run_summary=False,
        executor_kwargs={"timeout_seconds": _code_timeout("GAIA_CODE_TIMEOUT", CODE_TIMEOUT_S)},
    )
    web_tools = _drop_disabled(web, disabled)
    if web_tools:
        team.append(CodeAgent(
            tools=web_tools,
            instructions=WEB_RESEARCHER_INSTRUCTIONS + _disabled_note(disabled, {t.name for t in web}),
            name=WEB_AGENT_NAME,
            description=WEB_RESEARCHER_DESCRIPTION,
            **common,
        ))
    media_tools = _drop_disabled(media, disabled)
    if media_tools:
        team.append(CodeAgent(
            tools=media_tools,
            instructions=MEDIA_ANALYST_INSTRUCTIONS + _disabled_note(disabled, {t.name for t in media}),
            name=MEDIA_AGENT_NAME,
            description=MEDIA_ANALYST_DESCRIPTION,
            **common,
        ))
    missing = [n for n in (WEB_AGENT_NAME, MEDIA_AGENT_NAME) if n not in {a.name for a in team}]
    note = ""
    if missing:
        note = (f"\nTEAM MEMBERS UNAVAILABLE IN THIS CONFIGURATION (do not call them): {', '.join(missing)}. "
                "Work with the team you have, or from what you know.\n")
    return CodeAgent(
        tools=[],
        model=model,
        managed_agents=team,
        instructions=MANAGER_INSTRUCTIONS + note,
        max_steps=max_steps,
        verbosity_level=LogLevel.ERROR,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        final_answer_checks=[_check_answer_shape],
        step_callbacks=step_callbacks,
        executor_kwargs={"timeout_seconds": _code_timeout("GAIA_MANAGER_CODE_TIMEOUT", MANAGER_CODE_TIMEOUT_S)},
    )


def _sub_agents(agent) -> list:
    out, stack = [], list((getattr(agent, "managed_agents", None) or {}).values())
    while stack:
        a = stack.pop()
        if a not in out:
            out.append(a)
            stack.extend((getattr(a, "managed_agents", None) or {}).values())
    return out


def _clip(x: Any, n: int) -> str | None:
    if x is None:
        return None
    s = x if isinstance(x, str) else str(x)
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more chars]"


def _trace(agent) -> tuple[int, list[dict]]:
    """Compact per-step summary kept on the solve() result (the full trace goes to disk)."""
    from smolagents.memory import ActionStep

    steps, trace = 0, []
    for st in getattr(getattr(agent, "memory", None), "steps", None) or []:
        if isinstance(st, ActionStep):
            steps += 1
            trace.append(
                {
                    "step": st.step_number,
                    "code": _clip(st.code_action, 4000),
                    "observations": _clip(st.observations, 3000),
                    "error": _clip(st.error, 1500) if st.error else None,
                    "final": bool(st.is_final_answer),
                }
            )
    return steps, trace


def build_task(question: str, file_path: str | None, file_name: str | None) -> str:
    task = question.strip()
    if file_path:
        task += f"\n\nAttached file: {file_path}"
    elif file_name:
        # Fail fast: in run gaia-v1 every question with a missing attachment burned all 12 steps
        # hunting for the file (read_file on guessed paths, web searches for the file name).
        task += (
            f"\n\n(An attached file named {file_name!r} was expected but could not be downloaded. "
            "It is NOT available anywhere: do not search for it, guess its path or try to reconstruct it. "
            "If the question can be answered without it, do so; otherwise call final_answer('unknown') now.)"
        )
    return task


# --------------------------------------------------------------------------- verifier (GAIA_VERIFY=1)

VERIFIER_PROMPT = """\
You check the FORMAT of an answer to a benchmark question that is graded by exact string match.

QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

FORMAT RULES THE GRADER EXPECTS:
{rules}
Your job is to fix the format only. You may: drop units, currency or percent signs the question
did not ask for; remove articles, filler words or a sentence wrapped around the answer; remove
thousands separators; reorder or re-delimit a list when the question asks (e.g. alphabetical,
comma-separated); change capitalization, singular/plural or number words vs digits when the
question asks; and apply rounding or decimal places the question explicitly requests.
You must NOT research, recompute, add information, or replace the answer with a different one.
If the candidate already follows the format, return it unchanged.

Reply with exactly one line and nothing else:
FINAL ANSWER: <answer>"""

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
}
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+", re.UNICODE)


def _tokens(s: str) -> list[str]:
    s = s.lower().replace(",", "")
    return [_NUMBER_WORDS.get(t, t) for t in _TOKEN_RE.findall(s)]


def _has_token(tok: str, pool: set[str]) -> bool:
    if tok in pool:
        return True
    variants = {tok + "s", tok + "es", tok[:-1] if tok.endswith("s") else None, tok[:-2] if tok.endswith("es") else None}
    if any(v and v in pool for v in variants):
        return True
    try:
        x = float(tok)
        return any(abs(float(p) - x) < 1e-9 for p in pool if re.fullmatch(r"\d+(?:\.\d+)?", p))
    except ValueError:
        return False


def _is_rounding(candidate: str, revised: str) -> bool:
    """revised is candidate rounded to revised's number of decimals (both lone numbers)."""
    num = r"[-+]?\d+(?:\.\d+)?"
    c, r = candidate.replace(",", "").strip(), revised.replace(",", "").strip()
    if not (re.fullmatch(num, c) and re.fullmatch(num, r)):
        return False
    decimals = len(r.split(".")[1]) if "." in r else 0
    return abs(float(c) - float(r)) <= 0.5 * 10 ** (-decimals) + 1e-9


def is_reformat(candidate: str, revised: str) -> tuple[bool, str]:
    """Whether ``revised`` only reformats ``candidate`` (the verifier may not change substance).

    Accepted: identical; every token of the revision already in the candidate (units, articles or
    wrapping words dropped, list reordered, case changed, number words <-> digits, plural/singular);
    or a lone number rounded. Anything that introduces new content is rejected.
    """
    if revised == candidate:
        return True, "unchanged"
    if not revised:
        return False, "verifier returned an empty answer"
    if len(revised) > 300:
        return False, "verifier answer too long"
    if _is_rounding(candidate, revised):
        return True, "rounded as the question asks"
    cand = set(_tokens(candidate))
    rev = _tokens(revised)
    if rev and all(_has_token(t, cand) for t in rev):
        return True, "format-only change"
    new = [t for t in rev if not _has_token(t, cand)]
    return False, f"rejected: verifier introduced new content {new[:5]}"


def _parse_verifier(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.match(r"^(?:\*\*|__)?\s*final\s*answer", ln, re.I):
            return normalize_answer(ln)
    return normalize_answer(lines[-1]) if lines else ""


def verify_answer(question: str, answer: str, model) -> dict:
    """One extra model call that may only reformat ``answer``. Never raises.

    Returns {answer, candidate, proposed, accepted, changed, reason, raw_output, model,
    input_tokens, output_tokens, seconds, error}; ``answer`` is what solve() should submit.
    """
    from smolagents.models import ChatMessage, MessageRole

    t0 = time.time()
    rec: dict = {"enabled": True, "candidate": answer, "answer": answer, "proposed": None, "accepted": False,
                 "changed": False, "reason": None, "raw_output": None, "model": getattr(model, "model_id", None),
                 "input_tokens": None, "output_tokens": None, "seconds": None, "error": None}
    prompt = VERIFIER_PROMPT.format(question=question.strip(), answer=answer, rules=FORMAT_RULES)
    try:
        msg = model.generate([ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])])
        raw = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        rec["raw_output"] = raw
        usage = getattr(msg, "token_usage", None)
        if usage is not None:
            rec["input_tokens"], rec["output_tokens"] = int(usage.input_tokens), int(usage.output_tokens)
        last = getattr(model, "last_call", None)
        if isinstance(last, dict) and last.get("model"):
            rec["model"] = last["model"]
        proposed = _parse_verifier(raw)
        rec["proposed"] = proposed
        ok, reason = is_reformat(answer, proposed)
        rec["accepted"], rec["reason"] = ok, reason
        if ok:
            rec["answer"] = proposed
            rec["changed"] = proposed != answer
    except Exception as e:  # a failed verifier must never cost the answer
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["reason"] = "verifier call failed; kept the candidate"
    rec["seconds"] = round(time.time() - t0, 3)
    return rec


def _add_verifier_to_trace(tr: dict, v: dict | None) -> None:
    """Record the verifier as its own step (before the final step) and count its model call."""
    if not v:
        return
    from .trace import clip

    tr["verifier"] = {k: (clip(val) if isinstance(val, str) else val) for k, val in v.items()}
    steps = tr.get("steps") or []
    step = {
        "index": None, "type": "verifier", "agent": "verifier",
        "model_output": clip(v.get("raw_output")), "code": None, "tool_calls": [],
        "observation": clip(f"{v.get('reason')} | candidate={v.get('candidate')!r} -> submitted={v.get('answer')!r}"),
        "error": clip(v.get("error")), "error_type": None,
        "input_tokens": v.get("input_tokens"), "output_tokens": v.get("output_tokens"),
        "duration_s": v.get("seconds"),
    }
    pos = len(steps) - 1 if steps and steps[-1].get("type") == "final" else len(steps)
    steps.insert(pos, step)
    for i, s in enumerate(steps):
        s["index"] = i
    tot = tr.setdefault("totals", {})
    tot["model_calls"] = (tot.get("model_calls") or 0) + 1
    tot["input_tokens"] = (tot.get("input_tokens") or 0) + (v.get("input_tokens") or 0)
    tot["output_tokens"] = (tot.get("output_tokens") or 0) + (v.get("output_tokens") or 0)
    tot["total_tokens"] = tot["input_tokens"] + tot["output_tokens"]
    if v.get("input_tokens") is None:
        tot["token_usage_complete"] = False
    by = tot.setdefault("by_agent", {}).setdefault(
        "verifier", {"steps": 0, "model_calls": 0, "input_tokens": 0, "output_tokens": 0, "tool_calls": 0})
    by["model_calls"] += 1
    by["input_tokens"] += v.get("input_tokens") or 0
    by["output_tokens"] += v.get("output_tokens") or 0


# --------------------------------------------------------------------------- solve


def _adhoc_task_id(question: str) -> str:
    return "adhoc-" + hashlib.sha1(question.encode("utf-8", "replace")).hexdigest()[:10]


def _trace_info(agent, arch_name: str) -> dict:
    """provider_info for the trace: the active config, plus what the model object reports."""
    try:
        info = dict(config.provider_info())
    except Exception as e:
        info = {"provider_error": f"{type(e).__name__}"}
    model = getattr(agent, "model", None)
    mid = getattr(model, "model_id", None)
    if isinstance(mid, str):
        info["model_id"] = mid
    chain = getattr(model, "chain", None)
    if isinstance(chain, list):
        info["model_chain"] = list(chain)
    stats = getattr(model, "stats", None)
    if isinstance(stats, dict):
        info["model_stats"] = {k: v for k, v in stats.items()}
    info["arch"] = arch_name
    info["disabled_tools"] = config.disabled_tools()
    info["verify"] = config.verify_enabled()
    return info


def solve(
    question: str,
    task_id: str | None = None,
    file_name: str | None = None,
    *,
    agent=None,
    stop_event: threading.Event | None = None,
    max_steps: int = MAX_STEPS,
    file_path: str | None = None,
    run_id: str | None = None,
    write_trace: bool = True,
) -> dict:
    """Solve one question. Never raises: failures are reported in `error` with an empty answer.

    Returns dict(answer, raw, steps, error, state, file_path, trace, seconds, run_id, task_id,
    trace_path, arch, verifier, input_tokens, output_tokens, model_calls).

    `file_path` attaches a local file directly (practice tasks); otherwise `file_name` + `task_id`
    download the course attachment. `stop_event`, when set by a watchdog, interrupts the agent
    (and its sub-agents) at their next step. The full trace is written to
    <GAIA_RUNS_DIR or runs>/traces/<run_id>/<task_id>.json unless write_trace=False.
    """
    t0 = time.time()
    rid = run_id or config.get_run_id()
    tid = task_id or _adhoc_task_id(question)
    result: dict = {"answer": "", "raw": None, "steps": 0, "error": None, "state": None, "file_path": None,
                    "trace": [], "run_id": rid, "task_id": tid, "trace_path": None, "arch": None, "verifier": None}

    if not file_path and file_name and task_id:
        try:
            file_path = download_task_file(task_id, file_name)
        except Exception as e:
            result["file_error"] = repr(e)
    result["file_path"] = file_path
    task = build_task(question, file_path, file_name)

    def _watch(step, agent=None):
        if stop_event is not None and stop_event.is_set() and agent is not None:
            agent.interrupt()

    recorder = None
    exc: BaseException | None = None
    arch_name = "single"
    try:
        if agent is None:
            agent = build_agent(max_steps=max_steps, step_callbacks=[_watch] if stop_event else None)
        elif stop_event is not None:
            from smolagents.memory import ActionStep

            agent.step_callbacks.register(ActionStep, _watch)
        subs = _sub_agents(agent)
        if subs:
            arch_name = "multi"
            from .trace import SubAgentRecorder

            recorder = SubAgentRecorder.attach(agent)
            if stop_event is not None:
                from smolagents.memory import ActionStep

                for sub in subs:
                    sub.step_callbacks.register(ActionStep, _watch)
        result["arch"] = arch_name
        run = agent.run(task, return_full_result=True)
        output = getattr(run, "output", run)
        result["state"] = getattr(run, "state", None)
        result["raw"] = output if isinstance(output, (str, int, float, bool)) or output is None else str(output)
        answer = normalize_answer(output)
        if answer and config.verify_enabled() and not (stop_event is not None and stop_event.is_set()):
            v = verify_answer(question, answer, getattr(agent, "model", None))
            result["verifier"] = v
            answer = v["answer"]
        result["answer"] = answer
    except Exception as e:
        exc = e
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        result["arch"] = arch_name
        if agent is not None and hasattr(agent, "memory"):
            try:
                result["steps"], result["trace"] = _trace(agent)
            except Exception:
                pass
        result["seconds"] = round(time.time() - t0, 1)
        if write_trace:
            try:
                from .trace import trace_from_agent
                from .trace import write_trace as _write

                tr = trace_from_agent(
                    agent,
                    task_id=tid,
                    question=question,
                    run_id=rid,
                    final_answer=result["answer"] or result["raw"],
                    error=exc if exc is not None else result["error"],
                    seconds=result["seconds"],
                    arch=arch_name,
                    provider_info=_trace_info(agent, arch_name),
                    recorder=recorder,
                )
                tr["file_name"] = file_name
                tr["attached_file"] = bool(file_path)
                _add_verifier_to_trace(tr, result["verifier"])
                result["trace_path"] = str(_write(tr, root=config.runs_dir()))
                tot = tr.get("totals") or {}
                result["input_tokens"] = tot.get("input_tokens")
                result["output_tokens"] = tot.get("output_tokens")
                result["model_calls"] = tot.get("model_calls")
                result["tool_calls_by_name"] = tot.get("tool_calls_by_name")
            except Exception as e:  # a tracing bug must never cost the answer
                result["trace_error"] = f"{type(e).__name__}: {e}"
    return result


def _load_tools() -> list:
    """Every web + media tool (single-agent default)."""
    web, media = _tool_groups()
    return web + media
