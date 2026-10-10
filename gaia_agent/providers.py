"""Model providers: presets, a client-side rate limiter, and a resilient model wrapper.

Free tiers fail in ways a benchmark run has to survive: per-minute caps (429), daily caps,
overloaded upstream providers (502/503), slow reasoning models (timeouts), and models that
silently ignore parameters such as ``stop``. This module turns those failures into a policy:

* ``RateLimiter``: a reservation-based token bucket (thread-safe; any 60 s window admits at
  most ``burst + rpm`` requests), plus a persistent, date-keyed daily request counter in
  ``runs/usage.json`` guarded by a file lock so concurrent runs share one budget. Hitting the
  cap raises ``BudgetExceeded`` *before* the request is sent.
* ``ResilientModel``: a smolagents ``OpenAIServerModel`` subclass. Every HTTP attempt goes
  through the limiter; retryable failures back off exponentially with jitter (honouring
  ``Retry-After`` and OpenRouter's ``X-RateLimit-Reset``); a model that keeps failing hands
  over to the next model in the preset's fallback chain; and every logical call is appended
  to ``runs/usage.jsonl`` (served model, latency, tokens, retries, fallbacks, error kinds).
* ``vision_complete``: one-shot image questions through the same limiter / retry / fallback.
* ``check_openrouter_key`` / ``check_models``: free metadata calls for a preflight report.

OpenRouter limits for ``:free`` model variants (OpenRouter docs, read 2026-10-10): 20 requests
per minute; 50 requests per UTC day for accounts that bought fewer than 10 credits, 1000 per day
otherwise. The daily allowance is shared by all free models on the account, so falling back to
another free model does not help once it is spent: that case raises ``RemoteQuotaExhausted``.

Env (all optional): GAIA_PROVIDER, GAIA_MODEL_ID, GAIA_VISION_MODEL_ID, GAIA_FALLBACKS
(comma list, "" disables), GAIA_VISION_FALLBACKS, GAIA_RPM, GAIA_DAILY_CAP (<n> or "auto"),
GAIA_RUN_ID, GAIA_REASONING_EFFORT (OpenRouter reasoning effort: low|medium|high),
GAIA_BASE_URL, GAIA_ALLOW_PAID (default off: non-":free" OpenRouter ids are refused, because a
key on an account with credits would bill them).

Run ``python -m gaia_agent.providers`` for a preflight report. Keys are never printed.
"""

from __future__ import annotations

import base64
import contextlib
import datetime as _dt
import io
import json
import mimetypes
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from smolagents.models import (
    ChatMessage,
    ChatMessageStreamDelta,
    OpenAIServerModel,
    TokenUsage,
    remove_content_after_stop_sequences,
    supports_stop_parameter,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
USAGE_JSON = RUNS_DIR / "usage.json"
USAGE_JSONL = RUNS_DIR / "usage.jsonl"
MODELS_CACHE = RUNS_DIR / "openrouter_models.json"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_FREE_LIMITS = {
    "requests_per_minute": 20,
    "requests_per_day_under_10_credits": 50,
    "requests_per_day_with_10_credits": 1000,
    "source": "https://openrouter.ai/docs/api/reference/limits (read 2026-10-10)",
}

# ----------------------------------------------------------------------------- presets
# The preset table lives in gaia_agent/presets.py (no heavy imports) so config.py and this
# module read ONE source of truth.
from .presets import DEFAULT_PRESET, PRESETS, default_provider  # noqa: E402


_DEFAULT: Any = object()  # sentinel: "use the module-level path at call time" (monkeypatchable)


class BudgetExceeded(RuntimeError):
    """The local daily request cap is spent (raised before any request is sent)."""


class RemoteQuotaExhausted(BudgetExceeded):
    """The provider says the account's daily allowance is spent; fallbacks cannot help."""


class AllModelsFailed(RuntimeError):
    """Every model in the chain failed. ``attempts`` lists each failure (no secrets)."""

    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


class FatalProviderError(RuntimeError):
    """Auth / payment problems: retrying or falling back would only burn budget."""


# ----------------------------------------------------------------------------- env helpers


def _load_env() -> None:
    """Reuse config.load_env (it skips paid keys such as XAI_API_KEY) when available."""
    try:
        from .config import _ensure_env

        _ensure_env()
    except Exception:
        pass


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else None


def resolve_preset(name: str | None = None) -> tuple[str, dict]:
    _load_env()
    name = (name or _env("GAIA_PROVIDER") or default_provider()).strip().lower()
    if name not in PRESETS:
        raise ValueError(f"Unknown provider preset {name!r}; choose one of {sorted(PRESETS)}")
    return name, PRESETS[name]


def _api_key(preset: dict) -> str:
    key_env = preset["key_env"]
    if key_env is None:
        return os.environ.get("GAIA_LOCAL_API_KEY", "lm-studio")
    key = os.environ.get(key_env)
    if not key:
        raise FatalProviderError(f"{key_env} is not set (put it in the environment or ~/.claude/.env).")
    return key


def _split_ids(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def model_chain(preset_name: str | None = None, model_id: str | None = None, vision: bool = False) -> list[str]:
    """Primary model followed by its de-duplicated fallbacks, after env overrides."""
    name, preset = resolve_preset(preset_name)
    if vision:
        primary = model_id or _env("GAIA_VISION_MODEL_ID") or preset["vision_model_id"] or preset["model_id"]
        env_fb = _split_ids(os.environ.get("GAIA_VISION_FALLBACKS"))
        fallbacks = env_fb if env_fb is not None else list(preset.get("vision_fallbacks") or [])
    else:
        primary = model_id or _env("GAIA_MODEL_ID") or preset["model_id"]
        env_fb = _split_ids(os.environ.get("GAIA_FALLBACKS"))
        fallbacks = env_fb if env_fb is not None else list(preset.get("fallbacks") or [])
    if primary is None:  # lmstudio: ask the local server
        primary = _first_local_model(_env("GAIA_BASE_URL") or preset["base_url"])
    chain: list[str] = []
    for m in [primary, *fallbacks]:
        if m and m not in chain:
            chain.append(m)
    if name == "openrouter":
        assert_free_models(chain)
    return chain


def assert_free_models(chain: Iterable[str]) -> None:
    """Refuse non-``:free`` OpenRouter ids: a key with credits WOULD bill a paid model.

    The project rule is zero API spend; set GAIA_ALLOW_PAID=1 only to override deliberately.
    """
    paid = [m for m in chain if not m.endswith(":free")]
    if paid and os.environ.get("GAIA_ALLOW_PAID") != "1":
        raise ValueError(
            f"Refusing paid OpenRouter model(s) {paid}: only ':free' variants are allowed "
            f"(set GAIA_ALLOW_PAID=1 to override)."
        )


def _first_local_model(base_url: str) -> str:
    import requests

    try:
        r = requests.get(base_url.rstrip("/") + "/models", timeout=5)
        ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        chat = [i for i in ids if "embed" not in i.lower()]
        return (chat or ids or ["local-model"])[0]
    except Exception:
        return "local-model"


def _utc_today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- file locking


@contextlib.contextmanager
def _file_lock(path: Path):
    """Exclusive inter-process lock on ``<path>.lock`` (POSIX flock; no-op elsewhere)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    fh = open(lock_path, "a+")
    try:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - non-POSIX
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover
            pass
        fh.close()


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


_jsonl_lock = threading.Lock()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _jsonl_lock, _file_lock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


# ----------------------------------------------------------------------------- rate limiter


class DailyCounter:
    """Persistent request counter: ``{"YYYY-MM-DD": {"<scope>": {"requests": n, "by_model": {...}}}}``.

    Days are UTC because OpenRouter's daily allowance resets at UTC midnight.
    """

    def __init__(self, path: Path = USAGE_JSON, scope: str = "default", today: Callable[[], str] = _utc_today):
        self.path = Path(path)
        self.scope = scope
        self.today = today

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def used(self, day: str | None = None) -> int:
        day = day or self.today()
        return int(self._read().get(day, {}).get(self.scope, {}).get("requests", 0))

    def reserve(self, cap: int | None, model: str | None = None) -> int:
        """Atomically check the cap and count one request. Returns the new count for today."""
        with _file_lock(self.path):
            data = self._read()
            day = self.today()
            entry = data.setdefault(day, {}).setdefault(self.scope, {"requests": 0, "by_model": {}})
            if cap is not None and entry["requests"] >= cap:
                raise BudgetExceeded(
                    f"Daily request cap reached for {self.scope!r}: {entry['requests']}/{cap} on {day} (UTC). "
                    f"Raise GAIA_DAILY_CAP only if the provider allowance really is higher."
                )
            entry["requests"] += 1
            if model:
                entry.setdefault("by_model", {})[model] = entry.get("by_model", {}).get(model, 0) + 1
            _atomic_write_json(self.path, data)
            return entry["requests"]


class RateLimiter:
    """Token bucket (``rpm`` refill, ``burst`` capacity) plus an optional persistent daily cap.

    Reservation-based: a caller takes a token even when the bucket is empty, which pushes the
    bucket negative; it then sleeps until its own slot arrives. Callers are served in arrival
    order and the lock is never held while sleeping. Over any window of T seconds at most
    ``burst + rpm * T / 60`` requests are admitted. ``clock`` and ``sleep`` are injectable so
    the arithmetic can be tested without waiting.
    """

    def __init__(
        self,
        rpm: float | None = 18,
        burst: float = 2,
        daily_cap: int | None = None,
        usage_path: Path | str | None = _DEFAULT,
        scope: str = "default",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], str] = _utc_today,
    ):
        self.rpm = float(rpm) if rpm else 0.0
        self.capacity = max(1.0, float(burst))
        self.rate = self.rpm / 60.0  # tokens per second
        self.daily_cap = int(daily_cap) if daily_cap else None
        if usage_path is _DEFAULT:
            usage_path = USAGE_JSON
        self.counter = DailyCounter(Path(usage_path), scope, today) if usage_path else None
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tokens = self.capacity
        self._last = clock()
        self.total_wait_s = 0.0
        self.acquired = 0

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def _reserve_slot(self) -> float:
        """Take one token; return how long the caller must wait for it (0 if available)."""
        with self._lock:
            now = self._clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            self._tokens -= 1.0
            return 0.0 if self._tokens >= 0 else -self._tokens / self.rate

    def acquire(self, model: str | None = None) -> float:
        """Block until a request may be sent. Returns seconds waited. Raises BudgetExceeded."""
        if self.counter is not None:
            self.counter.reserve(self.daily_cap, model)  # raises before we wait or send
        elif self.daily_cap is not None and self.acquired >= self.daily_cap:
            raise BudgetExceeded(f"In-memory request cap reached: {self.acquired}/{self.daily_cap}")
        wait = self._reserve_slot() if self.enabled else 0.0
        if wait > 0:
            self._sleep(wait)
        with self._lock:
            self.total_wait_s += wait
            self.acquired += 1
        return wait

    def used_today(self) -> int:
        return self.counter.used() if self.counter else self.acquired

    def remaining_today(self) -> int | None:
        return None if self.daily_cap is None else max(0, self.daily_cap - self.used_today())


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_limiter(preset_name: str | None = None) -> RateLimiter:
    """One limiter per preset per process: text and vision calls share one account budget."""
    name, preset = resolve_preset(preset_name)
    with _LIMITERS_LOCK:
        if name not in _LIMITERS:
            rpm = float(_env("GAIA_RPM") or preset["rpm"] or 0)
            lim = RateLimiter(rpm=rpm or None, burst=preset.get("burst", 1), daily_cap=None, scope=name)
            cap_raw = _env("GAIA_DAILY_CAP") or preset.get("daily_cap")
            if str(cap_raw).lower() == "auto":
                lim.daily_cap = _auto_daily_cap(lim, fallback=OPENROUTER_FREE_LIMITS["requests_per_day_under_10_credits"])
            elif cap_raw not in (None, "", "0", 0):
                lim.daily_cap = int(cap_raw)
            _LIMITERS[name] = lim
        return _LIMITERS[name]


def _auto_daily_cap(limiter: RateLimiter, fallback: int, reserve: int = 5) -> int:
    """Daily cap from the account's live free-model allowance (GET /api/v1/key, free).

    cap = requests this machine already counted today + what the server says remains, minus a
    small reserve, so the local counter stops us just before the provider would. Falls back to
    the conservative documented minimum (50/day) when the key check is unavailable.
    """
    try:
        info = check_openrouter_key(timeout=10)
    except Exception:
        info = {}
    remaining = info.get("daily_free_remaining")
    if isinstance(remaining, int):
        return limiter.used_today() + max(0, remaining - reserve)
    return fallback


# ----------------------------------------------------------------------------- error policy

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FALLBACK_STATUS = {400, 403, 404, 413, 422}  # model-specific: try the next model, don't retry
FATAL_STATUS = {401, 402}  # bad key / payment required: stop everything


@dataclass
class ErrorInfo:
    kind: str  # rate_limit | daily_quota | server | timeout | connection | empty_response |
    #            bad_request | not_found | context_length | auth | payment | unknown
    status: int | None
    retryable: bool  # retry the same model after a backoff
    fallback: bool  # if not retried (or retries exhausted), move to the next model
    retry_after: float | None
    message: str


def _header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    try:
        v = headers.get(name)
        if v is None:
            v = headers.get(name.lower())
        return v
    except Exception:
        return None


def parse_retry_after(headers: Any, now: float | None = None) -> float | None:
    """Seconds to wait from ``Retry-After`` (seconds or HTTP date) or ``X-RateLimit-Reset``.

    OpenRouter sends ``X-RateLimit-Reset`` as a Unix timestamp in milliseconds.
    """
    now = time.time() if now is None else now
    ra = _header(headers, "retry-after-ms")
    if ra:
        with contextlib.suppress(ValueError):
            return max(0.0, float(ra) / 1000.0)
    ra = _header(headers, "Retry-After")
    if ra:
        with contextlib.suppress(ValueError):
            return max(0.0, float(ra))
        with contextlib.suppress(Exception):
            from email.utils import parsedate_to_datetime

            return max(0.0, parsedate_to_datetime(ra).timestamp() - now)
    reset = _header(headers, "X-RateLimit-Reset")
    if reset:
        with contextlib.suppress(ValueError):
            v = float(reset)
            if v > 1e12:  # epoch milliseconds
                return max(0.0, v / 1000.0 - now)
            if v > 1e9:  # epoch seconds
                return max(0.0, v - now)
            return max(0.0, v)  # already a delta
    return None


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and 100 <= v < 600:
            return v
    resp = getattr(exc, "response", None)
    v = getattr(resp, "status_code", None)
    return v if isinstance(v, int) else None


def classify_error(exc: BaseException, max_retry_after: float = 120.0, now: float | None = None) -> ErrorInfo:
    """Map an exception from the OpenAI-compatible client to a retry / fallback decision."""
    status = _status_of(exc)
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    body = getattr(exc, "body", None)
    text = str(exc)
    if body is not None:
        with contextlib.suppress(Exception):
            text += " " + json.dumps(body, default=str)
    low = text.lower()
    msg = text[:400]
    retry_after = parse_retry_after(headers, now)
    name = type(exc).__name__.lower()

    if isinstance(exc, EmptyResponse):
        return ErrorInfo("empty_response", None, True, True, None, msg)
    if "timeout" in name or "timed out" in low or isinstance(exc, TimeoutError):
        return ErrorInfo("timeout", status, True, True, None, msg)
    if status is None and ("connection" in name or "connecterror" in name or isinstance(exc, ConnectionError)):
        return ErrorInfo("connection", None, True, True, None, msg)
    if status in FATAL_STATUS:
        return ErrorInfo("auth" if status == 401 else "payment", status, False, False, None, msg)

    is_rate = status == 429 or "rate limit" in low or "rate_limit" in low or "too many requests" in low
    # OpenRouter wraps an upstream provider's 429 as "Provider returned error" + metadata.provider_name.
    # That limit belongs to one model's upstream, so a fallback model is the right move, not a long wait.
    upstream = "provider returned error" in low or "provider_name" in low or "rate-limited upstream" in low
    if is_rate and upstream:
        return ErrorInfo("upstream_rate_limit", status, True, True, retry_after, msg)
    if is_rate:
        remaining = _header(headers, "X-RateLimit-Remaining")
        daily_text = "per-day" in low or "per day" in low or "daily" in low or "free-models-per-day" in low
        long_wait = retry_after is not None and retry_after > max_retry_after
        if daily_text or (remaining == "0" and long_wait):
            # Account-level daily allowance: shared across all free models on the account.
            return ErrorInfo("daily_quota", status, False, False, retry_after, msg)
        # A long upstream hint on one model: skip to the next model rather than sleeping for it.
        return ErrorInfo("rate_limit", status, not long_wait, True, retry_after, msg)

    if status in RETRYABLE_STATUS:
        return ErrorInfo("server", status, True, True, retry_after, msg)
    if status is not None and status >= 500:
        return ErrorInfo("server", status, True, True, retry_after, msg)
    if "context length" in low or "context_length" in low or "maximum context" in low or "too many tokens" in low:
        return ErrorInfo("context_length", status, False, True, None, msg)
    if status == 404:
        return ErrorInfo("not_found", status, False, True, None, msg)
    if status == 403:  # e.g. a free model restricted to listed apps, or a moderation block
        return ErrorInfo("forbidden", status, False, True, None, msg)
    if status in FALLBACK_STATUS:
        return ErrorInfo("bad_request", status, False, True, None, msg)
    if status is None and ("overloaded" in low or "provider returned error" in low or "upstream" in low):
        return ErrorInfo("server", None, True, True, retry_after, msg)
    return ErrorInfo("unknown", status, False, False, None, msg)


# Kinds that say "this model is unusable for this account right now", whatever the request.
CIRCUIT_BREAK_KINDS = {"not_found", "forbidden"}


class ModelHealth:
    """In-process circuit breaker: skip a model for ``ttl_s`` after a 403/404 on it.

    Without it, a model that is permanently unavailable to this key (observed live: an
    app-restricted free model answering 403) costs one request of daily allowance on every
    single call before the fallback kicks in.
    """

    def __init__(self, ttl_s: float = 3600.0, clock: Callable[[], float] = time.monotonic):
        self.ttl_s = ttl_s
        self._clock = clock
        self._down: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def mark_down(self, model: str, reason: str) -> None:
        with self._lock:
            self._down[model] = (self._clock() + self.ttl_s, reason[:200])

    def is_down(self, model: str) -> bool:
        with self._lock:
            entry = self._down.get(model)
            if entry and entry[0] <= self._clock():
                del self._down[model]
                return False
            return entry is not None

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {m: r for m, (until, r) in self._down.items() if until > self._clock()}


HEALTH = ModelHealth()


class EmptyResponse(RuntimeError):
    """200 OK but no usable choice: OpenRouter can put an upstream error in the body."""


@dataclass
class RetryPolicy:
    max_retries: int = 3  # per model, after the first attempt
    base_s: float = 2.0
    cap_s: float = 60.0
    max_retry_after_s: float = 120.0  # never sleep longer than this on a server hint
    empty_retries: int = 1  # empty 200s are usually deterministic: retry once, then fall back
    upstream_retries: int = 1  # an upstream provider's 429 rarely clears in seconds: one retry, then fall back

    def backoff(self, attempt: int, retry_after: float | None = None, rng: random.Random | None = None) -> float:
        """Exponential backoff with 'equal jitter'; a server hint is a floor, not a suggestion."""
        r = (rng or random).random()
        exp = min(self.cap_s, self.base_s * (2**attempt))
        delay = exp / 2 + r * exp / 2
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.max_retry_after_s) + r * 0.5)
        return delay


# ----------------------------------------------------------------------------- shared call core


@dataclass
class CallRecord:
    kind: str  # text | vision
    preset: str
    requested_model: str
    started: float = field(default_factory=time.time)
    attempts: list[dict] = field(default_factory=list)
    waited_limiter_s: float = 0.0
    waited_backoff_s: float = 0.0

    def to_dict(self, **extra) -> dict:
        d = {
            "ts": _utc_now_iso(),
            "run_id": _env("GAIA_RUN_ID"),
            "kind": self.kind,
            "preset": self.preset,
            "requested_model": self.requested_model,
            "latency_s": round(time.time() - self.started, 3),
            "waited_limiter_s": round(self.waited_limiter_s, 3),
            "waited_backoff_s": round(self.waited_backoff_s, 3),
            "attempts": sum(1 for a in self.attempts if a.get("kind") != "skipped_circuit_open")
            + (1 if extra.get("status") == "ok" else 0),
            "skipped_models": [a["model"] for a in self.attempts if a.get("kind") == "skipped_circuit_open"],
            "retries": sum(1 for a in self.attempts if a.get("retried")),
            "errors": self.attempts,
        }
        d.update(extra)
        return d


def resilient_call(
    send: Callable[[str], Any],
    chain: list[str],
    *,
    limiter: RateLimiter | None,
    policy: RetryPolicy,
    record: CallRecord,
    validate: Callable[[Any], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    on_fallback: Callable[[str, str], None] | None = None,
    health: ModelHealth | None = None,
) -> tuple[Any, str, int]:
    """Send with limiter + retry + fallback. Returns (response, model_used, fallback_index).

    ``send(model_id)`` performs one HTTP request. ``validate(response)`` may raise
    ``EmptyResponse`` to treat a 200 as a failure. Models the circuit breaker marked down are
    skipped, unless every model in the chain is down (then all are tried anyway).
    """
    health = HEALTH if health is None else health
    live = [m for m in chain if not health.is_down(m)]
    skipped = [m for m in chain if m not in live]
    if skipped:
        record.attempts.extend({"model": m, "kind": "skipped_circuit_open", "status": None,
                                "message": health.snapshot().get(m, ""), "attempt_s": 0.0, "retried": False}
                               for m in skipped)
    if not live:
        live = list(chain)
    for idx, model in enumerate(chain):
        if model not in live:
            continue
        attempt = 0
        empty_seen = 0
        while True:
            if limiter is not None:
                record.waited_limiter_s += limiter.acquire(model)
            t0 = time.time()
            try:
                resp = send(model)
                if validate is not None:
                    validate(resp)
                return resp, model, idx
            except (BudgetExceeded, FatalProviderError):
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                info = classify_error(exc, policy.max_retry_after_s)
                entry = {
                    "model": model,
                    "kind": info.kind,
                    "status": info.status,
                    "message": info.message[:300],
                    "attempt_s": round(time.time() - t0, 3),
                }
                if info.kind == "daily_quota":
                    record.attempts.append(entry)
                    raise RemoteQuotaExhausted(
                        f"Provider daily allowance exhausted (HTTP {info.status}); all free models share it. "
                        f"Retry after the UTC reset. Detail: {info.message[:200]}"
                    ) from exc
                if info.kind in ("auth", "payment"):
                    record.attempts.append(entry)
                    raise FatalProviderError(
                        f"HTTP {info.status} from provider ({info.kind}); check the API key / account. "
                        f"Detail: {info.message[:200]}"
                    ) from exc
                if info.kind == "unknown":
                    record.attempts.append(entry)
                    raise
                if info.kind in CIRCUIT_BREAK_KINDS:
                    health.mark_down(model, f"HTTP {info.status}: {info.message[:160]}")
                if info.kind == "empty_response":
                    empty_seen += 1
                    can_retry = empty_seen <= policy.empty_retries
                elif info.kind == "upstream_rate_limit":
                    can_retry = attempt < min(policy.upstream_retries, policy.max_retries)
                else:
                    can_retry = info.retryable and attempt < policy.max_retries
                entry["retried"] = bool(can_retry)
                record.attempts.append(entry)
                if can_retry:
                    delay = policy.backoff(attempt, info.retry_after, rng)
                    record.waited_backoff_s += delay
                    sleep(delay)
                    attempt += 1
                    continue
                remaining_live = [m for m in chain[idx + 1 :] if m in live]
                if info.fallback and remaining_live:
                    if on_fallback:
                        on_fallback(model, remaining_live[0])
                    break  # next model
                raise AllModelsFailed(
                    f"All {idx + 1} model(s) failed; last error on {model}: {info.kind} {info.status or ''} "
                    f"{info.message[:200]}",
                    record.attempts,
                ) from exc
    raise AllModelsFailed("Empty model chain", record.attempts)  # pragma: no cover


# ----------------------------------------------------------------------------- model metadata


def fetch_openrouter_models(cache_path: Path | None = _DEFAULT, max_age_s: float = 6 * 3600, timeout: float = 20) -> dict[str, dict]:
    """GET /api/v1/models (public, free). Returns {id: model}. Cached on disk for ``max_age_s``."""
    if cache_path is _DEFAULT:
        cache_path = MODELS_CACHE
    if cache_path is not None and cache_path.is_file() and time.time() - cache_path.stat().st_mtime < max_age_s:
        with contextlib.suppress(Exception):
            return {m["id"]: m for m in json.loads(cache_path.read_text(encoding="utf-8"))["data"]}
    import requests

    r = requests.get(f"{OPENROUTER_BASE}/models", timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(cache_path, data)
    return {m["id"]: m for m in data.get("data", [])}


def summarize_model(meta: dict | None) -> dict:
    if not meta:
        return {"exists": False}
    pricing = meta.get("pricing") or {}
    params = sorted(meta.get("supported_parameters") or [])
    free = all(str(pricing.get(k, "0")) in ("0", "0.0") for k in ("prompt", "completion"))
    return {
        "exists": True,
        "free": free,
        "context_length": meta.get("context_length"),
        "max_completion_tokens": (meta.get("top_provider") or {}).get("max_completion_tokens"),
        "input_modalities": (meta.get("architecture") or {}).get("input_modalities"),
        "supports_tools": "tools" in params,
        "supports_stop": "stop" in params,
        "supported_parameters": params,
    }


def check_models(preset_name: str = "openrouter", models: dict[str, dict] | None = None) -> dict[str, dict]:
    """Check every model the preset may use exists and is free (OpenRouter only)."""
    name, preset = resolve_preset(preset_name)
    if name != "openrouter":
        raise ValueError("check_models only knows OpenRouter's public model list")
    models = models if models is not None else fetch_openrouter_models()
    ids = [preset["model_id"], preset["vision_model_id"], *preset["fallbacks"], *preset["vision_fallbacks"]]
    out = {}
    for mid in dict.fromkeys(ids):
        out[mid] = summarize_model(models.get(mid))
    return out


def _supported_params_for(preset_name: str, chain: Iterable[str]) -> dict[str, set[str]]:
    """Best-effort supported-parameter sets (OpenRouter only; never raises)."""
    if preset_name != "openrouter":
        return {}
    try:
        models = fetch_openrouter_models()
    except Exception:
        return {}
    return {m: set(models[m].get("supported_parameters") or []) for m in chain if m in models}


# ----------------------------------------------------------------------------- the model


class ResilientModel(OpenAIServerModel):
    """OpenAIServerModel with a shared limiter, retry/backoff, a fallback chain and usage logs.

    smolagents' own retry (3 tries, 60 s base) and the OpenAI SDK's retries are disabled so
    this class is the single source of retry truth and every HTTP attempt is counted.
    """

    def __init__(
        self,
        model_id: str,
        fallbacks: Iterable[str] = (),
        *,
        preset_name: str = "custom",
        limiter: RateLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
        usage_log: Path | str | None = _DEFAULT,
        supported_params: dict[str, set[str]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        client_kwargs: dict | None = None,
        **kwargs,
    ):
        ck = dict(client_kwargs or {})
        ck["max_retries"] = 0
        kwargs.pop("requests_per_minute", None)
        kwargs.pop("retry", None)
        super().__init__(model_id=model_id, client_kwargs=ck, requests_per_minute=None, retry=False, **kwargs)
        self.fallbacks = [m for m in fallbacks if m and m != model_id]
        self.preset_name = preset_name
        self.limiter = limiter
        self.retry_policy = retry_policy or RetryPolicy()
        if usage_log is _DEFAULT:
            usage_log = USAGE_JSONL
        self.usage_log = Path(usage_log) if usage_log else None
        self.supported_params = supported_params or {}
        self._sleep = sleep
        self._rng = rng
        self._stats_lock = threading.Lock()
        self.stats = {
            "calls": 0,
            "failed_calls": 0,
            "http_attempts": 0,
            "retries": 0,
            "fallbacks": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "served_by": {},
        }
        self.last_call: dict | None = None

    @property
    def chain(self) -> list[str]:
        return [self.model_id, *self.fallbacks]

    def _stop_supported(self, model: str) -> bool:
        params = self.supported_params.get(model)
        if params is not None:
            return "stop" in params
        return supports_stop_parameter(model)

    def _validate(self, response: Any) -> None:
        choices = getattr(response, "choices", None)
        if not choices:
            extra = getattr(response, "model_extra", None) or {}
            raise EmptyResponse(f"No choices in response; body error: {str(extra.get('error'))[:300]}")
        msg = choices[0].message
        if not (msg.content or "").strip() and not msg.tool_calls:
            fr = getattr(choices[0], "finish_reason", None)
            raise EmptyResponse(f"Empty content and no tool calls (finish_reason={fr})")

    def _log(self, record: CallRecord, **extra) -> None:
        rec = record.to_dict(**extra)
        self.last_call = rec
        with self._stats_lock:
            s = self.stats
            s["calls"] += 1
            s["http_attempts"] += rec["attempts"]
            s["retries"] += rec["retries"]
            if extra.get("status") != "ok":
                s["failed_calls"] += 1
            if extra.get("fallback_index"):
                s["fallbacks"] += 1
            s["input_tokens"] += extra.get("input_tokens") or 0
            s["output_tokens"] += extra.get("output_tokens") or 0
            served = extra.get("model")
            if served:
                s["served_by"][served] = s["served_by"].get(served, 0) + 1
        if self.usage_log is not None:
            with contextlib.suppress(Exception):
                append_jsonl(self.usage_log, rec)

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        record = CallRecord("text", self.preset_name, self.model_id)
        trimmed: dict[str, bool] = {}

        def send(model: str):
            params = dict(completion_kwargs, model=model)
            # Some free models reject or silently ignore `stop`; drop it and trim client-side.
            if stop_sequences and not self._stop_supported(model):
                params.pop("stop", None)
            if stop_sequences and "stop" not in params:
                trimmed[model] = True
            return self.client.chat.completions.create(**params)

        try:
            response, served, idx = resilient_call(
                send,
                self.chain,
                limiter=self.limiter,
                policy=self.retry_policy,
                record=record,
                validate=self._validate,
                sleep=self._sleep,
                rng=self._rng,
            )
        except Exception as exc:
            self._log(record, status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}", model=None)
            raise

        choice = response.choices[0]
        content = choice.message.content
        if stop_sequences and trimmed.get(served):
            content = remove_content_after_stop_sequences(content, stop_sequences)
        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        extra = getattr(response, "model_extra", None) or {}
        self._log(
            record,
            status="ok",
            model=served,
            served_model=getattr(response, "model", None),
            upstream_provider=extra.get("provider"),
            fallback_index=idx,
            input_tokens=in_tok,
            output_tokens=out_tok,
            finish_reason=getattr(choice, "finish_reason", None),
            stop_trimmed_client_side=bool(trimmed.get(served)),
        )
        return ChatMessage(
            role=choice.message.role or "assistant",
            content=content,
            tool_calls=choice.message.tool_calls,
            raw=response,
            token_usage=TokenUsage(input_tokens=in_tok, output_tokens=out_tok),
        )

    def generate_stream(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        """Non-streaming under the hood (retries are only safe before any token is shown)."""
        msg = self.generate(messages, stop_sequences, response_format, tools_to_call_from, **kwargs)
        if msg.tool_calls:
            raise NotImplementedError("ResilientModel streams text only; use generate() for tool calls")
        yield ChatMessageStreamDelta(content=msg.content or "", token_usage=msg.token_usage)


def make_model(preset: str | None = None, model_id: str | None = None, **overrides) -> ResilientModel:
    """Build a ResilientModel for a preset (default: GAIA_PROVIDER, then openrouter)."""
    name, p = resolve_preset(preset)
    chain = model_chain(name, model_id)
    temperature = _env("GAIA_TEMPERATURE")
    kwargs: dict[str, Any] = {}
    if temperature is None:
        kwargs["temperature"] = 0.0
    elif temperature.lower() not in ("none", "default"):
        kwargs["temperature"] = float(temperature)
    effort = _env("GAIA_REASONING_EFFORT")
    if effort and name == "openrouter":
        kwargs["extra_body"] = {"reasoning": {"effort": effort}}
    params: dict[str, Any] = dict(
        model_id=chain[0],
        fallbacks=chain[1:],
        preset_name=name,
        api_base=_env("GAIA_BASE_URL") or p["base_url"],
        api_key=_api_key(p),
        client_kwargs={"timeout": p.get("timeout_s", 180), "default_headers": dict(p.get("headers") or {})},
        flatten_messages_as_text=p["flatten_messages_as_text"],
        limiter=get_limiter(name),
        supported_params=_supported_params_for(name, chain),
        **kwargs,
    )
    params.update(overrides)
    return ResilientModel(**params)


# ----------------------------------------------------------------------------- vision

_MAX_IMAGE_SIDE = 2048
_MAX_IMAGE_BYTES = 4 * 1024 * 1024


def image_data_uri(path: str | os.PathLike) -> str:
    """Base64 data URI for an image; very large or exotic formats are re-encoded as JPEG."""
    p = Path(path)
    raw = p.read_bytes()
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as img:
            too_big = max(img.size) > _MAX_IMAGE_SIDE or len(raw) > _MAX_IMAGE_BYTES
            if too_big or mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                img = img.convert("RGB")
                img.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                raw, mime = buf.getvalue(), "image/jpeg"
    except Exception:
        pass
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def vision_complete(
    prompt: str,
    image_paths: list[str],
    preset: str | None = None,
    *,
    model_id: str | None = None,
    max_tokens: int = 1500,
    client: Any = None,
    limiter: RateLimiter | None = None,
    retry_policy: RetryPolicy | None = None,
    usage_log: Path | str | None = _DEFAULT,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Ask the vision model about local images, with the shared limiter, retries and fallbacks."""
    name, p = resolve_preset(preset)
    chain = model_chain(name, model_id, vision=True)
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=_api_key(p),
            base_url=_env("GAIA_BASE_URL") or p["base_url"],
            timeout=p.get("timeout_s", 180),
            max_retries=0,
            default_headers=dict(p.get("headers") or {}),
        )
    if limiter is None:
        limiter = get_limiter(name)
    content: list[dict] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_data_uri(path)}})
    base = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0}

    def validate(resp):
        if not getattr(resp, "choices", None) or not (resp.choices[0].message.content or "").strip():
            raise EmptyResponse("vision model returned no text")

    record = CallRecord("vision", name, chain[0])
    if usage_log is _DEFAULT:
        usage_log = USAGE_JSONL
    log = Path(usage_log) if usage_log else None
    try:
        resp, served, idx = resilient_call(
            lambda m: client.chat.completions.create(model=m, **base),
            chain,
            limiter=limiter,
            policy=retry_policy or RetryPolicy(),
            record=record,
            validate=validate,
            sleep=sleep,
        )
    except Exception as exc:
        if log:
            with contextlib.suppress(Exception):
                append_jsonl(log, record.to_dict(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}", n_images=len(image_paths)))
        raise
    usage = getattr(resp, "usage", None)
    if log:
        with contextlib.suppress(Exception):
            append_jsonl(
                log,
                record.to_dict(
                    status="ok",
                    model=served,
                    fallback_index=idx,
                    input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    n_images=len(image_paths),
                ),
            )
    return resp.choices[0].message.content.strip()


# ----------------------------------------------------------------------------- key check


def check_openrouter_key(timeout: float = 15, session: Any = None) -> dict:
    """GET /api/v1/key (free). Returns limits and today's free-model usage. Never returns the key.

    The ``label`` field is dropped because OpenRouter's default label is a masked key prefix.
    """
    _load_env()
    out: dict[str, Any] = {"key_present": False, "documented_free_limits": OPENROUTER_FREE_LIMITS}
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        out["note"] = "OPENROUTER_API_KEY is not set; add it to the environment or ~/.claude/.env."
        return out
    out["key_present"] = True
    if session is None:
        import requests as session  # type: ignore[no-redef]
    try:
        r = session.get(f"{OPENROUTER_BASE}/key", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    except Exception as e:  # network failure: report the type only
        out.update(ok=False, error=f"{type(e).__name__}")
        return out
    out["http_status"] = r.status_code
    if r.status_code != 200:
        out.update(ok=False, error="key rejected" if r.status_code in (401, 403) else f"HTTP {r.status_code}")
        return out
    data = (r.json() or {}).get("data", {}) or {}
    out["ok"] = True
    for k in ("is_free_tier", "limit", "limit_remaining", "limit_reset", "usage", "usage_daily", "rate_limit", "free_model_daily_requests"):
        if k in data:
            out[k] = data[k]
    fmd = data.get("free_model_daily_requests") or {}
    if isinstance(fmd, dict) and fmd.get("limit") is not None:
        out["daily_free_allowance"] = fmd.get("limit")
        out["daily_free_remaining"] = fmd.get("remaining")
    else:
        # Older responses lack the field; the tier is set by all-time purchased credits.
        out["daily_free_allowance"] = (
            OPENROUTER_FREE_LIMITS["requests_per_day_under_10_credits"]
            if data.get("is_free_tier", True)
            else "50 or 1000 (depends on all-time credits purchased)"
        )
    return out


# ----------------------------------------------------------------------------- usage summary


def summarize_usage(path: Path | None = None, run_id: str | None = None) -> dict:
    """Aggregate runs/usage.jsonl (optionally for one run): calls, retries, fallbacks, tokens."""
    path = USAGE_JSONL if path is None else path
    agg: dict[str, Any] = {
        "calls": 0,
        "errors": 0,
        "retries": 0,
        "fallbacks": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_s": 0.0,
        "waited_s": 0.0,
        "by_model": {},
        "error_kinds": {},
    }
    if not Path(path).is_file():
        return agg
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if run_id and r.get("run_id") != run_id:
            continue
        agg["calls"] += 1
        agg["errors"] += r.get("status") != "ok"
        agg["retries"] += r.get("retries", 0)
        agg["fallbacks"] += bool(r.get("fallback_index"))
        agg["input_tokens"] += r.get("input_tokens") or 0
        agg["output_tokens"] += r.get("output_tokens") or 0
        agg["latency_s"] += r.get("latency_s") or 0
        agg["waited_s"] += (r.get("waited_limiter_s") or 0) + (r.get("waited_backoff_s") or 0)
        if r.get("model"):
            agg["by_model"][r["model"]] = agg["by_model"].get(r["model"], 0) + 1
        for e in r.get("errors") or []:
            agg["error_kinds"][e.get("kind")] = agg["error_kinds"].get(e.get("kind"), 0) + 1
    agg["latency_s"] = round(agg["latency_s"], 1)
    agg["waited_s"] = round(agg["waited_s"], 1)
    return agg


def _preflight() -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="Provider preflight: key limits, model availability, usage.")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    name, preset = resolve_preset(args.preset)
    print(f"preset: {name}  base_url: {preset['base_url']}")
    print("chain (text):  " + " -> ".join(model_chain(name)))
    print("chain (vision): " + " -> ".join(model_chain(name, vision=True)))
    if name == "openrouter":
        try:
            for mid, s in check_models(name).items():
                flag = "OK " if s.get("exists") and s.get("free") else "!! "
                print(
                    f"  {flag}{mid:55s} ctx={s.get('context_length')} tools={s.get('supports_tools')} "
                    f"stop={s.get('supports_stop')} in={s.get('input_modalities')}"
                )
        except Exception as e:
            print(f"  model check failed: {type(e).__name__}: {e}")
        print("key:", json.dumps(check_openrouter_key(), default=str))
    lim = get_limiter(name)
    print(f"limiter: rpm={lim.rpm} burst={lim.capacity} daily_cap={lim.daily_cap} used_today={lim.used_today()}")
    print("usage:", json.dumps(summarize_usage(run_id=args.run_id)))
    return 0


__all__ = [
    "PRESETS",
    "BudgetExceeded",
    "RemoteQuotaExhausted",
    "AllModelsFailed",
    "FatalProviderError",
    "EmptyResponse",
    "RateLimiter",
    "DailyCounter",
    "RetryPolicy",
    "ModelHealth",
    "HEALTH",
    "ErrorInfo",
    "ResilientModel",
    "classify_error",
    "parse_retry_after",
    "resilient_call",
    "make_model",
    "model_chain",
    "get_limiter",
    "vision_complete",
    "image_data_uri",
    "check_openrouter_key",
    "check_models",
    "fetch_openrouter_models",
    "summarize_usage",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_preflight())
