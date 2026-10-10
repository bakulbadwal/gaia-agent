"""Offline tests for gaia_agent.providers: limiter math, budgets, retry/fallback policy.

The HTTP layer is faked at the OpenAI-client boundary (``client.chat.completions.create``)
with real ``openai`` exception classes, so classification is tested against the exact types
the SDK raises. Only the test marked ``network`` touches the internet (a free, keyless GET).
"""

from __future__ import annotations

import json
import random
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest
from openai.types.chat import ChatCompletion

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaia_agent import providers as P  # noqa: E402

try:  # openai>=3 ships its own httpx fork; fall back to httpx for older SDKs
    import httpx2 as _httpx
except ImportError:  # pragma: no cover
    import httpx as _httpx

from smolagents.models import ChatMessage, MessageRole  # noqa: E402


# --------------------------------------------------------------------------- fakes


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


class FakeDay:
    def __init__(self, day="2026-10-10"):
        self.day = day

    def __call__(self):
        return self.day


def completion(content="ok", model="m", prompt_tokens=11, completion_tokens=7, provider="FakeProvider", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return ChatCompletion.model_validate(
        {
            "id": "gen-1",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "provider": provider,  # OpenRouter extra field -> model_extra
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
        }
    )


def no_choices():
    return ChatCompletion.model_validate(
        {"id": "gen-2", "object": "chat.completion", "created": 0, "model": "m", "choices": [], "error": {"message": "upstream exploded"}}
    )


def status_error(code: int, headers: dict | None = None, body: dict | None = None, msg: str | None = None):
    req = _httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = _httpx.Response(code, headers=headers or {}, request=req)
    cls = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        402: openai.APIStatusError,
        403: openai.PermissionDeniedError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
    }.get(code, openai.InternalServerError)
    return cls(msg or f"Error code: {code}", response=resp, body=body)


def timeout_error():
    return openai.APITimeoutError(request=_httpx.Request("POST", "https://x.test/v1/chat/completions"))


def connection_error():
    return openai.APIConnectionError(request=_httpx.Request("POST", "https://x.test/v1/chat/completions"))


class FakeCompletions:
    """Scripted outcomes per model id: each call pops the next item (exception or response)."""

    def __init__(self, script: dict[str, list]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            queue = self.script.get(kwargs["model"])
            if not queue:
                raise AssertionError(f"unexpected call for model {kwargs['model']!r}")
            outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome() if callable(outcome) else outcome


class FakeClient:
    def __init__(self, script):
        self.completions = FakeCompletions(script)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self):
        return self.completions.calls

    def models_called(self):
        return [c["model"] for c in self.calls]


def user(text="What is 6*7?"):
    return [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": text}])]


@pytest.fixture
def tmp_runs(tmp_path, monkeypatch):
    """Point every default path at tmp so tests never touch the real runs/ folder."""
    monkeypatch.setattr(P, "USAGE_JSON", tmp_path / "usage.json")
    monkeypatch.setattr(P, "USAGE_JSONL", tmp_path / "usage.jsonl")
    monkeypatch.setattr(P, "MODELS_CACHE", tmp_path / "models.json")
    monkeypatch.setattr(P, "_LIMITERS", {})
    for var in ("GAIA_PROVIDER", "GAIA_MODEL_ID", "GAIA_VISION_MODEL_ID", "GAIA_FALLBACKS", "GAIA_VISION_FALLBACKS",
                "GAIA_RPM", "GAIA_DAILY_CAP", "GAIA_RUN_ID", "GAIA_TEMPERATURE", "GAIA_REASONING_EFFORT", "GAIA_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def make(client, chain=("primary:free", "backup:free", "last:free"), tmp=None, limiter=None, policy=None, **kw):
    sleeps: list[float] = []
    model = P.ResilientModel(
        chain[0],
        fallbacks=list(chain[1:]),
        preset_name="openrouter",
        client=client,
        api_key="sk-test-not-real",
        limiter=limiter,
        retry_policy=policy or P.RetryPolicy(max_retries=2, base_s=1.0, cap_s=8.0),
        usage_log=(tmp / "usage.jsonl") if tmp else None,
        sleep=sleeps.append,
        rng=random.Random(0),
        **kw,
    )
    return model, sleeps


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# --------------------------------------------------------------------------- RateLimiter math


def test_bucket_burst_then_steady_rate():
    clock = FakeClock()
    lim = P.RateLimiter(rpm=18, burst=2, usage_path=None, clock=clock, sleep=clock.sleep)
    waits = [lim.acquire() for _ in range(5)]
    assert waits[:2] == [0.0, 0.0]  # burst capacity
    interval = 60 / 18
    assert waits[2:] == pytest.approx([interval] * 3)  # then exactly one token per 60/rpm s
    assert lim.total_wait_s == pytest.approx(3 * interval)


def test_bucket_refills_but_caps_at_capacity():
    clock = FakeClock()
    lim = P.RateLimiter(rpm=60, burst=3, usage_path=None, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        assert lim.acquire() == 0.0
    clock.t += 3600  # a long idle period refills to capacity, not to 3600 tokens
    assert [lim.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert lim.acquire() == pytest.approx(1.0)


def test_any_60s_window_admits_at_most_burst_plus_rpm():
    clock = FakeClock(0.0)
    rpm, burst = 18, 2
    lim = P.RateLimiter(rpm=rpm, burst=burst, usage_path=None, clock=clock, sleep=clock.sleep)
    times = []
    for _ in range(200):  # a greedy caller hammering the limiter
        lim.acquire()
        times.append(clock.t)
    for i, t0 in enumerate(times):
        in_window = sum(1 for t in times[i:] if t < t0 + 60)
        assert in_window <= burst + rpm  # 20 == OpenRouter's documented free per-minute cap


def test_reservations_queue_in_order_without_holding_lock():
    clock = FakeClock()
    lim = P.RateLimiter(rpm=60, burst=1, usage_path=None, clock=clock, sleep=clock.sleep)
    # Four callers arrive at the same instant: slots are handed out 0, 1, 2, 3 seconds out.
    assert [lim._reserve_slot() for _ in range(4)] == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_disabled_limiter_never_waits():
    clock = FakeClock()
    lim = P.RateLimiter(rpm=None, usage_path=None, clock=clock, sleep=clock.sleep)
    assert sum(lim.acquire() for _ in range(100)) == 0
    assert clock.sleeps == []


def test_limiter_threads_get_distinct_slots():
    lim = P.RateLimiter(rpm=600, burst=1, usage_path=None, sleep=lambda s: None)
    waits: list[float] = []
    lock = threading.Lock()

    def worker():
        w = lim._reserve_slot()
        with lock:
            waits.append(w)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    # Every caller got a different slot (no two share a token), spaced ~0.1 s apart.
    assert len({round(w, 1) for w in waits}) == 20


# --------------------------------------------------------------------------- daily budget


def test_daily_counter_persists_across_instances(tmp_path):
    day = FakeDay()
    path = tmp_path / "usage.json"
    a = P.RateLimiter(rpm=None, daily_cap=3, usage_path=path, scope="openrouter", today=day)
    a.acquire("m1")
    a.acquire("m2")
    b = P.RateLimiter(rpm=None, daily_cap=3, usage_path=path, scope="openrouter", today=day)  # "another process"
    assert b.used_today() == 2 and b.remaining_today() == 1
    b.acquire("m1")
    with pytest.raises(P.BudgetExceeded, match="3/3"):
        a.acquire("m1")
    data = json.loads(path.read_text())
    assert data["2026-10-10"]["openrouter"] == {"requests": 3, "by_model": {"m1": 2, "m2": 1}}


def test_daily_counter_rolls_over_at_new_utc_day(tmp_path):
    day = FakeDay("2026-10-10")
    lim = P.RateLimiter(rpm=None, daily_cap=1, usage_path=tmp_path / "u.json", today=day)
    lim.acquire()
    with pytest.raises(P.BudgetExceeded):
        lim.acquire()
    day.day = "2026-10-11"
    assert lim.acquire() == 0.0
    data = json.loads((tmp_path / "u.json").read_text())
    assert set(data) == {"2026-10-10", "2026-10-11"}


def test_scopes_are_independent(tmp_path):
    path, day = tmp_path / "u.json", FakeDay()
    P.RateLimiter(rpm=None, daily_cap=1, usage_path=path, scope="openrouter", today=day).acquire()
    P.RateLimiter(rpm=None, daily_cap=1, usage_path=path, scope="gemini", today=day).acquire()  # no raise


def test_budget_exceeded_raises_before_any_http_request(tmp_runs):
    day = FakeDay()
    lim = P.RateLimiter(rpm=None, daily_cap=1, usage_path=tmp_runs / "usage.json", today=day)
    client = FakeClient({"primary:free": [completion("a")]})
    model, _ = make(client, tmp=tmp_runs, limiter=lim)
    model.generate(user())
    with pytest.raises(P.BudgetExceeded):
        model.generate(user())
    assert len(client.calls) == 1  # the second call never reached the HTTP layer
    rows = read_jsonl(tmp_runs / "usage.jsonl")
    assert rows[-1]["status"] == "error" and "BudgetExceeded" in rows[-1]["error"]


def test_retries_count_against_the_daily_budget(tmp_runs):
    lim = P.RateLimiter(rpm=None, daily_cap=100, usage_path=tmp_runs / "usage.json", today=FakeDay())
    client = FakeClient({"primary:free": [status_error(503), status_error(503), completion("x")]})
    model, _ = make(client, tmp=tmp_runs, limiter=lim)
    model.generate(user())
    assert lim.used_today() == 3  # every HTTP attempt costs one request of allowance


# --------------------------------------------------------------------------- Retry-After parsing


def test_parse_retry_after_variants():
    now = 1_700_000_000.0
    assert P.parse_retry_after({"Retry-After": "7"}, now) == 7.0
    assert P.parse_retry_after({"retry-after-ms": "1500"}, now) == 1.5
    assert P.parse_retry_after({"X-RateLimit-Reset": str(int((now + 30) * 1000))}, now) == pytest.approx(30.0)
    assert P.parse_retry_after({"X-RateLimit-Reset": str(int(now + 12))}, now) == pytest.approx(12.0)
    assert P.parse_retry_after({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}, now) == 0.0  # past date
    assert P.parse_retry_after({}, now) is None
    assert P.parse_retry_after(None, now) is None


def test_backoff_is_exponential_capped_and_respects_hint():
    pol = P.RetryPolicy(base_s=2.0, cap_s=20.0, max_retry_after_s=60.0)
    rng = random.Random(1)
    for attempt, exp in [(0, 2), (1, 4), (2, 8), (3, 16), (4, 20), (9, 20)]:
        d = pol.backoff(attempt, None, rng)
        assert exp / 2 <= d <= exp  # equal jitter: [exp/2, exp]
    assert pol.backoff(0, 30.0, rng) >= 30.0  # server hint is a floor
    assert pol.backoff(0, 9999.0, rng) <= 60.5  # ...but never beyond max_retry_after


# --------------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    "exc, kind, retryable, fallback",
    [
        (status_error(429, {"Retry-After": "3"}), "rate_limit", True, True),
        (status_error(503), "server", True, True),
        (status_error(502, body={"message": "Provider returned error"}), "server", True, True),
        (status_error(404, msg="No endpoints found for model"), "not_found", False, True),
        (status_error(400, msg="This model's maximum context length is 8192 tokens"), "context_length", False, True),
        (status_error(400, msg="tools not supported"), "bad_request", False, True),
        (status_error(403, msg="only available on agentic harnesses"), "forbidden", False, True),
        (status_error(401), "auth", False, False),
        (status_error(402, msg="Insufficient credits"), "payment", False, False),
        (timeout_error(), "timeout", True, True),
        (connection_error(), "connection", True, True),
        (P.EmptyResponse("nothing"), "empty_response", True, True),
        (ValueError("bug in our code"), "unknown", False, False),
    ],
)
def test_classify_error(exc, kind, retryable, fallback):
    info = P.classify_error(exc)
    assert (info.kind, info.retryable, info.fallback) == (kind, retryable, fallback)


def test_classify_daily_quota_by_message_and_by_headers():
    by_msg = status_error(429, msg="Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000")
    assert P.classify_error(by_msg).kind == "daily_quota"
    now = 1_700_000_000.0
    hdrs = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int((now + 6 * 3600) * 1000))}
    assert P.classify_error(status_error(429, hdrs), now=now).kind == "daily_quota"
    # Remaining 0 but the reset is seconds away: that is the per-minute window, so retry.
    hdrs_soon = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int((now + 20) * 1000))}
    info = P.classify_error(status_error(429, hdrs_soon), now=now)
    assert info.kind == "rate_limit" and info.retryable and info.retry_after == pytest.approx(20.0)


def test_status_429_substring_in_text_is_not_a_status():
    # A request id containing "429" must not make an unrelated error look rate-limited.
    info = P.classify_error(status_error(404, msg="model not found (request id 4f429ab1)"))
    assert info.kind == "not_found"


# --------------------------------------------------------------------------- ResilientModel


def test_success_first_try_logs_usage(tmp_runs, monkeypatch):
    monkeypatch.setenv("GAIA_RUN_ID", "run-xyz")
    client = FakeClient({"primary:free": [completion("42", model="primary:free", prompt_tokens=100, completion_tokens=3)]})
    model, sleeps = make(client, tmp=tmp_runs)
    msg = model.generate(user())
    assert msg.content == "42"
    assert msg.token_usage.input_tokens == 100 and msg.token_usage.output_tokens == 3
    assert sleeps == []
    (row,) = read_jsonl(tmp_runs / "usage.jsonl")
    assert row["status"] == "ok" and row["model"] == "primary:free" and row["fallback_index"] == 0
    assert row["retries"] == 0 and row["attempts"] == 1 and row["run_id"] == "run-xyz"
    assert row["upstream_provider"] == "FakeProvider"
    assert row["input_tokens"] == 100 and row["output_tokens"] == 3
    assert "sk-test" not in json.dumps(row)
    assert model.stats["calls"] == 1 and model.stats["served_by"] == {"primary:free": 1}


def test_429_with_retry_after_is_retried_then_succeeds(tmp_runs):
    client = FakeClient({"primary:free": [status_error(429, {"Retry-After": "5"}), status_error(429, {"Retry-After": "5"}), completion("ok")]})
    model, sleeps = make(client, tmp=tmp_runs)
    assert model.generate(user()).content == "ok"
    assert client.models_called() == ["primary:free"] * 3
    assert len(sleeps) == 2 and all(s >= 5.0 for s in sleeps)  # honoured Retry-After
    row = read_jsonl(tmp_runs / "usage.jsonl")[0]
    assert row["retries"] == 2 and row["attempts"] == 3 and [e["kind"] for e in row["errors"]] == ["rate_limit"] * 2


def test_persistent_5xx_falls_back_to_next_model(tmp_runs):
    client = FakeClient({"primary:free": [status_error(503)], "backup:free": [completion("from backup", model="backup:free")]})
    model, sleeps = make(client, tmp=tmp_runs, policy=P.RetryPolicy(max_retries=2, base_s=1, cap_s=4))
    assert model.generate(user()).content == "from backup"
    assert client.models_called() == ["primary:free"] * 3 + ["backup:free"]
    assert len(sleeps) == 2  # backoff only between retries, not before the fallback
    assert sleeps[1] > sleeps[0] * 0.9  # exponential growth (with jitter)
    row = read_jsonl(tmp_runs / "usage.jsonl")[0]
    assert row["model"] == "backup:free" and row["fallback_index"] == 1
    assert model.stats["fallbacks"] == 1


def test_404_falls_back_immediately_without_sleeping(tmp_runs):
    client = FakeClient({"primary:free": [status_error(404, msg="No endpoints found")], "backup:free": [completion("b")]})
    model, sleeps = make(client, tmp=tmp_runs)
    assert model.generate(user()).content == "b"
    assert sleeps == [] and client.models_called() == ["primary:free", "backup:free"]


def test_timeouts_are_retried(tmp_runs):
    client = FakeClient({"primary:free": [timeout_error(), completion("late")]})
    model, sleeps = make(client, tmp=tmp_runs)
    assert model.generate(user()).content == "late" and len(sleeps) == 1


def test_daily_quota_stops_without_trying_fallbacks(tmp_runs):
    err = status_error(429, msg="Rate limit exceeded: free-models-per-day")
    client = FakeClient({"primary:free": [err], "backup:free": [completion("never")]})
    model, sleeps = make(client, tmp=tmp_runs)
    with pytest.raises(P.RemoteQuotaExhausted):
        model.generate(user())
    assert client.models_called() == ["primary:free"] and sleeps == []


def test_auth_error_is_fatal_and_not_retried(tmp_runs):
    client = FakeClient({"primary:free": [status_error(401)], "backup:free": [completion("never")]})
    model, _ = make(client, tmp=tmp_runs)
    with pytest.raises(P.FatalProviderError):
        model.generate(user())
    assert client.models_called() == ["primary:free"]


def test_all_models_failing_raises_with_attempt_history(tmp_runs):
    client = FakeClient({m: [status_error(503)] for m in ("primary:free", "backup:free", "last:free")})
    model, _ = make(client, tmp=tmp_runs, policy=P.RetryPolicy(max_retries=1, base_s=0.1, cap_s=0.2))
    with pytest.raises(P.AllModelsFailed) as ei:
        model.generate(user())
    assert len(ei.value.attempts) == 6  # 2 attempts x 3 models
    assert {a["model"] for a in ei.value.attempts} == {"primary:free", "backup:free", "last:free"}
    row = read_jsonl(tmp_runs / "usage.jsonl")[0]
    assert row["status"] == "error" and "AllModelsFailed" in row["error"]


def test_empty_200_retried_once_then_falls_back(tmp_runs):
    client = FakeClient({"primary:free": [no_choices()], "backup:free": [completion("b")]})
    model, sleeps = make(client, tmp=tmp_runs)
    assert model.generate(user()).content == "b"
    assert client.models_called() == ["primary:free", "primary:free", "backup:free"]
    assert len(sleeps) == 1


def test_blank_content_counts_as_empty(tmp_runs):
    client = FakeClient({"primary:free": [completion("   ")], "backup:free": [completion("real")]})
    model, _ = make(client, tmp=tmp_runs)
    assert model.generate(user()).content == "real"


def test_unknown_exception_propagates_unchanged(tmp_runs):
    client = FakeClient({"primary:free": [KeyError("bug")]})
    model, _ = make(client, tmp=tmp_runs)
    with pytest.raises(KeyError):
        model.generate(user())


def test_sdk_and_smolagents_retries_are_disabled():
    model = P.ResilientModel("m", api_key="sk-test-not-real", api_base="http://localhost:9/v1", usage_log=None)
    assert model.client.max_retries == 0
    assert model.retryer.max_attempts == 1
    assert model.rate_limiter._enabled is False  # our limiter replaces smolagents' throttle


# --------------------------------------------------------------------------- stop handling


def test_stop_dropped_and_trimmed_for_models_without_stop_support(tmp_runs):
    text = "Thought: x\n<code>\nfinal_answer('42')\n</code>\nObservation: hallucinated"
    client = FakeClient({"primary:free": [completion(text)]})
    model, _ = make(client, tmp=tmp_runs, supported_params={"primary:free": {"tools", "temperature"}})
    msg = model.generate(user(), stop_sequences=["Observation:", "</code>"])
    assert "stop" not in client.calls[0]
    assert msg.content == "Thought: x\n<code>\nfinal_answer('42')\n"
    assert read_jsonl(tmp_runs / "usage.jsonl")[0]["stop_trimmed_client_side"] is True


def test_stop_sent_when_supported(tmp_runs):
    client = FakeClient({"primary:free": [completion("x")]})
    model, _ = make(client, tmp=tmp_runs, supported_params={"primary:free": {"stop", "tools"}})
    model.generate(user(), stop_sequences=["</code>"])
    assert client.calls[0]["stop"] == ["</code>"]


def test_stop_decision_is_per_model_in_the_chain(tmp_runs):
    client = FakeClient({"primary:free": [status_error(404)], "backup:free": [completion("a</code>junk")]})
    model, _ = make(
        client, tmp=tmp_runs, supported_params={"primary:free": {"stop"}, "backup:free": {"tools"}}
    )
    msg = model.generate(user(), stop_sequences=["</code>"])
    assert "stop" in client.calls[0] and "stop" not in client.calls[1]
    assert msg.content == "a"


# --------------------------------------------------------------------------- smolagents integration


def test_works_inside_a_real_codeagent(tmp_runs):
    from smolagents import CodeAgent

    reply = "Thought: multiply.\n<code>\nfinal_answer(str(6*7))\n</code>"
    client = FakeClient({"primary:free": [status_error(429, {"Retry-After": "1"}), completion(reply)]})
    model, sleeps = make(client, tmp=tmp_runs, supported_params={"primary:free": {"tools"}})
    agent = CodeAgent(tools=[], model=model, max_steps=2, verbosity_level=0)
    assert agent.run("What is 6*7?") == "42"
    assert len(sleeps) == 1
    assert "stop" not in client.calls[-1]  # unsupported -> dropped, trimmed client side


def test_generate_stream_yields_one_delta(tmp_runs):
    client = FakeClient({"primary:free": [completion("streamed")]})
    model, _ = make(client, tmp=tmp_runs)
    deltas = list(model.generate_stream(user()))
    assert len(deltas) == 1 and deltas[0].content == "streamed" and deltas[0].token_usage.output_tokens == 7


# --------------------------------------------------------------------------- vision


@pytest.fixture
def png(tmp_path):
    from PIL import Image

    p = tmp_path / "board.png"
    Image.new("RGB", (8, 8), (200, 10, 10)).save(p)
    return p


def test_vision_complete_sends_data_uri_and_falls_back(tmp_runs, png, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    client = FakeClient(
        {
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": [status_error(404)],
            "dots-studio/dots-3-note-preview:free": [status_error(404)],
            "google/gemma-4-31b-it:free": [completion("A red square.")],
        }
    )
    lim = P.RateLimiter(rpm=None, usage_path=tmp_runs / "usage.json", today=FakeDay())
    out = P.vision_complete("Describe.", [str(png)], "openrouter", client=client, limiter=lim, sleep=lambda s: None)
    assert out == "A red square."
    parts = client.calls[0]["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "Describe."}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    row = read_jsonl(tmp_runs / "usage.jsonl")[0]
    assert row["kind"] == "vision" and row["model"] == "google/gemma-4-31b-it:free" and row["n_images"] == 1
    assert lim.used_today() == 3


def test_image_data_uri_downscales_huge_images(tmp_path):
    from PIL import Image

    p = tmp_path / "big.bmp"
    Image.new("RGB", (5000, 100)).save(p)
    uri = P.image_data_uri(p)
    assert uri.startswith("data:image/jpeg;base64,")


# --------------------------------------------------------------------------- presets / env


def test_openrouter_preset_shape():
    p = P.PRESETS["openrouter"]
    assert p["base_url"] == "https://openrouter.ai/api/v1" and p["key_env"] == "OPENROUTER_API_KEY"
    assert p["model_id"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert p["vision_model_id"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    assert p["vision_fallbacks"] == [
        "dots-studio/dots-3-note-preview:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
    ]
    every_id = [p["model_id"], p["vision_model_id"], *p["fallbacks"], *p["vision_fallbacks"]]
    assert all(m.endswith(":free") for m in every_id)
    assert not any("inkling" in m for m in every_id)  # 403 for this key: agentic harnesses only
    assert p["headers"] == {"HTTP-Referer": "https://github.com/bakulbadwal", "X-Title": "GAIA Agent (HF Agents Course)"}
    assert p["rpm"] + p["burst"] <= 20
    assert {"gemini", "groq", "lmstudio"} <= set(P.PRESETS)


def test_model_chain_env_overrides(tmp_runs, monkeypatch):
    assert P.model_chain("openrouter")[0] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    monkeypatch.setenv("GAIA_MODEL_ID", "google/gemma-4-31b-it:free")
    chain = P.model_chain("openrouter")
    assert chain[0] == "google/gemma-4-31b-it:free" and chain.count("google/gemma-4-31b-it:free") == 1
    monkeypatch.setenv("GAIA_FALLBACKS", "")
    assert P.model_chain("openrouter") == ["google/gemma-4-31b-it:free"]
    monkeypatch.setenv("GAIA_VISION_FALLBACKS", "a:free, b:free")
    assert P.model_chain("openrouter", vision=True) == [
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "a:free", "b:free"]


def test_make_model_wires_headers_limiter_and_chain(tmp_runs, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("GAIA_DAILY_CAP", "7")
    monkeypatch.setattr(P, "_supported_params_for", lambda name, chain: {})  # no network
    m = P.make_model("openrouter")
    assert isinstance(m, P.ResilientModel)
    assert m.chain == [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
    ]
    assert m.client.max_retries == 0
    assert m.client.default_headers["X-Title"] == "GAIA Agent (HF Agents Course)"
    assert m.limiter is P.get_limiter("openrouter")  # shared with vision_complete
    assert m.limiter.daily_cap == 7 and m.limiter.rpm == 18
    assert m.kwargs["temperature"] == 0.0


def test_make_model_without_key_fails_clearly(tmp_runs, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(P, "_load_env", lambda: None)
    with pytest.raises(P.FatalProviderError, match="OPENROUTER_API_KEY"):
        P.make_model("openrouter")


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        P.resolve_preset("not-a-provider")


# --------------------------------------------------------------------------- key check


class FakeSession:
    def __init__(self, status=200, payload=None):
        self.status, self.payload, self.seen = status, payload, []

    def get(self, url, headers=None, timeout=None):
        self.seen.append((url, headers))
        return SimpleNamespace(status_code=self.status, json=lambda: self.payload)


def test_check_key_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(P, "_load_env", lambda: None)
    out = P.check_openrouter_key(session=FakeSession())
    assert out["key_present"] is False and out["documented_free_limits"]["requests_per_minute"] == 20


def test_check_key_reports_limits_and_never_leaks_key(monkeypatch):
    secret = "sk-or-v1-SECRETSECRETSECRET"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    payload = {
        "data": {
            "label": "sk-or-v1-SEC...RET",
            "limit": None,
            "usage": 0,
            "is_free_tier": True,
            "rate_limit": {"requests": -1, "interval": "10s"},
            "free_model_daily_requests": {"used": 3, "limit": 50, "remaining": 47},
        }
    }
    sess = FakeSession(200, payload)
    out = P.check_openrouter_key(session=sess)
    assert sess.seen[0][0] == "https://openrouter.ai/api/v1/key"
    assert out["ok"] and out["is_free_tier"] is True
    assert out["daily_free_allowance"] == 50 and out["daily_free_remaining"] == 47
    assert "label" not in out
    assert "SECRET" not in json.dumps(out)


def test_check_key_rejected(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    out = P.check_openrouter_key(session=FakeSession(401, {}))
    assert out["ok"] is False and out["error"] == "key rejected"


# --------------------------------------------------------------------------- usage summary / metadata


def test_summarize_usage_filters_by_run(tmp_path):
    path = tmp_path / "usage.jsonl"
    rows = [
        {"run_id": "a", "status": "ok", "retries": 1, "fallback_index": 0, "input_tokens": 10, "output_tokens": 2, "model": "m1", "errors": [{"kind": "rate_limit"}]},
        {"run_id": "a", "status": "ok", "retries": 0, "fallback_index": 1, "input_tokens": 5, "output_tokens": 1, "model": "m2", "errors": [{"kind": "server"}]},
        {"run_id": "b", "status": "error", "retries": 3, "errors": []},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    s = P.summarize_usage(path, run_id="a")
    assert s["calls"] == 2 and s["retries"] == 1 and s["fallbacks"] == 1 and s["errors"] == 0
    assert s["input_tokens"] == 15 and s["by_model"] == {"m1": 1, "m2": 1}
    assert s["error_kinds"] == {"rate_limit": 1, "server": 1}


def test_check_models_from_metadata_offline():
    models = {
        "nvidia/nemotron-3-ultra-550b-a55b:free": {
            "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "context_length": 1000000,
            "pricing": {"prompt": "0", "completion": "0"},
            "supported_parameters": ["tools", "temperature"],
            "architecture": {"input_modalities": ["text"]},
        }
    }
    out = P.check_models("openrouter", models=models)
    ultra = out["nvidia/nemotron-3-ultra-550b-a55b:free"]
    assert ultra["exists"] and ultra["free"] and ultra["supports_tools"] and not ultra["supports_stop"]
    assert out["nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"] == {"exists": False}


# --------------------------------------------------------------------------- network (free, keyless)


@pytest.mark.network
def test_chosen_free_models_exist_on_openrouter(tmp_path):
    models = P.fetch_openrouter_models(cache_path=tmp_path / "models.json", max_age_s=0)
    report = P.check_models("openrouter", models=models)
    missing = [m for m, s in report.items() if not (s["exists"] and s["free"])]
    assert not missing, f"not present or not free on OpenRouter: {missing}"
    assert report[P.PRESETS["openrouter"]["model_id"]]["supports_tools"]
    vision = report[P.PRESETS["openrouter"]["vision_model_id"]]
    assert "image" in (vision["input_modalities"] or [])


# --------------------------------------------------------------------------- spend guard / auto cap


def test_paid_openrouter_models_are_refused(tmp_runs, monkeypatch):
    monkeypatch.setenv("GAIA_MODEL_ID", "nvidia/nemotron-3-ultra-550b-a55b")  # paid variant (no :free)
    with pytest.raises(ValueError, match="paid"):
        P.model_chain("openrouter")
    monkeypatch.setenv("GAIA_ALLOW_PAID", "1")
    assert P.model_chain("openrouter")[0] == "nvidia/nemotron-3-ultra-550b-a55b"


def test_paid_guard_also_covers_fallbacks_and_vision(tmp_runs, monkeypatch):
    monkeypatch.delenv("GAIA_ALLOW_PAID", raising=False)
    monkeypatch.setenv("GAIA_VISION_FALLBACKS", "openai/gpt-5o")
    with pytest.raises(ValueError):
        P.model_chain("openrouter", vision=True)


def test_auto_daily_cap_uses_live_remaining_allowance(tmp_runs, monkeypatch):
    monkeypatch.setattr(P, "check_openrouter_key", lambda timeout=10: {"daily_free_remaining": 997})
    lim = P.get_limiter("openrouter")
    assert lim.daily_cap == 997 - 5  # nothing counted locally yet; keep a 5-request reserve
    assert lim.rpm == 18 and lim.capacity == 2


def test_auto_daily_cap_counts_local_usage_already_spent(tmp_runs, monkeypatch):
    P.RateLimiter(rpm=None, usage_path=P.USAGE_JSON, scope="openrouter").acquire()
    P.RateLimiter(rpm=None, usage_path=P.USAGE_JSON, scope="openrouter").acquire()
    monkeypatch.setattr(P, "check_openrouter_key", lambda timeout=10: {"daily_free_remaining": 100})
    assert P.get_limiter("openrouter").daily_cap == 2 + 95


def test_auto_daily_cap_falls_back_to_documented_minimum(tmp_runs, monkeypatch):
    monkeypatch.setattr(P, "check_openrouter_key", lambda timeout=10: {"key_present": False})
    assert P.get_limiter("openrouter").daily_cap == 50


def test_explicit_daily_cap_and_no_cap_presets(tmp_runs, monkeypatch):
    monkeypatch.setenv("GAIA_DAILY_CAP", "12")
    assert P.get_limiter("openrouter").daily_cap == 12
    monkeypatch.delenv("GAIA_DAILY_CAP")
    assert P.get_limiter("lmstudio").daily_cap is None and not P.get_limiter("lmstudio").enabled


# --------------------------------------------------------------------------- circuit breaker / upstream 429
# Both behaviours come from a live smoke call on 2026-10-10: the app-restricted vision model
# answered 403 on every call, and an upstream provider's 429 did not clear across 3 retries.


@pytest.fixture(autouse=True)
def fresh_health(monkeypatch):
    monkeypatch.setattr(P, "HEALTH", P.ModelHealth())


def test_403_opens_circuit_so_next_call_skips_the_model(tmp_runs):
    forbidden = status_error(403, msg="model is only available on agentic harnesses")
    client = FakeClient({"primary:free": [forbidden], "backup:free": [completion("b")]})
    model, sleeps = make(client, tmp=tmp_runs)
    assert model.generate(user()).content == "b"
    assert model.generate(user()).content == "b"
    assert client.models_called() == ["primary:free", "backup:free", "backup:free"]  # 2nd call skipped primary
    rows = read_jsonl(tmp_runs / "usage.jsonl")
    assert rows[1]["skipped_models"] == ["primary:free"] and rows[1]["attempts"] == 1
    assert sleeps == []


def test_circuit_closes_after_ttl():
    clock = FakeClock()
    h = P.ModelHealth(ttl_s=60, clock=clock)
    h.mark_down("m", "HTTP 404")
    assert h.is_down("m") and h.snapshot() == {"m": "HTTP 404"}
    clock.t += 61
    assert not h.is_down("m")


def test_all_models_down_still_tries_them(tmp_runs):
    for m in ("primary:free", "backup:free", "last:free"):
        P.HEALTH.mark_down(m, "HTTP 404")
    client = FakeClient({"primary:free": [completion("recovered")]})
    model, _ = make(client, tmp=tmp_runs)
    assert model.generate(user()).content == "recovered"


def test_upstream_429_retries_once_then_falls_back(tmp_runs):
    upstream = status_error(
        429,
        body={"message": "Provider returned error", "code": 429,
              "metadata": {"raw": "x:free is temporarily rate-limited upstream", "provider_name": "Google AI Studio"}},
        msg="Provider returned error",
    )
    assert P.classify_error(upstream).kind == "upstream_rate_limit"
    client = FakeClient({"primary:free": [upstream], "backup:free": [completion("b")]})
    model, sleeps = make(client, tmp=tmp_runs, policy=P.RetryPolicy(max_retries=4, base_s=1, cap_s=4))
    assert model.generate(user()).content == "b"
    assert client.models_called() == ["primary:free", "primary:free", "backup:free"]
    assert len(sleeps) == 1


def test_upstream_daily_quota_is_not_mistaken_for_account_quota():
    e = status_error(429, body={"message": "Provider returned error", "metadata": {"raw": "quota per day exceeded", "provider_name": "X"}})
    assert P.classify_error(e).kind == "upstream_rate_limit"
