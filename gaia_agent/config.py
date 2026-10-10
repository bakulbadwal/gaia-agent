"""Configuration: paths, env loading, provider selection and the run-level env contract.

Provider is chosen with GAIA_PROVIDER (openrouter | gemini | groq | lmstudio). Default:
openrouter when OPENROUTER_API_KEY is set, else gemini. The preset table lives in
gaia_agent/presets.py; the models themselves are built by gaia_agent/providers.py
(rate limiter, daily budget, backoff, fallback chain, usage log), which get_model() and
vision_complete() delegate to.

Env overrides:
    GAIA_PROVIDER         openrouter | gemini | groq | lmstudio
    GAIA_MODEL_ID         model used by the CodeAgent
    GAIA_VISION_MODEL_ID  model used by vision_complete (default: the preset's vision model,
                          else the text model)
    GAIA_FALLBACKS / GAIA_VISION_FALLBACKS  comma lists ("" disables fallbacks)
    GAIA_BASE_URL         override the preset base URL
    GAIA_RPM              client-side requests-per-minute cap (0 = off)
    GAIA_DAILY_CAP        local daily request cap (<n> or "auto")
    GAIA_TEMPERATURE      sampling temperature (default 0)
    GAIA_FLATTEN          1/0: force flatten_messages_as_text
    GAIA_ALLOW_PAID       keep OFF: non-":free" OpenRouter ids are refused (a key with
                          credits would bill them)

Run-level contract (read by gaia_agent.agent and the runners):
    GAIA_RUN_ID           groups traces (runs/traces/<run_id>/) and usage rows; default: a
                          timestamp chosen once per process
    GAIA_ARCH             single | multi
    GAIA_DISABLE_TOOLS    comma list of tool names to remove from the agent
    GAIA_VERIFY           1 = one extra format-only verifier call after the answer
    GAIA_RUNS_DIR         where traces and error analysis go (default <project>/runs)

Secrets are read from the environment (plus ~/.claude/.env when it exists) and are never
printed or returned by provider_info().
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from .presets import PRESETS as PROVIDERS  # one table for config + providers
from .presets import default_provider

API_URL = "https://agents-course-unit4-scoring.hf.space"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
FILES_DIR = DATA_DIR / "files"
ENV_FILE = Path.home() / ".claude" / ".env"

# Keys this project must never use (paid APIs). load_env skips them entirely.
_BLOCKED_ENV_KEYS = {"XAI_API_KEY"}

DEFAULT_PROVIDER = "openrouter"  # when OPENROUTER_API_KEY is present; see presets.default_provider()

_env_loaded = False


def load_env(path: str | os.PathLike | None = None, override: bool = False) -> list[str]:
    """Load KEY=VALUE lines from ~/.claude/.env (or `path`) into os.environ.

    Existing environment variables win unless override=True. Returns the names
    (never the values) of the keys it set.
    """
    global _env_loaded
    env_path = Path(path) if path else ENV_FILE
    loaded: list[str] = []
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key.replace("_", "").isalnum() or key in _BLOCKED_ENV_KEYS:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if override or key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    if path is None:
        _env_loaded = True
    return loaded


def _ensure_env() -> None:
    if not _env_loaded:
        load_env()


# --------------------------------------------------------------------------- run-level contract


def runs_dir() -> Path:
    """Root for traces and error analysis (GAIA_RUNS_DIR, default <project>/runs)."""
    return Path(os.environ.get("GAIA_RUNS_DIR") or RUNS_DIR)


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def get_run_id() -> str:
    """GAIA_RUN_ID, or a timestamp id chosen once per process (exported so usage rows match)."""
    rid = (os.environ.get("GAIA_RUN_ID") or "").strip()
    if not rid:
        rid = new_run_id()
        os.environ["GAIA_RUN_ID"] = rid
    return rid


def arch() -> str:
    a = (os.environ.get("GAIA_ARCH") or "single").strip().lower() or "single"
    if a not in ("single", "multi"):
        raise ValueError(f"GAIA_ARCH must be 'single' or 'multi', got {a!r}")
    return a


def disabled_tools() -> list[str]:
    return sorted({t.strip() for t in (os.environ.get("GAIA_DISABLE_TOOLS") or "").split(",") if t.strip()})


def verify_enabled() -> bool:
    return (os.environ.get("GAIA_VERIFY") or "0").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- provider


def provider_name() -> str:
    _ensure_env()
    name = (os.environ.get("GAIA_PROVIDER") or "").strip().lower() or default_provider()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown GAIA_PROVIDER {name!r}; choose one of {sorted(PROVIDERS)}")
    return name


def _preset() -> dict:
    return PROVIDERS[provider_name()]


def _base_url() -> str:
    return os.environ.get("GAIA_BASE_URL") or _preset()["base_url"]


def _rpm() -> float | None:
    raw = os.environ.get("GAIA_RPM")
    rpm = float(raw) if raw not in (None, "") else float(_preset()["rpm"] or 0)
    return rpm if rpm > 0 else None


def _temperature() -> float | None:
    raw = os.environ.get("GAIA_TEMPERATURE", "0")
    if raw.strip().lower() in ("", "none", "default"):
        return None
    return float(raw)


_local_model_cache: dict[str, str] = {}


def _first_local_model(base_url: str) -> str:
    """Ask a local OpenAI-compatible server (LM Studio) which model is loaded."""
    if base_url in _local_model_cache:
        return _local_model_cache[base_url]
    import requests

    try:
        r = requests.get(base_url.rstrip("/") + "/models", timeout=5)
        r.raise_for_status()
        ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        chat_ids = [i for i in ids if "embed" not in i.lower()]
        model = (chat_ids or ids or ["local-model"])[0]
    except Exception:
        model = "local-model"
    _local_model_cache[base_url] = model
    return model


def model_id() -> str:
    _ensure_env()
    explicit = os.environ.get("GAIA_MODEL_ID")
    if explicit:
        return explicit
    preset = _preset()
    if preset["model_id"]:
        return preset["model_id"]
    return _first_local_model(_base_url())


def vision_model_id() -> str:
    """GAIA_VISION_MODEL_ID, else the preset's vision model, else the text model.

    GAIA_MODEL_ID alone does not move vision calls onto a text model: a text-only model swap
    (e.g. an ablation) would otherwise send images to a model that cannot see them.
    """
    _ensure_env()
    explicit = os.environ.get("GAIA_VISION_MODEL_ID")
    if explicit:
        return explicit
    return _preset()["vision_model_id"] or model_id()


def _split(raw: str | None) -> list[str] | None:
    return None if raw is None else [x.strip() for x in raw.split(",") if x.strip()]


def provider_info() -> dict:
    """Describe the active provider. Contains no secrets (only whether a key is present)."""
    _ensure_env()
    name = provider_name()
    preset = PROVIDERS[name]
    key_env = preset["key_env"]
    text_model = os.environ.get("GAIA_MODEL_ID") or preset["model_id"] or "(first local model)"
    fb = _split(os.environ.get("GAIA_FALLBACKS"))
    vfb = _split(os.environ.get("GAIA_VISION_FALLBACKS"))
    return {
        "provider": name,
        "base_url": _base_url(),
        "model_id": text_model,
        "vision_model_id": os.environ.get("GAIA_VISION_MODEL_ID") or preset["vision_model_id"] or text_model,
        "fallbacks": fb if fb is not None else list(preset.get("fallbacks") or []),
        "vision_fallbacks": vfb if vfb is not None else list(preset.get("vision_fallbacks") or []),
        "key_env": key_env,
        "key_present": bool(os.environ.get(key_env)) if key_env else None,
        "rpm": _rpm(),
        "temperature": _temperature(),
        "allow_paid": os.environ.get("GAIA_ALLOW_PAID") == "1",
    }


def get_model(**overrides):
    """The agent's model: a providers.ResilientModel for the active provider.

    Shares the process-wide rate limiter and daily budget with vision_complete, retries with
    backoff, falls back along the preset's chain, and logs every call to runs/usage.jsonl.
    On OpenRouter a non-":free" model id raises ValueError (unless GAIA_ALLOW_PAID=1).
    """
    from .providers import make_model

    _ensure_env()
    name = provider_name()
    mid = overrides.pop("model_id", None) or model_id()
    flatten = os.environ.get("GAIA_FLATTEN")
    params: dict = {}
    if flatten is not None:
        params["flatten_messages_as_text"] = flatten.lower() in ("1", "true", "yes")
    params.update(overrides)
    return make_model(name, model_id=mid, **params)


def vision_complete(prompt: str, image_paths: list[str], max_tokens: int = 1500) -> str:
    """Ask the vision model about local images (same limiter / retries / fallbacks / usage log)."""
    from . import providers

    _ensure_env()
    return providers.vision_complete(
        prompt, image_paths, provider_name(), model_id=vision_model_id(), max_tokens=max_tokens
    )


def list_models() -> list[str]:
    """List model ids the active provider exposes (free metadata call)."""
    from openai import OpenAI

    preset = _preset()
    key_env = preset["key_env"]
    key = os.environ.get(key_env) if key_env else os.environ.get("GAIA_LOCAL_API_KEY", "lm-studio")
    if not key:
        raise RuntimeError(f"{key_env} is not set (put it in the environment or {ENV_FILE}).")
    client = OpenAI(api_key=key, base_url=_base_url(), timeout=60, max_retries=2)
    return sorted(m.id for m in client.models.list())
