"""Provider presets: one table shared by gaia_agent.config and gaia_agent.providers.

Kept dependency-free so ``import gaia_agent.config`` stays cheap. Every preset is an
OpenAI-compatible endpoint on a free tier or a local server. On OpenRouter only ``:free``
model ids are allowed (providers.assert_free_models): the account has paid credits, so any
other id WOULD be billed.

Keys per preset: base_url, key_env, model_id, vision_model_id, fallbacks, vision_fallbacks,
headers, rpm, burst, daily_cap, flatten_messages_as_text, timeout_s.
"""

from __future__ import annotations

import os
from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        # Verified free + present in GET /api/v1/models, and answered live with this account's key,
        # on 2026-10-10:
        #   nemotron-3-ultra       ctx 1,000,000  text              tools yes  stop NO   (text primary)
        #   nemotron-3-super       ctx   262,144  text              tools yes  stop NO
        #   gemma-4-31b-it         ctx   262,144  text+image        tools yes  stop NO   (often 429 upstream)
        #   nemotron-3-nano-omni   ctx   256,000  text+image+audio  tools yes  stop NO   (vision primary)
        #   dots-3-note-preview                   text+image                             (vision fallback)
        # "stop NO" matters for a CodeAgent: ResilientModel trims those outputs client-side.
        # thinkingmachines/inkling(:free, -small:free) were removed: they answer 403 to this key
        # ("only available on agentic harnesses").
        "model_id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "vision_model_id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "fallbacks": ["nvidia/nemotron-3-super-120b-a12b:free", "google/gemma-4-31b-it:free"],
        "vision_fallbacks": [
            "dots-studio/dots-3-note-preview:free",
            "google/gemma-4-31b-it:free",
            "google/gemma-4-26b-a4b-it:free",
        ],
        "headers": {
            "HTTP-Referer": "https://github.com/bakulbadwal",
            "X-Title": "GAIA Agent (HF Agents Course)",
        },
        "rpm": 18,  # 10% under the documented 20/min
        "burst": 2,  # burst + rpm = 20 => never more than 20 requests in any 60 s window
        # "auto": read today's remaining free-model allowance from GET /api/v1/key (50 or 1000/day
        # depending on credits ever purchased); falls back to 50. GAIA_DAILY_CAP=<n> overrides.
        "daily_cap": "auto",
        "flatten_messages_as_text": False,
        "timeout_s": 240,  # reasoning models can think for minutes
    },
    # The presets below were NOT verified live (no key on this machine); a wrong id only costs
    # one 404, which triggers the next fallback.
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model_id": "gemini-2.5-flash",
        "vision_model_id": "gemini-2.5-flash",
        "fallbacks": ["gemini-2.5-flash-lite"],
        "vision_fallbacks": ["gemini-2.5-flash-lite"],
        "headers": {},
        "rpm": 8,
        "burst": 1,
        "daily_cap": None,
        "flatten_messages_as_text": False,
        "timeout_s": 180,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_id": "openai/gpt-oss-120b",
        "vision_model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "fallbacks": ["llama-3.3-70b-versatile"],
        "vision_fallbacks": [],
        "headers": {},
        "rpm": 25,
        "burst": 2,
        "daily_cap": None,
        "flatten_messages_as_text": True,
        "timeout_s": 180,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "key_env": None,
        "model_id": None,  # None = first model the local server lists
        "vision_model_id": None,
        "fallbacks": [],
        "vision_fallbacks": [],
        "headers": {},
        "rpm": 0,  # local: no limit
        "burst": 1,
        "daily_cap": None,
        "flatten_messages_as_text": False,
        "timeout_s": 600,
    },
}
DEFAULT_PRESET = "openrouter"
FALLBACK_PRESET = "gemini"  # used only when no OpenRouter key is configured


def default_provider() -> str:
    """openrouter when OPENROUTER_API_KEY is set, else gemini (GAIA_PROVIDER always wins upstream)."""
    return DEFAULT_PRESET if os.environ.get("OPENROUTER_API_KEY") else FALLBACK_PRESET
