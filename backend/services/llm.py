"""
Single LLM-provider seam.

Every chat / vision / transcription call in the webapp builds its client here,
so switching provider — OpenAI ↔ a local Ollama or vLLM server ↔ Azure — is an
env change, not a code change. Ollama and vLLM both speak the OpenAI wire
format, so pointing `LLM_BASE_URL` at them (e.g. http://localhost:11434/v1) is
all it takes. Embeddings have their own seam in embeddings.py.
"""

import os

import openai
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# OpenAI-compatible endpoint override; unset → OpenAI's own API.
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
_API_KEY = os.getenv("OPENAI_API_KEY", "")


def has_api_key() -> bool:
    """True when a usable key is configured (not the .env placeholder).

    For a keyless local provider (Ollama) set OPENAI_API_KEY to any non-empty
    sentinel like `ollama` so this returns True and real calls are attempted.
    """
    return bool(_API_KEY) and not _API_KEY.startswith("sk-your")


def is_mock() -> bool:
    """No usable key → callers should emit canned output instead of calling out."""
    return not has_api_key()


def get_client() -> openai.OpenAI:
    """The one place an LLM client is constructed."""
    return openai.OpenAI(api_key=_API_KEY or "sk-noop", base_url=LLM_BASE_URL)
