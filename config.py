import os
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()


# API configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_openrouter_key() -> Optional[str]:
    """Read the API key lazily so tests and local demos can run without network access."""
    return os.getenv("OPENROUTER_API_KEY")


# Model configuration. Prices are USD per 1M tokens and are used for portfolio
# benchmarks plus live usage accounting.
MODEL_REGISTRY: Dict[str, Dict[str, float]] = {
    "x-ai/grok-4-fast": {
        "max_output": 4000,
        "input_cost_per_1m": 0.20,
        "output_cost_per_1m": 0.50,
    },
    "openai/gpt-4o-mini": {
        "max_output": 4000,
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
    },
    "anthropic/claude-3.5-haiku": {
        "max_output": 4000,
        "input_cost_per_1m": 0.80,
        "output_cost_per_1m": 4.00,
    },
    # Deterministic local model used by tests, demos, and benchmark validation.
    "mock/echo": {
        "max_output": 1000,
        "input_cost_per_1m": 0.0,
        "output_cost_per_1m": 0.0,
    },
}

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "x-ai/grok-4-fast")

MODEL_CONFIG = {
    "name": DEFAULT_MODEL,
    **MODEL_REGISTRY.get(DEFAULT_MODEL, MODEL_REGISTRY["x-ai/grok-4-fast"]),
}


def get_model_config(model: Optional[str] = None) -> Dict[str, float]:
    selected = model or DEFAULT_MODEL
    config = MODEL_REGISTRY.get(selected, MODEL_REGISTRY[DEFAULT_MODEL])
    return {"name": selected, **config}


# Strategy settings
RECENT_MESSAGE_COUNT = int(os.getenv("RECENT_MESSAGE_COUNT", "15"))
SUMMARIZE_THRESHOLD = RECENT_MESSAGE_COUNT
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "2000"))
MESSAGE_COMPRESS_THRESHOLD = int(os.getenv("MESSAGE_COMPRESS_THRESHOLD", "2500"))
MESSAGE_COMPRESSED_SIZE = int(os.getenv("MESSAGE_COMPRESSED_SIZE", "800"))
TARGET_INPUT_TOKENS = int(os.getenv("TARGET_INPUT_TOKENS", "20000"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "50000"))


# Rate limiting
MIN_REQUEST_INTERVAL = float(os.getenv("MIN_REQUEST_INTERVAL", "2"))


# Database
DB_NAME = os.getenv("DB_NAME", "story_conversations.db")


# Prompts
STORY_SYSTEM_PROMPT = (
    "You are a continuity-focused writing assistant. Preserve established facts, "
    "character motivations, unresolved threads, and the user's requested tone. "
    "When prior context is summarized, treat it as authoritative memory."
)

SUMMARY_PROMPT = """Analyze the conversation and produce a rich, accurate story summary.

The summary must preserve established events, character arcs, relationships, motives,
current emotional state, world rules, active conflicts, unresolved threads, and the
latest scene position. Do not invent new events. Write in flowing prose, not bullets.
Keep the result concise enough to serve as long-term memory for future LLM calls."""

COMPRESS_PROMPT = (
    "Compress this long story segment while preserving plot events, character actions, "
    "facts, motivations, unresolved threads, and continuity-critical details."
)
