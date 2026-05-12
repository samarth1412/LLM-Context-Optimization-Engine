import math
import re
from typing import Dict, Optional


TOKEN_RE = re.compile(r"[a-z0-9_]+")
ENTITY_RE = re.compile(r"\b(?:Project|User|Client|Team|Service|Region|Atlas|Zephyr|Vega|Orion|Boreal|Atala)\b")

PREFERENCE_TERMS = {
    "prefer",
    "prefers",
    "preference",
    "favorite",
    "likes",
    "dislikes",
    "allergy",
    "allergic",
    "avoid",
    "never",
    "always",
}
PERSISTENCE_TERMS = {
    "stable",
    "durable",
    "remember",
    "memory",
    "constraint",
    "must",
    "requirement",
    "policy",
    "rule",
}
CONTRADICTION_TERMS = {
    "old",
    "previous",
    "previously",
    "used to",
    "changed",
    "replaced",
    "updated",
    "current",
    "now",
    "instead",
}
EPISODIC_TERMS = {
    "meeting",
    "incident",
    "release",
    "launch",
    "rollout",
    "decision",
    "deadline",
    "ticket",
}
NOISE_TERMS = {
    "routine",
    "chatter",
    "formatting",
    "ordinary",
    "neutral",
    "filler",
}


def _tokenize(text: str):
    return TOKEN_RE.findall((text or "").lower())


def _contains_any(text: str, terms) -> bool:
    return any(term in text for term in terms)


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def score_memory(
    content: str,
    *,
    role: str = "user",
    message_id: Optional[int] = None,
    latest_message_id: Optional[int] = None,
    retrieval_count: int = 0,
) -> Dict:
    """Score a memory for long-session retention and hierarchy assignment."""
    lower = (content or "").lower()
    tokens = _tokenize(content)
    token_count = max(1, len(tokens))

    if message_id is not None and latest_message_id is not None:
        age = max(0, int(latest_message_id) - int(message_id))
        recency = math.exp(-age / 500.0)
    else:
        recency = 0.5

    preference = 1.0 if _contains_any(lower, PREFERENCE_TERMS) else 0.0
    persistence = 1.0 if _contains_any(lower, PERSISTENCE_TERMS) else 0.0
    contradiction = 1.0 if _contains_any(lower, CONTRADICTION_TERMS) else 0.0
    stale_marker = 1.0 if _contains_any(lower, {"old", "previous", "previously", "used to"}) else 0.0
    fresh_update = 1.0 if _contains_any(lower, {"current", "updated", "now"}) else 0.0
    episodic = 1.0 if _contains_any(lower, EPISODIC_TERMS) else 0.0
    noise = 1.0 if _contains_any(lower, NOISE_TERMS) else 0.0
    entity_count = len(ENTITY_RE.findall(content or ""))
    entity_importance = _bounded(entity_count / 3.0)
    retrieval_frequency = _bounded(math.log1p(max(0, retrieval_count)) / math.log(10))
    length_signal = _bounded(min(token_count, 80) / 80.0)

    role_signal = 1.0 if role == "user" else 0.75
    score = (
        0.13 * recency
        + 0.18 * persistence
        + 0.18 * preference
        + 0.15 * entity_importance
        + 0.14 * retrieval_frequency
        + 0.12 * contradiction
        + 0.06 * episodic
        + 0.04 * length_signal
    ) * role_signal
    score -= 0.22 * noise
    score -= 0.32 * stale_marker * (1.0 - retrieval_frequency)
    score = _bounded(score)

    if stale_marker >= 1.0 and recency < 0.22 and retrieval_frequency < 0.5:
        action = "evict"
    elif (
        score >= 0.72
        or (preference >= 1.0 and stale_marker == 0.0 and recency >= 0.25)
        or (contradiction >= 1.0 and fresh_update >= 1.0 and stale_marker == 0.0)
    ):
        action = "preserve"
    elif score >= 0.34 or entity_importance >= 0.34 or episodic >= 1.0:
        action = "compress"
    else:
        action = "evict"

    if recency >= 0.88:
        layer = "working"
    elif preference >= 1.0 or persistence >= 1.0 or entity_importance >= 0.67:
        layer = "semantic"
    elif episodic >= 1.0 or contradiction >= 1.0:
        layer = "episodic"
    else:
        layer = "archived"

    signals = {
        "recency": _bounded(recency),
        "persistence": persistence,
        "contradiction_risk": contradiction,
        "stale_marker": stale_marker,
        "fresh_update": fresh_update,
        "retrieval_frequency": retrieval_frequency,
        "entity_importance": entity_importance,
        "user_preference_durability": preference,
        "episodic_event": episodic,
        "noise": noise,
        "length_signal": length_signal,
    }
    return {
        "importance_score": score,
        "memory_action": action,
        "memory_layer": layer,
        "signals": signals,
    }
