from typing import Dict, List, Optional

import logging

from config import (
    CONTEXT_POLICY,
    MAX_INPUT_TOKENS,
    MESSAGE_COMPRESS_THRESHOLD,
    MESSAGE_COMPRESSED_SIZE,
    RECENT_MESSAGE_COUNT,
    RETRIEVAL_MIN_SCORE,
    RETRIEVAL_TOP_K,
    STORY_SYSTEM_PROMPT,
    SUMMARY_MAX_TOKENS,
)
from database import (
    cache_summary,
    count_messages,
    estimate_tokens,
    get_all_messages,
    get_cached_summary,
    get_last_n_messages,
    get_latest_cached_summary,
    get_messages_range,
    get_story_context,
    record_llm_usage,
)
from llm_utils import compress_message, generate_summary
from logging_utils import configure_logging
from semantic_memory import retrieve


configure_logging()
logger = logging.getLogger(__name__)


VALID_POLICIES = {
    "full_history",
    "sliding_window",
    "summary",
    "retrieval",
    "hybrid",
    "adaptive",
}

MEMORY_QUERY_HINTS = (
    "remember",
    "recall",
    "earlier",
    "previous",
    "before",
    "original",
    "mentioned",
    "told",
    "said",
    "favorite",
    "prefer",
    "who is",
    "what is",
    "when did",
    "where did",
)

BROAD_QUERY_HINTS = (
    "summarize",
    "recap",
    "so far",
    "overall",
    "state",
    "status",
    "current",
    "story",
    "plot",
    "relationship",
    "all",
)

CURRENT_STATE_HINTS = (
    "current",
    "latest",
    "now",
    "today",
    "updated",
)


def _usage_tokens(usage: Dict) -> tuple:
    return (
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
        bool(usage.get("estimated", False)),
        int(usage.get("cached_input_tokens", 0) or 0),
        int(usage.get("cache_write_tokens", 0) or 0),
        usage.get("latency_ms"),
    )


def _record_usage_from_response(
    session_id: str,
    operation: str,
    model: Optional[str],
    usage: Dict,
) -> None:
    input_tokens, output_tokens, estimated, cached_tokens, cache_write_tokens, latency_ms = (
        _usage_tokens(usage)
    )
    record_llm_usage(
        session_id=session_id,
        operation=operation,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated=estimated,
        cached_input_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        latency_ms=latency_ms,
    )


def compress_if_needed(
    message: Dict,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict:
    token_count = estimate_tokens(message["content"], model)

    if token_count <= MESSAGE_COMPRESS_THRESHOLD:
        return message

    logger.info(
        "compressing long message",
        extra={"operation": "compression"},
    )
    compressed_content, usage = compress_message(
        message["content"], MESSAGE_COMPRESSED_SIZE, model=model
    )

    if session_id:
        _record_usage_from_response(session_id, "compression", model, usage)

    return {
        "role": message["role"],
        "content": f"[Compressed prior message]: {compressed_content}",
    }


def generate_summary_incremental(
    session_id: str,
    target_coverage: int,
    model: Optional[str] = None,
) -> str:
    latest = get_latest_cached_summary(session_id)

    if latest:
        prev_coverage, prev_summary = latest

        if prev_coverage >= target_coverage:
            return prev_summary

        new_messages = get_messages_range(session_id, prev_coverage + 1, target_coverage)
        if not new_messages:
            return prev_summary

        logger.info(
            "incremental summary update",
            extra={"session_id": session_id, "operation": "summary"},
        )
        new_summary_part, usage = generate_summary(new_messages, max_tokens=1000, model=model)
        _record_usage_from_response(session_id, "summary", model, usage)

        combined = f"{prev_summary}\n\nRecent developments: {new_summary_part}"

        if estimate_tokens(combined, model) > SUMMARY_MAX_TOKENS:
            logger.info(
                "summary too long; refreshing",
                extra={"session_id": session_id, "operation": "summary_refresh"},
            )
            all_messages = get_messages_range(session_id, 1, target_coverage)
            combined, usage = generate_summary(
                all_messages, max_tokens=SUMMARY_MAX_TOKENS, model=model
            )
            _record_usage_from_response(session_id, "summary_refresh", model, usage)

        return combined

    logger.info(
        "generating initial summary",
        extra={"session_id": session_id, "operation": "summary"},
    )
    messages = get_messages_range(session_id, 1, target_coverage)
    summary, usage = generate_summary(messages, max_tokens=SUMMARY_MAX_TOKENS, model=model)
    _record_usage_from_response(session_id, "summary", model, usage)
    return summary


def _system_message(
    session_id: str,
    summary: Optional[str] = None,
    retrieved: Optional[List[Dict]] = None,
) -> Dict:
    story = get_story_context(session_id)
    sections = [STORY_SYSTEM_PROMPT]

    if story:
        sections.append(f"Original story reference:\n{story}")
    if summary:
        sections.append(f"Conversation memory:\n{summary}")
    if retrieved:
        snippets = []
        for item in retrieved:
            snippets.append(
                f"- ({item.get('role')}, id={item.get('message_id')}, score={item.get('score')}): "
                f"{str(item.get('content', ''))[:500]}"
            )
        sections.append("Retrieved relevant prior facts:\n" + "\n".join(snippets))

    return {"role": "system", "content": "\n\n".join(sections)}


def _normalize_policy(policy: Optional[str]) -> str:
    selected = (policy or CONTEXT_POLICY or "adaptive").strip().lower()
    if selected not in VALID_POLICIES:
        logger.warning("unknown context policy; using adaptive", extra={"operation": "build_context"})
        return "adaptive"
    return selected


def _query_intent(query: Optional[str]) -> Dict[str, bool]:
    normalized = (query or "").lower()
    return {
        "memory": any(hint in normalized for hint in MEMORY_QUERY_HINTS),
        "broad": any(hint in normalized for hint in BROAD_QUERY_HINTS),
        "current_state": any(hint in normalized for hint in CURRENT_STATE_HINTS),
    }


def _filter_retrieved(items: Optional[List[Dict]]) -> List[Dict]:
    if not items:
        return []
    return [item for item in items if float(item.get("score", 0) or 0) >= RETRIEVAL_MIN_SCORE]


def _policy_decision(
    selected_policy: str,
    query: Optional[str],
    retrieved: Optional[List[Dict]],
) -> Dict:
    intent = _query_intent(query)
    strong_retrieval = bool(
        retrieved and float(retrieved[0].get("score", 0) or 0) >= RETRIEVAL_MIN_SCORE
    )

    include_retrieval = bool(selected_policy in {"retrieval", "hybrid"} and retrieved)
    include_summary = selected_policy in {"summary", "hybrid"}
    reason = f"fixed policy: {selected_policy}"

    if selected_policy == "full_history":
        return {
            "policy": selected_policy,
            "include_summary": False,
            "include_retrieval": False,
            "intent": intent,
            "strong_retrieval": strong_retrieval,
            "top_retrieval_score": float(retrieved[0].get("score", 0) or 0) if retrieved else 0.0,
            "reason": "full history baseline includes all stored messages",
        }

    if selected_policy == "sliding_window":
        return {
            "policy": selected_policy,
            "include_summary": False,
            "include_retrieval": False,
            "intent": intent,
            "strong_retrieval": strong_retrieval,
            "top_retrieval_score": float(retrieved[0].get("score", 0) or 0) if retrieved else 0.0,
            "reason": "sliding window baseline uses only recent messages",
        }

    if selected_policy == "adaptive":
        include_retrieval = bool(
            query
            and retrieved
            and not intent["current_state"]
            and (intent["memory"] or intent["broad"] or strong_retrieval)
        )
        include_summary = intent["broad"] or intent["current_state"] or not include_retrieval or not query

        if intent["current_state"]:
            reason = "current-state query: prefer summary plus recent context over broad retrieval"
        elif include_retrieval and not include_summary:
            reason = "memory/fact query with retrieved evidence: use retrieved facts plus recent context"
        elif include_retrieval and include_summary:
            reason = "broad memory query: combine summary and retrieval evidence"
        else:
            reason = "no strong retrieval signal: use summary plus recent context"

    return {
        "policy": selected_policy,
        "include_summary": include_summary,
        "include_retrieval": include_retrieval,
        "intent": intent,
        "strong_retrieval": strong_retrieval,
        "top_retrieval_score": float(retrieved[0].get("score", 0) or 0) if retrieved else 0.0,
        "reason": reason,
    }


def explain_policy(
    session_id: str,
    query: Optional[str] = None,
    policy: Optional[str] = None,
    retrieval_k: int = RETRIEVAL_TOP_K,
) -> Dict:
    selected_policy = _normalize_policy(policy)
    retrieved = _filter_retrieved(retrieve(session_id, query, top_k=retrieval_k)) if query else []
    return _policy_decision(selected_policy, query, retrieved)


def build_context(
    session_id: str,
    model: Optional[str] = None,
    query: Optional[str] = None,
    retrieval_k: int = RETRIEVAL_TOP_K,
    policy: Optional[str] = None,
) -> List[Dict]:
    """Build context from previously stored messages only.

    The current user prompt should be appended by the caller after this function
    returns. That avoids duplicating the newest prompt in the LLM request.
    """
    total_messages = count_messages(session_id)
    selected_policy = _normalize_policy(policy)
    logger.info(
        "build context",
        extra={"session_id": session_id, "operation": "build_context", "policy": selected_policy},
    )

    if selected_policy == "full_history":
        messages = get_all_messages(session_id)
        messages = [compress_if_needed(msg, session_id, model) for msg in messages]
        return [_system_message(session_id), *messages]

    if total_messages <= RECENT_MESSAGE_COUNT:
        messages = get_all_messages(session_id)
        messages = [compress_if_needed(msg, session_id, model) for msg in messages]
        retrieved = _filter_retrieved(retrieve(session_id, query, top_k=retrieval_k)) if query else []
        decision = _policy_decision(selected_policy, query, retrieved)
        return [
            _system_message(
                session_id,
                retrieved=retrieved if decision["include_retrieval"] else None,
            ),
            *messages,
        ]

    old_message_count = total_messages - RECENT_MESSAGE_COUNT
    logger.info(
        "long conversation",
        extra={"session_id": session_id, "operation": "build_context"},
    )

    recent_messages = get_last_n_messages(session_id, RECENT_MESSAGE_COUNT)
    recent_messages = [
        compress_if_needed(msg, session_id=session_id, model=model) for msg in recent_messages
    ]

    if selected_policy == "sliding_window":
        return [_system_message(session_id), *recent_messages]

    retrieved = _filter_retrieved(retrieve(session_id, query, top_k=retrieval_k)) if query else []
    decision = _policy_decision(selected_policy, query, retrieved)
    include_summary = bool(decision["include_summary"])
    include_retrieval = bool(decision["include_retrieval"])

    summary = None
    if include_summary:
        summary = get_cached_summary(session_id, old_message_count)
        if not summary:
            summary = generate_summary_incremental(session_id, old_message_count, model=model)
            cache_summary(session_id, old_message_count, summary)
        else:
            logger.info(
                "using cached summary",
                extra={"session_id": session_id, "operation": "build_context"},
            )

    context = [
        _system_message(
            session_id,
            summary=summary,
            retrieved=retrieved if include_retrieval else None,
        ),
        *recent_messages,
    ]
    total_tokens = sum(estimate_tokens(msg["content"], model) for msg in context)
    logger.info(
        "context tokens computed",
        extra={"session_id": session_id, "operation": "build_context"},
    )

    if total_tokens > MAX_INPUT_TOKENS:
        logger.info(
            "context exceeds safety limit; trimming",
            extra={"session_id": session_id, "operation": "build_context"},
        )
        context = [_system_message(session_id, summary), *recent_messages[-10:]]

    return context


def context_preview(
    session_id: str,
    model: Optional[str] = None,
    query: Optional[str] = None,
    policy: Optional[str] = None,
) -> Dict:
    """Return a compact view of the context that would be sent to the model."""
    messages = build_context(session_id, model=model, query=query, policy=policy)
    token_counts = [estimate_tokens(msg["content"], model) for msg in messages]
    return {
        "session_id": session_id,
        "policy": _normalize_policy(policy),
        "policy_decision": explain_policy(session_id, query=query, policy=policy),
        "message_count": len(messages),
        "context_tokens": sum(token_counts),
        "messages": [
            {
                "role": msg["role"],
                "tokens": token_counts[index],
                "preview": msg["content"][:300],
            }
            for index, msg in enumerate(messages)
        ],
    }
