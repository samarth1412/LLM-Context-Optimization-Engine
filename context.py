from typing import Dict, List, Optional

import logging

from config import (
    MAX_INPUT_TOKENS,
    MESSAGE_COMPRESS_THRESHOLD,
    MESSAGE_COMPRESSED_SIZE,
    RECENT_MESSAGE_COUNT,
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


def _usage_tokens(usage: Dict) -> tuple:
    return (
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
        bool(usage.get("estimated", False)),
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
        input_tokens, output_tokens, estimated = _usage_tokens(usage)
        record_llm_usage(
            session_id=session_id,
            operation="compression",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated=estimated,
        )

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
        input_tokens, output_tokens, estimated = _usage_tokens(usage)
        record_llm_usage(
            session_id,
            "summary",
            model,
            input_tokens,
            output_tokens,
            estimated=estimated,
        )

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
            input_tokens, output_tokens, estimated = _usage_tokens(usage)
            record_llm_usage(
                session_id,
                "summary_refresh",
                model,
                input_tokens,
                output_tokens,
                estimated=estimated,
            )

        return combined

    logger.info(
        "generating initial summary",
        extra={"session_id": session_id, "operation": "summary"},
    )
    messages = get_messages_range(session_id, 1, target_coverage)
    summary, usage = generate_summary(messages, max_tokens=SUMMARY_MAX_TOKENS, model=model)
    input_tokens, output_tokens, estimated = _usage_tokens(usage)
    record_llm_usage(
        session_id,
        "summary",
        model,
        input_tokens,
        output_tokens,
        estimated=estimated,
    )
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


def build_context(
    session_id: str,
    model: Optional[str] = None,
    query: Optional[str] = None,
    retrieval_k: int = 6,
) -> List[Dict]:
    """Build context from previously stored messages only.

    The current user prompt should be appended by the caller after this function
    returns. That avoids duplicating the newest prompt in the LLM request.
    """
    total_messages = count_messages(session_id)
    logger.info(
        "build context",
        extra={"session_id": session_id, "operation": "build_context"},
    )

    if total_messages <= RECENT_MESSAGE_COUNT:
        messages = get_all_messages(session_id)
        messages = [compress_if_needed(msg, session_id, model) for msg in messages]
        retrieved = retrieve(session_id, query, top_k=retrieval_k) if query else None
        return [_system_message(session_id, retrieved=retrieved), *messages]

    old_message_count = total_messages - RECENT_MESSAGE_COUNT
    logger.info(
        "long conversation",
        extra={"session_id": session_id, "operation": "build_context"},
    )

    summary = get_cached_summary(session_id, old_message_count)
    if not summary:
        summary = generate_summary_incremental(session_id, old_message_count, model=model)
        cache_summary(session_id, old_message_count, summary)
    else:
        logger.info(
            "using cached summary",
            extra={"session_id": session_id, "operation": "build_context"},
        )

    recent_messages = get_last_n_messages(session_id, RECENT_MESSAGE_COUNT)
    recent_messages = [
        compress_if_needed(msg, session_id=session_id, model=model) for msg in recent_messages
    ]

    retrieved = retrieve(session_id, query, top_k=retrieval_k) if query else None
    context = [_system_message(session_id, summary, retrieved=retrieved), *recent_messages]
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


def context_preview(session_id: str, model: Optional[str] = None) -> Dict:
    """Return a compact view of the context that would be sent to the model."""
    messages = build_context(session_id, model=model)
    token_counts = [estimate_tokens(msg["content"], model) for msg in messages]
    return {
        "session_id": session_id,
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
