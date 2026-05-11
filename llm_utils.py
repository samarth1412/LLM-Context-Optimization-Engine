import json
import logging
import time
from typing import Dict, Iterator, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    COMPRESS_PROMPT,
    DEFAULT_MODEL,
    GEMINI_URL_BASE,
    MODEL_CONFIG,
    OPENAI_URL,
    OPENROUTER_URL,
    SUMMARY_PROMPT,
    get_gemini_key,
    get_openai_key,
    get_openrouter_key,
)
from database import estimate_messages_tokens, estimate_tokens
from logging_utils import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=("POST",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _selected_model(model: Optional[str]) -> str:
    return model or MODEL_CONFIG["name"] or DEFAULT_MODEL


def _provider_model_name(model: str) -> str:
    if model.startswith(("openai/", "google/")):
        return model.split("/", 1)[1]
    return model


def _gemini_payload(
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
) -> Dict:
    system_parts = []
    contents = []

    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        if role == "system":
            system_parts.append(content)
            continue

        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": content}],
            }
        )

    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    if system_parts:
        payload["system_instruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}],
        }
    return payload


def _gemini_text(response_data: Dict) -> str:
    candidates = response_data.get("candidates") or []
    if not candidates:
        return ""
    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )
    return "".join(str(part.get("text", "")) for part in parts)


def _normalize_gemini_usage(usage: Dict, messages: List[Dict], content: str, model: str) -> Dict:
    prompt_tokens = usage.get("promptTokenCount")
    completion_tokens = usage.get("candidatesTokenCount")
    cached_tokens = usage.get("cachedContentTokenCount", 0)
    estimated = False

    if prompt_tokens is None:
        prompt_tokens = estimate_messages_tokens(messages, model)
        estimated = True
    if completion_tokens is None:
        completion_tokens = estimate_tokens(content, model)
        estimated = True

    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int((prompt_tokens or 0) + (completion_tokens or 0)),
        "cached_input_tokens": int(cached_tokens or 0),
        "cache_write_tokens": 0,
        "estimated": estimated,
    }


def _normalize_usage(usage: Dict, messages: List[Dict], content: str, model: str) -> Dict:
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    estimated = False
    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached_tokens = (
        prompt_details.get("cached_tokens")
        or usage.get("cached_tokens")
        or usage.get("cache_read_input_tokens")
        or 0
    )
    cache_write_tokens = (
        prompt_details.get("cache_write_tokens")
        or usage.get("cache_write_tokens")
        or usage.get("cache_creation_input_tokens")
        or 0
    )

    if prompt_tokens is None:
        prompt_tokens = usage.get("input_tokens")
    if completion_tokens is None:
        completion_tokens = usage.get("output_tokens")

    if prompt_tokens is None:
        prompt_tokens = estimate_messages_tokens(messages, model)
        estimated = True
    if completion_tokens is None:
        completion_tokens = estimate_tokens(content, model)
        estimated = True

    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int((prompt_tokens or 0) + (completion_tokens or 0)),
        "cached_input_tokens": int(cached_tokens or 0),
        "cache_write_tokens": int(cache_write_tokens or 0),
        "estimated": bool(usage.get("estimated", estimated)),
    }


def _mock_response(messages: List[Dict], model: str) -> Tuple[str, Dict]:
    last_user = next(
        (msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "user"),
        "",
    )
    content = f"Mock response: {last_user[:300]}"
    return content, _normalize_usage({}, messages, content, model)


def call_llm(
    messages: List[Dict],
    max_tokens: int = 4000,
    temperature: float = 0.8,
    model: Optional[str] = None,
) -> Tuple[str, Dict]:
    """Call OpenRouter or a deterministic local mock model."""
    selected_model = _selected_model(model)
    if selected_model.startswith("mock/"):
        return _mock_response(messages, selected_model)

    openai_key = get_openai_key()
    gemini_key = get_gemini_key()
    use_openai_direct = selected_model.startswith("openai/") and bool(openai_key)
    use_gemini_direct = selected_model.startswith("google/") and bool(gemini_key)
    api_key = (
        openai_key
        if use_openai_direct
        else gemini_key
        if use_gemini_direct
        else get_openrouter_key()
    )
    if not api_key:
        raise RuntimeError(
            "No provider key is set. Use OPENAI_API_KEY for openai/* models, "
            "GEMINI_API_KEY for google/* models, OPENROUTER_API_KEY for OpenRouter, "
            "or model='mock/echo' for local tests."
        )

    payload = _gemini_payload(messages, max_tokens, temperature) if use_gemini_direct else {
        "model": _provider_model_name(selected_model),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if use_gemini_direct:
        headers = {
            "x-goog-api-key": str(api_key),
            "Content-Type": "application/json",
        }

    try:
        start = time.perf_counter()
        response = _session().post(
            url=(
                f"{GEMINI_URL_BASE}/{_provider_model_name(selected_model)}:generateContent"
                if use_gemini_direct
                else OPENAI_URL
                if use_openai_direct
                else OPENROUTER_URL
            ),
            json=payload,
            headers=headers,
            timeout=45,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        response.raise_for_status()
        response_data = response.json()
        if use_gemini_direct:
            content = _gemini_text(response_data)
            usage = _normalize_gemini_usage(
                response_data.get("usageMetadata", {}),
                messages,
                content,
                selected_model,
            )
        else:
            content = response_data["choices"][0]["message"]["content"]
            usage = _normalize_usage(response_data.get("usage", {}), messages, content, selected_model)
        usage["latency_ms"] = latency_ms
        return content, usage
    except Exception as exc:
        logger.exception("LLM call failed", extra={"model": selected_model, "operation": "chat"})
        raise


def call_llm_stream(
    messages: List[Dict],
    max_tokens: int = 4000,
    temperature: float = 0.8,
    model: Optional[str] = None,
) -> Iterator[str]:
    """Stream text chunks from OpenRouter or a local mock model."""
    selected_model = _selected_model(model)
    if selected_model.startswith("mock/"):
        content, _ = _mock_response(messages, selected_model)
        yield content
        return

    if selected_model.startswith("google/") and get_gemini_key():
        content, _ = call_llm(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )
        yield content
        return

    openai_key = get_openai_key()
    use_openai_direct = selected_model.startswith("openai/") and bool(openai_key)
    api_key = openai_key if use_openai_direct else get_openrouter_key()
    if not api_key:
        raise RuntimeError(
            "No provider key is set. Use OPENAI_API_KEY for openai/* models, "
            "OPENROUTER_API_KEY for OpenRouter, or model='mock/echo' for local tests."
        )

    payload = {
        "model": _provider_model_name(selected_model),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = _session().post(
            url=OPENAI_URL if use_openai_direct else OPENROUTER_URL,
            json=payload,
            headers=headers,
            stream=True,
            timeout=45,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue

            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if "choices" not in data or not data["choices"]:
                continue

            delta = data["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content
    except Exception as exc:
        logger.exception(
            "Streaming LLM call failed",
            extra={"model": selected_model, "operation": "chat_stream"},
        )
        raise


def generate_summary(
    messages: List[Dict],
    max_tokens: int = 2000,
    model: Optional[str] = None,
) -> Tuple[str, Dict]:
    conversation_text = ""
    for msg in messages:
        conversation_text += f"{msg['role']}: {msg['content']}\n\n"

    summary_messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": f"Summarize this story conversation:\n\n{conversation_text}"},
    ]

    try:
        return call_llm(summary_messages, max_tokens=max_tokens, temperature=0.5, model=model)
    except Exception as exc:
        logger.exception(
            "Summary generation failed",
            extra={"model": _selected_model(model), "operation": "summary"},
        )
        fallback = "Summary unavailable; use recent messages as source of truth."
        usage = _normalize_usage({}, summary_messages, fallback, _selected_model(model))
        return fallback, usage


def compress_message(
    content: str,
    target_tokens: int = 800,
    model: Optional[str] = None,
) -> Tuple[str, Dict]:
    compress_messages = [
        {"role": "system", "content": COMPRESS_PROMPT},
        {"role": "user", "content": content},
    ]

    try:
        return call_llm(
            compress_messages,
            max_tokens=target_tokens,
            temperature=0.4,
            model=model,
        )
    except Exception as exc:
        logger.exception(
            "Message compression failed",
            extra={"model": _selected_model(model), "operation": "compression"},
        )
        fallback = content[: target_tokens * 4]
        usage = _normalize_usage({}, compress_messages, fallback, _selected_model(model))
        return fallback, usage
