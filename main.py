import json
import logging
import time
from threading import Lock

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import MIN_REQUEST_INTERVAL, MODEL_CONFIG, MODEL_REGISTRY, get_openrouter_key
from context import build_context, context_preview, generate_summary_incremental
from database import (
    cache_summary,
    count_messages,
    delete_session,
    estimate_messages_tokens,
    estimate_tokens,
    get_all_sessions,
    get_cached_summary,
    get_memory_hierarchy,
    get_messages,
    get_session_stats,
    get_usage_timeseries,
    init_database,
    record_llm_usage,
    save_story_context,
    store_message_with_usage,
)
from llm_utils import call_llm, call_llm_stream
from logging_utils import configure_logging
from semantic_memory import index_message
import benchmark as bench_mod


app = FastAPI(title="LLM Context Optimization Engine")
configure_logging()
logger = logging.getLogger(__name__)

last_request_time = 0.0
rate_limit_lock = Lock()
port = 9000

init_database()


def enforce_rate_limit():
    global last_request_time

    if MIN_REQUEST_INTERVAL <= 0:
        return

    current_time = time.time()
    with rate_limit_lock:
        time_since_last = current_time - last_request_time
        if time_since_last < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {wait_time:.1f} seconds",
            )
        last_request_time = current_time


def cost_response(stats):
    return {
        "input": f"${stats.get('input_cost_usd', 0.0):.6f}",
        "output": f"${stats.get('output_cost_usd', 0.0):.6f}",
        "total": f"${stats['total_cost_usd']:.6f}",
        "chat": f"${stats.get('chat_cost_usd', 0.0):.6f}",
        "background": f"${stats.get('background_cost_usd', 0.0):.6f}",
    }


@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "default_model": MODEL_CONFIG["name"],
        "openrouter_configured": bool(get_openrouter_key()),
    }


@app.get("/api/models")
def models():
    return {
        "default": MODEL_CONFIG["name"],
        "models": [
            {
                "name": name,
                "max_output": config["max_output"],
                "input_cost_per_1m": config["input_cost_per_1m"],
                "output_cost_per_1m": config["output_cost_per_1m"],
            }
            for name, config in MODEL_REGISTRY.items()
        ],
    }


class PromptIn(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = MODEL_CONFIG["name"]
    session_id: str = Field("default", min_length=1, max_length=160)
    max_tokens: int = Field(int(MODEL_CONFIG["max_output"]), ge=1, le=32000)


@app.post("/api/chat")
def chat(body: PromptIn):
    enforce_rate_limit()

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if body.model not in MODEL_REGISTRY and not body.model.startswith("mock/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{body.model}'. Allowed: {sorted(MODEL_REGISTRY.keys())}",
        )

    logger.info(
        "chat request",
        extra={"session_id": body.session_id, "model": body.model, "operation": "chat"},
    )

    context = build_context(body.session_id, model=body.model, query=prompt)
    context.append({"role": "user", "content": prompt})

    try:
        assistant_response, usage = call_llm(
            context,
            max_tokens=body.max_tokens,
            model=body.model,
        )

        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        estimated = bool(usage.get("estimated", False))
        cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
        cache_write_tokens = int(usage.get("cache_write_tokens", 0) or 0)
        latency_ms = usage.get("latency_ms")

        user_id = store_message_with_usage(body.session_id, "user", prompt)
        if user_id:
            index_message(body.session_id, int(user_id), "user", prompt)

        assistant_id = store_message_with_usage(
            body.session_id,
            "assistant",
            assistant_response,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        if assistant_id:
            index_message(body.session_id, int(assistant_id), "assistant", assistant_response)
        record_llm_usage(
            body.session_id,
            "chat",
            body.model,
            prompt_tokens,
            completion_tokens,
            estimated=estimated,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
            latency_ms=latency_ms,
        )

        logger.info(
            "chat response",
            extra={
                "session_id": body.session_id,
                "model": body.model,
                "operation": "chat",
            },
        )

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": assistant_response,
                    }
                }
            ],
            "usage": usage,
        }

    except Exception as exc:
        logger.exception(
            "chat failed",
            extra={"session_id": body.session_id, "model": body.model, "operation": "chat"},
        )
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/summary/{session_id}")
def get_summary_endpoint(session_id: str, model: str = MODEL_CONFIG["name"]):
    total_messages = count_messages(session_id)

    if total_messages == 0:
        return {"session_id": session_id, "summary": "No messages yet", "messages": 0}

    from config import RECENT_MESSAGE_COUNT

    if total_messages <= RECENT_MESSAGE_COUNT:
        return {
            "session_id": session_id,
            "summary": "Conversation too short for summary",
            "messages": total_messages,
        }

    old_count = total_messages - RECENT_MESSAGE_COUNT
    summary = get_cached_summary(session_id, old_count)

    if not summary:
        summary = generate_summary_incremental(session_id, old_count, model=model)
        cache_summary(session_id, old_count, summary)

    return {
        "session_id": session_id,
        "summary": summary,
        "messages": total_messages,
        "summary_covers": old_count,
    }


@app.get("/api/context/{session_id}")
def get_context_endpoint(
    session_id: str,
    model: str = MODEL_CONFIG["name"],
    query: str = "",
    policy: str = "",
):
    return context_preview(
        session_id,
        model=model,
        query=query.strip() or None,
        policy=policy.strip() or None,
    )


@app.get("/api/messages/{session_id}")
def get_messages_endpoint(session_id: str, limit: int = 100):
    safe_limit = max(1, min(limit, 500))
    return {"session_id": session_id, "messages": get_messages(session_id, safe_limit)}


@app.get("/api/memory/{session_id}")
def get_memory_endpoint(session_id: str, limit: int = 100):
    return get_memory_hierarchy(session_id, limit=limit)


@app.get("/api/stats/{session_id}")
def get_stats(session_id: str):
    stats = get_session_stats(session_id)
    chat_tokens = stats["chat_input_tokens"] + stats["chat_output_tokens"]
    background_tokens = stats["background_input_tokens"] + stats["background_output_tokens"]

    return {
        "session_id": session_id,
        **stats,
        "chat_tokens": chat_tokens,
        "background_tokens": background_tokens,
        "costs": cost_response(stats),
    }


class StoryIn(BaseModel):
    story: str


@app.post("/api/set-story/{session_id}")
def set_story(session_id: str, story: str = Body(..., media_type="text/plain")):
    save_story_context(session_id, story)
    return {"session_id": session_id, "story_tokens": estimate_tokens(story)}


@app.delete("/api/session/{session_id}")
def delete_session_endpoint(session_id: str):
    deleted = delete_session(session_id)
    return {"session_id": session_id, "deleted_messages": deleted}


@app.post("/api/chat/stream")
async def chat_stream(body: PromptIn):
    enforce_rate_limit()

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if body.model not in MODEL_REGISTRY and not body.model.startswith("mock/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{body.model}'. Allowed: {sorted(MODEL_REGISTRY.keys())}",
        )

    logger.info(
        "streaming request",
        extra={"session_id": body.session_id, "model": body.model, "operation": "chat_stream"},
    )

    context = build_context(body.session_id, model=body.model, query=prompt)
    context.append({"role": "user", "content": prompt})

    async def generate():
        full_response = ""

        try:
            for chunk in call_llm_stream(context, max_tokens=body.max_tokens, model=body.model):
                full_response += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"

            input_tokens = estimate_messages_tokens(context, body.model)
            output_tokens = estimate_tokens(full_response, body.model)
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cached_input_tokens": 0,
                "cache_write_tokens": 0,
                "estimated": True,
            }

            user_id = store_message_with_usage(body.session_id, "user", prompt)
            if user_id:
                index_message(body.session_id, int(user_id), "user", prompt)

            assistant_id = store_message_with_usage(
                body.session_id,
                "assistant",
                full_response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if assistant_id:
                index_message(body.session_id, int(assistant_id), "assistant", full_response)
            record_llm_usage(
                body.session_id,
                "chat",
                body.model,
                input_tokens,
                output_tokens,
                estimated=True,
            )

            yield f"data: {json.dumps({'done': True, 'usage': usage})}\n\n"
            logger.info(
                "stream complete",
                extra={
                    "session_id": body.session_id,
                    "model": body.model,
                    "operation": "chat_stream",
                },
            )

        except Exception as exc:
            logger.exception(
                "streaming error",
                extra={
                    "session_id": body.session_id,
                    "model": body.model,
                    "operation": "chat_stream",
                },
            )
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/sessions")
def get_sessions():
    sessions = get_all_sessions()
    return {"sessions": sessions, "total": len(sessions)}


@app.get("/api/usage_timeseries/{session_id}")
def usage_timeseries(session_id: str, operation: str = ""):
    op = operation.strip() or None
    return {"session_id": session_id, "operation": op, "points": get_usage_timeseries(session_id, op)}


@app.get("/api/benchmark")
def benchmark_api(
    model: str = MODEL_CONFIG["name"],
    turns: int = 100,
    words_per_message: int = 90,
    recent_messages: int = 15,
    output_tokens: int = 350,
    summary_ratio: float = 0.18,
):
    class Args:
        pass

    args = Args()
    args.model = model
    args.turns = turns
    args.words_per_message = words_per_message
    args.recent_messages = recent_messages
    args.output_tokens = output_tokens
    args.summary_ratio = summary_ratio
    results = bench_mod.run_benchmark(args)
    # convenience field for the dashboard charts
    for row in results.get("results", []):
        row["background_cost_usd"] = bench_mod.cost_for_tokens(
            model,
            int(row.get("background_input_tokens", 0) or 0),
            int(row.get("background_output_tokens", 0) or 0),
        )
    results["retrieval"] = True
    return results


if __name__ == "__main__":
    logger.info("starting server", extra={"operation": "startup"})
    uvicorn.run(app, host="0.0.0.0", port=port)
