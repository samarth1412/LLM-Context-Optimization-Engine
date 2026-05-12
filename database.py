import math
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import DB_NAME, DEFAULT_MODEL, get_model_config


def _connect():
    conn = sqlite3.connect(DB_NAME, timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database():
    """Initialize SQLite database and additive tables used by newer versions."""
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            session_id TEXT NOT NULL,
            messages_covered INTEGER NOT NULL,
            summary_text TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, messages_covered)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS story_context (
            session_id TEXT PRIMARY KEY,
            story_text TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            input_cost_usd REAL DEFAULT 0,
            output_cost_usd REAL DEFAULT 0,
            total_cost_usd REAL DEFAULT 0,
            estimated INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    _ensure_column(cursor, "llm_usage", "cached_input_tokens", "INTEGER DEFAULT 0")
    _ensure_column(cursor, "llm_usage", "cache_write_tokens", "INTEGER DEFAULT 0")
    _ensure_column(cursor, "llm_usage", "latency_ms", "INTEGER")

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_summary_session ON summaries(session_id, messages_covered DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_session ON llm_usage(session_id, operation)"
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            embedding_model TEXT DEFAULT 'mock/hash',
            vector_dim INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
        )
        """
    )
    _ensure_column(cursor, "memory_vectors", "embedding_model", "TEXT DEFAULT 'mock/hash'")
    _ensure_column(cursor, "memory_vectors", "vector_dim", "INTEGER DEFAULT 0")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_vectors(session_id, message_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory_vectors(session_id, embedding_model)"
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_metadata (
            session_id TEXT NOT NULL,
            message_id INTEGER NOT NULL,
            importance_score REAL NOT NULL,
            memory_layer TEXT NOT NULL,
            memory_action TEXT NOT NULL,
            retrieval_count INTEGER DEFAULT 0,
            last_retrieved_at DATETIME,
            signals_json TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, message_id),
            FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_metadata_session ON memory_metadata(session_id, memory_layer, memory_action)"
    )

    conn.commit()
    conn.close()


def _ensure_column(cursor, table: str, column: str, column_type: str) -> None:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = {row[1] for row in cursor.fetchall()}
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate tokens with tiktoken when installed, otherwise use a safe fallback."""
    if not text:
        return 0

    try:
        import tiktoken  # type: ignore

        try:
            encoder = tiktoken.encoding_for_model(model or DEFAULT_MODEL)
        except Exception:
            encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        return max(1, int(math.ceil(len(text) / 4)))


def estimate_messages_tokens(messages: List[Dict], model: Optional[str] = None) -> int:
    # Chat APIs add small role/format overhead; this estimate is used only when
    # provider usage is unavailable.
    return sum(estimate_tokens(msg.get("content", ""), model) + 4 for msg in messages)


def calculate_cost(
    model: Optional[str],
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> Dict:
    config = get_model_config(model)
    cached_input_tokens = max(0, min(int(cached_input_tokens or 0), int(input_tokens or 0)))
    uncached_input_tokens = max(0, int(input_tokens or 0) - cached_input_tokens)
    input_cost = (uncached_input_tokens / 1_000_000) * float(config["input_cost_per_1m"])
    input_cost += (cached_input_tokens / 1_000_000) * float(
        config.get("cached_input_cost_per_1m", config["input_cost_per_1m"])
    )
    output_cost = (output_tokens / 1_000_000) * float(config["output_cost_per_1m"])
    return {
        "input": input_cost,
        "output": output_cost,
        "total": input_cost + output_cost,
    }


def record_llm_usage(
    session_id: str,
    operation: str,
    model: Optional[str],
    input_tokens: int,
    output_tokens: int,
    estimated: bool = False,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    latency_ms: Optional[int] = None,
):
    """Record every paid or estimated LLM operation, including summaries."""
    selected_model = model or DEFAULT_MODEL
    costs = calculate_cost(selected_model, input_tokens, output_tokens, cached_input_tokens)

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO llm_usage (
            session_id, operation, model, input_tokens, output_tokens, total_tokens,
            input_cost_usd, output_cost_usd, total_cost_usd, estimated,
            cached_input_tokens, cache_write_tokens, latency_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            operation,
            selected_model,
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
            costs["input"],
            costs["output"],
            costs["total"],
            1 if estimated else 0,
            int(cached_input_tokens or 0),
            int(cache_write_tokens or 0),
            latency_ms,
        ),
    )
    conn.commit()
    conn.close()


def store_message_with_usage(
    session_id: str,
    role: str,
    content: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
):
    """Store a message. Token columns are retained for backwards compatibility."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO messages (session_id, role, content, input_tokens, output_tokens)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, role, content, input_tokens, output_tokens),
    )
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return message_id


def count_messages(session_id: str) -> int:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_all_messages(session_id: str) -> List[Dict]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, content FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,),
    )
    messages = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()
    return messages


def get_messages(session_id: str, limit: int = 100) -> List[Dict]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, role, content, timestamp FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "timestamp": row[3],
        }
        for row in reversed(rows)
    ]


def get_last_n_messages(session_id: str, n: int) -> List[Dict]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, content FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, n),
    )
    messages = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()
    return list(reversed(messages))


def get_messages_range(session_id: str, start: int, end: int) -> List[Dict]:
    """Get messages in a 1-indexed inclusive range ordered by insertion id."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, content FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        LIMIT ? OFFSET ?
        """,
        (session_id, end - start + 1, start - 1),
    )
    messages = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()
    return messages


def get_cached_summary(session_id: str, messages_covered: int) -> Optional[str]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT summary_text FROM summaries
        WHERE session_id = ? AND messages_covered = ?
        """,
        (session_id, messages_covered),
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def get_latest_cached_summary(session_id: str) -> Optional[tuple]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT messages_covered, summary_text FROM summaries
        WHERE session_id = ?
        ORDER BY messages_covered DESC
        LIMIT 1
        """,
        (session_id,),
    )
    result = cursor.fetchone()
    conn.close()
    return result if result else None


def cache_summary(session_id: str, messages_covered: int, summary: str):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO summaries (session_id, messages_covered, summary_text)
        VALUES (?, ?, ?)
        """,
        (session_id, messages_covered, summary),
    )
    conn.commit()
    conn.close()


def _empty_usage_stats() -> Dict:
    return {
        "chat_input_tokens": 0,
        "chat_output_tokens": 0,
        "background_input_tokens": 0,
        "background_output_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "estimated_usage_events": 0,
        "input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "chat_cost_usd": 0.0,
        "background_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "avg_latency_ms": 0,
    }


def get_session_stats(session_id: str) -> Dict:
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
    total_messages = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM summaries WHERE session_id = ?", (session_id,))
    summary_count = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT operation, SUM(input_tokens), SUM(output_tokens),
               SUM(input_cost_usd), SUM(output_cost_usd), SUM(total_cost_usd),
               SUM(estimated), COUNT(*), SUM(cached_input_tokens), SUM(cache_write_tokens),
               AVG(latency_ms)
        FROM llm_usage
        WHERE session_id = ?
        GROUP BY operation
        """,
        (session_id,),
    )
    usage_rows = cursor.fetchall()

    stats = _empty_usage_stats()
    if usage_rows:
        operation_counts = {}
        for (
            operation,
            input_tokens,
            output_tokens,
            input_cost,
            output_cost,
            total_cost,
            estimated,
            count,
            cached_input_tokens,
            cache_write_tokens,
            avg_latency_ms,
        ) in usage_rows:
            input_tokens = input_tokens or 0
            output_tokens = output_tokens or 0
            input_cost = input_cost or 0.0
            output_cost = output_cost or 0.0
            total_cost = total_cost or 0.0
            estimated = estimated or 0
            cached_input_tokens = cached_input_tokens or 0
            cache_write_tokens = cache_write_tokens or 0

            operation_counts[operation] = count
            if operation == "chat":
                stats["chat_input_tokens"] += input_tokens
                stats["chat_output_tokens"] += output_tokens
                stats["chat_cost_usd"] += total_cost
            else:
                stats["background_input_tokens"] += input_tokens
                stats["background_output_tokens"] += output_tokens
                stats["background_cost_usd"] += total_cost

            stats["input_tokens"] += input_tokens
            stats["output_tokens"] += output_tokens
            stats["input_cost_usd"] += input_cost
            stats["output_cost_usd"] += output_cost
            stats["total_cost_usd"] += total_cost
            stats["estimated_usage_events"] += estimated
            stats["cached_input_tokens"] += cached_input_tokens
            stats["cache_write_tokens"] += cache_write_tokens
        stats["operation_counts"] = operation_counts
        stats["avg_latency_ms"] = round(
            sum((row[10] or 0) for row in usage_rows) / max(1, len(usage_rows)),
            2,
        )
    else:
        cursor.execute(
            """
            SELECT SUM(input_tokens), SUM(output_tokens)
            FROM messages
            WHERE session_id = ?
            """,
            (session_id,),
        )
        result = cursor.fetchone()
        input_tokens = result[0] or 0
        output_tokens = result[1] or 0
        costs = calculate_cost(DEFAULT_MODEL, input_tokens, output_tokens)
        stats.update(
            {
                "chat_input_tokens": input_tokens,
                "chat_output_tokens": output_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost_usd": costs["input"],
                "output_cost_usd": costs["output"],
                "chat_cost_usd": costs["total"],
                "total_cost_usd": costs["total"],
                "operation_counts": {},
            }
        )

    stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]
    conn.close()

    return {
        "total_messages": total_messages,
        "cached_summaries": summary_count,
        **stats,
    }


def delete_session(session_id: str) -> int:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memory_metadata WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    messages_deleted = cursor.rowcount
    cursor.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM story_context WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM llm_usage WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM memory_vectors WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return messages_deleted


def upsert_memory_metadata(
    session_id: str,
    message_id: int,
    importance_score: float,
    memory_layer: str,
    memory_action: str,
    signals: Dict,
) -> None:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO memory_metadata (
            session_id, message_id, importance_score, memory_layer, memory_action, signals_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, message_id) DO UPDATE SET
            importance_score = excluded.importance_score,
            memory_layer = excluded.memory_layer,
            memory_action = excluded.memory_action,
            signals_json = excluded.signals_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            session_id,
            int(message_id),
            float(importance_score),
            memory_layer,
            memory_action,
            json.dumps(signals, sort_keys=True),
        ),
    )
    conn.commit()
    conn.close()


def increment_memory_retrievals(session_id: str, message_ids: List[int]) -> None:
    if not message_ids:
        return
    from memory_importance import score_memory

    conn = _connect()
    cursor = conn.cursor()
    cursor.executemany(
        """
        UPDATE memory_metadata
        SET retrieval_count = retrieval_count + 1,
            last_retrieved_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE session_id = ? AND message_id = ?
        """,
        [(session_id, int(message_id)) for message_id in message_ids],
    )
    cursor.execute(
        "SELECT MAX(id) FROM messages WHERE session_id = ?",
        (session_id,),
    )
    latest_message_id = cursor.fetchone()[0] or 0
    placeholders = ",".join("?" for _ in message_ids)
    cursor.execute(
        f"""
        SELECT msg.id, msg.role, msg.content, COALESCE(m.retrieval_count, 0)
        FROM messages msg
        LEFT JOIN memory_metadata m
          ON m.session_id = msg.session_id AND m.message_id = msg.id
        WHERE msg.session_id = ? AND msg.id IN ({placeholders})
        """,
        (session_id, *[int(message_id) for message_id in message_ids]),
    )
    for message_id, role, content, retrieval_count in cursor.fetchall():
        importance = score_memory(
            str(content),
            role=str(role),
            message_id=int(message_id),
            latest_message_id=int(latest_message_id),
            retrieval_count=int(retrieval_count or 0),
        )
        cursor.execute(
            """
            UPDATE memory_metadata
            SET importance_score = ?,
                memory_layer = ?,
                memory_action = ?,
                signals_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ? AND message_id = ?
            """,
            (
                float(importance["importance_score"]),
                importance["memory_layer"],
                importance["memory_action"],
                json.dumps(importance["signals"], sort_keys=True),
                session_id,
                int(message_id),
            ),
        )
    conn.commit()
    conn.close()


def get_memory_metadata_map(session_id: str, message_ids: List[int]) -> Dict[int, Dict]:
    if not message_ids:
        return {}
    placeholders = ",".join("?" for _ in message_ids)
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT message_id, importance_score, memory_layer, memory_action,
               retrieval_count, signals_json
        FROM memory_metadata
        WHERE session_id = ? AND message_id IN ({placeholders})
        """,
        (session_id, *[int(message_id) for message_id in message_ids]),
    )
    rows = cursor.fetchall()
    conn.close()
    out = {}
    for message_id, score, layer, action, retrieval_count, signals_json in rows:
        try:
            signals = json.loads(signals_json or "{}")
        except Exception:
            signals = {}
        out[int(message_id)] = {
            "importance_score": round(float(score or 0), 4),
            "memory_layer": layer,
            "memory_action": action,
            "retrieval_count": int(retrieval_count or 0),
            "signals": signals,
        }
    return out


def refresh_memory_metadata(session_id: str) -> None:
    from memory_importance import score_memory

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT MAX(id) FROM messages WHERE session_id = ?",
        (session_id,),
    )
    latest_message_id = cursor.fetchone()[0] or 0
    cursor.execute(
        """
        SELECT DISTINCT msg.id, msg.role, msg.content, COALESCE(m.retrieval_count, 0)
        FROM memory_vectors v
        JOIN messages msg
          ON msg.session_id = v.session_id AND msg.id = v.message_id
        LEFT JOIN memory_metadata m
          ON m.session_id = msg.session_id AND m.message_id = msg.id
        WHERE v.session_id = ?
        """,
        (session_id,),
    )
    for message_id, role, content, retrieval_count in cursor.fetchall():
        importance = score_memory(
            str(content),
            role=str(role),
            message_id=int(message_id),
            latest_message_id=int(latest_message_id),
            retrieval_count=int(retrieval_count or 0),
        )
        cursor.execute(
            """
            INSERT INTO memory_metadata (
                session_id, message_id, importance_score, memory_layer,
                memory_action, retrieval_count, signals_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, message_id) DO UPDATE SET
                importance_score = excluded.importance_score,
                memory_layer = excluded.memory_layer,
                memory_action = excluded.memory_action,
                signals_json = excluded.signals_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                session_id,
                int(message_id),
                float(importance["importance_score"]),
                importance["memory_layer"],
                importance["memory_action"],
                int(retrieval_count or 0),
                json.dumps(importance["signals"], sort_keys=True),
            ),
        )
    conn.commit()
    conn.close()


def get_memory_hierarchy(session_id: str, limit: int = 100) -> Dict:
    refresh_memory_metadata(session_id)
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT memory_layer, memory_action, COUNT(*), AVG(importance_score)
        FROM memory_metadata
        WHERE session_id = ?
        GROUP BY memory_layer, memory_action
        """,
        (session_id,),
    )
    groups = cursor.fetchall()
    cursor.execute(
        """
        SELECT m.message_id, msg.role, msg.content, m.importance_score, m.memory_layer,
               m.memory_action, m.retrieval_count, m.signals_json
        FROM memory_metadata m
        JOIN messages msg ON msg.session_id = m.session_id AND msg.id = m.message_id
        WHERE m.session_id = ?
        ORDER BY m.importance_score DESC, m.retrieval_count DESC
        LIMIT ?
        """,
        (session_id, max(1, min(int(limit), 500))),
    )
    rows = cursor.fetchall()
    conn.close()

    layers: Dict[str, Dict] = {}
    for layer, action, count, avg_score in groups:
        layer_bucket = layers.setdefault(
            layer,
            {"count": 0, "mean_importance": 0.0, "actions": {}},
        )
        layer_bucket["count"] += int(count or 0)
        layer_bucket["actions"][action] = int(count or 0)
    for layer, bucket in layers.items():
        layer_rows = [row for row in groups if row[0] == layer]
        total = sum(int(row[2] or 0) for row in layer_rows)
        if total:
            bucket["mean_importance"] = round(
                sum(float(row[3] or 0) * int(row[2] or 0) for row in layer_rows) / total,
                4,
            )

    memories = []
    for message_id, role, content, score, layer, action, retrieval_count, signals_json in rows:
        try:
            signals = json.loads(signals_json or "{}")
        except Exception:
            signals = {}
        memories.append(
            {
                "message_id": int(message_id),
                "role": role,
                "preview": str(content)[:240],
                "importance_score": round(float(score or 0), 4),
                "memory_layer": layer,
                "memory_action": action,
                "retrieval_count": int(retrieval_count or 0),
                "signals": signals,
            }
        )
    return {"session_id": session_id, "layers": layers, "top_memories": memories}


def get_usage_timeseries(session_id: str, operation: Optional[str] = None) -> List[Dict]:
    """Return per-day usage counts for charting."""
    conn = _connect()
    cursor = conn.cursor()
    if operation:
        cursor.execute(
            """
            SELECT substr(created_at, 1, 10) as day, COUNT(*)
            FROM llm_usage
            WHERE session_id = ? AND operation = ?
            GROUP BY day
            ORDER BY day ASC
            """,
            (session_id, operation),
        )
    else:
        cursor.execute(
            """
            SELECT substr(created_at, 1, 10) as day, COUNT(*)
            FROM llm_usage
            WHERE session_id = ?
            GROUP BY day
            ORDER BY day ASC
            """,
            (session_id,),
        )
    rows = cursor.fetchall()
    conn.close()
    return [{"day": row[0], "count": int(row[1] or 0)} for row in rows]


def save_story_context(session_id: str, story_text: str):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO story_context (session_id, story_text, created_at)
        VALUES (?, ?, ?)
        """,
        (session_id, story_text, datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_story_context(session_id: str) -> Optional[str]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT story_text FROM story_context WHERE session_id = ?", (session_id,)
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def get_all_sessions() -> List[Dict]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            session_id,
            COUNT(*) as message_count,
            MAX(timestamp) as last_activity
        FROM messages
        GROUP BY session_id
        ORDER BY last_activity DESC
        """
    )

    sessions = []
    for row in cursor.fetchall():
        sessions.append(
            {
                "session_id": row[0],
                "message_count": row[1],
                "last_activity": row[2],
            }
        )

    conn.close()
    return sessions
