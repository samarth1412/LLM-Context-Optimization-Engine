import json
import math
from typing import Dict, List, Optional, Tuple

from database import _connect  # local module private helper is fine within repo


def _hash_embedding(text: str, dim: int = 256) -> List[float]:
    """Deterministic, dependency-free embedding (hashing trick).

    This is not SOTA semantics, but it enables a real vector-retrieval pipeline
    without external services and keeps CI fast/offline.
    """
    vec = [0.0] * dim
    if not text:
        return vec
    for token in text.lower().split():
        h = hash(token)
        idx = h % dim
        sign = -1.0 if (h & 1) else 1.0
        vec[idx] += sign
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def index_message(session_id: str, message_id: int, role: str, content: str) -> None:
    vec = _hash_embedding(content)
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO memory_vectors (session_id, message_id, role, content, vector_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, message_id, role, content, json.dumps(vec)),
    )
    conn.commit()
    conn.close()


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def retrieve(
    session_id: str,
    query: str,
    *,
    top_k: int = 6,
    min_message_id: Optional[int] = None,
    max_message_id: Optional[int] = None,
) -> List[Dict]:
    """Return top-k relevant past snippets for the query."""
    q = _hash_embedding(query)
    conn = _connect()
    cursor = conn.cursor()

    where = ["session_id = ?"]
    params: List = [session_id]
    if min_message_id is not None:
        where.append("message_id >= ?")
        params.append(min_message_id)
    if max_message_id is not None:
        where.append("message_id <= ?")
        params.append(max_message_id)

    cursor.execute(
        f"""
        SELECT message_id, role, content, vector_json
        FROM memory_vectors
        WHERE {' AND '.join(where)}
        """,
        tuple(params),
    )
    rows = cursor.fetchall()
    conn.close()

    scored: List[Tuple[float, int, str, str]] = []
    for message_id, role, content, vector_json in rows:
        try:
            vec = json.loads(vector_json)
        except Exception:
            continue
        score = _cosine(q, vec)
        scored.append((float(score), int(message_id), str(role), str(content)))

    scored.sort(reverse=True, key=lambda x: x[0])
    out = []
    for score, message_id, role, content in scored[:top_k]:
        out.append(
            {
                "message_id": message_id,
                "role": role,
                "content": content,
                "score": round(score, 4),
            }
        )
    return out

