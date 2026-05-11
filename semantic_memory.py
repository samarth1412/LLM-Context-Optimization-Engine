import json
import math
import hashlib
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from database import _connect  # local module private helper is fine within repo


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def _stable_hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _hash_embedding(text: str, dim: int = 256) -> List[float]:
    """Deterministic, dependency-free embedding (hashing trick).

    This is not SOTA semantics, but it enables a real vector-retrieval pipeline
    without external services and keeps CI fast/offline.
    """
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        h = _stable_hash(token)
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


def _bm25_scores(query: str, documents: List[str]) -> List[float]:
    query_terms = _tokenize(query)
    if not query_terms or not documents:
        return [0.0 for _ in documents]

    doc_tokens = [_tokenize(doc) for doc in documents]
    doc_count = len(doc_tokens)
    avg_len = sum(len(tokens) for tokens in doc_tokens) / max(1, doc_count)
    dfs = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            dfs[term] += 1

    k1 = 1.5
    b = 0.75
    scores = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1 + ((doc_count - dfs[term] + 0.5) / (dfs[term] + 0.5)))
            numerator = tf[term] * (k1 + 1)
            denominator = tf[term] + k1 * (1 - b + b * (doc_len / max(avg_len, 1)))
            score += idf * (numerator / denominator)
        scores.append(score)
    return scores


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return [0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def retrieve(
    session_id: str,
    query: str,
    *,
    top_k: int = 6,
    min_message_id: Optional[int] = None,
    max_message_id: Optional[int] = None,
) -> List[Dict]:
    """Return top-k relevant past snippets for the query.

    The offline default is a hybrid of stable hashing vectors and BM25 lexical
    matching. That keeps CI deterministic while making exact fact recall much
    stronger than a pure hashing-trick embedding.
    """
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

    prepared = []
    for message_id, role, content, vector_json in rows:
        try:
            vec = json.loads(vector_json)
        except Exception:
            continue
        vector_score = float(_cosine(q, vec))
        prepared.append((vector_score, int(message_id), str(role), str(content)))

    lexical_scores = _bm25_scores(query, [item[3] for item in prepared])
    lexical_scores = _normalize_scores(lexical_scores)

    scored: List[Tuple[float, float, float, int, str, str]] = []
    for index, (vector_score, message_id, role, content) in enumerate(prepared):
        lexical_score = lexical_scores[index] if index < len(lexical_scores) else 0.0
        score = (0.45 * max(vector_score, 0.0)) + (0.55 * lexical_score)
        scored.append((float(score), float(vector_score), float(lexical_score), message_id, role, content))

    scored.sort(reverse=True, key=lambda x: x[0])
    out = []
    for score, vector_score, lexical_score, message_id, role, content in scored[:top_k]:
        out.append(
            {
                "message_id": message_id,
                "role": role,
                "content": content,
                "score": round(score, 4),
                "vector_score": round(vector_score, 4),
                "lexical_score": round(lexical_score, 4),
            }
        )
    return out
