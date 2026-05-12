import hashlib
import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import requests

from config import (
    EMBEDDING_MODEL,
    OPENAI_EMBEDDINGS_URL,
    RETRIEVAL_MODE,
    get_openai_key,
)
from database import _connect  # local module private helper is fine within repo


TOKEN_RE = re.compile(r"[a-z0-9_]+")
VALID_RETRIEVAL_MODES = {"bm25", "embedding", "hybrid"}
LOCAL_EMBEDDING_MODELS = {
    "local/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "local/e5-base-v2": "intfloat/e5-base-v2",
}
EMBEDDING_MODEL_ALIASES = {
    "mock": "mock/hash",
    "hash": "mock/hash",
    "mock/hash": "mock/hash",
    "bge-small-en-v1.5": "local/bge-small-en-v1.5",
    "e5-base-v2": "local/e5-base-v2",
    "text-embedding-3-small": "openai/text-embedding-3-small",
}
_LOCAL_MODEL_CACHE = {}


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def _stable_hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [float(v) / norm for v in vec]


def _hash_embedding(text: str, dim: int = 256) -> List[float]:
    """Deterministic, dependency-free embedding used for tests and CI."""
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        h = _stable_hash(token)
        idx = h % dim
        sign = -1.0 if (h & 1) else 1.0
        vec[idx] += sign
    return _l2_normalize(vec)


def normalize_embedding_model(model: Optional[str] = None) -> str:
    selected = (model or EMBEDDING_MODEL or "mock/hash").strip()
    return EMBEDDING_MODEL_ALIASES.get(selected, selected)


def normalize_retrieval_mode(mode: Optional[str] = None) -> str:
    selected = (mode or RETRIEVAL_MODE or "hybrid").strip().lower()
    if selected not in VALID_RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode '{selected}'. Use one of {sorted(VALID_RETRIEVAL_MODES)}.")
    return selected


def _format_for_embedding_model(text: str, model: str, purpose: str) -> str:
    if model == "local/e5-base-v2":
        prefix = "query: " if purpose == "query" else "passage: "
        return prefix + text
    if model == "local/bge-small-en-v1.5" and purpose == "query":
        return "Represent this sentence for searching relevant passages: " + text
    return text


def _local_embedding(text: str, model: str, purpose: str) -> List[float]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Local embedding models require sentence-transformers. "
            "Install optional dependencies with: pip install -r requirements-embeddings.txt"
        ) from exc

    model_name = LOCAL_EMBEDDING_MODELS.get(model)
    if not model_name:
        raise ValueError(f"Unsupported local embedding model: {model}")

    if model not in _LOCAL_MODEL_CACHE:
        _LOCAL_MODEL_CACHE[model] = SentenceTransformer(model_name)

    formatted = _format_for_embedding_model(text, model, purpose)
    encoded = _LOCAL_MODEL_CACHE[model].encode(
        [formatted],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return [float(value) for value in encoded]


def _openai_embedding(text: str, model: str) -> List[float]:
    api_key = get_openai_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for openai/* embedding models.")

    provider_model = model.split("/", 1)[1]
    response = requests.post(
        OPENAI_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": provider_model, "input": text},
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return _l2_normalize([float(value) for value in data["data"][0]["embedding"]])


def embed_text(text: str, model: Optional[str] = None, purpose: str = "document") -> List[float]:
    selected_model = normalize_embedding_model(model)
    if selected_model.startswith("mock/"):
        return _hash_embedding(text)
    if selected_model.startswith("local/"):
        return _local_embedding(text, selected_model, purpose)
    if selected_model.startswith("openai/"):
        formatted = _format_for_embedding_model(text, selected_model, purpose)
        return _openai_embedding(formatted, selected_model)
    raise ValueError(f"Unsupported embedding model: {selected_model}")


def index_message(
    session_id: str,
    message_id: int,
    role: str,
    content: str,
    *,
    embedding_model: Optional[str] = None,
) -> None:
    selected_model = normalize_embedding_model(embedding_model)
    vec = embed_text(content, selected_model, purpose="document")
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO memory_vectors (
            session_id, message_id, role, content, vector_json, embedding_model, vector_dim
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, message_id, role, content, json.dumps(vec), selected_model, len(vec)),
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
    mode: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> List[Dict]:
    """Return top-k relevant past snippets for the query.

    Supported modes:
    - bm25: lexical baseline
    - embedding: vector-only retrieval
    - hybrid: weighted vector + BM25 retrieval
    """
    selected_mode = normalize_retrieval_mode(mode)
    selected_model = normalize_embedding_model(embedding_model)
    query_vector = (
        embed_text(query, selected_model, purpose="query")
        if selected_mode in {"embedding", "hybrid"}
        else []
    )

    conn = _connect()
    cursor = conn.cursor()

    where = ["session_id = ?", "embedding_model = ?"]
    params: List = [session_id, selected_model]
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
        vector_score = 0.0
        if selected_mode in {"embedding", "hybrid"}:
            try:
                vec = json.loads(vector_json)
            except Exception:
                continue
            vector_score = float(_cosine(query_vector, vec))
        prepared.append((vector_score, int(message_id), str(role), str(content)))

    lexical_scores = (
        _normalize_scores(_bm25_scores(query, [item[3] for item in prepared]))
        if selected_mode in {"bm25", "hybrid"}
        else [0.0 for _ in prepared]
    )

    scored: List[Tuple[float, float, float, int, str, str]] = []
    for index, (vector_score, message_id, role, content) in enumerate(prepared):
        lexical_score = lexical_scores[index] if index < len(lexical_scores) else 0.0
        if selected_mode == "bm25":
            score = lexical_score
        elif selected_mode == "embedding":
            score = max(vector_score, 0.0)
        else:
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
                "retrieval_mode": selected_mode,
                "embedding_model": selected_model,
            }
        )
    return out
