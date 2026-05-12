# Retrieval

LLM-Context-Optimization-Engine evaluates retrieval before prompt assembly so retrieval quality can be separated from LLM answer quality.

## Retrieval Modes

- `bm25`: lexical baseline.
- `embedding`: vector-only semantic retrieval.
- `hybrid`: weighted combination of lexical and vector scores.

## Embedding Backends

- `mock/hash`: deterministic CI baseline, useful for reproducible local tests.
- `local/bge-small-en-v1.5`: local sentence-transformers backend.
- `local/e5-base-v2`: local sentence-transformers backend.
- `openai/text-embedding-3-small`: OpenAI embeddings API backend.

Install optional local embedding dependencies:

```powershell
pip install -r requirements-embeddings.txt
```

Run the full retrieval evaluation:

```powershell
python eval_retrieval_quality.py --embedding-models mock/hash,local/bge-small-en-v1.5,local/e5-base-v2,openai/text-embedding-3-small --json --export
```

Run only OpenAI embeddings:

```powershell
python eval_retrieval_quality.py --embedding-models openai/text-embedding-3-small --json --export
```

## Metrics

- **Recall@K**: whether required evidence appears in the top K retrieved items.
- **Precision@K**: proportion of retrieved items that are relevant.
- **MRR**: mean reciprocal rank of the first relevant result.
- **Distractor hit rate**: how often irrelevant distractor evidence is retrieved.
- **Stale evidence rate**: how often outdated evidence is retrieved.
- **Top-hit accuracy**: whether the top retrieved item is relevant.

## Current Results

| Retrieval mode | Embedding model | Recall@6 | Precision@6 | MRR | Distractor hit rate | Stale evidence rate | Top-hit accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| `bm25` | `mock/hash` | 0.8889 | 0.1667 | 0.8333 | 0.1333 | 0.0500 | 0.7778 |
| `embedding` | `mock/hash` | 0.8889 | 0.1667 | 0.8333 | 0.1333 | 0.0500 | 0.7778 |
| `hybrid` | `mock/hash` | 0.8889 | 0.1667 | 0.8333 | 0.1333 | 0.0500 | 0.7778 |
| `bm25` | `local/bge-small-en-v1.5` | 0.8889 | 0.1667 | 0.8333 | 0.1333 | 0.0500 | 0.7778 |
| `embedding` | `local/bge-small-en-v1.5` | 1.0000 | 0.1852 | 1.0000 | 0.1333 | 0.0500 | 1.0000 |
| `hybrid` | `local/bge-small-en-v1.5` | 1.0000 | 0.1852 | 0.8889 | 0.1333 | 0.0500 | 0.7778 |
| `bm25` | `local/e5-base-v2` | 0.8889 | 0.1667 | 0.8333 | 0.1333 | 0.0500 | 0.7778 |
| `embedding` | `local/e5-base-v2` | 1.0000 | 0.1852 | 0.8889 | 0.1333 | 0.0500 | 0.7778 |
| `hybrid` | `local/e5-base-v2` | 1.0000 | 0.1852 | 0.8889 | 0.1333 | 0.0500 | 0.7778 |
| `bm25` | `openai/text-embedding-3-small` | 0.8889 | 0.1667 | 0.8333 | 0.1333 | 0.0500 | 0.7778 |
| `embedding` | `openai/text-embedding-3-small` | 1.0000 | 0.1852 | 1.0000 | 0.1333 | 0.0500 | 1.0000 |
| `hybrid` | `openai/text-embedding-3-small` | 1.0000 | 0.1852 | 0.8889 | 0.1333 | 0.0500 | 0.7778 |

## Interpretation

- The deterministic hash backend is a mock baseline, not the serious semantic retrieval mode.
- BGE and OpenAI embedding-only are strongest on the current suite: both recover all relevant items and put the relevant result first.
- E5 improves Recall@6 over BM25/hash, but top-hit accuracy still shows distractor and stale-evidence pressure.
- The current hybrid weighting does not beat BGE embedding-only. That is useful evidence: hybrid retrieval needs tuning, not automatic adoption.

Artifacts:

- `results/retrieval_quality.json`
- `results/retrieval_quality.csv`
- `results/retrieval_quality_summary.csv`
- `results/retrieval_confusion_heatmap.png`
