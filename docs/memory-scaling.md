# Memory Scaling

The memory-scaling harness evaluates memory selection under long synthetic sessions. It focuses on whether a policy can preserve important long-range facts while avoiding stale and noisy context.

## Run

```powershell
python eval_memory_scaling.py --turns 1000,5000,10000 --json --export
```

## Stressors

- Retrieval noise
- Conflicting updates
- Evolving preferences
- Entity drift
- Forgotten constraints
- Routine filler turns

## Policies

- `full_archive`: keeps every generated memory.
- `sliding_window`: keeps only the newest budgeted messages.
- `importance`: selects preserved/compressed memories by importance score plus recent working memory.

## Metrics

- **Critical recall**: fraction of current critical facts kept.
- **Stale retention**: fraction of outdated facts retained.
- **Noise retention**: fraction of routine/distractor messages retained.
- **Retrieval precision estimate**: selected current facts divided by selected evidence-like items.
- **Kept messages**: selected memory count under policy budget.

## Current Results

| Turns | Policy | Critical recall | Stale retention | Noise retention | Retrieval precision estimate | Kept messages |
|---:|---|---:|---:|---:|---:|---:|
| 1,000 | `full_archive` | 1.0000 | 1.0000 | 1.0000 | 0.0290 | 1,000 |
| 1,000 | `sliding_window` | 0.3103 | 0.0000 | 0.1192 | 0.0750 | 120 |
| 1,000 | `importance` | 0.7931 | 0.5250 | 0.0816 | 0.1917 | 120 |
| 5,000 | `full_archive` | 1.0000 | 1.0000 | 1.0000 | 0.0206 | 5,000 |
| 5,000 | `sliding_window` | 0.1650 | 0.0167 | 0.0599 | 0.0567 | 300 |
| 5,000 | `importance` | 0.8252 | 0.3458 | 0.0283 | 0.2833 | 300 |
| 10,000 | `full_archive` | 1.0000 | 1.0000 | 1.0000 | 0.0194 | 10,000 |
| 10,000 | `sliding_window` | 0.1082 | 0.0386 | 0.0601 | 0.0350 | 600 |
| 10,000 | `importance` | 0.9072 | 0.3659 | 0.0262 | 0.2933 | 600 |

## Interpretation

- Full archive preserves every critical fact, but also retains every stale and noisy item. Precision collapses as sessions grow.
- Sliding window filters stale and noisy evidence by recency, but forgets most long-range critical facts at 5k-10k turns.
- Importance scoring keeps the same budget as sliding window while preserving far more critical facts and retaining less noise.
- Stale retention is not solved. At 10k turns, the importance policy still retains `0.3659` of stale facts. That is a useful next research target: stronger contradiction handling and recency-aware stale suppression.

Artifacts:

- `results/memory_scaling.json`
- `results/memory_scaling.csv`
- `results/memory_scaling_recall.png`
- `results/memory_scaling_retention.png`
