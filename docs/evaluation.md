# Evaluation

LLM-Context-Optimization-Engine uses offline deterministic evaluations for repeatability and optional live-provider evaluations for sanity checks.

## Evaluation Commands

```powershell
python benchmark.py --json --export
python eval_memory_quality.py --json --export
python eval_memory_scaling.py --turns 1000,5000,10000 --json --export
python eval_retrieval_quality.py --embedding-models mock/hash,local/bge-small-en-v1.5,local/e5-base-v2,openai/text-embedding-3-small --json --export
python eval_model_answers.py --model openai/gpt-4o-mini --json --export
python eval_model_answers.py --model google/gemini-3.1-flash-lite --strategies full_history,sliding_window,summary,adaptive --json --export
```

## Synthetic Token/Cost Benchmark

Run configuration:

- Model: `x-ai/grok-4-fast`
- Conversation: 100 turns, 200 messages
- Message length: 90 words/message
- Metric focus: token growth and estimated cost

| Strategy | Requests | Input tokens | Background input | Background output | Total tokens | Est. cost (USD) | Savings vs full |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_history` | 100 | 1,178,225 | 0 | 0 | 1,213,225 | 0.253145 | 0.00% |
| `sliding_window` | 100 | 180,393 | 0 | 0 | 215,393 | 0.053579 | 78.83% |
| `incremental_summary` | 100 | 326,654 | 21,341 | 3,774 | 386,769 | 0.088986 | 64.85% |

Notes:

- Latency is not measured in this synthetic benchmark.
- `incremental_summary` includes background summary overhead in total token and cost accounting.

## Memory-Quality Evaluation

Run configuration:

- Model: `mock/echo`
- Cases: 8 stress cases
- Metric focus: evidence quality before the LLM call

Stress cases:

- Long-range preference recall
- Temporal update handling
- Multi-hop project state recall
- Abstention/no-evidence behavior
- Distractor retrieval noise
- Similar-entity disambiguation
- Recent override vs stale summary
- Summary drift with missing early constraint

| Strategy | Mean quality | Required recall | Conflict pressure | Mean context tokens | Token reduction vs full | Failures |
|---|---:|---:|---:|---:|---:|---:|
| `full_history` | 0.8250 | 1.0000 | 0.5000 | 1,964.9 | 0.00% | 4 |
| `sliding_window` | 0.2500 | 0.2500 | 0.0000 | 370.4 | 81.15% | 6 |
| `summary` | 0.7875 | 0.8750 | 0.2500 | 383.5 | 80.48% | 3 |
| `retrieval` | 0.8250 | 1.0000 | 0.5000 | 429.3 | 78.15% | 4 |
| `hybrid` | 0.8250 | 1.0000 | 0.5000 | 442.4 | 77.49% | 4 |
| `adaptive` | 0.8688 | 1.0000 | 0.3750 | 418.3 | 78.71% | 3 |

Interpretation:

- `adaptive` has the best mean context quality while using far fewer context tokens than full history.
- `full_history`, `retrieval`, and `hybrid` keep recall high but can carry stale or distracting evidence into the prompt.
- `summary` is efficient but can fail when compression omits an early constraint.
- `sliding_window` is cheap but fails long-range and multi-hop recall by design.

Artifacts:

- `results/memory_quality.json`
- `results/memory_quality.csv`
- `results/memory_quality_failures.csv`
- `results/memory_quality_pareto.png`

## Live Model-Answer Evaluation

The live-model harness calls a provider model and scores generated answers for recall, conflict, abstention, token usage, cost, and latency. These results are useful as provider checks, but the live set is intentionally small and currently easier than the offline memory-quality suite.

### OpenAI `gpt-4o-mini`

| Strategy | Mean answer quality | Required recall | Conflict score | Prompt tokens | Cost (USD) | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| `full_history` | 1.0000 | 1.0000 | 0.0000 | 7,478 | 0.00114150 | 1,345.5 ms |
| `sliding_window` | 0.2500 | 0.2500 | 0.0000 | 1,644 | 0.00027240 | 990.8 ms |
| `summary` | 1.0000 | 1.0000 | 0.0000 | 1,695 | 0.00027405 | 852.3 ms |
| `retrieval` | 1.0000 | 1.0000 | 0.0000 | 1,791 | 0.00028845 | 1,220.0 ms |
| `hybrid` | 1.0000 | 1.0000 | 0.0000 | 1,842 | 0.00029610 | 1,209.8 ms |
| `adaptive` | 1.0000 | 1.0000 | 0.0000 | 1,743 | 0.00028125 | 844.5 ms |

### Gemini `gemini-3.1-flash-lite`

| Strategy | Mean answer quality | Required recall | Conflict score | Prompt tokens | Cost (USD) | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| `full_history` | 1.0000 | 1.0000 | 0.0000 | 6,804 | 0.00174600 | 12,636.8 ms |
| `sliding_window` | 0.2500 | 0.2500 | 0.0000 | 1,504 | 0.00041800 | 11,836.3 ms |
| `summary` | 1.0000 | 1.0000 | 0.0000 | 1,564 | 0.00044350 | 16,232.5 ms |
| `adaptive` | 1.0000 | 1.0000 | 0.0000 | 1,622 | 0.00045050 | 10,795.8 ms |

### Cross-Model Takeaway

| Model | Strategy | Quality | Prompt token reduction vs full | Cost reduction vs full |
|---|---|---:|---:|---:|
| `openai/gpt-4o-mini` | `adaptive` | 1.0000 | 76.69% | 75.36% |
| `google/gemini-3.1-flash-lite` | `adaptive` | 1.0000 | 76.16% | 74.20% |

The live answer results should not be oversold. The stronger engineering signal is the offline memory-quality suite because it exposes stale-evidence pressure and policy failures.
