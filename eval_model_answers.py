import argparse
import csv
import json
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, List

import context as context_engine
import database
from database import calculate_cost
from eval_memory_quality import STRATEGIES, _populate_case, build_cases
from llm_utils import call_llm


ABSTENTION_TERMS = (
    "unknown",
    "not provided",
    "not specified",
    "no evidence",
    "don't know",
    "do not know",
    "not in the context",
)


@dataclass
class AnswerResult:
    case: str
    strategy: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    cached_input_tokens: int
    latency_ms: int
    estimated_cost_usd: float
    required_recall: float
    conflict_score: float
    abstained: bool
    answer_quality: float


def _required_recall(answer: str, required_terms: List[str]) -> float:
    if not required_terms:
        return 1.0
    lower = answer.lower()
    return sum(1 for term in required_terms if term.lower() in lower) / len(required_terms)


def _answer_required_terms(case_name: str, required_terms: List[str]) -> List[str]:
    if case_name == "multi_hop_project_state":
        return ["dragonfly"]
    return required_terms


def _conflict_score(answer: str, conflict_terms: List[str], case_name: str) -> float:
    lower = answer.lower()
    terms = list(conflict_terms)
    if case_name == "temporal_update":
        terms.append("vim")
    if not terms:
        return 0.0
    return sum(1 for term in terms if term.lower() in lower) / len(terms)


def _abstained(answer: str) -> bool:
    lower = answer.lower()
    return any(term in lower for term in ABSTENTION_TERMS)


def _answer_prompt(query: str) -> str:
    return (
        f"{query}\n\n"
        "Answer only from the supplied conversation memory and recent context. "
        "If the answer is not present, say that it is unknown from the context."
    )


def run_eval(
    model: str,
    strategies: List[str] = None,
    max_tokens: int = 120,
) -> Dict:
    strategies = strategies or STRATEGIES
    cases = build_cases()
    rows: List[AnswerResult] = []

    original_db = database.DB_NAME
    with tempfile.TemporaryDirectory() as tmpdir:
        database.DB_NAME = os.path.join(tmpdir, "model_answer_eval.db")
        database.init_database()

        try:
            for case_index, case in enumerate(cases):
                session_id = f"case_{case_index}_{case.name}"
                _populate_case(session_id, case)

                for strategy in strategies:
                    context_messages = context_engine.build_context(
                        session_id,
                        model=model,
                        query=case.query,
                        policy=strategy,
                    )
                    prompt = _answer_prompt(case.query)
                    messages = [*context_messages, {"role": "user", "content": prompt}]
                    answer, usage = call_llm(
                        messages,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        model=model,
                    )

                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                    cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
                    cost = calculate_cost(
                        model,
                        prompt_tokens,
                        completion_tokens,
                        cached_input_tokens=cached_input_tokens,
                    )
                    answer_required_terms = _answer_required_terms(
                        case.name,
                        case.required_terms,
                    )
                    recall = _required_recall(answer, answer_required_terms)
                    abstained = _abstained(answer)

                    if answer_required_terms:
                        conflict = _conflict_score(answer, case.conflict_terms, case.name)
                        quality = max(0.0, recall - (0.35 * conflict))
                    else:
                        conflict = 0.0 if abstained else 1.0
                        quality = 1.0 if abstained else 0.0

                    rows.append(
                        AnswerResult(
                            case=case.name,
                            strategy=strategy,
                            answer=answer.strip().replace("\n", " "),
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            cached_input_tokens=cached_input_tokens,
                            latency_ms=int(usage.get("latency_ms", 0) or 0),
                            estimated_cost_usd=round(cost["total"], 8),
                            required_recall=round(recall, 4),
                            conflict_score=round(conflict, 4),
                            abstained=abstained,
                            answer_quality=round(quality, 4),
                        )
                    )
        finally:
            database.DB_NAME = original_db

    aggregate = []
    for strategy in strategies:
        strategy_rows = [row for row in rows if row.strategy == strategy]
        aggregate.append(
            {
                "strategy": strategy,
                "mean_answer_quality": round(
                    statistics.mean(row.answer_quality for row in strategy_rows), 4
                ),
                "mean_required_recall": round(
                    statistics.mean(row.required_recall for row in strategy_rows), 4
                ),
                "mean_conflict_score": round(
                    statistics.mean(row.conflict_score for row in strategy_rows), 4
                ),
                "total_prompt_tokens": sum(row.prompt_tokens for row in strategy_rows),
                "total_completion_tokens": sum(row.completion_tokens for row in strategy_rows),
                "total_cost_usd": round(
                    sum(row.estimated_cost_usd for row in strategy_rows), 8
                ),
                "mean_latency_ms": round(
                    statistics.mean(row.latency_ms for row in strategy_rows), 2
                ),
            }
        )

    return {
        "model": model,
        "cases": [case.name for case in cases],
        "strategies": strategies,
        "results": [asdict(row) for row in rows],
        "aggregate": aggregate,
    }


def export_results(results: Dict, out_dir: str = "results") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    suffix = (
        results.get("model", "model")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )
    json_path = os.path.join(out_dir, f"model_answer_eval_{suffix}.json")
    csv_path = os.path.join(out_dir, f"model_answer_eval_{suffix}.csv")
    latest_json_path = os.path.join(out_dir, "model_answer_eval.json")
    latest_csv_path = os.path.join(out_dir, "model_answer_eval.csv")

    for path in (json_path, latest_json_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    fieldnames = [
        "case",
        "strategy",
        "answer",
        "prompt_tokens",
        "completion_tokens",
        "cached_input_tokens",
        "latency_ms",
        "estimated_cost_usd",
        "required_recall",
        "conflict_score",
        "abstained",
        "answer_quality",
    ]
    for path in (csv_path, latest_csv_path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results["results"]:
                writer.writerow({field: row.get(field) for field in fieldnames})

    return {
        "json": json_path,
        "csv": csv_path,
        "latest_json": latest_json_path,
        "latest_csv": latest_csv_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run real model-answer metrics across context strategies."
    )
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-dir", default="results")
    args = parser.parse_args()

    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    results = run_eval(args.model, strategies=strategies, max_tokens=args.max_tokens)

    if args.export:
        export_results(results, args.export_dir)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"Model: {results['model']}")
    print("strategy                 quality  recall  conflict  prompt_tok  cost_usd  latency")
    print("-" * 86)
    for row in results["aggregate"]:
        print(
            f"{row['strategy']:<24}"
            f"{row['mean_answer_quality']:>7.3f}"
            f"{row['mean_required_recall']:>8.3f}"
            f"{row['mean_conflict_score']:>10.3f}"
            f"{row['total_prompt_tokens']:>12,}"
            f"{row['total_cost_usd']:>10.6f}"
            f"{row['mean_latency_ms']:>9.0f}ms"
        )


if __name__ == "__main__":
    main()
