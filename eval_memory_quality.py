import argparse
import csv
import json
import logging
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, List

import context as context_engine
import database
from database import calculate_cost, estimate_messages_tokens
from semantic_memory import index_message


logging.getLogger("context").setLevel(logging.WARNING)

STRATEGIES = [
    "full_history",
    "sliding_window",
    "summary",
    "retrieval",
    "hybrid",
    "adaptive",
]


@dataclass
class SummaryFact:
    message_index: int
    key: str
    text: str


@dataclass
class MemoryCase:
    name: str
    query: str
    messages: List[Dict]
    summary_facts: List[SummaryFact]
    required_terms: List[str]
    conflict_terms: List[str]


@dataclass
class CaseResult:
    case: str
    strategy: str
    context_tokens: int
    estimated_input_cost_usd: float
    message_count: int
    required_recall: float
    conflict_pressure: float
    quality_score: float


def _filler(index: int) -> Dict:
    role = "user" if index % 2 == 0 else "assistant"
    return {
        "role": role,
        "content": (
            f"Routine turn {index}: neutral planning chatter about deadlines, "
            "formatting, and ordinary implementation details."
        ),
    }


def _messages(total_messages: int, inserts: Dict[int, str]) -> List[Dict]:
    messages = [_filler(index) for index in range(total_messages)]
    for index, content in inserts.items():
        messages[index] = {"role": "user", "content": content}
    return messages


def build_cases() -> List[MemoryCase]:
    return [
        MemoryCase(
            name="long_range_preference",
            query="What is the user's favorite tea?",
            messages=_messages(
                80,
                {
                    4: "Stable user memory: the user's favorite tea is jasmine.",
                },
            ),
            summary_facts=[
                SummaryFact(4, "favorite_tea", "The user's favorite tea is jasmine.")
            ],
            required_terms=["jasmine"],
            conflict_terms=[],
        ),
        MemoryCase(
            name="temporal_update",
            query="What is the current preferred editor?",
            messages=_messages(
                80,
                {
                    6: "Old user memory: the preferred editor used to be Vim.",
                    58: "Updated user memory: the current preferred editor is Zed.",
                },
            ),
            summary_facts=[
                SummaryFact(6, "preferred_editor", "The preferred editor used to be Vim."),
                SummaryFact(58, "preferred_editor", "The current preferred editor is Zed."),
            ],
            required_terms=["zed"],
            conflict_terms=["preferred editor used to be vim"],
        ),
        MemoryCase(
            name="multi_hop_project_state",
            query="What logo does Zephyr use?",
            messages=_messages(
                80,
                {
                    8: "Project memory: the internal codename is Zephyr.",
                    50: "Brand memory: Zephyr uses a dragonfly logo.",
                },
            ),
            summary_facts=[
                SummaryFact(8, "codename", "The internal codename is Zephyr."),
                SummaryFact(50, "logo", "Zephyr uses a dragonfly logo."),
            ],
            required_terms=["zephyr", "dragonfly"],
            conflict_terms=[],
        ),
        MemoryCase(
            name="abstention_no_evidence",
            query="What is the user's passport number?",
            messages=_messages(80, {}),
            summary_facts=[],
            required_terms=[],
            conflict_terms=["passport number is"],
        ),
    ]


def _summary_for_case(case: MemoryCase, covered_messages: int) -> str:
    latest_by_key: Dict[str, SummaryFact] = {}
    for fact in case.summary_facts:
        if fact.message_index < covered_messages:
            latest_by_key[fact.key] = fact

    if not latest_by_key:
        return "No durable facts have been established yet."

    return "\n".join(fact.text for _, fact in sorted(latest_by_key.items()))


def _populate_case(session_id: str, case: MemoryCase) -> None:
    for message in case.messages:
        message_id = database.store_message_with_usage(
            session_id,
            message["role"],
            message["content"],
        )
        index_message(session_id, int(message_id), message["role"], message["content"])

    covered_messages = max(0, len(case.messages) - context_engine.RECENT_MESSAGE_COUNT)
    database.cache_summary(session_id, covered_messages, _summary_for_case(case, covered_messages))


def _score_context(case: MemoryCase, text: str) -> Dict[str, float]:
    lower = text.lower()
    if case.required_terms:
        required_recall = sum(1 for term in case.required_terms if term.lower() in lower) / len(
            case.required_terms
        )
    else:
        required_recall = 1.0

    if case.conflict_terms:
        conflict_pressure = sum(
            1 for term in case.conflict_terms if term.lower() in lower
        ) / len(case.conflict_terms)
    else:
        conflict_pressure = 0.0

    quality_score = max(0.0, required_recall - (0.35 * conflict_pressure))
    return {
        "required_recall": round(required_recall, 4),
        "conflict_pressure": round(conflict_pressure, 4),
        "quality_score": round(quality_score, 4),
    }


def run_eval(model: str = "mock/echo", strategies: List[str] = None) -> Dict:
    strategies = strategies or STRATEGIES
    cases = build_cases()
    rows: List[CaseResult] = []

    original_db = database.DB_NAME
    with tempfile.TemporaryDirectory() as tmpdir:
        database.DB_NAME = os.path.join(tmpdir, "memory_eval.db")
        database.init_database()

        try:
            for case_index, case in enumerate(cases):
                session_id = f"case_{case_index}_{case.name}"
                _populate_case(session_id, case)

                for strategy in strategies:
                    messages = context_engine.build_context(
                        session_id,
                        model=model,
                        query=case.query,
                        policy=strategy,
                    )
                    context_text = "\n".join(msg["content"] for msg in messages)
                    tokens = estimate_messages_tokens(messages, model)
                    score = _score_context(case, context_text)
                    rows.append(
                        CaseResult(
                            case=case.name,
                            strategy=strategy,
                            context_tokens=tokens,
                            estimated_input_cost_usd=round(
                                calculate_cost(model, tokens, 0)["total"],
                                8,
                            ),
                            message_count=len(messages),
                            required_recall=score["required_recall"],
                            conflict_pressure=score["conflict_pressure"],
                            quality_score=score["quality_score"],
                        )
                    )
        finally:
            database.DB_NAME = original_db

    aggregate = []
    for strategy in strategies:
        strategy_rows = [row for row in rows if row.strategy == strategy]
        total_tokens = sum(row.context_tokens for row in strategy_rows)
        aggregate.append(
            {
                "strategy": strategy,
                "mean_quality_score": round(
                    statistics.mean(row.quality_score for row in strategy_rows), 4
                ),
                "mean_required_recall": round(
                    statistics.mean(row.required_recall for row in strategy_rows), 4
                ),
                "mean_conflict_pressure": round(
                    statistics.mean(row.conflict_pressure for row in strategy_rows), 4
                ),
                "total_context_tokens": total_tokens,
                "mean_context_tokens": round(
                    statistics.mean(row.context_tokens for row in strategy_rows), 2
                ),
                "quality_per_1k_tokens": round(
                    (
                        statistics.mean(row.quality_score for row in strategy_rows)
                        / max(1, total_tokens / len(strategy_rows) / 1000)
                    ),
                    4,
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
    json_path = os.path.join(out_dir, "memory_quality.json")
    csv_path = os.path.join(out_dir, "memory_quality.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "case",
        "strategy",
        "context_tokens",
        "estimated_input_cost_usd",
        "message_count",
        "required_recall",
        "conflict_pressure",
        "quality_score",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["results"]:
            writer.writerow({field: row.get(field) for field in fieldnames})

    return {"json": json_path, "csv": csv_path}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate context strategies on deterministic long-memory tasks."
    )
    parser.add_argument("--model", default="mock/echo")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-dir", default="results")
    args = parser.parse_args()

    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    results = run_eval(model=args.model, strategies=strategies)

    if args.export:
        export_results(results, args.export_dir)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"Model: {results['model']}")
    print("strategy                 quality  recall  conflict  mean_tokens  q/1k_tok")
    print("-" * 78)
    for row in results["aggregate"]:
        print(
            f"{row['strategy']:<24}"
            f"{row['mean_quality_score']:>7.3f}"
            f"{row['mean_required_recall']:>8.3f}"
            f"{row['mean_conflict_pressure']:>10.3f}"
            f"{row['mean_context_tokens']:>13,.0f}"
            f"{row['quality_per_1k_tokens']:>10.3f}"
        )


if __name__ == "__main__":
    main()
