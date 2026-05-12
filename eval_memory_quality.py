import argparse
import csv
import json
import logging
import os
import re
import statistics
import tempfile
from dataclasses import asdict, dataclass, field
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
    category: str
    query: str
    messages: List[Dict]
    summary_facts: List[SummaryFact]
    required_terms: List[str]
    conflict_terms: List[str]
    relevant_message_indices: List[int] = field(default_factory=list)
    stale_message_indices: List[int] = field(default_factory=list)
    distractor_message_indices: List[int] = field(default_factory=list)
    note: str = ""


@dataclass
class CaseResult:
    case: str
    category: str
    strategy: str
    context_tokens: int
    estimated_input_cost_usd: float
    message_count: int
    included_summary: bool
    included_retrieval: bool
    top_retrieval_score: float
    policy_reason: str
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
            category="long_range_recall",
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
            relevant_message_indices=[4],
            note="Early fact is outside the recent window.",
        ),
        MemoryCase(
            name="temporal_update",
            category="temporal_update",
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
            relevant_message_indices=[58],
            stale_message_indices=[6],
            note="Older stale fact is replaced before the recent window.",
        ),
        MemoryCase(
            name="multi_hop_project_state",
            category="multi_hop",
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
            relevant_message_indices=[8, 50],
            note="Answer requires connecting codename and later brand fact.",
        ),
        MemoryCase(
            name="abstention_no_evidence",
            category="abstention",
            query="What is the user's passport number?",
            messages=_messages(80, {}),
            summary_facts=[],
            required_terms=[],
            conflict_terms=["passport number is"],
            relevant_message_indices=[],
            note="Correct behavior is to avoid inventing a missing private fact.",
        ),
        MemoryCase(
            name="distractor_retrieval_noise",
            category="retrieval_noise",
            query="What color is the Vega release badge?",
            messages=_messages(
                96,
                {
                    6: "Project memory: Vega release badge color is teal.",
                    18: "Distractor note: Orion release badge color is amber.",
                    38: "Distractor note: Vega release checklist mentions badges but not colors.",
                    60: "Distractor note: badge color decisions are often deferred.",
                },
            ),
            summary_facts=[
                SummaryFact(6, "vega_badge", "Vega release badge color is teal.")
            ],
            required_terms=["teal"],
            conflict_terms=["amber", "deferred"],
            relevant_message_indices=[6],
            distractor_message_indices=[18, 38, 60],
            note="Lexical retrieval can pull high-overlap but irrelevant badge snippets.",
        ),
        MemoryCase(
            name="similar_entity_confusion",
            category="entity_disambiguation",
            query="Which launch region belongs to Project Atlas?",
            messages=_messages(
                96,
                {
                    10: "Project memory: Project Atlas launches in the north region.",
                    28: "Project memory: Project Atala launches in the south region.",
                    52: "Project memory: Project Atlas support tickets use the amber queue.",
                },
            ),
            summary_facts=[
                SummaryFact(10, "atlas_region", "Project Atlas launches in the north region."),
                SummaryFact(28, "atala_region", "Project Atala launches in the south region."),
            ],
            required_terms=["north"],
            conflict_terms=["south"],
            relevant_message_indices=[10],
            distractor_message_indices=[28, 52],
            note="Near-name entity collision tests whether context includes confusing neighbors.",
        ),
        MemoryCase(
            name="recent_override",
            category="recent_update",
            query="What is the current escalation channel?",
            messages=_messages(
                96,
                {
                    8: "Old operations memory: the escalation channel is email.",
                    88: "Updated operations memory: the current escalation channel is PagerDuty.",
                },
            ),
            summary_facts=[
                SummaryFact(8, "escalation_channel", "The escalation channel is email."),
            ],
            required_terms=["pagerduty"],
            conflict_terms=["escalation channel is email"],
            relevant_message_indices=[88],
            stale_message_indices=[8],
            note="Recent message overrides a stale summary fact.",
        ),
        MemoryCase(
            name="summary_drift_missing_constraint",
            category="summary_drift",
            query="What constraint must the rollout preserve?",
            messages=_messages(
                96,
                {
                    12: "Rollout memory: the rollout must preserve audit logs.",
                    34: "Rollout memory: the rollout also needs faster dashboard load time.",
                },
            ),
            summary_facts=[
                SummaryFact(34, "rollout_speed", "The rollout needs faster dashboard load time."),
            ],
            required_terms=["audit logs"],
            conflict_terms=[],
            relevant_message_indices=[12],
            distractor_message_indices=[34],
            note="Simulates summary drift by omitting a critical early constraint.",
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
        required_recall = sum(1 for term in case.required_terms if _contains_term(lower, term)) / len(
            case.required_terms
        )
    else:
        required_recall = 1.0

    if case.conflict_terms:
        conflict_pressure = sum(
            1 for term in case.conflict_terms if _contains_term(lower, term)
        ) / len(case.conflict_terms)
    else:
        conflict_pressure = 0.0

    quality_score = max(0.0, required_recall - (0.35 * conflict_pressure))
    return {
        "required_recall": round(required_recall, 4),
        "conflict_pressure": round(conflict_pressure, 4),
        "quality_score": round(quality_score, 4),
    }


def _contains_term(lower_text: str, term: str) -> bool:
    normalized = term.lower().strip()
    if not normalized:
        return False
    if " " in normalized:
        return normalized in lower_text
    return re.search(rf"\b{re.escape(normalized)}\b", lower_text) is not None


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
                    decision = context_engine.explain_policy(
                        session_id,
                        query=case.query,
                        policy=strategy,
                    )
                    context_text = "\n".join(msg["content"] for msg in messages)
                    tokens = estimate_messages_tokens(messages, model)
                    score = _score_context(case, context_text)
                    rows.append(
                        CaseResult(
                            case=case.name,
                            category=case.category,
                            strategy=strategy,
                            context_tokens=tokens,
                            estimated_input_cost_usd=round(
                                calculate_cost(model, tokens, 0)["total"],
                                8,
                            ),
                            message_count=len(messages),
                            included_summary=bool(decision["include_summary"]),
                            included_retrieval=bool(decision["include_retrieval"]),
                            top_retrieval_score=round(float(decision["top_retrieval_score"]), 4),
                            policy_reason=str(decision["reason"]),
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
                "failure_count": sum(1 for row in strategy_rows if row.quality_score < 1.0),
            }
        )

    failures = [
        {
            "case": row.case,
            "category": row.category,
            "strategy": row.strategy,
            "quality_score": row.quality_score,
            "required_recall": row.required_recall,
            "conflict_pressure": row.conflict_pressure,
            "context_tokens": row.context_tokens,
            "policy_reason": row.policy_reason,
        }
        for row in rows
        if row.quality_score < 1.0
    ]

    return {
        "model": model,
        "cases": [
            {
                "name": case.name,
                "category": case.category,
                "note": case.note,
            }
            for case in cases
        ],
        "strategies": strategies,
        "results": [asdict(row) for row in rows],
        "aggregate": aggregate,
        "failure_analysis": failures,
    }


def export_results(results: Dict, out_dir: str = "results") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "memory_quality.json")
    csv_path = os.path.join(out_dir, "memory_quality.csv")
    failures_path = os.path.join(out_dir, "memory_quality_failures.csv")
    chart_path = os.path.join(out_dir, "memory_quality_pareto.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "case",
        "category",
        "strategy",
        "context_tokens",
        "estimated_input_cost_usd",
        "message_count",
        "included_summary",
        "included_retrieval",
        "top_retrieval_score",
        "policy_reason",
        "required_recall",
        "conflict_pressure",
        "quality_score",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["results"]:
            writer.writerow({field: row.get(field) for field in fieldnames})

    failure_fieldnames = [
        "case",
        "category",
        "strategy",
        "quality_score",
        "required_recall",
        "conflict_pressure",
        "context_tokens",
        "policy_reason",
    ]
    with open(failures_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=failure_fieldnames)
        writer.writeheader()
        for row in results.get("failure_analysis", []):
            writer.writerow({field: row.get(field) for field in failure_fieldnames})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    rows = results["aggregate"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        ax.scatter(
            row["mean_context_tokens"],
            row["mean_quality_score"],
            s=80,
        )
        ax.annotate(
            row["strategy"],
            (row["mean_context_tokens"], row["mean_quality_score"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_title("Memory quality vs context size")
    ax.set_xlabel("Mean context tokens")
    ax.set_ylabel("Mean quality score")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    return {
        "json": json_path,
        "csv": csv_path,
        "failures_csv": failures_path,
        "pareto_png": chart_path,
    }


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
