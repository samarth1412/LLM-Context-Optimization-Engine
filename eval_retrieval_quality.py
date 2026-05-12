import argparse
import csv
import json
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set

import database
from eval_memory_quality import MemoryCase, SummaryFact, _messages, build_cases
from semantic_memory import index_message, normalize_embedding_model, normalize_retrieval_mode, retrieve


DEFAULT_MODES = ["bm25", "embedding", "hybrid"]


@dataclass
class RetrievalCaseResult:
    case: str
    category: str
    mode: str
    embedding_model: str
    top_k: int
    has_relevant: bool
    recall_at_k: Optional[float]
    precision_at_k: Optional[float]
    mrr: Optional[float]
    distractor_hit_rate: float
    stale_evidence_rate: float
    retrieved_message_ids: List[int]
    relevant_message_ids: List[int]
    distractor_message_ids: List[int]
    stale_message_ids: List[int]
    top_hit_is_relevant: bool


def build_retrieval_cases() -> List[MemoryCase]:
    cases = build_cases()
    cases.extend(
        [
            MemoryCase(
                name="semantic_paraphrase_retrieval",
                category="semantic_gap",
                query="What kind of restaurant should we pick tonight?",
                messages=_messages(
                    80,
                    {
                        10: "Durable user preference: the user enjoys Japanese cuisine for evening meals.",
                        28: "Distractor note: the restaurant analytics dashboard runs a nightly batch job.",
                    },
                ),
                summary_facts=[
                    SummaryFact(
                        10,
                        "dinner_preference",
                        "The user enjoys Japanese cuisine for evening meals.",
                    )
                ],
                required_terms=["japanese"],
                conflict_terms=[],
                relevant_message_indices=[10],
                distractor_message_indices=[28],
                note="Tests whether embeddings can bridge a restaurant/dinner paraphrase that lexical retrieval overweights.",
            ),
            MemoryCase(
                name="abbreviation_entity_retrieval",
                category="entity_abbreviation",
                query="Where should EU launch traffic route?",
                messages=_messages(
                    80,
                    {
                        12: "Project Atlas launch routing target is the Europe region.",
                        30: "Project Atlas launch routing target used to be the United States region.",
                        48: "Project Boreal handles general traffic routing documentation.",
                    },
                ),
                summary_facts=[
                    SummaryFact(
                        12,
                        "atlas_region",
                        "Project Atlas launch routing target is the Europe region.",
                    )
                ],
                required_terms=["europe"],
                conflict_terms=["united states"],
                relevant_message_indices=[12],
                stale_message_indices=[30],
                distractor_message_indices=[48],
                note="Tests abbreviation/entity matching and stale launch-region evidence.",
            ),
        ]
    )
    return cases


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _populate_case(session_id: str, case: MemoryCase, embedding_model: str) -> Dict[int, int]:
    message_id_by_index: Dict[int, int] = {}
    for index, message in enumerate(case.messages):
        message_id = database.store_message_with_usage(
            session_id,
            message["role"],
            message["content"],
        )
        message_id_by_index[index] = int(message_id)
        index_message(
            session_id,
            int(message_id),
            message["role"],
            message["content"],
            embedding_model=embedding_model,
        )
    return message_id_by_index


def _ids_for(indices: List[int], mapping: Dict[int, int]) -> Set[int]:
    return {mapping[index] for index in indices if index in mapping}


def _metrics(
    case: MemoryCase,
    retrieved_ids: List[int],
    message_id_by_index: Dict[int, int],
    top_k: int,
) -> Dict:
    relevant = _ids_for(case.relevant_message_indices, message_id_by_index)
    distractors = _ids_for(case.distractor_message_indices, message_id_by_index)
    stale = _ids_for(case.stale_message_indices, message_id_by_index)
    retrieved = retrieved_ids[:top_k]

    relevant_hits = [message_id for message_id in retrieved if message_id in relevant]
    distractor_hits = [message_id for message_id in retrieved if message_id in distractors]
    stale_hits = [message_id for message_id in retrieved if message_id in stale]
    has_relevant = bool(relevant)

    if has_relevant:
        recall_at_k = len(set(relevant_hits)) / len(relevant)
        precision_at_k = len(relevant_hits) / max(1, top_k)
        first_rank = next(
            (rank for rank, message_id in enumerate(retrieved, start=1) if message_id in relevant),
            None,
        )
        mrr = 1 / first_rank if first_rank else 0.0
    else:
        recall_at_k = None
        precision_at_k = None
        mrr = None

    return {
        "has_relevant": has_relevant,
        "recall_at_k": round(recall_at_k, 4) if recall_at_k is not None else None,
        "precision_at_k": round(precision_at_k, 4) if precision_at_k is not None else None,
        "mrr": round(mrr, 4) if mrr is not None else None,
        "distractor_hit_rate": round(len(distractor_hits) / max(1, top_k), 4),
        "stale_evidence_rate": round(len(stale_hits) / max(1, top_k), 4),
        "relevant_message_ids": sorted(relevant),
        "distractor_message_ids": sorted(distractors),
        "stale_message_ids": sorted(stale),
        "top_hit_is_relevant": bool(retrieved and retrieved[0] in relevant),
    }


def run_eval(
    embedding_models: List[str] = None,
    modes: List[str] = None,
    top_k: int = 6,
    skip_unavailable: bool = False,
) -> Dict:
    cases = build_retrieval_cases()
    selected_models = [normalize_embedding_model(model) for model in (embedding_models or ["mock/hash"])]
    selected_modes = [normalize_retrieval_mode(mode) for mode in (modes or DEFAULT_MODES)]
    rows: List[RetrievalCaseResult] = []
    skipped_models = []

    original_db = database.DB_NAME
    with tempfile.TemporaryDirectory() as tmpdir:
        database.DB_NAME = os.path.join(tmpdir, "retrieval_eval.db")
        database.init_database()

        try:
            for embedding_model in selected_models:
                model_available = True
                for case_index, case in enumerate(cases):
                    session_id = f"retrieval_{_safe_name(embedding_model)}_{case_index}_{case.name}"
                    try:
                        message_id_by_index = _populate_case(session_id, case, embedding_model)
                    except Exception as exc:
                        if skip_unavailable:
                            skipped_models.append(
                                {"embedding_model": embedding_model, "reason": str(exc)}
                            )
                            model_available = False
                            break
                        raise

                    for mode in selected_modes:
                        retrieved = retrieve(
                            session_id,
                            case.query,
                            top_k=top_k,
                            mode=mode,
                            embedding_model=embedding_model,
                        )
                        retrieved_ids = [int(item["message_id"]) for item in retrieved]
                        metrics = _metrics(case, retrieved_ids, message_id_by_index, top_k)
                        rows.append(
                            RetrievalCaseResult(
                                case=case.name,
                                category=case.category,
                                mode=mode,
                                embedding_model=embedding_model,
                                top_k=top_k,
                                retrieved_message_ids=retrieved_ids,
                                **metrics,
                            )
                        )
                if not model_available:
                    continue
        finally:
            database.DB_NAME = original_db

    aggregate = []
    for embedding_model in selected_models:
        for mode in selected_modes:
            strategy_rows = [
                row
                for row in rows
                if row.embedding_model == embedding_model and row.mode == mode
            ]
            relevant_rows = [row for row in strategy_rows if row.has_relevant]
            if not strategy_rows:
                continue
            aggregate.append(
                {
                    "embedding_model": embedding_model,
                    "mode": mode,
                    "cases": len(strategy_rows),
                    "mean_recall_at_k": round(
                        statistics.mean(row.recall_at_k for row in relevant_rows), 4
                    )
                    if relevant_rows
                    else None,
                    "mean_precision_at_k": round(
                        statistics.mean(row.precision_at_k for row in relevant_rows), 4
                    )
                    if relevant_rows
                    else None,
                    "mean_mrr": round(statistics.mean(row.mrr for row in relevant_rows), 4)
                    if relevant_rows
                    else None,
                    "mean_distractor_hit_rate": round(
                        statistics.mean(row.distractor_hit_rate for row in strategy_rows), 4
                    ),
                    "mean_stale_evidence_rate": round(
                        statistics.mean(row.stale_evidence_rate for row in strategy_rows), 4
                    ),
                    "top_hit_accuracy": round(
                        statistics.mean(1.0 if row.top_hit_is_relevant else 0.0 for row in relevant_rows),
                        4,
                    )
                    if relevant_rows
                    else None,
                }
            )

    return {
        "top_k": top_k,
        "embedding_models": selected_models,
        "modes": selected_modes,
        "skipped_models": skipped_models,
        "cases": [
            {
                "name": case.name,
                "category": case.category,
                "relevant_message_indices": case.relevant_message_indices,
                "stale_message_indices": case.stale_message_indices,
                "distractor_message_indices": case.distractor_message_indices,
                "note": case.note,
            }
            for case in cases
        ],
        "results": [asdict(row) for row in rows],
        "aggregate": aggregate,
    }


def export_results(results: Dict, out_dir: str = "results") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "retrieval_quality.json")
    csv_path = os.path.join(out_dir, "retrieval_quality.csv")
    summary_csv_path = os.path.join(out_dir, "retrieval_quality_summary.csv")
    heatmap_path = os.path.join(out_dir, "retrieval_confusion_heatmap.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "case",
        "category",
        "mode",
        "embedding_model",
        "top_k",
        "has_relevant",
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "distractor_hit_rate",
        "stale_evidence_rate",
        "top_hit_is_relevant",
        "retrieved_message_ids",
        "relevant_message_ids",
        "distractor_message_ids",
        "stale_message_ids",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["results"]:
            writer.writerow({field: row.get(field) for field in fieldnames})

    summary_fieldnames = [
        "embedding_model",
        "mode",
        "cases",
        "mean_recall_at_k",
        "mean_precision_at_k",
        "mean_mrr",
        "mean_distractor_hit_rate",
        "mean_stale_evidence_rate",
        "top_hit_accuracy",
    ]
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in results["aggregate"]:
            writer.writerow({field: row.get(field) for field in summary_fieldnames})

    _export_confusion_heatmap(results, heatmap_path)

    return {
        "json": json_path,
        "csv": csv_path,
        "summary_csv": summary_csv_path,
        "confusion_heatmap_png": heatmap_path,
    }


def _export_confusion_heatmap(results: Dict, heatmap_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    rows = results["results"]
    cases = list(dict.fromkeys(row["case"] for row in rows))
    strategy_keys = list(
        dict.fromkeys(f"{row['mode']}\n{row['embedding_model']}" for row in rows)
    )
    if not cases or not strategy_keys:
        return

    matrix = []
    for case in cases:
        case_values = []
        for strategy_key in strategy_keys:
            mode, embedding_model = strategy_key.split("\n", 1)
            match = next(
                (
                    row
                    for row in rows
                    if row["case"] == case
                    and row["mode"] == mode
                    and row["embedding_model"] == embedding_model
                ),
                None,
            )
            if match:
                case_values.append(
                    float(match["distractor_hit_rate"]) + float(match["stale_evidence_rate"])
                )
            else:
                case_values.append(0.0)
        matrix.append(case_values)

    fig_width = max(8, len(strategy_keys) * 2.1)
    fig_height = max(5, len(cases) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=max(0.35, max(max(row) for row in matrix)))
    ax.set_title("Retrieval confusion pressure by case")
    ax.set_xlabel("Retrieval mode / embedding model")
    ax.set_ylabel("Evaluation case")
    ax.set_xticks(range(len(strategy_keys)), labels=strategy_keys, rotation=35, ha="right")
    ax.set_yticks(range(len(cases)), labels=cases)
    fig.colorbar(image, ax=ax, label="Distractor + stale evidence rate")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality before context assembly.")
    parser.add_argument("--embedding-models", default="mock/hash")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-dir", default="results")
    args = parser.parse_args()

    embedding_models = [
        item.strip() for item in args.embedding_models.split(",") if item.strip()
    ]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    results = run_eval(
        embedding_models=embedding_models,
        modes=modes,
        top_k=args.top_k,
        skip_unavailable=args.skip_unavailable,
    )

    if args.export:
        export_results(results, args.export_dir)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"Top K: {results['top_k']}")
    print("embedding_model               mode       recall  precision     mrr  distract  stale")
    print("-" * 88)
    for row in results["aggregate"]:
        print(
            f"{row['embedding_model']:<30}"
            f"{row['mode']:<10}"
            f"{row['mean_recall_at_k'] if row['mean_recall_at_k'] is not None else 0:>7.3f}"
            f"{row['mean_precision_at_k'] if row['mean_precision_at_k'] is not None else 0:>11.3f}"
            f"{row['mean_mrr'] if row['mean_mrr'] is not None else 0:>8.3f}"
            f"{row['mean_distractor_hit_rate']:>10.3f}"
            f"{row['mean_stale_evidence_rate']:>8.3f}"
        )


if __name__ == "__main__":
    main()
