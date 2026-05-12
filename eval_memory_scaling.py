import argparse
import csv
import json
import os
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, List

from memory_importance import score_memory


POLICIES = ["full_archive", "sliding_window", "importance"]
DEFAULT_TURNS = [1000, 5000, 10000]
PREFERENCE_KEYS = ["tea", "editor", "cuisine", "notification_channel", "dashboard_metric"]
PREFERENCE_VALUES = ["jasmine", "zed", "japanese", "pagerduty", "p95 latency"]
ENTITIES = ["Atlas", "Zephyr", "Vega", "Orion", "Boreal", "Atala"]


@dataclass
class SyntheticMemory:
    turn: int
    role: str
    content: str
    category: str
    key: str
    is_current_critical: bool
    is_stale: bool
    is_noise: bool
    retrieval_count: int
    importance_score: float = 0.0
    memory_layer: str = ""
    memory_action: str = ""


def _mark_previous_stale(events: List[SyntheticMemory], key: str) -> None:
    for event in reversed(events):
        if event.key == key and event.is_current_critical:
            event.is_current_critical = False
            event.is_stale = True
            event.content = f"Old stale memory superseded by a later update: {event.content}"
            event.retrieval_count = max(0, event.retrieval_count - 2)
            return


def generate_session(turns: int) -> List[SyntheticMemory]:
    events: List[SyntheticMemory] = []
    for turn in range(1, turns + 1):
        role = "user" if turn % 2 else "assistant"
        if turn % 37 == 0:
            idx = (turn // 37) % len(PREFERENCE_KEYS)
            key = f"preference:{PREFERENCE_KEYS[idx]}"
            _mark_previous_stale(events, key)
            value = PREFERENCE_VALUES[idx]
            content = f"Updated durable user preference: the current {PREFERENCE_KEYS[idx]} is {value}."
            events.append(
                SyntheticMemory(
                    turn,
                    role,
                    content,
                    "evolving_preference",
                    key,
                    True,
                    False,
                    False,
                    4 + (turn % 5),
                )
            )
        elif turn % 53 == 0:
            entity = ENTITIES[(turn // 53) % len(ENTITIES)]
            key = f"constraint:{entity}:{turn}"
            content = f"Durable project constraint: Project {entity} must preserve audit logs during rollout."
            events.append(
                SyntheticMemory(
                    turn,
                    role,
                    content,
                    "forgotten_constraint",
                    key,
                    True,
                    False,
                    False,
                    3,
                )
            )
        elif turn % 41 == 0:
            entity = ENTITIES[(turn // 41) % len(ENTITIES)]
            key = f"entity:{entity}"
            _mark_previous_stale(events, key)
            region = ["north", "south", "europe", "west"][(turn // 41) % 4]
            content = f"Current entity memory: Project {entity} launch region is {region}."
            events.append(
                SyntheticMemory(
                    turn,
                    role,
                    content,
                    "entity_drift",
                    key,
                    True,
                    False,
                    False,
                    2 + (turn % 3),
                )
            )
        elif turn % 29 == 0:
            entity = ENTITIES[(turn // 29) % len(ENTITIES)]
            content = f"Distractor note: Project {entity} release badge discussion mentioned colors but no final decision."
            events.append(
                SyntheticMemory(
                    turn,
                    role,
                    content,
                    "retrieval_noise",
                    f"noise:{turn}",
                    False,
                    False,
                    True,
                    0,
                )
            )
        else:
            events.append(
                SyntheticMemory(
                    turn,
                    role,
                    f"Routine turn {turn}: neutral implementation chatter about formatting and scheduling.",
                    "routine",
                    f"routine:{turn}",
                    False,
                    False,
                    True,
                    0,
                )
            )

    for event in events:
        scored = score_memory(
            event.content,
            role=event.role,
            message_id=event.turn,
            latest_message_id=turns,
            retrieval_count=event.retrieval_count,
        )
        event.importance_score = scored["importance_score"]
        event.memory_layer = scored["memory_layer"]
        event.memory_action = scored["memory_action"]
    return events


def _selection_budget(turns: int) -> int:
    return max(120, min(600, int(turns * 0.06)))


def select_memories(events: List[SyntheticMemory], policy: str) -> List[SyntheticMemory]:
    turns = len(events)
    budget = _selection_budget(turns)
    if policy == "full_archive":
        return list(events)
    if policy == "sliding_window":
        return events[-budget:]
    if policy != "importance":
        raise ValueError(f"Unknown policy: {policy}")

    preserved = [event for event in events if event.memory_action == "preserve"]
    compressed = sorted(
        [event for event in events if event.memory_action == "compress"],
        key=lambda event: event.importance_score,
        reverse=True,
    )
    selected = sorted(preserved + compressed, key=lambda event: event.importance_score, reverse=True)
    recent_working = events[-min(60, budget // 2):]
    selected_by_turn = {event.turn: event for event in selected[:budget]}
    for event in recent_working:
        selected_by_turn.setdefault(event.turn, event)
    return sorted(selected_by_turn.values(), key=lambda event: event.turn)[-budget:]


def _token_count(events: List[SyntheticMemory]) -> int:
    return sum(max(1, len(event.content.split())) for event in events)


def _layer_counts(events: List[SyntheticMemory]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        counts[event.memory_layer] = counts.get(event.memory_layer, 0) + 1
    return counts


def _action_counts(events: List[SyntheticMemory]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        counts[event.memory_action] = counts.get(event.memory_action, 0) + 1
    return counts


def _metrics(turns: int, policy: str, events: List[SyntheticMemory]) -> Dict:
    selected = select_memories(events, policy)
    selected_turns = {event.turn for event in selected}
    current = [event for event in events if event.is_current_critical]
    stale = [event for event in events if event.is_stale]
    noise = [event for event in events if event.is_noise]
    selected_current = [event for event in current if event.turn in selected_turns]
    selected_stale = [event for event in stale if event.turn in selected_turns]
    selected_noise = [event for event in noise if event.turn in selected_turns]

    selected_evidence = [
        event
        for event in selected
        if event.is_current_critical or event.is_stale or event.is_noise
    ]
    retrieval_precision_estimate = len(selected_current) / max(1, len(selected_evidence))

    return {
        "turns": turns,
        "policy": policy,
        "selection_budget": _selection_budget(turns),
        "kept_messages": len(selected),
        "kept_tokens": _token_count(selected),
        "critical_facts": len(current),
        "stale_facts": len(stale),
        "noise_messages": len(noise),
        "critical_recall": round(len(selected_current) / max(1, len(current)), 4),
        "stale_retention_rate": round(len(selected_stale) / max(1, len(stale)), 4),
        "noise_retention_rate": round(len(selected_noise) / max(1, len(noise)), 4),
        "retrieval_precision_estimate": round(retrieval_precision_estimate, 4),
        "mean_kept_importance": round(
            statistics.mean(event.importance_score for event in selected), 4
        ) if selected else 0.0,
        "layer_counts": _layer_counts(selected),
        "action_counts": _action_counts(selected),
    }


def run_eval(turns_list: List[int] = None) -> Dict:
    turns_list = turns_list or DEFAULT_TURNS
    rows = []
    for turns in turns_list:
        events = generate_session(turns)
        for policy in POLICIES:
            rows.append(_metrics(turns, policy, events))
    return {
        "turns": turns_list,
        "policies": POLICIES,
        "results": rows,
    }


def export_results(results: Dict, out_dir: str = "results") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "memory_scaling.json")
    csv_path = os.path.join(out_dir, "memory_scaling.csv")
    recall_chart_path = os.path.join(out_dir, "memory_scaling_recall.png")
    retention_chart_path = os.path.join(out_dir, "memory_scaling_retention.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "turns",
        "policy",
        "selection_budget",
        "kept_messages",
        "kept_tokens",
        "critical_facts",
        "stale_facts",
        "noise_messages",
        "critical_recall",
        "stale_retention_rate",
        "noise_retention_rate",
        "retrieval_precision_estimate",
        "mean_kept_importance",
        "layer_counts",
        "action_counts",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["results"]:
            writer.writerow({field: row.get(field) for field in fieldnames})

    _export_charts(results, recall_chart_path, retention_chart_path)
    return {
        "json": json_path,
        "csv": csv_path,
        "recall_png": recall_chart_path,
        "retention_png": retention_chart_path,
    }


def _export_charts(results: Dict, recall_path: str, retention_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    rows = results["results"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in results["policies"]:
        policy_rows = [row for row in rows if row["policy"] == policy]
        ax.plot(
            [row["turns"] for row in policy_rows],
            [row["critical_recall"] for row in policy_rows],
            marker="o",
            label=policy,
        )
    ax.set_title("Critical memory recall under scaling")
    ax.set_xlabel("Synthetic session turns")
    ax.set_ylabel("Critical recall")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(recall_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in results["policies"]:
        policy_rows = [row for row in rows if row["policy"] == policy]
        ax.plot(
            [row["turns"] for row in policy_rows],
            [row["stale_retention_rate"] for row in policy_rows],
            marker="o",
            label=f"{policy} stale",
        )
        ax.plot(
            [row["turns"] for row in policy_rows],
            [row["noise_retention_rate"] for row in policy_rows],
            marker="x",
            linestyle="--",
            label=f"{policy} noise",
        )
    ax.set_title("Stale/noise retention under scaling")
    ax.set_xlabel("Synthetic session turns")
    ax.set_ylabel("Retention rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(retention_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Large-scale synthetic memory policy evaluation.")
    parser.add_argument("--turns", default=",".join(str(turns) for turns in DEFAULT_TURNS))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-dir", default="results")
    args = parser.parse_args()

    turns_list = [int(item.strip()) for item in args.turns.split(",") if item.strip()]
    results = run_eval(turns_list)
    if args.export:
        export_results(results, args.export_dir)
    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("turns   policy          recall  stale   noise   precision  kept")
    print("-" * 70)
    for row in results["results"]:
        print(
            f"{row['turns']:<8}"
            f"{row['policy']:<16}"
            f"{row['critical_recall']:>6.3f}"
            f"{row['stale_retention_rate']:>8.3f}"
            f"{row['noise_retention_rate']:>8.3f}"
            f"{row['retrieval_precision_estimate']:>11.3f}"
            f"{row['kept_messages']:>7}"
        )


if __name__ == "__main__":
    main()
