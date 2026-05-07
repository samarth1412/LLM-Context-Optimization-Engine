import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

from config import (
    DEFAULT_MODEL,
    RECENT_MESSAGE_COUNT,
    SUMMARY_MAX_TOKENS,
    TARGET_INPUT_TOKENS,
    get_model_config,
)
from database import calculate_cost, estimate_tokens


SYSTEM_TOKENS = 120


@dataclass
class StrategyResult:
    strategy: str
    requests: int
    input_tokens: int
    background_input_tokens: int
    background_output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    savings_vs_full_history_pct: float
    summary_calls: int = 0
    max_request_tokens: int = 0


def synthetic_message(index: int, words_per_message: int) -> str:
    base = [
        f"turn-{index}",
        "character",
        "objective",
        "constraint",
        "memory",
        "conflict",
        "resolution",
        "detail",
    ]
    words = [base[(index + offset) % len(base)] for offset in range(words_per_message)]
    return " ".join(words)


def build_messages(turns: int, words_per_message: int) -> List[Dict]:
    messages = []
    for turn in range(turns):
        messages.append(
            {"role": "user", "content": synthetic_message(turn * 2, words_per_message)}
        )
        messages.append(
            {
                "role": "assistant",
                "content": synthetic_message(turn * 2 + 1, words_per_message),
            }
        )
    return messages


def message_tokens(messages: List[Dict], model: str) -> List[int]:
    return [estimate_tokens(msg["content"], model) + 4 for msg in messages]


def cost_for_tokens(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    return calculate_cost(model, input_tokens, output_tokens)["total"]


def full_history(tokens: List[int], model: str, output_tokens_per_request: int) -> StrategyResult:
    request_totals = []
    for index in range(0, len(tokens), 2):
        request_totals.append(SYSTEM_TOKENS + sum(tokens[: index + 1]))

    input_tokens = sum(request_totals)
    output_tokens = output_tokens_per_request * len(request_totals)
    return StrategyResult(
        strategy="full_history",
        requests=len(request_totals),
        input_tokens=input_tokens,
        background_input_tokens=0,
        background_output_tokens=0,
        total_tokens=input_tokens + output_tokens,
        estimated_cost_usd=cost_for_tokens(model, input_tokens, output_tokens),
        savings_vs_full_history_pct=0.0,
        max_request_tokens=max(request_totals),
    )


def sliding_window(
    tokens: List[int],
    model: str,
    output_tokens_per_request: int,
    recent_messages: int,
) -> StrategyResult:
    request_totals = []
    for index in range(0, len(tokens), 2):
        start = max(0, index + 1 - recent_messages)
        request_totals.append(SYSTEM_TOKENS + sum(tokens[start : index + 1]))

    input_tokens = sum(request_totals)
    output_tokens = output_tokens_per_request * len(request_totals)
    return StrategyResult(
        strategy="sliding_window",
        requests=len(request_totals),
        input_tokens=input_tokens,
        background_input_tokens=0,
        background_output_tokens=0,
        total_tokens=input_tokens + output_tokens,
        estimated_cost_usd=cost_for_tokens(model, input_tokens, output_tokens),
        savings_vs_full_history_pct=0.0,
        max_request_tokens=max(request_totals),
    )


def incremental_summary(
    tokens: List[int],
    model: str,
    output_tokens_per_request: int,
    recent_messages: int,
    summary_ratio: float,
) -> StrategyResult:
    request_totals = []
    summary_tokens = 0
    covered_messages = 0
    background_input_tokens = 0
    background_output_tokens = 0
    summary_calls = 0

    for index in range(0, len(tokens), 2):
        stored_before_current = index
        target_coverage = max(0, stored_before_current - recent_messages)

        if target_coverage > covered_messages:
            new_input = sum(tokens[covered_messages:target_coverage])
            new_output = max(1, int(new_input * summary_ratio))
            background_input_tokens += new_input
            background_output_tokens += new_output
            summary_calls += 1
            covered_messages = target_coverage
            summary_tokens = min(
                SUMMARY_MAX_TOKENS,
                summary_tokens + new_output,
            )

        recent_start = max(covered_messages, stored_before_current - recent_messages)
        recent_tokens = sum(tokens[recent_start : index + 1])
        request_totals.append(SYSTEM_TOKENS + summary_tokens + recent_tokens)

    input_tokens = sum(request_totals)
    output_tokens = output_tokens_per_request * len(request_totals)
    total_cost = cost_for_tokens(
        model,
        input_tokens + background_input_tokens,
        output_tokens + background_output_tokens,
    )
    return StrategyResult(
        strategy="incremental_summary",
        requests=len(request_totals),
        input_tokens=input_tokens,
        background_input_tokens=background_input_tokens,
        background_output_tokens=background_output_tokens,
        total_tokens=input_tokens
        + output_tokens
        + background_input_tokens
        + background_output_tokens,
        estimated_cost_usd=total_cost,
        savings_vs_full_history_pct=0.0,
        summary_calls=summary_calls,
        max_request_tokens=max(request_totals),
    )


def run_benchmark(args) -> Dict:
    model = args.model
    messages = build_messages(args.turns, args.words_per_message)
    tokens = message_tokens(messages, model)

    results = [
        full_history(tokens, model, args.output_tokens),
        sliding_window(tokens, model, args.output_tokens, args.recent_messages),
        incremental_summary(
            tokens,
            model,
            args.output_tokens,
            args.recent_messages,
            args.summary_ratio,
        ),
    ]

    full_cost = results[0].estimated_cost_usd
    full_tokens = results[0].total_tokens
    for result in results:
        if result.strategy != "full_history":
            result.savings_vs_full_history_pct = round(
                (1 - (result.estimated_cost_usd / full_cost)) * 100,
                2,
            )
        result.estimated_cost_usd = round(result.estimated_cost_usd, 6)

    return {
        "model": model,
        "model_config": get_model_config(model),
        "turns": args.turns,
        "messages": len(messages),
        "words_per_message": args.words_per_message,
        "target_input_tokens": TARGET_INPUT_TOKENS,
        "full_history_total_tokens": full_tokens,
        "results": [asdict(result) for result in results],
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_results(results: Dict, out_dir: str = "results") -> Dict[str, str]:
    """Export benchmark artifacts for README + dashboards."""
    _ensure_dir(out_dir)

    json_path = os.path.join(out_dir, "benchmark.json")
    csv_path = os.path.join(out_dir, "benchmark.csv")
    png_path = os.path.join(out_dir, "benchmark.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Flat CSV for quick spreadsheet use.
    fieldnames = [
        "strategy",
        "requests",
        "input_tokens",
        "background_input_tokens",
        "background_output_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "savings_vs_full_history_pct",
        "summary_calls",
        "max_request_tokens",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results.get("results", []):
            writer.writerow({key: row.get(key) for key in fieldnames})

    # Multi-panel chart (headless-safe).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    rows = results["results"]
    full = next((r for r in rows if r["strategy"] == "full_history"), None)
    if not full:
        raise RuntimeError("benchmark export requires full_history baseline")

    strategies = [r["strategy"] for r in rows]
    total_tokens = [r["total_tokens"] for r in rows]
    costs = [r["estimated_cost_usd"] for r in rows]
    bg_costs = [
        cost_for_tokens(
            results["model"],
            int(r.get("background_input_tokens", 0) or 0),
            int(r.get("background_output_tokens", 0) or 0),
        )
        for r in rows
    ]
    max_req = [r.get("max_request_tokens", 0) for r in rows]
    summary_calls = [r.get("summary_calls", 0) for r in rows]
    tokens_saved_pct = [
        0.0 if full["total_tokens"] <= 0 else (1 - (t / full["total_tokens"])) * 100
        for t in total_tokens
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        f"Context strategy benchmark ({results['model']}) — {results['turns']} turns",
        fontsize=14,
        fontweight="bold",
    )

    def bar(ax, values, title, ylabel, color="#2563eb"):
        ax.bar(strategies, values, color=color, alpha=0.9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.2)

    bar(axes[0, 0], tokens_saved_pct, "Tokens saved vs full history", "% saved", color="#0f8b5f")
    bar(axes[0, 1], costs, "Cost comparison", "USD", color="#2563eb")
    bar(axes[0, 2], bg_costs, "Background summary cost", "USD", color="#b7791f")
    bar(axes[1, 0], max_req, "Max context size per strategy", "tokens", color="#1d4ed8")

    # "Summary calls over time" (synthetic): spread summary calls across conversation to show cadence.
    total_turns = int(results.get("turns", 0) or 0)
    time_x = list(range(1, max(total_turns, 1) + 1))
    calls_over_time = [0 for _ in time_x]
    if len(rows) > 0:
        inc = next((r for r in rows if r["strategy"] == "incremental_summary"), None)
        if inc and total_turns > 0:
            calls = int(inc.get("summary_calls", 0) or 0)
            if calls > 0:
                step = max(1, total_turns // calls)
                for t in range(step, total_turns + 1, step):
                    calls_over_time[t - 1] += 1

    axes[1, 1].plot(time_x, calls_over_time, color="#c2410c", linewidth=2)
    axes[1, 1].set_title("Summary calls over time", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Turn")
    axes[1, 1].set_ylabel("calls")
    axes[1, 1].grid(alpha=0.2)

    # Total tokens (absolute).
    bar(axes[1, 2], total_tokens, "Total tokens (chat + background)", "tokens", color="#334155")

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    return {"json": json_path, "csv": csv_path, "png": png_path}


def main():
    parser = argparse.ArgumentParser(
        description="Compare context-memory strategies on a synthetic conversation."
    )
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--words-per-message", type=int, default=90)
    parser.add_argument("--recent-messages", type=int, default=RECENT_MESSAGE_COUNT)
    parser.add_argument("--output-tokens", type=int, default=350)
    parser.add_argument("--summary-ratio", type=float, default=0.18)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-dir", default="results")
    args = parser.parse_args()

    results = run_benchmark(args)

    if args.json:
        print(json.dumps(results, indent=2))
        if args.export:
            export_results(results, out_dir=args.export_dir)
        return

    print(f"Model: {results['model']}")
    print(
        f"Conversation: {results['turns']} turns, {results['messages']} messages, "
        f"{results['words_per_message']} words/message"
    )
    print("")
    print(
        "strategy                 input      bg_input   bg_output  max_req   cost       savings"
    )
    print("-" * 86)
    for result in results["results"]:
        print(
            f"{result['strategy']:<24}"
            f"{result['input_tokens']:>10,}"
            f"{result['background_input_tokens']:>11,}"
            f"{result['background_output_tokens']:>11,}"
            f"{result['max_request_tokens']:>9,}"
            f"  ${result['estimated_cost_usd']:<9.6f}"
            f"{result['savings_vs_full_history_pct']:>7.2f}%"
        )

    if args.export:
        paths = export_results(results, out_dir=args.export_dir)
        print("")
        print("Exported:")
        print(f"  - {paths['json']}")
        print(f"  - {paths['csv']}")
        print(f"  - {paths['png']}")


if __name__ == "__main__":
    main()
