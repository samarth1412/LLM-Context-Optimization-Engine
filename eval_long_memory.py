import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from llm_utils import call_llm


@dataclass
class EvalResult:
    model: str
    turns: int
    preference_turn: int
    query_turn: int
    recalled: bool
    contradiction: bool
    latency_s: float


def run_eval(
    *,
    model: str,
    turns: int = 80,
    preference_turn: int = 5,
    query_turn: int = 80,
    preference: str = "The user's favorite tea is jasmine.",
) -> EvalResult:
    messages: List[Dict] = [{"role": "system", "content": "You are a helpful assistant."}]

    start = time.time()
    recalled = False
    contradiction = False

    for t in range(1, turns + 1):
        if t == preference_turn:
            user = f"Preference to remember: {preference}"
        elif t == query_turn:
            user = "Question: What is the user's favorite tea?"
        else:
            user = f"Small talk turn {t}. Acknowledge briefly."

        messages.append({"role": "user", "content": user})
        assistant, _usage = call_llm(messages, model=model, max_tokens=200, temperature=0.2)
        messages.append({"role": "assistant", "content": assistant})

        if t == query_turn:
            recalled = "jasmine" in assistant.lower()
            contradiction = ("earl grey" in assistant.lower()) or ("green tea" in assistant.lower())

    latency = time.time() - start
    return EvalResult(
        model=model,
        turns=turns,
        preference_turn=preference_turn,
        query_turn=query_turn,
        recalled=recalled,
        contradiction=contradiction,
        latency_s=round(latency, 3),
    )


def main():
    parser = argparse.ArgumentParser(description="Long-memory evaluation harness (baseline).")
    parser.add_argument("--model", default="mock/echo")
    parser.add_argument("--turns", type=int, default=80)
    parser.add_argument("--preference-turn", type=int, default=5)
    parser.add_argument("--query-turn", type=int, default=80)
    args = parser.parse_args()

    result = run_eval(
        model=args.model,
        turns=args.turns,
        preference_turn=args.preference_turn,
        query_turn=args.query_turn,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()

