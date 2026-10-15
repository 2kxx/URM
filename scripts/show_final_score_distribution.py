#!/usr/bin/env python3
"""Print the ``final_score`` distribution of stage-3 SFT JSONL datasets."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


def extract_final_score(row: dict, path: Path, line_no: int) -> float:
    """Extract final_score from the assistant JSON stored in an SFT row."""
    if "final_score" in row:
        value = row["final_score"]
    else:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{path}:{line_no}: missing messages")
        assistant = next(
            (message for message in reversed(messages) if message.get("role") == "assistant"),
            None,
        )
        if assistant is None:
            raise ValueError(f"{path}:{line_no}: missing assistant message")
        content = assistant.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no}: assistant content is not valid JSON: {exc}"
                ) from exc
        if not isinstance(content, dict) or "final_score" not in content:
            raise ValueError(f"{path}:{line_no}: missing assistant final_score")
        value = content["final_score"]

    if isinstance(value, bool):
        raise ValueError(f"{path}:{line_no}: invalid final_score={value!r}")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{line_no}: invalid final_score={value!r}"
        ) from exc
    if not math.isfinite(score):
        raise ValueError(f"{path}:{line_no}: non-finite final_score={value!r}")
    return score


def load_scores(path: Path) -> list[float]:
    scores = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            scores.append(extract_final_score(row, path, line_no))
    if not scores:
        raise ValueError(f"{path}: no samples found")
    return scores


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile without third-party packages."""
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def format_score(score: float) -> str:
    return f"{score:g}"


def print_distribution(path: Path, scores: list[float], bar_width: int) -> None:
    counts = Counter(scores)
    maximum = max(counts.values())
    sorted_scores = sorted(scores)

    print(f"\n=== final_score distribution: {path} ===")
    print(
        "samples={n} mean={mean:.4f} std={std:.4f} min={minimum:g} "
        "p25={p25:.4f} median={median:.4f} p75={p75:.4f} max={maximum_score:g}".format(
            n=len(scores),
            mean=statistics.fmean(scores),
            std=statistics.pstdev(scores),
            minimum=sorted_scores[0],
            p25=percentile(sorted_scores, 0.25),
            median=percentile(sorted_scores, 0.50),
            p75=percentile(sorted_scores, 0.75),
            maximum_score=sorted_scores[-1],
        )
    )
    print(" score | count | percent | distribution")
    print("-------+-------+---------+----------------------------------------")
    for score in sorted(counts):
        count = counts[score]
        percent = 100 * count / len(scores)
        bar_len = max(1, round(bar_width * count / maximum))
        print(
            f"{format_score(score):>6} | {count:>5} | {percent:>6.2f}% | "
            f"{'#' * bar_len}"
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print exact-value final_score distributions from SFT JSONL files."
    )
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--bar-width", type=int, default=40)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.bar_width < 1:
        raise SystemExit("--bar-width must be >= 1")
    for dataset in args.datasets:
        print_distribution(dataset, load_scores(dataset), args.bar_width)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
