"""Shared helpers for the external benchmark pipeline (prepare + eval).

This module contains:
- the pointwise system prompt / message builder used at inference time
  (kept identical to ``scripts/build_pointwise.py`` so eval matches training),
- a robust parser for the model's ``Weighted Total Score`` output,
- the unified ``pairs.jsonl`` IO helpers,
- the pairwise preference metric computation,
- benchmark config loading from ``configs/benchmarks/*.json``.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Optional


# --------------------------------------------------------------------------- #
# Pointwise prompt (mirrors scripts/build_pointwise.py)
# --------------------------------------------------------------------------- #
POINTWISE_SYSTEM = (
    "You are an expert evaluator for text-to-image generation. Evaluate one "
    "candidate image against the user's generation prompt with a task-specific "
    "weighted rubric.\n\n"
    "Requirements:\n"
    "1. Select 3-5 atomic evaluation dimensions that best match the prompt.\n"
    "2. Assign weights summing to 100%.\n"
    "3. For each dimension, explain the image's strengths and weaknesses and "
    "give a score from 0 to 4.\n"
    "4. Compute the weighted total score on the 0-4 scale.\n"
    "5. Use the exact section structure shown in the training examples."
)

USER_TEMPLATE = "# Generation Prompt\n{prompt}\n\n# Candidate Image\n"


def build_pointwise_messages(prompt: str, image_path: str) -> list[dict[str, Any]]:
    """Build the messages list used to score one candidate image."""
    return [
        {"role": "system", "content": POINTWISE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_TEMPLATE.format(prompt=prompt)},
                {"type": "image", "image": image_path},
            ],
        },
    ]


# --------------------------------------------------------------------------- #
# Score parsing
# --------------------------------------------------------------------------- #
_TOTAL_WEIGHTED_RE = re.compile(
    r"Weighted\s+Total\s+Score\s*:.*?=\s*\*\*\s*([0-4](?:\.\d+)?)\s*/\s*4\s*\*\*",
    re.I | re.S,
)
_TOTAL_BARE_RE = re.compile(r"Weighted\s+Total\s+Score\s*:\s*([0-4](?:\.\d+)?)", re.I)
_SCORE_SLASH_RE = re.compile(r"Score\s*:\s*([0-4](?:\.\d+)?)\s*/\s*4", re.I)
_SCORE_BOLD_RE = re.compile(r"\*\*\s*([0-4](?:\.\d+)?)\s*/\s*4\s*\*\*")


def _section(text: str, heading: str) -> str:
    start = re.search(rf"(?im)^#{{2,4}}\s*\[{re.escape(heading)}\]\s*$", text)
    if not start:
        return ""
    end = re.search(r"(?im)^\s*#{{2,4}}\s*\[.*\]\s*$", text[start.end():])
    return text[start.end(): start.end() + (end.start() if end else len(text))].strip()


def parse_total_score(text: str) -> Optional[float]:
    """Extract the 0-4 weighted total score from a model rubric response.

    Tries, in order:
      1. "Weighted Total Score: ... = **X / 4**"   (canonical training format)
      2. "Weighted Total Score: X"
      3. the last "Score: X / 4" inside the Final Conclusion
      4. the last "**X / 4**" anywhere in the response
    """
    if not text:
        return None
    m = _TOTAL_WEIGHTED_RE.search(text)
    if m:
        return float(m.group(1))
    m = _TOTAL_BARE_RE.search(text)
    if m:
        return float(m.group(1))
    conclusion = _section(text, "Final Conclusion")
    if conclusion:
        m = list(_SCORE_SLASH_RE.finditer(conclusion))
        if m:
            return float(m[-1].group(1))
        m = list(_SCORE_BOLD_RE.finditer(conclusion))
        if m:
            return float(m[-1].group(1))
    m = list(_SCORE_SLASH_RE.finditer(text))
    if m:
        return float(m[-1].group(1))
    m = list(_SCORE_BOLD_RE.finditer(text))
    if m:
        return float(m[-1].group(1))
    return None


# --------------------------------------------------------------------------- #
# Unified pairs.jsonl IO
# --------------------------------------------------------------------------- #
def iter_pairs(pairs_path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(pairs_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_pairs(pairs_path: str | Path, rows: list[dict[str, Any]]) -> None:
    pairs_path = Path(pairs_path)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_meta(meta_path: str | Path, meta: dict[str, Any]) -> None:
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_pairwise_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    score_a_key: str = "score_a",
    score_b_key: str = "score_b",
) -> dict[str, Any]:
    """Compute pairwise preference accuracy from model scores.

    Each ``row`` is a unified pair dict with ``chosen`` in {"a", "b", None}
    and numeric ``score_a`` / ``score_b`` (or the key overrides).

    Reported metrics:
      - n, n_evaluable          total pairs / pairs with both scores + a label
      - n_ties                  pairs whose scores are exactly equal
      - accuracy                correct / n_evaluable (ties counted as wrong)
      - accuracy_no_tie         correct / (n_evaluable - n_ties)
      - accuracy_tie_half       (correct + 0.5 * n_ties) / n_evaluable
      - tie_rate                n_ties / n_evaluable
      - kendall_tau             rank correlation over scored pairs (ties -> 0.5)
    """
    correct = wrong = ties = 0
    scored = []
    for row in rows:
        chosen = row.get("chosen")
        sa = row.get(score_a_key)
        sb = row.get(score_b_key)
        if chosen not in ("a", "b") or sa is None or sb is None:
            continue
        diff = float(sa) - float(sb)
        if abs(diff) < 1e-9:
            ties += 1
        else:
            pred = "a" if diff > 0 else "b"
            correct += int(pred == chosen)
            wrong += int(pred != chosen)
        scored.append((chosen, sa, sb))

    n_evaluable = correct + wrong + ties
    n_ties = ties
    accuracy = (correct / n_evaluable) if n_evaluable else None
    accuracy_no_tie = (correct / (n_evaluable - n_ties)) if (n_evaluable - n_ties) else None
    accuracy_tie_half = ((correct + 0.5 * n_ties) / n_evaluable) if n_evaluable else None
    tie_rate = (n_ties / n_evaluable) if n_evaluable else None

    # Kendall's tau (pairwise comparison of model scores vs human preference)
    kt = None
    k = len(scored)
    if k >= 2:
        concord = discord = tie = 0
        for i in range(k):
            for j in range(i + 1, k):
                (c1, a1, b1), (c2, a2, b2) = scored[i], scored[j]
                h1 = 1 if c1 == "a" else -1
                h2 = 1 if c2 == "a" else -1
                s1 = 1 if (a1 - b1) > 0 else (-1 if (a1 - b1) < 0 else 0)
                s2 = 1 if (a2 - b2) > 0 else (-1 if (a2 - b2) < 0 else 0)
                if s1 == 0 or s2 == 0:
                    tie += 1
                elif s1 * h1 == s2 * h2:
                    concord += 1
                else:
                    discord += 1
        denom = concord + discord + tie
        if denom:
            kt = (concord - discord) / denom

    return {
        "n": len(scored),
        "n_evaluable": n_evaluable,
        "n_ties": n_ties,
        "n_correct": correct,
        "n_wrong": wrong,
        "accuracy": accuracy,
        "accuracy_no_tie": accuracy_no_tie,
        "accuracy_tie_half": accuracy_tie_half,
        "tie_rate": tie_rate,
        "kendall_tau": kt,
        "mean_score_a": statistics.fmean([r[1] for r in scored]) if scored else None,
        "mean_score_b": statistics.fmean([r[2] for r in scored]) if scored else None,
    }


def compute_genai_4way(
    rows: Iterable[dict[str, Any]],
    *,
    tie_gap: float = 0.25,
    bad_threshold: float = 1.0,
) -> dict[str, Any]:
    """Optional GenAI-Bench 4-way label accuracy (heuristic mapping).

    Maps the two scalar scores to the official labels:
      |score_a - score_b| <= tie_gap          -> A=B=Good if mean >= bad_threshold else A=B=Bad
      score_a > score_b (+gap)                -> A>B
      otherwise                               -> B>A
    This is a heuristic; pairwise_accuracy is the primary metric.
    """
    correct = total = 0
    by_label: dict[str, dict[str, int]] = {}
    for row in rows:
        sa, sb, gt = row.get("score_a"), row.get("score_b"), row.get("vote_label")
        if sa is None or sb is None or gt not in ("A>B", "B>A", "A=B=Good", "A=B=Bad"):
            continue
        total += 1
        diff = float(sa) - float(sb)
        mean = (float(sa) + float(sb)) / 2.0
        if abs(diff) <= tie_gap:
            pred = "A=B=Good" if mean >= bad_threshold else "A=B=Bad"
        else:
            pred = "A>B" if diff > 0 else "B>A"
        by_label.setdefault(gt, {"correct": 0, "total": 0})
        by_label[gt]["total"] += 1
        if pred == gt:
            by_label[gt]["correct"] += 1
            correct += 1
    return {
        "n": total,
        "accuracy_4way": (correct / total) if total else None,
        "per_label": by_label,
        "tie_gap": tie_gap,
        "bad_threshold": bad_threshold,
    }


def compute_rr_accuracy(
    rows: Iterable[dict[str, Any]],
    *,
    score_a_key: str = "score_a",
    score_b_key: str = "score_b",
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """RationalRewards-style 3-way pairwise accuracy (ties included as a class).

    Mirrors the baseline's ``reward_model_evaluation/modular_rm_eval/scoring.py``:
      - pred rank  = "A=B" if |score_a - score_b| <= tolerance else "A>B"/"B>A"
      - gt rank    = normalized from ``chosen`` ("a"->A>B, "b"->B>A, tie->A=B)
      - accuracy   = (pred == gt) / total, benchmark ties ARE a class (a model
        tie on a GT tie counts correct; a model tie on a preference pair counts
        wrong). Rows with unknown/None chosen or missing scores are skipped.
    """
    correct = total = skipped = 0
    for row in rows:
        chosen = row.get("chosen")
        sa = row.get(score_a_key)
        sb = row.get(score_b_key)
        if sa is None or sb is None:
            skipped += 1
            continue
        v = str(chosen).strip().lower() if chosen is not None else ""
        if v in ("a", "left", "leftvote", "1", "model_1"):
            gt = "A>B"
        elif v in ("b", "right", "rightvote", "2", "model_2"):
            gt = "B>A"
        elif v in ("tie", "draw", "equal", "a=b", "a=b=good", "a=b=bad"):
            gt = "A=B"
        else:
            skipped += 1
            continue
        d = float(sa) - float(sb)
        if abs(d) <= tolerance:
            pred = "A=B"
        else:
            pred = "A>B" if d > 0 else "B>A"
        total += 1
        if pred == gt:
            correct += 1
    return {
        "n": total,
        "skipped": skipped,
        "passed": correct,
        "accuracy": (correct / total) if total else None,
        "tolerance": tolerance,
    }


# --------------------------------------------------------------------------- #
# Benchmark config loading
# --------------------------------------------------------------------------- #
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "benchmarks"


def list_benchmarks() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def load_benchmark_config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No benchmark config at {path}. Available: {list_benchmarks()}"
        )
    return json.loads(path.read_text(encoding="utf-8"))
