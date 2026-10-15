#!/usr/bin/env python3
"""Audit the pairwise-judge RL data and calibrate the threshold-reward margin/tau.

Motivation
----------
The stage6/URM T2I-judge GRPO run (``stage6_pairwise_rl_hard4k_*``) trains on
``output/rl_hard_sel/rl_data/rl_pairwise_train_allwrong.verl.jsonl`` -- the 4k
pairs the *base* judge already gets WRONG.  Training/val reward sits flat at
~0.4 and ``pairwise_acc`` never leaves ~0.28-0.38, and ``positive_mean`` ~
``negative_mean`` (the judge scores the preferred and the rejected image almost
identically).  Two independent causes:

  1. data: every pair is from the hard pool, and ~38% of the "wrong" pairs are
     exact score ties (the judge gives A and B the same ``final_score``), so the
     pairwise reward is a coin flip on noise;
  2. reward: ``mean_threshold_format`` is a hard 0/1 step and never uses ``tau``,
     so even a correct-but-close call yields no graded signal (and the default
     ``margin=0.05`` normalized == 0.5 raw points is larger than the whole
     near-boundary band).

This script quantifies both and prints a calibration table so ``margin``/``tau``
can be picked from the observed score-gap distribution instead of guessed.

Usage
-----
    python3 scripts/audit_rl_pairwise_data.py \
        --analyzed output/rl_hard_sel/analyzed.jsonl \
        --train    output/rl_hard_sel/rl_data/rl_pairwise_train_allwrong.verl.jsonl \
        --val      output/rl_hard_sel/rl_data/rl_pairwise_val.verl.jsonl \
        --out      output/rl_data_audit/audit.json

No GPU / no verl import needed (pure python + numpy).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

BUCKETS = ("easy", "near", "tie", "wrong")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _quantiles(values, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if len(values) == 0:
        return {p: None for p in ps}
    arr = np.asarray(values, dtype=float)
    return {p: float(np.quantile(arr, p)) for p in ps}


def signed_gap(row) -> float | None:
    """Signed judge gap ``score(winner) - score(loser)`` in raw 0-10 points.

    ``> 0`` -> the base judge already prefers the labelled winner (correct side).
    ``= 0`` -> exact tie in ``final_score`` (pure noise for a pairwise reward).
    """
    sa, sb = row.get("score_a"), row.get("score_b")
    if sa is None or sb is None:
        return None
    chosen = str(row.get("chosen", "")).lower()
    if chosen == "a":
        return float(sa) - float(sb)
    if chosen == "b":
        return float(sb) - float(sa)
    return None


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def bucket_table(rows, score_max: float):
    """Per-difficulty separatedness stats (gap in raw and normalized units)."""
    by_bucket = collections.defaultdict(list)
    for row in rows:
        gap = signed_gap(row)
        if gap is None:
            continue
        by_bucket[row.get("difficulty", "unknown")].append(gap)

    table = {}
    for bucket in BUCKETS + tuple(sorted(set(by_bucket) - set(BUCKETS))):
        gaps = by_bucket.get(bucket)
        if not gaps:
            continue
        arr = np.asarray(gaps, dtype=float)
        absq = _quantiles(np.abs(arr))
        table[bucket] = {
            "n": int(arr.size),
            "frac_below_or_equal_zero": float((arr <= 0).mean()),
            "frac_exact_tie": float((arr == 0).mean()),
            "gap": {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())},
            "abs_gap_quantiles": absq,
            "abs_gap_quantiles_normalized": {p: (v / score_max if v is not None else None) for p, v in absq.items()},
        }
    return table


def tau_calibration(gaps_norm: np.ndarray, taus, margins=(0.0, 0.01, 0.05)):
    """How much reward separation a *soft* threshold buys, per (tau, margin).

    For a pair whose normalized |gap| equals ``g``, the best case for GRPO is a
    rollout on the winner side and one on the loser side, giving the reward
    difference ``|sigmoid((g-m)/tau) - sigmoid((-g-m)/tau)|``.  We report it at
    the p50 / p90 of the observed |gap| distribution: too small -> the reward is
    noise-dominated (that is what ``tau=0.2`` does today); ~1 -> the reward is a
    hard step again (no gradient through the transition band).
    """
    p50 = float(np.quantile(gaps_norm, 0.5))
    p90 = float(np.quantile(gaps_norm, 0.9))
    out = []
    for tau in taus:
        for margin in margins:
            def sep(g):
                return abs(sigmoid((g - margin) / tau) - sigmoid((-g - margin) / tau))
            out.append(
                {
                    "tau": tau,
                    "tau_raw_points": tau * 10.0,
                    "margin_normalized": margin,
                    "root_gap_p50_normalized": p50,
                    "root_gap_p90_normalized": p90,
                    "separation_p50": sep(p50),
                    "separation_p90": sep(p90),
                    "sigmoid_slope_at_0": 1.0 / (4.0 * tau),
                }
            )
    return out, {"p50": p50, "p90": p90}


def load_label_margins(pointwise_dir: Path) -> dict[int, float]:
    """``pair_index -> |label score A - label score B|`` from the annotation.

    The label score is the RubricRM weighted total (0-4) that *produced* the
    label; its ``argmax`` is exactly ``chosen``.  It is a different quantity from
    the judge ``score_a``/``score_b`` in ``analyzed.jsonl``.
    """
    scores: dict[int, dict[str, float]] = collections.defaultdict(dict)
    for path in sorted(pointwise_dir.glob("*.jsonl")):
        for row in _read_jsonl(path):
            meta = row.get("metadata") or {}
            if "pair_index" in meta and "candidate" in meta and meta.get("pointwise_score") is not None:
                scores[int(meta["pair_index"])][str(meta["candidate"]).upper()] = float(meta["pointwise_score"])
    return {idx: abs(s["A"] - s["B"]) for idx, s in scores.items() if "A" in s and "B" in s}


def _gen_pair_index(row) -> int | None:
    if row.get("source") != "gen":
        return None
    try:
        return int(str(row["id"]).rsplit("-", 1)[1])
    except (KeyError, ValueError):
        return None


def label_margin_section(rows, label_margins, taus, band=(0.05, 1.25), max_gap_prefilter=True):
    """Difficulty defined by the LABEL score, not by the judge's own verdict.

    Also answers "what does the judge already do on label-hard pairs?" -- the
    cross-tab below is the reason a label-hard split is *not* the same animal as
    the model-hard 4k set.
    """
    joined = []
    for row in rows:
        if row.get("source") != "gen":
            continue
        margin = label_margins.get(_gen_pair_index(row))
        gap = signed_gap(row)
        if margin is None or gap is None:
            continue
        joined.append((margin, gap, row))

    margins = np.asarray([m for m, _, _ in joined], dtype=float)
    out: dict = {
        "joined_pairs": len(joined),
        "label_margin_quantiles": _quantiles(margins),
        "label_margin_ties": int((margins == 0).sum()),
        "cross_tab_judge_bucket": {},
        "bands": {},
    }
    for bucket in BUCKETS:
        vals = np.asarray([m for m, _, row in joined if row.get("difficulty") == bucket], dtype=float)
        if vals.size:
            out["cross_tab_judge_bucket"][bucket] = {"n": int(vals.size), "label_margin_median": float(np.median(vals))}

    for lo, hi in ((0.0, 0.25), (0.0, 0.5), band, (0.0, 2.0)):
        sub = [(m, g, row) for m, g, row in joined if lo <= m <= hi and (not max_gap_prefilter or abs(g) > 0)]
        if not sub:
            continue
        gaps_norm = np.asarray([abs(g) for _, g, _ in sub], dtype=float) / 10.0
        entry = {
            "n": len(sub),
            "judge_buckets": dict(collections.Counter(row.get("difficulty") for _, _, row in sub)),
            "judge_abs_gap_quantiles_raw": _quantiles(gaps_norm * 10.0),
            "judge_frac_gap_le_zero": float((np.asarray([g for _, g, _ in sub]) <= 0).mean()),
        }
        entry["tau_grid"] = [
            {
                "tau": tau,
                "tau_raw_points": tau * 10.0,
                "mean_separation": float(
                    np.mean([abs(sigmoid(g / tau) - sigmoid(-g / tau)) for g in gaps_norm])
                ),
                "median_separation": float(
                    np.median([abs(sigmoid(g / tau) - sigmoid(-g / tau)) for g in gaps_norm])
                ),
            }
            for tau in taus
        ]
        out["bands"][f"label_margin_{lo}_{hi}"] = entry
    return out


def audit_current_split(path: Path, name: str):
    """Bucket mix / tie rate of a materialized ``*.verl.jsonl`` split."""
    seen_pairs = set()
    seen_rows = 0
    sides = collections.Counter()
    buckets = collections.Counter()
    sources = collections.Counter()
    missing_keys = collections.Counter()
    for row in _read_jsonl(path):
        seen_rows += 1
        extra = row.get("extra_info") or {}
        seen_pairs.add(extra.get("pair_id", row.get("index")))
        sides[str(extra.get("side", "")).upper()] += 1
        buckets[extra.get("difficulty", "untagged")] += 1
        sources[extra.get("source", "unknown")] += 1
        for key in ("instruction", "images", "data_source", "reward_model"):
            if key not in row:
                missing_keys[key] += 1
    return {
        "name": name,
        "path": str(path),
        "rows": seen_rows,
        "pairs": len(seen_pairs),
        "sides": dict(sides),
        "buckets": dict(buckets),
        "sources": dict(sources),
        "rows_missing_keys": dict(missing_keys),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyzed", default="output/rl_hard_sel/analyzed.jsonl",
                    help="per-pair scored pool from scripts/score_pairs_for_rl.py")
    ap.add_argument("--train", default="output/rl_hard_sel/rl_data/rl_pairwise_train_allwrong.verl.jsonl")
    ap.add_argument("--val", default="output/rl_hard_sel/rl_data/rl_pairwise_val.verl.jsonl")
    ap.add_argument("--score-max", type=float, default=10.0, help="raw final_score upper bound")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.2, 0.1, 0.05, 0.03, 0.02, 0.01],
                    help="tau grid (normalized score units)")
    ap.add_argument("--out", default="output/rl_data_audit/audit.json")
    ap.add_argument("--pointwise-dir", default="data/processed/pointwise",
                    help="annotation dir with metadata.pointwise_score (the LABEL score); "
                         "used for the label-margin section")
    ap.add_argument("--label-margin-band", type=float, nargs=2, default=[0.05, 1.25],
                    help="the all-hard band used by scripts/build_rl_pairwise_hard.py")
    ap.add_argument("--no-split-audit", action="store_true")
    args = ap.parse_args(argv)

    analyzed_path = Path(args.analyzed)
    if not analyzed_path.exists():
        print(f"ERROR: {analyzed_path} not found", file=sys.stderr)
        return 2
    rows = list(_read_jsonl(analyzed_path))
    report: dict = {"analyzed": str(analyzed_path), "n_pairs_scored": len(rows), "score_max": args.score_max}

    buckets = bucket_table(rows, args.score_max)
    report["bucket_table"] = buckets

    # ---- the pool actually used for training / validation -------------------
    wrong = [r for r in rows if r.get("difficulty") == "wrong" and signed_gap(r) is not None]
    wrong_gaps = np.asarray([signed_gap(r) for r in wrong], dtype=float)
    report["wrong_pool"] = {
        "n": int(wrong_gaps.size),
        "frac_exact_tie": float((wrong_gaps == 0).mean()),
        "n_gap_le_-0.5": int((wrong_gaps <= -0.5).sum()),
        "n_gap_le_-1.0": int((wrong_gaps <= -1.0).sum()),
        "n_gap_le_-2.0": int((wrong_gaps <= -2.0).sum()),
        "source_of_confident_wrong": dict(
            collections.Counter(r.get("source", "?") for r in wrong if (signed_gap(r) or 0.0) <= -1.0)
        ),
    }

    # ---- tau / margin grid on the *learnable* buckets -----------------------
    learnable = [r for r in rows if r.get("difficulty") in ("near", "easy")]
    learnable_hard = [r for r in rows if r.get("difficulty") == "wrong" and (signed_gap(r) or 0.0) <= -1.0]
    gaps_learnable = np.asarray([abs(signed_gap(r)) for r in learnable], dtype=float) / args.score_max
    gaps_hard = np.asarray([abs(signed_gap(r)) for r in learnable_hard], dtype=float) / args.score_max
    cal_learnable, q_learnable = tau_calibration(gaps_learnable, args.taus)
    cal_hard, q_hard = tau_calibration(gaps_hard, args.taus) if gaps_hard.size else ([], {})
    report["tau_calibration"] = {
        "near+easy": {"gap_quantiles_normalized": q_learnable, "grid": cal_learnable},
        "confidently_wrong": {"gap_quantiles_normalized": q_hard, "grid": cal_hard},
    }

    # ---- what the CURRENT hard-step reward does on this pool ----------------
    cur = {"margin_normalized": 0.05, "tau_normalized": 0.2}
    hard_step = []
    for bucket in ("near", "wrong", "easy"):
        gaps = [signed_gap(r) for r in rows if r.get("difficulty") == bucket and signed_gap(r) is not None]
        if not gaps:
            continue
        arr = np.asarray(gaps, dtype=float) / args.score_max
        # "winner rollout gets reward 1" under the hard step with the current margin
        hard_step.append(
            {
                "bucket": bucket,
                "n": int(arr.size),
                "P_reward_1_hard_step_margin_0.05": float((arr > cur["margin_normalized"]).mean()),
                "P_reward_1_hard_step_margin_0.00": float((arr > 0.0).mean()),
                "mean_soft_reward_tau_0.2": float(np.mean([sigmoid(g / cur["tau_normalized"]) for g in arr])),
            }
        )
    report["current_reward_on_pool"] = {"config": cur, "buckets": hard_step}

    if not args.no_split_audit:
        splits = {}
        for name, path in (("train", args.train), ("val", args.val)):
            p = Path(path)
            splits[name] = audit_current_split(p, name) if p.exists() else {"name": name, "path": str(p), "missing": True}
        if splits.get("train", {}).get("rows") and splits.get("val", {}).get("rows"):
            def pair_ids(path):
                out = set()
                for row in _read_jsonl(Path(path)):
                    out.add((row.get("extra_info") or {}).get("pair_id"))
                return out
            train_ids, val_ids = pair_ids(args.train), pair_ids(args.val)
            splits["overlap_train_val_pairs"] = len(train_ids & val_ids)
        report["splits"] = splits

    # ---- label-score difficulty (annotation score, NOT the judge's verdict) --
    pointwise_dir = Path(args.pointwise_dir)
    if pointwise_dir.exists():
        label_margins = load_label_margins(pointwise_dir)
        if label_margins:
            section = label_margin_section(
                rows, label_margins, args.taus, band=tuple(args.label_margin_band)
            )
            report["label_margin"] = {"pointwise_dir": str(pointwise_dir), **section}

    # ---- human-readable summary -------------------------------------------
    print(f"pairs scored: {len(rows)}  ({analyzed_path})")
    print("\n[1] separatedness by difficulty (signed gap = score(winner) - score(loser), raw 0-10)")
    print(f"{'bucket':10s} {'n':>6s} {'gap<=0':>7s} {'tie':>6s} {'|gap| p25':>9s} {'p50':>6s} {'p75':>6s} {'p90':>6s}")
    for bucket in BUCKETS:
        st = buckets.get(bucket)
        if not st:
            continue
        q = st["abs_gap_quantiles"]
        print(f"{bucket:10s} {st['n']:6d} {st['frac_below_or_equal_zero']:7.3f} {st['frac_exact_tie']:6.3f} "
              f"{q[0.25]:9.2f} {q[0.5]:6.2f} {q[0.75]:6.2f} {q[0.9]:6.2f}")

    print("\n[2] the 4k 'all-wrong' train pool (and the 300-pair val) live in the 'wrong' row above:")
    wp = report["wrong_pool"]
    print(f"    n={wp['n']}  exact score ties={wp['frac_exact_tie']:.1%}  "
          f"gap<=-0.5: {wp['n_gap_le_-0.5']}  gap<=-1.0: {wp['n_gap_le_-1.0']}  gap<=-2.0: {wp['n_gap_le_-2.0']}")

    print("\n[3] tau calibration (normalized score units; multiply by score_max for raw points)")
    print(f"{'tau':>6s} {'tau_raw':>8s} {'margin':>7s} {'sep@p50':>8s} {'sep@p90':>8s}")
    for entry in cal_learnable:
        print(f"{entry['tau']:6.3f} {entry['tau_raw_points']:8.2f} {entry['margin_normalized']:7.3f} "
              f"{entry['separation_p50']:8.3f} {entry['separation_p90']:8.3f}")

    print("\n[4] current reward (hard step, margin=0.05, tau unused) on the same pool")
    for entry in hard_step:
        print(f"    {entry['bucket']:8s} n={entry['n']:6d}  P(R=1 | margin 0.05)={entry['P_reward_1_hard_step_margin_0.05']:.3f}"
              f"  (margin 0.0 -> {entry['P_reward_1_hard_step_margin_0.00']:.3f})"
              f"  soft(tau=0.2) mean={entry['mean_soft_reward_tau_0.2']:.3f}")

    lm = report.get("label_margin")
    if lm:
        print("\n[5] LABEL-score difficulty (annotation score margin, 0-4 scale)")
        print(f"    joined pairs={lm['joined_pairs']} of {len(rows)} pooled; label ties={lm['label_margin_ties']}")
        q = lm["label_margin_quantiles"]
        print(f"    label margin p10/p25/p50/p75/p90 = {q[0.1]:.2f}/{q[0.25]:.2f}/{q[0.5]:.2f}/{q[0.75]:.2f}/{q[0.9]:.2f}")
        print("    judge bucket (model-based) vs label margin:")
        for bucket, st in lm["cross_tab_judge_bucket"].items():
            print(f"      {bucket:8s} n={st['n']:6d}  label margin median={st['label_margin_median']:.2f}")
        for name, entry in lm["bands"].items():
            jb = ", ".join(f"{k}:{v}" for k, v in sorted(entry["judge_buckets"].items()))
            print(f"    {name:24s} n={entry['n']:5d}  judge buckets: {jb}")
            for grid in entry["tau_grid"]:
                print(f"        tau_raw={grid['tau_raw_points']:4.1f}  mean|sep|={grid['mean_separation']:.3f}"
                      f"  median|sep|={grid['median_separation']:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
