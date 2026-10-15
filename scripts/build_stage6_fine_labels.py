#!/usr/bin/env python3
"""Build stage6 fine-grained labels: average Gemini 0-10 with aligned stage0 0-4.

Motivation
----------
Gemini's ``final_score`` on the 20k T2I set lives on only ~25 values (mostly
0.5-step landmarks), so the scorer trained on it saturates and produces ties on
hard preference benchmarks.  Stage0 (RubricRM pointwise, 0-4 continuous) is an
INDEPENDENT second judge.  Averaging the two (after aligning stage0's 0-4 scale
to Gemini's 0-10) yields finer, non-fabricated labels that break Gemini's coarse
collisions -- while the checklist / dimensions / score_reason stay Gemini's.

CRITICAL: the emitted file is a deep copy of the source where ONLY
``eval_result["final_score"]`` is replaced.  Everything else is byte-identical,
so downstream builders (twostage / stage4-5 style) produce the same SFT rows
except for the final score.

Usage
-----
  python scripts/build_stage6_fine_labels.py \
      --gemini  stage1/t2i_v4_20k_checklist_gemini_scored_clean.json \
      --stage0  stage1/stage0_annotations.jsonl \
      --out     stage1/t2i_v4_20k_checklist_gemini_scored_stage6_fine.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemini", type=Path,
                    default=ROOT / "stage1/t2i_v4_20k_checklist_gemini_scored_clean.json")
    ap.add_argument("--stage0", type=Path,
                    default=ROOT / "stage1/stage0_annotations.jsonl")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "stage1/t2i_v4_20k_checklist_gemini_scored_stage6_fine.json")
    ap.add_argument("--weight-stage0", type=float, default=0.5,
                    help="weight of aligned stage0 in the average (rest is Gemini)")
    ap.add_argument("--align", choices=["affine", "none"], default="affine",
                    help="fit stage0 0-4 -> 0-10 with a least-squares affine map "
                         "before averaging")
    ap.add_argument("--no-clip", action="store_true", help="do not clip fine target to [0,10]")
    args = ap.parse_args()

    gem = json.loads(args.gemini.read_text(encoding="utf-8"))
    if not isinstance(gem, list):
        print(f"ERROR: {args.gemini} is not a JSON list", file=sys.stderr)
        return 1

    ann = {}
    for line in args.stage0.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        a = json.loads(line)
        if not isinstance(a.get("stage0_score"), (int, float)):
            continue
        # prefer the stable relative image id (matches source json exactly);
        # fall back to whichever absolute path annotate recorded
        key = a.get("image_rel") or a.get("image")
        if key:
            ann[key] = float(a["stage0_score"])

    def lookup_s0(img: str) -> Optional[float]:
        """Find stage0 score for a source image id.  Source 'image' is relative
        ('images_v4/...'); annotation records image_rel (relative) and image
        (absolute, resolved under the image root).  Match relative first, then
        the absolute variants that could have been used."""
        if img in ann:
            return ann[img]
        p = Path(img)
        if not p.is_absolute():
            for cand in (ROOT / img, ROOT / "stage1" / "data" / img,
                         ROOT / "stage1" / img):
                if str(cand) in ann:
                    return ann[str(cand)]
        return ann.get(str(p)) if p.is_absolute() else None

    # collect paired (gemini, stage0) on usable rows for alignment
    pairs: list[tuple[float, float]] = []
    for r in gem:
        er = r.get("eval_result") or {}
        gf = er.get("final_score")
        if not (r.get("prompt") and r.get("image") and isinstance(gf, (int, float))):
            continue
        s0 = lookup_s0(r["image"])
        if s0 is not None:
            pairs.append((float(gf), s0))
    print(f"[stage6] source rows={len(gem)} usable={len(pairs)} "
          f"stage0 coverage={100 * len(pairs) / max(1, len(gem)):.1f}%")
    if len(pairs) < 100:
        print("ERROR: too few stage0 annotations to align (run annotate_stage0_t2i.py first)",
              file=sys.stderr)
        return 1

    # alignment affine: stage0(x, 0-4) -> a*x + b on the 0-10 Gemini scale
    xs = [s for _, s in pairs]
    ys = [g for g, _ in pairs]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    corr = cov / math.sqrt(vx * (sum((y - my) ** 2 for y in ys) / n)) if vx else 0.0
    a_coef = cov / vx if vx else 0.0
    b_coef = my - a_coef * mx
    print(f"[stage6] corr(gemini,stage0)={corr:.3f}  align y={a_coef:.3f}*x + {b_coef:.3f}")
    if args.align == "none":
        a_coef, b_coef = 1.0, 0.0

    def align(x: float) -> float:
        return a_coef * x + b_coef

    def fine_of(gf: float, s0: float) -> float:
        aligned = align(s0)
        w = args.weight_stage0
        v = (1 - w) * gf + w * aligned
        if not args.no_clip:
            v = min(10.0, max(0.0, v))
        return round(v, 2)

    # ---- rebuild (ONLY eval_result.final_score changes) ----
    out = []
    n_changed = 0
    orig_new: list[tuple[float, float]] = []  # (gemini, fine) for relabeled rows
    for r in gem:
        er = r.get("eval_result")
        if not isinstance(er, dict):
            out.append(r)
            continue
        gf = er.get("final_score")
        if not isinstance(gf, (int, float)):
            out.append(r)
            continue
        s0 = lookup_s0(r["image"])
        if s0 is None:
            out.append(r)
            continue
        new_row = copy.deepcopy(r)
        fine = fine_of(float(gf), s0)
        new_row["eval_result"]["final_score"] = fine
        out.append(new_row)
        orig_new.append((float(gf), fine))
        n_changed += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- stats ----
    def label_stats(vals):
        c = {}
        for v in vals:
            c[round(v, 2)] = c.get(round(v, 2), 0) + 1
        half_grid = sum(cnt for v, cnt in c.items() if abs(v - round(v)) < 1e-9
                        or abs((v * 2) - round(v * 2)) < 1e-9)
        return len(c), half_grid / len(vals), c.get(10.0, 0) / len(vals)

    old = [p[0] for p in orig_new]
    new = [p[1] for p in orig_new]
    u_old, h_old, sat_old = label_stats(old)
    u_new, h_new, sat_new = label_stats(new)
    delta = [b - a for a, b in zip(old, new)]
    print(f"[stage6] relabeled {n_changed}/{len(out)} rows")
    print(f"[stage6] unique final values : {u_old} -> {u_new}")
    print(f"[stage6] on 0.5-grid (or int) : {100 * h_old:.1f}% -> {100 * h_new:.1f}%")
    print(f"[stage6] frac == 10 (saturation): {100 * sat_old:.1f}% -> {100 * sat_new:.1f}%")
    print(f"[stage6] mean |final change| : {statistics.mean(abs(x) for x in delta):.3f}")
    stats = {
        "n_source": len(gem), "n_relabeled": n_changed, "n_stage0": len(ann),
        "corr_gemini_stage0": corr, "align_a": a_coef, "align_b": b_coef,
        "weight_stage0": args.weight_stage0,
        "unique_before": u_old, "unique_after": u_new,
        "half_grid_rate_before": h_old, "half_grid_rate_after": h_new,
        "sat10_before": sat_old, "sat10_after": sat_new,
        "mean_abs_change": statistics.mean(abs(x) for x in delta),
    }
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage6] wrote {args.out} (+ {stats_path.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
