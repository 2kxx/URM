#!/usr/bin/env python3
"""Build the final ``middle + easy-anchor`` pairwise RL split.

Selection rule (signed gap = score(winner) - score(loser), raw 0-10 points):

* ``middle``      -- ``0 < gap <= --mid-gap-max``  (default 1.0)
* ``easy anchor`` -- ``gap > --anchor-gap-min``    (default 1.5)

Judge-wrong pairs and exact ties are intentionally excluded: the run keeps the
hard ``mean_threshold_format`` reward with ``THRESHOLD_MARGIN=0.05`` (0.5 raw
points), under which those buckets are mostly dead groups.

The script reuses the readers / converters from ``build_rl_pairwise_mixed.py``
so the output schema stays identical to the other ``*.verl.jsonl`` splits.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_rl_pairwise_mixed import (  # noqa: E402
    read_jsonl,
    signed_gap,
    stratified_pick,
    to_pair_record,
    to_verl_rows,
)


def _quantiles(values, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not values:
        return {}
    vals = sorted(float(v) for v in values)

    def pick(q):
        idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
        return vals[idx]

    return {str(q): pick(q) for q in qs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyzed", default="output/rl_hard_sel/analyzed.jsonl")
    ap.add_argument("--out-dir", default="output/rl_mid_anchor/rl_data")
    ap.add_argument("--image-root", default="/efs/jacuzzitang/projects/URM")
    ap.add_argument("--old-root", default="/efs/jacuzzitang/projects/URM")
    ap.add_argument("--exclude-pairs-file", default="output/rl_hard_sel/rl_data/pairs_val.jsonl",
                    help="pairs to keep out of train; their prompts are also held out by default")
    ap.add_argument("--allow-train-val-prompt-overlap", action="store_true")
    ap.add_argument("--mid-gap-max", type=float, default=1.0)
    ap.add_argument("--anchor-gap-min", type=float, default=1.5)
    ap.add_argument("--train-mid", type=int, default=2700)
    ap.add_argument("--train-anchor", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    analyzed = Path(args.analyzed)
    if not analyzed.exists():
        print(f"ERROR: {analyzed} not found", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    excluded_rows = []
    ex_path = Path(args.exclude_pairs_file) if args.exclude_pairs_file else None
    if ex_path and ex_path.exists():
        excluded_rows = list(read_jsonl(ex_path))
    excluded_ids = {row["id"] for row in excluded_rows}
    excluded_prompts = {row["prompt"] for row in excluded_rows}

    middle_pool, anchor_pool = [], []
    for row in read_jsonl(analyzed):
        row_id = row.get("id")
        if row_id in excluded_ids:
            continue
        if not args.allow_train_val_prompt_overlap and row.get("prompt") in excluded_prompts:
            continue
        gap = signed_gap(row)
        if gap is None:
            continue
        # Keep the original judge-verdict taxonomy honest: a handful of rows
        # tagged "wrong" have a positive score gap after vote remapping, so
        # they must not leak into the middle pool.
        if row.get("difficulty") not in ("near", "easy"):
            continue
        if 0.0 < gap <= args.mid_gap_max:
            middle_pool.append((row, gap))
        elif gap > args.anchor_gap_min:
            anchor_pool.append((row, gap))

    picked_mid = stratified_pick([r for r, _ in middle_pool], args.train_mid, rng)
    picked_anchor = stratified_pick([r for r, _ in anchor_pool], args.train_anchor, rng)
    gap_by_id = {r["id"]: g for r, g in middle_pool + anchor_pool}

    selected = []
    for bucket, rows in (("mid", picked_mid), ("easy", picked_anchor)):
        for row in rows:
            selected.append(to_pair_record(row, bucket, gap_by_id[row["id"]], args.image_root, args.old_root))
    rng.shuffle(selected)

    train_rows = []
    for pair in selected:
        train_rows.extend(to_verl_rows(pair, "urm_t2i_pairwise"))

    summary = {
        "analyzed": str(analyzed),
        "seed": args.seed,
        "image_root": args.image_root,
        "selection": {
            "mid": f"0 < gap <= {args.mid_gap_max}",
            "easy_anchor": f"gap > {args.anchor_gap_min}",
            "wrong_or_tie": "excluded",
        },
        "excluded_pairs": len(excluded_ids),
        "allow_train_val_prompt_overlap": bool(args.allow_train_val_prompt_overlap),
        "pool_sizes_after_holdout": {
            "mid": len(middle_pool),
            "easy": len(anchor_pool),
        },
        "pool_gap_quantiles": {
            "mid": _quantiles([g for _, g in middle_pool]),
            "easy": _quantiles([g for _, g in anchor_pool]),
        },
        "train": {
            "pairs": len(selected),
            "rows": len(train_rows),
            "by_bucket": dict(collections.Counter(p["bucket"] for p in selected)),
            "by_source": dict(collections.Counter(p["source"] for p in selected)),
            "by_original_difficulty": dict(collections.Counter(p["difficulty"] for p in selected)),
            "gap_quantiles": _quantiles([p["base_gap"] for p in selected]),
        },
        "val": {
            "pairs": 0,
            "rows": 0,
            "note": "use output/rl_hard_sel/rl_data/rl_pairwise_val.verl.jsonl",
        },
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pairs_train.jsonl").open("w", encoding="utf-8") as fh:
        for pair in selected:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    with (out_dir / "rl_pairwise_train_mixed.verl.jsonl").open("w", encoding="utf-8") as fh:
        for row in train_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Keep the same file layout as build_rl_pairwise_mixed.py; train uses the
    # shared hard-wrong val set, so these val files stay empty on purpose.
    (out_dir / "pairs_val.jsonl").write_text("", encoding="utf-8")
    (out_dir / "rl_pairwise_val_mixed.verl.jsonl").write_text("", encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_dir}/rl_pairwise_train_mixed.verl.jsonl ({len(train_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
