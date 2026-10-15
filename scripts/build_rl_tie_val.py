#!/usr/bin/env python3
"""Build a *tie-diagnostic* validation set: pairs the current judge cannot order.

The existing 300-pair val file was selected to have non-zero judge gaps, so it
cannot show whether RL fixes the actual complaint ("RM 打同分 -> tie") -- 63% of
the hard4k train file are exact ``final_score`` ties.  This script emits a
held-out val file made of the pool's ``tie`` bucket (``|score_a - score_b| <
1e-9`` or a combined-score tie), which the 4k train selection never touched,
tagged ``data_source=urm_t2i_pairwise_tie`` so verl logs
``val-aux/urm_t2i_pairwise_tie/pairwise_acc/mean@1`` next to the normal val.

Usage
-----
    python3 scripts/build_rl_tie_val.py --n 240
    # then, in the launcher: data.val_files="[$TEST_FILE,$TIE_VAL_FILE]"
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_image(path: str, image_root: str, old_root: str) -> str:
    p = str(path)
    prefix = old_root.rstrip("/") + "/"
    if p.startswith(prefix):
        p = p[len(prefix):]
    if p.startswith("/"):
        return p
    return str(Path(image_root) / p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyzed", default="output/rl_hard_sel/analyzed.jsonl")
    ap.add_argument("--out", default="output/rl_hard_sel/rl_data/rl_pairwise_val_tie.verl.jsonl")
    ap.add_argument("--train-file", default="output/rl_hard_sel/rl_data/rl_pairwise_train_allwrong.verl.jsonl")
    ap.add_argument("--buckets", nargs="+", default=["tie"],
                    help="pool buckets to draw from (tie = judge cannot order them)")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--image-root", default="/efs/jacuzzitang/projects/URM")
    ap.add_argument("--old-root", default="/efs/jacuzzitang/projects/URM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    used = set()
    train_path = Path(args.train_file)
    if train_path.exists():
        used = {row["extra_info"]["pair_id"] for row in read_jsonl(train_path)}
    old_val = Path(args.out).parent / "rl_pairwise_val.verl.jsonl"
    if old_val.exists():
        used |= {row["extra_info"]["pair_id"] for row in read_jsonl(old_val)}

    candidates = []
    for row in read_jsonl(Path(args.analyzed)):
        if row.get("id") in used:
            continue
        if row.get("difficulty") not in args.buckets:
            continue
        if row.get("score_a") is None or row.get("score_b") is None:
            continue
        candidates.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    picked = candidates[: args.n]
    if not picked:
        print("ERROR: no held-out tie pairs found", file=sys.stderr)
        return 2

    rows = []
    for row in picked:
        chosen = str(row["chosen"]).lower()
        imgs = {s: (row["image_a"] if chosen == "a" else row["image_b"]) for s in ("A",)}
        a_img, b_img = (row["image_a"], row["image_b"]) if chosen == "a" else (row["image_b"], row["image_a"])
        base = {
            "pair_id": row["id"],
            "index": row["id"],
            "source": row.get("source", "?"),
            "axis": row.get("axis", ""),
            "file": row.get("file", ""),
            "difficulty": row.get("difficulty", ""),
            "judge_gap": float(row["score_a"]) - float(row["score_b"]),
        }
        for side, image in (("A", a_img), ("B", b_img)):
            rows.append(
                {
                    "data_source": "urm_t2i_pairwise_tie",
                    "instruction": row["prompt"],
                    "images": [resolve_image(image, args.image_root, args.old_root)],
                    "reward_model": {"ground_truth": "A>B"},
                    "extra_info": {**base, "side": side},
                }
            )

    summary = {
        "out": args.out,
        "pairs": len(picked),
        "rows": len(rows),
        "buckets": dict(collections.Counter(r.get("difficulty") for r in picked)),
        "sources": dict(collections.Counter(r.get("source") for r in picked)),
        "held_out_from_train_pairs": len(used),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
