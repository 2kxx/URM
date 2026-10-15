#!/usr/bin/env python3
"""Build a *learnable* mixed-difficulty pairwise dataset for the T2I judge RL.

Why
---
``rl_pairwise_train_allwrong.verl.jsonl`` (= the 4k hardest pairs) makes the
pairwise reward unlearnable: 38% of those pairs are exact ``final_score`` ties
and ~100% have ``score(winner) <= score(loser) + 0.5``, so with the reward
margin of 0.05 normalized (= 0.5 raw points) the winner-side rollout earns
R=1 on ~0% of them (see ``scripts/audit_rl_pairwise_data.py``).  The policy can
only fit noise -> scores inflate, ``pairwise_acc`` stays at ~0.3.

What this script does
---------------------
Selects a difficulty mix from the scored pool (``analyzed.jsonl``) instead:

  * ``near``       -- base judge is *correct but within the boundary band*
                      (gap <= 0.5 raw points).  RL target: widen the margin.
  * ``hard_wrong`` -- base judge confidently picks the loser (gap <= -0.5 raw).
                      RL target: flip it.  Pairs whose two scores are exactly
                      equal (pure coin flip) are dropped.
  * ``easy``       -- anchor pairs so the reward never saturates at 0/1 and the
                      format/score scale stays calibrated (kept a minority).

The validation split mirrors the same mix (and tags ``data_source`` per bucket,
so verl reports ``val-aux/urm_t2i_pairwise_{easy,near,hard}/...`` curves for
free).  Val pairs are held out from the train selection (id + prompt level).

Usage
-----
    python3 scripts/build_rl_pairwise_mixed.py \
        --analyzed output/rl_hard_sel/analyzed.jsonl \
        --out-dir  output/rl_mixed_sel/rl_data

Then point the launcher at the produced ``*.verl.jsonl`` files (see
``scripts/run_rl_v4_train.sh``).  Paths are rewritten under ``--image-root``
(default ``/efs/jacuzzitang/projects/URM``, i.e. the training node layout).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

DIFF_BUCKETS = ("easy", "near", "tie", "wrong")
VAL_DATA_SOURCE = {
    "easy": "urm_t2i_pairwise_easy",
    "near": "urm_t2i_pairwise_near",
    "hard_wrong": "urm_t2i_pairwise_hard",
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def signed_gap(row) -> float | None:
    sa, sb = row.get("score_a"), row.get("score_b")
    if sa is None or sb is None:
        return None
    chosen = str(row.get("chosen", "")).lower()
    if chosen == "a":
        return float(sa) - float(sb)
    if chosen == "b":
        return float(sb) - float(sa)
    return None


def resolve_image(path: str, image_root: str, old_root: str) -> str:
    """Rewrite a pool image path under ``image_root``.

    ``gen`` paths are relative (``data/extracted/gen/images/x.jpg``); HPDv3++
    paths are stored absolute under ``old_root``.  Both end up under
    ``image_root`` so the dataset can be rebuilt on a different node.
    """
    p = str(path)
    prefix = old_root.rstrip("/") + "/"
    if p.startswith(prefix):
        p = p[len(prefix):]
    if p.startswith("/"):
        return p  # foreign absolute path: leave untouched
    return str(Path(image_root) / p)


def classify(row, hard_wrong_gap: float):
    """Map a scored pair to a training bucket (``None`` = unusable)."""
    gap = signed_gap(row)
    if gap is None:
        return None, None
    diff = row.get("difficulty")
    if diff == "easy":
        return "easy", gap
    if diff == "near":
        return "near", gap
    if diff == "wrong":
        # exact ties carry no learnable direction -> drop
        if gap <= -hard_wrong_gap:
            return "hard_wrong", gap
        return None, gap
    return None, gap


def stratified_pick(candidates, n: int, rng: random.Random, source_weights=None):
    """Pick ``n`` candidates, keeping the pool's source mix (or explicit weights)."""
    by_source = collections.defaultdict(list)
    for row in candidates:
        by_source[row.get("source", "unknown")].append(row)
    for rows in by_source.values():
        rows.sort(key=lambda r: r["id"])
        rng.shuffle(rows)

    if not by_source:
        return []
    if source_weights:
        total = sum(source_weights.values())
        quota = {src: int(round(n * w / total)) for src, w in source_weights.items()}
    else:
        available = sum(len(v) for v in by_source.values())
        quota = {src: int(round(n * len(rows) / available)) for src, rows in by_source.items()}

    picked, leftover = [], []
    for src, rows in by_source.items():
        take = min(quota.get(src, 0), len(rows))
        picked.extend(rows[:take])
        leftover.extend(rows[take:])
    # backfill any shortfall from the remaining pool
    if len(picked) < n:
        leftover.sort(key=lambda r: r["id"])
        picked.extend(leftover[: n - len(picked)])
    return picked[:n]


def to_pair_record(row, difficulty: str, gap: float, image_root: str, old_root: str):
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "image_a": resolve_image(row["image_a"], image_root, old_root),
        "image_b": resolve_image(row["image_b"], image_root, old_root),
        "chosen": row["chosen"],
        "source": row.get("source", "unknown"),
        "axis": row.get("axis", ""),
        "file": row.get("file", ""),
        # ``difficulty`` keeps the pool taxonomy (easy/near/tie/wrong);
        # ``bucket`` is the training role (easy/near/hard_wrong).
        "difficulty": row.get("difficulty", ""),
        "bucket": difficulty,
        "base_gap": round(float(gap), 4),
    }


def to_verl_rows(pair, data_source: str, extra_extra=None):
    """Emit the A/B prompt rows the reward manager expects (winner = side A)."""
    a_img = pair["image_a"] if str(pair["chosen"]).lower() == "a" else pair["image_b"]
    b_img = pair["image_b"] if str(pair["chosen"]).lower() == "a" else pair["image_a"]
    base_extra = {
        "pair_id": pair["id"],
        "index": pair["id"],
        "source": pair["source"],
        "axis": pair["axis"],
        "file": pair["file"],
        "difficulty": pair["difficulty"],
        "bucket": pair["bucket"],
        "base_gap": pair["base_gap"],
    }
    if extra_extra:
        base_extra.update(extra_extra)
    rows = []
    for side, image in (("A", a_img), ("B", b_img)):
        rows.append(
            {
                "data_source": data_source,
                "instruction": pair["prompt"],
                "images": [image],
                "reward_model": {"ground_truth": "A>B"},
                "extra_info": {**base_extra, "side": side},
            }
        )
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyzed", default="output/rl_hard_sel/analyzed.jsonl")
    ap.add_argument("--out-dir", default="output/rl_mixed_sel/rl_data")
    ap.add_argument("--image-root", default="/efs/jacuzzitang/projects/URM",
                    help="node root the image paths are rewritten to")
    ap.add_argument("--old-root", default="/efs/jacuzzitang/projects/URM",
                    help="root already baked into the pool's absolute paths")
    ap.add_argument("--score-max", type=float, default=10.0)
    ap.add_argument("--hard-wrong-gap", type=float, default=0.5,
                    help="min |gap| (raw points) for a wrong pair to be usable")
    ap.add_argument("--train-near", type=int, default=2100)
    ap.add_argument("--train-hard-wrong", type=int, default=1500)
    ap.add_argument("--train-easy", type=int, default=600)
    ap.add_argument("--val-near", type=int, default=120)
    ap.add_argument("--val-hard-wrong", type=int, default=160)
    ap.add_argument("--val-easy", type=int, default=120)
    ap.add_argument("--exclude-pairs-file", default="output/rl_hard_sel/rl_data/pairs_val.jsonl",
                    help="pairs to keep out of the train selection (they are the old val set)")
    ap.add_argument("--allow-train-val-prompt-overlap", action="store_true",
                    help="by default train prompts that also appear in val are dropped")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    analyzed = Path(args.analyzed)
    if not analyzed.exists():
        print(f"ERROR: {analyzed} not found", file=sys.stderr)
        return 2

    excluded_ids: set[str] = set()
    ex_path = Path(args.exclude_pairs_file) if args.exclude_pairs_file else None
    if ex_path and ex_path.exists():
        excluded_ids = {row["id"] for row in read_jsonl(ex_path)}

    buckets = collections.defaultdict(list)
    for row in read_jsonl(analyzed):
        if row.get("id") in excluded_ids:
            continue
        bucket, gap = classify(row, args.hard_wrong_gap)
        if bucket is None:
            continue
        buckets[bucket].append((row, bucket, gap))

    val_easy = stratified_pick([r for r, _, _ in buckets["easy"]], args.val_easy, rng)
    val_near = stratified_pick([r for r, _, _ in buckets["near"]], args.val_near, rng)
    val_hard = stratified_pick([r for r, _, _ in buckets["hard_wrong"]], args.val_hard_wrong, rng)

    def gaps_of(rows):
        return {r["id"]: signed_gap(r) for r in rows}

    val_gaps = {}
    for rows in (val_easy, val_near, val_hard):
        val_gaps.update(gaps_of(rows))

    val_pairs = []
    for bucket, rows in (("easy", val_easy), ("near", val_near), ("hard_wrong", val_hard)):
        for row in rows:
            val_pairs.append(to_pair_record(row, bucket, val_gaps[row["id"]], args.image_root, args.old_root))

    # ---- train selection (val held out) ------------------------------------
    val_ids = {p["id"] for p in val_pairs}
    val_prompts = {p["prompt"] for p in val_pairs}

    pool = {"easy": buckets["easy"], "near": buckets["near"], "hard_wrong": buckets["hard_wrong"]}
    if not args.allow_train_val_prompt_overlap:
        for key, rows in pool.items():
            pool[key] = [r for r, _, _ in rows if r["id"] not in val_ids and r["prompt"] not in val_prompts]
    else:
        pool = {key: [r for r, _, _ in rows if r["id"] not in val_ids] for key, rows in pool.items()}

    train_gaps = {r["id"]: signed_gap(r) for bucket_rows in pool.values() for r in bucket_rows}
    train_pairs = []
    for bucket, quota in (("near", args.train_near), ("hard_wrong", args.train_hard_wrong), ("easy", args.train_easy)):
        picked = stratified_pick(pool[bucket], quota, rng)
        for row in picked:
            train_pairs.append(to_pair_record(row, bucket, train_gaps[row["id"]], args.image_root, args.old_root))

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    train_rows, val_rows = [], []
    for pair in train_pairs:
        train_rows.extend(to_verl_rows(pair, "urm_t2i_pairwise"))
    for pair in val_pairs:
        val_rows.extend(to_verl_rows(pair, VAL_DATA_SOURCE[pair["bucket"]]))

    out_dir = Path(args.out_dir)
    summary = {
        "analyzed": str(analyzed),
        "seed": args.seed,
        "hard_wrong_gap": args.hard_wrong_gap,
        "image_root": args.image_root,
        "excluded_pairs": len(excluded_ids),
        "pool_sizes_after_holdout": {k: len(v) for k, v in pool.items()},
        "train": {
            "pairs": len(train_pairs),
            "rows": len(train_rows),
            "by_bucket": dict(collections.Counter(p["bucket"] for p in train_pairs)),
            "by_source": dict(collections.Counter(p["source"] for p in train_pairs)),
        },
        "val": {
            "pairs": len(val_pairs),
            "rows": len(val_rows),
            "by_bucket": dict(collections.Counter(p["bucket"] for p in val_pairs)),
            "by_source": dict(collections.Counter(p["source"] for p in val_pairs)),
            "data_source_tags": VAL_DATA_SOURCE,
        },
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pairs in (("pairs_train.jsonl", train_pairs), ("pairs_val.jsonl", val_pairs)):
        with (out_dir / name).open("w", encoding="utf-8") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    with (out_dir / "rl_pairwise_train_mixed.verl.jsonl").open("w", encoding="utf-8") as fh:
        for row in train_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "rl_pairwise_val_mixed.verl.jsonl").open("w", encoding="utf-8") as fh:
        for row in val_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_dir}/rl_pairwise_train_mixed.verl.jsonl ({len(train_rows)} rows)")
    print(f"wrote {out_dir}/rl_pairwise_val_mixed.verl.jsonl ({len(val_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
