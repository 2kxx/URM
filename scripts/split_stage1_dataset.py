#!/usr/bin/env python3
"""Split the stage-1 dataset jsonl into train/val (prompt-leakage safe).

Grouping is done by prompt (all images sharing a prompt stay in the same split
to avoid prompt leakage), and groups are assigned to train/val stratified by
``dim`` with a deterministic seed -- same strategy as the stage-3 split
(``scripts/split_stage3_dataset.py``).

The build script skips source rows without a valid ``eval_result``, so the
jsonl is a subset of the stage-1 json. The exact source indices are read from
the sidecar written by ``scripts/build_stage1_dataset.py``
(``<output>.indices.json``) so the split stays perfectly aligned.

Usage:
    python scripts/split_stage1_dataset.py \
        --input stage1/t2i_v4_20k_final_prompt_3dim_v2.jsonl \
        --indices stage1/t2i_v4_20k_final_prompt_3dim_v2.jsonl.indices.json \
        --base stage1/t2i_v4_20k_checklist_gemini_scored_clean.json \
        --val-ratio 0.05 --seed 42 \
        --train-out stage1/t2i_v4_20k_final_prompt_3dim_v2_train.jsonl \
        --val-out   stage1/t2i_v4_20k_final_prompt_3dim_v2_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=ROOT / "stage1/t2i_v4_20k_final_prompt_3dim_v2.jsonl")
    ap.add_argument("--base", type=Path,
                    default=ROOT / "stage1/t2i_v4_20k_checklist_gemini_scored_clean.json")
    ap.add_argument("--indices", type=Path, default=None,
                    help="Sidecar with the source indices written by the build script "
                         "(default: <input>.indices.json)")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-out", type=Path,
                    default=ROOT / "stage1/t2i_v4_20k_final_prompt_3dim_v2_train.jsonl")
    ap.add_argument("--val-out", type=Path,
                    default=ROOT / "stage1/t2i_v4_20k_final_prompt_3dim_v2_val.jsonl")
    ap.add_argument("--val-indices-out", type=Path, default=None,
                    help="Sidecar with base indices of the val rows (default: <val-out>.indices.json); "
                         "used by scripts/eval_stage4.py")
    args = ap.parse_args()
    if args.indices is None:
        args.indices = Path(str(args.input) + ".indices.json")
    if args.val_indices_out is None:
        args.val_indices_out = Path(str(args.val_out) + ".indices.json")

    with args.base.open(encoding="utf-8") as f:
        base = json.load(f)
    with args.indices.open(encoding="utf-8") as f:
        indices = json.load(f)["indices"]

    rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if len(rows) != len(indices):
        ap.error(f"row count mismatch: jsonl={len(rows)} indices={len(indices)}")
    base_used = [base[i] for i in indices]
    if len(base_used) != len(rows):
        ap.error(f"base index out of range: base={len(base)} max_index={max(indices)}")

    # group row indices by prompt, keep dim info (base_used row-aligned with jsonl)
    prompt_groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(base_used):
        prompt_groups[r["prompt"]].append(i)

    prompts = sorted(prompt_groups.keys(), key=lambda p: (base_used[prompt_groups[p][0]]["dim"], p))
    rng = random.Random(args.seed)

    by_dim: dict[str, list[str]] = defaultdict(list)
    for p in prompts:
        by_dim[base_used[prompt_groups[p][0]]["dim"]].append(p)

    val_prompts: set[str] = set()
    for dim, plist in by_dim.items():
        n_rows = sum(len(prompt_groups[p]) for p in plist)
        quota = max(1, round(n_rows * args.val_ratio))
        rng.shuffle(plist)
        used_rows = 0
        for p in plist:
            if used_rows >= quota:
                break
            val_prompts.add(p)
            used_rows += len(prompt_groups[p])

    train_rows, val_rows = [], []
    for i, r in enumerate(rows):
        if base_used[i]["prompt"] in val_prompts:
            val_rows.append(r)
        else:
            train_rows.append(r)

    def write(path: Path, data: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(data)} rows -> {path}")

    write(args.train_out, train_rows)
    write(args.val_out, val_rows)
    val_indices = [indices[i] for i in range(len(rows)) if base_used[i]["prompt"] in val_prompts]
    with args.val_indices_out.open("w", encoding="utf-8") as f:
        json.dump({"source": str(args.base), "indices": val_indices}, f)
    print(f"val indices sidecar: {args.val_indices_out} ({len(val_indices)} rows)")

    print(f"total rows: {len(rows)}, unique prompts: {len(prompts)}")
    print(f"train: {len(train_rows)} rows")
    print(f"val:   {len(val_rows)} rows")
    print("val rows by dim:", dict(sorted(Counter(base_used[i]['dim'] for i in range(len(base_used))
                                                 if base_used[i]['prompt'] in val_prompts).items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
