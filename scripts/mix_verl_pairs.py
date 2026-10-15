#!/usr/bin/env python3
"""Merge several ``*.verl.jsonl`` pairwise splits into one, weight by source.

Why: the reward is GRPO-group-normalised, so only pairs whose group can
actually produce different rewards carry a gradient -- and which pairs those are
depends on the threshold margin (see the alive-rate table below).  Different
buckets therefore have to be mixed by hand, e.g. "judge-wrong dominant + a bit of
near/easy".

Behaviour:
  * dedupes pairs by ``extra_info.pair_id`` (first occurrence wins);
  * shuffles *pairs* (not rows), keeping the A/B rows adjacent -- required because
    the trainer slices batches of consecutive rows and the pairwise reward needs
    both sides of a pair in the same batch;
  * writes one file plus a short composition report (bucket / source / judge gap).

Usage:
  python3 scripts/mix_verl_pairs.py --in a.jsonl:3 --in b.jsonl:1 --out out.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_pairs(path: Path) -> dict[str, list[dict]]:
    pairs: dict[str, list[dict]] = collections.OrderedDict()
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        pid = (row.get("extra_info") or {}).get("pair_id") or row.get("index")
        pairs.setdefault(pid, []).append(row)
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="path[:weight] (repeatable); weight only affects how many pairs are taken"
                         " when --total is set")
    ap.add_argument("--total", type=int, default=0, help="cap the number of pairs (0 = all)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    chunks = []
    for spec in args.inputs:
        path_s, _, weight_s = spec.partition(":")
        path = Path(path_s)
        if not path.is_absolute():
            path = ROOT / path
        pairs = read_pairs(path)
        chunks.append((path, float(weight_s or 1.0), pairs))
        print(f"[mix] {path}: {len(pairs)} pairs (weight {weight_s or 1.0})")

    seen: set[str] = set()
    picked: list[list[dict]] = []
    for path, weight, pairs in chunks:
        take = list(pairs.items())
        rng.shuffle(take)
        if args.total:
            budget = max(1, int(round(args.total * weight / sum(w for _, w, _ in chunks))))
        else:
            budget = len(take)
        n = 0
        for pid, rows in take:
            if pid in seen or len(rows) != 2:
                continue
            seen.add(pid)
            picked.append(rows)
            n += 1
            if n >= budget:
                break
        print(f"[mix]   took {n} pairs from {path.name}")

    rng.shuffle(picked)
    rows = [row for pair in picked for row in pair]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    buckets = collections.Counter()
    sources = collections.Counter()
    gaps = []
    for pair in picked:
        extra = pair[0].get("extra_info") or {}
        buckets[extra.get("bucket") or extra.get("difficulty") or "?"] += 1
        sources[extra.get("source", "?")] += 1
        g = extra.get("judge_gap")
        if g is None:
            score = {}
            for row in pair:
                extra_i = row.get("extra_info") or {}
                score[extra_i.get("side")] = row
            g = extra.get("base_gap")
        gaps.append(abs(float(g)) if g is not None else 0.0)
    gaps.sort()
    med = gaps[len(gaps) // 2] if gaps else float("nan")
    print(f"[mix] wrote {out}  pairs={len(picked)} rows={len(rows)}")
    print(f"[mix] buckets={dict(buckets)}  sources={dict(sources)}  |judge_gap| median={med:.2f} raw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
