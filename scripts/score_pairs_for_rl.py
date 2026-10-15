#!/usr/bin/env python3
"""Score preference pairs with the current best scorer to hunt "hard" RL data.

Goal
----
The user wants ~4k genuinely hard preference pairs to train RL.  Hard == pairs
that the current best model (stage6) does NOT confidently get right: it either
picks the loser, ties the two images, or (optionally) is only barely above the
decision boundary.  We score both images of every pair with the stage6
checklist model (3-dim v2 prompt, the SAME format stage6 was trained/evaled on)
and emit per-pair verdicts + difficulty so pairs can be ranked and sliced.

Two supported input formats (they are auto-detected by field names):

  1. gen / stage0-source pairs  (``data/raw/gen_sft_data.json``)
     JSON list, one object per pair:
       {instruction: <t2i prompt>, images: [imgA, imgB],
        label: "A"|"B"  (winner side), ...}
     ``images`` are relative to ``data/extracted/gen`` (image_root).

  2. HPDv3-PlusPlus pairs (https://huggingface.co/datasets/Junjun2333/HPDv3-PlusPlus)
     JSON list, one object per pair: {path1: <preferred>, path2: <non-preferred>,
     prompt: <text>}.  ``path1/path2`` are relative to the repo root after the
     image tar is extracted (they live under ``images/qwen_image/...``).
     Ready-to-use files: train/train_tf.json, train/train_aes.json,
     test/test_tf.json, test/test_aes.json  (the aes / tf axis is inferred
     from the file name).

Scoring prompt
--------------
stage6 is a *checklist* scorer (0-10 final_score + 0/1 eval_points).  It MUST be
prompted with the 3-dim v2 checklist prompt (the training/eval template in
``stage3/t2i_final_prompt_3dim_v2.jsonl``), NOT stage0's RubricRM 0-4 prompt
(that one was only used by the stage0 *model* when it annotated the corpus).

Per pair we keep raw score = final_score and, like the project convention,
``combined = final_score + 0.1 * pts_sum`` (pts_sum = number of satisfied
checklist points).  A prediction is "a"/"b"/"tie" from either signal and we mark
correct / wrong against the stored human/RubricRM preference.

Two modes
---------
  * scoring  (default): vLLM inference, 8-GPU shardable + resumable.
  * --merge  : read all shards, print per-file metrics (raw + combined), write
               ``analyzed.jsonl`` and the ranked hard set ``rl_hard.jsonl``,
               and optionally pick the top ``--top-k`` hard pairs.

Environment
-----------
Run with the interpreter that has vLLM + ms-swift + transformers 5.15:
    /usr/local/bin/python3
The stage6 checkpoint is a rank-32 LoRA; the engine already passes
``max_lora_rank=64`` (see scripts/eval_stage3.py).  If a batch ever OOMs set
``VLLM_ENFORCE_EAGER=1``.

Usage (run on /efs)
-------------------
  1) one-off download of HPDv3-PlusPlus (images ~268GB, done once):
     pip install -U "huggingface_hub[cli]"
     hf download Junjun2333/HPDv3-PlusPlus --repo-type dataset \
         --local-dir data/hpdv3pp
     cd data/hpdv3pp && cat images.tar.part* | tar -xf -
     # JSON pair files -> data/hpdv3pp/train|test/*.json
     # images          -> data/hpdv3pp/images/qwen_image/...

  2) quick smoke (a few pairs per source, one GPU):
     CUDA_VISIBLE_DEVICES=0 /usr/local/bin/python3 scripts/score_pairs_for_rl.py \
         --hpdv3 data/hpdv3pp/test/test_tf.json --hpdv3-root data/hpdv3pp \
         --limit 8 --out-dir output/rl_hard_sel_test

  3) full 8-GPU parallel scoring of the RL *candidate* pool
     (HPDv3++ train + gen pairs; keep test/ files for evaluation only):
     for i in 0 1 2 3 4 5 6 7; do
       CUDA_VISIBLE_DEVICES=$i /usr/local/bin/python3 scripts/score_pairs_for_rl.py \
           --hpdv3 data/hpdv3pp/train/train_tf.json \
           --hpdv3 data/hpdv3pp/train/train_aes.json \
           --hpdv3-root data/hpdv3pp \
           --gen data/raw/gen_sft_data.json --gen-root data/extracted/gen \
           --shard $i --num-shards 8 --out-dir output/rl_hard_sel \
           > logs/rl_hard_sel_shard_$i.log 2>&1 &
     done
     wait

  4) merge + analyse + pick the 4k hardest:
     /usr/local/bin/python3 scripts/score_pairs_for_rl.py --merge \
         --out-dir output/rl_hard_sel --top-k 4000

The pick logic (worst first): wrong (model prefers the loser on raw OR
combined) -> tie -> near-boundary (correct but margin <= --hard-margin), then
margin ascending inside each class.  Use ``--per-source-topk N`` to instead take
exactly N from every source/file (balanced pool), or ``--sample-per-file N`` in
scoring mode to first run a deterministic random N-sample of each file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmark_utils import compute_pairwise_metrics  # noqa: E402
from scripts.eval_stage3 import (  # noqa: E402
    build_messages,
    load_engine,
    load_template,
)

COMB_W = 0.1                      # combined = final_score + 0.1 * pts_sum
EPS = 1e-9

DEFAULT_CKPT = ROOT / "output/skyjm_gen_9b_stage6/v0-20260902-172103/checkpoint-250"
DEFAULT_TEMPLATE = ROOT / "stage3/t2i_final_prompt_3dim_v2.jsonl"
DEFAULT_GEN_ROOT = ROOT / "data/extracted/gen"
DEFAULT_HPDV3_ROOT = ROOT / "data/hpdv3pp"
DEFAULT_OUT = ROOT / "output/rl_hard_sel"


# --------------------------------------------------------------------------- #
# output parsing (mirrors scripts/recompute_benchmark_checklist.py)
# --------------------------------------------------------------------------- #
def parse_output_full(text: Optional[str]):
    """Return (final_score, pts_sum, n_pts) from a checklist JSON output."""
    if not text:
        return None, None, None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    fs = pts = npts = None
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
        except Exception:
            d = None
        if isinstance(d, dict):
            v = d.get("final_score")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                fs = float(v)
            pl = d.get("eval_points")
            if isinstance(pl, list):
                vals = [
                    p["score"] for p in pl
                    if isinstance(p, dict) and isinstance(p.get("score"), (int, float))
                    and not isinstance(p["score"], bool)
                ]
                if vals:
                    pts = float(sum(vals))
                    npts = len(vals)
    if fs is None:
        for pat in (r'"final_score"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)',
                    r'final_score\s*[:：]\s*"?([0-9]+(?:\.[0-9]+)?)'):
            mm = re.search(pat, s)
            if mm:
                fs = float(mm.group(1))
                break
    return fs, pts, npts


def _combined(fs: Optional[float], pts: Optional[float]) -> Optional[float]:
    if fs is None:
        return None
    return fs + COMB_W * (pts or 0.0)


def _resolve(rel: str, root: Path) -> str:
    p = Path(rel)
    return str(p if p.is_absolute() else root / p)


def _verdict(sa: Optional[float], sb: Optional[float]) -> Optional[str]:
    """Return 'a', 'b' or 'tie' (None when unparseable)."""
    if sa is None or sb is None:
        return None
    if abs(float(sa) - float(sb)) < EPS:
        return "tie"
    return "a" if float(sa) > float(sb) else "b"


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_gen_rows(files: list[Path], root: Path, sample: Optional[int],
                  seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    skipped = 0
    for f in files:
        if not f.is_file():
            raise SystemExit(f"ERROR: gen file not found: {f}")
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"ERROR: {f} is not a JSON list")
        pool = [r for r in data if isinstance(r, dict)]
        if sample is not None:
            pool = random.Random(seed + len(out)).sample(pool, min(sample, len(pool)))
        for i, r in enumerate(pool):
            imgs = r.get("images") or []
            label = r.get("label")
            prompt = r.get("instruction") or r.get("prompt") or ""
            if len(imgs) != 2 or label not in ("A", "B") or not prompt:
                skipped += 1
                continue
            if str(imgs[0]) == str(imgs[1]):
                skipped += 1
                continue
            chosen = "a" if label == "A" else "b"
            out.append({
                "source": "gen",
                "axis": "gen",
                "file": f.stem,
                "id": f"gen-{f.stem}-{i:06d}",
                "prompt": prompt,
                "image_a": _resolve(str(imgs[0]), root),
                "image_b": _resolve(str(imgs[1]), root),
                "img_a_rel": str(imgs[0]),
                "img_b_rel": str(imgs[1]),
                "chosen": chosen,
                "vote_label": "A>B" if chosen == "a" else "B>A",
                "extra": {
                    "classification_en": r.get("classification_en"),
                    "raw_label": label,
                },
            })
    if skipped:
        print(f"[gen] skipped {skipped} malformed/dup rows", flush=True)
    return out


def load_hpdv3_rows(files: list[Path], root: Path, preferred: int,
                    sample: Optional[int], seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    skipped = 0
    for f in files:
        if not f.is_file():
            raise SystemExit(f"ERROR: HPDv3++ file not found: {f}  "
                             "(download + extract images first)")
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"ERROR: {f} is not a JSON list")
        stem = f.stem.lower()
        axis = "aes" if "aes" in stem else ("tf" if "tf" in stem else stem)
        pool = [r for r in data if isinstance(r, dict)]
        if sample is not None:
            pool = random.Random(seed + len(out)).sample(pool, min(sample, len(pool)))
        for i, r in enumerate(pool):
            p1, p2 = r.get("path1"), r.get("path2")
            prompt = r.get("prompt") or r.get("instruction") or ""
            if not (p1 and p2 and prompt):
                skipped += 1
                continue
            if str(p1) == str(p2):
                skipped += 1
                continue
            if preferred == 1:
                image_a, image_b, chosen = str(p1), str(p2), "a"
            else:
                image_a, image_b, chosen = str(p2), str(p1), "b"
            out.append({
                "source": "hpdv3",
                "axis": axis,
                "file": f.stem,
                "id": f"hpdv3-{stem}-{i:06d}",
                "prompt": prompt,
                "image_a": _resolve(image_a, root),
                "image_b": _resolve(image_b, root),
                "img_a_rel": image_a,
                "img_b_rel": image_b,
                "chosen": chosen,
                "vote_label": "A>B" if chosen == "a" else "B>A",
                "extra": {},
            })
    if skipped:
        print(f"[hpdv3] skipped {skipped} malformed rows", flush=True)
    return out


def collect_pairs(args) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    if args.gen:
        pairs += load_gen_rows([Path(p) for p in args.gen], args.gen_root,
                               args.sample_per_file, args.seed)
    if args.hpdv3:
        pairs += load_hpdv3_rows([Path(p) for p in args.hpdv3], args.hpdv3_root,
                                 args.hpdv3_preferred, args.sample_per_file,
                                 args.seed)
    seen, uniq = set(), []
    for p in pairs:
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        uniq.append(p)
    return uniq


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def infer_texts(engine, request_config, messages_list: list) -> list[str]:
    from swift import InferRequest
    reqs = [InferRequest(messages=m) for m in messages_list]
    resp = engine.infer(reqs, request_config)
    out = []
    for r in resp:
        out.append(r.choices[0].message.content or "")
    return out


def score() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", action="append", default=[],
                    help="stage0-source pair JSON (RubricRM gen data); repeatable")
    ap.add_argument("--gen-root", type=Path, default=DEFAULT_GEN_ROOT,
                    help="root resolving the gen 'images/...' paths")
    ap.add_argument("--hpdv3", action="append", default=[],
                    help="HPDv3-PlusPlus pair JSON (path1/path2/prompt); repeatable")
    ap.add_argument("--hpdv3-root", type=Path, default=DEFAULT_HPDV3_ROOT,
                    help="root that contains images/ (path1/path2 resolve here)")
    ap.add_argument("--hpdv3-preferred", type=int, choices=[1, 2], default=1,
                    help="which path is preferred (default 1 = path1)")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                    help="scorer checkpoint (default: stage6 best raw ck250)")
    ap.add_argument("--full-model", action="store_true",
                    help="load --ckpt as a full (merged) model with no LoRA adapter")
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                    help="3-dim v2 checklist eval template jsonl")
    ap.add_argument("--engine", choices=["vllm", "transformers"], default="vllm")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="max pairs for this shard (quick checks)")
    ap.add_argument("--sample-per-file", type=int, default=None,
                    help="deterministically random-sample N pairs per input file "
                         "before sharding (first-version fast pass)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore already-scored rows in the output dir")
    ap.add_argument("--plan-only", type=Path, default=None,
                    help="dry-run: apply --sample-per-file, write the exact list "
                         "of image paths this run would need to PATH and exit "
                         "(extract only those from the tar, then run without this "
                         "flag to score)")
    ap.add_argument("--drop-output", action="store_true",
                    help="do not store the raw model JSON text per image "
                         "(saves disk; parsed scores are always stored)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--merge", action="store_true",
                    help="analyse mode: combine shards, metrics + hard-pair pick")
    ap.add_argument("--top-k", type=int, default=4000,
                    help="how many hard pairs to write in analyse mode")
    ap.add_argument("--per-source-topk", type=int, default=None,
                    help="analyse: take exactly N from each source/file "
                         "(balanced pool) instead of a global --top-k")
    ap.add_argument("--hard-margin", type=float, default=0.5,
                    help="combined/raw |diff| <= this counts as near-boundary")
    ap.add_argument("--include-easy", action="store_true",
                    help="analyse: also include easy pairs in rl_hard.jsonl")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        return analyse(args)

    if not args.gen and not args.hpdv3:
        ap.error("provide at least one --gen or --hpdv3 input file")

    pairs = collect_pairs(args)
    if not pairs:
        raise SystemExit("ERROR: no usable pairs after loading")

    if args.plan_only:
        paths = sorted({p["img_a_rel"] for p in pairs} |
                       {p["img_b_rel"] for p in pairs})
        args.plan_only.parent.mkdir(parents=True, exist_ok=True)
        args.plan_only.write_text("\n".join(paths) +
                                  ("\n" if paths else ""), encoding="utf-8")
        print(f"[plan] pairs={len(pairs)} unique_images={len(paths)} "
              f"-> {args.plan_only}", flush=True)
        print(f"[plan] extract ONLY these with (run from the HPDv3 repo root, "
              f"e.g. data/hpdv3pp):", flush=True)
        print(f"       cat images.tar.part* | tar -xf - -T {args.plan_only.name}",
              flush=True)
        return 0

    # image-existence filter (vLLM will fail on a missing file)
    ok_pairs, missing = [], 0
    for p in pairs:
        if os.path.isfile(p["image_a"]) and os.path.isfile(p["image_b"]):
            ok_pairs.append(p)
        else:
            missing += 1
    if missing:
        print(f"[score] {missing} pairs skipped (missing image file) "
              f"-- check --gen-root/--hpdv3-root", flush=True)
    if not ok_pairs:
        raise SystemExit("ERROR: none of the pairs have both images present")

    idx = list(range(len(ok_pairs)))
    idx = idx[args.shard::args.num_shards]
    if args.limit:
        idx = idx[: args.limit]
    by_source: dict[str, int] = {}
    for j in idx:
        by_source[ok_pairs[j]["source"]] = by_source.get(ok_pairs[j]["source"], 0) + 1
    print(f"[score] total_pairs={len(ok_pairs)} "
          f"shard={args.shard}/{args.num_shards} this_shard={len(idx)} "
          f"by_source={by_source} engine={args.engine}", flush=True)

    shard_fp = args.out_dir / f"shard_{args.shard}.jsonl"
    done_ids: set[str] = set()
    if not args.no_resume:
        for sf in sorted(args.out_dir.glob("shard_*.jsonl")):
            with sf.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        done_ids.add(json.loads(line)["id"])
        todo = [j for j in idx if ok_pairs[j]["id"] not in done_ids]
    else:
        todo = list(idx)
    print(f"[score] resume: {len(idx) - len(todo)} already scored, "
          f"{len(todo)} to run", flush=True)
    if not todo:
        print(f"[score] nothing to do for shard {args.shard}", flush=True)
        return 0

    system, user_body, tpl_prompt = load_template(args.template)
    engine, request_config = load_engine(
        args.ckpt, None, args.batch_size, args.max_new_tokens,
        engine_type=args.engine, tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        no_adapter=args.full_model)

    mode = "a" if (shard_fp.is_file() and not args.no_resume) else "w"
    with shard_fp.open(mode, encoding="utf-8") as f:
        n_done = n_parse = 0
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            msgs = []
            for j in chunk:
                p = ok_pairs[j]
                msgs.append(build_messages(system, user_body, tpl_prompt,
                                           p["prompt"], p["image_a"]))
                msgs.append(build_messages(system, user_body, tpl_prompt,
                                           p["prompt"], p["image_b"]))
            try:
                texts = infer_texts(engine, request_config, msgs)
            except Exception as ex:
                print(f"[score] batch error at {start}: {ex}", file=sys.stderr,
                      flush=True)
                continue
            for k, j in enumerate(chunk):
                p = ok_pairs[j]
                t_a, t_b = texts[2 * k], texts[2 * k + 1]
                fs_a, pts_a, np_a = parse_output_full(t_a)
                fs_b, pts_b, np_b = parse_output_full(t_b)
                row = {
                    "source": p["source"],
                    "axis": p["axis"],
                    "file": p["file"],
                    "id": p["id"],
                    "prompt": p["prompt"],
                    "image_a": p["image_a"],
                    "image_b": p["image_b"],
                    "chosen": p["chosen"],
                    "vote_label": p["vote_label"],
                    "score_a": fs_a,
                    "score_b": fs_b,
                    "pts_sum_a": pts_a,
                    "pts_sum_b": pts_b,
                    "n_pts_a": np_a,
                    "n_pts_b": np_b,
                    "comb_a": _combined(fs_a, pts_a),
                    "comb_b": _combined(fs_b, pts_b),
                }
                if not args.drop_output:
                    row["output_a"] = t_a
                    row["output_b"] = t_b
                if p.get("extra"):
                    row["extra"] = p["extra"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
                if fs_a is not None and fs_b is not None:
                    n_parse += 1
            f.flush()
            if n_done and n_done % (args.batch_size * 20) == 0:
                print(f"[score] shard {args.shard}: {n_done}/{len(todo)} "
                      f"parsed={n_parse}", flush=True)
    print(f"[score] shard {args.shard} complete: {n_done} pairs -> {shard_fp} "
          f"(both scores parsed on {100 * n_parse / max(1, n_done):.1f}%)",
          flush=True)
    return 0


# --------------------------------------------------------------------------- #
# analyse / merge
# --------------------------------------------------------------------------- #
def _enrich(row: dict[str, Any], hard_margin: float) -> dict[str, Any]:
    ch = row.get("chosen")
    fs_a, fs_b = row.get("score_a"), row.get("score_b")
    ca, cb = row.get("comb_a"), row.get("comb_b")
    raw_pred = _verdict(fs_a, fs_b)
    comb_pred = _verdict(ca, cb)
    raw_corr = None
    if ch in ("a", "b"):
        raw_corr = raw_pred in ("a", "b") and raw_pred == ch
        comb_corr = comb_pred in ("a", "b") and comb_pred == ch
    else:
        comb_corr = None
    margin_raw = abs(float(fs_a) - float(fs_b)) if fs_a is not None and fs_b is not None else None
    margin_comb = abs(float(ca) - float(cb)) if ca is not None and cb is not None else None
    margins = [m for m in (margin_raw, margin_comb) if m is not None]
    min_margin = min(margins) if margins else None

    if fs_a is None or fs_b is None:
        cat = "unparseable"
    elif comb_corr is False:
        cat = "wrong"
    elif (comb_pred == "tie") or (raw_pred == "tie"):
        cat = "tie"
    elif (raw_corr is False) or (min_margin is not None and min_margin <= hard_margin):
        cat = "near"
    else:
        cat = "easy"

    row = dict(row)
    row.update({
        "pred_raw": raw_pred,
        "pred_comb": comb_pred,
        "correct_raw": raw_corr,
        "correct_comb": comb_corr,
        "margin_raw": margin_raw,
        "margin_comb": margin_comb,
        "min_margin": min_margin,
        "difficulty": cat,
    })
    return row


def analyse(args) -> int:
    def fmt(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:.4f}"

    shards = sorted(args.out_dir.glob("shard_*.jsonl"))
    if not shards:
        raise SystemExit(f"ERROR: no shard_*.jsonl under {args.out_dir} -- "
                         "run scoring first")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sf in shards:
        with sf.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                rows.append(_enrich(r, args.hard_margin))
    print(f"[analyse] {len(rows)} unique scored pairs across {len(shards)} shards")
    if not rows:
        raise SystemExit("ERROR: no rows to analyse")

    # group keys: source / file (keep train vs test separate)
    def group(r):
        return r.get("source", "?") + ":" + r.get("file", "?")

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(group(r), []).append(r)

    report: dict[str, Any] = {}
    hard_pool: list[dict] = []
    for g, gs in sorted(groups.items()):
        m_raw = compute_pairwise_metrics(gs, score_a_key="score_a", score_b_key="score_b")
        m_comb = compute_pairwise_metrics(gs, score_a_key="comb_a", score_b_key="comb_b")
        n_eval = m_raw["n_evaluable"]
        wrong = sum(1 for r in gs if r.get("difficulty") == "wrong")
        tie = sum(1 for r in gs if r.get("difficulty") == "tie")
        near = sum(1 for r in gs if r.get("difficulty") == "near")
        report[g] = {
            "n": len(gs),
            "n_eval": n_eval,
            "raw": m_raw,
            "combined": m_comb,
            "difficulty": {"wrong": wrong, "tie": tie, "near": near,
                           "easy": sum(1 for r in gs if r["difficulty"] == "easy"),
                           "unparseable": sum(1 for r in gs if r["difficulty"] == "unparseable")},
        }
        print(f"[analyse] {g:24s} n={len(gs):6d}  "
              f"raw acc={fmt(m_raw['accuracy'])} "
              f"tie={fmt(m_raw['tie_rate'])} "
              f"no_tie={fmt(m_raw['accuracy_no_tie'])} | "
              f"comb acc={fmt(m_comb['accuracy'])} "
              f"tie={fmt(m_comb['tie_rate'])} "
              f"| wrong={wrong} tie={tie} near={near}", flush=True)
        hard_pool += [r for r in gs if r["difficulty"] != "easy" or args.include_easy]

    # rank worst-first: wrong > tie > near > easy; margin ascending within class
    order = {"wrong": 0, "tie": 1, "near": 2, "easy": 3, "unparseable": 4}
    hard_pool.sort(key=lambda r: (order.get(r["difficulty"], 4),
                                  (r["min_margin"] if r["min_margin"] is not None else 10.0)))

    with (args.out_dir / "analyzed.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (args.out_dir / "rl_hard.jsonl").open("w", encoding="utf-8") as f:
        for r in hard_pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # final selection
    if args.per_source_topk:
        pick: list[dict] = []
        for g, gs in groups.items():
            ranked = [r for r in hard_pool if group(r) == g]
            pick += ranked[: args.per_source_topk]
        title = f"per_source_top{args.per_source_topk}"
    else:
        pick = hard_pool[: args.top_k]
        title = f"top{args.top_k}"
    pick_fp = args.out_dir / f"rl_hard_{title}.jsonl"
    with pick_fp.open("w", encoding="utf-8") as f:
        for r in pick:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "report": report,
        "hard_pool_size": len(hard_pool),
        "picked": len(pick),
        "picked_file": str(pick_fp),
        "by_difficulty": {k: sum(1 for r in hard_pool if r["difficulty"] == k)
                          for k in ("wrong", "tie", "near", "easy", "unparseable")},
        "config": {"hard_margin": args.hard_margin, "top_k": args.top_k,
                   "per_source_topk": args.per_source_topk},
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"[analyse] hard candidates = {len(hard_pool)}; picked {len(pick)} -> {pick_fp}")
    print(f"[analyse] wrote analyzed.jsonl / rl_hard.jsonl / meta.json under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(score())
