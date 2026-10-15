#!/usr/bin/env python3
"""Annotate a (prompt, image) corpus with the stage0 pointwise reward model.

Stage0 (``skyjm_gen_9b_pointwise_stage0``) is the ORIGINAL RubricRM-style
pointwise scorer: it judges ONE image against the prompt with a 3-5 dimension
weighted rubric and emits a continuous ``Weighted Total Score`` on the **0-4**
scale.  It must be prompted with ITS OWN rubric prompt (NOT the 3-dim v2 0-10
checklist prompt used by stage3/4/5 / twostage) -- that prompt/format lives in
``scripts/benchmark_utils.py`` (``POINTWISE_SYSTEM`` / ``build_pointwise_messages``)
and mirrors the exact message layout stage0 was trained on.

The resulting stage0 scores feed ``build_stage6_fine_labels.py``, which aligns
them to the Gemini 0-10 scale and averages with Gemini's ``final_score`` to build
the fine-grained training target for stage6.

Usage (run on /efs where GPU + vLLM + the images live):
  # stage0's own merged checkpoint + its own 0-4 rubric prompt:
  python scripts/annotate_stage0_t2i.py \
      --source stage1/t2i_v4_20k_checklist_gemini_scored_clean.json \
      --ckpt  output/skyjm_gen_9b_pointwise_stage0/v3-20260818-224412/checkpoint-1500-merged \
      --engine vllm \
      --out   stage1/stage0_annotations.jsonl \
      > logs/stage0_annotate.log 2>&1

  # optional: parallel shards (8 GPUs, one per shard):
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i python scripts/annotate_stage0_t2i.py \
        --source ... --ckpt ... --engine vllm \
        --out stage1/stage0_annotations.jsonl --shard $i --num-shards 8 \
        > logs/stage0_annotate_shard_$i.log 2>&1 &
  done; wait
  python scripts/annotate_stage0_t2i.py --merge \
      --out stage1/stage0_annotations.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmark_utils import build_pointwise_messages, parse_total_score  # noqa: E402


def usable_rows(source: Path) -> list[dict[str, Any]]:
    """Rows that have a prompt, an image and a Gemini final_score (to align to later)."""
    data = json.loads(source.read_text(encoding="utf-8"))
    out = []
    for r in data:
        er = r.get("eval_result") or {}
        if r.get("prompt") and r.get("image") and isinstance(er.get("final_score"), (int, float)):
            out.append(r)
    return out


def resolve(path: str, image_root: Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else image_root / p)


def load_engine(ckpt: Path, engine_type: str, *, batch_size: int,
                tp: int = 1, gpu_mem: float = 0.9):
    """Full-model (no adapter) engine for a *merged* checkpoint, both backends."""
    import torch  # noqa: F401
    ckpt = ckpt.resolve()
    from swift import RequestConfig
    if engine_type == "vllm":
        import os
        from swift import VllmEngine
        engine = VllmEngine(
            str(ckpt),
            torch_dtype=torch.bfloat16,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem,
            max_num_seqs=2 * batch_size,
            max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384")),
            enforce_eager=os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
            limit_mm_per_prompt={"image": 1},
        )
    else:
        from swift import TransformersEngine
        engine = TransformersEngine(
            str(ckpt),
            torch_dtype=torch.bfloat16,
            max_batch_size=batch_size,
        )
    request_config = RequestConfig(max_tokens=1024, temperature=0, stop=["<|im_end|>"])
    return engine, request_config


def score_batch(engine, request_config, prompts: list[str], image_paths: list[str]):
    from swift import InferRequest
    reqs = [InferRequest(messages=build_pointwise_messages(p, img))
            for p, img in zip(prompts, image_paths)]
    resp_list = engine.infer(reqs, request_config)
    out = []
    for resp in resp_list:
        text = resp.choices[0].message.content
        out.append({"stage0_score": parse_total_score(text), "raw": text})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path,
                    default=ROOT / "stage1/t2i_v4_20k_checklist_gemini_scored_clean.json",
                    help="Gemini-scored source json (list) to annotate")
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "output/skyjm_gen_9b_pointwise_stage0"
                                   "/v3-20260818-224412/checkpoint-1500-merged")
    ap.add_argument("--out", type=Path, default=ROOT / "stage1/stage0_annotations.jsonl")
    ap.add_argument("--engine", choices=["vllm", "transformers"], default="vllm")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-root", type=Path, default=ROOT / "stage1" / "data",
                    help="root that resolves the source relative image paths "
                         "('images_v4/...' live under stage1/data)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="max rows for a quick test")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing shard output (default: skip rows already annotated)")
    ap.add_argument("--merge", action="store_true",
                    help="concatenate <out>.shard<i>.jsonl into <out>")
    args = ap.parse_args()

    if args.merge:
        parts = sorted(args.out.parent.glob(f"{args.out.name}.shard*.jsonl"))
        if not parts:
            print(f"ERROR: no {args.out.name}.shard*.jsonl found", file=sys.stderr)
            return 1
        with args.out.open("w", encoding="utf-8") as f:
            n = 0
            for p in parts:
                for line in p.open(encoding="utf-8"):
                    f.write(line)
                    n += 1
        print(f"[stage0-annotate] merged {n} rows -> {args.out}")
        return 0

    if not args.ckpt.is_dir():
        print(f"ERROR: stage0 ckpt not found: {args.ckpt}", file=sys.stderr)
        return 1
    rows = usable_rows(args.source)
    if not rows:
        print(f"ERROR: no usable rows in {args.source}", file=sys.stderr)
        return 1
    # deterministic shard split by row index
    idx = list(range(len(rows)))
    idx = idx[args.shard::args.num_shards]
    if args.limit:
        idx = idx[: args.limit]
    print(f"[stage0-annotate] rows total={len(rows)} shard={args.shard}/{args.num_shards} "
          f"this_shard_n={len(idx)} engine={args.engine}", flush=True)

    shard_out = args.out.parent / f"{args.out.name}.shard{args.shard}.jsonl"
    done = {}
    if not args.no_resume and shard_out.exists():
        for line in shard_out.open(encoding="utf-8"):
            rec = json.loads(line)
            done[rec["image"]] = rec
    todo = [rows[i] for i in idx if resolve(rows[i]["image"], args.image_root) not in done]
    print(f"[stage0-annotate] already done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print(f"[stage0-annotate] nothing to do for shard {args.shard}", flush=True)
        return 0

    engine, request_config = load_engine(
        args.ckpt, args.engine, batch_size=args.batch_size,
        tp=args.tp, gpu_mem=args.gpu_mem_util)
    ok = parse = 0
    with shard_out.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            prompts = [r["prompt"] for r in chunk]
            imgs = [resolve(r["image"], args.image_root) for r in chunk]
            try:
                results = score_batch(engine, request_config, prompts, imgs)
            except Exception as ex:  # keep partial progress on a bad batch
                print(f"[stage0-annotate] batch error at {start}: {ex}", file=sys.stderr, flush=True)
                continue
            for r, res in zip(chunk, results):
                rec = {
                    "prompt": r["prompt"],
                    "image": resolve(r["image"], args.image_root),
                    "image_rel": r["image"],
                    "gemini_final": r["eval_result"]["final_score"],
                    "stage0_score": res["stage0_score"],
                    "raw_len": len(res["raw"] or ""),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
                parse += 1 if res["stage0_score"] is not None else 0
            if (ok % (args.batch_size * 10)) == 0:
                print(f"[stage0-annotate] {ok}/{len(todo)} parsed={parse}", flush=True)
    print(f"[stage0-annotate] done {ok} rows -> {shard_out} "
          f"(parseable={parse}, pct={100 * parse / ok:.1f}% if ok)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
