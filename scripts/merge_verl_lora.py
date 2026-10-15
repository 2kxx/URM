#!/usr/bin/env python3
"""Merge a verl LoRA-only FSDP checkpoint into a full HF model (CPU, no GPU).

verl saved the actor as PEFT-style LoRA (``base_model.model.*.lora_A/B.default.weight``)
sharded across ranks as DTensors (shard dim 0).  ``verl.model_merger`` does NOT
support LoRA-only checkpoints, so we gather the shards ourselves and add
``scaling * B@A`` to the corresponding base weight of ``--base``.

Usage:
    python3 scripts/merge_verl_lora.py \
        --actor <global_step_N>/actor \
        --base  output/skyjm_gen_9b_stage6/ck250-merged \
        --out   output/rl_t2i/merged_stepN
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch


def _local(t):
    """Return the local (unsharded-on-this-rank) data of a (D)tensor."""
    if hasattr(t, "_local_tensor"):
        return t._local_tensor
    if hasattr(t, "to_local"):
        try:
            return t.to_local()
        except Exception:
            return t
    return t


def load_lora(actor: Path):
    """Gather LoRA A/B across rank shards. Returns {target_weight: {"A":..., "B":...}}."""
    shards = sorted(actor.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        raise SystemExit(f"no rank shards under {actor}")
    world = None
    per_rank = []
    for p in shards:
        name = p.name
        # model_world_size_{W}_rank_{R}.pt
        parts = name.replace(".pt", "").split("_")
        w, r = int(parts[3]), int(parts[5])
        world = w if world is None else world
        per_rank.append((r, torch.load(p, map_location="cpu", weights_only=False)))
    per_rank.sort(key=lambda x: x[0])
    if len(per_rank) != world:
        print(f"warning: found {len(per_rank)} shards but world={world}")

    gathered: dict[str, dict[str, torch.Tensor]] = {}
    first_keys = per_rank[0][1].keys()
    for key in first_keys:
        if not (key.endswith("lora_A.default.weight") or key.endswith("lora_B.default.weight")):
            continue
        if not key.startswith("base_model.model."):
            continue
        # strip peft prefix / adapter suffix -> target weight key in the HF model
        mid = key[len("base_model.model."):].replace(".lora_A.default.weight", "").replace(
            ".lora_B.default.weight", "")
        tgt = mid + ".weight"
        pieces = [_local(per_rank[r][1][key]) for r, _ in enumerate(per_rank) if key in per_rank[r][1]]
        full = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        gathered.setdefault(tgt, {})
        if "lora_A" in key:
            gathered[tgt]["A"] = full
        else:
            gathered[tgt]["B"] = full
    print(f"gathered {len(gathered)} LoRA targets (rank shards={len(per_rank)})")
    return gathered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actor", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=None, help="lora_alpha (default from lora_train_meta.json or 2*r)")
    ap.add_argument("--rank", type=int, default=None, help="lora rank (default from lora_train_meta.json)")
    args = ap.parse_args()

    import json
    meta = {}
    mf = args.actor / "lora_train_meta.json"
    if mf.is_file():
        meta = json.loads(mf.read_text())
    r = int(args.rank if args.rank is not None else meta.get("r", 32))
    alpha = float(args.alpha if args.alpha is not None else meta.get("lora_alpha", 2 * r))
    scaling = alpha / r
    print(f"lora r={r} alpha={alpha} scaling={scaling}")

    from transformers import AutoModelForImageTextToText
    print(f"loading base {args.base} (CPU, bf16)...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    params = {n: p for n, p in model.named_parameters()}

    gathered = load_lora(args.actor)
    n_apply = 0
    missing = 0
    for tgt, ab in sorted(gathered.items()):
        if tgt not in params:
            missing += 1
            print(f"  WARN missing base weight: {tgt}")
            continue
        A, B = ab.get("A"), ab.get("B")
        if A is None or B is None:
            print(f"  WARN incomplete lora for {tgt}")
            continue
        delta = (B.float() @ A.float()) * scaling
        p = params[tgt]
        with torch.no_grad():
            p.data = (p.data.float() + delta).to(p.dtype)
        n_apply += 1
    print(f"applied LoRA to {n_apply} weights (missing={missing})")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    # copy tokenizer/processor files from base (config/weights already saved)
    for f in args.base.iterdir():
        if f.is_file() and (
            f.name.startswith("tokenizer") or f.name.startswith("special_tokens")
            or f.name.startswith("preprocessor") or f.name.startswith("vocab")
            or f.name == "merges.txt" or f.name.endswith(".tiktoken")
            or f.name.startswith("chat_template") or f.name == "generation_config.json"
            or f.name.startswith("added_tokens")
        ):
            dst = args.out / f.name
            if not dst.exists():
                shutil.copyfile(f, dst)
    print(f"saved merged model -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
