#!/usr/bin/env python3
# Convert URM T2I pairwise rows (scripts/score_pairs_for_rl.py output, e.g.
# output/rl_hard_sel/rl_hard_top4000.jsonl or analyzed.jsonl) into the verl
# two-row-per-pair format consumed by the pairwise judge RL launcher.
#
# Input row (one per pair) expected fields:
#   id / prompt / image_a / image_b / chosen("a"|"b")   [+ source,file,axis]
#   (exactly what score_pairs_for_rl.py writes; image_a/b are absolute)
#
# Output: two JSONL rows per pair sharing extra_info.pair_id:
#   side=A row -> the winner image, side=B row -> the loser image.
#   The reward manager / GRPO grouping later compares the judge's final_score
#   between the two rows of a pair (positive side A = human/RubricRM preferred).
#
# Usage:
#   python3 examples/reward_function/convert_urm_t2i_pairwise_jsonl.py \
#       --input  /efs/jacuzzitang/projects/URM/output/rl_hard_sel/rl_hard_top4000.jsonl \
#       --output /efs/jacuzzitang/projects/URM/rl_data/rl_pairwise_4k_train.verl.jsonl
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SYSTEM_PROMPT_FILE = Path(__file__).resolve().with_name("t2i_3dim_v2_system_prompt.txt")
DEFAULT_USER_TEMPLATE_FILE = Path(__file__).resolve().with_name("t2i_3dim_v2_user_prompt_template.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=None,
                        help="prefix for relative image paths (default: use paths as-is)")
    parser.add_argument("--limit", type=int, default=None, help="max pairs")
    parser.add_argument("--embed_prompts", action="store_true",
                        help="embed system+user prompt into every row (default: rely on "
                             "data.system_prompt_file / data.user_prompt_template_file)")
    args = parser.parse_args()

    system_prompt = DEFAULT_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    user_tpl = DEFAULT_USER_TEMPLATE_FILE.read_text(encoding="utf-8").strip()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            src = json.loads(line)
            pair_id = src.get("id")
            instruction = src.get("prompt") or src.get("instruction")
            chosen = src.get("chosen")
            if not pair_id or not instruction or chosen not in ("a", "b"):
                print(f"skip {line_no}: missing id/prompt/chosen")
                continue
            if chosen == "a":
                winner, loser = src["image_a"], src["image_b"]
            else:
                winner, loser = src["image_b"], src["image_a"]

            def path_of(p: str) -> str:
                pp = Path(p)
                if pp.is_absolute() or args.image_root is None:
                    return str(pp)
                return str(args.image_root / pp)

            winner, loser = path_of(winner), path_of(loser)
            for side, img in (("A", winner), ("B", loser)):
                row = {
                    "data_source": "urm_t2i_pairwise",
                    "instruction": instruction,
                    "images": [img],
                    "reward_model": {"ground_truth": "A>B"},
                    "extra_info": {
                        "pair_id": str(pair_id),
                        "side": side,
                        "index": str(pair_id),
                        "source": src.get("source"),
                        "axis": src.get("axis"),
                        "file": src.get("file"),
                    },
                }
                if args.embed_prompts:
                    row["prompt"] = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_tpl.format(instruction=instruction)},
                    ]
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.limit is not None and n >= args.limit:
                break
    print(f"converted {n} pairs -> {n * 2} rows: {args.output}")


if __name__ == "__main__":
    main()
