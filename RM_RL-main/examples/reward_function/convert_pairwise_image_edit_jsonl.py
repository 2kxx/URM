#!/usr/bin/env python3
# Copyright 2026 Individual Contributor: Pairwise image-edit reward example
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT_FILE = Path(__file__).resolve().with_name("pairwise_image_edit_stage3_system_prompt.txt")
DEFAULT_USER_TEMPLATE_FILE = Path(__file__).resolve().with_name("pairwise_image_edit_stage3_user_prompt_template.txt")

DEFAULT_USER_TEMPLATE = """<image><image>
The images are ordered as source image(s) followed by the edited image.
Editing instruction: {instruction}

Generate checklist-style evaluation points for this image-editing case, score each point as 0 or 1, assign a final score from 0 to 10, and output the JSON result.
"""


def build_prompt(instruction: str, system_prompt: str | None, user_template: str) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append(
        {
            "role": "user",
            "content": user_template.format(instruction=instruction).strip(),
        }
    )
    return messages


def load_user_template(user_template: str | None, user_template_file: Path | None, embed_user_prompt: bool) -> str | None:
    if user_template_file is not None:
        return user_template_file.read_text(encoding="utf-8")
    if user_template is not None:
        return user_template
    if embed_user_prompt:
        return DEFAULT_USER_TEMPLATE_FILE.read_text(encoding="utf-8")
    return None


def load_system_prompt(
    system_prompt: str | None, system_prompt_file: Path | None, embed_system_prompt: bool, no_system_prompt: bool
) -> str | None:
    if no_system_prompt:
        return None
    if system_prompt_file is not None:
        return system_prompt_file.read_text(encoding="utf-8")
    if system_prompt is not None:
        return system_prompt
    if embed_system_prompt:
        return DEFAULT_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    return None


def build_row(
    source: dict[str, Any], side: str, image_key: str, system_prompt: str | None, user_template: str | None
) -> dict[str, Any]:
    pair_id = source.get("id")
    if pair_id is None:
        raise ValueError("input row is missing required field: id")

    if side == "A":
        edited_image = source.get("winner_image")
        reference_score = source.get("winner_score")
    elif side == "B":
        edited_image = source.get("loser_image")
        reference_score = source.get("loser_score")
    else:
        raise ValueError(f"unsupported side: {side}")

    source_image = source.get("source_image")
    instruction = source.get("instruction")
    if source_image is None:
        raise ValueError(f"{pair_id}: missing source_image")
    if edited_image is None:
        raise ValueError(f"{pair_id}: missing edited image for side={side}")
    if instruction is None:
        raise ValueError(f"{pair_id}: missing instruction")

    pair_metadata = {
        "pair_id": str(pair_id),
        "side": side,
        "index": str(pair_id),
    }

    row = {
        "data_source": "pairwise_image_edit",
        "instruction": instruction,
        image_key: [source_image, edited_image],
        "reward_model": {"ground_truth": "A>B"},
        "extra_info": {
            **pair_metadata,
            "instruction": instruction,
            "source_image": source_image,
            "edited_image": edited_image,
            "reference_avg_score": reference_score.get("avg_score") if isinstance(reference_score, dict) else None,
            "reference_num_points": reference_score.get("num_points") if isinstance(reference_score, dict) else None,
        },
    }
    if user_template is not None or system_prompt is not None:
        if user_template is None:
            user_template = DEFAULT_USER_TEMPLATE_FILE.read_text(encoding="utf-8")
        row["prompt"] = build_prompt(instruction, system_prompt, user_template)
    return row


def convert_file(
    input_path: Path,
    output_path: Path,
    image_key: str,
    system_prompt: str | None,
    user_template: str | None,
    limit: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_pairs = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            source = json.loads(line)
            for side in ("A", "B"):
                fout.write(
                    json.dumps(build_row(source, side, image_key, system_prompt, user_template), ensure_ascii=False)
                    + "\n"
                )
            num_pairs += 1
            if limit is not None and num_pairs >= limit:
                break
    print(f"converted {num_pairs} pairs into {num_pairs * 2} rows: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one-row pairwise image-edit JSONL into verl RL JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image_key", default="images")
    parser.add_argument("--system_prompt", default=None, help="Inline system prompt override.")
    parser.add_argument("--system_prompt_file", type=Path, default=None)
    parser.add_argument(
        "--embed_system_prompt",
        action="store_true",
        help="Embed the default shared system prompt into every JSONL row.",
    )
    parser.add_argument("--no_system_prompt", action="store_true")
    parser.add_argument("--user_template", default=None, help="Inline user prompt template override.")
    parser.add_argument("--user_template_file", type=Path, default=None)
    parser.add_argument(
        "--embed_user_prompt",
        action="store_true",
        help="Embed the default user prompt template into every JSONL row.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    system_prompt = load_system_prompt(
        args.system_prompt, args.system_prompt_file, args.embed_system_prompt, args.no_system_prompt
    )
    user_template = load_user_template(args.user_template, args.user_template_file, args.embed_user_prompt)

    convert_file(
        input_path=args.input,
        output_path=args.output,
        image_key=args.image_key,
        system_prompt=system_prompt,
        user_template=user_template,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
