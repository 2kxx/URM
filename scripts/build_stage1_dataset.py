#!/usr/bin/env python3
"""Build the stage-1 (expanded URM T2I v4) checklist SFT dataset (jsonl).

Adapts the previous stage-3 build pipeline (``scripts/build_stage3_dataset.py``)
to the newly expanded scored data in ``stage1/``:

    stage1/t2i_v4_20k_checklist_gemini_scored_clean.json

Each output row mimics ``stage3/t2i_final_prompt_3dim_v2.jsonl``:

    {
      "messages": [
        {"role": "system",    "content": "<3-dim v2 system prompt>"},
        {"role": "user",      "content": "<image>\\n...\\nText prompt: <prompt>\\n\\nGenerate ..."},
        {"role": "assistant", "content": "<label JSON: eval_points / dimension_summary / score_reason / final_score>"}
      ],
      "images": ["/abs/path/to/image.png"]
    }

* The system prompt and the user-content template (including the trailing
  ``Generate ...`` instruction) are taken verbatim from the 3-dim v2 template
  file (default ``stage3/t2i_final_prompt_3dim_v2.jsonl``) -- i.e. the latest
  3-dimension system prompt.
* One row is written per source record that carries a valid ``eval_result``
  (prompt + image + Gemini checklist label). Rows whose evaluation failed
  (``eval_error`` set / no ``eval_result``) are skipped and reported, so the
  emitted quantity is exactly aligned with the usable rows of the stage-1 file.
* ``images`` points at the downloaded image under ``--image-dir`` (default
  ``stage1/data``, the target of ``scripts/download_urm_t2i.sh``).

Usage:
    bash scripts/download_urm_t2i.sh          # first: fetch images into stage1/data
    python scripts/build_stage1_dataset.py     # -> stage1/t2i_v4_20k_final_prompt_3dim_v2.jsonl
    python scripts/split_stage1_dataset.py     # -> train/val jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl_first(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.loads(f.readline())


def load_json(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list (got {type(data).__name__})")
    return data


def extract_user_template(user_content: str) -> tuple[str, str]:
    """Return (user_template_with_placeholder, template_prompt).

    The template's example prompt is located and replaced with ``{prompt}`` so
    the surrounding phrasing is preserved exactly. Prompts may contain newlines,
    so we replace the prompt *string* rather than editing by line.
    """
    marker = "Text prompt: "
    idx = user_content.find(marker)
    if idx < 0:
        raise ValueError("Cannot find 'Text prompt: ' inside the template user content")
    prompt_start = idx + len(marker)
    rest = user_content[prompt_start:]
    nl = rest.find("\n")
    template_prompt = rest[:nl] if nl >= 0 else rest
    template_prompt = template_prompt.strip()
    if not template_prompt:
        raise ValueError("Could not extract the template's example prompt")
    user_template = user_content.replace(template_prompt, "{prompt}", 1)
    return user_template, template_prompt


def build_user_content(user_template: str, prompt: str) -> str:
    return user_template.replace("{prompt}", prompt, 1)


def build_label(er: dict) -> dict:
    """Build the assistant label from an ``eval_result`` dict."""
    return {
        "eval_points": er["eval_points"],
        "dimension_summary": er["dimension_summary"],
        "score_reason": er["score_reason"],
        "final_score": er["final_score"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the stage-1 T2I checklist SFT dataset from the expanded scored data."
    )
    ap.add_argument("--template", type=Path,
                    default=ROOT / "stage3/t2i_final_prompt_3dim_v2.jsonl",
                    help="Reference row whose system/user phrasing is reused "
                         "(default: stage3/t2i_final_prompt_3dim_v2.jsonl, the latest 3-dim prompt)")
    ap.add_argument("--source", type=Path,
                    default=ROOT / "stage1/t2i_v4_20k_checklist_gemini_scored_clean.json",
                    help="Stage-1 scored json (prompt + image + eval_result)")
    ap.add_argument("--image-dir", type=Path, default=ROOT / "stage1/data",
                    help="Directory that ``image`` paths are relative to "
                         "(default: stage1/data, target of scripts/download_urm_t2i.sh)")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "stage1/t2i_v4_20k_final_prompt_3dim_v2.jsonl",
                    help="Output jsonl")
    ap.add_argument("--indices-out", type=Path, default=None,
                    help="Sidecar json with the source indices written (default: "
                         "<output>.indices.json); used by split_stage1_dataset.py")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only build the first N rows (0 = all) for quick checks")
    args = ap.parse_args()

    if args.indices_out is None:
        args.indices_out = args.output.with_suffix(args.output.suffix + ".indices.json")

    template = load_jsonl_first(args.template)
    msgs = template.get("messages", [])
    system_content = next((m["content"] for m in msgs if m["role"] == "system"), None)
    user_content = next((m["content"] for m in msgs if m["role"] == "user"), None)
    if system_content is None or user_content is None:
        ap.error("Template must contain system and user messages")

    user_template, template_prompt = extract_user_template(user_content)
    src = load_json(args.source)

    n = len(src) if args.limit <= 0 else min(args.limit, len(src))
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped_no_label, skipped_missing_image = 0, 0, 0
    written_indices: list[int] = []
    dim_counter: Counter = Counter()
    with args.output.open("w", encoding="utf-8") as f:
        for i in range(n):
            s = src[i]
            er = s.get("eval_result")
            if not isinstance(er, dict):
                skipped_no_label += 1
                continue
            for key in ("eval_points", "dimension_summary", "score_reason", "final_score"):
                if key not in er or er[key] is None:
                    skipped_no_label += 1
                    break
            else:
                img = (args.image_dir / s["image"]).resolve()
                if not img.exists():
                    skipped_missing_image += 1
                    print(f"WARNING row {i}: image not found: {img}", file=sys.stderr)
                    continue
                row = {
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": build_user_content(user_template, s["prompt"])},
                        {"role": "assistant", "content": json.dumps(build_label(er), ensure_ascii=False, indent=2)},
                    ],
                    "images": [str(img)],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                written_indices.append(i)
                dim_counter[s.get("dim", "?")] += 1

    with args.indices_out.open("w", encoding="utf-8") as f:
        json.dump({"source": str(args.source), "indices": written_indices}, f)

    print(f"Template prompt: {template_prompt!r}")
    print(f"System content length: {len(system_content)} (3-dim v2)")
    print(f"Source rows inspected : {n}")
    print(f"Rows written          : {written}")
    print(f"Skipped (no label)    : {skipped_no_label}")
    print(f"Skipped (image missing): {skipped_missing_image}")
    print(f"Per-dim rows written  : {dict(sorted(dim_counter.items()))}")
    print(f"Output: {args.output}")
    print(f"Indices sidecar: {args.indices_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
