#!/usr/bin/env python
"""Smoke-test the stage-6 judge endpoint before starting an RL run.

Checks, in order:
  1. the endpoint answers ``/health`` and ``/v1/models`` with the expected model id;
  2. one real chat-completion round-trip returns the stage-6 JSON rubric;
  3. the reply parses into a ``final_score`` through the same code path the
     reward model uses (``flow_factory.rewards.stage6_judge_t2i``);
  4. the *actual* prompt/completion token counts are reported, so you can confirm
     ``--max-model-len`` and ``max_tokens`` are large enough.

Usage (with the Flow-Factory environment, on the training box):

    python sd35_rl/scripts/check_judge_server.py \
        --api-base-url http://127.0.0.1:8000/v1 --model stage6-judge \
        --image /path/to/one/sd35/sample.png

Exit code 0 = the judge is ready for RL. Non-zero = do not start training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from PIL import Image, ImageDraw


def _bypass_proxy_for(url: str) -> str:
    """Add the judge host to NO_PROXY.

    ``HTTP_PROXY`` is normally set for HuggingFace downloads, and the corporate
    ``no_proxy`` list does not always contain the loopback address, which makes
    localhost judge calls fail with a ProxyError.
    """
    host = urlparse(url).hostname or ""
    if not host:
        return ""
    for var in ("NO_PROXY", "no_proxy"):
        entries = [entry for entry in os.environ.get(var, "").split(",") if entry]
        if host not in entries:
            entries.append(host)
            os.environ[var] = ",".join(entries)
    return host


def _make_synthetic_image(size: int = 512) -> Image.Image:
    """Fallback image: a red circle on a blue background (clearly not a real sample)."""
    image = Image.new("RGB", (size, size), (40, 90, 200))
    draw = ImageDraw.Draw(image)
    draw.ellipse((size * 0.2, size * 0.2, size * 0.8, size * 0.8), fill=(220, 60, 40))
    return image


def _check_endpoint(base_url: str, expected_model: str, timeout: float) -> bool:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]

    ok = True
    try:
        health = requests.get(f"{root}/health", timeout=timeout)
        print(f"  /health            -> {health.status_code}")
    except requests.RequestException as exc:
        print(f"  /health            -> UNREACHABLE ({exc})")
        return False

    try:
        models = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        models.raise_for_status()
        served = [entry.get("id") for entry in models.json().get("data", [])]
        print(f"  /v1/models         -> {served}")
        if expected_model not in served:
            print(
                f"  ERROR: model id {expected_model!r} is not served. "
                f"Set VLM_MODEL to one of {served} (must equal vLLM --served-model-name)."
            )
            ok = False
    except requests.RequestException as exc:
        print(f"  /v1/models         -> FAILED ({exc})")
        ok = False
    return ok


async def _run_probe(args: argparse.Namespace, image: Image.Image, prompt: str) -> int:
    import base64
    import io
    import re

    from openai import AsyncOpenAI

    # Standalone-first: this script is the entry point for a plain "RM as a
    # server" checkout, so it must run without flow-factory installed. When the
    # flow-factory overlay IS importable we reuse it; otherwise the same
    # prompt/parsing logic is inlined below (kept intentionally small).
    try:
        from flow_factory.rewards.stage6_judge_t2i import (
            DEFAULT_ALLOWED_DIMENSIONS,
            DEFAULT_SYSTEM_PROMPT,
            DEFAULT_USER_TEMPLATE,
            count_eval_points,
            extract_final_score,
            extract_json_payload,
            resize_to_max_pixels,
        )
        from flow_factory.utils.image import pil_image_to_base64

        print("  protocol           : flow_factory.rewards.stage6_judge_t2i")
    except ImportError:
        from pathlib import Path

        protocol_dir = Path(__file__).resolve().parent.parent / "reward_protocol"
        DEFAULT_SYSTEM_PROMPT = (protocol_dir / "system_prompt.txt").read_text(encoding="utf-8").strip()
        DEFAULT_USER_TEMPLATE = (protocol_dir / "user_prompt_template.txt").read_text(encoding="utf-8").strip()
        DEFAULT_ALLOWED_DIMENSIONS = (
            "Prompt Alignment",
            "Aesthetics",
            "Technical Quality",
        )

        def pil_image_to_base64(img: Image.Image, format: str = "PNG") -> str:
            buf = io.BytesIO()
            img.save(buf, format=format)
            return f"data:image/{format.lower()};base64," + base64.b64encode(buf.getvalue()).decode()

        def resize_to_max_pixels(img: Image.Image, max_pixels):
            if not max_pixels or max_pixels <= 0:
                return img
            w, h = img.size
            if w * h <= max_pixels:
                return img
            import math

            scale = math.sqrt(max_pixels / float(w * h))
            new_w = max(32, int(w * scale) // 32 * 32)
            new_h = max(32, int(h * scale) // 32 * 32)
            return img if (new_w, new_h) == (w, h) else img.resize((new_w, new_h), Image.LANCZOS)

        _FENCE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)
        _THINK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
        _FINAL = re.compile(r'"final_score"\s*:\s*(?P<value>"?[^",}\s]+"?)')
        _LOOSE = re.compile(
            r"final[_ ]score\s*[\"']?\s*[:=]\s*[\"']?\s*(?P<value>-?\d+(?:\.\d+)?)", re.IGNORECASE
        )

        def extract_json_payload(text: str):
            import json

            if not text:
                return None
            for candidate in (_THINK.sub("", text), text):
                stripped = candidate.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        return parsed
            for match in _FENCE.finditer(text):
                try:
                    parsed = json.loads(match.group("body").strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
            return None

        def extract_final_score(text: str, payload=None):
            import json
            import math

            if payload is None:
                payload = extract_json_payload(text)
            if isinstance(payload, dict):
                raw = payload.get("final_score")
                if isinstance(raw, (int, float)):
                    return float(raw), "json"
                if isinstance(raw, str):
                    m = re.search(r"-?\d+(?:\.\d+)?", raw)
                    if m:
                        return float(m.group(0)), "json"
            if text:
                for pattern in (_FINAL, _LOOSE):
                    hits = list(pattern.finditer(text))
                    if hits:
                        m = re.search(r"-?\d+(?:\.\d+)?", hits[-1].group("value"))
                        if m and math.isfinite(float(m.group(0))):
                            return float(m.group(0)), "regex"
            return None, "missing"

        def count_eval_points(payload, allowed_dimensions=DEFAULT_ALLOWED_DIMENSIONS):
            if not isinstance(payload, dict):
                return 0, 0
            points = payload.get("eval_points")
            if not isinstance(points, list):
                return 0, 0
            allowed = {d.strip().lower() for d in allowed_dimensions} if allowed_dimensions else None
            violations = 0
            if allowed is not None:
                for point in points:
                    label = point.get("dimension") if isinstance(point, dict) else None
                    if not isinstance(label, str) or label.strip().lower() not in allowed:
                        violations += 1
            return len(points), violations

        print(f"  protocol           : {protocol_dir} (standalone)")

    resized = resize_to_max_pixels(image, args.image_max_pixels)
    print(
        f"  image              -> {image.size} -> {resized.size} "
        f"(cap {args.image_max_pixels} px, "
        f"~{(resized.size[0] // 32) * (resized.size[1] // 32)} vision tokens)"
    )

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": pil_image_to_base64(resized, format="PNG")},
                },
                {"type": "text", "text": DEFAULT_USER_TEMPLATE.format(instruction=prompt)},
            ],
        },
    ]

    failures = 0
    extra_body: dict = {}
    if args.chat_template_kwargs:
        extra_body["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    if args.no_repeat_ngram_size:
        extra_body["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    async with AsyncOpenAI(base_url=args.api_base_url, api_key=args.api_key) as client:
        for trial in range(args.repeat):
            started = time.perf_counter()
            try:
                request: dict = dict(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    frequency_penalty=args.frequency_penalty,
                    timeout=args.timeout,
                )
                if extra_body:
                    request["extra_body"] = extra_body
                completion = await client.chat.completions.create(**request)
            except Exception as exc:  # noqa: BLE001
                print(f"  trial {trial + 1}: REQUEST FAILED ({type(exc).__name__}: {exc})")
                failures += 1
                continue

            latency = time.perf_counter() - started
            choice = completion.choices[0]
            text = choice.message.content or ""
            usage = completion.usage
            payload = extract_json_payload(text)
            score, source = extract_final_score(text, payload)
            points, violations = count_eval_points(payload, DEFAULT_ALLOWED_DIMENSIONS)
            finish = choice.finish_reason

            print(f"  trial {trial + 1}:")
            print(f"    latency            : {latency:.2f}s")
            print(f"    sent max_tokens    : {args.max_tokens} "
                  f"(temperature={args.temperature}, frequency_penalty={args.frequency_penalty})")
            print(f"    prompt_tokens      : {getattr(usage, 'prompt_tokens', '?')}")
            print(f"    completion_tokens  : {getattr(usage, 'completion_tokens', '?')} "
                  f"(cap {args.max_tokens})")
            print(f"    finish_reason      : {finish}")
            print(f"    reply length       : {len(text)} chars")
            print(f"    eval_points        : {points} (off-taxonomy: {violations})")
            print(f"    final_score        : {score} (source={source})")
            print(f"    reward             : "
                  f"{None if score is None else round(score / 10.0, 4)}")

            prompt_tokens = getattr(usage, "prompt_tokens", None)
            if prompt_tokens:
                needed = prompt_tokens + args.max_tokens
                print(f"    max-model-len need : >= {needed} "
                      f"(prompt {prompt_tokens} + completion {args.max_tokens})")

            if score is None:
                print("    VERDICT            : FAIL (no final_score; reward would "
                      "fall back to 0.0)")
                if args.dump_raw:
                    print("    ---- full raw reply ----")
                    print(text)
                    print("    ---- end raw reply ----")
                else:
                    print(f"    raw reply head     : {json.dumps(text[:300])}")
                    print(f"    raw reply tail     : {json.dumps(text[-300:])}")
                failures += 1
            elif finish == "length":
                print("    VERDICT            : WARN (hit max_tokens; score was rescued, "
                      "but raise --max-model-len/max_tokens or the repetition penalty)")
            else:
                print("    VERDICT            : OK")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="stage6 judge endpoint smoke test")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="stage6-judge")
    parser.add_argument("--image", default=None, help="a real generated image (PNG/JPG)")
    parser.add_argument("--prompt", default="a photo of a yellow bus and an orange handbag")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="keep 0.0 for this judge: a non-zero penalty also hits JSON syntax "
        "tokens and makes replies abort mid-JSON",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--skip-http-check", action="store_true")
    parser.add_argument(
        "--dump-raw",
        action="store_true",
        help="print the complete judge reply (use this when parsing fails)",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        default=None,
        help='JSON passed to the chat template, e.g. \'{"enable_thinking": false}\'',
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="vLLM extra_body knob; 0 disables",
    )
    args = parser.parse_args()

    print("=" * 72)
    print(f"stage6 judge smoke test -> {args.api_base_url} (model={args.model})")
    print("=" * 72)
    print(f"  NO_PROXY +{_bypass_proxy_for(args.api_base_url)} "
          f"(env NO_PROXY={os.environ.get('NO_PROXY', '')})")

    if not args.skip_http_check and not _check_endpoint(
        args.api_base_url, args.model, args.timeout
    ):
        print("ENDPOINT NOT READY")
        return 2

    if args.image:
        image = Image.open(args.image).convert("RGB")
        print(f"  using real image   : {args.image}")
    else:
        image = _make_synthetic_image()
        print("  using SYNTHETIC image (pass --image <sd35 sample> to test the real path)")

    failures = asyncio.run(_run_probe(args, image, args.prompt))
    print("=" * 72)
    if failures:
        print(f"RESULT: {failures} failing probe(s) -- fix the judge before training")
        return 1
    print("RESULT: judge is ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
