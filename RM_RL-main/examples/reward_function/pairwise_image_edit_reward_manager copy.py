# Copyright 2026 Individual Contributor: Pairwise image-edit reward example
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


@dataclass
class ParsedRollout:
    score: float | None
    format_reward: float
    valid_score: bool
    score_count: int


@dataclass
class ScoredRollout:
    idx: int
    pair_id: Any
    side: str
    parsed: ParsedRollout


class PairwiseImageEditRewardManager(RewardManagerBase):
    """Batch reward manager for pairwise image-edit checkpoint scoring.

    Expected dataset metadata:
      - ``extra_info[pair_id_key]`` or top-level ``index`` identifies the A/B pair.
      - ``extra_info[side_key]`` is ``"A"`` for the preferred edit and ``"B"`` for the worse edit.

    Expected rollout text:
      - JSON object containing checkpoint/eval-point scores, e.g.
        ``{"checkpoints": [{"score": 8}, {"score": 7}]}``.
      - Scores are normalized to ``[0, 1]``; values above 1 are divided by ``score_max``.

    Reward:
      - Valid A_i rollouts compare against valid B_j rollouts from the same pair.
      - The pairwise term uses the Bradley-Terry log-likelihood:
        ``log sigmoid((s_A - s_B - margin) / tau)``.
      - Invalid rollouts do not enter pairwise comparisons and only receive format reward.
      - ``R_i = R_bt_i + format_bonus * R_format_i``.
    """

    batch_reward = True

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        reward_cfg = config.reward.get("pairwise_image_edit", {})
        self.margin = float(reward_cfg.get("margin", 0.05))
        self.tau = float(reward_cfg.get("tau", 0.1))
        self.format_bonus = float(reward_cfg.get("format_bonus", 0.1))
        self.score_max = float(reward_cfg.get("score_max", 10.0))
        self.pair_id_key = reward_cfg.get("pair_id_key", "pair_id")
        self.side_key = reward_cfg.get("side_key", "side")
        self.positive_side = str(reward_cfg.get("positive_side", "A")).upper()
        self.negative_side = str(reward_cfg.get("negative_side", "B")).upper()
        self.min_checkpoint_count = int(reward_cfg.get("min_checkpoint_count", 1))
        max_checkpoint_count = reward_cfg.get("max_checkpoint_count", None)
        self.max_checkpoint_count = int(max_checkpoint_count) if max_checkpoint_count is not None else None

    async def run_single(self, data: DataProto) -> dict:
        return (await self.run_batch(data))[-1]

    async def run_batch(self, data: DataProto) -> list[dict]:
        rollouts = []
        for i in range(len(data)):
            data_item = data[i]
            response_str = await self._decode_response(data_item)
            extra_info = self._extract_extra_info(data_item.non_tensor_batch)
            pair_id = extra_info.get(self.pair_id_key, data_item.non_tensor_batch.get("index", i))
            side = str(extra_info.get(self.side_key, data_item.non_tensor_batch.get(self.side_key, ""))).upper()
            rollouts.append(
                ScoredRollout(
                    idx=i,
                    pair_id=pair_id,
                    side=side,
                    parsed=self._parse_rollout(response_str),
                )
            )

        pair2side2rollouts: dict[Any, dict[str, list[ScoredRollout]]] = {}
        for rollout in rollouts:
            pair2side2rollouts.setdefault(rollout.pair_id, {}).setdefault(rollout.side, []).append(rollout)

        outputs = [None] * len(rollouts)
        for rollout in rollouts:
            pair_group = pair2side2rollouts.get(rollout.pair_id, {})
            bt_reward, valid_counterparts = self._bt_reward(rollout, pair_group)
            format_reward = rollout.parsed.format_reward
            reward = bt_reward + self.format_bonus * format_reward
            outputs[rollout.idx] = {
                "reward_score": reward,
                "reward_extra_info": {
                    "pairwise_bt": bt_reward,
                    "format_reward": format_reward,
                    "parsed_avg_score": rollout.parsed.score if rollout.parsed.score is not None else math.nan,
                    "valid_score": rollout.parsed.valid_score,
                    "score_count": rollout.parsed.score_count,
                    "valid_counterparts": valid_counterparts,
                    "pair_id": str(rollout.pair_id),
                    "side": rollout.side,
                },
            }

        return outputs

    def _extract_extra_info(self, non_tensor_batch: dict[str, Any]) -> dict[str, Any]:
        extra_info = non_tensor_batch.get("extra_info", {}) or {}
        if extra_info:
            return extra_info

        raw_prompt = non_tensor_batch.get("raw_prompt")
        if isinstance(raw_prompt, list):
            for message in raw_prompt:
                if isinstance(message, dict) and isinstance(message.get("extra_info"), dict):
                    return message["extra_info"]
        return {}

    async def _decode_response(self, data_item) -> str:
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        return await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

    def _bt_reward(self, rollout: ScoredRollout, pair_group: dict[str, list[ScoredRollout]]) -> tuple[float, int]:
        if not rollout.parsed.valid_score:
            return 0.0, 0

        positives = self._valid_scores(pair_group.get(self.positive_side, []))
        negatives = self._valid_scores(pair_group.get(self.negative_side, []))

        if rollout.side == self.positive_side:
            counterparts = negatives
            if not counterparts:
                return 0.0, 0
            reward = sum(
                self._log_sigmoid((rollout.parsed.score - score - self.margin) / self.tau) for score in counterparts
            )
        elif rollout.side == self.negative_side:
            counterparts = positives
            if not counterparts:
                return 0.0, 0
            reward = sum(
                self._log_sigmoid((score - rollout.parsed.score - self.margin) / self.tau) for score in counterparts
            )
        else:
            return 0.0, 0

        return reward / len(counterparts), len(counterparts)

    def _valid_scores(self, rollouts: list[ScoredRollout]) -> list[float]:
        return [rollout.parsed.score for rollout in rollouts if rollout.parsed.valid_score]

    def _parse_rollout(self, response: str) -> ParsedRollout:
        payload = self._load_json(response)
        if payload is None:
            return ParsedRollout(score=None, format_reward=0.0, valid_score=False, score_count=0)

        raw_scores = self._collect_scores(payload)
        normalized_scores = [self._normalize_score(score) for score in raw_scores]
        valid_scores = [score for score in normalized_scores if score is not None]
        score_count = len(valid_scores)
        has_score = score_count > 0
        count_ok = score_count >= self.min_checkpoint_count and (
            self.max_checkpoint_count is None or score_count <= self.max_checkpoint_count
        )

        format_reward = 0.4
        if has_score:
            format_reward += 0.3
        if raw_scores and len(raw_scores) == len(valid_scores):
            format_reward += 0.2
        if count_ok:
            format_reward += 0.1

        if not has_score or not count_ok:
            return ParsedRollout(score=None, format_reward=format_reward, valid_score=False, score_count=score_count)

        return ParsedRollout(
            score=sum(valid_scores) / len(valid_scores),
            format_reward=format_reward,
            valid_score=True,
            score_count=score_count,
        )

    def _load_json(self, response: str) -> Any | None:
        candidates = [response.strip()]
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(item.strip() for item in fenced)

        start = response.find("{")
        end = response.rfind("}")
        if 0 <= start < end:
            candidates.append(response[start : end + 1])

        for candidate in candidates:
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def _collect_scores(self, value: Any) -> list[float]:
        if isinstance(value, list):
            scores = []
            for item in value:
                scores.extend(self._collect_scores(item))
            return scores

        if not isinstance(value, dict):
            return []

        checkpoint_scores = []
        for key in ("checkpoints", "checkpoint", "eval_points", "rubric", "rubrics", "criteria", "items"):
            child = value.get(key)
            if isinstance(child, list):
                checkpoint_scores.extend(self._collect_scores(child))
        if checkpoint_scores:
            return checkpoint_scores

        for key in ("score", "final_score", "average_score", "avg_score"):
            score = self._coerce_number(value.get(key))
            if score is not None:
                return [score]

        scores = []
        for child in value.values():
            if isinstance(child, (list, dict)):
                scores.extend(self._collect_scores(child))
        return scores

    def _coerce_number(self, value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        return float(match.group(0)) if match else None

    def _normalize_score(self, score: float) -> float | None:
        if 0.0 <= score <= 1.0:
            return score
        if 1.0 < score <= self.score_max:
            return score / self.score_max
        return None

    @staticmethod
    def _log_sigmoid(value: float) -> float:
        if value >= 0:
            return -math.log1p(math.exp(-value))
        return value - math.log1p(math.exp(value))
