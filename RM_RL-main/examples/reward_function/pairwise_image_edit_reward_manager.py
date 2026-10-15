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
    eval_points_ok: bool
    dimension_summary_ok: bool
    score_reason_ok: bool
    # True when the eval_points dimension labels all belong to the configured
    # whitelist (``allowed_dimensions``). Trivially True when no whitelist is set.
    dimension_ok: bool
    # non-canonical dimension labels seen in eval_points (for logging only)
    bad_dimensions: tuple[str, ...] = ()


@dataclass
class ScoredRollout:
    idx: int
    pair_id: Any
    side: str
    parsed: ParsedRollout


@dataclass
class RewardOutput:
    score: float
    info: dict[str, Any]


def _as_bool(value: Any) -> bool:
    """Config-robust bool: Hydra usually hands us a real bool, but a stringified
    override (``+a.b="False"``) must not silently become ``True``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class PairwiseImageEditRewardManager(RewardManagerBase):
    """Batch reward manager for pairwise image-edit checkpoint scoring.

    Expected dataset metadata:
      - ``extra_info[pair_id_key]`` or top-level ``index`` identifies the A/B pair.
      - ``extra_info[side_key]`` is ``"A"`` for the preferred edit and ``"B"`` for the worse edit.

    Expected rollout text:
      - JSON object containing ``final_score``.
      - ``final_score`` is expected on the configured ``[0, score_max]`` scale and is normalized to ``[0, 1]``.

    Reward:
      - Valid A_i rollouts compare against valid B_j rollouts from the same pair.
      - The pairwise term uses the Bradley-Terry probability:
        ``sigmoid((s_A - s_B - margin) / tau)``.
      - ``mean_threshold_format`` / ``median_threshold_format`` are *hard steps*
        on the counterpart threshold (``float(s - thr > margin)``): ``tau`` is
        NOT used, so a correct-but-close call and a wildly wrong call get the
        same 0/1.  ``soft_threshold_format`` replaces the step with the
        temperature-smoothed ``sigmoid((s - thr - margin) / tau)`` so the reward
        is graded in the decision band.
      - Format reward is binary and requires the expected JSON schema:
        valid final_score, eval_points, dimension_summary, and score_reason.
      - ``R_i = R_bt_i + format_bonus * R_format_i``.
    """

    batch_reward = True

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        reward_cfg = config.reward.get("pairwise_image_edit", {})
        self.margin = float(reward_cfg.get("margin", 0.05))
        self.tau = float(reward_cfg.get("tau", 0.1))
        self.format_bonus = float(reward_cfg.get("format_bonus", 0.1))
        self.reward_type = str(reward_cfg.get("reward_type", "bt_sigmoid_format"))
        self.acc_margin = float(reward_cfg.get("acc_margin", 0.0))
        self.threshold_margin = float(reward_cfg.get("threshold_margin", 0.0))
        self.score_max = float(reward_cfg.get("score_max", 10.0))
        # ``tau`` lives in normalized score units (final_score / score_max).  When
        # ``tau_raw`` is given it is interpreted in raw final_score points on the
        # ``[0, score_max]`` scale, which is far easier to pick from the data:
        # the observed |gap| between the preferred and rejected image is ~0.3 raw
        # points for near-boundary pairs, so ``tau_raw=0.3`` is a good default
        # (``tau=0.03`` normalized) while the old ``tau=0.2`` normalized is 2 raw
        # points -- far too flat to move the reward.
        tau_raw = reward_cfg.get("tau_raw", None)
        if tau_raw is not None:
            self.tau = float(tau_raw) / self.score_max
        # ``use_dynamic_threshold`` makes the temperature noise-aware: the
        # effective temperature is at least ``dynamic_scale_mult`` times the
        # spread of the counterpart scores, so a gap that is smaller than the
        # policy's own rollout noise is not rewarded as if it were real signal.
        self.use_dynamic_threshold = _as_bool(reward_cfg.get("use_dynamic_threshold", False))
        self.dynamic_scale_mult = float(reward_cfg.get("dynamic_scale_mult", 1.0))
        # counterpart statistic used as the threshold: "mean" (default) or "median"
        self.threshold_type = str(reward_cfg.get("threshold_type", "mean"))
        # Two-tier variant (reward_type=mean_threshold_tiered):
        #   r = 1[gap > threshold_margin] + tier_weight * 1[gap > tier_margin]
        # The bottom tier is the plain indicator (m=0 keeps the judge-wrong pairs
        # alive); the top tier adds the "you must separate by tier_margin" bonus on
        # pairs that can actually reach it.  A single fixed margin > 0 cannot do
        # both: the judge-wrong pairs sit ~1.4 sigma on the wrong side of the
        # counterpart mean, so any positive bar turns those groups into all-zero
        # (zero advantage) groups.
        self.tier_margin = float(reward_cfg.get("tier_margin", 0.05))
        self.tier_weight = float(reward_cfg.get("tier_weight", 1.0))
        self.pair_id_key = reward_cfg.get("pair_id_key", "pair_id")
        self.side_key = reward_cfg.get("side_key", "side")
        self.positive_side = str(reward_cfg.get("positive_side", "A")).upper()
        self.negative_side = str(reward_cfg.get("negative_side", "B")).upper()
        self.min_checkpoint_count = int(reward_cfg.get("min_checkpoint_count", 1))
        max_checkpoint_count = reward_cfg.get("max_checkpoint_count", None)
        self.max_checkpoint_count = int(max_checkpoint_count) if max_checkpoint_count is not None else None
        # dimension_summary keys required for the format bonus.  The image-edit
        # example defaults to its three dimensions; for a T2I judge pass e.g.
        #   +reward.pairwise_image_edit.dimensions="Prompt Alignment,Aesthetics,Technical Quality"
        # as either a list or a comma/pipe separated string.
        self.required_dimensions = self._parse_dimensions(reward_cfg.get("dimensions"))
        # eval_points dimension-label whitelist. If not given explicitly it
        # defaults to ``dimensions`` (so a T2I judge that pins its summary keys
        # to the 3 canonical dims also gets its eval_points labels whitelisted).
        # Unlike required_dimensions, an unset value stays EMPTY -> no whitelist
        # enforcement (keeps the bare image-edit example backward compatible).
        _allowed_cfg = reward_cfg.get("allowed_dimensions", reward_cfg.get("dimensions"))
        self.allowed_dimensions = self._parse_dimensions(_allowed_cfg) if _allowed_cfg is not None else ()
        # dimension_summary keys must be EXACTLY the required dims (no extra
        # stray keys such as an old-taxonomy "Physical Plausibility" section).
        self.strict_summary = bool(reward_cfg.get("strict_summary", False))
        # extra penalty subtracted when the checklist drifts outside the
        # whitelist (dimension_penalty) or the rollout is not parseable /
        # truncated / has no final_score (invalid_rollout_penalty).
        self.dimension_penalty = float(reward_cfg.get("dimension_penalty", 0.0))
        self.invalid_rollout_penalty = float(reward_cfg.get("invalid_rollout_penalty", 0.0))
        self.reward_fns = {
            "bt_sigmoid_format": self._reward_bt_sigmoid_format,
            "pairwise_acc_format": self._reward_pairwise_acc_format,
            "mean_threshold_format": self._reward_mean_threshold_format,
            "median_threshold_format": self._reward_median_threshold_format,
            "soft_threshold_format": self._reward_soft_threshold_format,
            "mean_threshold_tiered": self._reward_mean_threshold_tiered,
        }
        if self.reward_type not in self.reward_fns:
            available = ", ".join(sorted(self.reward_fns))
            raise ValueError(f"Unknown pairwise image-edit reward_type={self.reward_type!r}. Available: {available}")

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

        pair_metrics = {
            pair_id: self._pair_metrics(side2rollouts) for pair_id, side2rollouts in pair2side2rollouts.items()
        }

        outputs = [None] * len(rollouts)
        pair_reward_scores: dict[Any, list[float]] = {}
        for rollout in rollouts:
            pair_group = pair2side2rollouts.get(rollout.pair_id, {})
            pair_metric = pair_metrics.get(rollout.pair_id, {})
            reward_output = self.reward_fns[self.reward_type](rollout, pair_group)
            pair_reward_scores.setdefault(rollout.pair_id, []).append(reward_output.score)
            score = reward_output.score
            if not rollout.parsed.valid_score:
                # non-terminating / truncated / unparseable rollouts get an
                # explicit negative so GRPO advantage clearly pushes them down
                # (they already contribute 0 to the threshold term).
                score -= self.invalid_rollout_penalty
            elif not rollout.parsed.dimension_ok:
                score -= self.dimension_penalty
            outputs[rollout.idx] = {
                "reward_score": score,
                "reward_extra_info": {
                    **reward_output.info,
                    "parsed_final_score": rollout.parsed.score if rollout.parsed.score is not None else math.nan,
                    "parsed_avg_score": rollout.parsed.score if rollout.parsed.score is not None else math.nan,
                    "valid_score": rollout.parsed.valid_score,
                    "score_count": rollout.parsed.score_count,
                    "eval_points_ok": rollout.parsed.eval_points_ok,
                    "dimension_summary_ok": rollout.parsed.dimension_summary_ok,
                    "score_reason_ok": rollout.parsed.score_reason_ok,
                    "dimension_ok": rollout.parsed.dimension_ok,
                    # numeric only: a list-valued field makes the val metric aggregation
                    # (process_validation_metrics -> bootstrap_metric -> np.max) crash with
                    # "zero-size array" as soon as val_kwargs.n > 1
                    "bad_dimension_count": float(len(rollout.parsed.bad_dimensions)),
                    "pair_id": str(rollout.pair_id),
                    "side": rollout.side,
                    "acc": pair_metric.get("acc", 0.0),
                    "pairwise_acc": pair_metric.get("acc", 0.0),
                    "pairwise_acc_margin": pair_metric.get("acc_margin", 0.0),
                    "pairwise_evaluable": pair_metric.get("evaluable", 0.0),
                    "pairwise_score_diff": pair_metric.get("score_diff", 0.0),
                    "pairwise_positive_mean": pair_metric.get("positive_mean", 0.0),
                    "pairwise_negative_mean": pair_metric.get("negative_mean", 0.0),
                },
            }

        # Within-pair reward spread: the quantity GRPO's advantage is built from.
        # ~0 means the pair's rollouts are indistinguishable to the reward (the
        # degenerate "everything ties" regime that made the hard4k run flat).
        for rollout in rollouts:
            scores = pair_reward_scores.get(rollout.pair_id) or []
            info = outputs[rollout.idx]["reward_extra_info"]
            info["pair_reward_std"] = self._std(scores)
            info["pair_reward_range"] = (max(scores) - min(scores)) if scores else 0.0

        return outputs

    def _reward_bt_sigmoid_format(
        self, rollout: ScoredRollout, pair_group: dict[str, list[ScoredRollout]]
    ) -> RewardOutput:
        bt_reward, valid_counterparts = self._bt_reward(rollout, pair_group)
        format_reward = rollout.parsed.format_reward
        return RewardOutput(
            score=bt_reward + self.format_bonus * format_reward,
            info={
                "pairwise_bt": bt_reward,
                "format_reward": format_reward,
                "valid_counterparts": valid_counterparts,
            },
        )

    def _reward_pairwise_acc_format(
        self, rollout: ScoredRollout, pair_group: dict[str, list[ScoredRollout]]
    ) -> RewardOutput:
        acc_reward, valid_counterparts = self._pairwise_acc_reward(rollout, pair_group)
        format_reward = rollout.parsed.format_reward
        return RewardOutput(
            score=acc_reward + self.format_bonus * format_reward,
            info={
                "pairwise_acc_reward": acc_reward,
                "format_reward": format_reward,
                "valid_counterparts": valid_counterparts,
            },
        )

    def _reward_mean_threshold_format(
        self, rollout: ScoredRollout, pair_group: dict[str, list[ScoredRollout]]
    ) -> RewardOutput:
        threshold_reward, valid_counterparts, threshold = self._threshold_reward(rollout, pair_group, "mean")
        format_reward = rollout.parsed.format_reward
        return RewardOutput(
            score=threshold_reward + self.format_bonus * format_reward,
            info={
                "mean_threshold_reward": threshold_reward,
                "format_reward": format_reward,
                "valid_counterparts": valid_counterparts,
                # -1.0 sentinel (never a legal score) instead of NaN: NaN makes
                # the whole aggregated val metric NaN.
                "threshold": threshold if threshold is not None else -1.0,
                "threshold_valid": float(threshold is not None),
            },
        )

    def _reward_median_threshold_format(
        self, rollout: ScoredRollout, pair_group: dict[str, list[ScoredRollout]]
    ) -> RewardOutput:
        threshold_reward, valid_counterparts, threshold = self._threshold_reward(rollout, pair_group, "median")
        format_reward = rollout.parsed.format_reward
        return RewardOutput(
            score=threshold_reward + self.format_bonus * format_reward,
            info={
                "median_threshold_reward": threshold_reward,
                "format_reward": format_reward,
                "valid_counterparts": valid_counterparts,
                "threshold": threshold if threshold is not None else -1.0,
                "threshold_valid": float(threshold is not None),
            },
        )

    def _reward_mean_threshold_tiered(
        self, rollout: ScoredRollout, pair_group: dict[str, list[ScoredRollout]]
    ) -> RewardOutput:
        """``1[gap > threshold_margin] + tier_weight * 1[gap > tier_margin]``.

        Bottom tier = the plain indicator (keep ``threshold_margin=0`` so the
        judge-wrong pairs still carry a (tail-driven) gradient); top tier = the
        "decisively separated" bonus, which is what makes the already-easy pairs --
        all-identical at the bottom tier -- carry a gradient again.
        """
        context = self._threshold_context(rollout, pair_group, self.threshold_type)
        format_reward = rollout.parsed.format_reward
        if context is None:
            return RewardOutput(
                score=self.format_bonus * format_reward,
                info={"tiered_base_reward": 0.0, "tiered_bonus": 0.0, "format_reward": format_reward,
                      "valid_counterparts": 0, "threshold": -1.0, "threshold_valid": 0.0, "pairwise_gap": 0.0},
            )
        gap, threshold, counterpart_std, valid_counterparts = context
        base = float(gap > self.threshold_margin)
        bonus = float(gap > self.tier_margin)
        return RewardOutput(
            score=base + self.tier_weight * bonus + self.format_bonus * format_reward,
            info={
                "tiered_base_reward": base,
                "tiered_bonus": bonus,
                "format_reward": format_reward,
                "valid_counterparts": valid_counterparts,
                "threshold": threshold,
                "threshold_valid": 1.0,
                "threshold_std": counterpart_std,
                "pairwise_gap": gap,
            },
        )

    def _reward_soft_threshold_format(
        self, rollout: ScoredRollout, pair_group: dict[str, list[ScoredRollout]]
    ) -> RewardOutput:
        """Temperature-smoothed ``a > mean(b)`` reward.

        ``R = sigmoid((gap - margin) / tau_eff)`` where ``gap = s - thr`` for the
        positive side (sign flipped for the negative side) and ``thr`` is the
        counterpart mean/median.  Unlike the hard-step variants this keeps a
        graded signal inside the decision band, so a rollout that moves the
        judgment in the right direction is rewarded even before it crosses the
        threshold -- which is what the hard step is missing on near-boundary and
        hard-wrong pairs.
        """
        context = self._threshold_context(rollout, pair_group, self._soft_threshold_type())
        format_reward = rollout.parsed.format_reward
        if context is None:
            return RewardOutput(
                score=self.format_bonus * format_reward,
                info={
                    "soft_threshold_reward": 0.0,
                    "format_reward": format_reward,
                    "valid_counterparts": 0,
                    "threshold": -1.0,
                    "threshold_valid": 0.0,
                    "threshold_std": 0.0,
                    "dynamic_tau": self.tau,
                    "pairwise_gap": 0.0,
                },
            )

        gap, threshold, counterpart_std, valid_counterparts = context
        dynamic_tau = self.tau
        if self.use_dynamic_threshold:
            dynamic_tau = max(self.tau, self.dynamic_scale_mult * counterpart_std)
        reward = self._sigmoid((gap - self.threshold_margin) / dynamic_tau)
        return RewardOutput(
            score=reward + self.format_bonus * format_reward,
            info={
                "soft_threshold_reward": reward,
                "format_reward": format_reward,
                "valid_counterparts": valid_counterparts,
                "threshold": threshold,
                "threshold_valid": 1.0,
                "threshold_std": counterpart_std,
                "dynamic_tau": dynamic_tau,
                "pairwise_gap": gap,
            },
        )

    def _soft_threshold_type(self) -> str:
        """Counterpart statistic used as the soft threshold (mean by default)."""
        return "median" if self.threshold_type == "median" else "mean"

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
                self._sigmoid((rollout.parsed.score - score - self.margin) / self.tau) for score in counterparts
            )
        elif rollout.side == self.negative_side:
            counterparts = positives
            if not counterparts:
                return 0.0, 0
            reward = sum(self._sigmoid((score - rollout.parsed.score - self.margin) / self.tau) for score in counterparts)
        else:
            return 0.0, 0

        return reward / len(counterparts), len(counterparts)

    def _pairwise_acc_reward(
        self, rollout: ScoredRollout, pair_group: dict[str, list[ScoredRollout]]
    ) -> tuple[float, int]:
        if not rollout.parsed.valid_score:
            return 0.0, 0

        positives = self._valid_scores(pair_group.get(self.positive_side, []))
        negatives = self._valid_scores(pair_group.get(self.negative_side, []))

        if rollout.side == self.positive_side:
            counterparts = negatives
            if not counterparts:
                return 0.0, 0
            reward = sum(float(rollout.parsed.score - score > self.acc_margin) for score in counterparts)
        elif rollout.side == self.negative_side:
            counterparts = positives
            if not counterparts:
                return 0.0, 0
            reward = sum(float(score - rollout.parsed.score > self.acc_margin) for score in counterparts)
        else:
            return 0.0, 0

        return reward / len(counterparts), len(counterparts)

    def _threshold_reward(
        self, rollout: ScoredRollout, pair_group: dict[str, list[ScoredRollout]], threshold_type: str
    ) -> tuple[float, int, float | None]:
        """Hard 0/1 step on the counterpart threshold (``tau`` is unused here)."""
        context = self._threshold_context(rollout, pair_group, threshold_type)
        if context is None:
            return 0.0, 0, None
        gap, threshold, _counterpart_std, valid_counterparts = context
        return float(gap > self.threshold_margin), valid_counterparts, threshold

    def _threshold_context(
        self, rollout: ScoredRollout, pair_group: dict[str, list[ScoredRollout]], threshold_type: str
    ) -> tuple[float, float, float, int] | None:
        """``(gap, threshold, counterpart_std, n_counterparts)`` for one rollout.

        ``gap`` is signed towards the correct answer: positive when the rollout
        already sits on the right side of the counterpart threshold (the winner
        side scores above it, the loser side below it).
        """
        if not rollout.parsed.valid_score:
            return None

        if rollout.side == self.positive_side:
            counterparts = self._valid_scores(pair_group.get(self.negative_side, []))
            sign = 1.0
        elif rollout.side == self.negative_side:
            counterparts = self._valid_scores(pair_group.get(self.positive_side, []))
            sign = -1.0
        else:
            return None

        threshold = self._score_threshold(counterparts, threshold_type)
        if threshold is None:
            return None
        gap = sign * (rollout.parsed.score - threshold)
        return gap, threshold, self._std(counterparts), len(counterparts)

    def _score_threshold(self, scores: list[float], threshold_type: str) -> float | None:
        if not scores:
            return None
        if threshold_type == "mean":
            return sum(scores) / len(scores)
        if threshold_type == "median":
            return self._median(scores)
        raise ValueError(f"Unknown threshold_type={threshold_type!r}")

    def _valid_scores(self, rollouts: list[ScoredRollout]) -> list[float]:
        return [rollout.parsed.score for rollout in rollouts if rollout.parsed.valid_score]

    @staticmethod
    def _median(scores: list[float]) -> float:
        sorted_scores = sorted(scores)
        mid = len(sorted_scores) // 2
        if len(sorted_scores) % 2 == 1:
            return sorted_scores[mid]
        return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2

    @staticmethod
    def _std(scores: list[float]) -> float:
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        return math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))

    def _pair_metrics(self, pair_group: dict[str, list[ScoredRollout]]) -> dict[str, float]:
        positives = self._valid_scores(pair_group.get(self.positive_side, []))
        negatives = self._valid_scores(pair_group.get(self.negative_side, []))
        if not positives or not negatives:
            return {
                "acc": 0.0,
                "acc_margin": 0.0,
                "evaluable": 0.0,
                "score_diff": 0.0,
                "positive_mean": 0.0,
                "negative_mean": 0.0,
            }

        positive_mean = sum(positives) / len(positives)
        negative_mean = sum(negatives) / len(negatives)
        score_diff = positive_mean - negative_mean
        return {
            "acc": float(score_diff > 0),
            "acc_margin": float(score_diff > self.margin),
            "evaluable": 1.0,
            "score_diff": score_diff,
            "positive_mean": positive_mean,
            "negative_mean": negative_mean,
        }

    def _parse_rollout(self, response: str) -> ParsedRollout:
        payload = self._load_json(response)
        if payload is None:
            return ParsedRollout(
                score=None,
                format_reward=0.0,
                valid_score=False,
                score_count=0,
                eval_points_ok=False,
                dimension_summary_ok=False,
                score_reason_ok=False,
                dimension_ok=False,
            )

        raw_point_scores = self._collect_eval_point_scores(payload)
        normalized_point_scores = [self._normalize_binary_score(score) for score in raw_point_scores]
        valid_point_scores = [score for score in normalized_point_scores if score is not None]
        score_count = len(valid_point_scores)
        raw_final_score = self._extract_final_score(payload)
        final_score = self._normalize_score(raw_final_score) if raw_final_score is not None else None
        has_score = final_score is not None
        eval_points_ok = self._eval_points_ok(payload, valid_point_scores)
        dimension_summary_ok = self._has_dimension_summary(payload)
        score_reason_ok = self._has_score_reason(payload)
        count_ok = score_count >= self.min_checkpoint_count and (
            self.max_checkpoint_count is None or score_count <= self.max_checkpoint_count
        )
        dimension_ok, bad_dimensions = self._dimension_labels_ok(payload)

        format_reward = (
            1.0
            if has_score
            and eval_points_ok
            and count_ok
            and dimension_summary_ok
            and score_reason_ok
            and dimension_ok
            else 0.0
        )

        if not has_score:
            return ParsedRollout(
                score=None,
                format_reward=format_reward,
                valid_score=False,
                score_count=score_count,
                eval_points_ok=eval_points_ok,
                dimension_summary_ok=dimension_summary_ok,
                score_reason_ok=score_reason_ok,
                dimension_ok=dimension_ok,
                bad_dimensions=bad_dimensions,
            )

        return ParsedRollout(
            score=final_score,
            format_reward=format_reward,
            valid_score=True,
            score_count=score_count,
            eval_points_ok=eval_points_ok,
            dimension_summary_ok=dimension_summary_ok,
            score_reason_ok=score_reason_ok,
            dimension_ok=dimension_ok,
            bad_dimensions=bad_dimensions,
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

    def _extract_final_score(self, value: Any) -> float | None:
        if not isinstance(value, dict):
            return None
        return self._coerce_number(value.get("final_score"))

    def _collect_eval_point_scores(self, value: Any) -> list[float]:
        if not isinstance(value, dict):
            return []

        eval_points = value.get("eval_points")
        if not isinstance(eval_points, list):
            return []

        scores = []
        for point in eval_points:
            if isinstance(point, dict):
                score = self._coerce_number(point.get("score"))
                if score is not None:
                    scores.append(score)
        return scores

    def _eval_points_ok(self, value: Any, valid_point_scores: list[float]) -> bool:
        if not isinstance(value, dict):
            return False

        eval_points = value.get("eval_points")
        if not isinstance(eval_points, list) or not eval_points:
            return False

        if len(eval_points) != len(valid_point_scores):
            return False

        for point in eval_points:
            if not isinstance(point, dict):
                return False
            question = point.get("question")
            dimension = point.get("dimension")
            if not isinstance(question, str) or not question.strip():
                return False
            if not isinstance(dimension, str) or not dimension.strip():
                return False
            if self._normalize_binary_score(self._coerce_number(point.get("score"))) is None:
                return False
        return True

    def _dimension_labels_ok(self, value: Any) -> tuple[bool, tuple[str, ...]]:
        """Check eval_points ``dimension`` labels against the whitelist.

        Returns ``(clean, bad_dimensions)``. ``clean`` is True when no whitelist
        is configured or every label belongs to it; otherwise it lists the
        offending labels (deduped, insertion-ordered) for logging.
        """
        allowed = self.allowed_dimensions
        if not allowed:
            return True, ()
        if not isinstance(value, dict):
            return False, ()
        eval_points = value.get("eval_points")
        if not isinstance(eval_points, list):
            return False, ()

        allowed_set = set(allowed)
        bad: list[str] = []
        seen: set[str] = set()
        for point in eval_points:
            if not isinstance(point, dict):
                continue
            dim = point.get("dimension")
            if not isinstance(dim, str) or not dim.strip():
                continue
            if dim not in allowed_set and dim not in seen:
                bad.append(dim)
                seen.add(dim)
        return not bad, tuple(bad)

    def _has_dimension_summary(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False

        summary = value.get("dimension_summary")

        if isinstance(summary, dict):
            required_dimensions = self.required_dimensions
            if not required_dimensions:
                # nothing configured -> require a non-empty summary dict
                return bool(summary) and any(
                    isinstance(v, str) and v.strip() for v in summary.values()
                )
            present_ok = all(
                isinstance(summary.get(dimension), str) and summary[dimension].strip()
                for dimension in required_dimensions
            )
            if not present_ok:
                return False
            if self.strict_summary:
                # no extra (old-taxonomy) summary sections allowed
                return set(summary.keys()) == set(required_dimensions)
            return True
        return False

    @staticmethod
    def _parse_dimensions(value: Any) -> tuple[str, ...]:
        if value is None:
            return ("Instruction Following", "Visual Consistency", "Visual Quality")
        if isinstance(value, (list, tuple)):
            items = [str(x) for x in value if str(x).strip()]
            return tuple(items) if items else ()
        if isinstance(value, str):
            items = [x.strip() for x in re.split(r"[,|]", value) if x.strip()]
            return tuple(items) if items else ()
        return ()

    def _has_score_reason(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        score_reason = value.get("score_reason")
        return isinstance(score_reason, str) and bool(score_reason.strip())

    def _coerce_number(self, value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        return float(match.group(0)) if match else None

    def _normalize_score(self, score: float) -> float | None:
        if 0.0 <= score <= self.score_max:
            return score / self.score_max
        return None

    @staticmethod
    def _normalize_binary_score(score: float) -> float | None:
        if score in (0.0, 1.0):
            return score
        return None

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)
