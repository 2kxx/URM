# URM T2I judge RL — 可执行 runbook（2026-09-15 冻结版）

一句话现状：**reward 用你的硬公式 `1[gap>m]`；数据和门槛必须配对（wrong 为主→m=0，easy/near 为主→m=0.5 分）；n=8 已设；辅助项压小、KL 0.04；评测只跑 T2I。**

---

## 0. 文件清单（谁需要同步到 /efs）

| 文件 | 状态 | 说明 |
|---|---|---|
| `RM_RL-main/examples/reward_function/pairwise_image_edit_reward_manager.py` | **待同步（整份）** | 软/硬/双层台阶三种 reward；`bad_dimensions` 改成数值 `bad_dimension_count`（val n>1 崩溃根因） |
| `scripts/run_rl_v4_train.sh` | **待同步（整份）** | `ROLLOUT_N=8`、`VAL_N=1`（单点验证）等改为环境变量，不再需要 sed |
| `RM_RL-main/verl/trainer/ppo/metric_utils.py` | **待同步** | val n>1 崩溃修复：非数值字段（str/list/tuple/dict/None）直接跳过 |
| `RM_RL-main/verl/trainer/ppo/v1/trainer_base.py` | **不需要** | 本轮新增的观测已按要求回退，与你 /efs 上那份一致 |
| `output/rl_mix_4000/rl_pairwise_train_4000.verl.jsonl`（5.4M） | **待同步** | **主训练集（4000 对，问题样本压到 50%）**，见 §2 |
| `output/rl_mix_wrongheavy/rl_pairwise_train_wrongheavy.verl.jsonl`（7.4M） | 备选 | 5700 对、问题样本 69% 的版本，想要"更偏问题样本"时用 |

同步（在本机执行，`-R` 保留目录结构，已 dry-run 验证目标为 `output/rl_mix_wrongheavy/...`）：

```bash
cd /group/40063/jacuzzitang/projects/URM
rsync -avR ./output/rl_mix_4000 $REMOTE:/efs/jacuzzitang/projects/URM/
```

数据成品校验（已通过）：8000 行 / 4000 对，A/B 相邻 0 违规、字段齐全、图片路径均为 `/efs/...`。

---

## 1. /efs 上的前置检查（整份同步后核对，无需 sed）

```bash
cd /efs/jacuzzitang/projects/URM
bash -n scripts/run_rl_v4_train.sh
PYTHON=/bin/echo bash scripts/run_rl_v4_train.sh | tr ' ' '\n' | \
  grep -E "rollout\.n=|val_kwargs\.n=|reward_type|threshold_margin|tier_|experiment_name"
#  期望：rollout.n=8 / val_kwargs.n=1 / 前缀都完整
python3 scripts/test_soft_threshold_reward.py     # 若已同步测试脚本，期望 11/11
```

## 2. Arm A（主推）：wrong 为主 + m=0

**数据**：`output/rl_mix_4000/rl_pairwise_train_4000.verl.jsonl`（4000 对 / 8000 行）
成分：near 39% + **judge 同分 32% + judge 判错 18%（问题样本 = 50%，旧版 69%）** + easy 11%
`|judge gap|` p50/p75 = 0.25/0.50 raw；加权活组率 **69%**，预期越线率 **≈0.56**

```bash
nohup env EXPERIMENT_NAME=stage6_pairwise_rl_wrongheavy_n8_v4 \
  TRAIN_FILE=/efs/jacuzzitang/projects/URM/output/rl_mix_4000/rl_pairwise_train_4000.verl.jsonl \
  REWARD_TYPE=mean_threshold_format THRESHOLD_MARGIN=0.0 \
  USE_TIE_VAL=0 KL=0.04 \
  FORMAT_BONUS=0.02 INVALID_PENALTY=0.05 DIM_PENALTY=0.02 \
  bash scripts/run_rl_v4_train.sh > logs/rl_t2i_v4_wrongheavy_n8.log 2>&1 &
```

**为什么 m=0**（活组率 = 组内有梯度的比例，n=8）：

| 数据桶（本 mix 占比） | m=0 | m=0.5 分 |
|---|---|---|
| tie（44%） | **99%** | 65% |
| judge-wrong（25%） | **65%** | **5% ← 死** |
| near（21%） | 64% | 99% |
| easy（9%） | 5%（死） | 99% |
| **加权** | **74%** | 60%（死的是那 25% 判错对） |

---

## 3. Arm B（备选）：easy/near 为主 + m=0.5 分（专治"降 tie"）

```bash
# 数据（在 /efs 上生成；需要 scripts/build_rl_pairwise_mixed.py）
python3 scripts/build_rl_pairwise_mixed.py --out-dir output/rl_easy_sel/rl_data \
  --train-near 2000 --train-easy 2000 --train-hard-wrong 0 \
  --val-near 0 --val-easy 0 --val-hard-wrong 0

nohup env EXPERIMENT_NAME=stage6_pairwise_rl_easynear_m0.5_n8_v4 \
  TRAIN_FILE=/efs/jacuzzitang/projects/URM/output/rl_easy_sel/rl_data/rl_pairwise_train_mixed.verl.jsonl \
  REWARD_TYPE=mean_threshold_format THRESHOLD_MARGIN=0.05 \
  USE_TIE_VAL=0 KL=0.04 \
  FORMAT_BONUS=0.02 INVALID_PENALTY=0.05 DIM_PENALTY=0.02 \
  bash scripts/run_rl_v4_train.sh > logs/rl_t2i_v4_easynear_m0.5.log 2>&1 &
```

两个 arm 只能先后跑（8 卡），别同时开；优先跑 Arm A。

---

## 4. 怎么读（硬 m=0 时）

```bash
sleep 600
# 单点 val 的主指标 + 训练侧组内/两侧差异（都不需要 VAL_N>1）
grep -a "val-core/urm_t2i_pairwise/acc/mean@1\|pairwise_score_diff/mean@1\|point_count/by_case/std_mean\|point_count/by_pair/std_mean" \
  logs/*n8*.log | tail -6
```

| 读数 | 期望 / 判据 |
|---|---|
| `val-core/urm_t2i_pairwise/acc/mean@1` | **主指标**（单点判定，同 v2/v3b/v4 口径） |
| `越线率 ≈ critic/rewards/mean − 0.1×format_reward` | Arm A（m=0）≈ **0.56**；<0.3 或 >0.7 = 数据/门槛没配上 |
| `1 − (by_case/std)²/(by_pair/std)²`（两侧差异占比） | base 24%；**应上涨** = 开始用分数表达两张图差别 |
| `parsed_final_score`（均分） | 不应持续上涨（通胀=失败信号） |
| `format_reward` | 不应继续掉（上次掉 6 个点） |
| `actor/entropy` | 别掉到 0.12 |

**三种终局判读**（bench 层面）：
* tie 率降 + 非平局 acc 不降 → 真在修 tie ✅
* tie 率降 + 非平局 acc 也降 → 只是变"乱判" ❌
* tie 率不变 + 两侧差异占比不涨 → 只在同侧抖，没学到判别 ❌

---

## 5. bench 评测（T2I only，8 卡并行）

```bash
RUN_DIR=/efs/jacuzzitang/projects/URM/checkpoints/URM_T2I_RL/stage6_pairwise_rl_wrongheavy_n8_v4 \
BENCH_SUFFIX=_n8v4 MERGE_PREFIX=_n8v4 STEPS="20 40 60 80 100" NDEVICES=8 \
EVAL_TASKS="image_generation t2i" \
nohup bash scripts/run_rl_bench_all.sh > logs/rl_bench_all_n8v4.log 2>&1 &
```

base 参照（同一 2306 对）：**genai acc 0.610 / tie 13.3%，mmrb2 acc 0.521 / tie 20.7%**；非平局 acc **0.703 / 0.657**。
上一轮软版本（τ=0.5, m=0）s160：genai 0.575 / tie 18.1%，mmrb2 0.426 / tie 34.8% —— 全面差于 base。

回传给我分析：

```bash
rsync -avR "$REMOTE:/efs/jacuzzitang/projects/URM/./output/bench_rl_t2i_n8v4/" "$DST/"
rsync -avR "$REMOTE:/efs/jacuzzitang/projects/URM/./logs/rl_t2i_v4_wrongheavy_n8.log" "$DST/"
```

---

## 6. 已知的坑

1. `EXPERIMENT_NAME` 必须显式给，否则默认名 `stage6_pairwise_rl_tau0.5_v4` 会覆盖 9/15 那次的 checkpoint 目录。
2. `run_rl_bench.sh` 的 eval 调用**不带 `--merge`** → 不产 `*_report.json` → `run_rl_bench_all.sh` 的"已完成跳过"认不出来，重跑会重算（无数据损坏）。要补报告：
   `python3 scripts/eval_benchmarks_stage3.py --merge --benchmark genai_bench mmrb2 --output-dir <step目录>`
3. eval 是 `--no-resume`：中断的 step 会从零重跑。
4. 训练 rollout 4→8 后每步 token 翻倍（每步 ≈9 分钟；想保持 5 分钟就把 `data.train_batch_size=32` 改 16，epoch 步数翻倍、总时长不变）。
5. 不要用 `m>0 + 小 τ` 或"低于门槛钳到 0"：前者饱和、后者把组内差异清零（v2/v3b 的实测死组）。
6. **val n>1 崩溃**（`np.max` zero-size array）：根因是 `bad_dimensions` 这个 list 型 `reward_extra_info` 字段
   进了 bootstrap 聚合。已修两处：manager 改成数值 `bad_dimension_count`；`metric_utils.process_validation_metrics`
   跳过所有非数值字段。两处都要同步，否则 `VAL_N>1` 时会在 `val_before_train` 那一步直接崩。
