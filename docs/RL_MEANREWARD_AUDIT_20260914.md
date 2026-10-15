# Pairwise-judge RL: mean-reward 诊断 + tau 修正 (2026-09-14, rev3)

对象：`stage6_pairwise_rl_hard4k_v2` / `_v3b`（base = stage6 ck250）、
`mean_threshold_format` reward。**数据集不变**（仍是 `rl_pairwise_train_allwrong`
那 4k 对 judge 答错的样本，同分对保留），只修 reward。

```bash
python3 scripts/audit_rl_pairwise_data.py          # -> output/rl_data_audit/audit.json
python3 scripts/test_soft_threshold_reward.py      # 10/10 reward 单测（含同分对行为）
bash scripts/run_rl_tau_sweep.sh                   # 打印 3 个 tau 的命令
```

## 0. 两个"难度"口径（别混）

| 口径 | 来源 | 含义 |
|---|---|---|
| **judge 口径**（你要解决的） | `analyzed.jsonl` 的 `difficulty`、`score_a/score_b` = stage6 judge 自己的输出 | judge 判对/判错/同分。**judge 打同分（tie）正是这次 RL 要解决的问题**，必须留在训练集里 |
| label 口径 | `data/processed/pointwise/*.jsonl` 的 `metadata.pointwise_score`（RubricRM 0–4，除 `chosen` 外还有 margin） | 标注本身给 A/B 的分数差，越小 = 标注越犹豫 |

两者弱相关（judge 判 "easy" 的对，label margin 中位数 2.05；judge 判 "wrong" 只有 1.30）。
按你的要求：**本版 train 不变、仍用 judge 口径的全难集（含同分对）**；
按 label margin 的版本留在 `output/rl_label_hard_sel/`，想试再换（§6）。

## 1. 三个问题的实测答案

| 你的判断 | 实测 |
|---|---|
| "`a > mean(b)` 太弱，得不到有效奖励" | **是硬阶跃的问题，不是"太弱"**：`mean_threshold_format` 是 0/1 阶跃、`tau` 完全不参与；而且在**同分对**上（train 的 63%），`margin=0.05` 让两侧都拿 **R=0** → 整个 pair group reward 完全相同 → **advantage 恒等于 0，完全没有梯度**。 |
| "需要设置合理的 tau" | **对，而且这是本版唯一的改动。** 同分对上软阈值给两侧 0.5，并且对"改善"有可调响应：同分对 +0.5 raw 分的改善，`tau_raw=2.0`（＝旧 tau=0.2）只把 reward 从 0.500 抬到 0.562，`0.5` → 0.731，`0.3` → 0.841。旧配置在这个池子上正确侧 `P(R=1)=0.0%`。 |
| "数据集区分性要强一点" | 不用改数据集：**同分对留着**（那就是要解决的），问题在于旧 config 在这些对上是死组。 |

## 2. 已跑 run 的证据

`logs/jmigu28k_metrics.csv`（v3b，477 步）与 `logs/rl_v2_key_metrics.csv`（v2，500 步）：

| val 指标 | step 0 | step 200 | step 470/500 | 解读 |
|---|---|---|---|---|
| `pairwise_acc` | 0.277 | 0.293 | 0.290（v2 最好 0.383） | 平，≈猜 |
| `pairwise_score_diff` | -0.0138 | +0.003 | +0.001 | judge 给 A/B 的分几乎一样 |
| `positive_mean` / `negative_mean` | 0.6965 / 0.7104 | 0.7932 / 0.7914 | 0.7741 / 0.7806 | 两侧一起上升 = 分数通胀，不是判别力 |
| `parsed_final_score` | 0.71 | 0.79 | 0.73 | 通胀后回落 |
| `critic/rewards/mean`（train） | 0.72 | 0.61 | 0.69 | 看着健康，其实是抛硬币率（val n=1 时 0.38） |
| `actor/entropy` | 0.44 | 0.26 | 0.33 | 多样性掉了，没学到 |

`training/point_count/by_pair/group_size_mean: 8.0` → pair 分组（2 侧 × 4 rollout）是完整的，不是分组 bug。

## 3. 旧 reward 在这份数据上到底做了什么

旧 train 文件实测（`rl_pairwise_train_allwrong.verl.jsonl` 4000 对）：
**63.0% 是 judge 两侧 `final_score` 完全同分**，`|gap|` p50 = 0.00 / p75 = 0.25 raw 分；
旧 val 300 对则 0% 同分、`|gap|` p50 = 0.67 raw 分（所以旧 val 根本测不出"同分有没有被解决"）。

按配置算期望 reward（用离线 gap 当 rollout 分差的代理）：

| config | E[r_A] | E[r_B] | 系统性对比 r_A-r_B | 同分对 r_A / r_B | 处于有效带 (0.05<r<0.95) |
|---|---|---|---|---|---|
| hard-step, margin=0.05（**旧配置**） | 0.000 | 0.019 | -0.019 | **0.000 / 0.000** | 0.0% |
| hard-step, margin=0 | 0.014 | 0.356 | -0.342 | 0.000 / 0.000 | 0.0% |
| soft, tau_raw=2.0（＝旧 tau 0.2） | 0.485 | 0.515 | -0.030 | 0.500 / 0.500 | 100% |
| soft, tau_raw=1.0 | 0.470 | 0.530 | -0.059 | 0.500 / 0.500 | 100% |
| **soft, tau_raw=0.5** | 0.443 | 0.557 | -0.113 | 0.500 / 0.500 | 100% |
| soft, tau_raw=0.3 | 0.413 | 0.587 | -0.173 | 0.500 / 0.500 | 100% |
| soft, tau_raw=0.2 | 0.386 | 0.614 | -0.227 | 0.500 / 0.500 | 99.4% |

结论：
1. **硬阶跃 + margin=0.05 → 同分对两侧 reward 都=0**：表里 E[r_A]=E[r_B]=0.000，
   即"确定性同分"下整组 reward 完全相同、`pair_reward_std=0`、advantage 恒 0。
   在线时 rollout 之间总有些微差异（所以实测 train reward ≈0.6 是抛硬币），
   但 E[r_A]≈0.000 说明这些差异**不带系统性方向**，只是噪声；剩下 37% 的 A 侧也几乎从不越线（`P(R=1)=0.0%`）。
2. 旧 `tau=0.2`（归一化）等价于 `tau_raw=2.0`：即便换成软阈值，只要 tau 这么大，
   系统性对比只有 -0.030，基本还是"两侧都 0.5"——这就是为什么 v2/v3b 只是在通胀分数。

## 4. 同分对上的梯度（为什么 tau 是唯一的解）

同分对（gap=0）时两侧 reward 都=0.5，梯度体现在"**改善多少 → reward 抬多少**"：

| tau_raw | 同分对 +0.5 raw 分改善后 r | 增量 |
|---|---|---|
| 2.0（旧 tau=0.2） | 0.562 | +0.062 |
| 1.0 | 0.622 | +0.122 |
| **0.5** | 0.731 | +0.231 |
| 0.3 | 0.841 | +0.341 |
| 0.2 | 0.924 | +0.424 |
| hard-step（任意 margin） | 0.000（不越线永远 0） | 0 |

## 5. 本版改了什么（代码）

| 文件 | 内容 |
|---|---|
| `RM_RL-main/.../pairwise_image_edit_reward_manager.py` | 新增 `soft_threshold_format`（`sigmoid((s-thr-margin)/tau)`，可 mean/median）；`tau_raw` 支持用 raw 分标定；`use_dynamic_threshold` 真正实现（τ 下限 = 实测的对侧分数噪声，见 `threshold_std/dynamic_tau`）；新增 `pairwise_gap / pair_reward_std / pair_reward_range / threshold_valid`；`threshold` 不再输出 NaN |
| `scripts/run_rl_v4_train.sh` | **数据仍是旧 hard4k**；`REWARD_TYPE=soft_threshold_format`、`PAIRWISE_MARGIN=0`、`PAIRWISE_TAU_RAW=0.5`；实验名带 tau 便于 sweep；额外挂上同分诊断 val |
| `scripts/run_rl_tau_sweep.sh` | 三个 arm（`2.0` / `0.5` / `0.3+dynamic`）一条命令打印或串行跑 |
| `scripts/test_soft_threshold_reward.py` | 10/10：新增"同分对在旧硬阶跃下 reward 全 0（死组）"与"软阈值下有梯度"两个回归用例 |
| `scripts/build_rl_tie_val.py` | 可选：额外加一个 240 对 judge 同分的诊断 val（`data_source=urm_t2i_pairwise_tie`）。**默认不挂**（`USE_TIE_VAL=0`），同分是否被打破建议用后置复评的方式看（§6） |
| `scripts/audit_rl_pairwise_data.py` | 增加 label 分数段与配置对比；`build_rl_pairwise_hard.py`（label-margin 版本，可选）默认**保留**同分对 |

单测回归用例锁定的行为：
* 同分对 + 旧硬阶跃（margin 0.05）→ 四个 rollout reward 全相同、`pair_reward_std=0`（无梯度）；
* 同分对 + `soft τ_raw=0.5` → 两侧 0.5；A 侧提高 0.5 raw 分 → 0.731（有梯度）。

## 6. 怎么跑（不改数据，只跑 tau）

```bash
cd /efs/jacuzzitang/projects/URM
python3 scripts/test_soft_threshold_reward.py

bash scripts/run_rl_tau_sweep.sh             # 只打印 3 条命令，先看一眼
RUN=1 STEPS=80 bash scripts/run_rl_tau_sweep.sh   # 想直接串行跑就加 RUN=1
```
三个 arm 之间**只有 tau 不同**，数据/LR/KL/LoRA/rollout 与 v3b 完全一致，便于对照。

验证集就用原来的 300 对（全是"之前 judge 做不对"的对，`data_source=urm_t2i_pairwise`），
与 v2/v3b 的 `pairwise_acc` 可直接对比。要注意它的局限：这 300 对是当时**挑过 gap 的**
（0% 完全同分，|gap| p50 = 0.67 raw 分），所以它不单独度量"同分"这一类。
想单独看同分，用后置复评：拿合并后的 checkpoint 对 pool 里 1033 对 `difficulty=tie`
跑一遍 `scripts/score_pairs_for_rl.py`，比较 tie 率 / acc 与 base 的差异（不需要训练期挂 val）。

要看的指标（旧 val 与 tie val 分开看）：

| 指标 | 预期变化 | 含义 |
|---|---|---|
| `val-core/urm_t2i_pairwise/acc`（原 300 对） | 起点 0.277，**应上升** | "之前做不对的对"变对了多少（与 v2/v3b 同一口径） |
| `val-aux/urm_t2i_pairwise/pairwise_score_diff/mean@1` | 起点 ≈ -0.014，**应转正/变大** | 两侧是否被拉开（同分/近似同分在变少） |
| `critic/rewards/mean`（train） | 应**高于**抛硬币底线（≈0.5 + format_bonus），而不是只抖动 | 同分组不再"全 0"的直接体现 |
| `val-aux/*/soft_threshold_reward` | 0.4–0.7 且会动 | 渐变 reward 生效 |
| `val-aux/*/pair_reward_std` | **> 0** | =0 就是死组（旧配置在同分对上就是 0） |
| `val-aux/*/threshold_std`, `dynamic_tau` | 观察 | 实测 rollout 分数噪声；τ 低于它就该开 `use_dynamic_threshold` |
| `val-aux/*/parsed_final_score` | 稳定 | 上升 = 分数通胀（老毛病） |

### 6.1 把结果发回来分析（跑起来之后）

```bash
# ① /efs 上导出逐步指标（wandb 在线；离线就用 wandb sync 后再导）
/efs/jacuzzitang/projects/verl_venv/bin/python scripts/export_wandb_metrics.py --latest --curate

# ② 本机（/group/...）拉回来；-R 配合 /./ 保留相对路径
REMOTE=your_user@efs_host; SRC=/efs/jacuzzitang/projects/URM; DST=/group/40063/jacuzzitang/projects/URM
rsync -avR "$REMOTE:$SRC/./logs/rl_t2i_v4_tau0.5.log" \
           "$REMOTE:$SRC/./logs/<RUN_ID>_metrics.csv" \
           "$REMOTE:$SRC/./logs/<RUN_ID>_key.csv" \
           "$REMOTE:$SRC/./output/val_dump/stage6_pairwise_rl_tau0.5_v4/" \
           "$DST/"
```
只要 `logs/rl_t2i_v4_tau0.5.log` + metrics csv 这两样，就能把 §2 的表和 τ 响应都算出来
（val dump 用来逐对看 rollout 文本/分数，可选）。想看同分是否被打破，做一次**后置复评**：

```bash
CKPT=output/rl_t2i/merged_v4tau05_step80          # run_rl_bench.sh 产出的 merged full model
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i /usr/local/bin/python3 scripts/score_pairs_for_rl.py \
    --gen data/raw/gen_sft_data.json --gen-root data/extracted/gen \
    --hpdv3 data/hpdv3pp/train/train_tf.json --hpdv3 data/hpdv3pp/train/train_aes.json --hpdv3-root data/hpdv3pp \
    --ckpt "$CKPT" --full-model --sample-per-file 2000 --shard $i --num-shards 8 \
    --out-dir output/rl_hard_sel_v4tau05 > logs/score_v4_$i.log 2>&1 &
done
wait
python3 scripts/score_pairs_for_rl.py --merge --out-dir output/rl_hard_sel_v4tau05
# → 回传 output/rl_hard_sel_v4tau05/analyzed.jsonl，与 base 的 analyzed.jsonl 做同口径对比
```

## 7. tau 怎么选 / 这次能拿到多少增益（要诚实）

* **能拿到的**：把 63% 同分对从"零梯度死组"变成有梯度的组；把"改善多少"变成 reward 的
  量化响应（0.3–0.5 raw 分的改善分别对应 +0.34 / +0.23）；把旧配置下 0% 的越线率
  变成连续可调的 sigmoid。
* **拿不到的**：τ 不会给模型"看出差别"的能力。旧 val 上 judge 离线是系统性偏向错的一侧
  （|gap| p50 0.67 raw 分），所以 `r_A` 仍会偏低；如果策略在各对之间无法分辨，
  reward 只能推动"分数更极端"而不是"判得更对"。**所以必须看 tie val 的 acc/score_diff 是否真的在动**，
  而不是看 train reward 涨没涨。
* **推荐起点**：`tau_raw=0.5`、`margin=0`（保守）；想更激进用 `tau_raw=0.3` 并且
  `PAIRWISE_USE_DYNAMIC_THRESHOLD=True`（τ 下限跟实测噪声走，避免在噪声上刷分）。
  池子 |gap| p75 = 0.25 raw 分，所以再往小（≤0.2）就接近硬阶跃了，容易退回高方差抛硬币。
* 如果 tie val 的 acc 在两三个 arm 之间都不动，那说明瓶颈确实在"模型看不出差别"，
  这时才需要动数据（label margin 版本 `output/rl_label_hard_sel/` 或换 prompt/加负例），
  而不是继续调 τ。

## 8. threshold_margin（门槛）怎么定 —— τ 要跟它配对

`R = sigmoid((gap − threshold_margin) / τ)`，两者都用 raw 分衡量（`THRESHOLD_MARGIN=0.05` ＝ 0.5 raw 分）。
**σ 的区分度峰值出现在"工作点 = 门槛"处**，所以门槛要设在"当前典型分差"附近，而不是你想要的最终目标：

| τ_raw | m_raw | R(0) | R(0.25) | R(0.5) | R(1.0) | 斜率@0（相对最大） |
|---|---|---|---|---|---|---|
| 0.50 | 0.00（这次跑的） | 0.500 | 0.622 | 0.731 | 0.881 | 0.50（100%） |
| 0.25 | 0.00 | 0.500 | 0.731 | 0.881 | 0.982 | 1.00（100%） |
| 0.25 | **0.20（推荐起点）** | 0.310 | 0.550 | 0.769 | 0.961 | 0.86（86%） |
| 0.25 | 0.30 | 0.231 | 0.450 | 0.690 | 0.943 | 0.71（71%） |
| 0.25 | 0.50（"0.5 以下算 0"） | 0.119 | 0.269 | 0.500 | 0.881 | 0.42（42%） |
| 0.10 | 0.50 | 0.007 | 0.076 | 0.500 | 0.993 | 0.07（3%） |

* 硬阶跃 + m_raw=0.5（＝v2/v3b 的 `mean_threshold_format, threshold_margin=0.05`）：低于门槛的对**两侧都拿 0**
  → 组内 reward 完全相同 → advantage 恒 0（死组，实测 winner 侧 `P(R=1)=0.000`）。所以"低于门槛给 0"只能用软阈值实现。
* 直接上 m_raw=0.5 的代价：只保留 42%（τ=0.25）甚至 3%（τ=0.1）的梯度；一旦 σ 饱和，pairwise 项近似常数，
  组内 reward 的差异就由 format/penalty 决定 → 学到的还是格式。**加门槛必须同时把辅助项降权。**
* 建议课程：`m_raw 0.2 (THRESHOLD_MARGIN=0.02) + τ_raw 0.25` → 等 val 的 gap 中位数升到 0.4~0.5 raw 再加到 `m_raw 0.5`。
* 定 m 用数据而不是拍脑袋：reward manager 新增 `gap_ge_0p1/0p25/0p5/1p0` 四个 0/1 指标，
  在 val 上取均值即为"已有多少比例达到该门槛"（val 指标里直接可见）。

## 9. 代码核查：分组链路是对的，缺的是"组内区分度"的测量

逐行核过 pairwise 链路（`RM_RL-main/verl/`）：

| 环节 | 位置 | 结论 |
|---|---|---|
| dataset 每行带 `extra_info`(pair_id/side) | `utils/dataset/rl_dataset.py:438` | ✅ |
| `advantage_uid = extra_info[pair_id]` | `trainer/ppo/v1/trainer_base.py:1333-1345` | ✅ |
| advantage 计算读取 `advantage_uid` | `trainer_base.py:1638-1650` | ✅ |
| GRPO 用它替换 uid 作为组 id | `trainer/ppo/v1/utils.py:200-206` | ✅ |
| `adv = (r − 组均值)/(组std+1e-6)` | `trainer/ppo/core_algos.py:330` | ✅ |
| reward 按 `pair_id` 在 batch 内配对并回写 | `pairwise_image_edit_reward_manager.run_batch` | ✅ |

反证：`valid_counterparts≈1.0`、`training/point_count/by_pair/group_size_mean=8.0`（16 对 × 8 rollout）、
`critic/advantages` 有 ±2.2 的量级 → 分组与 advantage 都真的算出来了，**没有发现代码 bug**。

⚠️ 顺带一个反面结论：fork 里自带的 DAPO 组过滤（`algorithm.filter_groups`）是**按"同一 prompt 行的 n 个 rollout 的 metric 是否完全相同"**判定的
（`replay_buffer.py:286`），而我们的信号恰恰是"A 侧一致 / B 侧一致"的跨侧对比 → 开了会把最干净的样本剔掉，**不要开**。

**为什么"组内区分不开"**：GRPO 组内归一化后，一对能提供的信号只有组内差异
`Δr_within ≈ (1/(4τ_raw)) × σ_s`，其中 `σ_s` = 同一侧 4 个 rollout 的 `final_score` 标准差。
τ_raw=0.5 时系数只有 0.5/raw 分；若 4 个 rollout 分数相同 → Δr = 0 → **该对零梯度**。
（更正上一轮的误读：val 是 n=1，`r_A ≡ r_B` 恒成立，所以那里 `pair_reward_std≈0.001` 是构造使然，**不能**当作"组内无差异"的证据。）

现有指标看不到这个量（val n=1、bench dump 每 prompt 1 条、离线池每图只采一次）。
**为了不改动训练代码，这里不做新的观测埋点**；用已有指标做代理判断：

* `pairwise_score_diff`（val，mean）= 该池上唯一的"分得开吗"读数；τ 调小后它若不动，说明组内本来就没有差异；
* `critic/advantages/max|min` 在 ±2 附近而 acc 不动 → 优势被个别离群 rollout（格式/截断）主导，而不是被系统性的分数差主导；
* 离线佐证：judge 的 `final_score` 高度集中在少数吸引值（8.24 占 15.6%、7.41 占 9.5%、8.99 占 9.2%），
  且分数主要由**图像**决定 → 同一侧 4 个 rollout 共享同一张图，分数很容易撞同一个值 → 组内差异天然很小。

如果 τ=0.25 那一轮 `pairwise_score_diff` 仍然纹丝不动，下一步的候选（都需要改代码，先不动）：
让组内差异变大——提高 rollout 温度、或把比较量换成 `final_score + w×清单通过率`（离线挑难例用的就是这个 combined 约定）；
实在不行才回到数据/prompt 层。

困难样本的可视化（不改任何训练代码）：`scripts/build_hard_pairs_report.py`
→ `output/hard_pairs_report.html` + `_sheet1/_sheet2.png`（按 judge 分差和 label 分差排序，可按桶过滤）。


## 10. 数据 × 门槛 的活组表（2026-09-15 更新，最终数据方案）

reward = GRPO 组内标准化，**组 = 同一对**，所以"活组"（有梯度的组）要求组内 reward 不能全同。
n=8、σ_s(同侧分数抖动)≈0.35 分下的活组率：

| 数据桶（该对真实分差 Δ） | m=0（硬公式） | m=0.5 分 |
|---|---|---|
| easy（Δ>0.5，模型早判对） | 5%（死） | 99% |
| near（0<Δ≤0.5） | 64% | 99% |
| tie（Δ=0） | 99% | 65% |
| judge-wrong（Δ<0） | 65% | 5%（死） |

**结论：数据 mix 和 m 必须配对。"wrong 为主"只能用 m=0；"easy/near 为主"才配 m=0.5 分。**
（离线实测佐证：hard4k 池上 winner 侧越线率 m=0.5 分时为 0.0%，m=0.2 分为 0.7%。）

**最终训练集**：`output/rl_mix_wrongheavy/rl_pairwise_train_wrongheavy.verl.jsonl`
= judge 同分 2521 (44%) + judge 判错 1441 (25%) + near 1203 (21%) + easy 535 (9%)，共 **5700 对**，
A/B 相邻（batch 内成对）已校验，配套门槛 `THRESHOLD_MARGIN=0.0`。
可执行命令见 `RL_V4_RUNBOOK.md`。

**降 tie 的现成捷径**：评测侧用 `combined = final + 0.1×清单通过数` 做 tie-break，base 的 tie 率
13.3%→6.1%（genai）、20.7%→9.9%（mmrb2），零 RL 成本；只有当部署侧只能用 `final_score` 时才需要靠 RL 教它"在 final_score 上表达果断"。

## 11. 其他注意事项

1. 离线 `difficulty` / gap 是在**评测 prompt** 下打的，RL 用另一套模板 + temp 0.7，在线判别力更弱
   （同一批 300 对：离线 gap −0.61 raw，在线只有 −0.14）。所以起点绝对值会低于离线估计，看趋势。
2. 同分是"两侧 `final_score` 完全相等"（EPS=1e-9）；pool 里 1033 对整体被判 tie、
   另有 2521 对 raw 同分但 combined 判错（归在 wrong 里）。tie val 取的是前者（held-out）。
3. reward 是相对的、没有绝对锚点，通胀始终是可能的吸引子：`parsed_final_score` +
   `pair_reward_std` 用来区分"真学到"和"通胀"。
4. 额外的同分诊断 val 默认关闭（`USE_TIE_VAL=0`）；想临时挂上就 `USE_TIE_VAL=1`，
   前提是先 `python3 scripts/build_rl_tie_val.py`（只加 val，train 一个字节都不动）。
