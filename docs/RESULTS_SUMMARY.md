# URM stage4/stage5 评测与改进总结

日期：2026-09-01 · 数据：同一 GenAI-Bench image_generation (1306 对) + MMRB2 t2i (1000 对)

## 一、最终对比（raw = 模型 final_score；combined = final_score + 0.1×pts_sum）

### raw（模型直接输出的分数）

| 模型 | genai acc / no_tie / tie | mmrb2 acc / no_tie / tie |
|---|---|---|
| stage3 ck148（5k 数据） | 0.517 / 0.685 / 24.5% | 0.438 / 0.640 / 31.5% |
| stage4 ck250 | 0.527 / 0.703 / 25.1% | 0.466 / 0.663 / 29.8% |
| **stage4 ck500** | **0.545 / 0.698 / 22.0%** | 0.469 / 0.680 / 31.0% |
| stage4 ck552 | 0.535 / 0.697 / 23.2% | **0.470 / 0.674 / 30.3%** |
| stage5 ck500（cap-10） | 0.546 / 0.708 / 22.8% | 0.458 / 0.670 / 31.6% |
| stage5 ck526 | 0.543 / 0.707 / 23.2% | 0.469 / 0.681 / 31.1% |

### combined（final + 0.1×checklist pts_sum，无需重训）

| 模型 | genai acc / tie | mmrb2 acc / tie |
|---|---|---|
| stage3 ck148 | 0.596 / 8.8% | 0.547 / 10.2% |
| stage4 ck250 | 0.609 / 9.5% | 0.541 / 14.5% |
| **stage4 ck500** | **0.619 / 8.3%** | **0.550 / 14.8%** |
| stage4 ck552 | 0.611 / 9.3% | 0.547 / 13.3% |
| stage5 ck500 | 0.613 / 9.7% | 0.529 / 16.2% |
| stage5 ck526 | 0.610 / 9.8% | 0.536 / 15.9% |

## 二、三条核心结论

1. **stage4（20k 数据）是当前最优**：mmrb2 比 stage3（5k）raw acc +3~4 点、VAL pearson 0.67→0.69；genai 提升小（+2.8 点）。stage4 内 ck500/ck552 几乎持平。
2. **stage5（删 10 分样本 cap-10）是无效实验**：训练集 10 分 2352→1513（9%），但 tie 仍是 23%/31%、raw acc 打平、val eval_loss 相同（0.3608 vs 0.3595）。→ **tie 是结构性根因**（pointwise 回归 + 分数钉在 10/7.5/8.5 等地标值），数据侧删样本解决不了。
3. **checklist 组合分是唯一稳赢杠杆**：对所有模型都稳定把 tie 砍到 8–16%、acc +5~8 点。**best = stage4 ck500 + final+0.1×pts_sum**（genai 0.545→0.619、mmrb2 0.469→0.550）。

## 三、bad-case 分析（tie vs no-tie，6 个 checkpoint）

- **错误高度系统化**：约 1/4 的 pair 6 个模型全错；相邻 ckpt 全错 Jaccard ~0.8，跨代（stage3 vs 4）~0.54。
- **tie 错误 = 分数饱和在 10/0 地标**（如“tiger in tuxedo”全部模型 10/10），且跨模型、跨代、跨删 10 分实验都稳定出现 → 这是系统性病灶。
- **用组合分后**：系统性 tie 几乎清零（6 模型全错 tie 从 6–10% → <1%），但 no-tie 偏好方向错误上升为主（全错降到 ~15%，no-tie 跨代 Jaccard 仍 ~0.3–0.4）→ 剩下的是**偏好校准/品味**问题。

## 四、系数敏感性 & checklist-only

- `final + w×pts_sum` 在 **w=0.02~0.3 几乎平台**（acc 不变、tie 都 ~8%/15%），w=0.5+ 才略降 → **w=0.1 稳健默认，无需精调**。
- **只用 checklist 分数明显更差**（pts_sum/mean 单独用都不如 final、更不如组合）→ final（全局校准）+ checklist（细粒度分离）互补，缺一不可。

## 五、落地建议

- **最终模型**：`output/skyjm_gen_9b_stage4/v0-20260831-013010/checkpoint-500`
- **排序/reward 打分**：`final_score + 0.1 × pts_sum`
- **下一步（真正治本）**：pairwise/margin loss（用偏好对训分离、校准品味），目标同时改善 tie 与 no-tie。

## 六、工具（本仓库 scripts/）

| 脚本 | 用途 |
|---|---|
| `rerun_stage4_benchmarks.sh` / `run_stage4_bm_lean.sh` | 跑 benchmark（多 checkpoint 串行） |
| `recompute_benchmark_checklist.py --output-dir <dir>` | 单目录 raw + combined 对比，写 `checklist_recompute.json` |
| `compare_benchmark_runs.py --run label:dir ...` | 多模型一键对比表，写 `output/benchmarks_compare.json` |
| `build_stage5_dataset.py --pct10-cap 10` | 按类别裁 10 分样本（cap 模式） |
| `run_stage5_pipeline.sh` | stage5 全自动链（等 ckpt250→筛→训→评） |

---

# 补充：two-stage（twostage stage-2 → stage-3）结果

日期：2026-09-02 · 同一数据（GenAI-Bench image_generation 1306 对 + MMRB2 t2i 1000 对）

流程：stage-2（score given points）训到 checkpoint-855 → merge 进 stage-0 基座 →
stage-3（generate + score e2e，80% 用 stage-5 的 3-dim v2 system prompt + 20% score-given-points）
训练 570/570 步（batch2/accum4、IMAGE_MAX_TOKEN_NUM=1024，显存 ~36.7GB）。
评测口径同仓库统一 recompute 脚本、同权重复算。

## stage-3 ck570（eval_loss 最优 = last，0.3575）vs stage4/5 单阶段最优

### genai_bench / image_generation
| 模型 | raw acc / tie | combined(pts_sum/10) acc / tie | kendall(combined) |
|---|---|---|---|
| **twostage stage3 ck570** | 0.526 / 24.6% | **0.612 / 9.9%** | 0.104 |
| stage4 ck500 | 0.545 / 22.0% | 0.619 / 8.3% | 0.103 |
| stage5 ck526 | 0.543 / 23.2% | 0.610 / 9.8% | 0.101 |

### mmrb2 / t2i
| 模型 | raw acc / tie | combined(pts_sum/10) acc / tie | kendall(combined) |
|---|---|---|---|
| **twostage stage3 ck570** | 0.469 / 32.1% | **0.563 / 14.6%** | **0.073** |
| stage4 ck500 | 0.469 / 31.0% | 0.550 / 14.8% | 0.061 |
| stage5 ck526 | 0.469 / 31.1% | 0.536 / 15.9% | 0.053 |

## 小结
- raw final_score 口径与单阶段 stage4/5 打平（mmrb2 0.469；genai 0.526 略低 1.5-2 点），
  tie 仍结构性偏高（24-32%）→ 必须配合 checklist 组合分使用。
- 叠上 `final + 0.1×pts_sum` 后：**mmrb2 0.563 反超 stage4/5 最优（0.550 / 0.536）**，
  genai 0.612 ≈ stage5、略低于 stage4 ck500（0.619）。kendall 也在 mmrb2 上最高。
- 结论：twostage"自主生成 checklist + 打分"在 mmrb2 上已体现价值，但 genai 仍略逊于
  单阶段最优 stage4 ck500 → 若追求 genai，可考虑 stage-3 从 stage4 ck500 起训再做一轮。

### 三档 checkpoint 全量（2026-09-02 补全，combined=final+pts_sum/10）
| twostage stage3 | genai comb acc/tie | mmrb2 comb acc/tie | 备注 |
|---|---|---|---|
| ck250 | 0.619 / 8.6% | 0.550 / 14.0% | genai 最优档 |
| ck500 | 0.614 / 9.1% | 0.547 / 14.7% | |
| ck570 | 0.612 / 9.9% | 0.563 / 14.6% | mmrb2 最优档 = last |

genai 上"越训越掉"（ck250→ck570 0.619→0.612）、mmrb2 上"越训越涨"（0.550→0.563）两方向相反
→ 两阶段相对单阶段仅打平、无稳定优势；stage6（细标签）已改走单阶段配方。

---

# 补充：stage6（细标签，2026-09-03）

## 一句话结论
把 Gemini 的**粗 final_score**（~25 个值、0.5 步长、13% 是 10 分）换成
**Gemini+stage0 对齐平均后的细标签**（只改 final_score，其余逐字节不动），单阶段重训
（stage4 配方，从 stage0 基座，2 epochs）→ **raw tie 几乎腰斩、raw acc 大涨 5-8 点**，
单阶段 + 细标签 = 当前最优路径。两阶段结构确认无价值。

## 细标签怎么造的
- stage0（RubricRM pointwise，0-4 连续）用 vLLM 给 18.6k 训练图打分（stage0 自己的 0-4 rubric prompt）。
- 最小二乘仿射 `aligned = a×stage0 + b`（全局一条线，以 Gemini 的 0-10 为目标标尺拟合；实测 a=1.656, b=1.851）。
- `fine = (1−w)×Gemini + w×aligned`（w=0.5），clip[0,10]，round 2；**只写回 eval_result.final_score**。
- 效果（18.6k 全量）：`corr(gemini,stage0)=0.632`；unique 25→409；0.5-grid 99.8%→8.3%；
  10 分饱和 13.3%→0%。

## 评测（vLLM，raw + combined=final+pts_sum/10）
### genai_bench / image_generation
| 模型 | raw acc / tie | combined acc / tie |
|---|---|---|
| **stage6 ck250** | **0.610 / 13.3%** | **0.647 / 6.1%** |
| stage6 ck500 | 0.589 / 15.2% | 0.638 / 6.7% |
| stage6 ck552 | 0.590 / 15.3% | 0.641 / 5.8% |
| stage4 ck500 | 0.545 / 22.0% | 0.619 / 8.3% |
| stage5 ck526 | 0.543 / 23.2% | 0.610 / 9.8% |
| twostage ck570 | 0.526 / 24.6% | 0.612 / 9.9% |

### mmrb2 / t2i
| 模型 | raw acc / tie | combined acc / tie |
|---|---|---|
| **stage6 ck250** | **0.521 / 20.7%** | 0.568 / 9.9% |
| **stage6 ck500** | 0.518 / 21.6% | **0.579 / 9.5%** |
| stage6 ck552 | 0.521 / 22.1% | 0.573 / 11.3% |
| stage4 ck500 | 0.469 / 31.0% | 0.550 / 14.8% |
| stage5 ck526 | 0.469 / 31.1% | 0.536 / 15.9% |
| twostage ck570 | 0.469 / 32.1% | 0.563 / 14.6% |

## 结论
- raw 口径 stage6（0.59-0.61 / 0.52）全面反超 stage4/5/twostage（0.54 / 0.47），tie 腰斩 →
  印证"粗标签/高分饱和是 tie 主因"。模型**原生输出**即可排序，不再依赖后处理 combined。
- 单阶段 + 细标签 > 两阶段 > 早期 stage4/5。stage-2（score-given-points）+ stage-3(e2e) 结构没有加分。
- 选 checkpoint：原生 final_score 用 `output/skyjm_gen_9b_stage6/v0-20260902-172103/checkpoint-250`
  （raw 双榜最优）；用 combined 排序用 `checkpoint-500`（genai 0.638 / mmrb2 0.579 最均衡）。
- 注意 ck552=last 是 eval_loss 最优(0.3628)但 bench genai 略降 → eval_loss 与 bench 并非完全一致；
  "早停点最优"在 genai 上再次出现。

## 融合逻辑与局限（备忘）
- 不是"逐图把 stage0 对齐到它自己的 Gemini 分"：仿射是**全局一条线**，同 Gemini 档内不同 stage0
  会得到不同 aligned / fine —— 这正是破 tie 的来源。
- aligned ≈ "用 stage0 预测 Gemini"，所以 `fine = 0.5×Gemini + 0.5×aligned` ≈ 把 Gemini 标签
  朝 stage0 能预测的方向收缩；**新增信息全部来自 stage0 在 Gemini 粗档内的连续排序**。
- 若 corr(stage0, gemini)→1 该方法会退化（aligned≈Gemini，无变化）；实测 0.632 + bench 提升证明
  档内排序是真信号。两裁判的方向性分歧会被平均掉一半（局限，未建模各自信噪比）。
- 更严谨替代（未做）：按信噪比/方差的最优线性组合，或用 stage0 在档内相对排序微调，或 latent-Q 建模。

## 备注
- 评测引擎：stage6 用 vLLM（tp=1，LoRA rank=32 需 `max_lora_rank≥64`）；此前 stage4/5/twostage 为
  transformers 引擎。vLLM 评测单 checkpoint 约 11-13 分钟（对比 transformers ~1.6h）。
- 环境：vllm 需要卸掉与 torch ABI 不匹配的 flash-attn（swift 训练用 sdpa，不依赖 flash-attn）。
