# Stage-5 实验：按类别“对齐式”削减 10 分样本 → 重训 → 自动评测

## 1. 方案（已定稿：cap 12%，温和且与其它分数对齐）

**不是删空某类**，而是把每类的 10 分占比**截断到目标值**（默认 12%）：
每类保留 `min(n_10, 12% × n)` 条 10 分，超出的随机删掉。这样高分泛滥的类别
被拉回与其它类别接近的水平，分数分布保持对齐，只删 ~650 条（train 剩 ~96.3%）。

- 受影响：10 分占比 >12% 的类别被裁（`multiling_text, text_combo, count_space,
  poster_anno, action_scene, text_few, text_many, attribute, aes_color, aes_detail`）
- 其余类别原样保留
- cap 对照（全量 18606 行 / train≈17670）：
  | cap | 删 10 分 | train 剩余 |
  |---|---|---|
  | 10% | ~886 | ~16829 (95.2%) |
  | **12%** | **~687** | **~17018 (96.3%)** |
  | 13.3%(均值) | ~565 | ~17133 (97.0%) |
  | 15% | ~437 | ~17255 (97.7%) |
- 训练：复用 `train_skyjm_gen_9b_stage4.sh`（stage0 基座 + 3dim_v2 + 同样超参）

## 2. 需要同步的文件（从本机 → trainer 的 `/efs/jacuzzitang/projects/URM/scripts/`）

```bash
scripts/build_stage5_dataset.py
scripts/train_skyjm_gen_9b_stage5.sh
scripts/run_stage5_pipeline.sh
```

## 3. 数据文件（trainer 上应已存在，先检查）

```bash
ls -la \
  stage1/t2i_v4_20k_checklist_gemini_scored_clean.json \
  stage1/t2i_v4_20k_final_prompt_3dim_v2_train.jsonl \
  stage1/t2i_v4_20k_final_prompt_3dim_v2_val.jsonl \
  stage1/t2i_v4_20k_final_prompt_3dim_v2.jsonl.indices.json \
  stage1/t2i_v4_20k_final_prompt_3dim_v2_val.jsonl.indices.json \
  output/skyjm_gen_9b_pointwise_stage0/v3-20260818-224412/checkpoint-1500-merged
```

缺任何一个先同步/生成（train/val/indices 由 `build_stage1_dataset.py` +
`split_stage1_dataset.py` 生成，stage0 基座在 trainer 上）。

## 4. 先看分析（不写文件，确认删哪些）

```bash
cd /efs/jacuzzitang/projects/URM
python3 scripts/build_stage5_dataset.py \
  --input stage1/t2i_v4_20k_final_prompt_3dim_v2_train.jsonl \
  --drop-ge 10 --pct10-threshold 15 --analyze-only
```

看到每个类别的 `keep10/drop10`，即每种 cap 下保留/删除的数量，确认即可继续。

## 5. 跑整条链（等 ckpt250 测完 → 筛数据 → 训 stage5 → 自动测 2 benchmark）

```bash
cd /efs/jacuzzitang/projects/URM
STAGE5_PCT10_CAP=12 \
nohup bash scripts/run_stage5_pipeline.sh > logs/stage5_pipeline.log 2>&1 &
echo "pid=$!"
```

链路内的阶段：
1. 轮询 `output/benchmarks_stage4/v0-20260831-013010/checkpoint-250/meta.json`
   直到 ckpt250 benchmark 完成（默认 60s 一次）
2. 生成 `stage1/t2i_v4_20k_final_prompt_3dim_v2_train_cat10.jsonl`（cap 12%）
3. `scripts/train_skyjm_gen_9b_stage5.sh` → `output/skyjm_gen_9b_stage5`
4. 自动取 best/last checkpoint 在 `genai_bench + mmrb2` 上评测 →
   `output/benchmarks_stage5/<run>/<ckpt>/`

## 6. 监控

```bash
tail -f logs/stage5_pipeline.log
tail -n 3 logs/stage4_bm_eval_shard_0.log   # 评测阶段打分进度
nvidia-smi
```

## 7. 调参

```bash
# 调 cap 目标（想多删/少删）
STAGE5_PCT10_CAP=13.3 bash scripts/run_stage5_pipeline.sh
STAGE5_PCT10_CAP=10   bash scripts/run_stage5_pipeline.sh
# 激进模式：删空 pct10>=15% 的类别（不推荐，删 ~1600）
STAGE5_PCT10_THRESHOLD=15 bash scripts/run_stage5_pipeline.sh
# 额外指定删/留
STAGE5_REMOVE_CATS="aes_color,portrait" \
STAGE5_KEEP_CATS="multiling_text,text_combo" ...
# 跳过等待（数据已筛好、直接训）
SKIP_WAIT=1 bash scripts/run_stage5_pipeline.sh
```

## 8. 跑完后对比

- raw vs checklist 组合分：`scripts/recompute_benchmark_checklist.py`
  `--output-dir output/benchmarks_stage5/<run>/<ckpt>`
- 最终对比表（stage3/4/5 × raw/combined）：同步结果回来，我帮你出。
