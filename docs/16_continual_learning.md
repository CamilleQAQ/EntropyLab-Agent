# 持续学习 v1（`continual/`）

日期：2026-11-05
状态：**机制已实现并通过测试。VBL-VA001 上做了 5 个独立划分：
continual 0.9830 ± 0.0074 vs frozen 0.9800 ± 0.0069，只有 1/5 个划分胜出，
配对差 +0.0030 [0.0000, +0.0070]，McNemar p = 0.25 —— 统计上不能称为正向。**

## 1. 设计（按讨论确认的三条）

| 决策 | 选择 |
|---|---|
| 批次来源 | **显式 manifest**（文件/组级、互斥、可审计） |
| Agent 角色 | **固定配置只重训**（用 Agent 选出的特征族 + 分类器，不在每批重新选型） |
| Promote 判据 | challenger 必须超过 champion **至少 0.005** |

每批规则：

```text
训练集   = 全部历史批次 ∪ 当前批次（只增不减，永不重切）
validation = 冻结，每批都用它
holdout    = 冻结，最后只评一次
challenger = 在训练集上重训
promote 当且仅当 challenger validation macro-F1 ≥ champion + 0.005
```

产物：`batches.json`、`decisions.json`、`report.md`、`metrics.json`、
`challengers/batch-N/`、`joint/`。

## 2. 实现

| 文件 | 内容 |
|---|---|
| `continual/plan.py` | `BatchPlan`：格式校验、互斥校验、类别覆盖校验、`masks()`（含泄漏断言） |
| `continual/loop.py` | `run_continual()`：逐批训练/比较/提升，三臂 holdout（含逐记录预测）+ 报告 |
| `scripts/prepare_continual_plan.py` | 通用策略：`chronological`、`by-load`（含 `--stratified-validation`） |
| `scripts/prepare_vbl_continual_plan.py` | VBL 专用：unit 级 5 分区，`--seed` 控制单元打乱 |
| `scripts/aggregate_continual_runs.py` | 跨划分聚合：mean±std、配对差、逐记录 bootstrap、McNemar |
| `cli` | `entropylab continual run --spec ... --plan ... --output ...` |
| `tests/continual/` | 6 个测试：重复组、漏组、holdout 泄漏、只增不减、promote 规则、计划往返 |

295 tests 全过。

## 3. Aero：机制跑通，但没有信息量

3 批 × 2 组/类（每批 60 条），validation 10 组，holdout 10 组。

| 批次 | 训练记录 | challenger | champion | 决定 |
|---|---:|---:|---:|---|
| 0 | 60 | 1.0000 | 1.0000 | promote（初始） |
| 1 | 120 | 1.0000 | 1.0000 | reject |
| 2 | 180 | 1.0000 | 1.0000 | reject |

holdout 三臂全 1.0000。aero 已饱和，只能证明循环能跑通。

## 4. 齿轮运行 A：所有更新都被拒绝

10 Nm（14 组）→ 15 Nm（7 组）→ 20 Nm（6 组），validation 5 组、holdout 5 组。

| 批次 | 训练记录 | challenger | champion | 决定 |
|---|---:|---:|---:|---|
| 0 (10 Nm) | 84 | 0.6800 | 0.6800 | promote（初始） |
| 1 (15 Nm) | 126 | 0.6473 | 0.6800 | **reject** |
| 2 (20 Nm) | 162 | 0.6473 | 0.6800 | **reject** |

| 臂 | holdout accuracy |
|---|---:|
| frozen | 0.6333 |
| continual | 0.6333 |
| joint | 0.7667 |

原因：validation 只有初始域，加新载荷后旧域掉 3 个点 → 必须拒绝 → 退化为“永不更新”。

## 5. 齿轮运行 B：validation 与 holdout 不同分布

把 validation 改成按载荷分层（10 组）后：

| 批次 | challenger | champion | 决定 |
|---|---:|---:|---|
| 0 | 0.1084 | 0.1084 | promote |
| 1 | 0.1128 | 0.1084 | reject |
| 2 | 0.1490 | 0.1490 | promote |

但 validation 只有 0.11–0.15，holdout 却有 0.80：validation 覆盖三个载荷层（难），
holdout 只有 5 组落在一个层（易）。**选模型和报结果用的不是一把尺子。**

## 6. VBL-VA001：一个看似正向、实则站不住的结果

数据集：实验室台架振动，20 kHz，4 类（normal / bearing / misalignment / unbalance），
4000 个 CSV、739 个独立 unit。适配器 `scripts/prepare_vbl_4class.py` 每类抽样 250 文件、
每文件 1 个 2048 点窗口；**group 隔离按 unit**。

单次划分（seed 20261104、未打乱 unit）：

| 批次 | 训练记录 | challenger | champion | 决定 |
|---|---:|---:|---:|---|
| 0 | 260 | 0.9472 | 0.9472 | promote |
| 1 | 442 | 0.9472 | 0.9472 | reject（持平） |
| 2 | 654 | 0.9646 | 0.9646 | promote |

| 臂 | holdout accuracy（234 条） |
|---|---:|
| frozen | 0.9017 |
| continual | **0.9188** |
| joint | 0.9188 |

看起来是 +1.7 点且追平 joint。**但这个结论是错的**：它是单次划分的端点，
而且 1.7 点只等于 234 条里的 4 条记录。

## 7. 五次独立划分：结论反转

同一数据集、同一配置，只改变 unit 如何分配到各分区（`--seed` 打乱），跑 5 次：

| seed | frozen | continual | joint | 关键事件 |
|---:|---:|---:|---:|---|
| 20261104 | 0.9750 | 0.9750 | 0.9850 | validation 第 0 批就 1.0000，后续全 reject |
| 1 | 0.9900 | 0.9900 | 0.9950 | 同上 |
| 2 | 0.9756 | 0.9756 | 0.9951 | 同上 |
| 3 | 0.9845 | 0.9845 | 0.9948 | 同上 |
| 4 | 0.9747 | 0.9899 | 0.9899 | 第 0 批 validation 0.9599，才有提升空间 |

聚合（`scripts/aggregate_continual_runs.py`）：

| 臂 | accuracy | macro-F1 |
|---|---:|---:|
| frozen | 0.9800 ± 0.0069 | 0.9799 |
| continual | 0.9830 ± 0.0074 | 0.9828 |
| joint | **0.9920 ± 0.0045** | 0.9919 |

| 比较 | 结果 |
|---|---|
| continual − frozen | **+0.0030 ± 0.0068**，5 个划分里**只赢 1 个**，其余 4 个打平 |
| continual − joint | **−0.0090 ± 0.0073**，5/5 全部落后 |
| 逐记录配对 bootstrap | +0.0030，95% CI **[0.0000, +0.0070]**（下界贴 0） |
| McNemar（配对） | 只有 continual 对 3 条、只有 frozen 对 0 条，**p = 0.25** |

**结论：不能称为正向。** 方向是正的，但 5 个划分里 4 个根本没有更新被接受，
统计检验也不显著，而且明显低于 joint。

## 8. 为什么单次和五次差这么多：validation 饱和

5 个划分的共同点：第 0 批的 validation macro-F1 就已经是 0.98–1.00。
判据是 `challenger ≥ champion + 0.005`；当 champion 已经 1.0000 时
**不存在能超过它的分数**，于是每一批都被拒绝——机制退化成“永不更新”。

只有 seed 4 的第 0 批 validation 是 0.9599（留了空间），才发生 promote；
也只有那个划分里 continual 高于 frozen（0.9899 vs 0.9747），并追平 joint。

所以问题不在“持续学习无效”，而在**判据在验证集饱和时无法触发**。
这是可修复的设计缺陷，但修复前不能对外宣称正向结果。

## 9. 结论

成立的：

1. 批次计划可审计（互斥、全覆盖、holdout 永不入训练集，均有测试）；
2. champion/challenger 规则按设计执行（持平会被拒绝，不是无脑接受）；
3. 三臂对比、逐记录预测、跨划分聚合与配对检验都能自动产出；
4. 三个可复现的失败模式：初始域 validation（gear A）、validation/holdout 不同分布（gear B）、
   validation 饱和（VBL 的 4/5 个划分）。

不成立的（不许粉饰）：

1. **“持续学习在 VBL 上提升 1.7 点”**：那是单次划分的端点；5 次划分后是 +0.3 点、
   只有 1/5 胜出、p = 0.25；
2. **“追平 joint”**：5 个划分全部落后约 0.9 点；
3. 因此目前**没有任何证据表明持续学习在本仓库提升了最终诊断**。

## 10. 下一步（按优先级）

1. **修判据（必须）**：champion 已到满分时应允许“不退化即接受”，
   或改用不会饱和的判据（分层 validation / 逐记录配对比较）。
   这是当前唯一挡住结论的环节。
2. **验证修复**：同一 5 个划分重跑，看 continual 能否追平 joint；能，才算正向。
3. **遗忘探针**：每批留一组永不训练的 probe，量化旧域退化。
4. **更多轮次**：VBL 的 unit 数够切 5–6 轮。

## 11. 复现

```powershell
# VBL-VA001：打包
python scripts/prepare_vbl_4class.py `
  --source ..\VBL-VA001\VBL-VA001 --output data/paper/vbl_4class `
  --window 2048 --per-class 250

# 5 个独立划分 + 运行
foreach ($seed in 20261104,1,2,3,4) {
  python scripts/prepare_vbl_continual_plan.py `
    --manifest data/paper/vbl_4class/manifest.json `
    --output data/paper/vbl_4class/continual_plan_seed$seed.json `
    --dataset-id VBL_4CLASS_FUSION_TIME_EXTRA_TREES --seed $seed
  python -m entropy_agent.cli continual run `
    --spec configs/vbl_4class_fusion_time_extratrees.json `
    --data-root data/paper/vbl_4class `
    --plan data/paper/vbl_4class/continual_plan_seed$seed.json `
    --output results/continual-vbl-seed$seed-20261105
}

# 聚合
python scripts/aggregate_continual_runs.py `
  --runs results/continual-vbl-seed*/metrics.json `
  --output results/continual-vbl-aggregate-20261105/summary.json
```
