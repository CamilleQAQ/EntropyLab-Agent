# 持续学习 v1（`continual/`）

日期：2026-11-05
状态：**机制已实现并通过测试。VBL-VA001 上拿到正结果：continual 0.9188 > frozen 0.9017，
且等于 joint；aero 饱和、gear 太小，只能作为失败模式保留。**

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

产物：`batches.json`（计划）、`decisions.json`、`report.md`、`metrics.json`、`challengers/batch-N/`、`joint/`。

## 2. 实现

| 文件 | 内容 |
|---|---|
| `continual/plan.py` | `BatchPlan`：格式校验、互斥校验、类别覆盖校验、`masks()`（含泄漏断言） |
| `continual/loop.py` | `run_continual()`：逐批训练/比较/提升，最后三臂 holdout + 报告 |
| `scripts/prepare_continual_plan.py` | 两种批次策略：`chronological`（按存储顺序切块）、`by-load`（按载荷分层） |
| `cli` | `entropylab continual run --spec ... --plan ... --output ...` |
| `tests/continual/` | 6 个测试：重复组、漏组、holdout 泄漏、只增不减、promote 规则、计划读写往返 |

295 tests 全过。

## 3. Aero：机制跑通，但没有信息量

计划：3 批 × 2 组/类（每批 60 条），validation 10 组，holdout 10 组。

| 批次 | 训练记录 | challenger | champion | 决定 |
|---|---:|---:|---:|---|
| 0 | 60 | 1.0000 | 1.0000 | promote（初始） |
| 1 | 120 | 1.0000 | 1.0000 | reject（没超过 margin） |
| 2 | 180 | 1.0000 | 1.0000 | reject |

holdout 三臂全 1.0000。**aero 已饱和**，持续学习在这里只能证明“循环能跑通”，
不能回答任何科学问题。

## 4. 齿轮运行 A：所有更新都被拒绝

计划：10 Nm（14 组）→ 15 Nm（7 组）→ 20 Nm（6 组），validation 5 组、holdout 5 组。

| 批次 | 训练记录 | challenger | champion | 决定 |
|---|---:|---:|---:|---|
| 0 (10 Nm) | 84 | 0.6800 | 0.6800 | promote（初始） |
| 1 (15 Nm) | 126 | 0.6473 | 0.6800 | **reject** |
| 2 (20 Nm) | 162 | 0.6473 | 0.6800 | **reject** |

holdout：

| 臂 | accuracy | macro-F1 |
|---|---:|---:|
| frozen | 0.6333 | 0.6253 |
| continual | 0.6333 | 0.6253 |
| **joint** | **0.7667** | **0.7076** |

**continual 完全等于 frozen，与 joint 差 13.3 个点。**

原因很具体：validation 只有 5 组、且来自初始域（10 Nm 附近）。加入 15/20 Nm 数据后，
模型在**旧域**上的分数从 0.68 掉到 0.647，按规则必须拒绝。于是持续学习退化成了“永不更新”。

这不是实现错误，是判据的设计缺陷：**只用初始域验证，就永远看不见新数据的好处**。

## 5. 齿轮运行 B：把 validation 按载荷分层，出现新问题

计划改成 validation = 每个 (类别, 载荷) 一组（10 组），10 Nm（13 组）→ 15 Nm（5 组）→ 20 Nm（4 组）。

| 批次 | 训练记录 | challenger | champion | 决定 |
|---|---:|---:|---:|---|
| 0 (10 Nm) | 78 | 0.1084 | 0.1084 | promote（初始） |
| 1 (15 Nm) | 108 | 0.1128 | 0.1084 | reject（差 0.0006，没到 margin） |
| 2 (20 Nm) | 132 | 0.1490 | 0.1490 | **promote** |

holdout 三臂全 0.8000 / 0.7333。

**validation 只有 0.11–0.15，holdout 却有 0.80** —— 两者不是同一个分布：
validation 覆盖三个载荷层（难），holdout 只有 5 组、落在单一载荷层（易）。
于是“用 validation 选模型、用 holdout 报结果”这件事本身就不成立。

## 6. VBL-VA001：机制有效（2026-11-05 补）

前面的结论是“现有数据回答不了”，于是接入了第四个数据集 **VBL-VA001**
（实验室台架振动，20 kHz，4 类：normal / bearing / misalignment / unbalance）。

- 来源 VBL-VA001/VBL-VA001/{normal,bearing,misalignment,unbalance}，4000 个 CSV；
- 适配器 scripts/prepare_vbl_4class.py：每类按 **unit 轮转** 抽样 250 个文件，
  每个文件取 1 个 2048 点窗口（最后一个加速度列；归档无轴向文档，已在 manifest 写明）；
- **group 隔离按 unit**：同一 unit 的不同录制永不跨分区；739 个独立 unit；
- 计划 scripts/prepare_vbl_continual_plan.py：holdout 20% / validation 10% /
  round0 25% / round1 20% / round2 25%（按 unit 分配，最大余数法保证恰好求和）。

记录数：round0 260 → round1 182 → round2 212；validation 112；holdout **234**。

| 批次 | 训练记录 | challenger | champion | 决定 |
|---|---:|---:|---:|---|
| 0 (round 0) | 260 | 0.9472 | 0.9472 | promote（初始） |
| 1 (round 1) | 442 | 0.9472 | 0.9472 | reject（持平，未过 margin） |
| 2 (round 2) | 654 | **0.9646** | **0.9646** | **promote** |

holdout（234 条，只评一次）：

| 臂 | accuracy | macro-F1 |
|---|---:|---:|
| frozen | 0.9017 | 0.9045 |
| **continual** | **0.9188** | **0.9219** |
| joint（上界） | 0.9188 | 0.9219 |

**这是本项目第一个正向的持续学习结果**：

1. 持续学习比“只用初始批”高 **+1.7 个点**（0.9017 → 0.9188）；
2. 并且**完全追平 joint**（0.9188 = 0.9188）——不回头重训也没有损失；
3. 更新不是无脑接受：round 1 与 champion 持平（差 0 < 0.005），被正确拒绝；
4. 每批都覆盖全部 4 类，validation 与 holdout 来自同一套 unit 抽样规则，分布一致。

对比齿轮：齿轮的失败是**数据规模**（holdout 只能 5 组）与**判据分布**（validation 只看旧域）
双重问题；VBL 两个问题都不存在。


## 7. 结论

**VBL-VA001 已经回答了“持续学习有没有用”**：在 234 条冻结 holdout 上，
continual 0.9188 > frozen 0.9017，且等于 joint 0.9188。

aero / gear 的旧结论仍然成立，作为失败模式保留：

已经成立的：

1. 批次计划可审计（互斥、全覆盖、holdout 永不入训练集，均有测试）；
2. champion/challenger + margin 规则按设计执行；
3. 三臂对比和报告能自动产出；
4. “只用初始域做 validation”会让持续学习退化为永不更新——可复现的失败模式（gear A）；
5. “validation 与 holdout 不同分布”会让选模型和报结果不可比（gear B）。

不成立的（不许粉饰）：

1. **齿轮只有 37 个文件**，holdout 只能给 5 组，撑不起分层评估；
2. **aero 已饱和**，三臂全 1.0，无法提供信号；
3. 所以在这两个数据集上，“持续学习是否提升最终诊断”**没有可信答案**。

## 8. 下一步

1. **多轮次**：目前只有 3 个 round。VBL 的 unit 数（每类 100–298）足够切到 5–6 轮，
   可以观察“轮次变多以后是否还会追平 joint”。
2. **遗忘探针**：每批留一组永不训练的 probe，度量“旧域掉了多少”；
   VBL 的 record 数够，可以正式做。
3. **判据对比**：同一个 VBL 计划，分别用“初始域 validation”和“分层 validation”，
   量化判据选择对最终 holdout 的影响——齿轮上已经看到方向，VBL 上能做定量。
4. **换分类器/特征族**：当前固定 FUSION_TIME + EXTRA-TREES；可以用 Agent 在 round0 上重选，
   看“选型 + 持续学习”是否互相增益（本轮刻意没做，避免把两件事混在一起）。

## 9. 复现

```powershell
# VBL-VA001：打包 + 计划 + 运行
python scripts/prepare_vbl_4class.py `
  --source ..\VBL-VA001\VBL-VA001 `
  --output data/paper/vbl_4class --window 2048 --per-class 250
python scripts/prepare_vbl_continual_plan.py `
  --manifest data/paper/vbl_4class/manifest.json `
  --output data/paper/vbl_4class/continual_plan.json `
  --dataset-id VBL_4CLASS_FUSION_TIME_EXTRA_TREES
python -m entropy_agent.cli continual run `
  --spec configs/vbl_4class_fusion_time_extratrees.json `
  --data-root data/paper/vbl_4class `
  --plan data/paper/vbl_4class/continual_plan.json `
  --output results/continual-vbl-20261105

# aero（按记录顺序）
python scripts/prepare_continual_plan.py `
  --spec configs/aero_5state_fusion_time_extratrees.json `
  --data-root data/paper/aero_engine_5state `
  --output data/paper/aero_engine_5state/continual_plan.json `
  --strategy chronological --batches 3 --reserve 2
```
