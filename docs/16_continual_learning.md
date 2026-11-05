# 持续学习 v1（`continual/`）

日期：2026-11-05
状态：**机制已实现并通过测试；两个数据集都回答不了“持续学习有没有用”这个问题，原因已定位。**

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

## 6. 结论：机制可用，但现有数据回答不了问题

已经成立的：

1. 批次计划可审计（互斥、全覆盖、holdout 永不入训练集，均有测试）；
2. champion/challenger + margin 规则按设计执行；
3. 三臂对比和报告能自动产出；
4. “只用初始域做 validation”会让持续学习退化为永不更新——这是可复现的失败模式。

没成立的（不许粉饰）：

1. **齿轮只有 37 个文件**，holdout 只能给 5 组；它无法同时覆盖三个载荷层，
   所以 holdout 与 validation 分布不一致，三臂在 holdout 上区分不开；
2. **aero 已饱和**，无法提供信号；
3. 因此“持续学习是否提升最终诊断”在本仓库目前**没有可信答案**。

## 7. 下一步（按优先级）

1. **修判据**：validation 与 holdout 必须来自同一分层（同类别、同载荷覆盖），
   否则选模型和报结果用的不是一把尺子。齿轮做不到（组数不够），需要换数据。
2. **换数据**：需要“每批 ≥ 数十条、跨批有真实漂移、且留得住分层 holdout”的数据集。
   候选：Paderborn（多工况、多轴承，但当前 pipeline 在该数据上只有 24–40%，见 `docs/` 旧记录），
   或 ESWA 论文的 CWRU/Politecnico/Wind。
3. **probe 机制**：每批保留一组永不训练的 probe，作为该域的“遗忘探针”；
   本次因齿轮组数不足没做。

## 8. 复现

```powershell
# 计划
python scripts/prepare_continual_plan.py `
  --spec configs/aero_5state_fusion_time_extratrees.json `
  --data-root data/paper/aero_engine_5state `
  --output data/paper/aero_engine_5state/continual_plan.json `
  --strategy chronological --batches 3 --reserve 2

# 运行
python -m entropy_agent.cli continual run `
  --spec configs/aero_5state_fusion_time_extratrees.json `
  --data-root data/paper/aero_engine_5state `
  --plan data/paper/aero_engine_5state/continual_plan.json `
  --output results/continual-aero-20261105
```
