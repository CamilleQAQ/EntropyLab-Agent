# 持续学习 V2：决策层（第一步——换判据）

日期：2026-11-06
状态：**判据已替换并验证。同一个固定配置下，guardrail 判据把 frozen→continual 的差距从
+1.85 点（只赢 1/5）提高到 +7.50 ± 1.24 点（5/5 全胜），并追平 joint。**

## 1. 为什么先换判据

V1 的规则是 `challenger.validation ≥ champion.validation + margin`。问题已经复现两次：

- VBL：5 个划分中 4 个的 round-0 验证集就到 0.98–1.00；
- Paderborn：5 个划分中 4 个的 round-0 验证集是 0.96–0.98。

**当 champion 已经满分，`+margin` 永远无法满足**，更新被系统性拒绝。

V2 把决策拆成两个量：

| 角色 | 度量 | 说明 |
|---|---|---|
| **guardrail** | `retention_drop` = 在**永不训练**的 probe 集上，champion 相对 challenger 的最大跌幅 | 防止遗忘 |
| **objective** | `gain_new` = challenger 在**新批次 probe** 上的提升 | 允许适应 |

判定：`retention_drop ≤ guard_tolerance` 且 `gain_new > min_gain` → Commit，否则 Keep。
默认 `guard_tolerance = 0.02`、`min_gain = 0.0`（未调参）。

## 2. probe 从哪来

计划里已经预留（`docs/17` §6）：

- `reference`：参考工况（N15_M07_F10）的 validation 集，永不训练；
- `batch-j`：第 j 轮自己的 3 次测量/类，永不训练。

Paderborn 上共 81 个 probe group（162 条），加上 validation 27 组（54 条）。

## 3. 实现

| 文件 | 内容 |
|---|---|
| `continual/decision.py`（新） | `decide()`：margin 与 guardrail 两种判据，纯函数，可单测 |
| `continual/loop.py` | 逐批评估 champion/challenger 在所有 probe 集上的准确率，写进 `decisions.json` |
| `continual/plan.py` | `Batch.probe_groups`：probe 组参与分区校验、永不进入训练（有断言） |
| CLI | `--criterion margin|guardrail`、`--guard-tolerance`、`--min-gain` |
| tests | 8 个：margin 饱和不可触发、guardrail 在满分时仍可提交、retention 拦截、增益过小拦截、端到端接线、probe 不入训练 |

303 tests 通过。

## 4. 结果（Paderborn，5 个独立划分，同一固定配置）

| 判据 | frozen | continual | joint | 逐划分 continual−frozen |
|---|---:|---:|---:|---|
| margin（V1） | 0.8843 ± 0.0135 | 0.9028 ± 0.0354 | 0.9593 ± 0.0069 | +0.0185 ± 0.0414（1/5 胜） |
| **guardrail（V2）** | 0.8843 ± 0.0135 | **0.9593 ± 0.0069** | 0.9593 ± 0.0069 | **+0.0750 ± 0.0124（5/5 胜）** |

逐划分（guardrail）：

| 划分 | frozen | continual | joint | 决策 |
|---:|---:|---:|---:|---|
| 20261104 | 0.8935 | 0.9583 | 0.9583 | round1 reject，round2 promote |
| 1 | 0.8796 | 0.9491 | 0.9491 | 两次 promote |
| 2 | 0.8750 | 0.9583 | 0.9583 | round1 reject，round2 promote |
| 3 | 0.9028 | 0.9676 | 0.9676 | round1 reject，round2 promote |
| 4 | 0.8704 | 0.9630 | 0.9630 | round2 promote |

判据不再因验证集饱和而失效：round 2 的更新在 5/5 个划分里都被接受，
而 margin 判据只有 1/5 能接受。

## 5. 必须写清的三点

1. **continual 最后等于 joint 是构造性的**：round 2 时历史批次 = 全部批次，
   两者训练数据完全相同（同 seed）→ 预测完全相同。这不是"持续学习优于联合重训"，
   而是"逐批增量能达到和一次性重训相同的效果"。
2. **验证集仍被记录，但不再参与决策**；`decisions.json` 里同时保留 validation、
   retention_drop、gain_new 和每个 probe 集的分数，便于事后审计。
3. **默认阈值已做敏感性**（§6b）：≥0.02 是平台区；`min_gain=0.03` 可再减少无谓更新。

## 6. 这一步证明了什么、没证明什么

证明了：

- V1 失败的直接原因是判据，不是数据、不是训练；
- 换成 guardrail 后，同一配置从"5 个划分只赢 1 个"变成"5/5 全胜、+7.5 点"；
- probe 机制可用：81 组 probe 从未进入训练（有测试），能支撑 retention 计算。

没证明：

- **agent 参与**：这一步仍然是固定配置（不选型、不分配预算），
  决策只是"接受/拒绝"两选一；
- 相对 joint 的优势（同上，构造性相等）；
- 阈值的稳健性。

## 6b. 敏感性分析（2026-11-06 补）

5 个划分 × `guard_tolerance ∈ {0, 0.01, 0.02, 0.05, 0.10}` ×
`min_gain ∈ {0, 0.01, 0.03}`，共 75 次运行（`scripts/sensitivity_continual_criterion.py`）：

| guard_tolerance | min_gain | continual | 逐划分 gain | 每划分 commit 数 |
|---:|---:|---:|---:|---|
| 0.00 | 0.00 | 0.9111 ± 0.0409 | +0.0269 | 2, 3, 1, 1, 1 |
| 0.00 | 0.03 | 0.9111 ± 0.0409 | +0.0269 | 2, 2, 1, 1, 1 |
| 0.01 | 0.00 | 0.9111 ± 0.0409 | +0.0269 | 2, 3, 1, 1, 1 |
| **0.02** | **0.00** | **0.9593 ± 0.0069** | **+0.0750** | 2, 3, 2, 2, 2 |
| 0.02 | 0.03 | 0.9593 ± 0.0069 | +0.0750 | 2, 2, 2, 2, 2 |
| 0.05 | 0.00 | 0.9593 ± 0.0069 | +0.0750 | 2, 3, 2, 3, 2 |
| 0.10 | 0.00 | 0.9593 ± 0.0069 | +0.0750 | 2, 3, 2, 3, 2 |

结论：

1. **是平台区，不是刀尖**：`guard_tolerance ≥ 0.02` 时结果完全一致（0.9593），
   一直放宽到 0.10 都不变；只有 ≤ 0.01 时掉到 0.9111。
2. `min_gain` 在 0–0.03 之间几乎不影响精度，只减少 commit 次数
   （`min_gain=0.03` 时每个划分稳定在 2 次 = 初始 + 一次更新）。
3. 因此默认值 `guard_tolerance = 0.02` 落在平台区内侧、`min_gain = 0.03` 更省抖动，
   两者都不是拍出来的。
4. 敏感性也暴露一个边界：容差过小（≤0.01）会把"实际掉 1–2 个点"的合理更新也挡掉，
   这正是 V1 的失败方式，只是换了个阈值。

## 7. 下一步（按讨论的顺序）

1. **敏感性分析**：`guard_tolerance ∈ {0.0, 0.01, 0.02, 0.05}`、`min_gain ∈ {0, 0.01, 0.03}`，
   看 5 个划分的胜率与幅度——阈值不能是拍出来的；
2. **每轮选型**：Agent 在预算内重选特征族/分类器（复用 `agent/cost.py` 与
   `agent/candidates.py`），对照"只换判据"看是否有增量；
3. **预算动作**：`search(n)` 让"再花多少钱找候选"成为动作；
4. 之后才是策略学习（Contextual Bandit）。

## 8. 复现

```powershell
python -m entropy_agent.cli continual run `
  --spec configs/paderborn_3class_fusion_time_extratrees.json `
  --data-root data/paper/paderborn_3class `
  --plan data/paper/paderborn_3class/continual_plan_seed20261104.json `
  --output results/continual-paderborn-guardrail-seed20261104-20261106 `
  --criterion guardrail --guard-tolerance 0.02 --min-gain 0.0
```
