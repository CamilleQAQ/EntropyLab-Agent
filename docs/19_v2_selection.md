# V2 第二步：每轮重选特征族与分类器（Agent 真正参与）

日期：2026-11-06
状态：**Agent 已参与决策。每轮在秒预算内重选候选池，冠军方法可以变；
在 Paderborn 5 个独立划分上，continual 从 0.9593（固定配置）提升到 0.9880。**

## 1. 这一步和上一步的区别

| | V2.1（换判据） | V2.2（本步） |
|---|---|---|
| 训练什么 | 固定配置（FUSION_TIME + ExtraTrees） | 候选池里的任意组合 |
| 决策 | 接受 / 拒绝 | 接受 / 拒绝 + **选哪个** |
| 特征空间 | 固定 | 每个候选自己一套 |
| Agent 角色 | 无（规则判据） | **搜索 + 选择** |

## 2. 动作与预算

每个候选 = `(特征方法, 分类器)`，来自 `agent/candidates.py`（13 方法 × 6 分类器 = 78）。

```text
每轮 t：
  按"预估提取秒数"升序排列候选（复用 agent/cost.py 的先验价格）
  依次付费评估：在 历史∪新批 上训练 → 在 probe 集上打分
  开销：一个特征方法只付一次（同方法的第 2~6 个分类器免费，特征已缓存）
  候选必须过 guardrail（retention_drop ≤ 0.02 且 gain_new > 0）
  在通过者里选 gain_new 最大者提交；没有通过者则 Keep
```

计费口径与 v1.4 一致：**只对特征提取计费**（训练时间记录但不计费）。
每轮预算 12 秒；Paderborn 上 12 秒够付 MPE / MDE / FUSION_TIME / FUSION_TIME_SPEC
等方法（TSMEFuDE 一次就要 562 秒，必然被跳过）。

## 3. 实现

| 文件 | 内容 |
|---|---|
| `continual/selection.py`（新） | 候选排序、逐候选训练、probe 打分、guardrail 过滤、选最优、预算账本 |
| `continual/loop.py` | `select=True` 分支；champion 现在携带自己的 `spec`（特征空间可能变化）；holdout 按各臂自己的特征空间评估 |
| `cli` | `--select`、`--selection-budget`；默认池 78 个候选 |
| 测试 | 5 个新测试：候选 spec 派生、搜索日志、预算约束、无可提交时 Keep、每轮可换配置 |

308 tests 通过。

## 4. 结果（Paderborn，5 个独立划分）

| 策略 | frozen | continual | 逐划分 gain | 胜率 |
|---|---:|---:|---:|---:|
| V1：固定配置 + margin | 0.8843 ± 0.0135 | 0.9028 ± 0.0354 | +0.0185 | 1/5 |
| V2.1：固定配置 + guardrail | 0.8843 ± 0.0135 | 0.9593 ± 0.0069 | +0.0750 | 5/5 |
| **V2.2：选型 + guardrail** | 0.9019 ± 0.0192 | **0.9880 ± 0.0195** | +0.0861 | **5/5** |

**选型相对固定配置的净增益：0.9593 → 0.9880（+2.87 点）**，5/5 个划分都更高或持平。

注意冻结臂也变了（0.9019 vs 0.8843）：V2.2 的 frozen = round 0 选出来的模型，
所以"相对 frozen 的增益"两行不能直接比；**能直接比的是 continual 那一列**。

逐划分：

| 划分 | frozen | continual | joint | 最终配置 |
|---:|---:|---:|---:|---|
| 20261104 | 0.8935 | **1.0000** | 1.0000 | FUSION_TIME_SPEC + ExtraTrees |
| 1 | 0.9259 | 0.9954 | 0.9954 | FUSION_ALL + RandomForest |
| 2 | 0.8750 | 0.9537 | 0.9537 | FUSION_TIME_SPEC + SVM-RBF |
| 3 | 0.9120 | 0.9907 | 0.9907 | FUSION_TIME_SPEC + ExtraTrees |
| 4 | 0.9028 | **1.0000** | 1.0000 | FUSION_TIME_SPEC + ExtraTrees |

## 5. 决策日志（agent 真的在选）

以 seed 20261104 为例：

| 轮 | 付费候选数 | 花费 | 尝试的方法 | 通过 guardrail | 提交 |
|---:|---:|---:|---|---:|---|
| 0 | 24 | 8.2 s | MPE, MDE, FUSION_TIME, FUSION_TIME_SPEC | 24（初始轮） | ✅ |
| 1 | 36 | 10.8 s | + FUSION_TIME_ENVELOPE, FUSION_TIME_ENVELOPE_ENTROPY | 3 | ✅ |
| 2 | 42 | 8.6 s | + FUSION_ALL | 5 | ✅ |

关键点：

- **冠军方法变了**：锚点配置是 `FUSION_TIME`，agent 在 4/5 个划分里选了
  `FUSION_TIME_SPEC`，1 个划分选了 `FUSION_ALL`；
- 候选数逐轮增加，是因为已经提取过的特征方法在缓存里**不再计费**，
  同样的预算能覆盖更多方法；
- TSMEFuDE / MFE / RCMFuDE 从未被付费（预估 151–562 秒，远超 12 秒预算）。

## 6. 必须写清的局限

1. **continual 仍等于 joint**：最后一轮历史 = 全部数据，且 joint 现在用冠军的配置，
   所以相等是构造性的；这一步证明的是"选型 + 增量更新能拿到全部数据的收益"。
2. **初始轮按 validation 选，存在赢家诅咒**：24 个候选里挑验证集最高者，
   而验证集只有 54 条；holdout 1.0 说明这次没被诅咒，但长期要小心。
3. **预算只计特征提取**：训练时间未计费，所以"12 秒"不等于真实墙钟（实际每轮数十秒）。
4. **frozen 的定义变了**（round 0 也是选出来的），跨策略比较只能用 continual 列。
5. 仍然只有 3 类、9 个轴承；类别和工况的多样性有限。

## 7. 下一步

1. **预算动作** `search(n)`：把"再花多少预算找候选"变成 agent 的动作，
   而不是每轮固定 12 秒；
2. **策略学习**：把每轮的 `(state, action, reward)` 记全，等样本够（~100 条转移）
   再试 Contextual Bandit；现在样本是 5 划分 × 3 轮 = 15 条，不够；
3. **更严格的评测**：holdout 之外的"极端工况"（N09）作为额外诊断；
4. 扩大轴承数（每类 ≥5 个），降低选型的赢家诅咒。

## 8. 复现

```powershell
python -m entropy_agent.cli continual run `
  --spec configs/paderborn_3class_fusion_time_extratrees.json `
  --data-root data/paper/paderborn_3class `
  --plan data/paper/paderborn_3class/continual_plan_seed20261104.json `
  --output results/continual-paderborn-select2-seed20261104-20261106 `
  --criterion guardrail --select --selection-budget 12
```
