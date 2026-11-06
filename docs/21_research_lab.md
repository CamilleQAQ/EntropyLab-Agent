# Research Lab：LLM 驱动的自主实验闭环

日期：2026-11-06
状态：**可运行**。LangGraph 状态机 + DeepSeek 工具调用 + 假设账本 + 确定性审查员，
已在 Paderborn 上跑通一次真实研究循环（5 轮，2 个实验，1 次被审查员驳回）。

## 1. 它解决什么问题

之前的 Agent 是“在固定候选里选一个组合”，规则由人写死。这个模块让 LLM 决定
**下一步做什么实验**，并且用一套确定性规则约束它不能乱下结论。

区别：

| | 旧 Agent | Research Lab |
|---|---|---|
| 谁决定下一轮 | 我写的规则 | LLM（读数据画像 + 历史 + 预算） |
| 能否提出假设 | 不能 | 能，且假设入账本 |
| 能否否证自己 | 不能 | 能（审查员驳回） |
| 结论依据 | 指标 | 指标 + 证据 + 审查记录 |

## 2. 结构

```text
src/entropy_agent/lab/
  llm.py       DeepSeek 客户端（读 .env，JSON 模式，1 次重试）+ FakeLLM（测试用）
  tools.py     4 个工具：profile_dataset / run_experiment / analyze_errors / search_notes
  ledger.py    SQLite：假设、实验、审查意见
  reviewer.py  确定性审查：划分不足、波动过大、结论在噪声内
  graph.py     LangGraph：plan → execute → review → (plan | finalize)
```

CLI：

```powershell
python -m entropy_agent.cli lab run `
  --spec configs/paderborn_3class_fusion_time_extratrees.json `
  --data-root data/paper/paderborn_3class `
  --output results/lab-paderborn-20261106 `
  --goal "在验证集上找到最好的特征+分类器组合，并写清结论边界" `
  --max-rounds 5 --budget 600
```

## 3. 实测一次真实循环（Paderborn，DeepSeek）

| 轮 | 动作 | 结果 |
|---|---|---|
| 1 | `run_experiment` FUSION_TIME + ExtraTrees，5 划分 | macro-F1 **0.9611 ± 0.0070**（19.3 s） |
| 2 | `profile_dataset` | 1080 条、3 类各 360、540 组、64 kHz |
| 3 | `search_notes` "inner ring confusion" | 3 条本地笔记 |
| 4 | `search_notes` "best feature classifier fusion" | 3 条 |
| 5 | `run_experiment` FUSION_TIME + SVM-RBF，5 划分 | 0.9212 ± **0.0308** → 审查员驳回：`UNSTABLE` |

产物：`results/lab-paderborn-20261106/` 下有 `ledger.sqlite`（假设/实验/审查）和 `report.md`。

这一轮里最关键的是第 5 轮：**LLM 自己提出换分类器，审查员按波动阈值驳回了它**，
而不是让它把一个不稳定的结果写进结论。

## 4. 约束（防止 LLM 胡说）

1. **审查员是确定性的**，不由 LLM 控制：少于 3 个划分、波动 > 0.02、
   声称的提升小于 2 倍波动，都会被驳回；
2. **重复动作被拦截**：同样的工具 + 同样的参数不能跑第二次；
3. **没做实验时强制做实验**：规划提示里硬性要求 `run_experiment`；
4. **工具报错不崩**：错误作为结果返回给 LLM，由它决定下一步；
5. 报告由模板生成，数字只来自工具返回，避免 LLM 编造数字。

## 5. 已知不足

- 假设质量一般（LLM 倾向写“笔记可能说明……”这类空话），提示词需要收紧；
- 每轮重新提取特征，实验成本高（Paderborn 一次 5 划分约 19 秒）；
- 只有 4 个工具，尚不能自己写新工具；
- 没有跨数据集策略学习（下一步）。

## 6. 测试

`tests/lab/test_lab.py`：6 个测试，用 FakeLLM 覆盖闭环、审查规则、账本、
工具返回结构；不依赖网络和 API key。
