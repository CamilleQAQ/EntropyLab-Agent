# 人工中断与任务级评测

日期：2026-11-07
状态：两项都完成。人工中断在真实运行里会暂停等确认；任务级评测集在 3 个任务上
**全部 8/8 通过**（平均 1.00）。

## 1. Human-in-the-loop（LangGraph interrupt）

### 怎么工作

LangGraph 图里插入一个 `guard` 节点，位于 `plan` 与 `execute` 之间：

```text
plan → guard → execute → review → plan
              ↑
        关键动作暂停，等人确认（interrupt）
```

触发条件（`RISKY_ACTIONS`）：

- `run_custom_experiment`：调用由 LLM 自己生成并注册的特征；
- `finish`：准备下结论、写报告之前。

### 暂停 / 恢复

```python
graph = build_graph(ctx, llm=chat, checkpointer=MemorySaver())
result = graph.invoke(state, config)          # 在 guard 处返回 __interrupt__
result = graph.invoke(Command(resume=True), config)   # 人工批准后继续
```

- 批准 → 继续执行；
- 拒绝 → 状态置为 `rejected`，直接进入 `finalize`，不再执行该动作。

CLI：

```powershell
python -m entropy_agent.cli lab run ... --interactive
```

非交互模式不插 checkpointer，`guard` 直接放行，行为与之前一致。

### 测试

`tests/lab/test_hitl.py`：暂停发生在 `finish` 之前、拒绝会终止运行、非交互模式不暂停。

## 2. 任务级评测集

### 任务定义

`configs/lab_tasks.json`，每个任务 = 数据集 + 目标 + 期望：

```json
{
  "id": "cwru-class-imbalance",
  "dataset": "cwru",
  "goal": "找出 NORMAL 类样本少（4 个录制）对指标的影响，并给出改进方向",
  "max_rounds": 3,
  "expect": {"min_experiments": 1, "min_seeds": 3, "min_hypotheses": 1}
}
```

### 评分（8 项检查）

| 检查 | 含义 |
|---|---|
| `finished` | 正常结束（不是崩了/被中断） |
| `has_experiment` | 至少跑了一个实验 |
| `enough_seeds` | 每个实验的划分数量 ≥ 阈值 |
| `no_tool_failures` | 工具调用没有失败 |
| `no_repeated_calls` | 没有被拦截的重复调用 |
| `report_written` | 报告存在且是研究日志格式 |
| `has_hypothesis` | 至少提出一个假设 |
| `hypotheses_grounded` | 假设引用的数字来自实验历史 |

评分关注**过程**，不是「找到多好的模型」——这正是 agent 评测该测的东西。

### 实测（3 个任务，真实 LLM）

| 任务 | 数据集 | 得分 |
|---|---|---:|
| paderborn-baseline | paderborn | 8/8 |
| cwru-class-imbalance | cwru | 8/8 |
| gear-hard-case | gear | 8/8 |
| **平均** | | **1.00** |

产物：`results/lab-eval-20261107/{results.csv,summary.md,<task>/ledger.sqlite}`。

### 一个必须记下的 bug

第一次跑出来是 7/8，全部卡在 `enough_seeds`。原因是**评分函数读错了层级**：
账本把工具结果包在 `payload` 字段里，而评分直接读顶层。修正后重算（复用已保存的
账本，不重跑 LLM）得到 8/8。回归测试已加。

这说明任务级评测本身也需要测试——否则会给出假的失败。

## 3. 复现

```powershell
# 带人工中断的一次研究
python -m entropy_agent.cli lab run `
  --spec configs/cwru_4class_fusion_time_extratrees.json `
  --data-root data/paper/cwru_4class `
  --output results/lab-cwru-interactive `
  --goal "在验证集上取得最高 macro-F1" --interactive

# 任务级评测
python scripts/run_lab_eval.py --output results/lab-eval-20261107 --budget 300
```
