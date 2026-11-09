# 持续更新与模型发布判据

随着新工况批次到达，系统用已到达的数据训练候选模型，并将候选与当前模型在留出探测集上进行对照。

| 信号 | 计算方式 |
| --- | --- |
| 新工况收益 | 候选模型在当前批次探测集上的准确率增量 |
| 历史工况退化 | 候选模型在参考集及已到达的历史探测集上的最大准确率下降 |

当新工况收益超过阈值且历史退化在容差内，发布候选模型。有新工况收益但历史退化处于容差与两倍容差之间时，记录为暂缓。其余情况拒绝。探测集与训练集隔离，每轮仅能访问参考集和当时已到达的探测数据。

## Paderborn 评估

使用 FUSION_TIME + ExtraTrees 固定配置，在 5 个测量级分组划分上逐批引入工况。相较仅使用初始批次训练的冻结模型，持续更新后留出测试准确率平均提高 **3.51 个百分点**，3 个划分提升，2 个持平。结果对应这组数据和时序协议。

## 运行入口

使用 [Paderborn 批次计划脚本](../scripts/prepare_paderborn_continual_plan.py) 生成测量级分组计划。上述评估使用 20261104 至 20261108 五个种子。每个种子用对应计划单独运行，例如：

```powershell
python -m entropy_agent.cli continual run `
  --spec configs/paderborn_3class_fusion_time_extratrees.json `
  --data-root data/paper/paderborn_3class `
  --plan data/paper/paderborn_3class/continual_plan_seed20261104.json `
  --output results/continual-paderborn-seed20261104 `
  --criterion guardrail --guard-tolerance 0.02 --min-gain 0.0
```

`decisions.json` 记录每轮决策和依据，`metrics.json` 汇总留出测试指标。
