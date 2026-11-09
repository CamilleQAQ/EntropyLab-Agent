# 人工确认与任务级评测

## 人工确认

交互模式下，LangGraph 在规划和执行之间设置审批节点。调用已注册的自生成特征或提交结论前，节点暂停并等待确认。批准后执行该动作，拒绝后终止当前运行并生成报告。

```powershell
python -m entropy_agent.cli lab run `
  --spec configs/cwru_4class_fusion_time_extratrees.json `
  --data-root data/paper/cwru_4class `
  --output results/lab-cwru-interactive `
  --goal "在验证集上建立基线" --interactive
```

## 任务级评测

[任务配置](../configs/lab_tasks.json)定义数据集、目标、轮数和通过条件。通用检查涵盖实际执行的实验、划分数、工具错误、重复调用、假设证据和报告。

CWRU 配对任务将完整训练与移除一个正常训练录制进行对照。两组复用同一个验证集。评分器从 SQLite 账本中的样本数和正确预测数重算逐划分召回率，再核验 Agent 提交的证据编号、平均召回率和变化方向。结论适用范围限定在观察到的划分。

```powershell
python scripts/run_lab_eval.py --output results/lab-eval-demo --budget 300
```

使用新的输出目录运行配对任务。[人工中断测试](../tests/lab/test_hitl.py)与[评分测试](../tests/lab/test_eval.py)覆盖暂停、拒绝、证据引用和结论核验。
