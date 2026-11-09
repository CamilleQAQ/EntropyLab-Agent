# 自主实验闭环与特征工具工厂

Research Lab 通过 LangGraph 编排数据画像、假设生成、实验、审查和报告。DeepSeek 根据当前预算、可用工具和历史证据提出下一步，Pydantic 校验动作结构。SQLite 记录假设、工具输入输出和审查意见。

## 实验编排

- **规划与执行。** 读取数据画像和历史结果，选择训练评估、误差分析或特征验证工具。重复调用和工具异常会以结构化信息反馈给规划器。
- **规则审查。** 按划分数量、结果波动和证据检查实验与假设。报告从已记录的工具结果提取指标。
- **人工确认。** 交互模式可在调用自生成特征或提交结论前暂停。[人工确认与任务级评测](23_hitl_and_eval.md)

## 特征工具工厂

生成候选代码 → AST 静态检查 → 受限子进程执行 → 检查形状、有限值、确定性和非常量 → 同划分配对评估 → 通过后注册。

执行错误可以反馈给 LLM 修订。评估时保持划分、基础特征和分类器一致，只给候选组增加生成的特征。注册后的工具可通过 `run_custom_experiment` 在后续实验复用。执行约束由静态检查和子进程超时实现。

### CWRU 案例

在 CWRU 12 kHz 驱动端四分类任务上，以 64 个原始录制为组，每个录制提取 6 个窗口。将生成的包络谱特征叠加到 FUSION_TIME，保持 ExtraTrees 和 5 个录制级划分一致。平均验证 Macro-F1 从 **0.9571** 升至 **0.9715**，配对均值增加 **0.0144**，达到该次验收阈值。注册后重新调用该工具复现了对应的验证分数。

数据准备见 [CWRU 适配脚本](../scripts/prepare_cwru_4class.py) 和[实验配置](../configs/cwru_4class_fusion_time_extratrees.json)。

```powershell
python -m entropy_agent.cli lab factory `
  --spec configs/cwru_4class_fusion_time_extratrees.json `
  --data-root data/paper/cwru_4class `
  --output results/lab-cwru-demo `
  --goal "生成用于四分类的包络谱特征" --seeds 5
```

命令需要本地 CWRU 数据和 `DEEPSEEK_API_KEY`。每次运行生成的候选由模型当次输出决定。
