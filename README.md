# EntropyLab Agent

面向工业故障诊断的自主实验智能体。使用 LangGraph 和 DeepSeek 编排假设生成、工具实验、证据审查和重规划，并通过 SQLite 保存实验配置、指标和决策依据。

给定一个数据集配置和研究目标，Agent 可以先建立基线，再根据误差分布决定是否更换特征或分类器，最后提交带实验依据的报告。

**技术栈** · Python · LangGraph · DeepSeek · Pydantic · scikit-learn · NumPy / SciPy · SQLite

## 项目亮点

- **自主实验编排。** Agent 根据数据画像、预算和实验历史选择下一步。规则审查实验结果与假设引用的数字，关键动作可由人工确认。
- **特征工具自扩展。** 将 8 种熵特征方法和 5 类融合特征与 6 种分类器组合成 78 个候选。LLM 生成的特征代码经静态检查、受限子进程执行、契约测试和配对评估后，可以注册给后续实验复用。
- **预算感知选型。** 以预估特征提取成本调度候选，复用逐记录特征缓存，并在相同划分下比较固定顺序、随机搜索和穷举。
- **持续更新。** 用独立的留出探测集衡量新工况收益和历史工况性能变化，据此发布、暂缓或拒绝候选模型。

## 代表性结果

| 实验 | 结果 | 评估范围 |
| --- | --- | --- |
| 三轴轴承预算选型 | 在 60 秒预算下，以 **37.6 秒预估提取成本**达到穷举的验证 Macro-F1 **0.9500**，比同预算下的固定顺序高 **11.7 个百分点** | 7 类，5 个划分。成本采用统一的特征提取估算口径 |
| CWRU 特征工具验收 | 加入 Agent 生成的包络谱特征后，平均验证 Macro-F1 从 **0.9571** 提升到 **0.9715** | 4 类，5 个录制级划分，同一划分和分类器下配对比较 |
| Paderborn 持续更新 | 相比冻结初始模型，留出测试准确率平均提升 **3.51 个百分点**，5 个划分中 3 个提升、2 个持平 | 按工况顺序引入数据，每轮决策仅使用已到达的批次 |

实验协议与运行入口见[代表性实验](docs/experiments.md)。

## 架构

```text
DatasetSpec → 分组切分 → 特征提取 → 训练与评估 → 版本化模型产物
数据画像 → Agent 规划 → 人工确认（可选）→ 工具执行 → 规则审查 → 重规划或报告
```

| 模块 | 职责 |
| --- | --- |
| `lab/` | 编排实验、记录证据、验收生成的特征工具，核验 Agent 提交的配对实验结论 |
| `agent/` · `training/` | 管理特征缓存与候选预算，执行分组切分、训练和评估 |
| `continual/` · `diagnosis/` | 按工况逐批评估候选模型，通过显式注册的模型及其输入契约运行诊断 |

数据集配置定义标签、采样参数和分组规则。同一训练流水线可读取不同配置，诊断时由调用方明确选择适用的模型。

## 快速开始

### 本地验证

安装依赖后，可以直接检查仓库自带的 CSV 样例。测试使用模拟规划器和测试数据，支持离线验证实验编排、人工确认和证据评分。

```powershell
python -m pip install -e ".[ml,test]" langgraph
python -m entropy_agent.cli inspect --signal tests/fixtures/three_channels.csv
python -m pytest tests/lab -q
```

### 运行一次研究实验

准备 [CWRU 数据](docs/research-lab.md#cwru-案例)，并在环境中设置 `DEEPSEEK_API_KEY` 后，可运行以下示例。`--interactive` 会在提交结论或调用自生成特征前等待人工确认。

```powershell
python -m entropy_agent.cli lab run `
  --spec configs/cwru_4class_fusion_time_extratrees.json `
  --data-root data/paper/cwru_4class `
  --output results/cwru-lab-demo `
  --goal "建立四分类基线并分析各类别召回率" `
  --max-rounds 5 --budget 300 --interactive
```

### 实验输出

| 文件 | 内容 |
| --- | --- |
| `report.md` · `ledger.sqlite` | 研究报告，以及用于回查的假设、工具结果和审查记录 |
| `cost_model.json` | 预算选型中各策略的预估成本与候选选择 |
| `decisions.json` · `metrics.json` | 持续更新中各批次的决策、判定依据及留出集结果 |

## 技术文档

- [自主实验、特征工具与任务评测](docs/research-lab.md)
- [持续更新与模型发布判据](docs/continual-learning.md)
- [第三方来源与许可声明](THIRD_PARTY_NOTICES.md)
