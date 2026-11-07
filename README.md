# EntropyLab Agent · 轴承故障诊断自主研究智能体

**一句话**：把「选特征、选分类器、调预算、持续更新」交给一个会提假设、会做实验、
会被确定性规则驳回的 Agent；模型侧在 5 个公开数据集上完成验证。

- 语言/框架：Python 3.13 · LangGraph · DeepSeek Tool Calling · Pydantic · scikit-learn · PyTorch · SQLite
- 测试：**347 passed**
- 数据集：aero / triaxial / VBL / Paderborn / CWRU

---

## 1. 它能做什么

| 能力 | 说明 | 证据 |
|---|---|---|
| **自主研究闭环** | LangGraph 状态机：数据画像 → 提假设 → 选工具实验 → 审查 → 重规划 → 出报告 | [`docs/21`](docs/21_research_lab.md) |
| **假设接地** | 假设里的数字必须来自实验历史，否则驳回重写 | [`docs/24`](docs/24_ablation.md) |
| **工具工厂** | LLM 自己写特征提取器，经 AST 白名单、沙箱、契约测试、配对评估后注册 | [`docs/21`](docs/21_research_lab.md) §5c |
| **持续学习** | probe 判据解决「验证集饱和导致永不更新」，支持每轮重选与预算动作 | [`docs/18`](docs/18_v2_decision_layer.md) |
| **人工中断** | 关键动作（调用自生成特征、下结论）前暂停等待批准 | [`docs/23`](docs/23_hitl_and_eval.md) |
| **任务级评测** | 8 项过程检查，衡量「研究是否规范」而非单一精度 | [`docs/23`](docs/23_hitl_and_eval.md) |

## 2. 结果

| 数据集 | 类别数 | 指标 | 说明 |
|---|---:|---:|---|
| Aero engine | 5 | **99.3% ± 0.8** | 论文复现 + Agent 选型 |
| Triaxial | 7 | **93.6% ± 4.4** | 最难的公开分类任务 |
| VBL | 4 | **98.3%** | 持续学习交叉验证数据集 |
| Paderborn | 3 | **98.8%**（嵌套 CV 测试，3×3） | 持续学习主实验 |
| CWRU | 4 | **95.7% → 97.2%**（+自生成特征） | 工具工厂验收 |

**Agent 侧关键数字**

| 实验 | 结果 |
|---|---|
| 选型 vs 穷举（60 秒预算） | 37.6 秒匹配穷举；固定顺序 59.9 秒且低 12 点 |
| 持续学习（probe 判据） | **+7.50 点，5/5 划分胜出** |
| 每轮重选方法 | **+2.87 点** |
| 预算动作 | 成本减半，精度 −1.7 点（显式权衡） |
| 工具工厂 | 7 次生成 6 次被拒，1 次 **+1.44 点**注册成功 |
| 任务级评测 | 3 个任务 **8/8** |
| 消融 | 审查员在高方差数据上 **3/3** 触发，关闭后 0/3 |

## 3. 工作流

```text
① 数据层   适配器 → npy + manifest → DatasetSpec → 按组隔离切分（60/20/20 × 5 划分）
② 特征层   8 种熵 + 5 个融合族 → 候选池（13 单元 × 6 分类器）
③ 模型层   逐候选训练/验证/测试 → 选型 + 秒级预算 → 版本化产物
④ Agent 层 plan（LLM）→ guard（人工批准）→ execute（工具）→ review（规则）→ 重规划
⑤ 工厂层   LLM 写代码 → AST 白名单 → 子进程沙箱 → 契约测试 → 配对评估 → 注册
⑥ 评测层   347 个单元/集成测试 + 任务级 8 项检查
```

## 4. 快速开始

```powershell
pip install -e .
pytest -q                                   # 347 passed

# 看已注册的模型
python -m entropy_agent.cli models list

# 跑一次自主研究（不需要联网）
python -m entropy_agent.cli lab run `
  --spec configs/paderborn_3class_fusion_time_extratrees.json `
  --data-root data/paper/paderborn_3class `
  --output results/lab-demo --goal "在验证集上建立可靠基线" --max-rounds 4

# 带人工中断（需要 DEEPSEEK_API_KEY）
python -m entropy_agent.cli lab run ... --interactive
```

## 5. 限制（必须知道）

1. **5 个划分不是独立样本**：同一批测量重新分配，只能做逐划分配对比较，不能合并做
   bootstrap / McNemar。
2. **Paderborn 的历史「冻结测试 88.4%」不再可复现**：它依赖一份已删除的固定 manifest
   （另一批轴承）。**嵌套交叉验证（3 外层 × 3 内层）显示选型没有过拟合验证集**：
   朴素协议验证-测试差 +0.19 点，嵌套协议 +0.14 点，测试分 0.988 vs 0.987。
   因此那 10 点差距来自协议差异，不是选型偏差。
3. **策略学习只在允许试 3 个候选时优于固定顺序**（regret 0.0104 vs 0.0319），
   只允许试 1 个时仍落后（0.060 vs 0.034）。
4. **任务级评测只测过程合规**，不判断结论对错。
5. 齿轮数据集（5 类，37 组）因样本过少、划分方差达 ±17.6 点，已从对外结果中排除，
   仅保留在 `results/_archive/` 与内部分析文档中。

## 6. 目录

```text
src/entropy_agent/
  analysis/       熵特征后端、窗口、质量检查
  classifiers/    6 个分类器（含 BT-LSTSVM）
  io/             DatasetSpec、加载器、窗口
  training/       分组切分 + 通用训练流水线
  diagnosis/      模型产物、注册表、诊断运行器
  agent/          成本模型、SQLite 记忆、选型策略
  lab/            LangGraph 研究闭环、工具、工厂、审查、评测
configs/          数据集规格与实验配置
docs/             23 篇设计与实验文档
results/          可复现的实验产物（_archive/ 为历史版本）
tests/            347 个测试
```

## 7. 诚实说明

本项目对「负结果」的保留方式是：**失败实验移入 `results/_archive/`，并在
[`docs/25_experiments.md`](docs/25_experiments.md) 中记录为什么失败**，而不是删除。
论文级别的结论都标注了样本量与限制。
