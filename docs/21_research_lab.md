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
| 1 | `run_experiment` FUSION_TIME + ExtraTrees，5 划分 | macro-F1 **0.9611 ± 0.0070**（18.5 s） |
| 2 | `profile_dataset` | 1080 条、540 组、3 类各 360 |
| 3 | `search_notes` 内圈混淆 / 包络特征 | 3 条 |
| 4 | `search_notes` 最佳特征组合 | 3 条 |
| 5 | `analyze_errors` | 逐类召回 HEALTHY 1.0 / OR 0.9028 / IR 0.9722；OR→IR **7 例** |
| 6 | `run_experiment` FUSION_TIME + SVM-RBF，5 划分 | 0.9212 ± **0.0308** → 审查员驳回：`UNSTABLE` |
| 7 | 假设改口 | 引用 0.9212±0.0308 与「被驳回」，准备换方向（预算到） |

产物：`results/lab-paderborn-20261106/` 下有 `ledger.sqlite`（假设/实验/审查）和 `report.md`。

两处关键行为：

1. **第 3–5 轮的假设引用真实数字**（0.9611±0.0070、OR→IR 7 例），不是空话；
2. **第 6 轮 LLM 自己换分类器，审查员按波动阈值驳回**，第 7 轮它引用这次驳回调整假设。

## 4. 约束（防止 LLM 胡说）

1. **审查员是确定性的**，不由 LLM 控制：少于 3 个划分、波动 > 0.02、
   声称的提升小于 2 倍波动、工具调用失败，都会被驳回；
2. **重复动作被拦截**：同样的工具 + 同样的参数不能跑第二次；
3. **没做实验时强制做实验**：规划提示里硬性要求 `run_experiment`；
4. **工具报错不崩**：错误作为结果返回给 LLM，由它决定下一步；
5. 报告由模板生成，数字只来自工具返回，避免 LLM 编造数字；
6. **假设接地检查**：假设里出现的数字必须能在实验历史里找到（浮点按 2% 容差），
   否则驳回并要求重写一次；仍不通过则标记 `⚠ 未通过校验` 记入账本。

## 5. 已知不足

- 假设质量已通过「接地检查 + 重写」改善，但仍会出现泛泛而谈的表述；
- 每轮重新提取特征，实验成本高（Paderborn 一次 5 划分约 19 秒）；
- 工具工厂已注册 1 个新特征（CWRU），但注册结果尚未接回实验循环；
- 没有跨数据集策略学习（下一步）。

## 5b. 工具工厂（LLM 自己写特征提取器）

### 原理

```text
LLM 提议（name / rationale / code）
   → 静态检查（AST：只允许 numpy/scipy/math；禁 open/exec/标签参数）
   → 子进程沙箱（30s 超时，独立进程，产出特征矩阵）
   → 契约测试（形状、有限值、确定性、非常量）
   → 叠加评估（候选特征 + 项目现有特征，同一分类器、同一批划分）
   → 通过才注册到 lab_tools/，否则记为失败
```

关键设计：**新特征按“叠加”评估，不按“替代”评估**——新工具的价值是补充信息，
不是替换 90 维融合特征。

### 三次真实运行（Paderborn，DeepSeek 生成）

| 提议 | 划分 | 叠加后 macro-F1 | 基线 | 差值 | 阈值 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| `envelope_spectrum_band_ratio` | 3 | 0.8797 ± 0.0372 | 0.9629 ± 0.0080 | −0.0832 | 0.0744 | 拒绝 |
| `envelope_harmonic_kurtosis_band_energy` | 3 | 0.9768 ± 0.0046 | 0.9629 ± 0.0080 | +0.0139 | 0.0161 | 拒绝（差一点） |
| 同一个候选 | 5 | 0.9685 ± 0.0039 | 0.9611 ± 0.0070 | +0.0074 | 0.0141 | 拒绝 |

第三个结果很有意思：**同一个特征在 3 个划分上看着 +1.39 点，5 个划分只剩 +0.74 点**，
仍然低于噪声阈值。这正是审查员存在的意义。

随后把验收规则从「两个独立均值的差」改成**配对比较**：同一划分下算差值，
把“这次划分难不难”抵消掉。齿轮数据集上的效果：

| 提议 | 划分 | 叠加后 | 基线 | 配对差值 | 阈值 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| `gear_vibration_multifeature`（12 维） | 5 | 0.8082 ± 0.1508 | 0.7626 ± 0.2206 | **+0.0457** | 0.0793 | 拒绝 |
| 同一个候选 | 10 | 0.6214 ± 0.2283 | 0.6132 ± 0.2453 | +0.0081 | 0.0282 | 拒绝 |

齿轮上配对比较把阈值从 0.44 降到 0.079（相差 5 倍多），但没有改变结论：
**新特征 +4.6 点，而配对标准误也有 4 点**——37 个组的划分方差太大，测不出显著性。

六次生成、六次驳回。这不是失败：它说明**“能生成”不等于“能通过”**，
而验收门槛确实在拦东西。

### 局限（必须写清）

1. 注册的工具**还没有接进 `run_experiment`**——它能被验证，但实验循环还调不到它；
2. 只比较“叠加后 vs 基线”，没有单独消融候选特征的贡献；
3. 生成代码依靠 AST 白名单 + 子进程超时，没有真正的容器隔离。

### 复现

```powershell
python -m entropy_agent.cli lab factory `
  --spec configs/paderborn_3class_fusion_time_extratrees.json `
  --data-root data/paper/paderborn_3class `
  --output results/lab-paderborn-20261106 `
  --goal "为外圈/内圈混淆设计一个多特征提取器" --seeds 5
```

## 5c. CWRU 数据集与一次成功的工具注册

齿轮数据集只有 37 个独立组，划分方差把任何真实提升都吃掉了，所以工厂换到
**CWRU**（12k 驱动端）：

| 项 | 值 |
|---|---|
| 类别 | NORMAL / BALL / INNER_RACE / OUTER_RACE |
| 文件 | 64 个录制（正常 4、球 16、内圈 16、外圈 28） |
| 记录 | 每个文件 6 个 2048 点窗口 → 384 条 |
| 组 | 每个文件一组（64 组） |
| 采样率 | 12 kHz |
| 基线（FUSION_TIME + ExtraTrees，5 划分） | **0.9571 ± 0.0388** |

适配器：`scripts/prepare_cwru_4class.py`；配置：`configs/cwru_4class_fusion_time_extratrees.py.json`。

### 一次真实的「生成 → 失败 → 自我修复 → 注册」

1. LLM 生成 `envelope_spectral_kurtosis_features`；
2. 沙箱执行失败：`scipy.signal.butter` 的截止频率按归一化频率传入，`fs` 默认 1.0 →
   `ValueError: 0 < Wn < fs/2`；
3. **修复循环**把报错回给 LLM，它修正后重新提交；
4. 契约测试通过（形状、有限值、确定性、非常量）；
5. 配对评估：叠加后 **0.9715 ± 0.0302** vs 基线 0.9571 ± 0.0388，
   配对差值 **+0.0144**，阈值 0.0089 → **通过**；
6. 注册到 `results/lab-cwru-20261106/lab_tools/envelope_spectral_kurtosis_features.py`，
   `registry.json` 记录指标与逐划分差值 `[+0.026, +0.021, +0.011, +0.014, 0.000]`。

这是整个项目里**第一个由 LLM 生成、经沙箱与配对评估后注册成功的工具**。

## 6. 测试

`tests/lab/test_lab.py`：6 个测试，用 FakeLLM 覆盖闭环、审查规则、账本、
工具返回结构；不依赖网络和 API key。
