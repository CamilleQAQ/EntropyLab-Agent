# Phase 2 结果 — 契约、Paderborn DatasetSpec 与基线现实检查

日期：2026-11-04
状态：**契约完成；基线质量受阻**
模型训练：真实 Paderborn 数据
`final_holdout`：**未打开**

> **范围说明：** 读完两篇论文的 DOI 之后可以确认，论文主线用的是
> Aero-engine / Triaxial 两组基准，不是 Paderborn。本文档保留，作为跨轴承压力测试。
> 论文复现见 [05_aero_engine_reproduction.md](05_aero_engine_reproduction.md) 与
> [06_triaxial_reproduction.md](06_triaxial_reproduction.md)。

## 1. 已实现的内容

- `configs/paderborn_baseline.json`：MDE + SVM-RBF。
- `configs/paderborn_fusion_baseline.json`：FUSION_TIME_ENVELOPE + ExtraTrees。
- `manifests/paderborn/development.csv`：initial_train + initial_validation，
  960 条记录、12 个轴承；不含 final_holdout。
- 固定切分支持 `test=0`，用于只做开发的运行。
- 修正稀疏窗口契约：`stride` 与 `window size` 相互独立。
  `WindowPlan`、`WindowConfig`、`DiagnosisProfile` 和 artifact 校验
  不再要求 `stride <= size`。
- 单熵方法的配置会拒绝隐藏的 `feature.parameters.sample_stride`。
- `scripts/run_phase2_baseline.py` 跑一个配置，输出 validation 预测、指标和模型 artifact。
- `scripts/group_cv_phase2.py` 在 12 个开发轴承上做 leave-one-bearing-out。

## 2. 结果

| 配置 | 验证准确率 | 验证 Macro-F1 |
|---|---:|---:|
| MDE + SVM-RBF | 24.38% | 21.25% |
| FUSION_TIME_ENVELOPE + ExtraTrees | 28.13% | 16.67% |

MDE 模型训练集准确率 **87.5%**，验证集只有 **24.4%**。这是对训练轴承的过拟合，
不是流水线崩溃。

融合特征的逐轴承验证：

| 轴承 | 类别 | 准确率 |
|---|---|---:|
| K005 | HEALTHY | 0.0% |
| KA05 | OUTER_RACE | 100.0% |
| KA15 | OUTER_RACE | 12.5% |
| KI05 | INNER_RACE | 0.0% |

### 12 个开发轴承上的 leave-one-bearing-out

融合特征 + ExtraTrees，留一轴承：

```text
pooled accuracy = 40.10%
pooled macro-F1 = 39.59%
```

逐轴承准确率：

```text
K001 1.000   K003 1.000   K005 0.000
KA01 1.000   KA04 0.000   KA05 1.000
KA06 0.813   KA15 0.000
KI01 0.000   KI03 0.000   KI05 0.000   KI16 0.000
```

这才是关键结果：**模型学到的是“这是哪个轴承”，不是可迁移的故障特征**。
它能给少数几个轴承打高分，其余全部失败。

## 3. 解读

旧 Paderborn 记录级那个约 98.8% 的数字，来自“同一轴承同时出现在训练集和测试集”的切分。
新的轴承级切分是另一个、困难得多的问题。24–40% 不是新代码造成的回退，
而是把旧的泄漏暴露了出来。

当前证据：

- 更多窗口 / 融合特征并不能填平跨轴承的差距；
- 8 个轴承的训练集太小；
- 12 个轴承留一仍然失败；
- 目前没有任何结果支持一个可用的三分类轴承级模型。

## 4. Phase 2 的决定

**不要在这个基线上继续做 5 轮 Agent。** 当特征表示本身不可迁移时，
让 Agent 去挑方法是没有意义的。

Phase 2 拆成两段：

- **Phase 2a — 已完成：** 契约、DatasetSpec、真实数据通路、基线脚本、稀疏步长修复。
- **Phase 2b — 受阻：** 模型质量 / 域泛化。

## 5. Phase 2b 的备选方案

### 方案 A — 增加训练轴承数量

把全部 20 个非最终轴承用于开发（16 训练 / 4 验证，或分组交叉验证），
只保留 6 个最终轴承和较少的批次轴承。用它检验“更多轴承身份”能否填平差距。

代价：留给持续学习批次的轴承变少。

### 方案 B — 基于工况的训练与更新

在全部 20 个非最终轴承上，用 4 个工况中的 3 个训练；用留出的那个工况验证；
新工况作为更新批次；6 个最终轴承保持未见。

这是最贴近“同一台设备、新的运行数据”的协议，训练组也显著更多。
代价：验证集与训练集共享轴承，因此它衡量的是跨工况泛化，而不是跨轴承泛化。

### 方案 C — 域不变特征

加入逐记录归一化、无量纲特征、包络阶次特征，或显式剔除“轴承身份”信号的特征选择。
这是科学上最强的修法，但需要自己的一轮实验周期。

### 方案 D — 换数据集

CWRU/SEU/JNU 可能提供更多样本或不同机型，但它们同样有自己的身份与泄漏问题。
不应该用它来掩盖 Paderborn 的结果。

## 6. 建议

先做**方案 B**，用方案 A 作为对照：

1. 用 20 个开发轴承中 3 个工况训练；
2. 在留出的工况上验证；
3. 6 个最终轴承保持不动；
4. 与“轴承分组交叉验证”的对照比较。

如果基于工况的训练仍然失败，就转到方案 C，而不是上更大的 Agent。

## 7. 测试

```text
231 passed
```

新增/更新的契约：

- 接受 `test=0` 的开发用 manifest；
- 单熵方法拒绝隐藏的 `sample_stride`；
- 稀疏窗口步长（`stride > size`）被 `WindowPlan`、`WindowConfig`、
  `DiagnosisProfile` 和 artifact 校验接受；
- Paderborn 开发 manifest 排除 final_holdout 分组。

## 8. 产物

```text
configs/paderborn_baseline.json
configs/paderborn_fusion_baseline.json
manifests/paderborn/development.csv
results/phase2-baseline-20261104/
results/phase2-fusion-20261104/
results/phase2-groupcv-20261104/
scripts/run_phase2_baseline.py
scripts/group_cv_phase2.py
```

结果目录被 Git 忽略；manifest 和配置纳入版本管理。
