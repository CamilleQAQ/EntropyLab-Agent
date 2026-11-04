# Agent v1.1 — 逐记录缓存、多次切分、5 seed 对比

日期：2026-11-04
数据集：Aero-engine 5 状态（`AERO_ENGINE_5STATE_TSMEFUDE_BTLSSVM`）
状态：**工程目标全部达成；选型效果仍是负结果——Agent 没赢随机，也输给固定顺序**

## 1. v1.1 改了什么

对着 v1 实测出的四个缺陷逐项改：

| 改动 | 内容 | 验收结果 |
|---|---|---|
| ① 逐记录特征缓存 | key 只含数据集契约（不含 labels/split/classifier）；按 `record_id` 增量填充 | 5 个 seed × 4 臂 × 465 个 trial，**特征提取 0 次**（全部复用） |
| ② 成本感知选择器 | 已提取特征的方法优先；同方法内先扫其余分类器 | 15 次 trial 里出现了 `LSTSVM`、`OBT-LSTSVM`、`RANDOM-FOREST`（v1 一次都没有） |
| ③ 多次切分稳定性 | 每个 trial 在 3 个 group 隔离切分上取 mean ± std；冠军按 mean 选、std 做 tie-break | 报告里每个 trial 都有 ± 值；冠军 `1.0000 ± 0.0000` |
| ④ 多 seed + 同预算统计 | `--seeds` 跑同一套四臂，聚合 mean/std/胜场 | 5 seed 表见 §2；test 仍只开一次 |

另外修了两个真实实现缺陷（都是我自己写的）：

1. **`extracted_methods` 依赖缓存是否预热**：原条件 `if extracted_records or not cache_hit`
   在缓存已热时永不成立，于是 5 个 seed 全被当成“需要新提取”，成本感知那一维完全失效。
   改成一旦取到特征就记录。
2. **选择器打分把“已试分类器”排在“未试分类器”前面**：`quality = 2×方法分 + 分类器分`
   让未试分类器（分数 = −1）永远垫底，所谓“先扫完免费分类器”根本没发生。
   改成严格成本序：`(是否已提取, −方法分, pool 顺序)`。

## 2. 结果（5 seeds，每 trial 3 次切分）

| 臂 | 均值 Macro-F1 | 标准差 | 胜场（共 5） |
|---|---:|---:|---:|
| agent | 0.9922 | 0.0045 | 1 |
| random | 0.9944 | 0.0050 | 3 |
| fixed | **0.9967** | **0.0027** | 5 |
| exhaustive | 0.9967 | 0.0027 | 5 |

逐 seed 最佳 validation：

| seed | agent | random | fixed | exhaustive |
|---:|---:|---:|---:|---:|
| 20261104 | 0.9889 | 0.9889 | 0.9944 | 0.9944 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 0.9944 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 0.9889 | 0.9944 | 0.9944 | 0.9944 |
| 4 | 0.9889 | 0.9889 | 0.9944 | 0.9944 |

冠军：`RCMFuDE + SVM-RBF`（seed 1），validation `1.0000 ± 0.0000`，
**唯一一次 test = 98.33%**（59/60）。

注意：`fixed` 与 `exhaustive` 在 5 个 seed 上完全一致——候选池按论文顺序排列，
前 15 个里每个 seed 都含有全局最优组合。所以在这里“固定顺序”不是运气，而是一个强基线。

## 3. Agent 的实际轨迹（seed 20261104）

| 轮 | 方法 | 分类器 | val Macro-F1 | 理由 |
|---:|---|---|---:|---|
| 0 | TSMEFuDE | BT-LSTSVM | 0.9666 | anchor combination |
| 0 | RCMFuDE | SVM-RBF | 0.9832 | unvisited family fuzzy_dispersion |
| 0 | MFE | EXTRA-TREES | 0.9610 | unvisited family fuzzy |
| 1 | IMFRDE | BT-LSTSVM | 0.9833 | cover family renyi_fuzzy_dispersion |
| 1 | RCMDE | BT-LSTSVM | **0.9889** | cover family dispersion |
| 1 | MPE | BT-LSTSVM | 0.9280 | cover family permutation |
| 2 | RCMDE | SVM-RBF | 0.9833 | sweep SVM-RBF for RCMDE |
| 2 | RCMDE | EXTRA-TREES | 0.9833 | sweep EXTRA-TREES for RCMDE |
| 2 | RCMDE | LSTSVM | 0.9889 | sweep LSTSVM for RCMDE |
| 3 | RCMDE | OBT-LSTSVM | 0.9889 | sweep OBT-LSTSVM for RCMDE |
| 3 | RCMDE | RANDOM-FOREST | 0.9721 | sweep RANDOM-FOREST for RCMDE |
| 3 | IMFRDE | SVM-RBF | 0.9832 | sweep SVM-RBF for IMFRDE |
| 4 | IMFRDE | EXTRA-TREES | 0.9722 | sweep EXTRA-TREES for IMFRDE |
| 4 | IMFRDE | LSTSVM | 0.9889 | sweep LSTSVM for IMFRDE |
| 4 | IMFRDE | OBT-LSTSVM | 0.9833 | sweep OBT-LSTSVM for IMFRDE |

## 4. 为什么还是输

三个原因，按重要性排：

1. **方法选择只基于一次观测。** 第 1 轮 RCMDE 报了 0.9889（3 次切分均值，std 0.0079），
   Agent 就把它当成“最好的方法”，后面 9 个 trial 全花在 RCMDE 和 IMFRDE 上。
   而这两个方法的天花板就是 0.9889，真正能到 0.9967 的 `TSMEFuDE + SVM-RBF` /
   `RCMFuDE + LSTSVM` 从未被组合到。
2. **深度优先用光了预算。** 15 个 trial 里 9 个用于“把 2 个方法扫完”，
   8 个方法只覆盖了 6 个。
3. **aero 已经饱和。** 大量组合落在 0.98–1.00，agent/random 的差别（0.9922 vs 0.9944）
   只有 0.2 个百分点，而单 trial 的 std 就有 0.005。这个数据集已经无法区分策略好坏。

## 5. 成本（这次最值得记的数字）

| 项 | 数值 |
|---|---:|
| 5 seeds × 4 臂 = 465 个 trial 的特征提取 | **0 次 / 0 秒**（复用缓存） |
| 分类器拟合总时间 | 143.6 s |
| v1 冷启动时同样的 8 种熵提取一次 | 477–564 s |

逐记录缓存把“多 seed 实验”从“每个 seed 重提 8 次特征”变成了一次性成本。
这是 v1.1 真正的工程收益，即使选型结论是负的。

## 6. 结论

- v1.1 的工程目标全部达成：缓存、稳定性度量、多 seed、可解释理由、两个实现 bug 修复；
- **选型效果**：agent 0.9922 < random 0.9944 < fixed = exhaustive 0.9967；
  用 5 个 seed 和 3 次切分之后，差距仍然稳定存在，不是噪声；
- 因此现在仍然**不能**声称“Agent 优于随机搜索”或“优于固定顺序”。

## 7. v1.2 的假设（下一轮要验证的）

1. **广度优先**：15 个 trial 正好可以“8 个方法各 1 个分类器（8 次）+ 7 次深化”。
   当前是“6 个方法 + 9 次深化”，广度不够。
2. **用下置信界而不是均值选方法**：`mean − std` 会惩罚只测过一次的方法，
   避免被单个幸运观测锁死。
3. **在 triaxial 上验证**：那里准确率只有 77–83%，不同组合差距 3–6 个百分点，
   才能真正区分策略。aero 已饱和，继续在它上面调选择器没有信息量。

## 8. 复现

```powershell
python scripts/run_agent_v1.py `
  --spec configs/aero_engine_5state_tsmefude_btlssvm.json `
  --data-root data/paper/aero_engine_5state `
  --output results/agent-v1_1-aero-20261104 `
  --cache-root results/agent-v1_1-cache `
  --seeds 20261104,1,2,3,4 --validation-repeats 3
```

产物：

```text
results/agent-v1_1-aero-20261104/
  summary.json     每臂每 seed 的最佳 validation + 聚合 mean/std/胜场
  trials.csv       465 行，含 reason、extracted_records、val_macro_f1_std
  report.md        自动生成的报告
  memory.sqlite    runs / trials 两张表
  champion_model/  唯一一次 test 的 artifact
results/agent-v1_1-cache/   逐记录特征缓存（跨运行复用）
```

测试总数：**258 passed**（251 + 7 个新增/改写的 Agent 测试）。
