# Agent v1.2 — 预算按“特征提取”计

日期：2026-11-05
数据集：Aero-engine 5 状态（8 单元全付）、Triaxial 7 状态（5/8 单元）
状态：**记账修正确实有效；但“更智能”还没实现，而且验证了 selection bias**

## 1. 为什么要改记账

v1.1 的失败可以归到一句话：**预算的货币错了。**

| 资源 | 单次成本 |
|---|---:|
| 提一次特征（一个熵方法） | 40–350 秒 |
| 拟合一个分类器（特征已存在） | ~0.1 秒 |

一次提取 ≈ 400–3500 次分类器拟合。但 v1 和 v1.1 都把“一个 (方法, 分类器) 组合”当成一个 trial 预算单位，于是把 0.1 秒和 350 秒当成等价资源。结果就是：为了省下“免费”的分类器，Agent 反复换方法提取特征（贵），却在已提取的方法里只试一两个分类器（便宜）。

## 2. v1.2 改了什么

| 改动 | 内容 |
|---|---|
| **unit 预算** | `--units N`：1 个 unit = 1 个熵方法（含参数）。选中一个 unit 就把它的**全部分类器**评一遍 |
| **unit 选择策略** | 锚点方法优先，其余保持候选池顺序。没有比这更“聪明”的在线规则——见 §6 |
| **三种 unit 臂** | `agent`（锚点优先）、`fixed`（池序前 N）、`random`（随机 N 个方法）、`exhaustive`（全部方法） |
| **兼容** | 不给 `--units` 时仍然走 `--rounds/--per-round` 的老路径 |
| **文案** | 报告标题/表头/run_id 按模式区分（v1.2 / v1.1） |

实现位置：`loop.py` 的 `_method_units()` / `_unit_order()` / `_unit_arm()`。

## 3. Aero-engine：8 个 unit 全部付清（预算 = 全部方法）

| 臂 | 均值 Macro-F1 | 标准差 | 胜场 | trial/seed |
|---|---:|---:|---:|---:|
| agent | 0.9967 | 0.0027 | 5 | 48 |
| fixed | 0.9967 | 0.0027 | 5 | 48 |
| random | 0.9967 | 0.0027 | 5 | 48 |
| exhaustive | 0.9967 | 0.0027 | 5 | 48 |

冠军 `TSMEFuDE + SVM-RBF`：validation 1.0000 ± 0.0000，**唯一一次 test = 100.00%**。

成本：特征提取 **0 秒**（全部命中逐记录缓存），960 个 trial 的拟合共 728 秒。

**解读**：aero 只有 8 个方法单元。按正确货币记账后，“8 次提取”本来就等于穷举，所以四条臂必然并列——v1.1 里“15 个 trial 不够用”根本不是预算问题，是记账问题。

## 4. Triaxial：只付 5/8 个 unit（真正受限）

| 臂 | 均值 Macro-F1 | 标准差 | 胜场 |
|---|---:|---:|---:|
| agent | **0.8457** | 0.0228 | 5 |
| fixed | 0.8457 | 0.0228 | 5 |
| random | 0.8273 | 0.0195 | 3 |
| exhaustive | 0.8457 | 0.0228 | 5 |

- 付的 5 个方法是 `TSMEFuDE, RCMFuDE, MFE, MFuDE, IMFRDE`（池序前 5），
  恰好包含全局最优组合 `IMFRDE + OBT-LSTSVM`（0.8828），所以 agent = fixed = exhaustive；
- **随机挑 5 个方法会掉 1.8 个点**（0.8273 vs 0.8457）——“付哪几次提取”在 triaxial 上是真问题；
- 冠军 `IMFRDE + OBT-LSTSVM`：validation **0.8810**（3 次切分均值 0.8828），
  **唯一一次 test = 77.38%**。

成本：特征提取 756 秒（5 个 unit + 穷举臂补的 3 个），690 个 trial 拟合 257 秒。

## 5. 最重要的发现：validation 被挑高了，test 没有变好

| 模型 | validation | test |
|---|---:|---:|
| 论文 anchor（TSMEFuDE + BT-LSTSVM） | 80.95% | 77.38% |
| Agent 冠军（IMFRDE + OBT-LSTSVM） | **88.10%** | **77.38%** |

同样的 test 准确率（65/84），但 validation 高了 7.2 个点。
这就是 **selection bias**：在 84 条 validation 记录上挑 30 个候选的最优者，
最好那个的 validation 必然被高估；错误结构也变了（混淆矩阵不同，错误数相同），
说明它换了一种对验证集更“合身”的错法，而不是学到了更好的东西。

对项目的含义很直接：

- **不能**用 validation 的 88.1% 去说明 Agent 选得更好；
- 能说的是“同样的 test 表现（77.38%）下，Agent 把一个更好看的 validation 交给了我们——
  这恰好说明 validation 数字被挑选抬高了”；
- 84 条 test 记录意味着 **±1 条 = ±1.2 个百分点**，任何小于 2 个点的差异都不该下结论。

## 6. 为什么 unit 选择仍然不“智能”

在 unit 模式下，选哪个方法**必须付一次提取才能看到分数**。8 个方法、预算 5 个 unit，
意味着永远有 3 个方法无法评估。没有跨数据集的先验时，在线学习无法解决这个问题——
这就是为什么 v1.2 的 unit 选择退回到“锚点优先 + 池序”，而不是继续编一个看起来聪明的规则。

（早期草稿试过“优先未访问的熵家族”，在 aero 上把 `MFuDE` 挤掉了——它和 `RCMFuDE`
同属 `fuzzy_dispersion` 家族——直接更差。这个反例已经写进 `_unit_order()` 的注释。）

所以“更智能”的下一层不再是改规则，而是**利用跨数据集的实验历史**（见 §7）。

## 7. 下一步

| 方向 | 内容 | 前置条件 |
|---|---|---|
| L2 可行性过滤 | 用 `DataProfile` 剔除参数与窗口长度不匹配的方法（`scale/m/tau` 的最低点数要求），省掉必然失败的 unit | 无，纯代码 |
| L3 元学习 | 用 SQLite 里的历史（多个数据集 × 方法 × 参数 × 分数）训练排序器，为新数据集给出“先付哪几个 unit”的先验 | 需要 ≥5 个数据集；现在只有 2 个 |
| 评估协议 | test 只有 84 条，建议重复 test 切分或用 group-bootstrap 给区间 | 无 |

## 8. 复现

```powershell
# aero：8 个 unit 全付
python scripts/run_agent_v1.py `
  --spec configs/aero_engine_5state_tsmefude_btlssvm.json `
  --data-root data/paper/aero_engine_5state `
  --output results/agent-v1_2-aero-20261105 `
  --cache-root results/agent-v1_1-cache --units 8 `
  --seeds 20261104,1,2,3,4 --validation-repeats 3

# triaxial：只付 5/8 个 unit
python scripts/run_agent_v1.py `
  --spec configs/triaxial_7class_tsmefude_btlssvm.json `
  --data-root data/paper/triaxial_7class `
  --output results/agent-v1_2-triaxial-20261105 --units 5 `
  --seeds 20261104,1,2,3,4 --validation-repeats 3
```

测试总数：**262 passed**（258 + 4 个 unit 模式测试）。
