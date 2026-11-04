# Agent v1 — 5 轮选型 + 实验记忆

日期：2026-11-04
数据集：Aero-engine 5 状态（`AERO_ENGINE_5STATE_TSMEFUDE_BTLSSVM`）
状态：**v1 可用并已冻结；选型效果与随机臂持平，低于固定顺序与穷举上界**

## 1. 目标

最小闭环，不做 LangGraph：

```text
DataProfile → 候选排序 → 5 轮 × 3 个 trial
  → 每轮只看 validation → 写入 SQLite 实验记忆
  → 第 5 轮结束按 validation 选冠军 → test 只开一次
```

同时跑同预算对照，证明“选得比随机好”是否成立：

| 臂 | 内容 |
|---|---|
| `agent` | 5 轮 × 3 = 15 个 trial，分轮调整 |
| `random` | 同 15 个预算，固定 seed 随机抽 |
| `fixed` | 同 15 个预算，按候选池固定顺序 |
| `exhaustive` | 48 个组合全跑，作为上界 |

test 只对 **agent 冠军** 开一次；其余臂只报告 validation。

## 2. 实现

```text
src/entropy_agent/agent/
  profile.py     DataProfile：只从注入的 DatasetSpec + 已解析切分推导
  candidates.py  8 种熵 × 6 个分类器 = 48 个候选（每个方法有自己的参数名）
  cache.py       特征缓存：key = 数据集契约摘要 + 记录集摘要
  memory.py      SQLite：runs / trials，失败也落库
  selector.py    确定性排序：第 0 轮覆盖家族，第 1 轮按分数覆盖，第 2 轮起利用
  loop.py        四臂执行、冠军选择、唯一一次 test、报告
```

运行入口：

```powershell
python scripts/run_agent_v1.py `
  --spec configs/aero_engine_5state_tsmefude_btlssvm.json `
  --data-root data/paper/aero_engine_5state `
  --output results/agent-v1-aero-20261104 `
  --rounds 5 --per-round 3
```

**特征缓存**是按 `(数据集契约, 记录集)` 做 key 的：换分类器、换轮次都命中，
只有换熵方法或参数才重新提特征。这是整套设计里最关键的一点（见 §4）。

**候选池**不写死数据集：同一份池子可以被任何 `DatasetSpec` 使用。每个熵方法的参数名
不同（`kmax` / `scale` / `r`+`n` / `type_`+`q`），所以池子里为每个方法单独声明。

### 冠军元数据（2026-11-04 修复）

初版有一个元数据 bug：冠军 artifact 的 `diagnosis.display_name` 直接抄了 spec 里的
原始配置，于是出现「模型实际是 `RCMDE + SVM-RBF`，说明卡却写着 `TSMEFuDE + BT-LSTSVM`」。

修复方式（只动 Agent 层，不动核心 `train_dataset`）：

- `champion_diagnosis()`：如果配置的 `display_name` 以「配置的熵 + 配置的分类器」结尾，
  就把这一段替换成冠军的组合，得到
  `Aero-engine 5-state — agent-selected RCMDE + SVM-RBF`；
  没有这个后缀时退化为「原名称 — agent-selected …」；
- `profile_id` 保持不变：它描述的是诊断契约（信号类型、类别集、采样率），
  不是哪个算法碰巧赢了；
- `champion_model_id()`：`model_id` 现在包含冠军组合
  （`aero_engine_5state_tsmefude_btlssvm-agent-rcmde-svm-rbf`），
  不再是一个不说明任何信息的 `…-agent-champion`；
- 报告 §5 也会打印这个 `model_id`。

artifact 仍能正常加载（digest 校验通过），`labels.json` 不变。

## 3. 结果（Aero-engine 5 状态）

| 臂 | trial 数 | 最佳 val Macro-F1 | 对应 Accuracy | 缓存命中 |
|---|---:|---:|---:|---:|
| agent | 15 | 0.9833 | 0.9833 | 9/15 |
| random | 15 | 0.9833 | 0.9833 | 13/15 |
| fixed | 15 | 1.0000 | 1.0000 | 15/15 |
| exhaustive | 48 | 1.0000 | 1.0000 | 48/48 |

Agent 的 15 次选择：

| 轮 | trial | 方法 | 分类器 | val Macro-F1 |
|---:|---:|---|---|---:|
| 0 | 0 | TSMEFuDE | BT-LSTSVM | 0.9500 |
| 0 | 1 | RCMFuDE | SVM-RBF | 0.9664 |
| 0 | 2 | MFE | EXTRA-TREES | 0.9165 |
| 1 | 3 | IMFRDE | SVM-RBF | 0.9664 |
| 1 | 4 | RCMDE | SVM-RBF | 0.9833 |
| 1 | 5 | MPE | SVM-RBF | 0.9833 |
| 2 | 6 | RCMDE | BT-LSTSVM | 0.9833 |
| 2 | 7 | MPE | BT-LSTSVM | 0.9161 |
| 2 | 8 | TSMEFuDE | SVM-RBF | 0.9664 |
| 3 | 9 | RCMFuDE | BT-LSTSVM | 0.9833 |
| 3 | 10 | IMFRDE | BT-LSTSVM | 0.9833 |
| 3 | 11 | RCMDE | EXTRA-TREES | 0.9833 |
| 4 | 12 | RCMFuDE | EXTRA-TREES | 0.9667 |
| 4 | 13 | IMFRDE | EXTRA-TREES | 0.9499 |
| 4 | 14 | MPE | EXTRA-TREES | 0.9667 |

冠军：`RCMDE + SVM-RBF`（第 1 轮，trial 4）。

**唯一一次 test**：

| 分区 | Accuracy | Macro-F1 |
|---|---:|---:|
| validation | 98.33% | 98.33% |
| test | **100.00%** | **100.00%** |

测试集混淆矩阵为对角阵（每类 12/12 全对）。注意：test 只有 60 条，
且它比 validation 更容易，所以 100% 不应解读为“比 anchor 的 98.33% 更好”。

### 穷举上界（validation）

| 排名 | 方法 | 分类器 | val Macro-F1 |
|---:|---|---|---:|
| 1 | TSMEFuDE | LSTSVM | 1.0000 |
| 2 | RCMFuDE | LSTSVM | 1.0000 |
| 3 | MFuDE | LSTSVM | 1.0000 |
| 4 | IMFRDE | LSTSVM | 1.0000 |
| 5 | RCMFuDE | BT-LSTSVM | 0.9833 |

## 4. 成本分析（这次运行真正值得写下来的部分）

| 项 | 数值 |
|---|---:|
| 特征提取总耗时（8 种熵各一次） | 477 s |
| 分类器拟合总耗时（78 个 trial） | 12 s |
| 其中 TSMEFuDE 一次 | 346 s |
| 其中 MFE 一次 | 91 s |
| 穷举臂 48 个 trial 的额外提取耗时 | **0 s**（全部命中缓存）+ 5.5 s 拟合 |

结论：**成本由“熵方法数”决定，不由 trial 数决定。** 48 个组合全跑只比 15 个多花几秒，
因为分类器复用同一份特征。所以“15 个 trial 的预算”约束的是选择策略，不是算力。

## 5. 与上界的差距：一条验证记录

Agent 的 0.9833 等于 60 条验证记录里错了 1 条（59/60），穷举上界是 60/60。
所以差距是 **1 条记录**，不是档次差别。

但差距存在的原因很清楚：

1. Agent 从头到尾没试过 `LSTSVM`；而穷举前 4 名全是 `LSTSVM`。
2. 第 2 轮之后，选择器按“方法的历史最高分”做贪心，锁在 0.9833 这个平台上
   （RCMDE / MPE / IMFRDE / RCMFuDE 都能到这个分），于是把剩余预算全花在
   同一批方法换分类器，没有回到“TSMEFuDE 家族还有 5 个分类器没试”。
3. `fixed` 臂拿到 1.0 是**顺序碰巧**：它的第 4 个 trial 恰好是 TSMEFuDE + LSTSVM。
   这不是策略更好——正因为如此才需要多数据集、多种子才能下结论。

## 6. 结论

- v1 的工程目标全部达成：可运行、可审计、失败落库、缓存生效、test 只开一次；
- 选型效果：**15/48 的预算拿到 59/60，与随机臂持平，输给固定顺序和穷举**；
- 这是一个诚实的基线结果，**不能**写成“Agent 优于随机搜索”。

## 7. v1.1 要改什么（下一步）

1. **把分类器当免费资源**：只要某方法已提取过特征，就先扫完它的分类器，再考虑新提取
   （成本感知排序）。
2. **家族内不确定性**：一个方法只试过 1 个分类器时，不应当被判定为“差”；
   用“方法内已试分类器数”降低其置信度。
3. **冠军判定加稳定项**：validation 只有 60 条，1 条记录就会翻转排名，
   可以用重复切分或置信区间把“1 条记录的差别”变成“不确定”。
4. 之后再把同一份代码原样跑 Triaxial（不改一行），验证选型没有对 Aero-engine 过拟合。

## 8. 复现

```text
src/entropy_agent/agent/            实现
scripts/run_agent_v1.py             入口
tests/agent/                        20 个单元测试（合成小数据，5 秒）
results/agent-v1-aero-20261104/
  summary.json、trials.csv、report.md、memory.sqlite、champion_model/
results/agent-v1-aero-20261104-cache/
  feature_cache 的共享副本（重跑时用 --cache-root 指过来，全部命中）
```

重跑（复用缓存，几秒钟）：

```powershell
python scripts/run_agent_v1.py `
  --spec configs/aero_engine_5state_tsmefude_btlssvm.json `
  --data-root data/paper/aero_engine_5state `
  --output results/agent-v1-aero-20261104 `
  --cache-root results/agent-v1-aero-20261104-cache `
  --rounds 5 --per-round 3
```

测试总数：**251 passed**（231 旧 + 20 Agent/元数据）。
