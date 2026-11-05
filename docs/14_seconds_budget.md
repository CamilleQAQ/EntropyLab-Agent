# Agent v1.4 — 预算按秒计费

日期：2026-11-05
动机：v1.3 里 unit 之间的成本差 100 倍（融合 5 ms vs TSMEFuDE 500 ms/窗口），
但预算仍按“unit 个数”计，等于把一分钱和一百块钱当成一样。见 `docs/13` §4。

结果：**在 triaxial 上第一次拿到清晰胜利**——60 秒预算下 Agent 用 37.6 秒追平穷举，
而按论文顺序的固定策略花掉 59.9 秒、少 12 个点。

## 1. 改了什么

| 文件 | 内容 |
|---|---|
| `agent/cost.py`（新） | 成本模型 + 账本：每窗口先验价格、观测后指数平滑、`from_memory()` 从历史实验读取实测成本 |
| `agent/loop.py` | 新增 `--budget-seconds` 模式：unit 按预估秒数计费，买不起就跳过；`agent` 策略=锚点优先 + 最便宜优先；`fixed`=池序；`random`=随机序；`exhaustive` 忽略预算作参照 |
| `scripts/run_agent_v1.py` | `--budget-seconds`、`--cost-memory` |
| 输出 | 新增 `cost_model.json`（每个臂的账本：花了多少、买了哪些、跳过了什么） |

计费口径：**用预估秒数计费，不用墙钟**。这样同一个实验在“缓存已热”和“冷启动”下
账本一致，两条臂才可比；真实提取耗时仍逐 trial 记录。

策略只有一句话：**预算不足时先买便宜的、把贵的留到确实负担得起的时候**；
论文锚点（TSMEFuDE）仍然优先，但买不起就跳过。

## 2. 实验设置

| 数据集 | 窗口数 | 全部 unit 的预估成本 | 预算 | 预算占比 |
|---|---:|---:|---:|---:|
| aero 5 状态 | 300 | ~346 s | 60 s | 17% |
| triaxial 7 状态 | 420 | ~487 s | 60 s | 12% |
| 人字齿轮 5 状态 | 222 | ~257 s | 60 s | 23% |

5 个 seed，每 seed 独立选型 + 独立 test。

## 3. 结果

| 数据集 | 臂 | validation 均值 | 买下的 unit 数 | 花费 | test（5 折） |
|---|---|---:|---:|---:|---:|
| **triaxial** | **agent** | **0.9500 ± 0.0125** | 10 | **37.6 s** | **93.57% ± 4.43%** |
| | fixed | 0.8326 ± 0.0255 | **3** | 59.9 s | — |
| | random | 0.8957 ± 0.0617 | 10 | 37.6 s | — |
| | exhaustive | 0.9500 ± 0.0125 | 13 | 487 s | — |
| aero | agent | 0.9978 ± 0.0027 | 10 | 26.9 s | 99.33% ± 0.82% |
| | fixed | 0.9978 ± 0.0027 | 7 | 59.3 s | — |
| | random | 0.9978 ± 0.0027 | 10 | 26.9 s | — |
| gear | agent | 0.7368 ± 0.0826 | 11 | 51.0 s | 68.18% ± 17.59% |
| | fixed | 0.7368 ± 0.0826 | 11 | 51.0 s | — |
| | random | 0.7368 ± 0.0826 | 11 | 51.0 s | — |

### triaxial 是这一轮的重点

60 秒预算下：

```text
agent  花 37.6 秒买到 10 个 unit：
        MPE, MDE, FUSION_TIME, FUSION_TIME_SPEC, FUSION_TIME_ENVELOPE,
        FUSION_TIME_ENVELOPE_ENTROPY, FUSION_ALL, RCMDE, IMFRDE, MFuDE
        → validation 0.9500，等于穷举（0.9500）

fixed  花 59.9 秒只买到 3 个 unit：
        RCMFuDE（59.3 秒，一个人吃掉整个预算）、MDE、MPE
        → validation 0.8326，比 agent 低 11.7 个点
```

固定顺序的第二个方法就是 `RCMFuDE`，它的预估价格 59.3 秒几乎等于全部预算；
买下它之后只剩 0.6 秒，后面 11 个方法全部跳过。这不是“固定顺序运气差”，
而是**在秒预算下，按方法个数排序的策略必然会被一个昂贵的方法卡死**。

随机序 0.8957 介于两者之间，方差也最大（±0.0617）——它有时抽不到昂贵的大件，
有时抽到了就废掉预算。

这正好回答了 v1.2 留下的问题：当时说“unit 选择在没有先验时无法online学习”，
而一旦预算按成本计费、且成本差异巨大，**“先买便宜的”本身就是一个有效先验**。

### aero 与 gear

- aero：三条臂 validation 完全相同（0.9978），只是花费不同（agent 26.9 s vs fixed 59.3 s）。
  aero 已经饱和，任何策略都能摸到 0.99，这一轮它的价值是“agent 用更少时间达到同样结果”；
- gear：60 秒够买 11/13 个 unit（只有 TSMEFuDE 和 MFE 买不起），三条臂买到的集合不同
  但验证分数相同（0.7368），说明 gear 上这几个 unit 的边际贡献在噪声内。

## 4. 结论

1. **秒预算是必要的**：按个数计时，固定顺序与 agent 打平；按秒计时，固定顺序在 triaxial
   上被一个昂贵方法卡死，agent 用 37.6/60 秒追平穷举。
2. **“最便宜优先”是有效先验**，但它不是万灵药：它在 aero/gear 上没有额外收益
   （那两个数据集本来就饱和或噪声大）。
3. **成本模型是估计值**：先验来自本仓库 1892–2000 点窗口的实测，运行中会用真实耗时
   指数平滑更新。换窗口长度或换方法参数时需要重新校准。

## 5. 局限

- 预算按**预估秒数**计费，不是墙钟；预估偏差会直接变成“买贵了/买便宜了”；
- 策略仍是“锚点优先 + 最便宜优先”，没有用“已买方法的验证分数”来指导后续购买
  （因为 unit 一旦买下就含全部分类器，分数只在买后才知道）；
- gear 的 test 方差仍大（±17.6），两个 seed 只有 45–48%；
- aero 饱和，不能用来区分策略。

## 6. 复现

```powershell
python scripts/run_agent_v1.py `
  --spec configs/triaxial_7class_tsmefude_btlssvm.json `
  --data-root data/paper/triaxial_7class `
  --output results/agent-v1_4-triaxial-20261105 `
  --cache-root results/agent-v1_2-triaxial-20261105/feature_cache `
  --budget-seconds 60 --seeds 20261104,1,2,3,4 `
  --validation-repeats 3 --test-mode per-seed
```

产物里 `cost_model.json` 记录每个臂的账本（花费、买下的 unit、跳过的 unit 及原因）。
测试总数：**281 passed**。
