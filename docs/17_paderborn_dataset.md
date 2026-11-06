# 第四个数据集：Paderborn（3 类 + 4 工况）

日期：2026-11-06
状态：**已接入并跑通 5 个独立划分；这是目前唯一同时具备"真实工况漂移"和"足够 headroom"的实验台。**
V1 重训基线在它上面暴露了与 VBL 相同的判据失效：5 个划分里只有 1 个发生了更新。

## 1. 为什么选它

持续学习需要两样东西，之前的数据集各缺一样：

| 数据集 | headroom | 真实漂移 |
|---|---|---|
| Aero | 无（饱和） | 无 |
| 齿轮 | 有 | 有，但只有 37 组，holdout 无法分层 |
| VBL | 弱（frozen 0.98） | 弱（同台架） |
| **Paderborn** | **有（0.88）** | **有（4 个工况）** |

## 2. 数据结构

```text
Paderborn-source-verification-20261102/
  archives/<轴承>/<轴承>/*.mat      # 每个 .mat 一次测量
  all_documents/<轴承>/*.pdf        # factsheet + 测量日志（32 个轴承）
```

每个 `.mat` 的结构：

| 字段 | 内容 |
|---|---|
| `X` | 时间栅格：Mechanical 4 kHz（16001 点）/ HostService 64 kHz（256001 点）/ Temp 1 Hz |
| `Y` | 7 个信号：force、phase_current_1/2、speed、temp、torque、**vibration_1** |
| 文件名 | `{工况}_{轴承}_{第几次测量}.mat` |

适配器用 `Y.vibration_1`（64 kHz，4 秒），跳过前 20000 点，取 2 个 2048 点窗口，
`group_size = 2` → **一个 group = 一次测量**。

## 3. 标签：KA/KI 不是"人工/真实"

从 64 个本地 factsheet 里解析出的 `Component` 字段：

| 类型 | 数量 | 轴承 |
|---|---:|---|
| HEALTHY | 6 | K001–K006 |
| **OR 外圈** | 12 | KA01, KA03, KA04, KA05, KA06, KA07, KA09, KA15, KA16, KA22, KA30, KB27 |
| **IR 内圈** | 13 | KB23, KB24, KI01, KI03, KI04, KI05, KI07, KI08, KI14, KI16, KI17, KI18, KI21 |
| AR | 1 | KA08 |

**注意**：`KA04`、`KA15` 是**寿命试验**产生的退化，`KI08` 是**电刻**人工损伤。
按前缀理解成"KA=人工、KI=真实"会标错——这也是旧划分没过审的原因（README 写明）。

## 4. 原型采用 9 个轴承（3 类 × 3）

| 类 | 轴承 | 选择理由 |
|---|---|---|
| HEALTHY | K001、K002、K003 | — |
| OUTER_RING | KA04、KA15、KA16 | 全部**寿命试验**制造 |
| INNER_RING | KI04、KI14、KI16 | 全部**寿命试验**制造 |

**控制制造方式**是关键：如果 OR 用寿命试验、IR 用电刻，模型可能学到"电刻痕迹"
而不是"损伤位置"，那是一个混杂捷径。

## 5. 可行性探针（适配器之前先验证数据）

同一工况内、4 个轴承做 4 类分类（12 训练测量 / 8 测试测量，group 隔离）：

| 工况 | 同工况准确率 |
|---|---:|
| N09_M07_F10 | 1.0000 |
| N15_M01_F10 | 1.0000 |
| N15_M07_F04 | 1.0000 |
| N15_M07_F10 | 1.0000 |

跨工况（训练→测试）：

| 训练 \ 测试 | N09_M07_F10 | N15_M01_F10 | N15_M07_F04 | N15_M07_F10 |
|---|---:|---:|---:|---:|
| N09_M07_F10 | 1.000 | 0.294 | 0.350 | 0.331 |
| N15_M01_F10 | 0.431 | 1.000 | 0.931 | 1.000 |
| N15_M07_F04 | 0.625 | 0.950 | 1.000 | 0.875 |
| N15_M07_F10 | 0.487 | 1.000 | 0.956 | 1.000 |

结论：

1. **历史上的 24–40% 不是数据问题**，同工况内能到 100%；旧实验大概把工况混训了；
2. **N09（900 rpm）是极端域**，跨过去只有 0.29–0.35（3–4 类随机水平）；
3. **三个 N15 工况之间是中等漂移**（0.87–1.00），正好是持续学习该处理的量级。

## 6. 协议

| 项 | 取值 |
|---|---|
| 类别 | HEALTHY / OUTER_RING / INNER_RING |
| 轮次 | N15_M07_F10 → N15_M01_F10 → N15_M07_F04 |
| 每个 (类, 工况) 的 20 次测量 | 4 holdout + 3 probe + 3 validation（仅参考工况）+ 其余训练 |
| holdout | 108 group = 216 条，跨三个工况，**只评一次** |
| probe | 81 group，**永不训练**（为 V2 的判据准备） |
| 训练 | round0 90 group（180 条）→ round1 117 → round2 117 |
| 特征/分类器 | FUSION_TIME + ExtraTrees（V1.4 冠军配置，先固定） |
| 划分次数 | 5 个独立测量分配（`--seed`） |

## 7. 基线结果（V1 重训循环，5 个划分）

| 划分 | frozen | continual | joint | 关键事件 |
|---:|---:|---:|---:|---|
| 20261104 | 0.8935 | 0.8935 | 0.9583 | 验证集 0.9815，两次 reject |
| 1 | 0.8796 | 0.8796 | 0.9491 | 同上 |
| 2 | 0.8750 | 0.8750 | 0.9583 | 同上 |
| 3 | 0.9028 | 0.9028 | 0.9676 | 同上 |
| 4 | 0.8704 | **0.9630** | 0.9630 | round 2 的 challenger 达到 1.0000 → promote |

| 臂 | accuracy | macro-F1 |
|---|---:|---:|
| frozen | 0.8843 ± 0.0135 | 0.8828 |
| continual | 0.9028 ± 0.0354 | 0.9015 |
| **joint** | **0.9593 ± 0.0069** | 0.9592 |

| 比较 | 结果 |
|---|---|
| continual − frozen（逐划分配对） | **+0.0185 ± 0.0414**，5 个划分里**只赢 1 个** |
| continual − joint | **−0.0565**，5/5 落后 |

按工况拆开（seed 20261104）：

| 臂 | N15_M07_F10 | N15_M01_F10 | N15_M07_F04 |
|---|---:|---:|---:|
| frozen | 0.9583 | 0.9722 | **0.7500** |
| joint | 0.9306 | 0.9722 | **0.9722** |

按轴承拆开：KA04 0.667→0.875、KA15 0.583→0.875，其余基本 1.0。

两个重要观察：

1. **headroom 集中在 N15_M07_F04**（0.75 → 0.97），而且"补上"它会让 F10 掉 2.8 个点——
   **joint 并不是全面更好**，这正是 guardrail 要保护的；
2. **判据又一次失效**：4/5 个划分的 round-0 验证集已经是 0.96–0.98，
   只有 seed 4 在 round 2 让 challenger 摸到 1.0000 才发生更新，而那一次直接 +9.3 点。

## 8. 统计上的坑（必须写清）

聚合脚本给出的池化 bootstrap `+0.0185 [0.0111, 0.0269]`、McNemar `p = 1.1e-05` **不能当作显著**：

- 5 个划分是**同一批测量**的不同分配，记录跨划分重复；
- 池化后把重复记录当独立样本，等于人为放大样本量。

可信的表述是**逐划分配对**：`+0.0185 ± 0.0414`，只赢 1/5；其余 4 个划分差值为 0
（没有任何更新发生）。也就是说，**现有收益全部来自"判据碰巧允许了一次更新"**。

## 9. 对 V2 的含义

Paderborn 给了 V2 一个理想的实验台：

1. **有 headroom**：continual 与 joint 差 5.6 个点；
2. **有真实权衡**：更新 F04 会伤 F10，guardrail 有实际作用；
3. **判据是唯一的瓶颈**：只要允许正确的更新，seed 4 已经证明能追平 joint（0.9630 = 0.9630）；
4. **有 probe**：81 个 group 已预留，V2 的 State/Reward 可以直接用。

## 10. 复现

```powershell
# 解压（Windows 自带 tar 即可）
cd Paderborn-source-verification-20261102\archives
foreach ($b in @('K003','KA16','KI04','KI14','KI16')) { tar -xf "$b.rar" -C . }

# 打包 + 计划 + 基线（在 renovation/ 下）
python scripts/prepare_paderborn_3class.py `
  --source ..\Paderborn-source-verification-20261102\archives `
  --output data/paper/paderborn_3class
python scripts/prepare_paderborn_continual_plan.py `
  --manifest data/paper/paderborn_3class/manifest.json `
  --output data/paper/paderborn_3class/continual_plan_seed20261104.json `
  --dataset-id PADERBORN_3CLASS_FUSION_TIME_EXTRA_TREES --seed 20261104
python -m entropy_agent.cli continual run `
  --spec configs/paderborn_3class_fusion_time_extratrees.json `
  --data-root data/paper/paderborn_3class `
  --plan data/paper/paderborn_3class/continual_plan_seed20261104.json `
  --output results/continual-paderborn-seed20261104-20261106
```
