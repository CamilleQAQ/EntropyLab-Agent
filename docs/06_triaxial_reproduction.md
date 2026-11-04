# 三轴 7 状态复现 — TSMEFuDE + BT-LSTSVM

日期：2026-11-04
论文：*Time Shift Multiscale Ensemble Fuzzy Dispersion Entropy and Its
Application in Bearing Fault Diagnosis*（Coatings 2025, 15, 779）
DOI：10.3390/coatings15070779
状态：**已复现并冻结为 7 类 anchor；但论文宣称的准确率在该数据集上不成立**

> 语言约定：文档用中文，代码、配置、字段名、指标键保持英文，方便复现和检索。

## 1. 论文协议

依据论文正文和作者提供的代码：

- 三轴感应电机轴承试验台，一个传感器同时测 X/Y/Z；
- **7 类**：300 W 负载下 6 种内圈缺陷宽度 + 1 条健康记录；
- 每类 60 个不重叠窗口；
- 窗长：论文正文写 2000 点，作者代码用 1892（见 §2）；
- 采样率 10 kHz（名义值，见 §8）；
- TSMEFuDE：`m=2`、`nc=5`、`tau=1`、`kmax=20` → 每个窗口 20 个特征；
- 分类器 BT-LSTSVM；
- 论文切分：60% 训练 / 40% 测试。

## 2. 数据映射

数据来源：

```text
7数据源/TriaxialBearingVibrationDatasetofInductionMotorunderVaryingLoad/Dataset
```

这个映射不是猜的，而是由作者的加载函数
`DataloaderOfTraixialBearing.m` 固定：它从下列文件取 `B1`…`B7`，并读取 X 轴
（CSV 第 2 列，0 基索引为 1；第 1 列是时间戳）。

| 类别 ID | 标签 | 源文件（300 W） | 采样点数 | 1892 点窗口数 |
|---:|---|---|---:|---:|
| 0 | `INNER_RACE_0.7MM` | `0.7mm-bearing-faults/0.7inner-300watt-Zo8w7U.csv` | 113,520 | 60 |
| 1 | `INNER_RACE_0.9MM` | `0.9mm-bearing-faults/0.9inner-300watt.csv` | 132,423 | 69 |
| 2 | `INNER_RACE_1.1MM` | `1.1mm-bearing-faults/1.1inner-300watt.csv` | 147,041 | 77 |
| 3 | `INNER_RACE_1.3MM` | `1.3mm-bearing-faults/1.3inner-300watt.csv` | 127,832 | 67 |
| 4 | `INNER_RACE_1.5MM` | `1.5mm-bearing-faults/1.5inner-300watt.csv` | 133,989 | 70 |
| 5 | `INNER_RACE_1.7MM` | `1.7mm-bearing-faults/1.7inner-300watt.csv` | 128,888 | 68 |
| 6 | `HEALTHY` | `Healthy bearing data/healthy without pulley.csv` | 116,975 | 61 |

其他负载（100/200 W）、外圈文件、`Healthy with pulley.csv` **不**属于本协议。

### 为什么是 1892 点而不是 2000 点

作者的驱动函数 `fun_WindTurbineBearingFeatureExtraction.m` 里写的是
`number=60; test_positon=31; N=1892`，切窗方式是
`f = B(1 + N*(i-1) : i*N)`。最短的记录（0.7 mm）正好有
113,520 = 60 × 1892 个点，所以 1892 是唯一能让每类都切满 60 个不重叠窗口的窗长。
若用 2000 点，0.7 mm 只有 56 段、健康只有 58 段，论文说的“每类 60 个样本”就无法复现。

决策：复现作者代码（`N=1892`），并把与论文正文的差异如实记录，而不是补零或让窗口重叠。

## 3. 数据准备

```powershell
python scripts/prepare_triaxial_7class.py
```

产出：

```text
data/paper/triaxial_7class/class1.npy … class7.npy   (60, 1, 1892) float64
data/paper/triaxial_7class/manifest.json             每类的源路径 + SHA-256
```

每类都从第 0 个采样点开始取前 60 个不重叠窗口，与 MATLAB 的切片方式一致。

## 4. 与作者保存特征的交叉验证

作者保存了他们实际使用的特征表：

```text
matlab old code/7特征提取_风电机轴承/mat三相/
  FeatureExtraction_DataAll_WindTurbineBearing_D6H6_60samples.mat
```

`FeatureExtraction_DataAll` 的第 8 个 cell 是 TSMEFuDE，形状 `(420, 21)`，
即 420 个样本 ×（20 个特征 + 1 个从 1 开始的标签），按类别分块排列。

```powershell
python scripts/verify_triaxial_features.py `
  --data-root data/paper/triaxial_7class `
  --reference "…\FeatureExtraction_DataAll_WindTurbineBearing_D6H6_60samples.mat" `
  --output results/triaxial-7class-tsmefude-btlssvm-20261104/feature_parity.json
```

| 检查项 | 结果 |
|---|---|
| 参与比较的样本数 | 420 / 420 |
| 参考标签列与类别顺序一致 | 是 |
| 特征最大绝对差的中位数 | 3.3e-4 |
| 特征最大绝对差的 P95 | 7.0e-4 |
| 差值小于 5e-3 的窗口 | 419 / 420 |
| 异常窗口 | 第 4 类（1.3 mm）第 0 段：0.787 |

结论：数据映射、切窗规则、Python 版 TSMEFuDE 与作者自己的特征矩阵是对得上的。
唯一异常是那一个窗口的原始数据或存档问题；它被保留、没有被剔除，本文档也没有任何
指标依赖“剔除该窗口”。

## 5. 本地实验协议

训练流程需要 validation 分区，因此本次运行使用：

```text
train       252 个样本  （每类 36，60%）
validation   84 个样本  （每类 12，20%）
test         84 个样本  （每类 12，20%）
```

分组方式为「连续 6 个窗口为一组」，同一组的窗口不会跨分区（70 组 → 42 / 14 / 14）。
配置：

```text
configs/triaxial_7class_tsmefude_btlssvm.json
```

运行：

```powershell
python scripts/run_supervised_experiment.py `
  --config configs/triaxial_7class_tsmefude_btlssvm.json `
  --data-root data/paper/triaxial_7class `
  --output results/triaxial-7class-tsmefude-btlssvm-20261104 `
  --model-id triaxial-7class-tsmefude-btlssvm
```

## 6. 结果

| 分区 | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation | 80.95% | 80.82% |
| Test | **77.38%** | **77.26%** |

测试集混淆矩阵（行是真实标签，列是预测标签；顺序为 0.7、0.9、1.1、1.3、1.5、
1.7 mm、健康）：

```text
[[12,  0,  0,  0,  0,  0,  0],
 [ 0,  5,  6,  0,  1,  0,  0],
 [ 0,  3,  9,  0,  0,  0,  0],
 [ 0,  0,  0, 10,  0,  2,  0],
 [ 0,  0,  0,  0,  9,  3,  0],
 [ 0,  0,  0,  0,  4,  8,  0],
 [ 0,  0,  0,  0,  0,  0, 12]]
```

测试集各类召回率：

```text
0.7mm 1.000 | 0.9mm 0.417 | 1.1mm 0.750 | 1.3mm 0.833
1.5mm 0.750 | 1.7mm 0.667 | 健康 1.000
```

两个分区上「健康 vs 故障」都是 100%（12/12 健康，72 个故障记录全部被判为故障）；
所有错误都发生在相邻缺陷宽度之间。这些熵特征在相邻等级上本来就非常接近
（整个语料每个维度的标准差约 0.015，而特征值在 3.15 附近），所以这是一个很难的
7 分类“严重程度分级”问题，而不是流程坏了。

运行成本：总计 202 秒，其中 198 秒用于特征提取（420 个窗口）。

## 7. 与论文、与作者存档的诚实对比

两个独立参照互相矛盾，而本次复现站在较弱的那一边：

| 来源 | TSMEFuDE + BT-LSTSVM，三轴 7 状态 |
|---|---:|
| `docs/PAPER_CONTEXT.md` 记录的论文表格 | 97.46% |
| 作者保存的准确率矩阵（`8故障识别_风电机轴承/mat/BT_BTunify_OBT_OBTunify_…mat`） | 83.28% |
| 本次复现（单 seed，每类 12 个测试样本） | 77.38% |

说明：

1. `PAPER_CONTEXT.md` 里电机轴承表和发动机轴承表**每一行数值完全相同**
   （94.14、96.25、…、97.46），所以 97.46% 几乎可以确定是发动机那组的结果被抄到了
   三轴那一列。而 `docs/05` 的发动机复现确实做到了 98.33%，与 97.46% 属于同一水平。
2. 作者自存的该单元格准确率是 **83.28%**，TSMFuDE 为 86.35%、TSMDE 为 84.27%——
   在他们自己的表里，这个数据集上 TSMEFuDE 并不是最优特征。
3. 与 83.28% 的 6 个百分点差距来自协议和方差，不是数据错误（特征一致性见 §4）：
   - 这里每类 36 个训练窗口，作者代码是 30 个；
   - 这里是随机分组切分，作者代码 `test_positon=31` 是按时间顺序取前 30 / 后 30；
   - 只跑了 1 个 seed。用作者的特征矩阵换 seed 1–4 复跑，测试准确率分别是
     65.5%、66.7%、75.0%、78.6%，即在 ~73% 上下浮动 ±6 个点。

因此：**不要用 97.46% 描述三轴数据集。** 站得住的说法是“该协议下健康 vs 故障
100%；7 类严重程度分级约 75–83%，取决于切分和实现”。

## 8. 局限

- 窗长 1892 与论文正文的 2000 不一致（见 §2）；
- 采样率 10 kHz 取自论文知识库；CSV 时间戳列每秒重复、推算出约 440 点/秒，
  因此该值未经独立核实，只作为 metadata（熵特征计算不使用它）；
- 只用了一个通道（X 轴）、一个 seed、一组参数，没有对 `c1/c2/q` 做网格搜索——
  论文的 BT-LSTSVM 可能做过调参，这是剩余差距的一部分可能来源；
- 论文的 60/40 切分被近似为 60/20/20，因为训练流程必须有 validation；
- 这里的分组隔离避免了相邻窗口跨分区泄漏，因此数值不能与“逐窗口全随机切分”直接比较。

## 9. 冻结状态

```text
data/paper/triaxial_7class/                      准备好的数组 + manifest
configs/triaxial_7class_tsmefude_btlssvm.json    协议
scripts/prepare_triaxial_7class.py               数据准备
scripts/verify_triaxial_features.py              特征一致性审计
results/triaxial-7class-tsmefude-btlssvm-20261104/
  metrics.json、feature_parity.json、
  validation_predictions.csv、test_predictions.csv、model/
```

这个 anchor 已完成，不再重跑调参。它作为 Agent v1 可用的第二个数据集
（第一个是 Aero-engine 5 状态 anchor）；~77–83% 的严重程度分级区间，就是“智能体
搜索在只看 validation、不碰 test 的前提下需要打到的参照”。
