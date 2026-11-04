# Aero-engine 5 状态复现 — TSMEFuDE + BT-LSTSVM

日期：2026-11-04
论文：*Time Shift Multiscale Ensemble Fuzzy Dispersion Entropy and Its
Application in Bearing Fault Diagnosis*（Coatings 2025, 15, 779）
DOI：10.3390/coatings15070779
状态：**已在 Aero-engine 5 状态协议上复现**

## 1. 论文协议

来自论文第 5.2 节：

- 航空发动机中介轴承试验台；
- **5 个状态**：2 个健康 + 3 个故障；
- 采样率 25 kHz；
- 每个状态 60 个不重叠样本，每个 2000 点；
- 特征矩阵 60 × 5 × 20（60 样本、5 状态、20 个 TSMEFuDE 尺度）；
- TSMEFuDE：`m=2`、`nc=5`、`tau=1`、`kmax=20`；
- 分类器 BT-LSTSVM；
- 论文切分：每个状态 60% 训练 / 40% 测试；
- 论文报告 TSMEFuDE + BT-LSTSVM 准确率 97.46%。

## 2. 本地数据映射

来源：

```text
7数据源/Aero engine bearing fault diagnosis/dataset/data1.npy ... data5.npy
```

源文件形状：

```text
data1 (504, 8, 20480)
data2 (450, 8, 20480)
data3 (504, 8, 20480)
data4 (504, 8, 20480)
data5 (450, 8, 20480)
```

论文的 5 个状态对应 `data1 … data5`。
本次复现使用通道索引 2，与之前的 aero 配置一致。

数据准备：

```text
scripts/prepare_aero_engine_5state.py
  -> data/paper/aero_engine_5state/state1.npy ... state5.npy
  -> 每个文件：(60, 1, 2000)
```

## 3. 本地协议

训练流程必须有 validation 分区，因此本次运行使用：

```text
train       180 个样本  （每类 36，60%）
validation   60 个样本  （每类 12，20%）
test         60 个样本  （每类 12，20%）
```

论文使用每个状态 36 训练 / 24 测试。这里训练规模完全相同（每类 36），
本地测试集更小。这是一次接近的复现，不是对论文切分的逐位重跑。

配置：

```text
configs/aero_engine_5state_tsmefude_btlssvm.json
```

运行：

```powershell
python scripts/run_supervised_experiment.py `
  --config configs/aero_engine_5state_tsmefude_btlssvm.json `
  --data-root data/paper/aero_engine_5state `
  --output results/aero-engine-5state-tsmefude-btlssvm-20261104 `
  --model-id aero-engine-5state-tsmefude-btlssvm
```

## 4. 结果

| 分区 | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation | 95.00% | 95.00% |
| Test | **98.33%** | **98.33%** |

测试集混淆矩阵（行是真实标签，列是预测标签）：

```text
[[12, 0, 0, 0, 0],
 [ 0,12, 0, 0, 0],
 [ 0, 0,12, 0, 0],
 [ 1, 0, 0,11, 0],
 [ 0, 0, 0, 0,12]]
```

测试集各类召回率：

```text
state1 1.000
state2 1.000
state3 1.000
state4 0.917
state5 1.000
```

论文报告 97.46%；本次在更小的测试分区上得到 98.33%。差异在切分/种子波动范围内，
不应解读为严格意义上的提升。

## 5. 对早前一次运行的更正

早前一次运行报告 69.7% 测试准确率。那次用的是旧的**二分类**配置：
2 类（健康/故障）、2412 条记录、每条记录一个窗口、60/20/20 分组切分。
那**不是**论文协议。

论文用的是 **5 个状态**、每类 60 个 2000 点样本。修正后的 5 状态运行才是本文档记录的这次。

## 6. 局限

- 只用了 1 个通道（索引 2）；论文没有明确写出它的通道索引；
- 本地切分比论文的 60/40 测试集更小；
- 只跑了 1 个随机种子；
- 没有做加噪声实验；
- 还没有与 MPE/MSlopEn/MDE/MFuDE/RCMDE/TSMDE/TSMFuDE 的对比表。

## 7. 后续

1. ~~复现三轴 7 类协议。~~ 已完成 —— 见
   [三轴复现](06_triaxial_reproduction.md)。那次运行测试准确率 77.38%；
   论文的 97.46% 在三轴数据集上不成立，作者自存的该类结果是 83.28%。
2. 补上论文精确的 60/40 切分（或用 manifest 固定每个状态 36 训练 / 24 测试）。
3. 用多个种子重复。
4. 复现论文的方法对比表。
