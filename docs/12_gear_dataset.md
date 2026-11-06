# 第三个数据集：人字齿轮 5 状态

日期：2026-11-05
状态：**接入完成，核心零改动；数据集本身很难，且再次证明单切分数字不可信**

## 1. 为什么选它

前两个数据集都是**轴承**（aero 发动机轴承、triaxial 电机轴承）。第三个数据集换成
**齿轮箱**，才知道 Agent 不是只对轴承调过参。工作区里可选的齿轮数据有两类结构：

| 候选 | 结论 |
|---|---|
| 直齿轮 / 斜齿轮 | 健康类只有 2 个文件，group 隔离切分要求每类 ≥3 组 → 不可用 |
| **人字齿轮** | 5 类、37 个文件 → 采用 |
| 风电叶片 | 每个 CSV 只有约 600 个点，连一个 2000 点窗口都切不出 → 不可用 |

## 2. 数据与协议

来源：

```text
7数据源/GearDatasets/GearDatasets/人字齿轮/{健康,裂纹,磨损,断齿,剥落}/*.csv
```

每个 CSV 的结构：

```text
第 1 行  15360,460800,10,0,3     采样率 15360 Hz、460800 点、第 3 字段（不是载荷）
第 2-4 行 通道元信息（最大值/使能标志/零行）
第 5 行起 采样数据，10 列
```

- 前 9 列是振动通道，第 10 列是转速类信号；
- 归档里**没有任何文档说明 9 个振动通道的安装位置**，所以协议固定用**第 1 个振动列**，
  并把这一选择写进 manifest，而不是假装它有物理含义；
- 窗口 2000 点（0.13 s @15.36 kHz），不重叠，每个文件取前 6 个；
- **每个类的全部文件**都参与（见 §3 的载荷问题），因此类别记录数不均：30 / 60 / 48 / 36 / 48。

```text
data/paper/gear_5state/class1..5.npy   (30|60|48|36|48, 1, 2000)
data/paper/gear_5state/manifest.json   每个源文件的路径、采样率、载荷、SHA-256
```

## 3. 载荷混杂：差点把工况当成类别

归档把 10/15/20 Nm 混放在同一个类别文件夹里，各类的载荷分布并不一致：

| 类别 | 文件数 | 载荷分布 |
|---|---:|---|
| HEALTHY | 5 | 10×2, 15×2, 20×1 |
| CRACK | 10 | 10×5, 15×5 |
| WEAR | 8 | 10×4, 20×4 |
| MISSING_TOOTH | 6 | 10×3, 15×3 |
| SPALLING | 8 | 10×4, 20×4 |

最初我按“每类前 5 个文件”取数，那样会把**载荷**变成类别的代理特征——所以改成
用每类的全部文件，并让 manifest 记录每个源文件的载荷。

即便如此，随机 group 切分仍可能把某个类别的一个载荷整体分到 test：
实际切分里 HEALTHY 的 10 Nm 文件全部落在 validation/test，训练集只有 15/20 Nm。

**但这不是 test 失效的主因**——量化过：

| test 记录 | 数量 | 准确率 |
|---|---:|---:|
| (类别, 载荷) 组合在训练集出现过 | 60 | 53.3% |
| (类别, 载荷) 组合没出现过 | 6 | 50.0% |

两者几乎一样，说明模型的问题是**跨文件泛化**，不是没见过载荷。

## 4. 接入代价：只有两个新文件

```text
scripts/prepare_gear_5class.py
configs/gear_5state_tsmefude_btlssvm.json
```

核心代码（`analysis/`、`classifiers/`、`io/`、`training/`、`diagnosis/`、`agent/`）
**一行没改**。这是 AGENTS.md 里“加数据集只需适配器 + 配置”的第一次实测通过。

## 5. 论文 anchor 组合（TSMEFuDE + BT-LSTSVM）

| 分区 | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation | 83.33% | 82.38% |
| Test | **53.03%** | **50.06%** |

Test 混淆矩阵（行=真实，列=预测；顺序 HEALTHY, CRACK, WEAR, MISSING_TOOTH, SPALLING）：

```text
[[ 3, 3, 0, 0, 0],
 [ 0, 6, 1, 0, 5],
 [ 7, 1, 3, 5, 2],
 [ 0, 0, 0,12, 0],
 [ 0, 6, 1, 0,11]]
```

各类召回：HEALTHY 0.50、CRACK 0.50、WEAR 0.17、MISSING_TOOTH 1.00、SPALLING 0.61。

validation 83% 而 test 53%，差 30 个点——这是三个数据集里最难的：

- 只有 120 条训练记录（20 个文件）；
- 每个文件是独立一次运行，文件间幅值/相位差异大；
- 36 条 validation 太小，波动极大。

## 6. Agent 在这个数据集上

`--units 8`（8 个方法全付）× 5 个 seed × 3 次切分：

| 臂 | 均值 Macro-F1 | 标准差 | 胜场 |
|---|---:|---:|---:|
| agent / fixed / random / exhaustive | 0.5743 | 0.1543 | 5 |

四臂并列（因为 8 个 unit 就等于穷举），但**方差极大**：逐 seed 最佳 validation 分别是
0.848 / 0.634 / 0.508 / 0.454 / 0.428。这个数据集的 validation 非常不稳定。

### 单切分 vs 5 个独立切分

| 报告方式 | test accuracy |
|---|---:|
| 单切分（挑 validation 最高的 seed） | 68.18%（bootstrap 56.1–78.8%） |
| **5 个 seed 各自独立选型 + 独立 test** | **52.12% ± 12.66%** |

逐 seed：

| seed | 冠军 | validation | test |
|---:|---|---:|---:|
| 20261104 | MPE + EXTRA-TREES | 0.9722 | 0.6818 |
| 1 | RCMFuDE + RANDOM-FOREST | 0.8333 | 0.5455 |
| 2 | IMFRDE + OBT-LSTSVM | 0.7222 | 0.4848 |
| 3 | TSMEFuDE + EXTRA-TREES | 0.6389 | 0.5909 |
| 4 | TSMEFuDE + BT-LSTSVM | 0.3611 | 0.3030 |

**68.18% 是“挑了最容易的那个切分”的结果**：单切分模式选冠军是在所有 seed 里取
validation 最高者，而 validation 最高的 seed 恰好 test 也最容易。5 个独立切分后
真实水平是 52.1% ± 12.7%，和 anchor 的 53.0% 基本一致。

这再次说明：**凡是只有一个 test 数字的结论，都要先问它是哪个切分。**

## 7. 为什么只有 52%：三组对照（2026-11-05 补）

一开始我把原因写成“数据集很难”。做了对照之后要更正：**数据集本身是可分的，
是我们的表示（只用熵特征、只用一个任意挑的通道）在这里不够用。**

### 对照 1：同一文件组切分下，简单时频特征逐通道

不用任何熵特征，每个窗口只算 11 个统计量（RMS、峰值因子、峭度、偏度、
过零率、谱质心、5 个频段能量占比），RandomForest 300 棵：

| 通道 | test accuracy |
|---|---:|
| ch0（我们用的） | 81.8% |
| ch1 | 59.1% |
| ch2 | 69.7% |
| ch3 | 59.1% |
| ch4 | 78.8% |
| ch5 | 74.2% |
| ch6 | 65.2% |
| **ch7** | **92.4%** |
| ch8 | 90.9% |
| ch9（转速信号） | 83.3% |
| 九个振动通道合并 | **95.5%** |

两件事：**数据集可分**（95.5%），而且**通道选错要掉 10–30 个点**——
我们用的 ch0 恰好是中等偏上的那个，但不是最好的。

### 对照 2：8 种熵 × 3 个分类器的上限

直接用 Agent 缓存下来的 8 组熵特征，在同一 train/test 上换分类器：

| 方法 | BT-LSTSVM | RandomForest | SVM-RBF |
|---|---:|---:|---:|
| MPE | 37.9% | 63.6% | 60.6% |
| RCMFuDE | 50.0% | 56.1% | 31.8% |
| IMFRDE | 50.0% | 62.1% | 31.8% |
| MDE | 53.0% | 53.0% | 24.2% |
| TSMEFuDE | 53.0% | 66.7% | 33.3% |
| RCMDE | 69.7% | 63.6% | 22.7% |
| MFuDE | 53.0% | 53.0% | 28.8% |
| MFE | **71.2%** | 66.7% | 33.3% |

**换分类器救不回来**：8 种熵 × 3 个分类器的最好成绩只有 71.2%。
所以限制在表示，不在分类器。

### 对照 3：仓库里已有的非熵特征族

`FUSION_TIME_SPEC`（时域 + 频谱统计）+ ExtraTrees，同一份数据、同一个切分：

| 配置 | validation | test |
|---|---:|---:|
| TSMEFuDE + BT-LSTSVM（anchor） | 83.3% | 53.0% |
| FUSION_TIME_SPEC + ExtraTrees | 97.2% | **77.3%** |

### 为什么熵特征会输

熵方法在分档前都会做**幅值归一化**（最小最大或 NCDF），所以它们描述的是
“变化模式有多不规则”，**对整体能量/幅值不敏感**。而这个齿轮数据集的类别差异
主要就是能量与冲击性：

```text
HEALTHY  rms 3.08   kurtosis 3.1
CRACK    rms 5.72   kurtosis 3.1
WEAR     rms 22.36  kurtosis 17.0
MISSING  rms 3.47   kurtosis 3.4
SPALLING rms 6.91   kurtosis 3.3
```

熵特征把这部分信息归一化掉了，剩下的模式差异不足以区分 5 类。
这不是数据的问题，是**表示和任务不匹配**。

### 另一个限制：有效样本只有 37 个文件

同一个文件内 6 个窗口的特征余弦相似度是 **0.9977**，文件之间是 0.9807——
同一文件的窗口几乎互为副本。所以“222 条记录”的独立信息量接近 **37 个文件**，
训练集实际只有 20 个文件。这也解释了为什么 validation 在不同切分间从 36% 波动到 97%。

## 8. 结论与局限

- 第三个数据集接入完成，且只用了适配器 + 配置，核心零改动；
- **数据集本身可分**（简单特征 95.5%），低分来自表示不匹配：
  熵特征对幅值/能量不敏感，而这类差异正是齿轮故障的主要信号；
- 通道 0 是任意选的；ch7 在同一协议下高 11 个点，通道选择本身就是一个未开发的方向；
- 有效样本只有 37 个文件（文件内窗口近乎重复），所以方差大是结构性的；
- 载荷与类别有部分混杂，已量化：不是主因，但必须写；
- 该数据集**适合**用来暴露“候选池只有熵方法”这个结构性缺陷。

对 Agent 的直接含义：候选池目前全是熵方法，在这个数据集上结构上就摸不到 77–95%。
把 `FUSION_*`（时域/频谱/包络）加进候选池，比继续调熵的选择策略更有价值。


## 9. 复现

```powershell
python scripts/prepare_gear_5class.py

# anchor
python scripts/run_supervised_experiment.py `
  --config configs/gear_5state_tsmefude_btlssvm.json `
  --data-root data/paper/gear_5state `
  --output results/gear-5state-tsmefude-btlssvm-20261105 `
  --model-id gear-5state-tsmefude-btlssvm

# Agent（8 个 unit 全付）
python scripts/run_agent_v1.py `
  --spec configs/gear_5state_tsmefude_btlssvm.json `
  --data-root data/paper/gear_5state `
  --output results/agent-v1_2-gear-20261105 `
  --units 8 --seeds 20261104,1,2,3,4 --validation-repeats 3
```

非熵对照（FUSION_TIME_SPEC，配置见 `tmp/gear_fusion.json`）：

```powershell
python scripts/run_supervised_experiment.py `
  --config tmp/gear_fusion.json `
  --data-root data/paper/gear_5state `
  --output results/gear-5state-fusion-timespec-20261105 `
  --model-id gear-5state-fusion-timespec
```

诊断脚本（本轮分析用，未纳入测试）：

```text
tmp/gear_channel_scan.py    逐通道简单时频特征的可分性
tmp/gear_entropy_study.py   8 种熵 × 3 个分类器在同一 train/test 上的上限
```

