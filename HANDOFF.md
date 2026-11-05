# HANDOFF — EntropyLab Agent v2

最后更新：2026-11-04  
用途：跨对话交接。新对话先读本文件，再读 `docs/00`–`docs/06`。

语言约定：**文档（`docs/`、本文件、交接说明）用中文**；代码、配置、字段名、指标键、
数据集标签保持英文，方便复现、检索和不破坏已有脚本。

---

## 0. 一句话现状

两个 anchor 都已完成并冻结：Aero-engine 5 类 **98.33%** test，Triaxial 7 类
**77.38%** test（作者自存的同类结果是 83.28%，论文的 97.46% 对不上，见 `docs/06`）。
Agent v1.1（3 次切分 + 5 seed）与 v1.2（预算按特征提取计）都已完成。
v1.2 修好了记账：aero 8 个 unit 全付 → 四臂并列 0.9967（= oracle），test 100%；
triaxial 只付 5/8 个 unit → agent = exhaustive = 0.8457，随机选方法掉到 0.8273。
但 triaxial 上 **validation 88.1% 并没有换来更好的 test（仍是 77.38%）**，是 selection bias（见 `docs/10`）。
OOD 留一类验证是负结果：未知类误接受率 53%（见 `docs/08`）。
下一步：L2 可行性过滤 + 更大/可重复的 test 评估协议；L3 元学习要等数据集够（现有 2 个）。

---

## 1. 项目目标

这是一个故障诊断 Agent，不是“万能跨轴承模型”。

目标：

1. 对**同一台设备 / 同一试验台**的数据，Agent 根据数据类型选择熵方法 + 分类器；
2. 通过 **5 轮实验**，根据 validation 结果不断调整，选出最优组合；
3. 最终 test 只评估一次；
4. 之后支持新批次持续学习，让同一台设备的模型逐步变好；
5. 项目用于论文 + 简历/面试，必须可复现，不能编指标。

**不要**把目标理解成“学了 A 轴承去预测 B 轴承”。  
跨轴承泛化是另一个问题，只作为可选的 stress test。

---

## 2. 论文依据

### Coatings 2025

- 标题：*Time Shift Multiscale Ensemble Fuzzy Dispersion Entropy and Its Application in Bearing Fault Diagnosis*
- DOI: `10.3390/coatings15070779`
- 方法：**TSMEFuDE + BT-LSTSVM**
- 数据：
  - Triaxial bearing：7 类，60 样本/类，2000 点；
  - Aero-engine：5 类（2 健康 + 3 故障），60 样本/类，2000 点。

### ESWA 2025

- 标题：*A fault diagnosis approach using improved multiscale fluctuation Rényi dispersion entropy and optimized binary tree LSTSVM*
- DOI: `10.1016/j.eswa.2025.127820`
- 方法：**IMFRDE + optimized BT-LSTSVM**
- 数据线索：CWRU、Politecnico di Torino、wind turbine gearbox。

ESWA 的完整数据映射还没有做；先不要展开。

---

## 3. 仓库与路径

### 新仓库（开发主线）

```text
E:\Research\EntropyLab Agent\renovation
```

### 旧仓库（只读参考）

```text
E:\Research\EntropyLab Agent\entropy-agent
```

旧仓库不要继续开发、不要搬 HIT 专用栈、不要搬 MATLAB 兼容层、不要用旧 results/ 指标。

### 数据

```text
7数据源\Aero engine bearing fault diagnosis\dataset
7数据源\TriaxialBearingVibrationDatasetofInductionMotorunderVaryingLoad\Dataset
7数据源\GearDatasets
7数据源\WindTurbine...
```

新仓库自己的数据：

```text
renovation\data\paper\aero_engine_5state   # 5 个 state*.npy，已 prep
renovation\data\paderborn\raw              # 20.8GB，Git 忽略
```

`data/`、`results/`、`mlruns/` 都在 `.gitignore` 里。

---

## 4. 当前代码状态

### 已迁移

```text
src/msentropy/                 8 种熵
src/entropy_agent/
  analysis/                   熵后端、窗口、质量、融合特征
  classifiers/                6 种分类器、注册表、codec
  io/                         DatasetSpec、CSV/NPY/NPZ/MAT、窗口
  training/                   grouped split + generic training
  diagnosis/                  v4 artifact、runner、profile registry
  agent/                      v1.1 5 轮选型、逐记录缓存、多次切分、多 seed、SQLite 记忆
tests/                        262 passed（231 旧 + 31 Agent）
```

### 还没迁移/还没做

- `experiments/`；
- `continual/`；
- `cli/`；
- Agent L2：数据画像驱动的可行性过滤（window 长度 vs `scale/m/tau`）；
- Agent L3：跨数据集元学习（需要 ≥5 个数据集，现有 2 个）；
- 可重复的 test 评估协议（现在 test 只有 60/84 条，±1 条 = 1.2–1.7 个点）。

### 测试

```powershell
cd "E:\Research\EntropyLab Agent\renovation"
$env:PYTHONPATH="$PWD\src"
python -m pytest tests -q --basetemp="$PWD\.pytest-tmp"
```

当前结果：**262 passed**。

在 DSH 的受限沙箱里跑不通：pytest 要创建/删除临时目录，会被沙箱拒绝（表现为
大量 setup error + `sessionfinish` 的 `PermissionError`）。用「完全权限」跑即可。
`renovation` 与 `renovation\tmp` 的权限已在 2026-11-04 修过一次（回滚记录在
`_acl-repair-20261104-triaxial/`）。

---

## 5. 已验证结果（可以写进 resume）

### Aero-engine 5 类

协议：

```text
5 类：data1/2 健康，data3/4/5 故障
每类 60 个样本，每个 2000 点
channel = 2
TSMEFuDE: m=2, nc=5, tau=1, kmax=20
BT-LSTSVM: c1=1.0, c2=1.0, bandwidth=1.0, epsilon=1e-4
本地 split: 60% train / 20% validation / 20% test
```

结果：

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation | 95.00% | 95.00% |
| Test | **98.33%** | **98.33%** |

论文报告：97.46%。  
本地测试集比论文小（12/类 vs 24/类），所以是“同一水平”，不是“超过论文”。

文件：

```text
configs/aero_engine_5state_tsmefude_btlssvm.json
scripts/prepare_aero_engine_5state.py
scripts/run_supervised_experiment.py
docs/05_aero_engine_reproduction.md
results/aero-engine-5state-tsmefude-btlssvm-20261104/
```

### Triaxial 7 类（已冻结）

由作者的 `DataloaderOfTraixialBearing.m` 固定：300 W 下六个内圈缺陷
（0.7/0.9/1.1/1.3/1.5/1.7 mm）+ `healthy without pulley`，X 轴（CSV 第 2 列）。

```text
窗长 1892（作者代码 N=1892，不是正文的 2000；2000 点切不满 60/类）
每类 60 段、无重叠；feature 同 aero；BT-LSTSVM 同参数
本地 split: 60/20/20，group = 连续 6 段 → 36/12/12 每类
```

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation | 80.95% | 80.82% |
| Test | **77.38%** | **77.26%** |

健康 vs 故障 test 100%；错误全部是相邻缺陷尺寸互相混淆。
特征与作者保存的 MATLAB 特征一致（420 段里 419 段差 < 5e-3，中位数 3.3e-4）。

**不要引用 97.46% 作为这个数据集的结果**：作者自存的准确率是 83.28%，
`docs/PAPER_CONTEXT.md` 里三轴表与发动机表数值完全一样，应是抄写错误。

文件：

```text
scripts/prepare_triaxial_7class.py
scripts/verify_triaxial_features.py
configs/triaxial_7class_tsmefude_btlssvm.json
docs/06_triaxial_reproduction.md
results/triaxial-7class-tsmefude-btlssvm-20261104/
```

### Agent v1（Aero-engine 5 状态）

5 轮 × 3 = 15 个 trial，8 种熵 × 6 个分类器 = 48 个候选，同预算四条臂。

| 臂 | trial | 最佳 val Macro-F1 |
|---|---:|---:|
| agent | 15 | 0.9833 |
| random | 15 | 0.9833 |
| fixed | 15 | 1.0000（顺序碰巧） |
| exhaustive | 48 | 1.0000 |

冠军 `RCMDE + SVM-RBF`：validation 98.33% → **test 100.00%**（只开一次）。
冠军 artifact：`model_id = aero_engine_5state_tsmefude_btlssvm-agent-rcmde-svm-rbf`，
`display_name = Aero-engine 5-state — agent-selected RCMDE + SVM-RBF`
（初版这里抄了配置里的 `TSMEFuDE + BT-LSTSVM`，2026-11-04 已修，见 `docs/07` §2）。

关键成本事实：特征提取 477 s（8 次），分类器拟合 12 s（78 个 trial）；
穷举臂 48 个 trial 只多花 5.5 s，因为特征全部命中缓存。

**结论要如实写**：15/48 预算拿到 59/60 条 validation，与随机臂持平，
输给固定顺序和穷举上界（差 1 条记录）；不能说“Agent 优于随机搜索”。

文件：

```text
src/entropy_agent/agent/
scripts/run_agent_v1.py
docs/07_agent_v1.md
results/agent-v1-aero-20261104/
```

### Agent v1.1（Aero-engine 5 状态，5 seed）

逐记录缓存 + 每 trial 3 次切分 mean±std + 5 个 seed 的同预算对比。

| 臂 | 均值 Macro-F1 | 标准差 | 胜场（共 5） |
|---|---:|---:|---:|
| agent | 0.9922 | 0.0045 | 1 |
| random | 0.9944 | 0.0050 | 3 |
| fixed | **0.9967** | **0.0027** | 5 |
| exhaustive | 0.9967 | 0.0027 | 5 |

冠军 `RCMFuDE + SVM-RBF`：validation 1.0000 ± 0.0000，**唯一一次 test = 98.33%**。

关键成本事实：5 seed × 4 臂 = 465 个 trial 的**特征提取为 0 秒**（逐记录缓存复用），
拟合共 143.6 s。v1 冷启动时提 8 种熵要 477–564 s。

**结论仍然是负的**：agent < random < fixed = exhaustive，且 5 个 seed 一致。
原因是“方法只测一次就锁死”（RCMDE 0.9889 后 9 个 trial 全给它），
以及深度优先用光预算。见 `docs/09` §4、§7。

文件：

```text
src/entropy_agent/agent/
scripts/run_agent_v1.py
docs/09_agent_v1_1.md
results/agent-v1_1-aero-20261104/
results/agent-v1_1-cache/
```

### OOD 验证（aero，留一类）—— 负结果

5 折留一类，阈值只用各折 validation 校准；已知 240 条、未知 300 条。

| margin | 误拒率 FRR | 未知类误接受率 FAR |
|---:|---:|---:|
| 1.00 | 0.0% | **53.3%** |
| 1.05 | 0.0% | 57.0% |
| 1.20 | 0.0% | 64.3% |

已知类一侧完美（240/240 全对），但**未知类一半被硬分到某个已知类**。
原因：留出类的特征落在已知类的云团内部（state1/state5 的未知记录有 92%/83%
低于已知类的最大距离），只有 state3 能被干净拒掉。
真实诊断入口复核 40 条信号，判定 0 处不一致——所以不是实现 bug，是度量不可分。

**结论**：两个 anchor 保持 `ood.calibration = "DISABLED"` 是对的；
现在只能说“实现了拒识机制”，**不能说“支持未知信号拒识”**。

```text
scripts/run_ood_validation.py
docs/08_ood_validation.md
results/ood-aero-20261104/
```

### 作废结果，不要再引用

- Aero-engine 二分类 69.7%：旧配置，不是论文 5 类协议；
- Paderborn 24%–40%：cross-bearing stress test，不是论文主线；
- MDE 不是论文主方法，不要围绕它排查；
- Triaxial 97.46%：见上，论文正文数字与作者存档不一致，未复现；
- “OOD 能识别未知信号”：见上，aero 留一类 FAR 53%，不成立。

---

## 6. 下一步任务

### 第一步：Triaxial 7 类 anchor —— 已完成

见 `docs/06_triaxial_reproduction.md`。冻结，不再重跑。

### 第二步：Agent v1 / v1.1 —— 已完成

见 `docs/07_agent_v1.md`（v1）与 `docs/09_agent_v1_1.md`（v1.1）。
可运行、可审计、test 只开一次；但两版的选型效果都是负结果。

已实现的闭环：

```text
DataProfile
  -> 候选熵方法 + 分类器排序
  -> 5 轮实验
  -> 每轮只跑 validation
  -> 结果写入 SQLite experiment memory
  -> 第 5 轮结束选 validation 最优
  -> final test 只开一次
```

实现位置：

```text
src/entropy_agent/agent/
  profile.py      DataProfile
  candidates.py   8 种熵 × 6 个分类器 = 48
  cache.py        逐记录特征缓存（key 只含数据集契约）
  memory.py       SQLite runs / trials（含 seed、mean±std、reason）
  selector.py     确定性排序（成本感知 + 理由）
  loop.py         多 seed、多次切分、四臂、冠军、唯一一次 test、报告
```

v1.1 已解决、v1.2 已解决记账问题；剩下的见 §10。

先不要用 LLM 直接决定模型；LLM 只做候选排序/解释，所有执行走确定性 runner。

### 第三步：在 Aero-engine 5 类上验证 Agent —— 已完成（结论见第 4 步）

- Anchor：TSMEFuDE + BT-LSTSVM = 98.33% test；
- v1：15 个 trial，val 98.33%，test 100%（只开一次），与随机臂持平；
- v1.1：5 seed × 4 臂，agent 0.9922 < random 0.9944 < fixed = exhaustive 0.9967；
- 结论：aero 已饱和，策略差异小于单 trial 波动，**不能**声称 Agent 更好。

### 第四步：Agent v1.2 与 Triaxial 验证 —— 已完成

- v1.2 把预算单位从 trial 换成特征提取（unit = 一次提取，买下该方法全部分类器）；
- aero（8 unit 全付）：四臂并列 0.9967 = oracle，冠军 test 100%；
- triaxial（只付 5/8 unit）：agent = fixed = exhaustive = 0.8457，random 0.8273
  → “付哪几个方法”在 triaxial 上是真问题，随机会掉 1.8 个点；
- **triaxial 冠军 validation 88.1% 但 test 仍是 77.38%**：selection bias，
  不能拿 validation 的提升当成果；
- 下一步：L2 可行性过滤 + 可重复 test 评估协议；L3 元学习等更多数据集。

### 第五步：持续学习

等 Agent 与评估协议稳定后再做：

- 新批次接入；
- 固定历史 train，不允许重新随机划分；
- Champion/Challenger；
- 固定 holdout 比较；
- promote/reject/rollback。

---

## 7. 关键设计约束

1. **同一设备/同一试验台**，不是跨轴承。
2. 5 轮只看 validation；final test 只一次。
3. 特征缓存必须按 dataset manifest + method + params 做 key。
4. 所有指标必须能从逐记录 prediction 复算。
5. 不伪造指标；失败 run 保留。
6. 核心不能 import 具体数据集；数据集适配器只在 boundary。
7. 不要把 HIT 专用栈、MATLAB 兼容层、旧 results 带回来。
8. 不要先做持续学习，先做 5 轮选型。

---

## 8. 当前已知问题/待确认

- Triaxial 类别映射已确认（作者 `DataloaderOfTraixialBearing.m`）；窗长用 1892
  与论文正文 2000 不一致，已在 `docs/06` 说明；
- Triaxial 采样率写的是论文的 10 kHz，但 CSV 时间戳列推出来约 440 点/秒，未独立核实；
  该值只进 metadata，不参与熵计算；
- 论文是 60/40 split；本地 pipeline 需要 validation，用 60/20/20 近似；
- 作者自存的三轴准确率 83.28% 与论文表里的 97.46% 对不上，尚未向作者求证；
- ESWA 论文的 CWRU / Politecnico / Wind 数据尚未复现；
- Agent v1 选择器偏贪心，15 次里没试过 `LSTSVM`，见 §10；
- OOD 在 aero 上不可用（留一类 FAR 53%），且 schema 把 `margin` 限制在 ≥1，
  连收紧阈值都做不到，见 `docs/08`；
- 特征缓存按“记录集”做 key，换一次切分就要重提 8 次特征；多 seed 对比前要改成逐记录；
- 持续学习 replay 仍待重写；
- git 已初始化并完成首个提交（`c97c332`），`data/`、`results/`、`tmp/` 均被忽略。

---

## 9. 新对话的第一步

```text
1. 读 HANDOFF.md
2. 读 docs/05（aero）、docs/06（triaxial）、docs/07（v1）、docs/08（OOD）、
   docs/09（v1.1）、docs/10（v1.2）
3. 用「完全权限」跑 262 tests（受限沙箱会挡住 pytest 临时目录）
4. 检查 anchor / agent-v1_1 / agent-v1_2 / ood 的 results/ 是否还在
5. 做 L2（可行性过滤）+ 可重复 test 评估协议，再谈 L3 元学习
```

不要重新论证论文方法是否有效，也不要再重跑两个 anchor。  
Aero-engine 98.33% + Triaxial 77.38% 已经证明主链路可用；Agent v1.2 证明
“按特征提取记账”能用 5/8 个 unit 达到穷举水平。  
OOD、Agent 选型、selection bias 三个负结果已写进 `docs/08`–`docs/10`，
不要把它们说成正向能力。

---

## 10. Agent 已知缺陷（下一轮输入）

1. **unit 选择仍靠先验**：付哪几个方法必须付了才知道好坏；预算 5/8 时永远有 3 个
   方法无法评估，只有跨数据集历史（L3）才能给出先验。
2. **selection bias 已被实测**：triaxial 冠军 validation 88.1%，test 仍 77.38%；
   在 84 条 test 上 ±1 条 = 1.2 个点，任何小差异都不能下结论。
3. **没有可行性过滤**：某些方法参数与窗口长度不匹配时仍会占掉一个 unit（L2）。
4. **aero 已饱和**：0.9967 是天花板，继续在它上面调策略没有信息量。
