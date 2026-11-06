# HANDOFF — EntropyLab Agent v2

最后更新：2026-11-04  
用途：跨对话交接。新对话先读本文件，再读 `docs/00`–`docs/06`。

语言约定：**文档（`docs/`、本文件、交接说明）用中文**；代码、配置、字段名、指标键、
数据集标签保持英文，方便复现、检索和不破坏已有脚本。

---

## 0. 一句话现状

三个数据集都已接入并跑过：Aero-engine 5 类（轴承）、Triaxial 7 类（轴承）、
**人字齿轮 5 类（齿轮箱，`docs/12`）**。第三个只新增了 prepare 脚本 + config，核心零改动。
Agent v1.1 / v1.2、L2 可行性过滤、可重复 test 协议都已完成；
v1.3 把 `FUSION_*` 接进候选池（`docs/13`，池 48→78、unit 8→13）；
**v1.4 把预算从“unit 个数”改成“预估秒数”**（`docs/14`）。

**现在的数字（5 个独立切分，每折独立选型 + 独立 test）**：

| 数据集 | 只有熵 | v1.3 全付（487–346 s） | v1.4 60 s 预算 |
|---|---:|---:|---:|
| aero 5 状态 | 98.0% ± 1.6% | 98.3% ± 1.8% | **99.3% ± 0.8%** |
| triaxial 7 状态 | 82.6% ± 2.9% | 93.6% ± 4.4% | **93.6% ± 4.4%** |
| 人字齿轮 5 状态 | 52.1% ± 12.7% | 68.2% ± 17.6% | **68.2% ± 17.6%** |

**v1.4 triaxial 是本项目第一个清晰的 Agent 胜利**：60 秒预算下
agent 花 37.6 秒买到 10 个 unit → validation 0.9500（= 穷举）；
按论文顺序的 fixed 花 59.9 秒只买到 3 个 unit → 0.8326，低 11.7 个点
（`RCMFuDE` 一个人吃掉 59.3 秒）。

anchor 文档里的单切分数字（aero 98.33%、triaxial 77.38%、gear 53.03%）仍然有效，
但**它们只是某一次切分**；5 折均值才是该写的数字。

OOD 留一类验证是负结果：未知类误接受率 53%（见 `docs/08`）。
**CLI 与诊断入口已打通**（`docs/15`）：`entropylab agent run / train / models / diagnose /
diagnose-batch / continual / inspect`，真实冠军已注册，5 条真实信号 5/5 诊断正确。
**持续学习机制已实现，但没有正向证据**（`docs/16`）：新增第四个数据集 **VBL-VA001**
（20 kHz、4 类、739 个 unit）。5 个独立划分下 continual 0.9830 ± 0.0074 vs
frozen 0.9800 ± 0.0069，**只赢 1/5、McNemar p = 0.25、5/5 落后 joint 约 0.9 点**。
根因：validation 在第 0 批就饱和到 0.98–1.00，`+0.005` 的 margin 判据无法触发，
4/5 个划分里没有任何更新被接受。**下一步必须先修判据，修好之前不得宣称正向。**

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
  agent/                      v1.4 选型：8 熵 + 5 融合族、逐记录缓存、秒预算、多 seed
  cli/                        统一入口：agent run / train / models / diagnose / continual
（新增数据集只用适配器脚本：VBL = prepare_vbl_4class.py，Paderborn = prepare_paderborn_3class.py）
  continual/                  持续学习：批次计划、champion/challenger、三臂对比
tests/                        295 passed（231 旧 + 64 Agent/评估协议/CLI/持续学习）
```

### 还没迁移/还没做

- 可视化（特征随尺度曲线、混淆矩阵、预算-性能曲线；全仓库 0 行 matplotlib）；
- 数据集 catalog（现在 spec 仍要手传路径，不能按元数据选择）；
- Agent L3：跨数据集元学习（需要 ≥5 个数据集，现有 3 个）；
- 持续学习的数据前提：现有三个数据集都回答不了“CL 有没有用”，需要新数据集（`docs/16` §6）；
- `experiments/`：**不做**（用户 2026-11-05 明确放弃论文全表复现）。

### 测试

```powershell
cd "E:\Research\EntropyLab Agent\renovation"
$env:PYTHONPATH="$PWD\src"
python -m pytest tests -q --basetemp="$PWD\.pytest-tmp"
```

当前结果：**295 passed**。

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

### 人字齿轮 5 状态（第三个数据集，`docs/12`）

数据：`7数据源/GearDatasets/GearDatasets/人字齿轮/{健康,裂纹,磨损,断齿,剥落}`，
37 个 CSV、15360 Hz、每文件 460800 点、10 列（前 9 列振动、第 10 列转速）。
协议：全部文件、每文件前 6 个 2000 点窗口、**第 1 个振动列**（无文档说明通道位置）。

| 模型 | validation | test（单切分） | test（5 个独立切分） |
|---|---:|---:|---:|
| anchor：TSMEFuDE + BT-LSTSVM | 83.33% | 53.03% | — |
| Agent v1.2（8 unit 全付） | 86.11% | 68.18% | **52.12% ± 12.66%** |

关键点：

- 单切分的 68.18% 是“在所有 seed 里挑 validation 最高的那个”，而它恰好 test 最容易；
  5 折后是 52.1%，与 anchor 的 53.0% 基本一致——**Agent 在这里没有真实提升**；
- **低分不是数据集的问题，是表示不匹配**（2026-11-05 三组对照）：
  - 同一文件组切分、简单时频特征：ch0 81.8%、ch7 92.4%、九通道合并 **95.5%**；
  - 8 种熵 × 3 个分类器的最好成绩只有 **71.2%**（MFE + BT）——换分类器救不回来；
  - `FUSION_TIME_SPEC + ExtraTrees`：validation 97.2%、test **77.3%**；
  - 原因：熵方法分档前做幅值归一化，对“能量/冲击性”这类齿轮故障主信号不敏感
    （rms：健康 3.1、磨损 22.4；kurtosis：3.1 对 17.0）；
- 通道 0 是任意选的，ch7 在同一协议下高 11 个点；
- 同一文件内 6 个窗口的特征余弦相似度 0.9977（文件间 0.9807）→ 有效样本≈37 个文件，
  训练只有 20 个文件，这是 validation 从 36% 波动到 97% 的结构性原因；
- 载荷与类别部分混杂，已量化：不是主因
  （训练集见过的 (类别,载荷) 组合 53.3%，没见过的 50.0%）；
- 接入只新增 `scripts/prepare_gear_5class.py` + `configs/gear_5state_tsmefude_btlssvm.json`，
  核心代码零改动。

**这个数据集暴露的结构性缺陷**：Agent 候选池里只有 8 种熵，没有时域/频域/包络特征，
所以它在这个数据集上**结构上**摸不到 77–95%。已在 v1.3 修复（见下）。

### v1.3：把 `FUSION_*` 接进候选池（`docs/13`）

- 候选池 **48 → 78**，unit **8 → 13**（8 熵 + 5 融合族：TIME / TIME_SPEC /
  TIME_ENVELOPE / TIME_ENVELOPE_ENTROPY / ALL）；
- `METHOD_FAMILIES` / `V1_AUTO_METHODS` 保持不变（它们必须等于熵工具箱注册表），
  融合族放在独立的 `FUSION_FAMILIES`，Agent 用并集 `FEATURE_FAMILIES`；
- 融合特征本来就在 `feature_pipeline.py` 里，数值代码零改动；
- 成本：5 ms/窗口 vs TSMEFuDE 500 ms/窗口，三个数据集上融合提取只花 5–9 秒。

| 数据集 | 只有熵 | 加融合族 | 每 seed 选中的家族 |
|---|---:|---:|---|
| aero 5 状态 | 98.00% ± 1.63% | 98.33% ± 1.83% | 熵（4/5） |
| triaxial 7 状态 | 82.62% ± 2.88% | **93.57% ± 4.43%** | 融合（5/5） |
| 人字齿轮 5 状态 | 52.12% ± 12.66% | **68.18% ± 17.59%** | 融合（5/5） |

这是“根据数据类型选方法”第一次真正发生。局限：gear 方差仍大（两个 seed 只有 45–48%）、
triaxial 的 93.6% 不能当成“超过论文”（协议不同）。

### v1.4：预算按秒计费（`docs/14`）—— 第一个清晰的 Agent 胜利

- `agent/cost.py`：每窗口先验价格 + 观测后指数平滑 + `from_memory()` 从历史实验读实测成本；
- `--budget-seconds`：unit 按预估秒数计费，买不起就跳过；输出 `cost_model.json` 账本；
- 计费口径是**预估秒数**（不是墙钟），这样冷/热缓存下两条臂才可比。

60 秒预算（全部 unit 需要 257–487 秒）：

| 数据集 | 臂 | validation | 买下 unit | 花费 | test |
|---|---|---:|---:|---:|---:|
| triaxial | **agent** | **0.9500** | 10 | **37.6 s** | **93.57%** |
| triaxial | fixed | 0.8326 | 3 | 59.9 s | — |
| triaxial | random | 0.8957 | 10 | 37.6 s | — |
| triaxial | exhaustive | 0.9500 | 13 | 487 s | — |
| aero | agent | 0.9978 | 10 | 26.9 s | 99.33% |
| gear | agent | 0.7368 | 11 | 51.0 s | 68.18% |

**机制**：固定顺序的第二个方法 `RCMFuDE` 预估 59.3 秒，几乎等于全部预算，
买下它之后只剩 0.6 秒、后面 11 个方法全部跳过；agent 先买便宜的 10 个 unit，
37.6 秒就追平穷举。aero/gear 上三条臂打平（分别饱和 / 噪声大）。

### CLI 与诊断入口（`docs/15`）

统一入口 `entropylab`（`pyproject.toml` 的 `[project.scripts]`，也可 `python -m entropy_agent.cli`）：

```text
agent run / train / models list|register|show|delete / diagnose / diagnose-batch / inspect
```

- 训练核心抽到 `training/experiment.py::run_supervised_experiment()`，
  `scripts/run_supervised_experiment.py` 改成薄封装，两条入口不会再漂移；
- 诊断有两条路径：`--artifact` 直接加载，或 `--registry + --profile` 走
  `ModelRegistry.as_diagnosis_registry()` → `DiagnosisRegistry.match()`；
- **元数据必须显式**：采样率不符 → `INCOMPATIBLE`；缺 `signal_type` → `INCOMPLETE`
  并列出缺项。绝不从信号推断；
- 端到端演示：注册两个 v1.4 真实冠军，5 条真实 aero 信号 5/5 诊断正确；
- 诚实限制：注册表只保证元数据契约，**不保证信号属于该设备**（那属于 OOD，而 OOD 不可靠）。


### 持续学习 v1（docs/16）—— 机制可用，数据不支持结论

- continual/：BatchPlan（批次/validation/holdout 互斥，组级）+ 
un_continual()
  （逐批重训、champion/challenger、margin 0.005、三臂 holdout）；
- 计划生成器 scripts/prepare_continual_plan.py：chronological 与 y-load 两种策略；
- CLI：entropylab continual run --spec --data-root --plan --output；
- 测试：6 个（重复组/漏组/holdout 泄漏/只增不减/promote 规则/计划往返）。

| 运行 | frozen | continual | joint | 说明 |
|---|---:|---:|---:|---|
| aero（3 批×60 条） | 1.0000 | 1.0000 | 1.0000 | 饱和，无信息量 |
| gear A（按类 validation） | 0.6333 | 0.6333 | **0.7667** | 三次更新全被拒（旧域掉了 3 点） |
| gear B（按载荷分层 validation） | 0.8000 | 0.8000 | 0.8000 | validation 0.11–0.15 与 holdout 0.80 分布不一致 |
| VBL-VA001（单次划分） | 0.9017 | 0.9188 | 0.9188 | 看起来 +1.7 点，但只是单次端点 |
| **VBL-VA001（5 个独立划分）** | 0.9800 ± 0.0069 | 0.9830 ± 0.0074 | **0.9920 ± 0.0045** | **只赢 1/5，p=0.25，全程落后 joint** |

两个可复现的失败模式：

1. **只用初始域做 validation**，持续学习退化为“永不更新”（gear A）；
2. **validation 与 holdout 不同分布**，选模型和报结果用的不是一把尺子（gear B，
   validation 覆盖三载荷层、holdout 只有 5 组落在一个层）。

结论：机制成立、测试齐全，但**持续学习没有可证实的收益**。
VBL 的 5 划分结果显示 +0.3 点（1/5 胜出、p=0.25）且落后 joint 0.9 点；
根因是 validation 饱和导致 margin 判据 4/5 个划分里从未触发（docs/16 §8）。
修判据之前不得对外说正向。


### Paderborn：第三个能撑住实验的数据集（docs/17）

- 数据：Paderborn-source-verification-20261102，32 个轴承、4 个工况、每工况 20 次测量；
  信号 Y.vibration_1（64 kHz）；
- 标签来自 factsheet 的 Component 字段，**不是** KA/KI 前缀（KA04/KA15 是寿命试验退化，
  KI08 是电刻损伤）；
- 原型 9 轴承（HEALTHY/OR/IR 各 3），损伤全部选“寿命试验”，避免“制造方式”成为捷径；
- 适配器 scripts/prepare_paderborn_3class.py，计划 prepare_paderborn_continual_plan.py。

可行性探针：同工况内 4 轴承分类 = 1.0000；跨工况 0.87–1.00（N15 内部）、0.29–0.35（N09）。
**历史 24–40% 是跨工况混训导致的，不是数据问题。**

基线（5 个划分，FUSION_TIME + ExtraTrees）：

| 臂 | accuracy | macro-F1 |
|---|---:|---:|
| frozen | 0.8843 ± 0.0135 | 0.8828 |
| continual | 0.9028 ± 0.0354 | 0.9015 |
| joint | **0.9593 ± 0.0069** | 0.9592 |

- 逐划分配对 continual − frozen = **+0.0185 ± 0.0414，只赢 1/5**；
- 唯一赢的那次是 seed 4：round 2 的 challenger 摸到 1.0000 → promote，holdout 直接 +9.3 点，
  并与 joint 持平（0.9630 = 0.9630）；
- 其余 4 个划分的 round-0 验证集已是 0.96–0.98，**判据无法触发**；
- 按工况：headroom 全在 N15_M07_F04（0.75 → 0.97），而补上它会伤 N15_M07_F10（−2.8 点）
  → guardrail 有真实用武之地。

**统计陷阱**：池化 bootstrap（+0.0185 [0.0111, 0.0269]）与 McNemar（p=1.1e-05）不可信，
因为 5 个划分是同一批测量的重复分配；只能报逐划分配对。


### 作废结果，不要再引用

- Aero-engine 二分类 69.7%：旧配置，不是论文 5 类协议；
- Paderborn 24%–40%：cross-bearing stress test，不是论文主线；
- MDE 不是论文主方法，不要围绕它排查；
- Triaxial 97.46%：见上，论文正文数字与作者存档不一致，未复现；
- “OOD 能识别未知信号”：见上，aero 留一类 FAR 53%，不成立；
- 齿轮 68.18%：单切分挑了最容易的 seed，5 折后只有 52.1%；
- “齿轮数据集本身不可分”：错的，简单特征 95.5%，是熵表示不匹配。

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
- **triaxial 冠军 validation 88.1% 但单切分 test 77.38%**：当时判为 selection bias；
  5 折重跑后修正为 val ≈ 84.1%、test ≈ 82.6%，真实差距约 1.4 个点（见 `docs/11`）。

### 第五步：L2 可行性过滤 + 可重复 test 协议 —— 已完成

- `agent/feasibility.py`：`window >= scale x ((m-1)*tau+1)`，在预算检查前过滤候选池，
  跳过的候选写进 `budget.skipped_candidates` 与报告 §1b（现有两个数据集 0 跳过，是守卫）；
- `--test-mode per-seed`：每个 seed 独立选型 + 独立 test，报 mean ± std；
- `single` 模式加 bootstrap 区间（2000 次重采样，95%）；
- **修正后的数字**：aero test 98.0% ± 1.6%，triaxial test 82.6% ± 2.9%；
- 结论：单切分的 ±1 条记录会改变结论方向，之前 77.38% / 100% 都只是单次切分的端点。

### 第六步：持续学习

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
- 特征缓存已于 v1.1 改为逐记录；现在残留的是 unit 预算未按成本加权
  （融合 5 ms vs TSMEFuDE 500 ms，仍按个数计）；
- 持续学习 replay 仍待重写；
- git 已初始化并完成首个提交（`c97c332`），`data/`、`results/`、`tmp/` 均被忽略。

---

## 9. 新对话的第一步

```text
1. 读 HANDOFF.md
2. 读 docs/05（aero）、docs/06（triaxial）、docs/07（v1）、docs/08（OOD）、
   docs/09（v1.1）、docs/10（v1.2）、docs/11（L2 + test 协议）、docs/12（齿轮数据集）、
   docs/13（融合族）、docs/14（秒预算）、docs/15（CLI + 诊断入口）、docs/16（持续学习）、
   docs/17（Paderborn 数据集）
3. 用「完全权限」跑 295 tests（受限沙箱会挡住 pytest 临时目录）
4. 检查 anchor / agent / ood 的 results/ 是否还在（三个数据集都有）
5. 可视化 / 通道选择 / 找一个能支撑持续学习结论的新数据集
```

不要重新论证论文方法是否有效，也不要再重跑两个 anchor。  
三个数据集的 5 折 test（v1.4，60 秒预算）：aero 99.3%、triaxial 93.6%、gear 68.2%。  
OOD、Agent 贪心选型、单切分高估这些负结果写在 `docs/08`–`docs/12`，
不要把它们说成正向能力；triaxial 的 93.6% 也不能说成“超过论文”。

---

## 10. Agent 已知缺陷（下一轮输入）

1. **unit 选择仍靠先验**：付哪几个方法必须付了才知道好坏；预算不足时永远有方法无法评估，
   只有跨数据集历史（L3）才能给出先验。
2. **评估已修好，结论要按新数字写**：v1.4 aero 99.3% ± 0.8%、triaxial 93.6% ± 4.4%、
   gear 68.2% ± 17.6%（各 5 个独立切分、60 秒预算）。单切分的 100% / 77.38% / 68.18% 都是端点。
3. **成本模型是估计值**：先验来自本仓库 1892–2000 点窗口，换窗口长度或方法参数要重新校准；
   预算按预估秒数计费，预估偏差会直接变成“买贵/买便宜”。
4. **通道是写死的**：齿轮 9 个振动通道里 ch7 比 ch0 高 11 个点，通道还没进入候选维度。
5. **CLI 只保证元数据契约，不保证信号属于该设备**：把 aero 信号按 gear 的 profile 输入
   照样会给出 `SPALLING`；真正的“不属于任何已知类别”要靠 OOD，而 OOD 已实测不可靠
   （`docs/08`）。不要把 CLI 当安全门。
6. **没有可视化**：论文/简历需要的特征曲线、混淆矩阵图、预算-性能曲线都还没有（0 行 matplotlib）。
7. **aero 已饱和**：validation 五折全 1.0，继续在它上面调策略没有信息量；
   策略对比要用 triaxial（中等）和 gear（难，方差大）。
8. **持续学习没有可证实的收益**：VBL 5 个划分下 +0.3 点（1/5 胜出、p=0.25），落后 joint 0.9 点；
   根因是 validation 饱和使 margin 判据无法触发（docs/16 §8），必须先修判据。
