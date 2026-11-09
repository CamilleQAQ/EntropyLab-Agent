# 统一 CLI 与诊断入口

日期：2026-11-05
状态：**已实现并端到端验证**（5 条真实信号，5/5 正确）

之前所有能力都散在 `scripts/` 里，而且 `ModelRegistry` / `DiagnosisRegistry` 写好了却
没有任何入口用它。这一轮把两者补齐，并把训练脚本的核心抽成共享函数，避免 CLI 与脚本
两份实现漂移。

## 1. 命令一览

```text
entropylab agent run       跑 Agent（units 或 seconds 预算，多 seed，四臂对照）
entropylab train           一次监督实验（validation + 只开一次的 test）
entropylab models list     列出注册的模型
entropylab models register 注册一个 artifact 目录
entropylab models show     查看某个注册模型的 profile / feature / classifier / test
entropylab models delete   注销（可选删文件）
entropylab diagnose        诊断一条信号文件
entropylab diagnose-batch  诊断一个目录下的所有匹配文件
entropylab inspect         查看信号文件的列/数组结构（不猜）
```

入口注册在 `pyproject.toml`：

```toml
[project.scripts]
entropylab = "entropy_agent.cli:main"
```

没装包时也能用：`python -m entropy_agent.cli ...`。

## 2. 诊断入口的两种来源

```powershell
# A. 直接指向一个 artifact 目录
entropylab diagnose --artifact results/.../champion_model --trusted `
  --signal sig.npy --sampling-rate 25000

# B. 走注册表 + 诊断契约（推荐）
entropylab diagnose --registry models/registry --trusted `
  --profile AERO_ENGINE_5STATE `
  --signal sig.npy --sampling-rate 25000 `
  --signal-type ACCELERATION --signal-role AERO_ENGINE_CHANNEL_2
```

B 路径会走 `ModelRegistry.as_diagnosis_registry()` → `DiagnosisRegistry.match()`，
**元数据必须显式给出**，不会从信号本身推断：

| 情况 | 结果 |
|---|---|
| 采样率不符（20000 vs 25000） | `INCOMPATIBLE`，退出码 2，打印不匹配原因 |
| 缺 `signal_type` | `INCOMPLETE`，列出还缺哪些元数据 |
| 元数据齐全且匹配 | 正常诊断 |

这是 AGENTS.md 里“profile 必须显式或从可验证元数据选择”的落地。

## 3. 端到端演示（真实 champion 模型）

注册两个 v1.4 冠军：

```powershell
entropylab models --registry models/registry --trusted register `
  --artifact results/agent-v1_4-aero-20261105/champion_model/seed-20261104
entropylab models --registry models/registry --trusted register `
  --artifact results/agent-v1_4-gear-20261105/champion_model/seed-1
```

```text
aero_...-agent-fusion-time-extra-trees-seed-20261104  AERO_ENGINE_5STATE  ACCELERATION/AERO_ENGINE_CHANNEL_2
gear_...-agent-fusion-time-extra-trees-seed-1         GEAR_5STATE         ACCELERATION/GEARBOX_VIBRATION_CHANNEL_1
```

单条诊断（aero 的 state3 第 0 条记录，导出成 npy）：

```text
signal       : tmp\demo_signals\state3_record0.npy
status       : DIAGNOSED
model        : aero_...-agent-fusion-time-extra-trees-seed-20261104
diagnosis    : state3 (class 2)
vote share   : 1.00
windows      : 1
```

批量诊断 5 条（state1–state5 各一条）：

```text
state1_record0.npy,DIAGNOSED,state1,0,1.0,1,
state2_record0.npy,DIAGNOSED,state2,1,1.0,1,
state3_record0.npy,DIAGNOSED,state3,2,1.0,1,
state4_record0.npy,DIAGNOSED,state4,3,1.0,1,
state5_record0.npy,DIAGNOSED,state5,4,1.0,1,
```

5/5 正确。注意这些信号来自**训练过的数据集**，所以是“入口能用”的验证，不是泛化证据。

## 4. 实现结构

| 文件 | 作用 |
|---|---|
| `src/entropy_agent/cli/__init__.py` | 全部子命令；每个 handler 只调用库函数 |
| `src/entropy_agent/cli/__main__.py` | `python -m entropy_agent.cli` |
| `src/entropy_agent/training/experiment.py`（新） | `run_supervised_experiment()`：训练+一次 test+预测+metrics 的唯一实现 |
| `scripts/run_supervised_experiment.py` | 改成薄封装，调用上面那个函数（行为与输出不变） |
| `tests/cli/test_cli.py` | 8 个测试：train / register / list / show / diagnose / batch / 契约拒绝 / inspect |

`tests/agent/conftest.py` 的 tiny 数据集 fixture 上移到了 `tests/conftest.py`，
CLI 测试与 Agent 测试共用。

## 5. 必须写清的局限

1. **注册表校验显式选择的模型及其输入元数据契约。** 实际设备来源需要由调用方确认，
   未知故障拒识需要独立验证。
2. `--trusted` 是必需的：scikit-learn joblib artifact 默认拒绝加载；
   CLI 会在没有该标志时报错（`error: ... trusted=True is required ...`）。
3. `diagnose-batch` 目前逐文件串行，没有并发；大目录会慢。
4. 没有 `datasets` 子命令：数据集目录（catalog）还没做，spec 仍要手传路径。

## 6. 下一步

- 可视化（特征随尺度曲线、混淆矩阵、预算-性能曲线）；
- [持续更新与模型发布判据](18_v2_decision_layer.md)；
- 数据集 catalog（把“手传 spec”变成“按元数据选择”）。

测试总数：**289 passed**。
