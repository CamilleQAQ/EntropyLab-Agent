# 迁移日志 — 干净核心的抽取

日期：2026-11-04
来源：`E:\Research\EntropyLab Agent\entropy-agent`（只读参考）
目标：`E:\Research\EntropyLab Agent\renovation`

## 1. 结果

新仓库已经存在一个干净核心，并通过了迁移过来的测试：

```text
Phase 2 契约测试补完后 231 passed（迁移收尾时为 226）
```

没有复制任何 HIT 代码、MATLAB 兼容层、旧 results 或旧报告。

## 2. 复制的模块

| 来源 | 目标 | 说明 |
|---|---|---|
| `src/msentropy/` | `src/msentropy/` | 8 种熵方法、公开注册表 |
| `classifiers/__init__.py` | 同路径 | 分类器导出 |
| `classifiers/bt_lstsvm.py` | 同路径 | 二分类 LSTSVM 参考实现 |
| `classifiers/lstsvm.py` | 同路径 | OvO 多分类 LSTSVM |
| `classifiers/lstsvm_artifact.py` | 同路径 | NPZ 编解码 |
| `classifiers/obt_lstsvm.py` | 同路径 | OBT-LSTSVM |
| `classifiers/obt_artifact.py` | 同路径 | OBT 编解码 |
| `classifiers/torch_bt_lstsvm.py` | 同路径 | PyTorch 版 BT-LSTSVM |
| `classifiers/torch_artifact.py` | 同路径 | torch 编解码 |
| `classifiers/sklearn_classifiers.py` | 同路径 | SVM-RBF / 随机森林 / Extra Trees |
| `classifiers/registry.py` | 同路径 | 惰性注册表 |
| `analysis/*.py` | 同路径 | 9 个文件 |
| `io/*.py` | 同路径 | 5 个文件 |
| `training/grouped.py` | 同路径 | 分组切分 |
| `training/pipeline.py` | 同路径 | 通用训练流程 |
| `diagnosis/*.py` | 同路径 | 6 个文件 |

## 3. 有意不复制的内容

| 来源 | 原因 |
|---|---|
| `src/entropy_agent/datasets/hit/` | HIT 专用栈；新仓库不含 HIT |
| `src/entropy_agent/compatibility/` | 放弃 MATLAB 一致性对齐 |
| `src/entropy_agent/agent/` | 旧的信号图不是新的 5 轮选型器 |
| `src/entropy_agent/experiments/` | 等新 Agent 循环定义之后再迁 |
| `src/entropy_agent/continual/` | replay/训练逻辑必须先重写 |
| `src/entropy_agent/cli/` | 等核心稳定后再暴露 |
| `src/entropy_agent/training/compatibility.py` | 遗留的 MATLAB 路径 |
| `experiments/hit/` | 旧的 HIT 实验 |
| `results/`、`mlruns/` | 已被污染 / 是生成物 |
| `docs/archive/` | 历史报告 |
| 旧的 MATLAB 文档 | 不属于新的验收范围 |

## 4. 测试迁移

复制过来的测试套件：

- `tests/classifiers/`（7 个文件）；
- `tests/entropy_toolbox/`（2 个文件）；
- `tests/analysis/`（4 个文件）；
- `tests/io/test_dataset.py`；
- `tests/training/test_grouped.py`；
- `tests/training/test_pipeline_e2e.py`；
- `tests/training/test_cross_dataset_isolation.py`；
- `tests/diagnosis/test_artifact_contract.py`；
- `tests/diagnosis/test_label_contract.py`；
- `tests/diagnosis/test_registry.py`；
- `tests/fixtures/{three_channels.csv,missing_cell.csv}`。

迁移时删掉的内容：

- `tests/io/test_fixed_split_mat.py` —— 依赖旧的 Paderborn 文档和旧 Agent；
- `tests/io/test_dataset.py` 里“内置 HIT/VBL/NEU 配置”的断言。

这些测试绑定的是旧的数据集栈，不是新的核心。

## 5. 架构守卫

新增 `tests/test_dataset_boundary.py`：

- 用 `ast` 解析核心模块的 import；
- 若核心 import 了 `entropy_agent.datasets.*` 或
  `entropy_agent.compatibility.*` 则失败；
- 断言核心树中不存在具体数据集 / 兼容层包。

这是一个依赖方向测试，不是关键词扫描。

## 6. 已知缺口

- `agent/` 的 5 轮选型尚未实现；
- `experiments/` 未迁移；
- `continual/` 的 replay/训练未迁移；
- `cli/` 未迁移；
- Paderborn 的 `DatasetSpec` 当时还没写（Phase 2 已补齐，见 `docs/04`）；
- 包还没有在全新环境里做过 editable 安装；
- Windows 上 pytest 需要可写的 `--basetemp`（DSH 受限沙箱下更要完全权限）。

## 7. 下一步（当时记录）

Phase 2：

1. 定义规范的窗口/步长契约；
2. 在应用边界加 Paderborn `DatasetSpec`；
3. 为“黄金路径”选定一种特征方法和一种分类器；
4. 补契约测试；
5. 在 `initial_train` 上做第一次真实训练，在 `initial_validation` 上评估，
   保持 `final_holdout` 关闭。

> 后续进展：Phase 2 的契约与 Paderborn 基线见 `docs/04`；论文主线的两个 anchor
> 见 `docs/05`（发动机）与 `docs/06`（三轴）。
