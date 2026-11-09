# 把 FUSION_* 接进 Agent 候选池（v1.3）

日期：2026-11-05
动机：在已有的 8 种熵特征之外，引入能够描述幅值、频谱和调制成分的融合特征。
结果：候选池从 48 个候选 / 8 个 unit 扩到 **78 个候选 / 13 个 unit**，
三个数据集上重跑，**triaxial +11 点、gear +16 点**，aero 不变（已饱和）。

## 1. 改了什么

| 文件 | 改动 |
|---|---|
| `analysis/methods.py` | 新增 `FUSION_FAMILIES` 与 `FEATURE_FAMILIES`；**`METHOD_FAMILIES` / `V1_AUTO_METHODS` 保持不变**（有测试断言它们等于熵工具箱注册表） |
| `agent/candidates.py` | `DEFAULT_METHODS` 追加 5 个融合族：`FUSION_TIME`、`FUSION_TIME_SPEC`、`FUSION_TIME_ENVELOPE`、`FUSION_TIME_ENVELOPE_ENTROPY`、`FUSION_ALL`；family 查 `FEATURE_FAMILIES` |
| `agent/feasibility.py` | 融合族没有 `scale/m/tau` 约束，改为“窗长 ≥ 16”的下限检查 |
| `tests/agent/test_fusion_pool.py` | 4 个新测试：池子包含融合族、unit 数为 13、可行性下限、Agent 真的能评一个融合 unit |

没有改任何数值代码：融合特征本来就在 `analysis/feature_pipeline.py` 里，
`extract_features()` 之前就支持 `FUSION_*`，只是 Agent 的池子没列它们。

## 2. 成本

| 特征族 | 单窗口耗时 | 维度 |
|---|---:|---:|
| FUSION_TIME | 3.0 ms | 90 |
| FUSION_TIME_SPEC | 5.1 ms | 204 |
| FUSION_TIME_ENVELOPE | 5.1 ms | 294 |
| FUSION_TIME_ENVELOPE_ENTROPY | 4.9 ms | 349 |
| FUSION_ALL | 8.2 ms | 523 |
| （对照）TSMEFuDE | ~500 ms | 20 |

融合族比最贵的熵方法便宜 **两个数量级**。三个数据集上，5 个融合 unit 的提取总耗时
只有 5–9 秒，熵特征全部命中已有缓存。

## 3. 结果：三个数据集，5 个 seed，每 seed 独立选型 + 独立 test

| 数据集 | 只有熵（`docs/09`–`docs/11`） | 加入融合族 | 变化 |
|---|---:|---:|---:|
| Aero-engine 5 状态 | 98.00% ± 1.63% | **98.33% ± 1.83%** | +0.3（已饱和） |
| Triaxial 7 状态 | 82.62% ± 2.88% | **93.57% ± 4.43%** | **+10.9** |
| 人字齿轮 5 状态 | 52.12% ± 12.66% | **68.18% ± 17.59%** | **+16.1** |

### 每个 seed 选的家族

**Triaxial（5/5 选融合）**

| seed | 冠军 | validation | test |
|---:|---|---:|---:|
| 20261104 | FUSION_TIME_SPEC + EXTRA-TREES | 0.9762 | 0.9881 |
| 1 | FUSION_TIME_SPEC + EXTRA-TREES | 0.9762 | 0.9762 |
| 2 | FUSION_TIME_SPEC + EXTRA-TREES | 0.9643 | 0.8690 |
| 3 | FUSION_TIME_SPEC + EXTRA-TREES | 0.9048 | 0.9405 |
| 4 | FUSION_TIME_SPEC + SVM-RBF | 0.9524 | 0.9048 |

**Gear（5/5 选融合，但家族分散）**

| seed | 冠军 | validation | test |
|---:|---|---:|---:|
| 20261104 | FUSION_TIME + EXTRA-TREES | 0.9167 | 0.7727 |
| 1 | FUSION_TIME_ENVELOPE_ENTROPY + EXTRA-TREES | 0.8333 | 0.8333 |
| 2 | FUSION_TIME_ENVELOPE + RANDOM-FOREST | 0.6389 | 0.8636 |
| 3 | FUSION_TIME_ENVELOPE + RANDOM-FOREST | 0.7778 | 0.4848 |
| 4 | FUSION_TIME_ENVELOPE_ENTROPY + RANDOM-FOREST | 0.8611 | 0.4545 |

**Aero（熵仍占优，5 个 seed 里 4 个选熵）**

| seed | 冠军 | validation | test |
|---:|---|---:|---:|
| 20261104 | FUSION_TIME + EXTRA-TREES | 1.0000 | 1.0000 |
| 1 | RCMFuDE + SVM-RBF | 1.0000 | 0.9833 |
| 2 | TSMEFuDE + SVM-RBF | 1.0000 | 1.0000 |
| 3 | RCMFuDE + LSTSVM | 1.0000 | 0.9833 |
| 4 | TSMEFuDE + LSTSVM | 1.0000 | 0.9500 |

这正好是“Agent 根据数据类型选方法”第一次真正发生的证据：

- 轴承上的**严重程度分级**（triaxial）与**齿轮故障**（gear）主要由能量/频谱差异驱动 → 融合族赢；
- 轴承上的**健康状态判别**（aero）主要由波形调制模式驱动 → 熵族赢。

## 4. 必须写清的局限

1. **gear 的测试方差仍然很大**（±17.6），单看均值会掩盖“两个 seed 只有 45–48%”。
   5 个 seed 的 test 分布是 83.3 / 86.4 / 48.5 / 45.5 / 77.3。
2. **triaxial 的 93.6% 不能当成“超过论文”**。论文该数据集的 97.46% 与作者存档不一致
   （见 `docs/06`），而且这里的 93.6% 是 5 个切分的均值、每折独立选型，协议不同。
   只能写“在本仓库协议下”。
3. **unit 预算还没按成本加权**：一个融合 unit 只要 5 ms，一个 TSMEFuDE unit 要 500 ms，
   但预算仍按“unit 个数”计。`--units 13` 时这个问题不明显（全付），
   收紧预算时会偏向便宜/昂贵的方法，需要改成按估算耗时计费（见 §5）。
4. **aero 仍然饱和**，它的 98.3% 不能用来区分策略好坏。

## 5. 下一步

| 方向 | 内容 |
|---|---|
| 成本加权预算 | unit 预算从“个数”改成“预估秒数”（用历史提取耗时估计），让融合族与熵族可比 |
| 通道选择 | 齿轮数据 9 个振动通道里 ch7 高 11 个点；把通道作为候选维度是下一个自然的扩展 |
| 第 4/5 个数据集 | L3 需要 ≥5 个（当前 3 个） |

## 6. 复现

```powershell
python scripts/run_agent_v1.py `
  --spec configs/triaxial_7class_tsmefude_btlssvm.json `
  --data-root data/paper/triaxial_7class `
  --output results/agent-v1_3-triaxial-20261105 `
  --cache-root results/agent-v1_2-triaxial-20261105/feature_cache `
  --units 13 --seeds 20261104,1,2,3,4 --validation-repeats 3 --test-mode per-seed
```

`units=13` 即全部方法单元（8 熵 + 5 融合）。测试总数：**274 passed**。
