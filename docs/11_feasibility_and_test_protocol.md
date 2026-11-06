# L2 可行性过滤 + 可重复的 test 评估

日期：2026-11-05
状态：两项都完成；其中“可重复 test”改变了我们对 triaxial 的结论

## 1. L2：可行性过滤

### 规则从哪来

不是拍脑袋，是从 `msentropy` 的真实约束推出来的：

| 约束 | 出处 |
|---|---|
| 信号长度不能被 scale 整除成 0 份（`L // scale >= 1`） | `core/decomposition.py::coarse_grain` 直接抛错 |
| 粗粒化后的序列至少要能取出一个 `m` 维、延迟 `tau` 的模式：`L // scale >= (m-1)*tau + 1` | `dis_en_ncdf_cumulative` 的 `W = N - (m-1)*tau`，`W == 0` 时得到 0/0 |

两条合起来就是：

```text
window_size >= scale x ((m-1) * tau + 1)
```

其中 `scale` 在大多数方法里叫 `scale`，在 `TSMEFuDE` 里叫 `kmax`。

### 实现

`src/entropy_agent/agent/feasibility.py`：

```python
required_window(method, parameters) -> int      # scale * ((m-1)*tau + 1)
check(method, parameters, window_size) -> Feasibility(feasible, required_window, reason)
```

`run_agent()` 在**做任何预算检查之前**先过滤候选池，把不可行的候选记进
`budget.skipped_candidates`（含所需窗长与原因），报告里单独一节列出。
如果全部候选都不可行，直接报错，不会跑到一半才失败。

### 结果

| 数据集 | 窗长 | 候选数 | 被过滤 |
|---|---:|---:|---:|
| aero | 2000 | 48 | 0 |
| triaxial | 1892 | 48 | 0 |

**默认参数下没有任何候选被过滤**——因为 8 个方法的要求最高也只有 100 点
（`MFE`：20 × ((2-1)×1+1) = 40；`MPE`：20 × 3 = 60；`MDE` 改 `m=3,tau=2` 也只有 100）。

所以这一项是**守卫**，不是优化：它防的是“把 350 秒的提取花在一个必然报错的方法上”，
以及短窗口数据集（<100 点）接入时静默失败。单测里用一个 60 倍 scale 的配置验证了
过滤链路（`MFuDE` 需要 300 点、窗长 200 → 被剔除，报告里可见）。

## 2. 可重复的 test 评估

### 问题

之前 triaxial 只有一个 test 切分：84 条记录，**±1 条 = 1.2 个百分点**。
而 agent 冠军的 validation 是 88.1%，test 是 77.38%——我据此在 `docs/10` 里
写了“selection bias 有 10.7 个点”。那个结论只基于一个切分。

### 做法

新增 `--test-mode per-seed`：**每个 seed 是一次完整独立实验**——
在该 seed 的 validation 上选冠军，在该 seed 的 test 上评一次，然后报 mean ± std。
同一个分区不会被评两次，所以仍然是“一次实验一个 test”。

另加 **bootstrap 区间**（`single` 模式默认输出）：
对固定的那份预测做 2000 次重采样，给出 accuracy / macro-F1 的 95% 分位区间。

### Triaxial 结果（5 个 seed，每 seed 独立选冠军并 test 一次）

| seed | 冠军 | validation | test |
|---:|---|---:|---:|
| 20261104 | IMFRDE + OBT-LSTSVM | 0.8810 | 0.7738 |
| 1 | IMFRDE + OBT-LSTSVM | 0.8333 | 0.8571 |
| 2 | IMFRDE + LSTSVM | 0.7738 | 0.8333 |
| 3 | IMFRDE + LSTSVM | 0.8095 | 0.8214 |
| 4 | IMFRDE + LSTSVM | 0.9048 | 0.8452 |

**test = 0.8262 ± 0.0288**（macro-F1 0.8222 ± 0.0288）。

### 这改变了两个结论

1. **原来那句“test 只有 77.38%”是单切分的下沿。**
   5 个独立切分下，triaxial 的 test 是 **82.6% ± 2.9%**（77.4–85.7）。
   之前把它当“这个数据集的上限 77%”是错的。
2. **`docs/10` 里的 “validation 88.1 vs test 77.4 = 10.7 点 selection bias” 被高估了。**
   5 个 seed 平均：validation ≈ 0.8405，test ≈ 0.8262，差距约 **1.4 个点**。
   selection bias 仍然存在（挑 30 个候选的最优者必然抬高 validation），
   但它的大小是 1–2 个点，不是 10 个点；那个 10.7 是单次切分的波动被当成了偏差。

这也是为什么“可重复 test”必须做：**单次 test 的 1 条记录就能改变结论方向。**

## 3. 现在能说的话

- triaxial 7 状态：**test accuracy 82.6% ± 2.9%**（5 个独立切分，每折独立选型）；
- aero 5 状态：**test accuracy 98.0% ± 1.6%**（5 个独立切分，见 §4）；
- L2 过滤在现有两个数据集上不触发，是守卫不是优化；
- **不能**再说 triaxial 的 ceiling 是 77.4%，也不能再说 selection bias 有 10 个点；
- 也不能再说 aero 的 test 是 100%——那是单切分的上沿。

## 4. Aero 的 per-seed 结果

只跑 agent 臂（其余三臂的对比已经在 `docs/10` 里做过），5 个 seed 各自独立选型 + 独立 test：

| seed | 冠军 | validation | test |
|---:|---|---:|---:|
| 20261104 | RCMFuDE + LSTSVM | 1.0000 | 0.9833 |
| 1 | RCMFuDE + SVM-RBF | 1.0000 | 0.9833 |
| 2 | TSMEFuDE + SVM-RBF | 1.0000 | 1.0000 |
| 3 | RCMFuDE + LSTSVM | 1.0000 | 0.9833 |
| 4 | TSMEFuDE + LSTSVM | 1.0000 | 0.9500 |

**test = 0.9800 ± 0.0163**。

和 triaxial 一样：单切分报 100%（seed 2）或 95%（seed 4）都可能，
5 折平均 98.0% 才是这个数据集上该写的数字。validation 全 1.0000 而 test 有高有低，
说明 aero 的 60 条 validation 已经饱和，分辨不出 1–2 条记录的差别。

## 5. 复现

```powershell
# aero：5 个 seed，每个独立选型 + 独立 test（只跑 agent 臂）
python scripts/run_agent_v1.py 
  --spec configs/aero_engine_5state_tsmefude_btlssvm.json 
  --data-root data/paper/aero_engine_5state 
  --output results/agent-v1_2-aero-perseed-20261105 
  --cache-root results/agent-v1_1-cache --units 8 
  --seeds 20261104,1,2,3,4 --validation-repeats 3 --test-mode per-seed --arms agent

# triaxial：5 个 seed，每个独立选型 + 独立 test
python scripts/run_agent_v1.py `
  --spec configs/triaxial_7class_tsmefude_btlssvm.json `
  --data-root data/paper/triaxial_7class `
  --output results/agent-v1_2-triaxial-perseed-20261105 `
  --cache-root results/agent-v1_2-triaxial-20261105/feature_cache `
  --units 5 --seeds 20261104,1,2,3,4 --validation-repeats 3 --test-mode per-seed
```

产物里 `report.md` 的 §1b 是可行性过滤，§5 是 per-seed test 表与均值；
`single` 模式则在 §5 额外打印 bootstrap 区间。

测试总数：**270 passed**（262 + 8 个新增测试）。
