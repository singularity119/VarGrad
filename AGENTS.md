# Task

在 `/root/VarGrad` 原始 VarGrad 项目基础上，直接面向 NYUv2 做最小侵入式扩展，用来重新验证 VarGrad + FairGrad + PSMGD 相关实验。

当前第一阶段优先聚焦一个完整方法链：

- Gradient preprocessing: `identity`, `vargrad`
- Baseline solver: `fairgrad`
- Weight scheduler: `every_step`, `psmgd_periodic`, `psmgd_dynamic`

后续再按同一接口扩展：

- `uniform`
- `mgda`
- `cagrad`
- `nashmtl`

必须优先复用原项目的：

- 数据流
- trainer 入口
- 模型结构
- 数据集定义
- 日志
- delta_m / 评估逻辑
- stats 保存逻辑

不要推倒重写，不要直接照搬 `/root/Vargrad_PSMGD_modular` 的 `ComposableMTL` 作为新中心类。这个项目的目标是沿着师兄原始 VarGrad 项目的组织方式续写代码。

---

# Current Repository Notes

当前执行目标是新增 `/root/VarGrad/VarGrad-code/experiments/nyuv2/`，复用 `/root/Vargrad_PSMGD_modular/experiments/nyuv2` 的 dataset/model/trainer/run script 结构，但方法实现来自 `/root/VarGrad/VarGrad-code/methods/weight_methods_vargrad.py`。

NYUv2 数据集路径：

- `/root/autodl-tmp/dataset/nyuv2`

输出路径必须与 modular 项目隔离：

- `/root/autodl-tmp/exp_logs_save/vargrad_reimpl/nyuv2/save`
- `/root/autodl-tmp/exp_logs_save/vargrad_reimpl/nyuv2/log`

核心文件：

- `methods/weight_methods_vargrad.py`
- `experiments/utils.py`
- `experiments/nyuv2/data.py`
- `experiments/nyuv2/models.py`
- `experiments/nyuv2/utils.py`
- `experiments/nyuv2/trainer.py`
- `experiments/nyuv2/run_vargrad_fairgrad_psmgd.sh`
- `experiments/cityscapes/trainer_vargrad.py`
- `experiments/cityscapes/run_vargrad.sh`

已知需要优先修正的问题：

- `methods/__init__.py` 当前引用 `methods.weight_methods`，但仓库中实际文件是 `methods/weight_methods_vargrad.py`。
- `FairGrad` 当前内部硬编码 `beta=0.85`，后续应改为使用 trainer/CLI 传入的 VarGrad beta。
- 原项目中的 VarGrad 逻辑分散写在各 solver 类里，本项目扩展时应保持这种组织风格，但抽出少量共享 helper，避免重复和不一致。

---

# Required Pipeline

每个 step 必须严格按下面顺序执行：

1. 显式计算每个任务的原始共享梯度 `g_t^k`
2. 做梯度预处理，得到 solver/update 使用的 `U_t^k`
3. scheduler 判断当前 step 是否需要调用 solver
4. 如果调用 solver，用 baseline solver 根据 `U_t` 计算候选权重 `lambda_hat_t`
5. 用 scheduler 生成当前权重 `lambda_t`
6. 聚合梯度并更新参数

不能用总 loss 的混合梯度代替单任务梯度。

---

# Math

## Raw Gradient

对每个任务 k：

$$
g_t^k = \nabla_{\theta} L_t^k
$$

必须显式拿到单任务梯度。

## VarGrad Preprocessing

如果 `preprocessing=vargrad`：

$$
c_t^k = g_t^k + \gamma \frac{\beta}{1 - \beta}\left(g_t^k - g_{t-1}^k\right)
$$

并沿用师兄 VarGrad 的状态递推：

$$
U_t^k = \beta U_{t-1}^k + (1 - \beta)c_t^k
$$

默认 `gamma=1.0`。`last_grads` 和 `exp_avg` 初始为 `None`，第一次调用时按零张量初始化，使第一步输出满足 `U_1^k = g_1^k`。

如果 `preprocessing=identity`：

$$
U_t^k = g_t^k
$$

---

# Solver

solver 输入默认使用 `U_t^k`。

当前第一阶段只要求完整支持：

- `fairgrad`

FairGrad 必须沿用原始 FairGrad 项目的定义：

- 不要重写 FairGrad 数学公式
- 保留 least-squares 求解逻辑
- `alpha` 是 FairGrad alpha，和 `psmgd_alpha` 分开
- 默认实验中 FairGrad alpha 应显式传为 `2.0`

FairGrad 权重尺度要谨慎处理：

- 原始 FairGrad 的 `w_cpu` 不一定是和为 1 的 simplex 权重
- 为了对齐原实现，第一阶段不要擅自把 FairGrad 权重强制归一化
- 如果后续要测试归一化 FairGrad 权重，应作为单独消融开关，而不是默认行为

---

# Scheduler

必须支持下面三种 scheduler：

- `every_step`
- `psmgd_periodic`
- `psmgd_dynamic`

参数校验必须包含：

```python
if scheduler not in ["every_step", "psmgd_periodic", "psmgd_dynamic"]:
    raise ValueError(f"unknown scheduler {scheduler}.")
```

## every_step

每个 step 调用 solver：

$$
\lambda_t = \hat{\lambda}_t
$$

## psmgd_periodic

固定周期调用 solver。

当 `t % R == 0`：

$$
\lambda_t = \alpha_{\mathrm{psmgd}}\lambda_{\mathrm{prev}} + (1 - \alpha_{\mathrm{psmgd}})\hat{\lambda}_t
$$

当 `t % R != 0`：

$$
\lambda_t = \lambda_{t-1}
$$

非刷新 step 必须跳过 solver，复用上一组权重。

## psmgd_dynamic

动态判断是否调用 solver。第一步必须强制调用 solver，用来建立初始权重和 `U_last_refresh` anchor。

至少支持两个监控指标：

- `refresh_rel_fro`
- `step_rel_fro`

其中：

$$
\mathrm{refresh\_rel\_fro} =
\frac{\|U_t - U_{\mathrm{last\_refresh}}\|_F}
{\|U_{\mathrm{last\_refresh}}\|_F + \epsilon}
$$

$$
\mathrm{step\_rel\_fro} =
\frac{\|U_t - U_{t-1}\|_F}
{\|U_{t-1}\|_F + \epsilon}
$$

动态策略必须支持方向：

- `above`: score > threshold 时刷新
- `below`: score <= threshold 时刷新

非刷新 step 必须：

- 不调用 solver
- 复用上一组 `lambda_t`
- 保留上一组 `candidate_weights`
- 记录 `solver_called=False`
- 记录 `updated_weights=False`

刷新 step 必须：

- 调用 solver
- 更新 `candidate_weights`
- 更新 `lambda_t`
- 更新 `U_last_refresh`
- 记录 `solver_called=True`
- 记录 `updated_weights=True`

`refresh_rel_fro` 必须在更新 `U_last_refresh` 前计算。

---

# Update

共享参数更新使用：

$$
g_t^{\mathrm{agg}} = \sum_k \lambda_t^k U_t^k
$$

FairGrad 需要对齐原项目 `overwrite_grad` 行为：

$$
g_t^{\mathrm{agg}} = K \sum_k \lambda_t^k U_t^k
$$

其中 K 是任务数。

实现上可以先用未加权的 `sum(losses)` 保留 task-specific head 梯度，再覆盖 shared parameters 的梯度为 `g_t^{agg}`。不要让 shared 参数实际使用 `sum(losses)` 的混合梯度更新。

---

# Strict Separation

必须保持三层职责解耦。

## preprocessing

只负责：

- `g -> c`
- `c -> U`

## solver

只负责：

- 根据当前 `U_t` 生成 `lambda_hat_t`

## scheduler

只负责：

- 根据 scheduler 类型、历史权重、历史 anchor、step 和阈值决定是否调用 solver
- 根据 `lambda_hat_t` 生成 `lambda_t`

不要把：

- VarGrad 写进 FairGrad solver 数学公式
- PSMGD 写进 FairGrad solver 数学公式
- solver 写进 VarGrad
- dynamic threshold 写进 solver

---

# Implementation Preference

优先在这些位置扩展：

- `methods/weight_methods_vargrad.py`
- `methods/__init__.py`
- `experiments/utils.py`
- `experiments/cityscapes/trainer_vargrad.py`
- 原始 run scripts

避免新建一整套并行 trainer。

建议实现方式：

- 保留原 solver 类组织方式
- 在 `weight_methods_vargrad.py` 中抽少量共享 helper
- 第一阶段先完成 `FairGrad` 的全流程
- 等 `fairgrad + vargrad + psmgd_periodic/dynamic` 被验证后，再推广到其他 solver

---

# Logging And Telemetry

后续实验必须能明确看出 solver 在哪些 step 被调用。

至少记录：

- `global_step`
- `scheduler_step`
- `solver_called`
- `updated_weights`
- `weights`
- `candidate_weights`
- `scheduler`
- `preprocessing`
- `solver`

动态 PSMGD 还要记录：

- `dynamic_refresh_metric`
- `dynamic_refresh_direction`
- `dynamic_refresh_score`
- `dynamic_refresh_threshold`
- `dynamic_refresh_triggered`
- `refresh_rel_fro`
- `step_rel_fro`
- `last_refresh_step`

不保存完整 `U_t` 或完整 `U_last_refresh` 到主日志。需要分析阈值时，只保存轻量 JSONL telemetry。

---

# Compatibility Rules

## Original FairGrad Compatibility

当：

- `preprocessing=identity`
- `solver=fairgrad`
- `scheduler=every_step`

时，行为应尽量接近原始 FairGrad。

具体要求：

- FairGrad solver 公式不变
- FairGrad alpha 和原脚本一致，实验中显式传 `2.0`
- shared gradient 写回前乘任务数 `n_tasks`
- 不引入额外 PSMGD 平滑
- 不引入动态 gate

## VarGrad + FairGrad Compatibility

当：

- `preprocessing=vargrad`
- `solver=fairgrad`
- `scheduler=every_step`

时，应接近师兄当前项目中的 VarGrad + FairGrad 行为，但要修正硬编码 beta，使 CLI 传入的 beta 生效。

## Uniform Baseline

后续扩展 `uniform` 时：

- `preprocessing=identity`
- `solver=uniform`
- `scheduler=every_step`

应退化为默认均匀权重 baseline。

---

# Config Requirements

至少支持这些配置项：

- `preprocessing`
- `solver`
- `scheduler`
- `use_vargrad`
- `use_psmgd`
- `use_momentum`

以及这些超参数：

- `beta`
- `beta_v`
- `beta_m`
- `psmgd_R`
- `psmgd_alpha`
- `psmgd_dynamic_metric`
- `psmgd_dynamic_direction`
- `psmgd_dynamic_threshold`
- `alpha`

命名必须区分：

- `alpha`: FairGrad alpha
- `psmgd_alpha`: PSMGD 平滑系数
- `beta` / `beta_v`: VarGrad beta
- `beta_m`: 额外 momentum beta

---

# Non-Goals

不要实现以下内容，除非明确要求：

- VarGrad 里的 SMO
- 新定义的 FairGrad 数学公式
- 重写新的训练系统
- 重写数据集和模型
- 在第一阶段同时大改 MGDA/CAGrad/NashMTL

---

# First Implementation Target

第一阶段只完成下面三组可运行实验：

1. `identity + fairgrad + every_step`
2. `vargrad + fairgrad + every_step`
3. `vargrad + fairgrad + psmgd_periodic`
4. `vargrad + fairgrad + psmgd_dynamic`

每组实验都必须能通过日志确认：

- solver 是否被调用
- 权重是否刷新
- `psmgd_periodic` 是否严格按周期刷新
- `psmgd_dynamic` 是否严格按阈值和方向刷新
