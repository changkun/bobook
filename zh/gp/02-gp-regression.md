---
status: done
synopsis: "以观测为条件更新高斯过程：预测方程、后验均值与不确定性对数据和噪声的响应，以及稳定的计算方法。"
sources: ["Rasmussen and Williams 2006, ch. 2", "Garnett 2023, ch. 2"]
---

# 高斯过程回归 {#sec-gp-regression}

@sec-function-space 最终得到一个先验：在观测任何数据之前，高斯过程已为每个函数赋予概率。本章引入数据：在少数几个输入处评估未知函数。问题是：结合先验与这些观测值，能对函数在其余各处得出什么结论？

回答这个问题无需新的工具。按照高斯过程的定义，任意有限个函数值服从联合高斯分布；@sec-gaussian-conditioning 已经说明了如何让联合高斯分布以其中部分坐标为条件。高斯过程回归就是这同一个公式，一侧是已观测的输入，另一侧是待预测的输入。本章其余内容都是对结果的解读：后验均值在数据之间和数据之外如何变化；不确定性为什么取决于观测的位置，而与观测到的值无关；观测带噪声时有哪些变化；以及如何在计算中避免数值问题。

后验不确定性是本书其余部分要花费的资源。@sec-acquisition 中的每个采集函数，都是把它转化为下一次查询的规则，因此有必要透彻了解它的形态。

## 以观测为条件 {#sec-gp-conditioning}

记 $f$ 为定义在输入定义域 $\X$ 上的未知函数，其先验是均值为零、核函数为 $k$ 的高斯过程：

$$
f \sim \GP(0, k).
$$

在 $n$ 个输入 $\vx_1, \dots, \vx_n$ 处观测 $f$，这些输入构成集合 $X$。暂且假设观测是精确的：$y_i = f(\vx_i)$。将观测值堆叠成向量 $\vy = (y_1, \dots, y_n)^\T$。目标是求 $f$ 在 $m$ 个新输入 $X_*$ 处的分布，这些位置上的未知值堆叠为 $\vf_*$。

最简单的情形已在 @ex-gauss-two-inputs 中算过：两个函数值的相关系数为 0.8，其中一个观测为 1.2，对另一个的信念随之变为均值 0.96、标准差 0.6。下面的计算与之相同，只是观测值有 $n$ 个，未观测值可以任意多，相关性由核函数给出。

由高斯过程的定义，$\vy$ 与 $\vf_*$ 服从联合高斯分布。其协方差由核函数逐元素构造，分为四块：已观测输入之间的协方差、已观测输入与新输入之间的协方差，以及新输入之间的协方差。

$$
\begin{bmatrix} \vy \\ \vf_* \end{bmatrix}
\sim \N\!\left( \mathbf{0},\;
\begin{bmatrix} \mK & \mK_* \\ \mK_*^\T & \mK_{**} \end{bmatrix} \right).
$$ {#eq-gp-joint}

其中 $[\mK]_{ij} = k(\vx_i, \vx_j)$ 是已观测输入之间的协方差，$[\mK_*]_{ij} = k(\vx_i, \vx_{*j})$ 是已观测输入与新输入之间的协方差，$[\mK_{**}]_{ij} = k(\vx_{*i}, \vx_{*j})$ 是新输入之间的协方差。

以 $\vy$ 为条件，正是 @sec-gaussian-conditioning 中的运算。

::: {.derivation title="预测分布"}
由 @eq-gauss-conditional，若联合高斯分布的分块为 $\vmu_a, \vmu_b$ 与 $\mSigma_{aa}, \mSigma_{ab}, \mSigma_{bb}$，则给定 $\mathbf{a}$ 时 $\mathbf{b}$ 的条件分布是高斯分布，均值为 $\vmu_b + \mSigma_{ab}^\T \mSigma_{aa}^{-1}(\mathbf{a} - \vmu_a)$，协方差为 $\mSigma_{bb} - \mSigma_{ab}^\T \mSigma_{aa}^{-1} \mSigma_{ab}$。

1. 取 $\mathbf{a} = \vy$，$\mathbf{b} = \vf_*$。两者的先验均值均为零，即
   $\vmu_a = \vmu_b = \mathbf{0}$。
2. 由 @eq-gp-joint 读出各块：$\mSigma_{aa} = \mK$，$\mSigma_{ab} = \mK_*$，$\mSigma_{bb} = \mK_{**}$。
3. 代入，得条件均值 $\mK_*^\T \mK^{-1} \vy$，条件协方差 $\mK_{**} - \mK_*^\T \mK^{-1} \mK_*$。
:::

于是新值的后验为

$$
\vf_* \mid X, \vy, X_* \sim \N\!\left(\mK_*^\T \mK^{-1} \vy,\;
\mK_{**} - \mK_*^\T \mK^{-1} \mK_*\right).
$$ {#eq-gp-posterior}

上述推导与新输入的个数和位置都无关。后验仍是高斯过程：对任意有限个新输入，预测服从联合高斯分布，均值与协方差如上。这种封闭性使该方法切实可用：数据只需吸收一次，结果可以在任何位置查询。

对单个新输入 $\vx$，各块退化为一个向量和两个数。记 $\vk(\vx) = (k(\vx, \vx_1), \dots, k(\vx, \vx_n))^\T$ 为 $\vx$ 与各已观测输入之间的协方差，则后验均值与方差为

$$
\mu(\vx) = \vk(\vx)^\T \mK^{-1} \vy,
\qquad
\sigma^2(\vx) = k(\vx, \vx) - \vk(\vx)^\T \mK^{-1} \vk(\vx).
$$ {#eq-gp-pointwise}

这两个式子是全章实际使用的形式。需要强调观测个数 $n$ 时，本书其余部分写作 $\mu_n(\vx)$ 和 $\sigma_n(\vx)$，并始终带自变量。它们不同于 @sec-gp-noise 中的噪声标准差 $\sigma_n$：后者不带自变量，下标表示噪声（noise）。

::: {.keyidea title="不确定性取决于观测位置，与观测值无关"}
@eq-gp-pointwise 中的方差包含输入与核函数，但不包含观测值 $\vy$。核函数固定时，模型在 $\vx$ 处的不确定程度只取决于在哪些位置做过观测，与观测到什么值无关。观测值只出现在均值中。若核函数的超参数由数据拟合（@sec-kernels），观测值会经由拟合得到的长度尺度与幅度间接影响方差。
:::

## 解读后验 {#sec-reading-posterior}

下图在 160 个输入构成的网格上计算 @eq-gp-pointwise。蓝线为后验均值；阴影带覆盖均值上下 1.96 个后验标准差的范围，在每个输入处包含 95% 的后验概率。下方条带单独画出标准差。

```{figure}
//| figure: gp-posterior
//| label: fig-gp-posterior
//| fig-cap: "基于四个观测、使用径向基函数核的高斯过程回归。点击图像可添加观测，点击某个点可将其删除；调节长度尺度，可以看到每个观测的影响延伸多远。下方条带显示后验标准差：噪声很小时，它在数据处为零，在约两个长度尺度之外回到先验值 1。各点仅作示意。"
```

下面通过几个实验具体理解这些方程。

**在远离其他点处添加一个点**。阴影带在新输入处收窄到几乎为零，向两侧又重新张开。张开的快慢由核函数的长度尺度 $\ell$ 决定：径向基函数核 $k(x, x') = \exp\!\left(-(x - x')^2 / 2\ell^2\right)$ 在两个长度尺度之外已降到约 $0.14$，因此一个观测对两个长度尺度以外的输入几乎不提供信息。

**观察均值在数据之间与数据之外的变化**。在相邻观测之间，均值光滑地插值。远离所有观测时，均值回到先验均值零，阴影带也恢复到先验宽度。模型不会外推趋势，而是退回到观测数据之前的信念。

**缩短长度尺度**。均值在点与点之间曲折地回落到零，阴影带在每个间隙中都明显变宽。加长长度尺度，均值则变成一条僵硬的曲线；若各点彼此不一致，曲线可能完全偏离这些点。长度尺度的选择是 @sec-kernels 的主题。

只有一个观测时，无需矩阵即可看清其中的结构。

::: {.example #ex-one-observation title="单个观测"}
在单位幅度的径向基函数核先验下观测 $y_1 = f(x_1)$，于是 $k(x_1, x_1) = 1$。此时 $\mK = [1]$，$\vk(x) = k(x, x_1)$，@eq-gp-pointwise 化为

$$
\mu(x) = k(x, x_1)\, y_1,
\qquad
\sigma^2(x) = 1 - k(x, x_1)^2.
$$

均值就是以 $x_1$ 为中心的核函数，经缩放后穿过 $y_1$。方差在 $x_1$ 处为零，随 $k(x, x_1)$ 降到零而升至 1。@fig-gp-posterior 中后验的各种特征，都是这一图景的叠加，再按观测之间的重叠程度作修正。
:::

### 均值是鼓包之和 {#sec-gp-bumps}

单个观测的结论可以推广。定义权重 $\bm{\alpha} = \mK^{-1}\vy$，则 @eq-gp-pointwise 中的均值为

$$
\mu(\vx) = \sum_{i=1}^n \alpha_i\, k(\vx, \vx_i),
$$ {#eq-gp-representer}

即 $n$ 个核函数的加权和，每个核函数以一个观测为中心。在图中打开“显示核函数鼓包”，即可画出每一项。两个观测靠得很近时，它们的核函数相互重叠，权重必须彼此补偿；因此，某个权重可能远大于它参与拟合的观测值，甚至符号相反。

```{figure}
//| figure: gp-posterior
//| label: fig-gp-bumps
//| fig-cap: "后验均值（蓝色）是加权核函数（虚线）之和，每个观测对应一项，如 @eq-gp-representer 所示。左侧两个相距很近的观测得到符号相反的权重，在两者间隙之外几乎相互抵消。"
points: "0.15:0.2,0.24:0.9,0.6:-0.5,0.85:0.6"
bumps: true
lengthscale: 0.1
```

核岭回归是一种没有概率解释的方法；当其正则化强度等于下一节引入的噪声方差时，它的预测也是 @eq-gp-representer。高斯过程在此基础上还给出方差。岭回归没有方差，而贝叶斯优化恰恰需要它[@kanagawa2018gaussian]。

## 带噪声的观测 {#sec-gp-noise}

真实的评估很少精确。换一个随机种子，训练得到的准确率就不同；穿着外骨骼行走的人，步伐有好有坏。标准模型给每个观测加上独立的高斯噪声：

$$
y_i = f(\vx_i) + \varepsilon_i,
\qquad
\varepsilon_i \sim \N(0, \sigma_n^2).
$$ {#eq-gp-noise-model}

噪声与 $f$ 独立，不同观测的噪声之间也相互独立，因此它只给每个观测的方差加上 $\sigma_n^2$，不改变任何协方差。联合分布（@eq-gp-joint）形式不变，只是 $\mK$ 换成 $\mK + \sigma_n^2 \mI$；推导同样如此：

$$
\mu(\vx) = \vk(\vx)^\T (\mK + \sigma_n^2 \mI)^{-1} \vy,
\qquad
\sigma^2(\vx) = k(\vx, \vx) - \vk(\vx)^\T (\mK + \sigma_n^2 \mI)^{-1} \vk(\vx).
$$ {#eq-gp-noisy}

图中有两处变化。在 @fig-gp-posterior 中移动噪声滑块，均值不再穿过各点，而是像岭回归那样在拟合与光滑之间折中。阴影带也不再收窄到零，因为带噪声的观测无法精确确定 $f$。

::: {.pitfall title="潜在方差与预测方差"}
@eq-gp-noisy 给出的是潜在值 $f(\vx)$ 的后验方差。若在 $\vx$ 处再做一次带噪声的观测 $y$，其波动更大，多出的部分正是噪声：$\Var[y \mid \text{data}] = \sigma^2(\vx) + \sigma_n^2$。方法需要哪一种方差，取决于要回答的问题。@sec-ei 中的期望改进关心的是 $f$，因此使用 $\sigma^2(\vx)$；下一次测量的预测区间则使用两者之和。各软件库默认返回哪一种并不统一。
:::

噪声方差通常未知。它是一个超参数，在 @sec-kernels 中与长度尺度一同拟合。噪声取得太小，模型会追逐每一次波动；取得太大，模型会忽视真实的结构。两者之间还可能相互替代：短长度尺度配小噪声，长长度尺度配大噪声，都能解释同样曲折的数据。这也是拟合出的超参数需要审慎看待的一个原因。

## 计算方法 {#sec-gp-computation}

公式中含有矩阵的逆，但严谨的实现从不显式求逆。矩阵 $\mK + \sigma_n^2 \mI$ 对称正定，因而存在 Cholesky 分解 $\mL\mL^\T$，其中 $\mL$ 为下三角矩阵（@sec-cholesky）。求解三角方程组的代价为 $O(n^2)$，且数值稳定；对近乎奇异的矩阵求逆，这两点都无法保证。

::: {.algorithm #alg-gp-regression title="高斯过程回归"}
输入：输入 $X$、观测 $\vy$、核函数 $k$、噪声方差 $\sigma_n^2$、测试输入 $\vx$。

1. $\mL \leftarrow \operatorname{cholesky}(\mK + \sigma_n^2 \mI)$，使得 $\mL \mL^\T = \mK + \sigma_n^2 \mI$。
2. $\bm{\alpha} \leftarrow \mL^\T \backslash (\mL \backslash \vy)$，即两次三角求解。
3. $\mu(\vx) \leftarrow \vk(\vx)^\T \bm{\alpha}$。
4. $\mathbf{v} \leftarrow \mL \backslash \vk(\vx)$。
5. $\sigma^2(\vx) \leftarrow k(\vx, \vx) - \mathbf{v}^\T \mathbf{v}$。

其中 $\mA \backslash \mathbf{b}$ 表示方程 $\mA\mathbf{z} = \mathbf{b}$ 的解 $\mathbf{z}$。此即 @rasmussen2006gaussian 中的算法 2.1；该算法还返回 @sec-marginal-likelihood 用到的对数边际似然。
:::

第 4 步是方差公式的另一种写法：$\mathbf{v}^\T\mathbf{v} =
\vk^\T \mL^{-\T}\mL^{-1}\vk = \vk^\T(\mK + \sigma_n^2\mI)^{-1}\vk$。

计算代价分为两部分：每个数据集只需付出一次的部分，以及每次预测都要付出的部分。第 1 步的分解需要 $O(n^3)$ 的时间和 $O(n^2)$ 的内存；此后每个均值的代价为 $O(n)$，每个方差为 $O(n^2)$。笔记本电脑分解几千行的矩阵用时远不到一秒，而一次贝叶斯优化运行的观测很少超过几百个，因此在本书中，三次方的代价很少成为瓶颈。更大的数据集需要近似方法，用较小的诱导点集合概括数据[@quinonero2005unifying; @titsias2009variational]。

::: {.code title="NumPy"}
```python
import numpy as np

def rbf(a, b, ell=0.12):
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)

def gp_posterior(x, y, xs, noise=1e-4, ell=0.12):
    L = np.linalg.cholesky(rbf(x, x, ell) + noise * np.eye(len(x)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = rbf(x, xs, ell)                 # n x m
    mean = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)           # n x m
    var = 1.0 - np.sum(v**2, axis=0)     # 此核函数满足 k(x, x) = 1
    return mean, var
```
生产环境中的实现会使用三角求解器（`scipy.linalg.solve_triangular`），速度更快，也明确体现了矩阵的结构。@sec-minimal-implementation 以这个函数为基础展开。
:::

即使核矩阵本身良定，当两个输入几乎相同时，浮点运算中的分解也可能失败：两行几乎相等，最小特征值经舍入变为零或负数。实现上的做法是在对角线上加一个小常数作为“抖动项”（jitter），例如 $10^{-6}\sigma_f^2$；若分解仍然失败，就换用更大的值重试。有观测噪声时这一问题很少出现，因为 $\sigma_n^2$ 本身就相当于抖动项。

## 后验样本 {#sec-gp-posterior-samples}

均值和阴影带逐个输入地概括后验，无法展示单个合理的函数是什么样子，因为相邻的值高度相关。要看到完整的函数，需要在输入网格上从联合后验（@eq-gp-posterior）中抽取样本。记 $\mSigma_*$ 为后验协方差，$\mL_*$ 为其 Cholesky 因子，则每个样本为

$$
\vf_* = \vmu_* + \mL_* \vz, \qquad \vz \sim \N(\mathbf{0}, \mI),
$$

这就是 @sec-gaussian-linear 中的采样方法。在 @fig-gp-posterior 中打开“显示样本”：每条紫色曲线都穿过（或接近）所有观测，大部分位于阴影带之内，光滑程度与核函数允许的一致。每一条都是模型认为可能的函数。

样本的用处不限于可视化。@sec-thompson 中的 Thompson 采样就是一种采集规则：抽取一个后验样本，在该样本取最大值处评估目标函数。在 $m$ 个点的网格上采样，分解的代价为 $O(m^3)$，网格因此限于几千个点以内；对更大的或连续的定义域，可以借助随机特征或路径式更新，把样本作为函数抽取[@rahimi2007random; @wilson2020efficiently]。

## 易错点 {#sec-gp-pitfalls}

实践中，以下三个习惯可以避免大多数意外。

**标准化输出**。零先验均值和单位幅度假定函数值以零为中心、离散程度在一左右。取值在 1000 附近的函数，在远离数据处会被拉向零。拟合前应减去观测值的均值、除以其标准差，并对预测结果做逆变换。BoTorch 等软件库用结果变换（outcome transform）完成这一步[@balandat2020botorch]。

**缩放输入**。单一的长度尺度假定所有输入方向的变化尺度相当。先把每个输入映射到 $[0, 1]$；若各维度的重要程度不同，再为每个维度设置各自的长度尺度（@sec-ard）。

**不要轻信外推**。在数据范围之外，后验按其构造回到先验。若目标函数的某种趋势延续到观测范围之外，平稳核无法预测出这一趋势。在有界定义域上做优化时，这一点通常可以接受，但也说明定义域不宜设得比必要的更大。

## 习题 {#sec-gp-exercises}

::: {.exercise #exr-two-points}
在无噪声、单位幅度、长度尺度为 $\ell$ 的径向基函数核先验下，得到两个观测 $y_1 = f(0)$ 和 $y_2 = f(\delta)$。令 $\rho = \exp(-\delta^2 / 2\ell^2)$。用 $\rho$ 表示中点 $x = \delta/2$ 处的后验方差，并验证当 $\delta \to 0$ 时它趋于单个观测时的值。

::: {.solution}
这里 $\mK = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}$；记 $r = \exp(-\delta^2 / 8\ell^2)$ 为中点与每个观测之间的核函数值，则 $\vk = (r, r)^\T$。逆矩阵为 $\mK^{-1} = \frac{1}{1 - \rho^2}\begin{bmatrix} 1 & -\rho \\ -\rho & 1 \end{bmatrix}$，因此 $\vk^\T \mK^{-1}\vk = \frac{2r^2(1 - \rho)}{1 - \rho^2} = \frac{2r^2}{1 + \rho}$，于是

$$
\sigma^2(\delta/2) = 1 - \frac{2r^2}{1 + \rho}.
$$

由于 $r^2 = \rho^{1/2}$，当 $\delta \to 0$ 时 $\rho$ 与 $r$ 都趋于 1，方差趋于 $1 - 2/2 = 0$，与直接观测中点的结果相同。$\delta$ 很小时，第二个观测几乎没有带来新信息：两个几乎重合的输入所含的信息与一个输入相差无几。
:::
:::

::: {.exercise #exr-noise-floor}
设噪声方差为 $\sigma_n^2$，$n$ 个观测都在同一输入 $x_0$ 处。证明对单位幅度的核函数，$x_0$ 处的后验方差为 $\sigma_n^2 / (n + \sigma_n^2)$。这一结果说明，重复一次评估而不尝试新输入，会有什么效果？

::: {.solution}
$\mK$ 的所有元素都是 1，所以 $\mK = \mathbf{1}\mathbf{1}^\T$，$\vk(x_0) = \mathbf{1}$。由于 $\mathbf{1}^\T\mathbf{1} = n$，有 $(\mathbf{1}\mathbf{1}^\T + \sigma_n^2\mI)\mathbf{1} = (n + \sigma_n^2)\mathbf{1}$，因此 $(\mathbf{1}\mathbf{1}^\T + \sigma_n^2\mI)^{-1}\mathbf{1} = \mathbf{1}/(n + \sigma_n^2)$（@exr-id-sherman 用 Sherman-Morrison 公式得到同样的结果）。于是 $\vk^\T(\mK + \sigma_n^2\mI)^{-1}\vk = n/(n + \sigma_n^2)$，方差为 $1 - n/(n + \sigma_n^2) = \sigma_n^2/(n + \sigma_n^2)$。重复评估使该输入处的不确定性按 $1/n$ 缩小，与对 $n$ 次带噪声测量取平均的速率相同，但对其他位置几乎不提供信息。只有当噪声相对于所要分辨的差异很大时，贝叶斯优化才会重复评估同一输入。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @rasmussen2006gaussian 的第 2 章给出标准推导，兼用权重空间与函数空间两种视角，并包含本章所用的算法。
- @garnett2023bayesian 的第 2 至 4 章从贝叶斯优化的角度阐述高斯过程，也讨论了本章视为固定的各项推断选择。
- @gortler2019visual 是一篇交互式的可视化入门，可与本章的图互为补充。
- @williams1996gaussian 将高斯过程回归引入机器学习。同样的预测方法早已在地质统计学中使用，称为克里金法（kriging）[@krige1951statistical; @matheron1963principles]。
- @kanagawa2018gaussian 阐明了高斯过程回归与核岭回归等核方法之间的确切对应关系。
