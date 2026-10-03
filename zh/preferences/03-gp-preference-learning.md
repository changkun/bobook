---
status: done
synopsis: "Chu 与 Ghahramani 的模型：效用服从高斯过程，只能通过带噪声的比较来观测，用 Newton 法与 Laplace 近似拟合。本章逐步推导拟合过程，预测新的比较，展示模型在一维、二维及更高维中的表现，说明比较无法识别哪些量、比较图如何影响后验，最后考察 BoTorch 中 PairwiseGP 的实现。"
sources: ["Chu and Ghahramani 2005", "Houlsby et al. 2011", "Shah et al. 2016", "Hendrickx et al. 2019"]
---

# 高斯过程偏好学习 {#sec-gp-preference}

@sec-comparisons 把“你更喜欢哪一个？”的回答转化为似然，即概率单位选择模型：按照这一模型，偏好某个选项的概率随两个选项效用之差的增大而增大。这种似然导致后验不再是高斯分布，@sec-approx-inference 介绍了相应的处理方法。本章借助 @sec-gp-regression 的高斯过程先验把两者结合起来。有了这一先验，针对几十个设计的几十个回答就能为每个设计提供信息，包括从未展示过的设计。

由此得到的就是 @chu2005preference 的模型。二十年后，它已成为偏好贝叶斯优化的默认模型，@sec-pbo 中的图运行的也是这一模型。本章先建立模型，推导拟合方法，再用于预测，然后讨论回归模型中不曾出现的两个问题：一是比较根本无法提供哪些信息；二是比较了哪些对，会如何影响比较所能提供的信息。

## 模型 {#sec-pref-model}

人对定义域 $\X$ 中的设计有潜在效用 $g$。设计可以是海报的颜色、外骨骼控制器的参数、照片滤镜的设置。为效用赋予高斯过程先验，

$$
g \sim \GP(0, k),
$$

于是在获得任何回答之前，任意有限个设计的效用都服从联合高斯分布，协方差由核函数 $k$ 给出（@sec-function-space）。核函数体现了一个关键假设：在 $\X$ 中相近的设计，效用也相近。正是这一假设，使得从少量回答中学习成为可能。

数据由 $n$ 个不同设计 $\vx_1, \dots, \vx_n$ 之间的 $m$ 个回答组成。第 $k$ 个回答表明设计 $v_k$ 优于设计 $u_k$，这里 $v_k$ 与 $u_k$ 是设计在列表中的下标。同一设计可以出现在多个回答中，因此 $n$ 至多为 $2m$，通常还要小得多。将各设计处的效用排成向量 $\vf = (g(\vx_1), \dots, g(\vx_n))^\T$，其先验为 $\N(\mathbf{0}, \mK)$，其中 $[\mK]_{ij} = k(\vx_i, \vx_j)$。每个回答服从 Thurstone 第五种情形（@eq-cmp-case-v），且给定效用时各回答相互独立：

$$
p(\D \mid \vf) = \prod_{k=1}^{m} \Phi\!\left(\frac{f_{v_k} - f_{u_k}}{\sqrt{2}\,\sigma}\right).
$$ {#eq-pref-lik}

@chu2005preference 推导这一似然的方式与 @sec-thurstone 完全相同：先设想一个理想的判断者，只要 $g(\vx_{v_k}) \ge g(\vx_{u_k})$ 就偏好 $v_k$；再给每个潜在值叠加方差为 $\sigma^2$ 的独立高斯噪声。噪声尺度 $\sigma$ 与核函数的长度尺度、幅度一样，都是超参数。

$\vf$ 的后验即 @eq-approx-posterior：高斯先验乘以 @eq-pref-lik 后归一化。这一后验不是高斯分布，@chu2005preference 用 @sec-laplace 的 Laplace 近似代替它。此后，新设计处的预测与回归完全相同，直接由高斯过程得到：给定 $\vf$，其他任何位置的效用都服从条件高斯分布。

同一模型还可以从另一个角度理解。似然只依赖于差值，因此可以定义一个以选项对为自变量的函数 $h(\vx, \vx') = g(\vx) - g(\vx')$，把每个回答视为该对上的二元标签。$h$ 是 $g$ 的线性变换，因而本身也是高斯过程，其协方差为

$$
\Cov\big(h(\vx, \vx'),\, h(\mathbf{y}, \mathbf{y}')\big) = k(\vx, \mathbf{y}) + k(\vx', \mathbf{y}') - k(\vx, \mathbf{y}') - k(\vx', \mathbf{y}),
$$

将两个差值的协方差逐项展开即得此式。@houlsby2011bayesian（预印本）推导出这一**偏好核**（preference kernel），并据此得出结论：Chu 与 Ghahramani 的模型“等价于采用某一类特定核函数的 GPC”，即定义在选项对上的高斯过程分类。因此，分类问题的全部工具，包括 @sec-approx-inference 中的各种近似方法，都适用于这一模型。

这类偏好学习很早就应用于真人。@brochu2007active 采用同样的 Thurstone 模型，让人在渲染出的样例之间做选择，以此帮助他们找到计算机图形学所需的材质外观。此后，该模型又结合 Laplace 近似，用于根据用户的比较和序数标签学习用户偏好的外骨骼步态 [@li2021roial]。@sec-cs-exoskeleton 与 @sec-cs-photo 完整演示了两个此类问题的求解过程。

## 拟合模型 {#sec-pref-laplace}

拟合模型，就是求出后验众数 $\hat\vf$ 以及众数处的曲率。@eq-approx-newton 给出了适用于任意似然的 Newton 步，所需的只是似然的梯度和负 Hessian 矩阵 $\mW$。对比较数据，这两者具有特定的结构，本章后文的许多结论都源于此，因此下面给出完整推导。记 $s = \sqrt{2}\,\sigma$；对第 $k$ 个回答，记 $z_k = (f_{v_k} - f_{u_k})/s$，$\mathbf{a}_k = \mathbf{e}_{v_k} - \mathbf{e}_{u_k}$，该向量在获胜选项处为 $+1$，在落败选项处为 $-1$，其余位置为 0，从而 $z_k = \mathbf{a}_k^\T\vf/s$。

::: {.derivation title="成对概率单位对数似然的梯度与 Hessian 矩阵"}
1. 对 @eq-pref-lik 取对数：$\ell(\vf) = \sum_k \log\Phi(z_k)$。
2. 对单独一项求导。$\tfrac{\dd}{\dd z}\log\Phi(z) = \phi(z)/\Phi(z)$，记为 $\lambda(z)$，称为**逆 Mills 比**（inverse Mills ratio）；又 $\partial z_k / \partial\vf = \mathbf{a}_k/s$，由链式法则得 $\nabla \log\Phi(z_k) = \tfrac{\lambda(z_k)}{s}\,\mathbf{a}_k$。
3. 对所有回答求和：$\nabla\ell(\vf) = \sum_k \tfrac{\lambda(z_k)}{s}\,\mathbf{a}_k$。每个回答使获胜选项对应的分量增加 $\lambda(z_k)/s$，使落败选项对应的分量减少同样的量。
4. 再求一次导。由 $\phi'(z) = -z\phi(z)$ 和商的求导法则得 $\lambda'(z) = -\lambda(z)\big(z + \lambda(z)\big)$，故 $\nabla\nabla \log\Phi(z_k) = -\tfrac{\lambda(z_k)(z_k + \lambda(z_k))}{s^2}\,\mathbf{a}_k\mathbf{a}_k^\T$。
5. 定义权重 $w_k = \lambda(z_k)\big(z_k + \lambda(z_k)\big)/s^2$，则 $\mW = -\nabla\nabla\ell(\vf) = \sum_k w_k\, \mathbf{a}_k\mathbf{a}_k^\T$。
6. 各权重均为正：$1 - \lambda(z)(z + \lambda(z))$ 是标准正态变量在大于 $-z$ 的条件下的方差，取值介于 0 与 1 之间，因此 $0 < \lambda(z)(z + \lambda(z)) < 1$，且 $0 < w_k < 1/s^2$。
7. 于是 $\mW$ 是半正定矩阵 $\mathbf{a}_k\mathbf{a}_k^\T$ 的正倍数之和，$\ell$ 为凹函数。对数先验同样是凹函数，因此对数后验有唯一的极大值，这正是 @chu2005preference 证明的结论（其引理 1）。
:::

$$
\nabla\ell(\vf) = \sum_{k=1}^{m} \frac{\lambda(z_k)}{s}\,\big(\mathbf{e}_{v_k} - \mathbf{e}_{u_k}\big),
\qquad
\mW = \sum_{k=1}^{m} w_k\,\big(\mathbf{e}_{v_k} - \mathbf{e}_{u_k}\big)\big(\mathbf{e}_{v_k} - \mathbf{e}_{u_k}\big)^\T.
$$ {#eq-pref-grad-hess}

下面逐个分量解读 @eq-pref-grad-hess。对每个设计，它每赢得一个回答，梯度就将其推高 $\lambda(z_k)/s$；每输掉一个回答，梯度就将其压低同样的量。回答越出乎意料，推力越大：当 $z$ 取很大的负值，即按当前效用本应是落败选项获胜时，$\lambda(z)$ 近似按 $-z$ 增长；回答在意料之中时，它趋于零。矩阵 $\mW$ 的对角元是各设计所参与回答的权重之和，非对角元是两个设计之间回答的权重之和的相反数。这正是**比较图**（comparison graph）的**加权图 Laplace 矩阵**（weighted graph Laplacian）；比较图以设计为节点，以已回答的对为边。其含义留待 @sec-comparison-graph 讨论。

由此可直接得到两个推论。其一，每个 $\mathbf{a}_k$ 的分量之和为零，故梯度的分量之和也为零，即 $\mathbf{1}^\T\nabla\ell = 0$，且 $\mW\mathbf{1} = \mathbf{0}$：所有效用同时增加同一个量，任何回答的概率都不变。其二，权重依赖于 $\vf$，因此每个 Newton 步都要重新计算 $\mW$。

有了这两项，拟合过程即为下面的算法。本书图形代码中的 `fitPreference` 实现的就是这一算法。

::: {.algorithm #alg-pref-fit title="用 Newton 法拟合偏好模型"}
输入：设计 $\vx_1, \dots, \vx_n$，回答 $(v_k, u_k)$（$k = 1, \dots, m$），核函数 $k(\cdot, \cdot)$，噪声 $\sigma$。

1. 构造 $\mK$，在对角线上加一个小的抖动项；令 $\vf \leftarrow \mathbf{0}$。
2. 由 @eq-pref-grad-hess 计算 $\nabla\ell(\vf)$ 和 $\mW$。
3. 执行 Newton 步 @eq-approx-newton：$\vf_{\text{new}} \leftarrow \mK(\mI + \mW\mK)^{-1}\big(\mW\vf + \nabla\ell(\vf)\big)$。
4. 若对数后验 $\ell(\vf) - \tfrac12\vf^\T\mK^{-1}\vf$ 减小，则将步长减半，$\vf_{\text{new}} \leftarrow \vf + \tfrac12(\vf_{\text{new}} - \vf)$，直至其增大。
5. 令 $\vf \leftarrow \vf_{\text{new}}$，返回第 2 步，直至最大变化量低于容差。将结果记为 $\hat\vf$。
6. 返回 $\hat\vf$、$\hat{\bm\beta} = \nabla\ell(\hat\vf)$ 以及在 $\hat\vf$ 处计算的 $\hat\mW$。
:::

初始时每个 $z_k = 0$，在模型看来每个回答都如同抛硬币，所有权重都等于 $\lambda(0)^2/s^2 = (2/\pi)/s^2$。因此第一步对所有回答一视同仁，后续各步再按各回答实际的意外程度重新加权。在本章的图中，拟合在五至十三步内收敛。每步需求解一个 $n \times n$ 线性方程组，代价为 $O(n^3)$，与回归相同。

在众数处，对数后验的梯度为零：$\nabla\ell(\hat\vf) - \mK^{-1}\hat\vf = \mathbf{0}$，因此

$$
\hat\vf = \mK\,\hat{\bm\beta}, \qquad \hat{\bm\beta} = \nabla\ell(\hat\vf),
$$ {#eq-pref-mode}

即 @chu2005preference 的式（11）。向量 $\hat{\bm\beta}$ 相当于回归中的 $\bm\alpha = \mK^{-1}\vy$（@eq-gp-representer）：每个设计对应一个权重，胜出多于模型预期的设计权重为正，落败的设计权重为负。

::: {.example #ex-pref-two title="两个设计之间的单个回答"}
考虑两个先验相关系数为 $\rho$ 的设计，即 $\mK = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}$，以及一个回答：$A$ 优于 $B$。此时 $\mathbf{a} = (1, -1)^\T$，$\mW = w \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$，其中 $w$ 为众数处的权重。Laplace 协方差为 $(\mK^{-1} + \mW)^{-1}$，由 Sherman-Morrison 公式可写成 $\mK - \frac{\mK\mathbf{a}\mathbf{a}^\T\mK}{1/w + \mathbf{a}^\T\mK\mathbf{a}}$。其中 $\mathbf{a}^\T\mK\mathbf{a} = 2 - 2\rho$ 是差值 $\Delta = f_A - f_B$ 的先验方差，$\mK\mathbf{a} = (1 - \rho)(1, -1)^\T$。

- 差值的后验方差为 $\mathbf{a}^\T\Sigma\mathbf{a} = v_\Delta - \frac{v_\Delta^2}{1/w + v_\Delta} = \frac{v_\Delta}{1 + w v_\Delta}$，其中 $v_\Delta = 2 - 2\rho$。回答使这一方差减小。
- 和 $f_A + f_B$ 的后验方差不变，因为 $(1, 1)\,\mK\mathbf{a} = 0$。回答不提供任何关于整体水平的信息。

这正是 @fig-approx-2d 所示的二维图景，此处以公式表达：一次比较只在差值方向上产生影响。
:::

同样的计算还给出 Laplace 模型证据 @eq-approx-laplace-evidence。@chu2005preference 最大化的正是这一证据，优化变量为核函数超参数和 $\sigma$（其式（12））；BoTorch 拟合偏好模型的方式与此相同（@sec-pairwisegp）。

::: {.code title="NumPy"}
```python
import numpy as np
from scipy.stats import norm

def terms(f, duels, s):
    """Gradient and negative Hessian of the pairwise probit log-likelihood."""
    n = len(f)
    grad, W = np.zeros(n), np.zeros((n, n))
    for i, j in duels:                        # 设计 i 优于设计 j
        z = (f[i] - f[j]) / s
        lam = np.exp(norm.logpdf(z) - norm.logcdf(z))   # phi(z) / Phi(z)
        grad[i] += lam / s
        grad[j] -= lam / s
        w = lam * (z + lam) / s**2
        W[i, i] += w; W[j, j] += w; W[i, j] -= w; W[j, i] -= w
    return grad, W

def fit_preference(K, duels, sigma=0.1, iters=50):
    n, s = len(K), np.sqrt(2) * sigma
    f = np.zeros(n)
    for _ in range(iters):
        grad, W = terms(f, duels, s)
        f_new = K @ np.linalg.solve(np.eye(n) + W @ K, W @ f + grad)
        done = np.max(np.abs(f_new - f)) < 1e-8
        f = f_new
        if done:
            break
    grad, W = terms(f, duels, s)              # 众数处的 beta 与 W
    return f, grad, W
```
这段示意代码省略了 @alg-pref-fit 中的步长减半，这一步只在最初几步越过众数时才有影响；代码还使用了通用求解器，更细致的实现会利用矩阵的对称性，如 @rasmussen2006gaussian 对分类问题的处理。@sec-impl-pref 在最小实现中构建了完整的模型。
:::

## 预测偏好 {#sec-pref-predict}

求得众数和曲率后，$\vf$ 的 Laplace 后验为高斯分布 $\N(\hat\vf, (\mK^{-1} + \hat\mW)^{-1})$，新设计处的效用预测便归结为高斯条件化，与 @sec-gp-conditioning 相同。对新设计 $\vx$，记 $\vk(\vx)$ 为它与各已比较设计之间的先验协方差。

::: {.derivation title="新设计处潜在值的后验"}
1. 给定 $\vf$，由高斯过程先验得 $g(\vx) \mid \vf \sim \N\big(\vk(\vx)^\T\mK^{-1}\vf,\; k(\vx, \vx) - \vk(\vx)^\T\mK^{-1}\vk(\vx)\big)$（即 @eq-gp-posterior，以 $\vf$ 作为无噪声的“观测”）。
2. 对 $\vf$ 的 Laplace 后验取平均。由 @eq-pref-mode，均值为 $\vk(\vx)^\T\mK^{-1}\hat\vf = \vk(\vx)^\T\hat{\bm\beta}$。
3. 方差等于条件方差加上条件均值的离散程度（全方差公式）：$k(\vx, \vx) - \vk(\vx)^\T\mK^{-1}\vk(\vx) + \vk(\vx)^\T\mK^{-1}(\mK^{-1} + \hat\mW)^{-1}\mK^{-1}\vk(\vx)$。
4. 由 Woodbury 恒等式（@sec-id-woodbury），当 $\hat\mW$ 可逆时，$\mK^{-1} - \mK^{-1}(\mK^{-1} + \hat\mW)^{-1}\mK^{-1} = (\mK + \hat\mW^{-1})^{-1}$；一般情形下该式等于 $(\mI + \hat\mW\mK)^{-1}\hat\mW$，这一形式无需求 $\hat\mW$ 的逆。
5. 因此方差为 $k(\vx, \vx) - \vk(\vx)^\T(\mI + \hat\mW\mK)^{-1}\hat\mW\,\vk(\vx)$。
:::

$$
\mu(\vx) = \vk(\vx)^\T \hat{\bm\beta},
\qquad
\sigma^2(\vx) = k(\vx, \vx) - \vk(\vx)^\T (\mI + \hat\mW\mK)^{-1}\hat\mW\, \vk(\vx).
$$ {#eq-pref-predict}

这与 @chu2005preference 的式（17）、式（18）一致，也与 Rasmussen 与 Williams 给出的分类预测方程 [@rasmussen2006gaussian, 第 3 章] 一致，只是以比较数据得到的 $\hat\mW$ 代替了其中的对角矩阵。第 4 步的第二种形式不可或缺：$\hat\mW$ 是图 Laplace 矩阵，总有零特征值，因此 $\hat\mW^{-1}$ 不存在。

偏好优化器最常用到的量，是人偏好新设计 $\vx$ 而非另一个新设计 $\vx'$ 的概率。在 Laplace 近似下，两个潜在值服从联合高斯分布，均值为 $\mu, \mu'$，方差为 $\sigma^2(\vx), \sigma^2(\vx')$；协方差 $c$ 由同一公式得到，只需把右侧换成 $\vk(\vx')$。二者之差的均值为 $\mu - \mu'$，方差为 $\sigma^2(\vx) + \sigma^2(\vx') - 2c$，代入 @eq-cmp-predictive 得

$$
\Prob(\vx \succ \vx' \mid \D) = \Phi\!\left(\frac{\mu - \mu'}{\sqrt{2\sigma^2 + \sigma^2(\vx) + \sigma^2(\vx') - 2c}}\right),
$$ {#eq-pref-pair}

即 @chu2005preference 的式（19）。协方差 $c$ 不可忽略。两个相近设计的潜在值高度相关，二者之差的不确定性因而远小于其中任何一个值；即使模型不确定任何一个设计有多好，也可以对二者的顺序相当有把握。

下图在贯穿全书的示例目标函数上运行这一模型，即 @sec-bo-loop 中的双峰函数。模拟用户以该函数为效用，每次比较的回答带有少量概率单位噪声；比较哪些对由你选择。图上方的比较图为每个回答画一条弧。

```{figure}
//| figure: pref-duels-1d
//| label: fig-pref-duels
//| fig-cap: "高斯过程偏好模型从比较中学习示例目标函数。点击两个输入，请模拟用户比较二者，或按“随机一对”。蓝色曲线与区间带是 Laplace 后验 @eq-pref-predict；虚线是目标函数，经过平移和缩放以便对照，因为比较无法确定效用的水平与尺度；上方的弧是已回答的对，按连通分量着色。状态行显示 @alg-pref-fit 的 Newton 步数，以及模拟用户作答之前模型赋予最后一个获胜选项的概率 @eq-pref-pair。切换到“你”可亲自回答。模拟用户的噪声（σ = 0.05）与模型设置均仅作示意。"
duels: "0.05>0.85,0.65>0.95,0.4>0.55,0.75>0.3,0.55>0.85,0.15>0.85"
```

可以尝试以下几点：

- **按两次“随机十对”。** 图中初始只有六个回答，不足以找到任何一个峰。再增加 20 个回答后，均值曲线呈现出两个峰，最大值位于 $x = 0.72$，紧邻位于 0.73 的真实最大值。每次比较只在其两端处移动曲线，并通过核函数影响两端的邻域。
- **观察区间带。** 区间带在已比较的设计附近收窄，但整体上仍然很宽，即使有 20 个回答，也仍约为 $\pm 1$。这并非模型失效，原因见 @sec-pref-identifiability。
- **查看最后一次预测。** 最初几个回答的预测概率较为分散，一些获胜选项事先被预测会落败（概率低于一半）；之后，多数获胜选项的预测概率在 0.8 或以上。出乎意料的回答使曲线移动最大，因为其 $\lambda(z)$ 较大。
- **改变长度尺度。** 当 $\ell = 0.03$ 时，每个回答只形成一个局部凸起，附近没有比较的地方，曲线都回落到零附近。当 $\ell = 0.5$ 时，曲线过于僵硬，无法同时容纳两个峰，最大值落在错误的位置。
- **反复按“最优对随机”。** 此时每一对都包含当前最优设计，比较图成为星形。最优设计周围的区域学得很好，其余区域学得很差。同样的模式还会出现在 @sec-choosing-pairs 的擂主与挑战者规则中，以及 EUBO 向当前最优点坍缩的现象中（@sec-pbo-failure-modes）。

### 二维及更高维 {#sec-pref-higher-dim}

从 @eq-pref-lik 到 @eq-pref-pair，所有公式都与输入维度无关：设计只通过核函数、以距离的形式进入模型。下图在二维设计空间上运行同一模型，模拟用户的效用为 Branin 函数。这是一个标准测试函数，沿一条弯曲的山谷有三个同样好的最大值；此处将其缩放到单位正方形上，并使数值越大越好。

```{figure}
//| figure: pref-duels-2d
//| label: fig-pref-2d
//| fig-cap: "二维设计空间上的偏好模型。左：隐藏效用（经缩放的 Branin 函数），按秩着色。中：后验均值，按秩着色，并标出已比较的设计（圆点，按连通分量着色）、已回答的对（线段）和推荐设计（星形），即后验均值最高的已比较设计。右：各设计效用相对于正方形上平均值的后验标准差。四种布局以不同方式分配同样数量的比较：在设计池中随机取对、链、围绕一个设计的星形、孤立对。读数给出两项指标：已比较设计对中被后验均值正确排序的比例，以及后验均值与隐藏效用在整个正方形上的秩相关系数。模拟用户与模型的噪声均为 σ = 0.1；所有数值仅作示意。"
```

可以尝试以下几点：

- **对照左图与中图。** 只用 17 个设计之间的 20 个随机对，后验均值就恢复了山谷的大致形状。在默认抽取下，后验均值对 89% 的已比较设计对排序正确，在整个正方形上与隐藏效用的秩一致性为 0.87；多按几次“重新抽取”，可见该一致性在约 0.4 至 0.9 之间变化。
- **查看右图。** 不确定性在比较过的设计处最低，在任何比较都未触及的角落最高。
- **切换到“星形”。** 每个回答都涉及同一个中心设计。中心设计与其他设计的比较关系得到了学习，但其他设计之间从未相互比较，模型在这种布局下表现最差：30 次抽取的平均结果是 81% 的对排序正确，秩一致性为 0.63；随机对则分别为 85% 和 0.71。
- **切换到“孤立对”。** 此时二十个回答涉及 40 个设计，分属 20 个互不相连的分量。每个回答确定了两个设计的顺序，但不直接说明它们与其他对之间的关系，因此对与对之间的偏移只能来自核函数。核函数如此光滑时，这种做法可行：30 次抽取中，模型仍对 82% 的对排序正确；40 个设计对正方形的覆盖优于 17 个设计，因此模型在正方形上的秩一致性略高（0.79）。
- **将长度尺度缩短到 0.08。** 此时核函数对设计之间的关联减弱。孤立对的正确排序比例降至 76%，随机对降至 82%：先验对设计之间的关联越弱，就越需要依靠回答本身来建立关联。

维度更高时，模型的表现如何？同一段代码可以给出答案。@tbl-pref-dims 给出一项示意性计算：在单位立方体中取 40 个随机设计，由噪声为 $\sigma = 0.1$ 的模拟用户回答其间的 $m$ 个随机对，使用长度尺度为 0.52 的径向基函数核（BoTorch 默认先验的众数，@sec-pairwisegp），重复 20 次。效用取 3 维和 6 维的 Hartmann 函数（两个标准基准），以及嵌入 20 维空间的 6 维 Hartmann 函数，其中 14 个输入不起作用。“核函数值”一列是两个设计之间的平均先验相关系数。

::: {.table #tbl-pref-dims title="用本章模型所做的示意性计算：随着维度增加，40 个随机设计之间的 20、40 或 80 次比较对效用的学习效果。ρ 是在 300 个新随机设计上后验均值与真实效用之间的秩相关系数（两种排序完全一致时为 1，互不相关时为 0）。名次是推荐设计在 40 个已比较设计中的真实名次（1 为最好）。均为 20 次重复的平均值。"}
| 效用 | 核函数值 | ρ，20 | ρ，40 | ρ，80 | 名次，20 | 名次，40 | 名次，80 |
|---|---|---|---|---|---|---|---|
| Hartmann，3 维 | 0.47 | 0.75 | 0.79 | 0.86 | 4.7 | 5.0 | 3.1 |
| Hartmann，6 维 | 0.23 | 0.49 | 0.55 | 0.61 | 7.7 | 5.5 | 3.1 |
| 20 维中的 6 维 Hartmann | 0.006 | 0.17 | 0.26 | 0.26 | 10.8 | 8.1 | 2.3 |
:::

有两个规律值得注意。一是模型对新设计的把握随维度增加而崩溃：在 20 维中，按核函数衡量，随机设计彼此相距太远（平均核函数值为 0.006），每个回答几乎只能提供其自身两个设计的信息。二是模型对已见过的设计的排序，在每个维度上都随回答增多而持续改善，因为这些设计由直接比较排序。若让长度尺度随输入个数 $d$ 增长，取 $0.3\sqrt{d}$，20 维中的平均核函数值可提高到 0.41，但新设计上的秩相关系数仅提高到 0.19、0.30 和 0.28：问题出在 14 个不起作用的输入上，而所有输入共用同一长度尺度的核函数（各向同性核）无法忽略它们。这些结果只来自一个示意性设置，并非基准测试（推断）。它们提示了高维偏好贝叶斯优化为何既依赖核函数的结构，例如每个输入各有一个长度尺度（@sec-ard），也依赖比较对的选择（@sec-pbo）；相关研究见 @sec-high-dimensions。

## 哪些量可以识别 {#sec-pref-identifiability}

@fig-pref-duels 中的区间带在 20 个回答之后仍然很宽；而有 20 个观测的回归模型，其区间带会在每个观测处收窄。差别不在于近似，而在于比较能够测量什么。

**平移。** 每个回答的概率都依赖于效用之差，因此给所有效用加上同一常数 $c$，即 $\vf \mapsto \vf + c\mathbf{1}$，似然不变。在 @eq-pref-grad-hess 中，这体现为 $\mathbf{1}^\T\nabla\ell = 0$ 和 $\mW\mathbf{1} = \mathbf{0}$：任何回答都不会沿方向 $\mathbf{1}$ 施加推力，也不会在该方向上增加曲率。关于整体水平的信息只来自先验，后验因而保留了先验对整体水平的不确定性；正是这部分为所有设计共有的不确定性使区间带一直很宽。下图展示同一模型，但区间带改为针对 $g(\vx)$ 减去其在定义域上平均值后的量，这是比较能够测量的量。

```{figure}
//| figure: pref-duels-1d
//| label: fig-pref-relative
//| fig-cap: "20 个回答之后的后验，区间带为 g(x) 减去其在 [0, 1] 上的平均值（点线）后的 95% 区间带。去除无法测量的整体水平后，区间带约缩窄一半：平均标准差从 0.59 降至 0.33，最小标准差从 0.51 降至 0.17。关闭“相对平均值的区间带”可作对比。"
duels: "0.05>0.85,0.65>0.95,0.4>0.55,0.75>0.3,0.55>0.85,0.15>0.85,0.4>0.9,0.25>0.6,0.55>0.5,0.65>0.4,0.25>0.05,0.65>0.6,0.1>0.85,0.05>0.4,0.2>0.15,0.8>0.5,0.35>0.6,0.85>0.9,0.6>0.95,0.25>0.65"
relative: true
```

平移不变性对优化无害，因为优化只需要效用的顺序；但若把后验方差当作已学到多少信息的度量，就会因此产生误判。有些模型在构造上消除了这种不变性。@byk2020active 从偏好中学习机器人奖励函数时，把任意选定的参考设计处的效用固定为零，并将这一约束内置于核函数中，原因是查询的回答对这类平移保持不变。@chau2022inconsistent 同样指出，效用只能确定到相差一个全局平移的程度。

**尺度与噪声。** 似然依赖于 $\vf/\sigma$。将所有效用和噪声同时加倍，每个回答的概率完全不变。因此，比较是以回答者自身的噪声为单位来测量效用的，正如 Thurstone 以辨别离散度为单位测量量表值（@sec-thurstone）。在模型中，只有先验能把二者区分开：核函数的幅度规定了效用的先验大小，幅度固定后，$\sigma$ 便可相对于幅度得到识别。仅凭比较同时拟合二者，实际上只拟合了一个比值。BoTorch 明确采用了这一处理：固定噪声，学习幅度（@sec-pairwisegp）。在 @fig-pref-relative 中，幅度固定为 1，因此“模型噪声 σ”滑块改变的就是这一比值。当 $\sigma = 0.03$ 时，很小的差值（$\sigma$ 的数倍）即可满足这些回答，后验均值从最低点到最高点的跨度约为 0.7；当 $\sigma = 0.1$ 时，跨度约为 1.4，最大值仍位于 $x = 0.72$。回答确定了顺序和大致形状，而差值的大小取决于为模型设定的噪声。当 $\sigma = 0.3$ 时，回答的分量过轻，最大值移到了较宽的那个峰上。

**仍可识别的量。** 效用的顺序、最大值的位置以及未来回答的概率都可以识别；水平不可识别，尺度只能相对于噪声识别。这对跨人或跨会话比较所学效用有实际影响：水平或尺度不同的两个后验可能描述的是同样的偏好，比较二者需要重新校准，例如加入共同的参照比较（推断）。@sec-many-users 将针对多人模型再次讨论这一问题。

## 比较图 {#sec-comparison-graph}

@eq-pref-grad-hess 表明，对数似然的曲率是加权图 Laplace 矩阵：设计是节点，已回答的对是权重为 $w_k$ 的边。图 Laplace 矩阵的性质已有透彻研究，其中三条可以直接转化为关于回答能确定哪些量的结论。

**连通分量即零特征值。** 对任意向量 $\vf$，$\vf^\T\mW\vf = \sum_k w_k (f_{v_k} - f_{u_k})^2$。该式为零当且仅当 $\vf$ 在图的每个连通分量上为常数，因此每个连通分量对应 $\mW$ 的一个零特征值。其中一个对应 @sec-pref-identifiability 中的全局平移；其余每一个都对应两组设计之间的偏移，这两组设计之间既无直接比较，也无经由其他设计构成的比较链。沿这些方向，回答不携带任何信息，后验只能依靠先验。下图以比较次数相同的四种布局展示这一点；@fig-pref-2d 已在地图上展示过同样的现象。

```{figure}
//| figure: comparison-graph
//| label: fig-pref-spectrum
//| fig-cap: "比较次数相同而布局不同时的比较图，以及似然曲率 W 的特征值，每次比较的权重为 1。每个零特征值（下方）都对应一个不受任何回答约束的方向：第一个是全局平移，其余各个（红色）是连通分量之间的偏移。这些布局仅作示意。"
design: chain
comparisons: 8
```

**连通程度即第二小特征值。** 连通图恰有一个零特征值，第二小的特征值，即 @fiedler1973algebraic 所称的**代数连通度**（algebraic connectivity），衡量图的连通程度。链的代数连通度很小，星形或稠密连通图的代数连通度则较大。保持比较次数不变，将 @fig-pref-spectrum 在“链”与“星形”之间切换：链的几个最小非零特征值向零聚集，星形的则都等于 1。特征值小的方向几乎不受回答约束。

**差值即电阻。** 把每个回答看作连接其两个设计、电导为 $w_k$ 的电阻。当先验很弱时，两个设计之差的不确定性就等于二者之间的**有效电阻**（effective resistance）。

::: {.derivation title="平坦先验下差值的不确定性"}
1. 若先验足够宽，在相关方向上 $\mK^{-1} \approx \mathbf{0}$，则 $\vf$ 的 Laplace 精度即为 $\mW$，这是一个奇异矩阵。但在同一连通分量内，满足 $\mathbf{1}^\T\mathbf{a} = 0$ 的差值 $\mathbf{a}^\T\vf$ 仍有良好定义。
2. 这类差值的方差为 $\mathbf{a}^\T\mW^{+}\mathbf{a}$，其中 $\mW^{+}$ 是伪逆，即在零特征向量的正交补空间上的逆。
3. 取 $\mathbf{a} = \mathbf{e}_i - \mathbf{e}_j$。按定义，$(\mathbf{e}_i - \mathbf{e}_j)^\T\mW^{+}(\mathbf{e}_i - \mathbf{e}_j)$ 就是电导为 $w_k$ 的电网络中节点 $i$ 与 $j$ 之间的有效电阻 $R_{ij}$。
4. 因此 $\Var(f_i - f_j) \approx R_{ij}$。串联电阻相加，所以长的比较链会留下很大的不确定性；并联电阻相互合并，所以两个设计之间若有许多条独立路径，二者之差就能牢牢确定下来。
:::

$$
\Var(f_i - f_j \mid \D) \approx R_{ij} = (\mathbf{e}_i - \mathbf{e}_j)^\T \mW^{+} (\mathbf{e}_i - \mathbf{e}_j).
$$ {#eq-pref-resistance}

在由八次等权比较构成的链中，两端之间相隔八个电阻，两端之差的方差是单次比较的八倍；在星形中，任意两片叶子之间只隔两个电阻。先验会把每个方差限制在其先验值以内，但若按差值的确定程度对设计排序，所得顺序仍与电阻一致。

这幅图景不只是对高斯过程模型的类比。针对在 Thurstone 模型和 Bradley-Terry 模型下估计有限个项目的效用，@shah2016estimation 证明了极小极大误差界，这些误差界通过 Laplace 谱依赖于比较图的拓扑；@hendrickx2019graph 证明相关的量是“比较图电阻的平方根”，并给出了在相差对数因子意义下与之匹配的下界。因此，询问哪些对，对任何比较模型都很重要，而不仅限于本章的模型。

**实践意义。** 在高斯过程先验下，Laplace 精度 $\mK^{-1} + \mW$ 总是可逆的，因为先验贡献了 $\mK^{-1}$ 一项；@fig-pref-2d 也表明，光滑的核函数可以弥补许多缺失的关联。先验无法弥补时，问题就出现了：沿 $\mW$ 的零方向或近零方向，精度完全来自先验；若先验在该方向上的方差很大，精度就很小，矩阵随之病态。一篇 2026 年的预印本正是将默认流程中的数值问题归因于此：该文观察到，EUBO 倾向于选择与先前查询没有共同设计的对，导致似然的 Hessian 矩阵秩亏；按先验不确定性缩放的对角修正，在 5 至 20 维的 11 个基准上使结果最多改善 10.9%，$p = 0.003$（**p 值**（p-value）指：若修正毫无作用，观察到至少如此大差异的概率）[@shao2026adaptive]。ICML 2026 上的一篇论文也论证了比较图的间隔与连通性决定 Bradley-Terry 估计的样本效率 [@pukdee2026preference]。@sec-obs-graphs 报告这些工作，@sec-pbo-failure-modes 将其列为这一循环的已知失效模式。

::: {.keyidea title="比较图是数据的一部分"}
比较沿比较图的边约束差值。图不连通时，连通分量之间的偏移只能由先验决定；长链则使两端之间只有松散的联系。询问哪些对，与这些对得到怎样的回答，同样决定着后验。
:::

## 软件实现 {#sec-pairwisegp}

拟合这一模型时，多数人使用 BoTorch 的 `PairwiseGP`。其文档字符串将它描述为“一个使用概率单位似然、通过成对比较数据学习的高斯过程，它对所估计效用值的后验采用 Laplace 近似”[@botorch2026pairwisegp]。该模型以张量形式接收设计，以下标对列表的形式接收回答（受偏好的设计在前），并通过最大化 Laplace 模型证据拟合超参数 [@botorch2026pairwise]。

::: {.code title="BoTorch"}
```python
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize

X = torch.rand(10, 3, dtype=torch.double)     # 10 个设计，每个有 3 个参数
comparisons = torch.tensor([[0, 1], [2, 0], [3, 4], [4, 2]])  # 每行 (i, j)：i 优于 j

model = PairwiseGP(X, comparisons, input_transform=Normalize(d=X.shape[-1]))
mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

post = model.posterior(torch.rand(5, 3, dtype=torch.double))
post.mean, post.variance                      # 5 个新设计处的潜在效用
```
这段代码沿用 BoTorch 偏好教程的结构；完整的循环见 @sec-impl-botorch。
:::

这些默认设置值得重视，因为它们实际上就是整个领域的默认设置。阅读 0.18.1 版的源代码，可以看到以下选择 [@botorch2026pairwisegp; @botorch2026likelihood]。

::: {.table #tbl-pref-botorch title="BoTorch 的 PairwiseGP 内部的各项选择及其在本章中的解释位置。"}
| 选择 | 默认值 | 解释位置 |
|---|---|---|
| 似然 | 概率单位，$\Phi\big((g(v) - g(u))/\sqrt{2}\big)$，噪声隐式固定为 1；自变量截断到 $[-3, 3]$ | @eq-pref-lik、@sec-random-utility |
| 备选似然 | 逻辑似然，logit 截断到 $[-8, 8]$ | @sec-bradley-terry |
| 噪声尺度 $\sigma$ | 省略；其作用由核函数的输出尺度承担，输出尺度是 BoTorch 对幅度 $\sigma_f^2$ 的称呼 | @sec-pref-identifiability |
| 后验众数 | 用 `scipy.optimize.fsolve` 求解，以上一次的解热启动 | @alg-pref-fit |
| 数值处理 | Cholesky 分解中加 $10^{-6}$ 的抖动项；相距 $10^{-4}$ 以内的重复设计合并为一个 | @sec-gp-computation |
| 超参数 | `PairwiseLaplaceMarginalLogLikelihood`，即 Chu 与 Ghahramani 式（12）的 Laplace 模型证据 | @eq-approx-laplace-evidence |
| 核函数 | 带缩放的径向基函数核，每个输入一个长度尺度；常数均值不参与优化 | @sec-ard |
| 长度尺度先验 | Gamma(2.4, 2.7)，初始值取其众数，每个维度约为 0.52 | @sec-pref-higher-dim |
| 输出尺度先验 | $[0.01, 100]$ 上的平滑箱形先验，约束为 $[0.005, 200]$ | @sec-pref-identifiability |
:::

其中三项与本章直接相关。第一，固定噪声为 1、学习输出尺度，就是把 @sec-pref-identifiability 中尺度与噪声的权衡明确化：幅度大意味着回答果断。第二，将概率单位的自变量截断在 $\pm 3$，使拟合时任何单次比较的似然至多约为 0.9987，从而限制了单个出乎意料的回答的拉力，这对应 @fig-cmp-choice-curve 中概率单位尾部的问题；截至 2026 年 9 月，我们没有找到将这一点记录为建模选择的论文（推断）。第三，长度尺度先验不随维度缩放：0.12.0 版（2024 年 9 月）把大多数 BoTorch 模型改为随维度缩放的长度尺度先验，但明确排除了 `PairwiseGP`，后者在 0.18.1 版（2026 年 6 月）中仍使用 Gamma(2.4, 2.7) [@botorch2026changelog]。如 @tbl-pref-dims 所示，约 0.5 的固定长度尺度会使高维空间中的典型设计处于核函数值接近零的区域；`PairwiseGP` 本身是否如此，尚未得到直接验证（推断）。

其他库的选择有所不同。optuna-dashboard 中的偏好采样器基于 @takeno2023practicalc，采用 Matérn 3/2 核，每个输入各有一个长度尺度，先验为 Gamma(5, 10)，同样与维度无关 [@optuna2026sampler]，并采用 @sec-approx-inference 中的近似方法。版本历史和这些选择的细节见 @sec-obs-pairwisegp。

## 习题 {#sec-pref-exercises}

::: {.exercise #exr-pref-zero-sum}
证明 @eq-pref-mode 中的权重 $\hat{\bm\beta}$ 之和恒为零。由此，在远离所有已比较设计之处，后验均值 @eq-pref-predict 有何性质？当核函数平稳、定义域很大时，$\mu(\vx)$ 在许多设计上的平均值又如何？

::: {.solution}
$\hat{\bm\beta} = \nabla\ell(\hat\vf) = \sum_k (\lambda(z_k)/s)(\mathbf{e}_{v_k} - \mathbf{e}_{u_k})$，而每个 $\mathbf{e}_{v_k} - \mathbf{e}_{u_k}$ 的分量之和为零，故 $\mathbf{1}^\T\hat{\bm\beta} = 0$。远离所有已比较设计时，每个 $k(\vx, \vx_i)$ 都接近零，$\mu(\vx) \approx 0$，即回到先验均值，这与回归相同。后验均值是若干核函数凸起之和，其权重相互抵消：核函数平稳时，在很大的定义域上每个凸起的积分相同，因此 $\mu$ 在定义域上的积分等于 $\sum_i \hat\beta_i$ 乘以该积分值，即为零。数据抬高一些区域、压低另一些区域，升降总量相等；数据从不抬高整体水平，因为数据无法测量整体水平。
:::
:::

::: {.exercise #exr-pref-two-weight}
在 @ex-pref-two 中取 $\rho = 0$、$\sigma = 1/\sqrt{2}$（从而 $s = 1$）。已知众数满足 $\hat\Delta / 2 = \lambda(\hat\Delta)$，其中 2 是差值的先验方差。用数值方法求差值的众数 $\hat\Delta$ 与权重 $w$，再计算差值的 Laplace 标准差，并与先验标准差比较。

::: {.solution}
$\Delta$ 的对数后验为 $-\Delta^2/4 + \log\Phi(\Delta)$，其导数在 $\hat\Delta/2 = \lambda(\hat\Delta)$ 处为零。代入几个值试算：在 $\Delta = 0.8$ 处，$\lambda(0.8) \approx 0.29/0.79 \approx 0.37$，而 $\Delta/2 = 0.40$；在 $\Delta = 0.75$ 处，$\lambda \approx 0.301/0.773 \approx 0.39$，而 $\Delta/2 = 0.375$。因此 $\hat\Delta \approx 0.77$，此处 $\lambda \approx 0.38$。于是 $w = \lambda(\hat\Delta + \lambda) \approx 0.38 \times 1.15 \approx 0.44$，差值的 Laplace 方差为 $v_\Delta/(1 + w v_\Delta) = 2/(1 + 0.88) \approx 1.06$，标准差约为 1.03，先验标准差则为 $\sqrt{2} \approx 1.41$。在中等噪声下，一个回答消除了差值约一半的方差，而和的方差丝毫未减。
:::
:::

::: {.exercise #exr-pref-resistance}
三个设计以单位权重相互比较。布局（a）中回答构成一条链：1 与 2 比较，2 与 3 比较。布局（b）再加上 1 与 3 的比较，构成三角形。利用 @eq-pref-resistance 分别求两种布局在平坦先验下的 $\Var(f_1 - f_3)$，并回答：若只直接比较 1 与 3，需要比较多少次才能与三角形相当？

::: {.solution}
（a）两个单位电阻串联：$R_{13} = 2$。（b）直接相连的边（电阻 1）与经过 2 的路径（电阻 2）并联，故 $R_{13} = (1 \cdot 2)/(1 + 2) = 2/3$。将 1 与 3 直接比较 $r$ 次，相当于 $r$ 个单位电阻并联，电阻为 $1/r$；因此 $r = 1.5$ 次直接比较即与三角形相当，而一次直接比较（电阻 1）已优于链。经过 2 的间接路径相当于半次直接比较。
:::
:::

::: {.exercise #exr-pref-scale}
证明：在先验 $\N(\mathbf{0}, a^2\mK)$ 与噪声 $\sigma$ 下，$\vf/\sigma$ 的后验仅通过比值 $a/\sigma$ 依赖于 $a$ 和 $\sigma$。若通过最大化模型证据同时拟合幅度与噪声，这意味着什么？

::: {.solution}
令 $\mathbf{u} = \vf/\sigma$，其先验为 $\N(\mathbf{0}, (a/\sigma)^2\mK)$，似然 @eq-pref-lik 为 $\prod_k \Phi\big((u_{v_k} - u_{u_k})/\sqrt{2}\big)$，其中不含 $\sigma$。$\mathbf{u}$ 的后验是先验与似然之积，因而只依赖于 $a/\sigma$。模型证据是同一乘积的归一化常数，同样只依赖于 $a/\sigma$：比值相同的任意一对 $(a, \sigma)$ 对回答的拟合效果完全相同。二者只能学到其一，这正是 BoTorch 固定噪声、拟合幅度的原因。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @chu2005preference 篇幅简短而内容完整，涵盖似然、凸性证明、Laplace 近似、模型证据以及新偏好的预测概率。
- @rasmussen2006gaussian 第 3 章给出了分类问题的数值稳定算法，将 $\mW$ 换成 Laplace 矩阵后同样适用。
- @houlsby2011bayesian 推导了偏好核，并提出一种基于信息量选择比较对的规则。
- @shah2016estimation 与 @hendrickx2019graph 针对有限项目集的排序问题，解释了比较图的谱与电阻为何决定误差；@fiedler1973algebraic 提出了代数连通度。
- BoTorch 的偏好教程 [@botorch2026pairwise] 拟合 `PairwiseGP`，并运行 @sec-pbo 的循环。
