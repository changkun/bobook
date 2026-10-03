---
status: done
synopsis: "遗憾界背后的数学：把函数看作向量；核函数定义的再生核 Hilbert 空间，以及其范数界的含义；Mercer 特征展开；Bochner 定理给出的谱观点与随机特征；特征值衰减如何决定信息增益；以及高斯过程作为随机对象的精确含义。"
sources: ["Rasmussen and Williams 2006, ch. 4 and 6", "Kanagawa et al. 2018", "Vakili et al. 2021", "Srinivas et al. 2010", "Rahimi and Recht 2007", "Da Costa et al. 2026"]
---

# 核函数背后的分析 {#sec-kernel-analysis}

前三章建立了一个可用的模型：核函数刻画未知函数在各处的取值如何共同变化（@sec-function-space），以观测为条件得到预测（@sec-gp-regression），边际似然确定其超参数（@sec-kernels）。这些足以运行贝叶斯优化，却不足以说明它为何有效。@sec-regret 中的理论保证分为两类。贝叶斯保证针对从先验中抽取的函数；频率派保证，以及 @sec-dueling-bandits 与 @sec-pbo-theory 中针对比较的对应结果，则针对单个固定的函数，其再生核 Hilbert 空间范数不超过某个数 $B$。两类保证的速率都取决于核函数特征值衰减的快慢。在这些表述中，核函数不再是构造协方差矩阵的规则，而是一个独立的数学对象：作用于函数的线性映射，有特征值，有频率谱，还规定了哪些函数算是简单的。

本章从读者已掌握的知识出发建立这一观点：内积与特征向量（@sec-linear-algebra），以及高斯模型从带噪声数据中获得的信息（@sec-gp-information-gain）。思路是把函数看作一个很长的向量，将有限维中的结论逐一推广过来。本章最后回到这些结果的用处：最大信息增益的增长速率，以及高斯过程作为随机对象的精确含义。使用 @sec-part-bo 的方法不需要这些内容，读懂其理论保证则需要。

## 函数作为向量 {#sec-ka-functions}

向量 $\vx \in \R^n$ 是 $n$ 个数构成的列表，也就是从下标集 $\{1, \dots, n\}$ 到 $\R$ 的函数：给定下标 $i$，返回 $x_i$。$[0, 1]$ 上的函数 $f$ 是同一类对象，只是下标连续变化。@sec-gp-prior-samples 把函数画成它在网格上的取值构成的向量，依据的正是这一理解。

### 函数的内积 {#sec-ka-inner}

取中点网格 $x_i = (i - \tfrac12)/n$，记两个函数在网格上的取值向量为 $\vf$ 与 $\vg$。内积 $\vf^\T\vg$（@eq-linalg-inner）随 $n$ 增大而增大，因为求和的项越来越多。除以 $n$ 后它趋于稳定：与 @sec-fs-rbf-limit 一样，Riemann 和化为积分：

$$
\frac{1}{n}\sum_{i=1}^n f(x_i)\, g(x_i) \;\longrightarrow\; \int_0^1 f(x)\, g(x)\, \dd x .
$$

这一极限就是**两个函数的内积**（inner product of two functions）。与 @sec-linalg-inner-products 相同，由内积可以得到长度 $\lVert f \rVert = (\int_0^1 f^2\, \dd x)^{1/2}$ 与正交性。用权重密度 $q(x) > 0$ 可以让某些输入占更大的比重：$\langle f, g \rangle_q = \int f(x)\, g(x)\, q(x)\, \dd x$。除非另作说明，本章在 $[0, 1]$ 上使用均匀权重 $q = 1$。

### 标准正交基：以 Fourier 基为例 {#sec-ka-fourier}

标准正交基为向量提供坐标，也能为函数提供坐标。$[0, 1]$ 上最常见的基是 Fourier 基：常数 $1$，以及 $j = 1, 2, \dots$ 时的函数 $\sqrt{2}\cos(2\pi jx)$ 与 $\sqrt{2}\sin(2\pi jx)$；每个基函数的长度为 1，且两两正交。$f$ 的坐标就是它的 Fourier 系数 $a_j$（余弦）与 $b_j$（正弦），其长度的平方等于各系数的平方和，这就是 Parseval 恒等式。

系数反映光滑度。设 $f$ 可微且 $f(0) = f(1)$，分部积分把导数转移到基函数上，并带出因子 $1/(2\pi j)$：$f$ 的余弦系数等于 $f'$ 的正弦系数乘以 $-1/(2\pi j)$，$f$ 的正弦系数等于 $f'$ 的余弦系数乘以 $1/(2\pi j)$。再对 $f'$ 应用 Parseval 恒等式，得

$$
\int_0^1 f'(x)^2\, \dd x = \sum_{j \ge 1} (2\pi j)^2 \left(a_j^2 + b_j^2\right).
$$ {#eq-ka-parseval-derivative}

斜率的能量是粗糙程度的自然度量，它等于各系数平方的加权和，权重随频率增长。高频代价高，低频代价低。@sec-ka-rkhs 中的范数也具有这种形式，只是权重由核函数决定。

### 核矩阵：算子的有限视图 {#sec-ka-operator}

矩阵把向量映射为向量，核函数则把函数映射为函数：

$$
(\mathcal{K} g)(x) = \int k(x, x')\, g(x')\, q(x')\, \dd x' ,
$$ {#eq-ka-operator}

即 $g$ 各处取值的组合，权重是 $x$ 与各个 $x'$ 的相关程度。在网格上，这一积分化为 $\frac1n\sum_j k(x_i, x_j)\, g(x_j)$，即矩阵与向量的乘积 $\frac1n\mK\vg$。因此 $\mK/n$ 是算子 $\mathcal{K}$ 在 $n$ 个点上的表现，其特征值近似于算子的特征值。

@sec-linalg-kernel-spectrum 中已出现过这样一个矩阵：长度尺度为 0.1 的径向基函数核作用于 100 个等距输入，最大的特征值依次为 23.9、21.2 与 17.5。除以 100 后为 0.239、0.212 与 0.175。这些输入包含 $[0, 1]$ 的两个端点。若改取 @sec-ka-inner 中的 100 个网格中点（与下文 @fig-ka-mercer 相同），结果为 0.241、0.214 与 0.176；中点网格再加密，这几位数字也不再变化。在那一节中使 Cholesky 分解失败的快速衰减，属于算子本身，而非网格。

## 核函数定义的函数空间 {#sec-ka-rkhs}

@sec-regret 的频率派遗憾定理针对单个固定的函数，界定优化器相对于本可找到的最优值损失多少。这些定理需要一个函数类：它要足够大，能容纳实际的目标函数；又要足够小，使有限次评估就能确定其中的成员。核函数恰好定义了这样一个函数类，构造从 @sec-function-space 的权重空间观点出发。

### 由鼓包构成的函数 {#sec-ka-bumps}

考虑由特征得到的核函数 $k(\vx, \vx') = \boldsymbol{\phi}(\vx)^\T\boldsymbol{\phi}(\vx')$，即 @eq-fs-induced-kernel 中权重先验协方差取 $\mI$ 的情形。每个权重向量 $\vw$ 给出一个函数 $\vw^\T\boldsymbol{\phi}(\vx)$，核函数鼓包 $k(\cdot, \vx')$ 就是权重为 $\boldsymbol{\phi}(\vx')$ 的那个函数。因此鼓包之和 $f = \sum_i \alpha_i k(\cdot, \vx_i)$ 的权重为 $\sum_i \alpha_i \boldsymbol{\phi}(\vx_i)$；对另一个和 $g = \sum_j \beta_j k(\cdot, \vx'_j)$，两个权重向量的内积为

$$
\langle f, g \rangle_k = \sum_{i}\sum_{j} \alpha_i \beta_j\, k(\vx_i, \vx'_j).
$$ {#eq-ka-inner}

右边不涉及特征，因此可以作为任意半正定核的定义。它也与 $f$、$g$ 写成和式的方式无关：按 $j$ 分组得 $\sum_j \beta_j f(\vx'_j)$，按 $i$ 分组得 $\sum_i \alpha_i g(\vx_i)$，两者都只依赖于函数的取值。$f$ 的范数平方为 $\lVert f \rVert_k^2 = \bm{\alpha}^\T\mK\bm{\alpha} \ge 0$。

### 再生性 {#sec-ka-reproducing}

取 $g = k(\cdot, \vx)$，即系数为 1 的单个鼓包，同样的分组给出

$$
\langle f, k(\cdot, \vx) \rangle_k = f(\vx).
$$ {#eq-ka-reproducing}

与鼓包作内积，就是在该点求函数值。这一性质称为**再生性**（reproducing property）。取 $f = k(\cdot, \vx')$，得 $\langle k(\cdot, \vx'), k(\cdot, \vx) \rangle_k = k(\vx, \vx')$：鼓包本身就是该核函数的特征。把 Cauchy-Schwarz 不等式 $\lvert\langle f, g\rangle_k\rvert \le \lVert f\rVert_k\lVert g\rVert_k$ 分别用于 @eq-ka-reproducing 和两个鼓包之差，得到两个界[@chowdhury2017kernelized]：

$$
\lvert f(\vx) \rvert \le \lVert f \rVert_k \sqrt{k(\vx, \vx)},
\qquad
\lvert f(\vx) - f(\vx') \rvert \le \lVert f \rVert_k \sqrt{k(\vx, \vx) - 2k(\vx, \vx') + k(\vx', \vx')}.
$$ {#eq-ka-bounds}

这两个界说明了范数控制什么。若核函数满足 $k(\vx, \vx) \le 1$，范数不超过 $B$ 的函数，其绝对值处处不超过 $B$。这样的函数也不能变化太快：对径向基函数核，距离为 $r$ 时第二个平方根为 $\sqrt{2 - 2e^{-r^2/2\ell^2}}$，当 $\ell = 0.1$ 时它在 $r = 0.01$ 处为 0.0999，接近 $r/\ell$。在远小于长度尺度的距离上，这样的函数至多变化约 $B\,r/\ell$。取值大、变化快，都要消耗范数。

第一个界还表明：只有零函数的范数为零，因此 $\lVert\cdot\rVert_k$ 是真正的长度；按这一范数收敛的鼓包之和在每个输入处都收敛，因此其极限也是函数。把这些极限加进来，构造就完成了。

::: {.definition #def-ka-rkhs title="再生核 Hilbert 空间"}
设 $k$ 是 $\X$ 上的对称半正定核。它的**再生核 Hilbert 空间**（reproducing kernel Hilbert space，RKHS）$\mathcal{H}_k$ 是鼓包之和在 @eq-ka-inner 所给范数下完备化得到的函数空间。它是唯一满足以下条件的 Hilbert 空间：由函数构成，包含每个鼓包 $k(\cdot, \vx)$，且 @eq-ka-reproducing 对其中每个成员和每个输入都成立[@aronszajn1950theory; @rasmussen2006gaussian, 定理 6.1]。Hilbert 空间即完备的内积空间：其中任一序列只要各项彼此任意接近，就在该空间中有极限。
:::

对常见的核函数，这个空间可以具体写出。直线核 $k(x, x') = xx'$（@ex-fs-line 中取 $\sigma_0 = 0$、$\sigma_1 = 1$）的 RKHS 由直线 $f(x) = wx$ 组成，范数为斜率的绝对值 $\lvert w \rvert$。对由有限个特征构成、权重协方差为 $\mI$ 的核函数，$f$ 的范数等于能产生它的最短权重向量的长度[@steinwart2008support, 第 4 章]。$d$ 维有界定义域上光滑度为 $\nu$ 的 Matérn 核，当 $\nu + d/2$ 为整数且边界正则时，其 RKHS 与一个 Sobolev 空间所含的函数相同，即 $\nu + d/2$ 阶及以下各阶导数平方可积的函数，且两者的范数等价[@kanagawa2018gaussian, 例 2.6]。径向基函数核的 RKHS 只包含 Fourier 变换按指数速度衰减的函数，因此这些函数极其光滑[@kanagawa2018gaussian, 例 2.7]。

### 后验均值属于这一空间 {#sec-ka-representer}

高斯过程回归的后验均值是鼓包的加权和，每个观测对应一个鼓包（@eq-gp-representer），因此它属于 $\mathcal{H}_k$。它还是一个不涉及概率的优化问题的解。

::: {.derivation title="平方误差下的表示定理"}
求使 $L(f) = \sum_{i=1}^n \big(y_i - f(\vx_i)\big)^2 + \sigma_n^2 \lVert f \rVert_k^2$ 最小的 $f \in \mathcal{H}_k$。

1. 记 $S$ 为 $k(\cdot, \vx_1), \dots, k(\cdot, \vx_n)$ 张成的子空间，令 $f = f_S + f_\perp$，其中 $f_\perp$ 与 $S$ 中的每个鼓包都正交。
2. 由 @eq-ka-reproducing，$f(\vx_i) = \langle f_S + f_\perp, k(\cdot, \vx_i)\rangle_k = f_S(\vx_i)$。数据项只与 $f_S$ 有关。
3. 由勾股定理，$\lVert f\rVert_k^2 = \lVert f_S\rVert_k^2 + \lVert f_\perp\rVert_k^2$，因此去掉 $f_\perp$ 会降低惩罚项，最小值点必在 $S$ 中：$f = \sum_j \alpha_j k(\cdot, \vx_j)$。
4. 此时拟合值为 $\mK\bm{\alpha}$，且 $L = \lVert\vy - \mK\bm{\alpha}\rVert^2 + \sigma_n^2\bm{\alpha}^\T\mK\bm{\alpha}$。
5. 梯度 $-2\mK\big(\vy - (\mK + \sigma_n^2\mI)\bm{\alpha}\big)$ 在 $\bm{\alpha} = (\mK + \sigma_n^2\mI)^{-1}\vy$ 处为零，因此 $f(\vx) = \vk(\vx)^\T(\mK + \sigma_n^2\mI)^{-1}\vy$，正是 @eq-gp-noisy 中的后验均值。
:::

在无穷维空间上做带惩罚的拟合，解却只有 $n$ 项，这一结论称为**表示定理**（representer theorem），最早由 @kimeldorf1971some 在平方误差下给出；它与高斯过程后验均值的对应关系可追溯到 @kimeldorf1970correspondence[另见 @kanagawa2018gaussian, 命题 3.6]。像这里这样数据项为简单求和时，惩罚权重就是噪声方差，即 @sec-gp-bumps 中的核岭回归。在权重空间中，这种对应在意料之中：取 $\vw \sim \N(\mathbf{0}, \mI)$，负二倍的对数后验在相差一个常数的意义下为 $\sigma_n^{-2}\sum_i(y_i - f(\vx_i))^2 + \lVert\vw\rVert^2$，而产生 $f$ 的最短 $\vw$ 的长度为 $\lVert f\rVert_k$。范数的平方相当于负二倍的对数先验。

@fig-gp-posterior 默认设置下（长度尺度 0.12，四个观测，绝对值最大为 0.85）的后验均值满足 $\lVert\mu\rVert_k^2 = \bm{\alpha}^\T\mK\bm{\alpha} = 1.39$，因此 @eq-ka-bounds 给出的上限为 1.18，高于其实际最大值：范数给出的是保证，而不是描述。

### 范数界假定了什么 {#sec-ka-norm-bound}

在频率派遗憾定理（@sec-regret-other-settings、@sec-kernelized-dueling、@sec-pbo-theory）中，$f$ 是一个固定的函数，唯一的随机性来自评估噪声。这些定理假定 $f$ 属于 $\mathcal{H}_k$，且其范数有已知的上界。由以上各节，若 $\lVert f\rVert_k \le B$ 且 $k(\vx, \vx) \le 1$，则 $f$ 处处不超过 $B$，每个长度尺度内至多变化约 $B$，并且在核函数认为代价高的方向上权重很小。这一假设不仅取决于 $f$，还取决于核函数及其长度尺度：长度尺度越长，快速变化的代价越高；对径向基函数核，宽度不超过 $\ell/\sqrt{2} \approx 0.71\,\ell$ 的高斯鼓包，其范数为无穷大（@exr-ka-narrow-bump）。

文献中有两种约定，相差一个平方。@srinivas2010gaussian 假定 $\lVert f \rVert_k^2 \le B$，@sec-regret-other-settings 也采用这一约定；@chowdhury2017kernelized 以及 @sec-kernelized-dueling 与 @sec-pbo-theory 中的偏好论文则假定 $\lVert f\rVert_k \le B$。在后一种约定下，界直接进入置信宽度。若噪声是 $R$-次高斯的（尾部不比标准差为 $R$ 的高斯分布更重），Chowdhury 与 Gopalan 证明：以至少 $1 - \delta$ 的概率，$\lvert\mu_{t-1}(\vx) - f(\vx)\rvert \le \beta_t^{1/2}\sigma_{t-1}(\vx)$ 对所有 $\vx$ 和所有轮次 $t \le T$ 成立，其中 $\beta_t^{1/2} = B + R\sqrt{2(\gamma_{t-1} + 1 + \log(1/\delta))}$[@chowdhury2017kernelized, 定理 2]；他们的后验与 $\gamma_{t-1}$ 都以噪声方差 $1 + 2/T$ 代替 $\sigma_n^2$，他们的 $\beta_t$ 是本书中相应量的平方根。这类宽度从何而来，见 @sec-regret。实践中 $B$ 是未知的，而决定其单位的核函数又由同一批数据拟合（@sec-regret-bounds-not-say）。

### 样本比空间中的函数更粗糙 {#sec-ka-samples}

@sec-regret-gp-setting 中的贝叶斯定理则假定 $f$ 是从 $\GP(0, k)$ 中抽取的样本。人们自然会猜想，典型样本在 $\mathcal{H}_k$ 中的范数不大。实际上，样本在其中根本没有范数可言。

::: {.derivation title="先验样本几乎必然不属于其 RKHS"}
设 $\vx_1, \vx_2, \dots$ 是互不相同的输入，其核矩阵 $\mK_n$ 均可逆；对径向基函数核与 Matérn 核，任意互不相同的输入都满足这一条件（@sec-ka-bochner-theorem）。

1. **插值界**。设 $g \in \mathcal{H}_k$ 在 $\vx_1, \dots, \vx_n$ 处的取值为 $\vg_n$。由上文第 1 至 3 步（去掉数据项）可知，它在前 $n$ 个鼓包张成的子空间中的分量 $g_S$ 取值相同，而范数不更大。记 $g_S = \sum_j \alpha_j k(\cdot, \vx_j)$，$\mK_n\bm{\alpha} = \vg_n$，即得 $\lVert g\rVert_k^2 \ge \bm{\alpha}^\T\mK_n\bm{\alpha} = \vg_n^\T\mK_n^{-1}\vg_n$。
2. **样本的同一个量**。把样本的取值写成 $\vf_n = \mL_n\vz$，其中 $\mL_n$ 是 $\mK_n$ 的 Cholesky 因子，$\vz$ 服从标准正态分布（@sec-gauss-sampling）。于是 $Q_n = \vf_n^\T\mK_n^{-1}\vf_n = \vz^\T\vz = z_1^2 + \dots + z_n^2$。
3. **嵌套**。加入 $\vx_{n+1}$ 只是在 $\mL_n$ 下方添加一行，原有各行不变，因此 $\vz$ 的前 $n$ 个分量保持不变，$Q_{n+1} = Q_n + z_{n+1}^2$。
4. **无界**。$Q_n$ 的均值为 $n$，标准差为 $\sqrt{2n}$，因此对任意固定的 $c$，$Q_n \le c$ 的概率趋于零。$Q_n$ 只增不减，所以它对所有 $n$ 都保持在 $c$ 以下的概率为零。
5. 若样本属于 $\mathcal{H}_k$，由第 1 步，对每个 $n$ 都有 $Q_n \le \lVert f\rVert_k^2$，因此存在某个整数 $c$，使 $Q_n \le c$ 对每个 $n$ 成立。范数随样本而变，而第 4 步针对的是固定的 $c$，不能直接用于这里。但对可数个 $c = 1, 2, 3, \dots$ 中的每一个，由第 4 步，这一事件的概率为零；这些事件之并的概率也为零，因为并的概率至多为各事件概率之和（@eq-regret-union，此处为可数个事件）。
:::

一般的结论是一条零一律：高斯过程属于某个给定 RKHS 的概率为 0 或 1；只要其自身核函数的 RKHS 是无穷维的，它属于该空间的概率就为 0[@driscoll1973reproducing; @lukic2001stochastic; @kanagawa2018gaussian, 定理 4.9 与推论 4.10]。通过 $n$ 个输入观察，样本看起来像范数平方约为 $n$ 的函数，每增加一个输入，范数平方约增加 1。后验均值则是有限个鼓包之和，求平均消除了粗糙性[@rasmussen2006gaussian, 第 6.1 节]。

样本确实属于由更粗糙的函数构成、稍大一些的空间。对径向基函数核，这一差别很少有实际影响[@kanagawa2018gaussian, 推论 4.13 与注 4.13]。对 Matérn 核则不然：RKHS 要求 Sobolev 光滑度 $\nu + d/2$，而样本具有低于 $\nu$ 的各阶光滑度，仅此而已，比前者粗糙了 $d/2$[@kanagawa2018gaussian, 推论 4.15，注 4.14 与注 4.15]。

因此，@sec-regret 的两种设定所作的假定不同。贝叶斯定理针对样本，任何界 $B$ 都覆盖不了它们；频率派定理针对 $\mathcal{H}_k$ 的成员，而先验赋予这个集合的概率为零。正如 @srinivas2010gaussian 所指出的，两者互不包含。在 $d$ 维中选用 Matérn 5/2 核，再引用频率派的界，就等于假定了 $5/2 + d/2$ 的光滑度，高于同一先验下样本所具有的光滑度（推断）。

## Mercer 定理 {#sec-ka-mercer}

鼓包相互重叠、并不正交，用作坐标并不方便。对于对称矩阵，@sec-linalg-spectral 找到了更好的坐标，即特征向量。算子 $\mathcal{K}$ 是对称矩阵的连续版本，同样的做法依然适用。

### 特征函数与核函数的展开 {#sec-ka-mercer-theorem}

$\mathcal{K}$ 的**特征函数**（eigenfunction）是只被该算子缩放的函数 $\varphi$：$\mathcal{K}\varphi = \lambda\varphi$，其中 $\lambda$ 为特征值。（本章把特征函数记作 $\varphi_i$，以区别于 @sec-function-space 中的特征 $\boldsymbol{\phi}$ 和正态密度 $\phi$。）

::: {.theorem #thm-ka-mercer title="Mercer 定理"}
设 $\X$ 是 $\R^d$ 的有界闭子集，$k$ 是 $\X$ 上连续、对称、半正定的核，$q$ 是对 $\X$ 的每个开区域都赋予正权重的权重（例如满足 $q > 0$ 的密度，更一般地，支撑为 $\X$ 的有限测度）。则 $\mathcal{K}$ 有有限个或可数个特征值 $\lambda_1 \ge \lambda_2 \ge \dots > 0$，对应的特征函数 $\varphi_1, \varphi_2, \dots$ 在 $\langle\cdot,\cdot\rangle_q$ 下标准正交，且

$$
k(\vx, \vx') = \sum_{i} \lambda_i\, \varphi_i(\vx)\, \varphi_i(\vx')
$$ {#eq-ka-mercer}

对所有 $\vx, \vx' \in \X$ 成立，级数绝对且一致收敛[@mercer1909functions; @steinwart2008support, 定理 4.49; @kanagawa2018gaussian, 定理 4.1]。
:::

@eq-ka-mercer 就是谱定理 @thm-linalg-spectral，即 $\mA = \sum_i \lambda_i\mathbf{u}_i\mathbf{u}_i^\T$，只是以函数代替了向量。定理的条件不可省略：在权重为零的区域，展开可能不成立[@kanagawa2018gaussian, 注 4.2]。特征值与特征函数依赖于权重，核函数及其 RKHS 则不依赖[@kanagawa2018gaussian, 注 4.1 与注 4.3]。由此得到三个推论。

**特征值是方差的预算**。在 @eq-ka-mercer 中令 $\vx' = \vx$，再对 $q$ 积分，由标准正交性得 $\sum_i \lambda_i = \int k(\vx, \vx)\, q(\vx)\, \dd\vx$。对平稳核和概率密度 $q$，特征值之和等于先验方差 $\sigma_f^2$，各特征值说明这一方差如何分配到各个方向。

**每个核函数都是特征的内积**。令 $\boldsymbol{\phi}(\vx) = (\sqrt{\lambda_1}\varphi_1(\vx), \sqrt{\lambda_2}\varphi_2(\vx), \dots)$，则 @eq-ka-mercer 可写成 $k(\vx, \vx') = \boldsymbol{\phi}(\vx)^\T\boldsymbol{\phi}(\vx')$。这就是 @sec-fs-valid-kernels 中引用的逆命题。

**特征坐标下的 RKHS 范数**。按特征函数展开，$f = \sum_i c_i\varphi_i$，其中 $c_i = \langle f, \varphi_i\rangle_q$。则 $f \in \mathcal{H}_k$ 当且仅当下式中的和有限，且

$$
\lVert f \rVert_k^2 = \sum_i \frac{c_i^2}{\lambda_i}
$$ {#eq-ka-norm-eigen}

[@rasmussen2006gaussian, 第 6.1 节; @kanagawa2018gaussian, 定理 4.2]。用再生性可以验证这一点：由 @eq-ka-mercer，鼓包 $k(\cdot, \vx)$ 的系数为 $\lambda_i\varphi_i(\vx)$，因此 $\langle f, k(\cdot,\vx)\rangle_k = \sum_i c_i\lambda_i\varphi_i(\vx)/\lambda_i = f(\vx)$。它与 @eq-ka-parseval-derivative 形式相同，只是权重 $1/\lambda_i$ 由核函数决定；它也是以 $\mK$ 的特征向量表示的 $\vf^\T\mK^{-1}\vf$ 的连续形式[@rasmussen2006gaussian, 第 6.1 节]。特征值小的方向代价高：沿该方向的系数 $c$ 要花费 $c^2/\lambda_i$。

### 样本的特征展开 {#sec-ka-karhunen-loeve}

同样的坐标也可以描述先验。取相互独立的标准正态随机数 $z_1, z_2, \dots$，令

$$
f(\vx) = \sum_i \sqrt{\lambda_i}\, z_i\, \varphi_i(\vx).
$$ {#eq-ka-kl}

则 $\E[f(\vx)f(\vx')] = \sum_{i,j}\sqrt{\lambda_i\lambda_j}\,\E[z_iz_j]\,\varphi_i(\vx)\varphi_j(\vx') = \sum_i\lambda_i\varphi_i(\vx)\varphi_i(\vx') = k(\vx, \vx')$，因为 $\E[z_iz_j]$ 在 $i = j$ 时为 1，否则为 0。因此 @eq-ka-kl 是以特征基表示的 $\GP(0, k)$ 的样本，各系数相互独立，方差为 $\lambda_i$。这就是 **Karhunen-Loève 展开**（Karhunen-Loève expansion）；该级数在整个定义域上一致地均方收敛[@kanagawa2018gaussian, 定理 4.3; @berlinet2004reproducing, 第 2.3 节]。由标准正交性，只保留前 $m$ 项时，平均平方误差为

$$
\E\!\left[\int \big(f(\vx) - f_m(\vx)\big)^2 q(\vx)\, \dd\vx\right] = \sum_{i > m} \lambda_i ,
$$ {#eq-ka-kl-error}

即被略去的特征值总量。

对小特征值的两种解读是一致的：先验给这个方向的方差很小，为 $\lambda_i$；范数对它收费很高，为 $1/\lambda_i$。由 @eq-ka-norm-eigen，截断后的样本满足 $\lVert f_m\rVert_k^2 = \sum_{i \le m} z_i^2$，约为 $m$，这正是 @sec-ka-samples 中的图景在特征坐标下的表现。最后这一步只是直观说明，而非证明，因为完整的级数是均方收敛，而不是按范数收敛[@kanagawa2018gaussian, 注 4.9]；证明已在 @sec-ka-samples 中给出。

### 数值计算 {#sec-ka-mercer-figure}

特征函数很少有闭式表达，通常按 @sec-ka-operator 提示的方法计算：在均匀权重下取 $n$ 个网格点，用 $\mK/n$ 的特征值估计 $\lambda_i$，用特征向量乘以 $\sqrt{n}$ 估计 $\varphi_i$ 在网格点上的值。这就是 Nyström 方法；它对较大特征值的估计比对较小特征值更准确[@rasmussen2006gaussian, 第 4.3.2 节]。

```{figure}
//| figure: ka-mercer
//| label: fig-ka-mercer
//| fig-cap: "Mercer 定理的数值计算。左上：$[0, 1]$ 上均匀权重下，四个单位幅度核函数的特征值 $\lambda_i$，双对数坐标，由 100 个网格中点上的 Nyström 方法求得[@rasmussen2006gaussian, 第 4.3.2 节]；小于 $10^{-14}$ 的值是舍入误差，不予绘出。右上：所选核函数的前四个特征函数。下：由 Karhunen-Loève 展开 @eq-ka-kl 的前 $m$ 项构造的样本（实线），与使用相同随机数、由全部 100 项构造的样本（虚线）对照，后者是该过程在网格上的精确样本。读数给出前 $m$ 项所占先验方差的比例，以及截断样本的 RKHS 范数平方 $\sum_{i \le m} z_i^2$。按“重新抽取”可换一组随机数。"
```

**查看默认设置**。长度尺度为 0.1 的径向基函数核，最大的特征值依次为 0.241、0.214 与 0.176，与 @sec-ka-operator 中的值相同；全部特征值之和为 1.00，即先验方差；到第 32 个时已降至舍入误差的水平。前四个特征函数形似余弦，分别变号 0、1、2、3 次。

**拖动“项数 m”**。一项占先验方差的 24%，十项占 99.5%；十项构成的样本与精确样本几乎无法区分。截断样本的范数平方却持续增长：十项时为 8.7，全部 100 项时为 96.5，而样本本身几乎没有变化。

**把“核函数”切换为“Matérn 1/2”**。特征值在双对数坐标中落在一条直线上，即幂律。十项占方差的 80%，二十项约占 90%。截断样本光滑，精确样本则呈锯齿状：粗糙性来自大量的小特征值。

**比较斜率**。$\nu = 1/2$、$3/2$、$5/2$ 时，Matérn 核的直线斜率分别接近 $-2$、$-4$、$-6$；在 3,000 个点的网格上，第 20 至第 60 个特征值之间的斜率为 $-2.03$、$-4.00$ 与 $-5.89$。特征值按 $i^{-(2\nu + 1)}$ 衰减，这与 Ritter 等人的结果一致：$[0, 1]$ 上具有 $r$ 阶均方导数的过程，其特征值按 $i^{-(2r + 2)}$ 衰减[@rasmussen2006gaussian, 第 4.3 节]。

**把“长度尺度 ℓ”加长到 0.3**。径向基函数核的第一个特征值升至 0.590，四项即占方差的 99.6%：长度尺度越长，先验越集中在少数几个方向上。

## Bochner 定理 {#sec-ka-bochner}

Mercer 特征函数依赖于定义域和权重，而且需要数值计算。平稳核只依赖于 $\mathbf{r} = \vx - \vx'$（@sec-fs-stationarity），对它有一种两者都不需要的描述：一组频率，以及每个频率所承载的方差。

### 核函数即频谱 {#sec-ka-bochner-theorem}

从一个振幅随机的余弦出发。若 $f(x) = a\cos(\omega x) + b\sin(\omega x)$，其中 $a, b$ 是相互独立的标准正态变量，则 $\Cov[f(x), f(x')] = \cos\omega x\cos\omega x' + \sin\omega x\sin\omega x' = \cos\big(\omega(x - x')\big)$，这是一个平稳核。对频率取混合可以得到更多的平稳核；Bochner 定理表明，所有平稳核都可以这样得到。

::: {.theorem #thm-ka-bochner title="Bochner 定理"}
$\R^d$ 上的连续函数 $k$ 是平稳核（即 $k(\vx - \vx')$ 半正定），当且仅当存在频率 $\boldsymbol{\omega}$ 上的有限非负测度 $\Lambda$，使 $k(\mathbf{r}) = \int e^{i\boldsymbol{\omega}^\T\mathbf{r}}\, \Lambda(\dd\boldsymbol{\omega})$[@bochner1933monotone; @rasmussen2006gaussian, 定理 4.1]。
:::

对本书中的核函数，$\Lambda$ 有密度 $s(\boldsymbol{\omega})$，称为**谱密度**（spectral density），它关于 $\boldsymbol{\omega}$ 对称，且 $k(\mathbf{r}) = \int s(\boldsymbol{\omega})\cos(\boldsymbol{\omega}^\T\mathbf{r})\,\dd\boldsymbol{\omega}$。这里频率的单位是每单位输入的弧度；Rasmussen 与 Williams 以周数为单位，$\boldsymbol{\omega} = 2\pi\mathbf{s}$，这会改变密度的尺度，但不改变其形状[@rasmussen2006gaussian, 式（4.6）]。由于 $\int s = k(\mathbf{0}) = \sigma_f^2$，$p = s/\sigma_f^2$ 是一个概率密度，且

$$
k(\mathbf{r}) = \sigma_f^2\, \E_{\boldsymbol{\omega} \sim p}\!\left[\cos(\boldsymbol{\omega}^\T\mathbf{r})\right].
$$ {#eq-ka-bochner}

定理的一个方向很容易证明。对输入 $\vx_1, \dots, \vx_n$ 与系数 $a_i$，由恒等式 $\cos(\theta - \theta') = \cos\theta\cos\theta' + \sin\theta\sin\theta'$ 得

$$
\sum_{i,j} a_ia_j\, k(\vx_i - \vx_j) = \int s(\boldsymbol{\omega})\left[\Big(\sum_i a_i\cos\boldsymbol{\omega}^\T\vx_i\Big)^2 + \Big(\sum_i a_i\sin\boldsymbol{\omega}^\T\vx_i\Big)^2\right]\dd\boldsymbol{\omega} \;\ge\; 0 .
$$

若 $s$ 在每个频率上都为正（径向基函数核与 Matérn 核正是如此），则除非所有 $a_i$ 都为零，该积分必为正，因为对互不相同的输入，两个和不可能在每个频率上同时为零。于是，互不相同的输入对应的核矩阵都可逆，@sec-ka-samples 用到了这一点[@wendland2004scattered, 第 6 章]。另一个方向，即每个平稳核都有这样的频谱，才是定理的深刻之处。

### 径向基函数核与 Matérn 核的频谱 {#sec-ka-spectra}

对径向基函数核，频率服从高斯分布 $p = \N(\mathbf{0}, \ell^{-2}\mI)$，因此典型频率约为 $1/\ell$（@exr-ka-gauss-spectrum）。对 Matérn 核，频率服从自由度为 $2\nu$、尺度为 $1/\ell$ 的 Student t 分布，

$$
p(\boldsymbol{\omega}) \propto \left(1 + \frac{\ell^2\lVert\boldsymbol{\omega}\rVert^2}{2\nu}\right)^{-(\nu + d/2)},
$$ {#eq-ka-matern-spectrum}

即 @rasmussen2006gaussian 中的式（4.15）改用弧度表示。一维中 $\nu = 1/2$ 时，它是 Cauchy 分布 $p(\omega) = (\ell/\pi)/(1 + \ell^2\omega^2)$。

两者的尾部不同。高斯分布的尾部比任何幂函数下降得都快，@eq-ka-matern-spectrum 则按 $\lVert\boldsymbol{\omega}\rVert^{-(2\nu + d)}$ 下降，在每个高频上都保留少量方差，$\nu$ 越小保留得越多。尾部决定光滑度。在一维中把 @eq-ka-bochner 在 $\mathbf{r} = 0$ 处求两次导数，得 $-k''(0) = \sigma_f^2\int\omega^2 p(\omega)\,\dd\omega$，即均方导数 $f'(x)$ 的方差（@sec-fs-smoothness）。只有当尾部下降得比 $\lvert\omega\rvert^{-3}$ 更快时它才有限，对 Matérn 核而言即 $2\nu + 1 > 3$，也就是 $\nu > 1$。把 $\omega^2$ 换成 $\omega$ 的更高偶次幂，同样的论证给出 @tbl-kern-family 中的规则：低于 $\nu$ 的每个整数阶都存在均方导数，$\nu$ 阶及以上则都不存在。均方根频率 $\sqrt{-k''(0)/k(0)}$ 正是 @eq-fs-upcrossings 所计数的量：对 Matérn 5/2，自由度为 5 的 Student t 分布方差为 $5/3$（以 $1/\ell^2$ 为单位），对应 @exr-fs-matern 中的 $k''(0) = -5/(3\ell^2)$。

Mercer 与 Bochner 描述的是同一个核函数，在没有端点的定义域上两者完全吻合。把 $[0, 1]$ 的两端接成圆周，并把核函数绕在圆周上：$k_\circ(r) = \sum_m k(r + m)$，对所有整数 $m$ 求和。在一圈上对 $k_\circ(x - y)\cos(2\pi jy)$ 积分，等于在整条实轴上对 $k(x - y)\cos(2\pi jy)$ 积分：代换 $u = y - m$ 把 $k(x - y + m)\cos(2\pi jy)$ 一项在 $[0, 1]$ 上的积分化为 $k(x - u)\cos(2\pi ju)$ 在 $[-m, 1 - m]$ 上的积分，因为余弦以 1 为周期，而这些区间合起来覆盖整条实轴。令 $r = x - y$，余弦可拆为 $\cos(2\pi jx)\cos(2\pi jr) + \sin(2\pi jx)\sin(2\pi jr)$。$k$ 是偶函数，因此正弦部分的积分为零；余弦部分给出 $2\pi s(2\pi j)\cos(2\pi jx)$，因为谱密度可由核函数按 $s(\omega) = \frac{1}{2\pi}\int k(r)\cos(\omega r)\,\dd r$ 求得[@rasmussen2006gaussian, 式（4.6）]。正弦基函数的情形同理。因此圆周上的特征函数就是 Fourier 基函数，特征值为 $\lambda_j = 2\pi s(2\pi j)$，即谱密度在恰好绕圆周整数个周期的频率上的取样。特征值的衰减就是频谱的衰减。第 $i$ 个特征值位于频率 $\pi i$ 附近，因此按 $\lvert\omega\rvert^{-(2\nu+1)}$ 下降的 Matérn 频谱给出按 $i^{-(2\nu + 1)}$ 下降的特征值，这就是 @fig-ka-mercer 中的斜率。区间的端点改变特征值的数值，却不改变尾部：长度尺度为 0.1 时，Matérn 1/2 的第 80 个特征值在区间上为 $3.24 \times 10^{-4}$，在圆周上为 $3.16 \times 10^{-4}$。

### 随机 Fourier 特征 {#sec-ka-rff}

@eq-ka-bochner 把核函数写成期望，而期望可以用平均值估计。从 $p$ 中抽取频率 $\boldsymbol{\omega}_1, \dots, \boldsymbol{\omega}_M$，使用以下 $2M$ 个特征：

$$
\mathbf{z}(\vx) = \frac{\sigma_f}{\sqrt{M}}\big(\cos\boldsymbol{\omega}_1^\T\vx, \dots, \cos\boldsymbol{\omega}_M^\T\vx, \sin\boldsymbol{\omega}_1^\T\vx, \dots, \sin\boldsymbol{\omega}_M^\T\vx\big).
$$ {#eq-ka-rff}

由余弦恒等式，$\mathbf{z}(\vx)^\T\mathbf{z}(\vx') = \frac{\sigma_f^2}{M}\sum_{j}\cos\boldsymbol{\omega}_j^\T(\vx - \vx')$，这是一个期望为 $k(\vx - \vx')$ 的平均值。这就是 @rahimi2007random 提出的**随机 Fourier 特征**（random Fourier features）。取 $\sigma_f = 1$，每一项都落在 $[-1, 1]$ 中，方差为 $\tfrac12(1 + k(2r)) - k(r)^2$，距离很大时趋于 $\tfrac12$（@exr-ka-rff），因此单个距离上的误差通常约为 $1/\sqrt{2M}$。要在有界定义域中的所有输入对上同时达到精度 $\varepsilon$，$M$ 需为 $(d/\varepsilon^2)\log(1/\varepsilon)$ 量级，定义域的直径与 $p$ 的分散程度通过对数项进入[@rahimi2007random, 断言 1]。

加权和 $\mathbf{w}^\T\mathbf{z}(\vx)$（$\mathbf{w} \sim \N(\mathbf{0}, \mI)$）就是 @eq-fs-linear-model 中具有 $2M$ 个特征的线性模型：它是 $\GP(0, k)$ 的近似样本，有显式公式，每个输入的计算代价为 $O(Md)$，可以像任何函数一样求最大值。Thompson 采样在后验随机样本取最大值处评估（@sec-thompson）；如 @sec-gp-posterior-samples 所述，它需要在连续定义域上抽取整个函数，随机 Fourier 特征是实现这一点的一种方法[@rahimi2007random; @wilson2020efficiently]。

```{figure}
//| figure: ka-bochner
//| label: fig-ka-bochner
//| fig-cap: "Bochner 定理与随机 Fourier 特征（单位幅度的核函数）。左上：所选核函数的谱密度 $p(\omega)$ 与其峰值之比，纵轴为对数刻度，横轴为缩放频率 $\omega\ell$；虚线为径向基函数核的密度，供对照；底部的刻线标出抽取的频率（至多 200 个，超出坐标轴的堆在末端）。右上：核函数 $k(r)$（虚线）及其由 $M$ 个随机特征得到的估计，即 $\cos(\omega_j r)$ 的平均值（实线）。下：由 $M$ 个特征以标准正态权重构造的样本，与用 @sec-gauss-sampling 的 Cholesky 方法从同一核函数抽取的精确样本对照。读数给出估计与核函数在 $[0, 1]$ 内各距离上的均方根差距。增大 $M$ 只会添加新的频率，不替换已有的频率；按“重新抽取频率”可换一组频率。"
```

**查看默认设置**。Matérn 3/2 的密度按幂律下降，在 $\omega\ell = 10$ 处仍约为峰值的 $10^{-3}$，而径向基函数核的密度在那里已经消失。$M = 20$ 时，估计的核函数在真实核函数周围起伏，均方根差距为 0.152，对照值 $1/\sqrt{2M} = 0.158$。

**把 M 增至 200，再增至 2000**。差距降至 0.058，再降至 0.023，对照值分别为 0.050 与 0.016：大致符合平均值的 $1/\sqrt{M}$ 速率，$M$ 每增大十倍，差距约缩小为原来的三分之一。

**在 M = 20 时把“核函数”切换为“Matérn 1/2”**。Cauchy 分布的重尾使二十个频率中有几个远在高频处，它们在核函数估计和特征样本中都表现为规则的波纹，而特征样本在其他方面是光滑的。精确样本则在每个尺度上都粗糙。重尾频谱把粗糙性分散到许多高频上，每个高频的方差都很小，二十个特征只能取到其中少数几个。

**切换到“RBF”**。频率都在 $1/\ell$ 的几倍以内；$M$ 较小时，特征样本就已具有精确样本的特征，因为径向基函数核的样本正是由这样的频率构成的。核函数估计并不比之前更准确：$M = 20$ 时差距为 0.161。

## 从特征值到信息增益 {#sec-ka-infogain}

@sec-regret 中的遗憾界（即优化器相对最优值的总损失的界）取决于最大信息增益 $\gamma_T$（@def-regret-gamma），即 $T$ 次带噪声的评估至多能揭示多少关于 $f$ 的信息。@sec-gp-information-gain 引用了它对径向基函数核与 Matérn 核的增长速率。这些速率来自特征值。

### 计数已分辨的方向 {#sec-ka-resolved}

借助 @eq-ka-mercer，把 $T$ 个输入的核矩阵写成 $\mK_A = \boldsymbol{\Phi}\boldsymbol{\Lambda}\boldsymbol{\Phi}^\T$，其中 $\boldsymbol{\Phi}$ 的第 $t$ 行是各特征函数在 $\vx_t$ 处的值，$\boldsymbol{\Lambda}$ 的对角线上是各特征值。行列式引理（@eq-id-det-lemma）把信息增益 @eq-info-gp-gain 从各次评估转移到各特征方向上：$\det(\mI + \sigma_n^{-2}\boldsymbol{\Phi}\boldsymbol{\Lambda}\boldsymbol{\Phi}^\T) = \det(\mI + \sigma_n^{-2}\boldsymbol{\Lambda}^{1/2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi}\boldsymbol{\Lambda}^{1/2})$。若输入按 $q$ 的比例分布，则与 @sec-ka-inner 中的 Riemann 和一样，$\frac1T\sum_t\varphi_i(\vx_t)\varphi_j(\vx_t)$ 近似于 $\int\varphi_i\varphi_j\, q$，其值为 1 或 0，因此 $\boldsymbol{\Phi}^\T\boldsymbol{\Phi} \approx T\mI$，且

$$
I(\vy_A; f) \approx \frac12\sum_i \log\!\left(1 + \frac{T\lambda_i}{\sigma_n^2}\right).
$$ {#eq-ka-info-eigen}

每个特征方向都是一次独立的实验：其系数的先验方差为 $\lambda_i$，$T$ 次分散的评估以约 $\sigma_n^2/T$ 的噪声方差测量它，由 @eq-info-mi-gp 即得上式中的对应项。满足 $T\lambda_i \gg \sigma_n^2$ 的方向称为**已分辨**（resolved）的方向，贡献约 $\tfrac12\log(T\lambda_i/\sigma_n^2)$；满足 $T\lambda_i \ll \sigma_n^2$ 的方向贡献约 $T\lambda_i/(2\sigma_n^2)$。信息量大致等于已分辨方向的个数乘以对数的一半，再加上 $T/(2\sigma_n^2)$ 乘以未分辨方向的特征值总量。

@eq-ka-info-eigen 只是近似。与 $[0, 1]$ 上 $T$ 次等距评估的精确信息量相比（长度尺度 0.1，$\sigma_n = 0.1$），$T = 100$ 时，它对径向基函数核以及 Matérn 5/2、Matérn 3/2 核的误差都在 0.25% 以内。对 Matérn 1/2，$T = 100$ 时它高估 46%，因为它计入了 143 个已分辨的方向，超过了 100 次评估所能分辨的数目；$T = 1000$ 时高估 7.4%。

### 由特征值得到的界 {#sec-ka-gamma-bound}

对任意设计，都可以把这一计数变成严格的界：在第 $m$ 个方向处截断，再为其后的特征值总量付出代价。

::: {.theorem #thm-ka-gamma title="由特征值衰减界定信息增益（Vakili、Khezeli 与 Picheny，2021）"}
设 $k$ 满足 @thm-ka-mercer 的条件，且对所有 $i$ 与 $\vx$ 有 $\lvert k(\vx, \vx')\rvert \le \bar k$ 与 $\lvert\varphi_i(\vx)\rvert \le \psi$。则对每个 $m = 1, 2, \dots$，

$$
\gamma_T \le \frac{m}{2}\log\!\left(1 + \frac{\bar k\, T}{\sigma_n^2\, m}\right) + \frac{T\,\delta_m}{2\sigma_n^2},
\qquad
\delta_m = \psi^2\sum_{i > m}\lambda_i .
$$ {#eq-ka-gamma-bound}

这是 @vakili2021information 的定理 3，原文中的 $\tau$ 即这里的 $\sigma_n^2$，$D$ 即这里的 $m$。
:::

尾部 $\delta_m$ 是特征值总量，而不是 @sec-ka-norm-bound 中 $\delta$ 那样的失效概率；这一记号沿用原文。特征函数有界是一个假设，作者称它对实践中使用的核函数成立；我们没有找到它对径向基函数核与 Matérn 核在任意定义域和权重下都成立的证明。该定理的证明只需要 @sec-mutual-information 中的工具。

::: {.derivation title="界的证明"}
固定 $T$ 个输入构成的集合 $A$，在第 $m$ 项处把核函数分成两部分：$k = k_P + k_O$，其中 $k_P = \sum_{i \le m}\lambda_i\varphi_i\varphi_i$，$k_O$ 为其余部分，两者都是半正定的。

1. **拆分函数**。相互独立的 $f_P \sim \GP(0, k_P)$ 与 $f_O \sim \GP(0, k_O)$ 之和是核函数为 $k$ 的过程（@exr-fs-sum）。因此在 $A$ 上取 $\vy = \vf_P + \vf_O + \bm{\varepsilon}$，$\tfrac12\log\det(\mI + \sigma_n^{-2}\mK_A)$ 就是 $\vy$ 所含的关于这个和的信息（@eq-info-mi-gp）。这个和由二元组 $(f_P, f_O)$ 算出，因此由数据处理不等式（@eq-info-data-processing），这一信息至多为关于二元组的信息。
2. **链式法则**。关于二元组的信息等于 $I(\vy; f_P) + I(\vy; f_O \given f_P)$（@sec-info-sequential）[@cover2006elements, 第 2 章]。给定 $f_P$ 时，数据就是 $f_O$ 加上噪声，因此第二项为 $\tfrac12\log\det(\mI + \sigma_n^{-2}\mK_{O})$。第一项至多为 $\vf_P + \bm{\varepsilon}$ 所含的关于 $f_P$ 的信息，因为加上独立的 $\vf_O$ 只是进一步的处理：$\tfrac12\log\det(\mI + \sigma_n^{-2}\mK_{P})$。
3. **前 $m$ 个方向**。$\mK_P = \boldsymbol{\Phi}_m\boldsymbol{\Lambda}_m\boldsymbol{\Phi}_m^\T$，因此由 @eq-id-det-lemma，相应的行列式等于 $m \times m$ 矩阵 $\mI + \mathbf{G}$ 的行列式，其中 $\mathbf{G} = \sigma_n^{-2}\boldsymbol{\Lambda}_m^{1/2}\boldsymbol{\Phi}_m^\T\boldsymbol{\Phi}_m\boldsymbol{\Lambda}_m^{1/2}$。记其特征值为 $g_j \ge 0$，由对数函数的凹性得 $\sum_j\log(1 + g_j) \le m\log(1 + \tfrac1m\sum_j g_j)$。矩阵乘积的各因子循环轮换后迹不变，因此 $\sum_j g_j = \tr\mathbf{G} = \sigma_n^{-2}\tr\mK_P = \sigma_n^{-2}\sum_t k_P(\vx_t, \vx_t)$。由于 $k_O(\vx, \vx) = \sum_{i > m}\lambda_i\varphi_i(\vx)^2 \ge 0$，每一项都满足 $k_P(\vx, \vx) = k(\vx, \vx) - k_O(\vx, \vx) \le k(\vx, \vx) \le \bar k$，因此 $\sum_j g_j \le \sigma_n^{-2}T\bar k$。
4. **其余方向**。由于 $\log(1 + g) \le g$，有 $\log\det(\mI + \sigma_n^{-2}\mK_O) \le \sigma_n^{-2}\tr\mK_O = \sigma_n^{-2}\sum_t k_O(\vx_t, \vx_t)$，而 $k_O(\vx, \vx) = \sum_{i > m}\lambda_i\varphi_i(\vx)^2 \le \delta_m$。
5. 把第 3、4 步的结果相加，除以 2，再对 $A$ 取最大值。
:::

### 多项式衰减与指数衰减 {#sec-ka-rates}

最佳的 $m$ 使 @eq-ka-gamma-bound 的两项相互平衡：一项是保留的方向数乘以 $\log T$，另一项是 $T$ 乘以其后的特征值总量。这一平衡只取决于特征值下降的快慢。

::: {.derivation title="两种衰减给出的速率"}
**多项式衰减**：$\lambda_i \le C i^{-\beta}$，其中 $\beta > 1$。

1. $\sum_{i > m} i^{-\beta} \le \int_m^\infty u^{-\beta}\,\dd u = m^{1-\beta}/(\beta - 1)$，因此第二项为 $T m^{1-\beta}$ 量级，第一项为 $m\log T$ 量级。
2. 当 $m \approx (T/\log T)^{1/\beta}$ 时两者相当，此时都为 $T^{1/\beta}(\log T)^{1 - 1/\beta}$ 量级。

对 $d$ 维中 $\nu > 1/2$ 的 Matérn 核，$\lambda_i = O(i^{-(2\nu + d)/d})$[@santin2016approximation; @vakili2021information, 注 2]，因此 $1/\beta = d/(2\nu + d)$，$\gamma_T = O\big(T^{d/(2\nu + d)}(\log T)^{2\nu/(2\nu + d)}\big)$。

**指数衰减**：一维中 $\lambda_i \le Ce^{-ci}$。

3. $\sum_{i > m}e^{-ci} \le e^{-cm}/(1 - e^{-c})$。
4. 取 $m = \lceil(\log T)/c\rceil$，则 $Te^{-cm} \le 1$，因此第二项以常数为界，第一项为 $(\log T)^2/c$ 量级。

对 $d$ 维中的径向基函数核，$\lambda_i = O(e^{-c\,i^{1/d}})$[@belkin2018approximation; @vakili2021information, 注 2]，取 $(\log T)^d$ 量级的 $m$，得 $\gamma_T = O\big((\log T)^{d+1}\big)$[@vakili2021information, 推论 1]。
:::

这些正是 @sec-info-gamma 与 @tbl-regret-gamma 中的速率：Matérn 核的速率出自 @vakili2021information，适用于 $\nu > 1/2$；径向基函数核的速率最早由 @srinivas2010gaussian 证明，用同一定理也可以重新得到。由 $d$ 个特征构成的核函数（例如 $d$ 维中的线性核）至多有 $d$ 个非零特征值，尾部在 $m = d$ 处消失，界给出 $O(d\log T)$（@exr-ka-linear）。概括地说：随着 $T$ 增大，多项式衰减的频谱约有 $T^{d/(2\nu + d)}$ 个方向可以分辨，指数衰减的频谱只有约 $(\log T)^d$ 个，每个已分辨的方向花费一个对数。光滑度降低指数，维度提高指数。

这些速率是渐近的。@fig-ka-infogain 精确计算了均匀分布在 $[0, 1]$ 上的 $T$ 次评估所获得的信息。$\gamma_T$ 取的是最好的设计，因此这一信息量是 $\gamma_T$ 的下界。

```{figure}
//| figure: ka-infogain
//| label: fig-ka-infogain
//| fig-cap: "在 $[0, 1]$ 中等距的输入上做 $T$ 次评估所获得的关于 $f$ 的信息 $\tfrac12\log\det(\mI + \sigma_n^{-2}\mK_T)$，单位为奈特，四个单位幅度的核函数，双对数坐标。该值为精确计算，是 $\gamma_T$ 的下界。虚线为单次评估信息量的 $T$ 倍，即 $T$ 次全新的评估所能获得的信息。$T = 100$ 至 $1000$ 之间的点线段，斜率为 $1/(2\nu + 1)$，即一维中各 Matérn 核的特征值衰减对大 $T$ 预言的斜率。读数给出各核函数在所选 $T$ 处的信息量及其局部斜率，即双对数坐标中 $T/2$ 与 $2T$ 之间的斜率（在右端为 $T/2$ 与 $T$ 之间）。"
```

**查看默认设置**。$T = 100$、长度尺度 0.1、噪声标准差 0.1 时，径向基函数核获得 36.5 奈特，Matérn 5/2 为 53.4，Matérn 3/2 为 70.2，Matérn 1/2 为 150.4；一百次全新的评估则为 230.8。各曲线在 $T = 10$ 至 $T = 22$ 之间离开虚线（降到虚线值的 90% 以下），此时各次评估开始相互重叠。

**把“评估次数 T”调到 1000**。局部斜率为 0.16、0.22、0.28 与 0.58，仍高于大 $T$ 时的极限值：径向基函数核为 0，各 Matérn 核为 $1/6$、$1/4$ 与 $1/2$。在一维中，一千次评估还远未进入渐近阶段。

**把“长度尺度 ℓ”加长到 0.3**。$T = 1000$ 时，四个值降至 25.2、41.5、63.9 与 398.0 奈特。速率与长度尺度无关，速率前面的常数则与之有关：长度尺度决定有多少个特征值较大。

### 从信息增益到遗憾 {#sec-ka-regret}

GP-UCB 规则在后验均值加若干倍后验标准差最大处评估（@eq-regret-gpucb）。@tbl-ka-rates 把上述速率代入 GP-UCB 的遗憾界，并注明每个结果所假定的设定。

::: {.table #tbl-ka-rates title="d 维紧定义域上，从特征值衰减到各项速率。遗憾上界针对先验样本上的 GP-UCB，置信参数按 log T 增长；在连续定义域上，它要求径向基函数核或 ν > 2 的 Matérn 核。下界针对 RKHS 范数有界的函数上的任意算法。"}
| 核函数 | 特征值 $\lambda_i$ | $\gamma_T$ | 遗憾上界（贝叶斯） | 下界（固定函数） |
|---|---|---|---|---|
| 径向基函数核 | $O(e^{-c\,i^{1/d}})$ [@belkin2018approximation] | $O\big((\log T)^{d+1}\big)$ | $O\big(\sqrt{T}(\log T)^{d/2 + 1}\big)$ | $\Omega\big(\sqrt{T(\log T)^{d/2}}\big)$ |
| Matérn 核，$\nu > 1/2$ | $O(i^{-(2\nu + d)/d})$ [@santin2016approximation] | $O\big(T^{\frac{d}{2\nu + d}}(\log T)^{\frac{2\nu}{2\nu + d}}\big)$ | $O\big(T^{\frac{\nu + d}{2\nu + d}}(\log T)^{\frac{4\nu + d}{4\nu + 2d}}\big)$ | $\Omega\big(T^{\frac{\nu + d}{2\nu + d}}\big)$ |
:::

$\gamma_T$ 与遗憾两列取自 @vakili2021information；遗憾即把 $\gamma_T$ 代入 $\sqrt{T\log T\,\gamma_T}$，对径向基函数核就是 @sec-regret-gpucb-bound 中的 $\sqrt{T}(\log T)^{(d+2)/2}$。下界取自 @scarlett2017lower。在连续定义域上，贝叶斯界要求样本路径足够光滑，以便离散化；径向基函数核与 $\nu > 2$ 的 Matérn 核满足这一要求（@sec-regret-other-settings、@sec-ka-regularity）。

对 RKHS 中的固定函数，GP-UCB 的频率派分析给出相差对数因子意义下 $\gamma_T\sqrt{T}$ 量级的遗憾[@chowdhury2017kernelized]；代入 Matérn 核的速率，其指数为 $\tfrac12 + d/(2\nu + d)$，一旦 $d \ge 2\nu$ 便达到 1（推断，与 @sec-regret-bounds-not-say 相同）。这一上界与下界之间相差的因子 $\sqrt{\gamma_T}$，是 @sec-dueling-rates 与 @sec-theory-rates 讨论的主题。

## 高斯过程的精确定义 {#sec-ka-process}

@def-fs-gp 把高斯过程定义为一族随机变量，其中任意有限个服从联合高斯分布；@sec-gp-definition 把它理解为一个接口，回答关于有限个输入的查询。遗憾界的要求更多：它们要取 $f$ 在连续定义域上的最大值，同时涉及无穷多个输入。本节说明高斯过程是什么样的数学对象，以及这类问题何时有答案。

### 以输入为指标的随机变量 {#sec-ka-stochastic-process}

随机变量是随机试验结果 $\omega$ 的函数（@sec-prob-random-variables；这里的 $\omega$ 表示结果，而非频率）。定义域 $\X$ 上的**随机过程**（stochastic process）是一个函数 $f(\vx, \omega)$，对每个固定的 $\vx$，$f(\vx, \cdot)$ 都是随机变量。反过来固定结果，得到**样本路径**（sample path）$\vx \mapsto f(\vx, \omega)$。高斯过程是这样一种随机过程：它在任意有限个输入处的取值服从联合高斯分布[@dacosta2026sample, 定义 2.1]。

有限个输入处的联合分布称为**有限维分布**（finite-dimensional distributions），它们必须在两方面相互一致：以另一种顺序列出输入，分布随之作相应的置换；去掉一个输入，得到其余输入的边际分布。@sec-gp-definition 已对任意均值函数与核函数验证了第二条。

::: {.theorem #thm-ka-kolmogorov title="Kolmogorov 扩展定理"}
在上述两方面一致的任何有限维分布族，都属于 $\X$ 上的某个随机过程；对涉及可数个输入的每个事件，该分布族决定了这个过程赋予它的概率[@kolmogorov1933grundbegriffe, 第 III 章第 4 节]。
:::

@sec-gp-definition 中“深入一步”提示框提到的正是这一定理。对高斯过程而言，它表明均值函数与半正定核在任意定义域上都定义了一个随机函数[@dacosta2026sample, 第 2 节]。证明需要测度论，本书从略。

### 有限维分布不能决定的性质 {#sec-ka-versions}

路径是否连续、在 $[0, 1]$ 上的最大值是多少，这些问题涉及不可数个输入，有限维分布无法决定。设 $U$ 在 $[0, 1]$ 上均匀分布，对每个 $x$ 令 $f(x) = 0$；若 $x = U$ 则令 $g(x) = 1$，否则令其为 0。在任意固定的 $x$ 处，除非 $U$ 恰好落在 $x$ 上（其概率为零），否则 $g(x) = 0$，因此 $f$ 与 $g$ 的有限维分布相同。然而 $f$ 的每条路径都连续且最大值为 0，$g$ 的每条路径都有跳跃，且最大值为 1。

在每个固定输入处以概率 1 相等的两个过程，互称为对方的**版本**（versions）[@kanagawa2018gaussian, 定义 4.8]；关于样本路径的论断，意思是存在某个版本具有这样的路径。有界闭定义域上的高斯过程若有连续版本，所指的就是这个版本；其最大值 $f^\star$ 与最大值点 $\vx^\star$ 都存在，因为这类定义域上的连续函数必能取到最大值。熵搜索根据评估能揭示多少关于 $\vx^\star$ 位置的信息来选择评估（@sec-entropy-search），它以此为前提。

### 样本路径的正则性 {#sec-ka-regularity}

连续版本是否存在，取决于核函数在距离趋于零时的性质。对均值为零、核函数平稳的过程，两个相邻取值之差的平方的期望为

$$
\E\big[(f(\vx + \mathbf{h}) - f(\vx))^2\big] = 2\big(k(\mathbf{0}) - k(\mathbf{h})\big).
$$

若对某个介于 0 与 1 之间的 $\eta$，它按 $\lVert\mathbf{h}\rVert^{2\eta}$ 缩小，则该过程有一个版本，其路径连续，并且对每个阶数 $\eta' < \eta$ 都是 Hölder 连续的，即在每一点附近有 $\lvert f(\vx) - f(\vx')\rvert \le C\lVert\vx - \vx'\rVert^{\eta'}$[@dacosta2026sample, 定理 3.1]。对高斯过程而言，这是 Kolmogorov 连续性定理的一种精确形式，其证明也以该定理为基础。

对 Matérn 1/2 核，$k(0) - k(h) = 1 - e^{-\lvert h\rvert/\ell} \le \lvert h\rvert/\ell$，因此 $2\eta = 1$：路径连续，与 Brown 运动（随机游走在连续时间中的极限）一样，对低于 1/2 的每个阶数都是 Hölder 连续的，但仅此而已，因而不可微[@dacosta2026sample, 注 3.4]。把这一判据用于导数过程（其核函数为 $-k''$），可以逐级向上推：$\nu$ 不是整数时，Matérn 核的路径（对适当的版本）恰好 $\lfloor\nu\rfloor$ 次连续可微，不再更多，因此 $\nu = 1/2, 3/2, 5/2$ 分别给出 0、1、2 阶连续导数[@dacosta2026sample, 推论 1.2 与命题 3.1]。对这样的 $\nu$，这些关于样本路径的结论与 @tbl-kern-family 中均方意义下的阶梯一致，后者同样给出低于 $\nu$ 的每个整数阶导数，不再更多。径向基函数核的路径具有任意阶导数[@dacosta2026sample, 注 3.5]。

这种一致是定理，而非定义。均方可微性（@sec-fs-smoothness）涉及差商的二阶矩，仅由 $k''(0)$ 决定[@rasmussen2006gaussian, 第 4.1.1 节]；样本路径的可微性则涉及抽取出的每一个函数。@sec-regret-other-settings 中连续定义域上的遗憾界需要后者，而且要求还更高一些：其证明把定义域离散化，要求相邻的取值以高概率彼此接近，径向基函数核与 $\nu > 2$ 的 Matérn 核满足这一要求[@srinivas2010gaussian]。

总之，对 Matérn 核，样本有 $\lfloor\nu\rfloor$ 阶连续导数，Sobolev 光滑度略低于 $\nu$；RKHS 中的函数（包括后验均值）的 Sobolev 光滑度则为 $\nu + d/2$（@sec-ka-samples）。每个遗憾界都假定了其中一种光滑度。

## 习题 {#sec-ka-exercises}

::: {.exercise #exr-ka-narrow-bump}
设 $k$ 是 $\R$ 上单位幅度、长度尺度为 $\ell$ 的径向基函数核。（a）证明 $\lVert k(\cdot, a) - k(\cdot, b)\rVert_k^2 = 2 - 2k(a, b)$，并计算 $\lvert a - b\rvert = \ell$ 与 $3\ell$ 时的值。（b）对 $\R$ 上的平稳核，$f \in \mathcal{H}_k$ 当且仅当 $\int \lvert\hat f(\omega)\rvert^2/s(\omega)\,\dd\omega$ 有限，其中 $\hat f$ 是 $f$ 的 Fourier 变换，$s$ 是谱密度[@wendland2004scattered, 定理 10.12; @kanagawa2018gaussian, 定理 2.4]。对鼓包 $f(x) = e^{-x^2/2w^2}$，已知 $\lvert\hat f(\omega)\rvert^2 \propto e^{-w^2\omega^2}$，求使 $f \in \mathcal{H}_k$ 的宽度 $w$。

::: {.solution}
（a）由 @eq-ka-inner，取系数 $1$ 与 $-1$，范数平方为 $k(a,a) - 2k(a,b) + k(b,b) = 2 - 2k(a,b)$。距离为 $\ell$ 时，$k = e^{-1/2} = 0.607$，范数平方为 $0.787$；距离为 $3\ell$ 时，$k = e^{-9/2} = 0.011$，范数平方为 $1.978$，接近 2，即两个互不重叠的鼓包对应的值。（b）径向基函数核的谱密度正比于 $e^{-\ell^2\omega^2/2}$，因此被积函数正比于 $e^{-(w^2 - \ell^2/2)\omega^2}$，积分有限当且仅当 $w > \ell/\sqrt{2} \approx 0.71\,\ell$。更窄的鼓包范数为无穷大：核函数赋予其高频成分的先验方差太小，不足以抵偿这些高频。核函数自身的鼓包宽度为 $\ell$，理应满足条件，也确实满足。
:::
:::

::: {.exercise #exr-ka-gauss-spectrum}
设 $\omega \sim \N(0, 1/\ell^2)$，$g(r) = \E[\cos(\omega r)]$。（a）用分部积分证明：对增长不太快的可微函数 $h$，有 $\E[\omega h(\omega)] = \ell^{-2}\E[h'(\omega)]$。（b）利用这一结果证明 $g'(r) = -(r/\ell^2)g(r)$，并由此得出 $g(r) = e^{-r^2/2\ell^2}$。

::: {.solution}
（a）密度 $p(\omega) = c\,e^{-\ell^2\omega^2/2}$ 满足 $p'(\omega) = -\ell^2\omega\,p(\omega)$，因此分部积分得 $\E[\omega h(\omega)] = -\ell^{-2}\int h\,p'\,\dd\omega = \ell^{-2}\int h'\,p\,\dd\omega$；边界项为零，因为 $p$ 的衰减快于 $h$ 的增长。（b）$g'(r) = -\E[\omega\sin(\omega r)]$。取 $h(\omega) = \sin(\omega r)$，则 $h' = r\cos(\omega r)$，因此 $g'(r) = -(r/\ell^2)g(r)$。结合 $g(0) = 1$，唯一解为 $e^{-r^2/2\ell^2}$，即径向基函数核：标准差为 $1/\ell$ 的高斯频率给出长度尺度 $\ell$，长度尺度短就需要高频。
:::
:::

::: {.exercise #exr-ka-rff}
单个随机特征以 $\cos(\omega r)$（$\omega \sim p$）估计单位幅度核函数在距离 $r$ 处的值。（a）证明其方差为 $\tfrac12\big(1 + k(2r)\big) - k(r)^2$。（b）求 $r = 0$ 时的值，以及 $r \to \infty$ 时的极限。（c）对径向基函数核，证明该方差不超过 $\tfrac12$，从而 $M$ 个特征的平均值的标准差至多为 $1/\sqrt{2M}$。

::: {.solution}
（a）$\cos^2\theta = \tfrac12(1 + \cos 2\theta)$，由 @eq-ka-bochner 得 $\E[\cos^2(\omega r)] = \tfrac12(1 + k(2r))$，再减去均值的平方 $k(r)^2$。（b）$r = 0$ 时方差为 $\tfrac12 \cdot 2 - 1 = 0$，因为每个特征都恰好给出 $k(0) = 1$；$r \to \infty$ 时趋于 $\tfrac12$。（c）对径向基函数核有 $k(2r) = k(r)^4$，因此当 $0 \le k \le 1$ 时，方差满足 $\tfrac12 - k^2(1 - \tfrac12k^2) \le \tfrac12$。$M$ 项相互独立，因此平均值的方差至多为 $1/(2M)$，即 @fig-ka-bochner 中的对照值。
:::
:::

::: {.exercise #exr-ka-linear}
（a）$\R^d$ 单位球上的线性核 $k(\vx, \vx') = \vx^\T\vx'$ 至多有 $d$ 个非零特征值。取 $m = d$，用 @thm-ka-gamma 证明 $\gamma_T \le \frac d2\log\big(1 + T/(\sigma_n^2 d)\big)$。（b）与 @exr-regret-independent 中 $K$ 条独立的臂比较，后者有 $\gamma_T = \frac K2\log\big(1 + T/(K\sigma_n^2)\big)$，并解释两者形式相同的原因。

::: {.solution}
（a）取 $m = d$，尾部 $\delta_d$ 为零；在单位球上 $\lvert k\rvert \le 1$，因此 $\bar k = 1$，@eq-ka-gamma-bound 只剩第一项：$\gamma_T \le \frac d2\log(1 + T/(\sigma_n^2 d))$，即 $O(d\log T)$，也就是 @sec-info-gamma 中线性核的速率。界 $\psi$ 不出现，因为证明的第 4 步用不到。（b）两个核函数都只有固定数目的方向（$d$ 或 $K$），承载全部方差。每个方向都可以反复测量，但只贡献一个对数，最好的设计把 $T$ 次评估均匀分配到这些方向上。有无穷多个特征值的核函数，表现得像有 $m$ 个方向的核函数，而 $m$ 随 $T$ 增长，其快慢取决于特征值衰减所允许的程度。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @kanagawa2018gaussian 是一篇以预印本形式发布的长篇综述，把高斯过程与 RKHS 两种观点并列讨论，内容包括 Mercer 定理、Karhunen-Loève 展开、样本路径的零一律，以及 Matérn 核对应的 Sobolev 空间。
- @rasmussen2006gaussian 第 4 章讨论平稳核、Bochner 定理与特征函数分析；第 6 章介绍 RKHS 及其与正则化的联系。
- @aronszajn1950theory 奠定了再生核理论；@steinwart2008support 第 4 章与 @berlinet2004reproducing 给出了现代的论述，后者着眼于概率论。
- @wendland2004scattered 从逼近论出发，讨论正定函数（第 6 章）与核函数的原生空间（第 10 章）。
- @rahimi2007random 提出了随机 Fourier 特征，并给出了本章引用的一致收敛界。
- @vakili2021information 由特征值衰减推导信息增益的速率；@thm-ka-gamma 即其定理 3。
- @dacosta2026sample 给出了样本路径具有给定光滑度时核函数所需满足的充分必要条件，并以 Matérn 核与径向基函数核为例。
