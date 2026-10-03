---
status: done
synopsis: "本书使用的符号，按主题分组，每个符号都注明引入它的小节。"
---

# 记号 {#sec-notation}

本书通篇使用同一套记号，本附录将其汇总。每个条目都注明引入该符号的小节；在那里，符号先以文字解释，再用于公式。文献中有多种约定并存时，条目注明本书采用哪一种。

以下排版规则全书通用。标量用斜体（$x$、$\sigma$），向量用粗体小写（$\vx$、$\vy$），矩阵用粗体大写（$\mK$、$\mL$）。转置记作 $^\T$。输入位于定义域 $\X$ 中，经缩放后通常为单位立方体 $[0, 1]^d$，其中 $d$ 是输入个数；下标遍历各个输入时（例如每个输入各有一个长度尺度），记作 $j = 1, \dots, d$。有限的候选输入集合同样记作 $\X$，含 $|\X|$ 个元素，即使书中引述的论文将其记作 $D$。全书约定越大越好：本书求最大值，天然需要最小化的函数（例如错误率）则取其相反数。

## 集合、向量与矩阵 {#sec-notation-linear}

::: {.table #tbl-notation-linear title="线性代数。"}
| 符号 | 含义 | 引入位置 |
|---|---|---|
| $\R$, $\R^d$ | 实数；由 $d$ 个实数组成的向量 | @sec-vectors |
| $d$ | 输入个数（$\X$ 的维度） | @sec-usual-tools、@sec-vectors |
| $\vx$, $x_i$ | 向量及其第 $i$ 个分量 | @sec-vectors |
| $\vx^\T$, $\mA^\T$ | 向量与矩阵的转置 | @sec-vectors、@sec-linalg-transpose |
| $\mI$ | 单位矩阵 | @sec-linalg-composition |
| $\mA^{-1}$ | $\mA$ 的逆矩阵（通过解方程计算，从不显式构造） | @sec-linalg-composition |
| $\mL$ | 下三角 Cholesky 因子，$\mL\mL^\T = \mA$ | @sec-linalg-cholesky-def |
| $\mA \backslash \mathbf{b}$ | $\mA\mathbf{z} = \mathbf{b}$ 的解 $\mathbf{z}$ | @sec-gp-computation |
| $\det \mA$, $\log\det\mA$ | 行列式及其对数（由 Cholesky 因子的对角元计算） | @sec-determinants、@sec-linalg-logdet |
| $\tr \mA$ | 迹，即对角元之和 | @sec-info-kl-examples、@sec-kern-gradient |
:::

## 概率 {#sec-notation-probability}

::: {.table #tbl-notation-probability title="概率。"}
| 符号 | 含义 | 引入位置 |
|---|---|---|
| $\Prob(A)$ | 事件的概率 | @sec-prob-axioms |
| $p(x)$, $p(x \given y)$ | 密度函数或质量函数；以 $y$ 为条件 | @sec-prob-random-variables |
| $X \sim p$ | $X$ 服从分布 $p$ | @sec-prob-random-variables |
| $\D$ | 观测数据 | @sec-bayes-rule |
| $\E[X]$, $\E_n[\cdot]$ | 期望；$n$ 次观测后关于后验的期望 | @sec-prob-expectation-def、@sec-acq-definition |
| $\Var[X]$, $\Cov[X, Y]$ | 方差；协方差 | @sec-prob-variance、@sec-prob-covariance |
| $\N(\mu, \sigma^2)$, $\N(\vmu, \mSigma)$ | 以均值和方差为参数的高斯分布；以均值向量和协方差矩阵为参数的高斯分布 | @sec-gaussian-1d、@sec-gaussian-nd |
| $\phi(z)$, $\Phi(z)$ | 标准正态分布的密度函数与累积分布函数 | @sec-gauss-standard |
| $\operatorname{sigmoid}(z)$ | 逻辑函数 $1/(1 + e^{-z})$，即 Bradley-Terry 链接 | @sec-bradley-terry |
| $H(p)$, $H(X)$ | 分布或随机变量的熵 | @sec-info-entropy-def |
| $\KL(p \,\Vert\, q)$ | Kullback-Leibler 散度 | @sec-info-kl-def |
| $I(X; Y)$ | 互信息 | @sec-mutual-information |
| $\delta$ | 失效概率：以高概率成立的论断，其成立的概率至少为 $1 - \delta$ | @sec-regret-concentration |
| $R$-次高斯 | 均值为零、且对所有 $\lambda$ 满足 $\E[e^{\lambda Z}] \le e^{\lambda^2 R^2/2}$ 的随机变量 $Z$；这里的 $R$ 不是遗憾 $R_T$ | @sec-regret-concentration |
:::

与多数统计学教材一样，本书 $\N(\mu, \sigma^2)$ 中的第二个参数是方差，需要时另行给出标准差 $\sigma$。

## 高斯过程 {#sec-notation-gp}

::: {.table #tbl-notation-gp title="高斯过程。"}
| 符号 | 含义 | 引入位置 |
|---|---|---|
| $f$ | 未知的目标函数（或潜在效用） | @sec-cost-of-evaluation |
| $\X$ | 输入的定义域 | @sec-cost-of-evaluation |
| $\lvert\X\rvert$ | 定义域为有限集时的候选个数 | @sec-ucb |
| $\GP(m, k)$ | 均值函数为 $m$、核函数为 $k$ 的高斯过程 | @sec-gp-definition |
| $k(\vx, \vx')$ | 核函数（协方差函数） | @sec-linalg-inner-products、@sec-fs-prior-values |
| $\ell$, $\ell_j$ | 长度尺度；每个输入 $j$ 各有一个长度尺度（自动相关性确定，ARD） | @sec-linalg-inner-products、@sec-ard |
| $\sigma_f^2$ | 幅度的平方，对平稳核而言等于 $k(\vx, \vx)$ | @sec-fs-rbf-limit |
| $\sigma_n^2$ | 观测噪声方差 | @sec-prob-total-variance、@sec-gp-noise |
| $X$, $\vy$ | 已观测的输入与观测值 | @sec-gp-conditioning |
| $\mK$ | 已观测输入的核矩阵，$[\mK]_{ij} = k(\vx_i, \vx_j)$ | @sec-gp-definition、@sec-gp-conditioning |
| $\vk(\vx)$ | $\vx$ 与各已观测输入之间的协方差 | @sec-gp-conditioning |
| $\mu(\vx)$, $\mu_n(\vx)$ | 后验均值（$n$ 次观测后） | @sec-gp-conditioning、@sec-bo-algorithm |
| $\sigma^2(\vx)$, $\sigma_n^2(\vx)$ | 潜在值的后验方差（$n$ 次观测后） | @sec-gp-conditioning、@sec-bo-algorithm |
| $\bm{\alpha}$ | 后验均值中的权重 $(\mK + \sigma_n^2\mI)^{-1}\vy$ | @sec-linalg-why-not-invert、@sec-gp-bumps |
| $M$ | 线性模型的特征个数；对随机 Fourier 特征，指抽取的频率个数 | @sec-weights-to-functions、@sec-ka-rff |
| $\mathcal{H}_k$, $\langle f, g\rangle_k$, $\lVert f\rVert_k$ | $k$ 的再生核 Hilbert 空间（RKHS）及其内积与范数 | @sec-ka-rkhs |
| $B$ | RKHS 范数的上界；有的论文界定 $\lVert f\rVert_k$，有的界定 $\lVert f\rVert_k^2$ | @sec-ka-norm-bound |
| $\mathcal{K}$, $q(\vx)$ | 核函数的积分算子，以及积分所用的权重密度 | @sec-ka-operator |
| $\lambda_i$, $\varphi_i$ | $\mathcal{K}$ 的特征值与特征函数（Mercer 定理） | @sec-ka-mercer |
| $s(\boldsymbol{\omega})$, $p(\boldsymbol{\omega})$ | 平稳核的谱密度；归一化为概率密度后的谱密度 | @sec-ka-bochner |
:::

有一处记号冲突需要留意。不带自变量的 $\sigma_n^2$ 表示噪声方差，沿用 @rasmussen2006gaussian的写法，下标代表噪声（noise）。$\sigma_n(\vx)$ 总是带自变量书写，表示 $n$ 次观测后的后验标准差，沿用 @srinivas2010gaussian的写法。两者靠自变量 $(\vx)$ 区分。两者出现在同一公式中时（@sec-knowledge-gradient），噪声方差记作 $\sigma_\varepsilon^2$。在 @sec-kernel-analysis 中，带下标的 $\lambda_i$ 表示特征值，$q$ 表示权重密度；在其他章节中，$\lambda$ 表示逆 Mills 比或失误率，$q$ 表示一次查询中的选项数。

## 优化与偏好 {#sec-notation-bo}

::: {.table #tbl-notation-bo title="优化与偏好。"}
| 符号 | 含义 | 引入位置 |
|---|---|---|
| $\vx^\star$, $f^\star$ | 某个最大值点与最大值 | @sec-info-eig-limits、@sec-bo-problem |
| $f^*_n$ | 当前最优值，即 $n$ 次评估后观测到的最好值 | @sec-pi |
| $a_n(\vx)$ | $n$ 次观测后的采集函数 | @sec-bo-algorithm |
| $\PI$, $\EI$, $\UCB$ | 改进概率、期望改进、上置信界 | @sec-pi、@sec-ei、@sec-ucb |
| $\beta$, $\beta_t$ | 上置信界 $\mu_n(\vx) + \beta^{1/2}\sigma_n(\vx)$ 中的探索权重 | @sec-loop-first-acq、@sec-ucb |
| $\xi$ | 改进概率与期望改进中的改进裕量（在 @sec-expected-information-gain中另指对实验的一种选择） | @sec-pi |
| $r_t$, $R_T$ | 瞬时遗憾；$T$ 轮的累积遗憾 | @sec-regret-definitions |
| $\gamma_T$ | 核函数在 $T$ 次观测后的最大信息增益 | @sec-info-gamma |
| $\vx \succ \vx'$ | $\vx$ 优于 $\vx'$ | @sec-thurstone |
| $g$ 或 $f$ | 人的潜在效用（与高斯过程共用同一套机制时，本书记作 $f$） | @sec-pref-model |
| $\sigma$（链接函数中） | 人评价单个选项时的噪声尺度 | @sec-thurstone |
| $\tau$ | 逻辑链接的尺度，$\operatorname{sigmoid}\big((g_i - g_j)/\tau\big)$ | @sec-bradley-terry |
| $\lambda(z)$ | 逆 Mills 比 $\phi(z)/\Phi(z)$，只在定义它的章节中使用 | @sec-ep、@sec-pref-laplace |
| $\lambda_{\text{lapse}}$ | 失误率，即回答属于随机失误的概率 | @sec-query-confidence |
| $\mW$ | 对数似然的负 Hessian 矩阵；对成对回答而言为加权图 Laplace 矩阵 | @sec-laplace、@sec-pref-laplace |
| $\EUBO(\vx_1, \vx_2)$ | 最优选项期望效用 | @sec-eubo |
| $q$ | 一次查询展示的选项数（qEUBO） | @sec-eubo |
:::

遗憾界用 $\tilde O(\cdot)$ 表述，该记号隐去对数因子；书中的每个速率都连同其假设与遗憾单位一并给出（@sec-theory-rates）。
