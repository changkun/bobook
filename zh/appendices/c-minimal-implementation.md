---
status: done
synopsis: "用约 180 行 NumPy 代码实现贝叶斯优化与偏好贝叶斯优化，与书中的图和记号一一对应；附三维示例，以及同样的循环在 BoTorch 中的实现。"
sources: ["Rasmussen and Williams 2006, ch. 2 and 3", "Chu and Ghahramani 2005", "Lin et al. 2022", "BoTorch 0.18.1 source"]
---

# 最小实现 {#sec-minimal-implementation}

本书前四部分的所有方法都可以写进一个短小的程序，本附录给出这个程序：高斯过程回归、期望改进、一维与三维中的优化循环、Laplace 偏好模型、EUBO，以及偏好贝叶斯优化的循环，共约 180 行 NumPy 代码。代码沿用本书的记号和贯穿全书的示例目标函数，每个函数都可以与其实现的公式对照阅读，每个循环也可以与演示它的动画图对照阅读。

这些代码以可读性为先：每一步都从头重新拟合模型，在网格或随机候选集上搜索而不使用梯度，并固定了大多数超参数。最后一节给出同样两个循环的 BoTorch 写法，BoTorch 会妥善处理上述各点，该节也说明了其中的变化。本附录的目标函数都是测试函数；@sec-part-cases 的案例研究在实测问题上运行同样的循环。

下面五段 NumPy 代码清单构成一个完整脚本：按顺序拼接即可原样运行，除 NumPy 外没有其他依赖。正文引用的输出来自在 Python 3.14 上用 NumPy 2.5.3 运行该脚本的结果；结果依赖于随机数生成器，因此其他版本的最后几位数字可能不同。

## 高斯过程回归 {#sec-impl-gp}

第一段清单即 @alg-gp-regression。`rbf` 构造核矩阵；`gp_posterior` 用 Cholesky 分解计算 @eq-gp-noisy 的后验均值与方差；`log_marginal` 计算 @sec-marginal-likelihood 的对数边际似然，后面的清单用它选择长度尺度。输入是形状为 $(n, d)$ 的数组；辅助函数 `as2d` 使由 $n$ 个数构成的普通数组表示一维中的 $n$ 个点，同一组函数因此可以同时用于两个例子。

::: {.code title="NumPy：高斯过程回归"}
```python
import numpy as np
from math import erf, sqrt, pi

def as2d(x):
    """Inputs as an (n, d) array; a 1-D array is n points in one dimension."""
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x

def rbf(A, B, ell=0.12):
    """RBF kernel matrix k(a, b) = exp(-|a - b|^2 / (2 ell^2)), unit amplitude."""
    A, B = as2d(A) / ell, as2d(B) / ell
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-0.5 * np.maximum(d2, 0.0))

def gp_posterior(X, y, Xs, ell=0.12, noise=1e-4):
    """Posterior mean and variance at Xs (Algorithm 2.1 of Rasmussen and Williams)."""
    L = np.linalg.cholesky(rbf(X, X, ell) + noise * np.eye(len(y)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = rbf(X, Xs, ell)                       # n x m
    v = np.linalg.solve(L, Ks)
    return Ks.T @ alpha, np.maximum(1.0 - (v**2).sum(0), 1e-12)

def log_marginal(X, y, ell, noise=1e-4):
    """Log marginal likelihood, for choosing the lengthscale."""
    L = np.linalg.cholesky(rbf(X, X, ell) + noise * np.eye(len(y)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return -0.5 * y @ alpha - np.log(np.diag(L)).sum() - 0.5 * len(y) * np.log(2 * pi)
```
:::

有三处细节需要说明。平方距离按 $\lVert a - b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2a^\T b$ 计算，速度快，但在浮点运算中可能略小于零，因此使用了 `np.maximum`。出于同样的原因，方差在一个极小的正值处截断：$-10^{-17}$ 的方差开平方后会得到 `nan`。另外，核函数的幅度为 1，方差中的 `1.0` 即代表 $k(\vx, \vx)$；为弥补这一点，代码在拟合前对输出做了标准化（@sec-gp-pitfalls）。

## 期望改进与优化循环 {#sec-impl-bo}

第二段清单加入采集函数与循环。`ei` 是 @sec-ei 中的闭式解 $\EI_n(\vx) = (\mu_n(\vx) - f^*_n)\,\Phi(z) + \sigma_n(\vx)\,\phi(z)$，其中 $z = (\mu_n(\vx) - f^*_n)/\sigma_n(\vx)$，$f^*_n$ 是迄今观测到的最优值（即 @sec-pi 中的当前最优值）。`Phi` 与 `phi` 分别是标准正态分布的累积分布函数与密度；NumPy 没有误差函数，所以清单将 Python `math` 模块中的误差函数向量化。`bo_loop` 在示例目标函数上实现 @sec-bo-algorithm 的循环：标准化、拟合、在 501 个候选点的网格上最大化期望改进、评估，如此反复。

::: {.code title="NumPy：期望改进与一维中的循环"}
```python
_erf = np.vectorize(erf)
def Phi(z): return 0.5 * (1.0 + _erf(np.asarray(z) / sqrt(2.0)))
def phi(z): return np.exp(-0.5 * np.asarray(z) ** 2) / sqrt(2.0 * pi)

def ei(mean, var, best):
    """Expected improvement over `best`, in closed form."""
    sd = np.sqrt(var)
    z = (mean - best) / sd
    return (mean - best) * Phi(z) + sd * phi(z)

def f(x):  # 本书的示例目标函数
    return (0.62 * np.exp(-(x - 0.25) ** 2 / (2 * 0.1**2))
            + np.exp(-(x - 0.73) ** 2 / (2 * 0.055**2))
            + 0.1 * np.sin(11 * x + 0.6) - 0.35 * x)

def bo_loop(f, n_init=2, n_iter=10, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 501)
    X = rng.uniform(0, 1, n_init)
    Y = f(X)
    for _ in range(n_iter):
        y = (Y - Y.mean()) / (Y.std() + 1e-9)    # 标准化输出
        mean, var = gp_posterior(X, y, grid, ell=0.08, noise=1e-6)
        x_next = grid[np.argmax(ei(mean, var, y.max()))]
        X, Y = np.append(X, x_next), np.append(Y, f(x_next))
    return X, Y

X, Y = bo_loop(f)
print(f"1-D: recommend x = {X[np.argmax(Y)]:.3f}, f = {Y.max():.3f} "
      f"(best on [0, 1]: f = {f(np.linspace(0, 1, 4001)).max():.3f})")
```
:::

输出为

```text
1-D: recommend x = 0.730, f = 0.816 (best on [0, 1]: f = 0.818)
```

12 次评估即找到了又高又窄的峰。这并非种子选得凑巧：在种子 0 至 19 上，找到的最优值中位数为 0.817，最小值为 0.788，每一次运行都到达了高峰。@fig-loop-animation 以交互方式运行同一循环；在该图中选择期望改进，可以看到模型一旦对宽峰有了把握，循环便离开宽峰。

### 三维情形 {#sec-impl-hartmann}

`gp_posterior` 与 `ei` 都与维度无关。与维度有关的是内层搜索：若每个轴取 501 个点，三维网格将有 1.25 亿个点。@sec-acq-optimization 介绍了替代网格的方法。本清单采用最简单的可行做法：对几千个随机候选点打分，再加上在迄今评估过的最优点周围施加小扰动得到的一团点，从中取最优者。随机候选负责探索，局部点团负责细化：期望改进的峰值常常位于当前最优点附近的一个小区域内，随机候选很容易错过。

测试问题是三维 Hartmann 函数。这是一个具有若干局部最大值的标准基准，此处取其相反数，使数值越大越好；它在单位立方体上的最大值为 3.86278。长度尺度不再手工固定：每一步，循环从五个候选值中选出边际似然最高者。

::: {.code title="NumPy：三维中的循环"}
```python
H_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
H_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470],
                       [1091, 8732, 5547], [381, 5743, 8828]])
H_C = np.array([1.0, 1.2, 3.0, 3.2])

def hartmann3(X):  # [0, 1]^3 上取相反数的 Hartmann-3；最大值 3.86278
    X = as2d(X)
    return (H_C * np.exp(-(H_A * (X[:, None, :] - H_P) ** 2).sum(-1))).sum(-1)

def next_point(X, y, ell, rng, n_global=2000, n_local=1000, step=0.05):
    """Maximize EI over random candidates plus perturbations of the best points."""
    d = X.shape[1]
    top = X[np.argsort(y)[-5:]]
    local = top[rng.integers(0, len(top), n_local)] + step * rng.normal(size=(n_local, d))
    cand = np.clip(np.vstack([rng.uniform(0, 1, (n_global, d)), local]), 0, 1)
    mean, var = gp_posterior(X, y, cand, ell, noise=1e-6)
    return cand[np.argmax(ei(mean, var, y.max()))]

def bo_nd(f, d, n_init=6, n_iter=30, seed=0, ells=(0.1, 0.15, 0.2, 0.3, 0.5)):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_init, d))
    Y = f(X)
    for _ in range(n_iter):
        y = (Y - Y.mean()) / (Y.std() + 1e-9)
        ell = max(ells, key=lambda l: log_marginal(X, y, l, noise=1e-6))
        X = np.vstack([X, next_point(X, y, ell, rng)])
        Y = np.append(Y, f(X[-1:]))
    return X, Y

X3, Y3 = bo_nd(hartmann3, d=3)
R3 = hartmann3(np.random.default_rng(1).uniform(0, 1, (36, 3)))
print(f"3-D: best of 36 evaluations {Y3.max():.3f} at {np.round(X3[np.argmax(Y3)], 3)}; "
      f"random search {R3.max():.3f}; maximum 3.863")
```
:::

输出为

```text
3-D: best of 36 evaluations 3.850 at [0.172 0.539 0.848]; random search 3.614; maximum 3.863
```

在种子 0 至 9 上，36 次评估中的最优值介于 3.801 至 3.850 之间，中位数为 3.837；预算相同的随机搜索（同样的十个种子，各用 `np.random.default_rng(s)` 抽取 36 个均匀分布的点）介于 2.771 至 3.782 之间，中位数为 3.495。下图展示同类运行：可以旋转由已评估点构成的立方体，并在穿过最优点的切片上查看模型。

```{figure}
//| figure: dim-bo-3d
//| label: fig-impl-hartmann
//| fig-cap: "三维 Hartmann 函数上的贝叶斯优化，设置与上面的清单相同：6 个初始点，此后用期望改进，共 36 次评估。不同之处在于，图中的初始设计为拉丁超立方，长度尺度由滑块固定，而清单使用均匀随机点，并按边际似然选择长度尺度；循环的其余部分相同。左：单位立方体中已评估的点，真实最大值点以星号标出。右：穿过最优点的模型切片。下：已找到的最优值。"
init: 6
steps: 36
```

::: {.pitfall title="单一的长度尺度"}
`rbf` 对所有输入使用同一个长度尺度。Hartmann-3 沿第三个输入的变化比沿第一个输入快得多；每个输入各用一个长度尺度（@sec-ard）能刻画这一点，共用一个长度尺度则不能。循环在这里依然有效，是因为三维问题能容忍粗糙的模型；@sec-high-dimensions 说明了这种容忍在何处失效。
:::

## 偏好模型 {#sec-impl-pref}

第四段清单即 @sec-gp-preference 的模型：效用为高斯过程，通过比较来观测，采用概率单位似然 $\Prob(\vx_w \succ \vx_l) = \Phi\big((g(\vx_w) - g(\vx_l))/(\sqrt{2}\sigma)\big)$ 与 Laplace 近似（@sec-pref-laplace）。数据包括两个列表：被比较输入的列表，以及比较的列表；每个比较是一对指向前一列表的索引（获胜者，落败者）。未知量是各被比较输入处的效用 $\vf$。

Laplace 近似需要 $\vf$ 的后验众数，用 Newton 法求得。Newton 法的每一步都需要对数似然的梯度 $\nabla\ell$ 与负 Hessian 矩阵 $\mW$（即 @eq-pref-grad-hess），二者由 `laplace_terms` 计算。输入 $w$ 与 $l$ 之间的一次比较，使获胜者的梯度分量增加 $\lambda(z)/(\sqrt{2}\sigma)$，落败者的梯度分量减去同样的量，其中 $\lambda(z) = \phi(z)/\Phi(z)$ 是该比较的标准化差 $z$ 处的逆 Mills 比（代码中的 `r`）。这次比较还把权重 $w_k = \lambda(z)\big(z + \lambda(z)\big)/(2\sigma^2)$（代码中的 `c`）加到 $\mW$ 中对应这两个输入的 $2 \times 2$ 块上，对角元取正号，非对角元取负号。对所有比较求和后，$\mW$ 就是 @sec-comparison-graph 中比较图的加权 Laplace 矩阵。

Newton 步即以 $\nabla\ell(\vf)$ 为对数似然梯度的 @eq-approx-newton：$\vf_{\text{new}} = \mK(\mI + \mW\mK)^{-1}\big(\mW\vf + \nabla\ell(\vf)\big)$。`fit_preference` 只用一次线性求解完成这一计算，无须求 $\mK$ 的逆；代码中的变量 `g` 存放 $\nabla\ell(\vf)$。另有一个事实可以在最后节省计算：在众数处，对数后验的梯度为零，因此 $\mK^{-1}\hat{\vf} = \nabla\ell(\hat\vf)$（@eq-pref-mode），预测均值 $\vk(\vx)^\T \mK^{-1}\hat{\vf}$ 就等于 $\vk(\vx)^\T \nabla\ell(\hat\vf)$，无须任何求解。

预测协方差与“观测”噪声协方差为 $\mW^{-1}$ 的高斯过程相同，即 $k(\vx, \vx') - \vk(\vx)^\T(\mK + \mW^{-1})^{-1}\vk(\vx')$。代码将其计算为先验减去 $\vk(\vx)^\T(\mI + \mW\mK)^{-1}\mW\,\vk(\vx')$，因为 $\mW$ 是奇异矩阵，不存在逆（图 Laplace 矩阵的每一行之和都为零，这正是 @sec-pref-identifiability 的平移不变性）。

::: {.code title="NumPy：Laplace 偏好模型"}
```python
def mills(z):
    """phi(z) / Phi(z), stable far into the lower tail."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    lo = z < -5
    out[~lo] = phi(z[~lo]) / Phi(z[~lo])
    out[lo] = -z[lo] / (1 - 1 / z[lo] ** 2 + 3 / z[lo] ** 4)
    return out

def laplace_terms(f, w, l, sigma):
    """Gradient and negative Hessian W of the probit log-likelihood of the comparisons."""
    s = sqrt(2.0) * sigma
    z = (f[w] - f[l]) / s
    r = mills(z)                               # lambda(z)，即逆 Mills 比
    g = np.zeros(len(f)); np.add.at(g, w, r / s); np.add.at(g, l, -r / s)
    c = r * (z + r) / s**2                     # 每次比较的权重 w_k，为正
    W = np.zeros((len(f), len(f)))
    np.add.at(W, (w, w), c); np.add.at(W, (l, l), c)
    np.add.at(W, (w, l), -c); np.add.at(W, (l, w), -c)
    return g, W                                # W 是加权图 Laplace 矩阵

def fit_preference(X, comps, ell=0.12, sigma=0.1, iters=50):
    """Laplace approximation for comparisons [(winner, loser), ...] indexing rows of X."""
    X = as2d(X)
    w, l = np.array(comps).T
    K = rbf(X, X, ell) + 1e-8 * np.eye(len(X))
    f = np.zeros(len(X))
    for _ in range(iters):                     # 对对数后验使用 Newton 法
        g, W = laplace_terms(f, w, l, sigma)
        f_new = K @ np.linalg.solve(np.eye(len(X)) + W @ K, W @ f + g)
        done = np.max(np.abs(f_new - f)) < 1e-9
        f = f_new
        if done:
            break
    g, W = laplace_terms(f, w, l, sigma)
    return dict(X=X, K=K, g=g, W=W, ell=ell)

def predict_preference(m, Xs):
    """Latent posterior mean and covariance at Xs under the Laplace approximation."""
    Ks = rbf(m["X"], Xs, m["ell"])
    mean = Ks.T @ m["g"]                       # 在众数处，K^-1 f 等于梯度
    B = np.eye(len(m["X"])) + m["W"] @ m["K"]
    cov = rbf(Xs, Xs, m["ell"]) - Ks.T @ np.linalg.solve(B, m["W"] @ Ks)
    return mean, cov
```
:::

此处的 Newton 迭代采用完整步长。概率单位对数似然是凹的，所以对数后验只有一个众数，在本附录的各次运行中完整步长均能收敛；生产环境中的实现为稳妥起见，会对步长加阻尼或采用信赖域，BoTorch 的 `PairwiseGP` 则把这一问题交给 SciPy 的求根器[@botorch2026pairwisegp]。`mills` 在 $z = -5$ 以下改用渐近展开，因为在该区域 $\Phi(z)$ 会下溢，直接相除将得到 $0/0$。

## EUBO 与偏好循环 {#sec-impl-pbo}

第五段清单完成 @alg-pbo。`eubo` 根据候选网格上的后验均值向量与协方差矩阵，一次性为每一对候选计算 @eq-eubo-closed：第 $i$ 行第 $j$ 列的元素是候选 $i$ 与 $j$ 中较好者的期望效用。`ask` 模拟一个人，其回答服从带隐藏效用的概率单位模型。`pbo_loop` 每一轮提出一次比较，重新拟合模型，再取 EUBO 矩阵中最大的非对角元素所对应的配对作为下一对。

::: {.code title="NumPy：EUBO 与偏好循环"}
```python
def eubo(mean, cov):
    """EUBO for every pair of candidates (Clark's formula), as a matrix."""
    v = np.diag(cov)
    s = np.sqrt(np.maximum(v[:, None] + v[None, :] - 2 * cov, 1e-12))
    d = mean[:, None] - mean[None, :]
    return mean[:, None] * Phi(d / s) + mean[None, :] * Phi(-d / s) + s * phi(d / s)

def ask(u, a, b, sigma, rng):
    """A simulated person: prefers a with the probit probability."""
    return rng.uniform() < Phi((u(a) - u(b)) / (sqrt(2.0) * sigma))

def pbo_loop(u, n_iter=15, sigma=0.1, ell=0.08, rule="eubo", seed=0):
    rng = np.random.default_rng(seed)
    cand = np.linspace(0, 1, 51)
    pts, comps = [], []
    def idx(x):                                # 每个不同的输入对应一个潜在值
        for i, p in enumerate(pts):
            if abs(p - x) < 1e-9:
                return i
        pts.append(x)
        return len(pts) - 1
    a, b = rng.choice(cand, 2, replace=False)  # 第一对随机选取
    for _ in range(n_iter):
        ia, ib = idx(a), idx(b)
        comps.append((ia, ib) if ask(u, a, b, sigma, rng) else (ib, ia))
        m = fit_preference(np.array(pts), comps, ell=ell, sigma=sigma)
        mean, cov = predict_preference(m, cand)
        if rule == "random":
            a, b = rng.choice(cand, 2, replace=False)
            continue
        E = eubo(mean, cov)
        np.fill_diagonal(E, -np.inf)           # 一对必须包含两个不同的选项
        i, j = np.unravel_index(np.argmax(E), E.shape)
        a, b = cand[i], cand[j]
    return cand[np.argmax(mean)], len(pts)

x_best, n_pts = pbo_loop(f)
print(f"PBO: after 15 comparisons among {n_pts} inputs, recommend x = {x_best:.2f}")
```
:::

以示例目标函数作为隐藏效用，输出为

```text
PBO: after 15 comparisons among 14 inputs, recommend x = 0.70
```

真正最偏好的点位于 $x = 0.73$，因此在本例中，15 个一比特的回答已足以在间距为 0.02 的网格上找到高峰。这一运行不如对应的数值版本可靠。在种子 0 至 19 上，比较 15 次后，有 15 次运行的推荐点与最偏好的点相距不超过 0.05；比较 30 次后有 16 次。在其余运行中，模型停留在 $x = 0.2$ 附近的宽峰上，不断询问该处几乎相同的配对，这正是 @sec-pbo-failure-modes 所描述的 EUBO 向当前最优坍缩的现象。随机配对（在清单中以 `rule="random"` 选择）在比较次数少时表现更差，比较次数多时则追了上来：比较 15 次后为 20 次运行中的 5 次，30 次后为 12 次。长度尺度也有影响：把 0.08 换成 0.12，比较 15 次后，EUBO 在 20 次运行中有 10 次找到高峰，因为如此光滑的模型无法表示如此窄的峰。

下图在同一目标函数上运行同一循环，每步做一次比较，并展示 `eubo` 返回的 EUBO 矩阵。

```{figure}
//| figure: eubo-map
//| label: fig-impl-eubo
//| fig-cap: "`eubo` 返回的矩阵：在示例目标函数上比较五次之后，长度尺度为 0.08，与清单所用的值相同。左：效用的后验与迄今的比较。右：每一对候选的 EUBO；最大的非对角元素即为下一对。按“询问下一对”执行 `pbo_loop` 的一步，回答由目标函数给出。"
lengthscale: 0.08
```

## 在 BoTorch 中实现 {#sec-impl-botorch}

BoTorch[@balandat2020botorch]以库组件的形式提供了这两个循环。下文的名称均已与 BoTorch 0.18.1 的源代码核对，清单也在该版本与 PyTorch 2.14.1、GPyTorch 1.15.2 下运行过。

首先是数值循环，仍使用同一 Hartmann 问题：

::: {.code title="BoTorch：使用期望改进的贝叶斯优化"}
```python
import warnings

warnings.filterwarnings("ignore")              # PairwiseGP 警告很多；见下文说明

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import PairwiseGP, SingleTaskGP
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.manual_seed(0)
torch.set_default_dtype(torch.double)
f3 = Hartmann(dim=3, negate=True)              # 在 [0, 1]^3 上的最大值为 3.86278
bounds = torch.stack([torch.zeros(3), torch.ones(3)])

# --- 使用期望改进的贝叶斯优化 ---------------------------------------------
X = torch.rand(6, 3)
Y = f3(X).unsqueeze(-1)                         # n x 1
for _ in range(30):
    gp = SingleTaskGP(X, Y)                     # 标准化 Y，为每个输入拟合一个长度尺度
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    acq = LogExpectedImprovement(gp, best_f=Y.max())
    x_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
    X = torch.cat([X, x_next])
    Y = torch.cat([Y, f3(x_next).unsqueeze(-1)])
print(f"BO: best of {len(Y)} evaluations {Y.max().item():.3f}")
```
:::

输出为 `BO: best of 36 evaluations 3.862`；在种子 0 至 4 上，结果介于 3.855 至 3.862 之间，NumPy 循环则为 3.801 至 3.850。结构与 `bo_nd` 相同，但每一行都是 NumPy 代码相应步骤的改进版本：

- `SingleTaskGP` 用结果变换（outcome transform）标准化输出，所用的径向基函数核为每个输入各设一个长度尺度（@sec-ard）。
- `fit_gpytorch_mll` 以基于梯度的优化，对各长度尺度与噪声最大化边际似然；`bo_nd` 则只比较了单一长度尺度的五个取值。
- `LogExpectedImprovement` 是期望改进的对数，计算方式数值稳定。它与期望改进有相同的最大值点；在大片区域中，期望改进本身过于接近零，无法引导搜索，它却仍保有有用的梯度[@ament2023unexpected]。
- `optimize_acqf` 用基于梯度的优化器最大化采集函数，优化器从若干起点出发，起点选自 512 个随机候选。

偏好循环使用 `PairwiseGP`（即 @sec-pairwisegp 中的模型）与解析形式的 EUBO：

::: {.code title="BoTorch：使用 EUBO 的偏好贝叶斯优化"}
```python
# 接续上面的脚本
def answer(pair, sigma=0.1):
    """A simulated person: index pair [winner, loser] within the two options."""
    u = f3(pair) + sigma * torch.randn(2)
    return torch.tensor([[0, 1]]) if u[0] > u[1] else torch.tensor([[1, 0]])

def fit_pairwise(X, comps):
    model = PairwiseGP(X, comps)
    try:
        fit_gpytorch_mll(PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model))
    except ModelFittingError:                   # 保留默认超参数
        model.eval()
    return model

X = torch.rand(2, 3)
comps = answer(X)                               # m x 2，每行为 [获胜者索引, 落败者索引]
for _ in range(30):
    model = fit_pairwise(X, comps)
    acq = AnalyticExpectedUtilityOfBestOption(pref_model=model)
    pair, _ = optimize_acqf(acq, bounds=bounds, q=2, num_restarts=8, raw_samples=256)
    comps = torch.cat([comps, answer(pair) + len(X)])
    X = torch.cat([X, pair])
best = X[fit_pairwise(X, comps).posterior(X).mean.argmax()]
print(f"PBO: after {len(comps)} comparisons, recommended point has utility {f3(best).item():.3f}")
```
:::

`PairwiseGP` 接受被比较的输入和一个整数比较张量，张量的每一行为（获胜者索引，落败者索引），与 `fit_preference` 的数据布局相同。它采用概率单位似然与 Laplace 近似，仅在参数化上有一处不同：不设噪声尺度 $\sigma$，而是固定似然的尺度、学习核函数的幅度。由于两者中只有比值可识别（@sec-pref-identifiability），这仍是同一个模型。`AnalyticExpectedUtilityOfBestOption` 即 @eq-eubo-closed；以 `q=2` 调用 `optimize_acqf`，会在六维的配对空间中对配对的两个成员联合优化该函数。对多于两个选项的查询，`qExpectedUtilityOfBestOption` 通过从后验采样估计 $\E_n[\max_i g(\vx_i)]$，用法相同，只是 `q` 更大。由于 `PairwiseGP` 只接受配对，此时在 $q$ 个选项中做出的一次选择，必须记录为被选选项与其余每个选项之间的比较，这是对 @eq-query-luce 的近似。

数值循环可以复现：每次运行都输出相同的 3.862。偏好循环则不然：在我们的环境中，即使固定种子也无法复现。脚本运行 11 次，输出的推荐点效用介于 2.18 至 3.84 之间，中位数为 3.23，而最大值为 3.863。拟合模型时细微的数值差异会改变下一次询问的配对，30 轮之后各次运行已经各不相同。尽管如此，与数值循环的对比依然成立：数值循环用 36 次评估达到 3.86，而 31 次比较（每次至多提供一比特信息）通常不足以完成三维搜索。@sec-query-design 的查询设计正是为了缩小这一差距。

::: {.pitfall title="PairwiseGP 的警告与拟合失败"}
清单在导入 BoTorch 之前屏蔽了警告，BoTorch 自己的教程也是如此[@botorch2026pairwise]，因为被比较的点彼此靠近时，`PairwiseGP` 会打印大量数值警告（“added jitter of 1.0e-06 to the diagonal”）。在我们使用 BoTorch 0.18.1 的部分运行中，超参数拟合还会直接以 `ModelFittingError` 失败；辅助函数 `fit_pairwise` 捕获这一错误，并在该轮保留模型的默认超参数。与人交互的循环绝不应因一次拟合失败而崩溃。
:::

## 习题 {#sec-impl-exercises}

::: {.exercise #exr-impl-ucb}
把 `bo_loop` 中的期望改进换成 @sec-ucb 的上置信界 $\mu_n(\vx) + \beta^{1/2}\sigma_n(\vx)$，取 $\beta^{1/2} = 2$。代码需要做什么改动？为什么输出的标准化对这一规则比对期望改进更重要？

::: {.solution}
只需改一行：`x_next = grid[np.argmax(mean + 2.0 * np.sqrt(var))]`。当前最优值 `y.max()` 不再需要。标准化之所以重要，是因为权重 2 乘的是模型单位下的标准差：输出经标准化、先验方差为 1 时，2 表示“两个先验标准差”。不做标准化，同一个数对变化幅度为 0.01 的输出太大，对变化幅度为 1000 的输出又太小。期望改进没有这样的常数，标准化只通过模型的拟合影响它。
:::
:::

::: {.exercise #exr-impl-choice}
把 `laplace_terms` 推广到从集合中选择的情形（@eq-query-luce）：每个观测由被选中的索引 `c` 和所展示索引的数组 `S`（包含 `c`）组成，温度为 `tau`。写出单个观测对梯度与对 $\mW$ 的贡献。

::: {.solution}
对 $j \in S$ 记 $p_j = e^{f_j/\tau}/\sum_{k \in S} e^{f_k/\tau}$，梯度贡献为 $(\mathbb{1}[j = c] - p_j)/\tau$，$\mW$ 在 $S$ 上的块为 $(\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\T)/\tau^2$（@sec-query-choice）：

```python
def choice_terms(f, c, S, tau, g, W):
    p = np.exp((f[S] - f[S].max()) / tau)
    p /= p.sum()
    g[S] += ((S == c) - p) / tau
    W[np.ix_(S, S)] += (np.diag(p) - np.outer(p, p)) / tau**2
```

取指数前先减去最大值，可以避免上溢。每个块都是半正定的，且每行之和为零，所以 $\mW$ 仍是加权图 Laplace 矩阵，`fit_preference` 的其余部分不变。
:::
:::

::: {.exercise #exr-impl-graph}
`pbo_loop` 运行之后，已回答的配对在被比较的输入上构成比较图（@sec-comparison-graph）。编写一个函数统计其连通分量的个数，并解释个数大于 1 对后验意味着什么。

::: {.solution}
```python
def components(n, comps):
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for w, l in comps:
        parent[find(w)] = find(l)
    return len({find(i) for i in range(n)})
```

连通分量多于一个时，没有任何一串回答把各组连接起来，因此似然不包含任何关于一组效用与另一组效用相对大小的信息：每个连通分量都给 $\mW$ 带来一个零特征值。只有先验能通过核函数把各组联系起来。EUBO 容易产生这样的图，因为它选出的配对常常与此前的配对没有共同的输入（@sec-pbo-failure-modes）。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @rasmussen2006gaussian 中用于回归的算法 2.1 与用 Newton 法做 Laplace 近似的第 3.4 节，分别是第一段与第四段清单的蓝本。
- @chu2005preference 是偏好模型的出处；@lin2022preference 与 @astudillo2023qeubob 分别是 EUBO 与 qEUBO 的出处。
- @balandat2020botorch 介绍了 BoTorch；其偏好教程[@botorch2026pairwise]在一个四维问题上运行偏好循环。
- @ament2023unexpected 解释了为何期望改进的对数比期望改进本身更易于优化。
