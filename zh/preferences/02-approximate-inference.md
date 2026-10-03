---
status: done
synopsis: "比较使后验不再是高斯分布。本章以一个效用差为例（其精确后验可以画出），推导并比较 Laplace 近似、期望传播、变分推断与采样；说明精确后验是偏斜正态分布（一般情形下是偏斜高斯过程），偏斜只出现在比较所涉及的方向上；最后报告近似方法的选择有多大影响。"
sources: ["Rasmussen and Williams 2006 ch. 3", "Kuss and Rasmussen 2005", "Nickisch and Rasmussen 2008", "Minka 2001", "Murray et al. 2010", "Benavoli et al. 2021", "Takeno et al. 2023"]
---

# 后验不是高斯分布时 {#sec-approx-inference}

高斯过程回归之所以有简洁的公式，依赖于一个事实：高斯先验与高斯观测噪声结合，得到的后验仍是高斯分布。@sec-comparisons 中的比较模型打破了这一事实。其似然是概率单位曲线或逻辑曲线，而非高斯函数，效用的后验因而不再是高斯分布。后续的一切，包括预测均值与预测带、下一个回答的概率以及 @sec-pbo 中的采集函数，都需要这一后验。

本章几乎完全围绕能体现这一困难的最小情形展开：一个效用差、一个高斯先验和少数几次比较。这一情形的精确后验可以在网格上计算并画出，每种近似都可以与真实答案对照。随后转向两个及更多潜在值的情形，那里的精确答案只在比较所涉及的方向上偏斜。最后讨论近似方法的选择在实践中影响多大，并给出相关证据。

## 非高斯似然 {#sec-non-gaussian-likelihoods}

先回顾回归为什么容易。先验为 $\vf \sim \N(\mathbf{0}, \mK)$，观测 $\vy = \vf + \bm{\varepsilon}$ 带高斯噪声，贝叶斯定理把关于 $\vf$ 的两个高斯密度相乘。高斯密度之积在相差一个常数的意义下仍是高斯密度（@sec-gaussian-sums），因此后验有闭式解，其归一化常数（即边际似然）也有闭式解。先验与似然构成一对共轭分布。

比较则不然。记 $\vf$ 为已比较输入处的效用，设已记录 $m$ 个回答，第 $k$ 个回答表示输入 $v_k$ 优于输入 $u_k$。采用 @eq-cmp-case-v 的概率单位模型，由贝叶斯定理得

$$
p(\vf \mid \D) = \frac{1}{Z}\, \N(\vf;\, \mathbf{0}, \mK) \prod_{k=1}^{m} \Phi\!\left(\frac{f_{v_k} - f_{u_k}}{\sqrt{2}\,\sigma}\right),
\qquad
Z = \int \N(\vf;\, \mathbf{0}, \mK) \prod_{k=1}^{m} \Phi\!\left(\frac{f_{v_k} - f_{u_k}}{\sqrt{2}\,\sigma}\right) \dd\vf.
$$ {#eq-approx-posterior}

先验是高斯的，但每个似然因子都是沿 $\vf$ 某一方向（即差 $f_{v_k} - f_{u_k}$ 的方向）变化的 S 形曲线。乘积不是高斯的；$Z$ 是高斯向量落入由 $m$ 堵软墙围成的区域的概率，即一个 $m$ 维积分，一般没有闭式解（@sec-no-closed-form）。高斯过程分类具有同样的结构：每个输入带有是或否的标签 $y_i = \pm 1$，而不是一个数，每个带标签的输入对应一个因子 $\Phi(y_i f(\vx_i))$ [@rasmussen2006gaussian, 第 3 章]。本章的方法最早正是在分类问题中发展和检验的。

### 最小情形 {#sec-approx-smallest}

把问题简化到只剩一个数。与 @sec-comparison-information 相同，令 $\Delta = g(A) - g(B)$ 为两个选项之间的效用差，其先验为高斯分布 $\Delta \sim \N(0, v_0)$。这里 $v_0$ 是方差；@eq-cmp-predictive 用标准差 $v$ 描述同类信念，故 $v_0 = v^2$。假设一个人回答了一次 $A$ 更好。记 $s = \sqrt{2}\,\sigma$ 为差上的噪声，则后验为

$$
p(\Delta \mid A \succ B) = 2\, \N(\Delta;\, 0, v_0)\, \Phi(\Delta/s).
$$ {#eq-approx-skew-normal}

因子 2 就是 $1/Z$：信念以零为中心时，由 @eq-cmp-predictive，这一回答的先验概率为 $\Phi(0) = 1/2$。高斯密度乘以正态分布函数，称为**偏斜正态**（skew-normal）密度，这一分布族由 @azzalini1985class 提出。它保留先验的上尾，切去下尾：噪声大时切得平缓，噪声小时切得陡峭。其均值有闭式解。

::: {.derivation title="一次比较后验的均值"}
1. 对 $\Delta \sim \N(0, v_0)$ 与可微函数 $h$，由分部积分得 $\E[\Delta\, h(\Delta)] = v_0\, \E[h'(\Delta)]$，因为密度 $\N(\Delta; 0, v_0)$ 的导数是 $-(\Delta/v_0)\, \N(\Delta; 0, v_0)$。这就是 **Stein 引理**（Stein's lemma）。
2. 取 $h(\Delta) = \Phi(\Delta/s)$，则 $h'(\Delta) = \phi(\Delta/s)/s$。于是 $\int \Delta\, \N(\Delta; 0, v_0)\, \Phi(\Delta/s)\, \dd\Delta = v_0\, \E[\phi(\Delta/s)/s]$。
3. $\phi(\Delta/s)/s$ 即密度 $\N(0; \Delta, s^2)$，只是视为 $\Delta$ 的函数。因此其先验期望等于 $\Delta - \eta$ 在 0 处的密度，其中 $\eta \sim \N(0, s^2)$ 为独立变量（@sec-gaussian-sums），即 $\N(0;\, 0, v_0 + s^2) = 1/\sqrt{2\pi(v_0 + s^2)}$。
4. 除以 $Z = 1/2$，得后验均值 $\E[\Delta \mid A \succ B] = 2 v_0 / \sqrt{2\pi(v_0 + s^2)} = \sqrt{2/\pi}\; v_0 / \sqrt{v_0 + s^2}$。
:::

当 $v_0 = 1$ 时，均值从 $\sigma = 1$ 时的约 0.46 增大到噪声消失时的 $\sqrt{2/\pi} \approx 0.80$，标准差则从约 0.89 降到 0.60（@exr-approx-variance）。**众数**（mode），即密度的峰值点，表现不同：它是方程 $\Delta/v_0 = \phi(\Delta/s)/(s\,\Phi(\Delta/s))$ 的解，随 $s$ 缩小而滑向零，$\sigma = 0.01$ 时为 0.05。在无噪声的极限下，后验是先验的上半部分，即半正态分布：峰值位于切口处，质量却大多远在切口右侧。

```{figure}
//| figure: approx-posterior-1d
//| label: fig-approx-exact
//| fig-cap: "A 与 B 经过比较之后，效用差 Δ = g(A) − g(B) 的后验，先验为标准正态分布。带阴影的曲线是在精细网格上计算的精确后验；竖直虚线与实线分别标出其众数与均值。表中给出均值、标准差，以及 B 实际上更好的概率 P(Δ < 0)。打开“显示先验与似然”，可以看到构成后验的乘积。"
method: exact
factors: true
```

可以做以下尝试：

- **缩小噪声。** 把 $\sigma$ 拖向 0.01。似然变为阶跃函数，后验变为半正态分布，众数与均值分离：众数移到切口处，均值停在 0.80 附近。
- **增大噪声。** $\sigma = 2$ 时，似然是平缓的斜坡，后验是略有偏移的钟形，众数与均值几乎重合。
- **让双方都获胜。** $A$ 赢三次、$B$ 也赢三次时，后验两侧都受到限制，接近以零为中心的高斯分布。偏斜来自单方面的证据。
- **让 $A$ 一直获胜。** 每多赢一次，质量就向上推移一些，但噪声小时左边缘始终陡峭：一连串相同的回答之后，证据更坚定地表明“$A$ 更好”，而不是表明“好多少”。

下文每种方法都要概括这种偏向一侧的形状，通常只用单个高斯分布。

## Laplace 近似 {#sec-laplace}

最简单的概括是在后验峰值处放置一个高斯分布，并赋予它峰值处的曲率。钟形曲线由最高点的位置以及从最高点下降的快慢决定；如果后验接近钟形，这两项信息就足以将其还原。这就是 **Laplace 近似**（Laplace approximation），得名于近似积分的 Laplace 方法。@tierney1986accurate 展示了它在贝叶斯计算中能发挥的作用：近似的后验矩与边际密度只需要一次最大化以及最大值点处的曲率。

记 $\Psi(\vf) = \log p(\D \mid \vf) + \log \N(\vf; \mathbf{0}, \mK)$ 为未归一化后验的对数，$\hat\vf$ 为其最大值点，即**众数**，也称最大后验估计。

::: {.derivation title="Laplace 近似"}
1. 记 $\nabla\Psi$ 为 $\Psi$ 的梯度，即其关于 $\vf$ 各分量的一阶导数组成的向量；记 $\nabla\nabla\Psi$ 为其 **Hessian 矩阵**，即二阶导数组成的矩阵。在 $\hat\vf$ 附近把 $\Psi$ 展开到二阶（Taylor 定理）：$\Psi(\vf) \approx \Psi(\hat\vf) + \nabla\Psi(\hat\vf)^\T(\vf - \hat\vf) - \tfrac12 (\vf - \hat\vf)^\T \mA\, (\vf - \hat\vf)$，其中 $\mA = -\nabla\nabla\Psi(\hat\vf)$。
2. 在最大值点处梯度为零，$\nabla\Psi(\hat\vf) = \mathbf{0}$，因此一次项消失。
3. 对数先验的 Hessian 矩阵为 $-\mK^{-1}$。记 $\mW = -\nabla\nabla \log p(\D \mid \vf)$（在 $\hat\vf$ 处取值）为对数似然的曲率，则 $\mA = \mK^{-1} + \mW$。
4. 取指数：$p(\vf \mid \D) \propto e^{\Psi(\vf)} \approx e^{\Psi(\hat\vf)} \exp\!\big(-\tfrac12 (\vf - \hat\vf)^\T \mA\, (\vf - \hat\vf)\big)$，右端在相差一个常数的意义下是高斯密度。
5. 因此 $q(\vf) = \N\big(\hat\vf,\, (\mK^{-1} + \mW)^{-1}\big)$。
:::

$$
p(\vf \mid \D) \approx \N\!\left(\hat\vf,\; (\mK^{-1} + \mW)^{-1}\right).
$$ {#eq-approx-laplace}

需要的只有两样：众数，以及众数处的曲率。对概率单位似然，由于 $\log\Phi$ 是凹函数，对数似然是凹的；对数先验是凹的二次函数，因此 $\Psi$ 只有一个最大值点，**Newton 法**（Newton's method）能很快找到它。Newton 法在当前点用二次近似代替原函数，然后跳到该二次函数的最大值点。

::: {.derivation title="求众数的 Newton 步"}
1. 梯度为 $\nabla\Psi(\vf) = \nabla \log p(\D \mid \vf) - \mK^{-1}\vf$，Hessian 矩阵为 $\nabla\nabla\Psi(\vf) = -(\mK^{-1} + \mW)$，此处 $\mW$ 在当前的 $\vf$ 处取值。
2. 局部二次函数的最大值点为 $\vf_{\text{new}} = \vf - (\nabla\nabla\Psi)^{-1}\nabla\Psi = \vf + (\mK^{-1} + \mW)^{-1}\big(\nabla \log p(\D \mid \vf) - \mK^{-1}\vf\big)$。
3. 写出 $\vf = (\mK^{-1} + \mW)^{-1}(\mK^{-1} + \mW)\vf$ 并合并各项：$\vf_{\text{new}} = (\mK^{-1} + \mW)^{-1}\big(\mW\vf + \nabla \log p(\D \mid \vf)\big)$。
4. 避免对 $\mK$ 求逆：由于 $(\mK^{-1} + \mW)\,\mK = \mI + \mW\mK$，有 $(\mK^{-1} + \mW)^{-1} = \mK(\mI + \mW\mK)^{-1}$，所以 $\vf_{\text{new}} = \mK(\mI + \mW\mK)^{-1}\big(\mW\vf + \nabla \log p(\D \mid \vf)\big)$。
:::

$$
\vf_{\text{new}} = \mK\,(\mI + \mW\mK)^{-1}\big(\mW\vf + \nabla \log p(\D \mid \vf)\big).
$$ {#eq-approx-newton}

从 $\vf = \mathbf{0}$ 出发反复应用 @eq-approx-newton，每当 $\Psi$ 没有增大就把步长减半，几次迭代即可收敛。分类问题中 $\mW$ 是对角矩阵，@rasmussen2006gaussian 给出了数值稳定的版本，即其算法 3.1。对比较而言，$\mW$ 不是对角矩阵；@sec-pref-laplace 推导了它的结构。

同一展开还可以近似归一化常数 $Z$，即用来拟合超参数的边际似然（@sec-marginal-likelihood）。对第 4 步中的高斯函数积分，得到

$$
\log Z \approx \log p(\D \mid \hat\vf) - \tfrac12\, \hat\vf^\T \mK^{-1} \hat\vf - \tfrac12 \log\left|\mI + \mK\mW\right|,
$$ {#eq-approx-laplace-evidence}

这就是 @chu2005preference 用于偏好问题的 Laplace 证据，即其论文中的式（12）；BoTorch 拟合偏好模型时最大化的也是这一量（@sec-pairwisegp）。

### 失效之处 {#sec-approx-laplace-wrong}

下图在精确后验上叠加了 Laplace 近似给出的高斯分布。

```{figure}
//| figure: approx-posterior-1d
//| label: fig-approx-laplace
//| fig-cap: "效用差后验的 Laplace 近似（蓝色）：以众数为中心、具有众数处曲率的高斯分布。把噪声调小到 0.01 附近，将其均值、标准差以及 B 更好的概率与表中的精确值对比。"
method: laplace
sigma: 0.1
```

噪声适中时，蓝色曲线与精确曲线十分接近。随着噪声缩小，近似以一种特定方式失效。$\sigma = 0.01$ 时，精确后验的均值为 0.80，标准差为 0.60，$B$ 更好的概率为 0.004。Laplace 近似的高斯分布以众数 0.05 为中心，标准差为 0.27，给出 $B$ 更好的概率为 0.43：在一个几乎无噪声的回答之后，它对顺序的不确定程度几乎与回答之前相同（@exr-approx-laplace-limit）。众数位于质量的边缘，以此为中心的高斯分布必然越过边缘。

这一普遍结论源自分类问题。@kuss2005assessing 在二分类高斯过程分类器上，把 Laplace 方法、期望传播与长时间采样的结果相比较，发现 Laplace 方法“系统性地低估均值”，因此潜在函数的近似后验“幅度太小”，预测概率过于保守，尽管潜在函数的符号大多正确。他们的结论是，该方法“如此不准确，以至于我们建议不要使用它，尤其是在需要认真对待预测概率的时候”。比较问题中也出现同样的模式：对决几乎无噪声时，众数可能离均值非常远 [@takeno2023practicalc]。

Laplace 方法至今仍在使用，原因是快速而简单。在从成对偏好中学习机器人奖励的研究中，@byk2020active 称期望传播“比 Laplace 近似更准确”，但“在实践中更慢”，最终出于计算效率选择了 Laplace 近似。它也是 BoTorch 的默认方法（@sec-pairwisegp）。

## 期望传播 {#sec-ep}

Laplace 近似只考察后验的一个点。更好的高斯分布应当匹配后验的均值与方差，即匹配其质量而不是峰值。对完整后验计算这些矩，与原问题一样困难；但对只含一个非高斯因子的后验，计算这些矩很容易。**期望传播**（expectation propagation，EP）正是由这类单因子问题构建近似的 [@minka2001expectation]。

每个似然因子 $t_k(\vf)$ 称为一个**位点**（site）。期望传播把每个位点替换为同一方向上的未归一化高斯函数 $\tilde t_k(\vf)$，使近似后验 $q(\vf) \propto \N(\vf; \mathbf{0}, \mK) \prod_k \tilde t_k(\vf)$ 为高斯分布，然后逐个细化各位点。

::: {.algorithm #alg-approx-ep title="期望传播"}
输入：先验 $\N(\mathbf{0}, \mK)$，位点 $t_1, \dots, t_m$。

1. 把每个位点的近似初始化为常数，使 $q$ 等于先验。
2. 选取位点 $k$，去掉其近似，得到**空腔分布**（cavity）$q_{\setminus k}(\vf) \propto q(\vf) / \tilde t_k(\vf)$，它是高斯分布。
3. 乘入精确因子，得到**倾斜分布**（tilted distribution）$\hat p_k(\vf) \propto q_{\setminus k}(\vf)\, t_k(\vf)$。
4. 计算 $\hat p_k$ 的均值与协方差。
5. 选择新的 $\tilde t_k$，使 $q_{\setminus k}\, \tilde t_k$ 恰好具有这些矩，并更新 $q$。
6. 对所有位点逐轮重复第 2 至 5 步，直到各位点的近似不再变化。
:::

第 3 步中的倾斜分布只有一个非高斯因子，且该因子只沿一个方向起作用，因此其矩可以归结为一维问题。对概率单位位点，答案有闭式解。设空腔分布下差 $\Delta$ 服从高斯分布 $\N(\mu, v)$，这里 $v$ 与 $v_0$ 一样表示方差；设位点为 $\Phi(y\, \Delta / s)$，$A$ 获胜时 $y = +1$，$B$ 获胜时为 $-1$。

::: {.derivation title="概率单位位点的矩"}
1. 由 @eq-cmp-predictive 的论证，归一化常数为 $Z(\mu) = \int \N(\Delta; \mu, v)\, \Phi(y\Delta/s)\, \dd\Delta = \Phi(z)$，其中 $z = y\mu / \sqrt{s^2 + v}$。
2. 在积分号下求导，$\partial_\mu \N(\Delta; \mu, v) = \tfrac{\Delta - \mu}{v}\, \N(\Delta; \mu, v)$，所以 $\partial_\mu \log Z = \E_{\hat p}[\Delta - \mu]/v$，倾斜分布的均值为 $\mu + v\, \partial_\mu \log Z$。
3. 再求一次导，得到倾斜分布的方差 $v + v^2\, \partial^2_\mu \log Z$。
4. 记 $\lambda = \phi(z)/\Phi(z)$，利用 $\tfrac{\dd}{\dd z}\lambda = -\lambda(z + \lambda)$，由链式法则得 $\partial_\mu \log Z = y\lambda / \sqrt{s^2 + v}$ 与 $\partial^2_\mu \log Z = -\lambda(z + \lambda)/(s^2 + v)$。
5. 因此，倾斜分布的均值为 $\mu + y v \lambda / \sqrt{s^2 + v}$，方差为 $v - v^2 \lambda (z + \lambda)/(s^2 + v)$。
:::

取 $s = 1$，这些量与概率单位分类的期望传播算法中的量相同 [@rasmussen2006gaussian, 第 3 章]。每次位点更新都是对协方差的秩一修改，对 $n$ 个潜在值的代价为 $O(n^2)$，因此遍历 $m$ 个位点一轮的代价为 $O(mn^2)$；当 $m$ 与 $n$ 同阶时，与一次 Newton 迭代相当。

```{figure}
//| figure: approx-posterior-1d
//| label: fig-approx-ep
//| fig-cap: "期望传播（绿色）与精确后验的对比。只有一次比较时，期望传播与精确的均值和标准差一致；同一对选项比较多次时，结果接近但不精确。在噪声很小时，将其 P(Δ < 0) 与精确值对比。"
method: ep
sigma: 0.03
```

只有一次比较时只有一个位点，倾斜分布就是精确后验，期望传播返回其精确的均值与方差：$\sigma = 0.03$ 时，均值为 0.80，标准差为 0.60，远优于 Laplace 近似的 0.12 与 0.32。不过，矩匹配也有自身的代价。均值和方差都正确的高斯分布，仍会把质量放在精确后验没有质量的地方：期望传播给出 $B$ 更好的概率为 0.093，而精确值为 0.013。在 $\sigma = 0.1$ 下赢五次时，多个位点作用在同一方向上，期望传播不再精确（均值 0.93、标准差 0.45，精确值为 0.90 与 0.58）。

这一微小的错位对比较问题尤为重要。@takeno2023practicalc 以采样方法的长时间运行结果为真实值（取 @sec-mcmc 中 Gibbs 采样器的 10,000 个样本，丢弃前 1,000 个，每十个保留一个），发现期望传播对均值与可信区间的估计非常准确；但对于已观测到 $\vx_w$ 胜过 $\vx_l$ 的对决，它高估了 $f(\vx_w) \le f(\vx_l)$ 的概率。在标准测试函数 Ackley 函数上，它低估了接近 0 或 1 的对决概率，其估计也没有保持各对之间的真实顺序，这可能改变采集函数选中的选项对。

在分类问题上，期望传播得到的评价是正面的。Kuss 与 Rasmussen 发现，其预测概率与边际似然估计都与长时间采样的结果非常接近 [@kuss2005assessing]；一项范围更广的研究比较了二分类高斯过程分类的各种近似方法，结论是：“除非计算预算非常紧张，期望传播算法几乎总是首选方法” [@nickisch2008approximations]。与凹函数上的 Newton 法不同，期望传播不能保证每一步都改进某个目标函数，因此位点更新出现振荡时，实现中会对更新施加阻尼。

## 变分推断 {#sec-vi}

第三条途径把近似转化为优化：选定一族简单分布（这里是高斯分布），从中找出与后验最接近的成员，接近程度用 Kullback-Leibler 散度 $\KL(q \,\|\, p)$ 度量（@sec-kl）。到后验的散度含有未知的 $\log Z$，无法直接计算，但仍然可以最小化。

::: {.derivation title="证据下界"}
1. 由贝叶斯定理，$\log p(\vf \mid \D) = \log p(\D \mid \vf) + \log p(\vf) - \log Z$。
2. 在 $q$ 下对 $\log q(\vf) - \log p(\vf \mid \D)$ 取期望：$\KL(q \,\|\, p(\cdot \mid \D)) = \E_q[\log q(\vf) - \log p(\vf)] - \E_q[\log p(\D \mid \vf)] + \log Z$。
3. 第一个期望就是到先验的 $\KL(q \,\|\, p)$。整理得 $\log Z = \underbrace{\E_q[\log p(\D \mid \vf)] - \KL(q \,\|\, p)}_{\mathcal{L}(q)} + \KL(q \,\|\, p(\cdot \mid \D))$。
4. 最后一项非负，所以 $\mathcal{L}(q) \le \log Z$；又因 $\log Z$ 不依赖于 $q$，最大化 $\mathcal{L}$ 等价于最小化到后验的散度。
:::

$\mathcal{L}(q)$ 称为**证据下界**（evidence lower bound，ELBO）。对高斯分布 $q$ 与概率单位似然，每一项的计算代价都很低：两个高斯分布之间的散度有闭式解；期望对数似然是若干一维积分之和，每次比较对应一个积分，积分对象是 $q$ 赋予某个差的高斯分布。因此，标准的基于梯度的优化器即可将其最大化。

散度的方向决定了拟合的特性。$\KL(q \,\|\, p)$ 在 $q$ 下对 $\log(q/p)$ 取平均，只要 $q$ 在某处有质量而 $p$ 在那里几乎没有，散度就会极大。因此最优的 $q$ 停留在后验的支撑集之内，往往过窄。期望传播的局部矩更新则按相反方向的散度 $\KL(\hat p_k \,\|\, q)$ 选择与倾斜分布最接近的高斯分布；这一散度惩罚 $q$ 遗漏的质量，往往使其过宽。

```{figure}
//| figure: approx-posterior-1d
//| label: fig-approx-vi
//| fig-cap: "高斯变分推断（品红色），即证据下界最高的高斯分布，与精确后验的对比。噪声很小时，它在不可能的一侧 Δ < 0 几乎不留质量，代价是标准差只有精确值的一半左右。选择“全部”，可与 Laplace 近似和期望传播对比。"
method: vi
sigma: 0.01
```

$\sigma = 0.01$ 时，变分高斯分布的均值为 0.88，标准差为 0.30，精确值为 0.80 与 0.60；它给出 $B$ 更好的概率为 0.002，接近精确值 0.004。它守住了期望传播所越过的硬边界，却把不确定性减半。每种方法都在其准则顾及不到之处出错。

变分推断是可扩展的近似方法。借助**诱导点**（inducing points），即概括函数的一小组伪输入 [@titsias2009variational]，再结合随机优化，它可以处理数千次比较，以及任何期望对数可估计的似然。正因如此，近来许多偏好建模工作都以它为基础：涉及数千名用户和物品的群体偏好学习 [@simpson2020scalable]、top-$k$ 排序 [@nguyen2021top]、选择函数 [@benavoli2023choice]、反应时 [@shvartsman2024response]、一篇 2025 年预印本中把比较与把握度评分相结合的混合似然 [@wu2025mixed]，以及 qEUBO 的实验 [@astudillo2023qeubob]。截至 2026 年 9 月，对于变分推断在偏好似然上的误差，我们没有找到可与 Laplace 近似和期望传播所受评估相比的系统评估。

## 采样 {#sec-mcmc}

以上三种方法都用高斯分布代替后验。**Markov 链蒙特卡洛**（Markov chain Monte Carlo，MCMC）方法则抽取一列样本，其分布收敛到后验本身。任何感兴趣的量，如均值、可信区间、一个选项胜过另一个的概率，都可以通过对样本取平均来估计。样本数增加时答案趋于精确，代价是计算量，以及需要判断链何时已运行得足够久。

对于高斯先验乘以似然的情形，**椭圆切片采样**（elliptical slice sampling）是自然的选择。@murray2010elliptical 为“具有多元高斯先验的模型”设计了这一方法：其“代码简单、通用”，“没有自由参数”需要调节，并且“在多种基于高斯过程的模型上效果很好”。其思路是沿一个椭圆移动，椭圆经过当前状态与从先验中新抽取的样本，因此每个提议在先验下都已合理，只需检查似然。

::: {.algorithm #alg-approx-ess title="椭圆切片采样的一步"}
输入：当前状态 $\vf$，先验 $\N(\mathbf{0}, \mK)$，对数似然 $\ell$。

1. 抽取 $\bm{\nu} \sim \N(\mathbf{0}, \mK)$，由其确定椭圆 $\vf' (\theta) = \vf\cos\theta + \bm{\nu}\sin\theta$。
2. 从 $(0, 1)$ 上的均匀分布抽取 $u$，令阈值 $\log y = \ell(\vf) + \log u$。
3. 从 $[0, 2\pi)$ 上的均匀分布抽取 $\theta$，令区间为 $[\theta - 2\pi, \theta]$。
4. 若 $\ell(\vf'(\theta)) > \log y$，接受 $\vf'(\theta)$ 并停止。
5. 否则向零收缩区间：用 $\theta$ 替换与它位于零同侧的端点，然后在区间内均匀抽取新的 $\theta$，返回第 4 步。
:::

收缩的区间始终包含 $\theta = 0$，即当前状态，因此循环必然终止。比较问题还允许进一步改进。概率单位因子是一个高斯噪声变量落在效用差之下的概率，所以精确后验是某个高斯分布的边际分布，这一高斯分布限制在由线性约束切出的区域内。@benavoli2021preferentialb 用 LinESS 对其采样。LinESS 是针对线性截断高斯分布的无拒绝椭圆切片采样器，时间与内存代价分别为 $O(n^3)$ 与 $O(n^2)$（以比较次数计）。@takeno2023practicalc 对同一分布采用 Gibbs 采样（每次在给定其余所有变量的条件下重新抽取一个变量），发现它更快：在其表 1 中约为 0.54 秒对 1.61 秒；不过当截断数量远超维度时，LinESS 应当更有优势。

```{figure}
//| figure: approx-posterior-1d
//| label: fig-approx-mcmc
//| fig-cap: "用椭圆切片采样对效用差的后验采样。紫色直方图显示样本；表格把样本的均值、标准差以及小于零的比例与精确值对比。按“更多样本”，样本数从 200 增至 1,000 和 5,000；按“新链”，换一个随机种子重新运行。"
method: mcmc
sigma: 0.03
samples: 200
```

只有 200 个样本时，直方图参差不齐；在默认随机种子下，均值偏差接近 0.2。Markov 链的相继样本彼此相关，因此 200 个这样的样本所含信息少于 200 次独立抽取。样本数为 1,000 时，误差低于 0.05；为 5,000 时，估计值与精确值一栏约两位数字一致。采样是本章唯一能够靠增加计算把误差压到零的方法，因此研究其他方法时都以它为参照。

## 精确答案：偏斜高斯过程 {#sec-skew-gp}

一次比较的后验 @eq-approx-skew-normal 有专门的名称，也有闭式解。一般的后验 @eq-approx-posterior 是否也有？答案是肯定的，理由正是 @sec-comparisons 中反复使用的随机效用技巧：概率单位因子就是一个隐藏的高斯变量为正的概率。

::: {.derivation title="后验即经过选择的高斯分布"}
1. 对每次比较 $k$，记 $\mathbf{a}_k = \mathbf{e}_{v_k} - \mathbf{e}_{u_k}$，使 $\mathbf{a}_k^\T\vf = f_{v_k} - f_{u_k}$，并引入相互独立的 $\eta_k \sim \N(0, s^2)$。于是 $\Phi(\mathbf{a}_k^\T\vf/s) = \Prob(\mathbf{a}_k^\T\vf - \eta_k > 0 \mid \vf)$。
2. 把各个 $\mathbf{a}_k^\T$ 堆叠成 $m \times n$ 矩阵 $\mathbf{D}$，把各个 $\eta_k$ 堆叠成 $\bm{\eta}$。由独立性，似然为 $\Prob(\mathbf{D}\vf - \bm{\eta} > \mathbf{0} \mid \vf)$，即所有不等式同时成立的概率。
3. 由贝叶斯定理，$p(\vf \mid \D)$ 就是在事件 $\mathbf{D}\vf - \bm{\eta} > \mathbf{0}$ 发生的条件下 $\vf$ 的分布，其中 $(\vf, \bm{\eta})$ 服从联合高斯分布。
4. 归一化常数 $Z$ 就是这一事件的概率。向量 $\mathbf{D}\vf - \bm{\eta}$ 服从均值为零、协方差为 $\mathbf{D}\mK\mathbf{D}^\T + s^2\mI$ 的高斯分布，所以 $Z$ 是一个 $m$ 维高斯向量各坐标全为正的概率，即**卦限概率**（orthant probability）。
:::

设一个高斯向量的某个线性变换加上独立高斯噪声后为正，以此为条件，该高斯向量的分布就是**统一偏斜正态**（unified skew-normal）分布。@durante2019conjugate 证明，对任意高斯先验，参数化概率单位回归的后验都属于这一分布族，因此该分布族与概率单位似然共轭。对于函数，@benavoli2021preferentialb 证明了“偏好函数的真实后验分布是一个偏斜高斯过程（SkewGP），其成对边际分布高度偏斜”，并据此论证 Laplace 方法“通常给出非常差的近似”。@thm-theory-skewgp 给出了这一定理及其参数。

由这一推导还可以看出后验在哪些方向上偏斜。每个因子都只通过差 $\mathbf{D}\vf$ 依赖于 $\vf$。在与所有 $\mathbf{a}_k$ 都正交的方向上，似然是常数，后验就是以这些差为条件的高斯先验。偏斜至多出现在 $m$ 个方向上，而且从不出现在把所有效用平移相同量的方向上，因为沿这一方向没有任何差会改变。下图展示最小的实例：两个效用以及它们之间的比较。

```{figure}
//| figure: approx-posterior-2d
//| label: fig-approx-2d
//| fig-cap: "A 获胜后两个效用 g(A) 与 g(B) 的后验；先验相关系数 ρ 相当于核函数赋予相近输入的相关性。阴影与粗等高线表示精确后验及其 50% 与 90% 最高密度区域；虚线椭圆是先验的 90% 区域；蓝色椭圆是相同水平下的 Laplace 近似。顶部的条带单独比较 g(A) 的边际分布。偏度读数是标准化三阶矩，高斯分布为 0，半正态分布约为 1。"
```

可以做以下尝试：

- **沿对角线看。** 精确等高线在直线 $g(A) = g(B)$ 处截断，并沿这条直线自由延伸。偏斜完全发生在横跨对角线的方向上，即差的方向；沿对角线方向，即和的方向，后验就是高斯先验。
- **看顶部条带。** $g(A)$ 的边际分布混合了偏斜的差与高斯的和。$\sigma = 0.05$、$\rho = 0.5$ 时，差的偏度约为 0.98，而单独的 $g(A)$ 偏度约为 0.04。即使后验远非高斯分布，单个效用看起来也可能接近高斯分布。@kuss2005assessing 在分类问题中做过同样的观察：高维截断高斯分布的边际分布“可能与高斯分布相当相似”。
- **增大相关系数。** 在核函数看来彼此接近的输入，其先验效用相关，因此它们之差的先验方差很小，为 $2 - 2\rho$。同样的噪声相对于这一离散程度就变大了，切口更平缓，差的偏度随之下降，从 $\rho = 0.5$ 时的约 0.98 降到 $\rho = 0.9$ 时的 0.90。模型已认为相似的两个选项，一次比较对它们的改变也最小。
- **增大噪声。** $\sigma = 1$ 时切口变得平缓，等高线接近椭圆；差的偏度降到约 0.06。

这一结论可以推广到多维情形。设一次会话在 60 个不同设计之间做了 30 次比较，每个设计是 6 维参数空间中的一个点，则后验定义在 60 个潜在效用上。后验只能在 30 个比较方向张成的子空间内偏斜；输入维度只通过核函数起作用，核函数决定各效用之间的相关程度。Laplace 近似与期望传播对其余方向的处理都是精确的。它们出错的是横跨各比较方向的形状，而下一个回答的概率恰恰取决于这一形状（推断）。

精确后验是有代价的：每次预测都需要从截断多元高斯分布中采样，边际似然需要计算高维正态卦限概率 [@benavoli2021preferentialb]。近似方法仍在使用，原因就在于此。

## 近似方法的影响有多大 {#sec-approximation-matters}

上面的图展示了各种近似在最小情形下的偏差。这些偏差是否会改变偏好优化器面对真实回答时的行为？讨论证据之前，先用 @tbl-approx-methods 对各方法做一汇总。

::: {.table #tbl-approx-methods title="偏好后验的近似推断：各方法匹配的对象、计算代价，以及在本章单个效用差情形下的失效方式。"}
| 方法 | 匹配的对象 | 每次拟合的代价 | 典型失效 | 使用者 |
|---|---|---|---|---|
| Laplace 近似 | 众数及众数处的曲率 | 几步 Newton 迭代，每步 $O(n^3)$ | 回答几乎无噪声时，中心落在质量的边缘；均值与离散程度都过小 | Chu 与 Ghahramani；BoTorch `PairwiseGP`；Bıyık 等（2020） |
| 期望传播 | 逐个位点匹配单个因子的矩 | 若干轮秩一更新，每轮 $O(mn^2)$ | 在不可能的一侧有质量；接近 0 或 1 时对决概率的顺序出错 | Siivola 等（2021）；optuna-dashboard |
| 变分推断（高斯） | 证据下界最高的高斯分布 | 一次优化；借助诱导点可扩展 | 过窄 | crowdGPPL；top-$k$ 排序；qEUBO 实验 |
| 采样 | 极限意义下的后验本身 | 大量样本；截断高斯分布 | 样本相关，需要长时间运行；蒙特卡洛误差 | Benavoli 等（2021）；Takeno 等（2023） |
:::

最有力的证据来自 @takeno2023practicalc。他们以 Gibbs 采样为真实值，使用径向基函数核、噪声方差 $10^{-4}$ 以及均匀随机的对决。Laplace 近似不准确，“因为众数可能离均值非常远，尤其是当”噪声方差很小时；他们的结论是：“尽管 Laplace 近似很快，基于 Laplace 近似的偏好贝叶斯优化将会失败”。期望传播对均值与可信区间非常准确，但扭曲了对决概率，如 @sec-ep 所述。他们也不认同 @benavoli2021preferentialb 早先的检验：那项检验中，一个区间内的输入全部落败，另一个区间内的输入全部获胜。他们称这种有偏的训练对决“不现实”，改用随机对决。尽管如此，他们仍用 Laplace 证据拟合超参数，认为它在噪声小时足够准确。

这些结果来自几乎无噪声的比较，而 @fig-approx-laplace 显示，Laplace 近似恰恰在这一区间表现最差。人的比较带有噪声，如前面各图所示，噪声会使切口变平缓、使偏斜缩小。在真实的人类噪声水平下，Laplace 近似与期望传播的误差有多大，我们找到的论文中没有一篇回答这个问题；我们也没有找到在真实人类比较上对 Takeno 等人的比较所做的独立复现（推断；两者均截至 2026 年 9 月）。@sec-obs-inference 完整报告了这方面的证据，以及各软件库的选择。

实践中，方法的选择取决于三个问题。如果后验通过成对结果的概率进入采集函数，如 EUBO（@sec-eubo），那么高斯近似放错位置的质量影响最大，值得为采样或至少为期望传播付出额外代价（推断）。如果只需要后验均值来推荐最终设计，任何能把效用顺序排对的方法都已足够。如果会话很长，或模型有很多用户，变分推断是可扩展的方法。

## 习题 {#sec-approx-exercises}

::: {.exercise #exr-approx-variance}
两次应用 Stein 引理，证明一次比较的后验 @eq-approx-skew-normal 的二阶矩为 $\E[\Delta^2 \mid A \succ B] = v_0$，与先验相同，从而方差为 $v_0 - \tfrac{2}{\pi} v_0^2/(v_0 + s^2)$。取 $v_0 = 1$，验证 $\sigma = 1$ 时的 0.89 与 $\sigma \to 0$ 时的 0.60 这两个值。

::: {.solution}
取 $h(\Delta) = \Delta\,\Phi(\Delta/s)$ 应用 Stein 引理：在先验下，$\E[\Delta^2\Phi(\Delta/s)] = v_0\,\E[\Phi(\Delta/s) + \Delta\,\phi(\Delta/s)/s]$。第一项为 $v_0 \cdot \tfrac12$。第二项中，$\N(\Delta; 0, v_0)\,\phi(\Delta/s)$ 正比于关于 $\Delta$ 的零均值高斯密度，因此 $\Delta$ 与之乘积的期望为零。除以 $Z = 1/2$，得 $\E[\Delta^2 \mid A \succ B] = v_0$。方差为 $v_0 - (\E[\Delta \mid A \succ B])^2 = v_0 - \tfrac{2}{\pi}\, v_0^2/(v_0 + s^2)$。当 $v_0 = 1$、$\sigma = 1$ 时，$s^2 = 2$，方差为 $1 - \tfrac{2}{3\pi} \approx 0.788$，标准差约为 0.89。当 $s \to 0$ 时，方差趋于 $1 - 2/\pi \approx 0.363$，标准差约为 0.60。比较移动了均值，却不改变二阶矩：它重新分配了先验的质量，而没有使其变小。
:::
:::

::: {.exercise #exr-approx-laplace-limit}
对 $v_0 = 1$ 时一次比较的后验，证明当 $s \to 0$ 时，Laplace 近似赋予事件 $\Delta < 0$ 的概率趋于 $1/2$，而精确概率趋于零。可以利用如下事实：$z$ 较大时，逆 Mills 比 $\lambda(z) = \phi(z)/\Phi(z)$ 满足 $\lambda(z) \approx \phi(z)$。

::: {.solution}
众数 $\hat\Delta$ 满足 $\hat\Delta = \lambda(\hat\Delta/s)/s$。记 $\hat z = \hat\Delta/s$，则 $s^2 \hat z = \lambda(\hat z) \approx \phi(\hat z)$，所以当 $s \to 0$ 时 $\hat z$ 无界增长，但只以 $\sqrt{2\log(1/s^2)}$ 的速度增长，且 $\hat\Delta = s\hat z \to 0$。利用 $\lambda(\hat z) \approx s^2\hat z$，曲率为 $1 + \lambda(\hat z)(\hat z + \lambda(\hat z))/s^2 \approx 1 + \hat z^2$，所以 Laplace 近似的标准差约为 $1/\sqrt{1 + \hat z^2}$。Laplace 近似给出的 $\Delta < 0$ 的概率为 $\Phi(-\hat\Delta\sqrt{1 + \hat z^2}) \approx \Phi(-s\hat z^2)$；由于 $\hat z^2$ 只按对数增长，$s\hat z^2 \to 0$，因此该概率趋于 $\Phi(0) = 1/2$。精确后验是半正态分布，在零以下没有质量。一个完全确定了顺序的无噪声回答，在 Laplace 近似下却显示为对顺序毫无信息。
:::
:::

::: {.exercise #exr-approx-kl}
设目标分布为半正态分布：$\Delta > 0$ 时 $p(\Delta) = 2\,\N(\Delta; 0, 1)$，否则为零。（a）解释为什么对任意高斯分布 $q$，$\KL(q \,\|\, p)$ 都是无穷大。（b）解释为什么使 $\KL(p \,\|\, q)$ 最小的高斯分布具有 $p$ 的均值与方差。（c）把这两个答案与 @fig-approx-vi 和 @fig-approx-ep 在噪声极小时显示的现象联系起来。

::: {.solution}
（a）$\KL(q \,\|\, p) = \E_q[\log q - \log p]$，而任何高斯分布在 $\Delta < 0$ 上都有正的质量，那里 $\log p = -\infty$，所以散度为无穷大。噪声很小但为正时，$p$ 在那里很小但不为零，在这一散度下最好的高斯分布会把几乎全部质量挤进 $\Delta > 0$，因而变窄。（b）$\KL(p \,\|\, q) = \E_p[\log p] - \E_p[\log q]$；只有第二项依赖于 $q$，而对高斯分布，$\E_p[\log q] = -\tfrac12\log(2\pi v) - \E_p[(\Delta - \mu)^2]/(2v)$，它在 $\mu = \E_p[\Delta]$、$v = \Var_p[\Delta]$ 时取最大值。（c）@fig-approx-vi 中的变分拟合避开 $\Delta < 0$，且过窄；@fig-approx-ep 中期望传播的矩匹配具有正确的均值与方差，但越过了边缘。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @rasmussen2006gaussian 第 3 章推导了高斯过程分类的 Laplace 近似与期望传播，给出的稳定算法可以沿用到比较问题。
- @kuss2005assessing 与 @nickisch2008approximations 细致比较了分类问题中的各种近似方法，解释了 Laplace 近似为何失效、期望传播为何表现良好。
- @minka2001expectation 提出了期望传播；@murray2010elliptical 提出了椭圆切片采样；@tierney1986accurate 是用 Laplace 方法求后验矩的经典文献。
- @durante2019conjugate 证明了概率单位模型与统一偏斜正态分布的共轭性；@benavoli2021preferentialb 将其推广到偏好后验，并对其精确采样。
- @takeno2023practicalc 测量了 Laplace 近似与期望传播在对决上的误差有多大，并提出了一种代价更低的精确采样替代方案。
