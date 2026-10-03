---
status: done
synopsis: "先以硬币、再以直线和平面为例，精确求出先验、似然、后验与预测分布。说明点估计为何无法指示下一步的评估位置，证据如何权衡不同模型，以及直线权重的后验为何是整个函数上的后验的铺垫。"
sources: ["textbooks", "Bishop 2006, sec. 2.1 and ch. 3", "Rasmussen and Williams 2006, sec. 2.1", "Gelman et al. 2013, ch. 2", "MacKay 2003, ch. 28"]
---

# 贝叶斯推断 {#sec-bayesian-inference}

@sec-bayes-rule 在由十一个可能偏向构成的网格上，每抛一次就更新一次对硬币的信念，最后提出警告：网格方法无法扩展。取十一个值的未知量需要十一个数，而未知函数在每个输入处都需要一个数。该节给出的出路是选择更新有闭式解的分布，用少数几个数概括整个信念。@sec-gaussian 随后给出了其中最主要的分布，即高斯分布，并说明高斯分布经条件化与相乘后仍属于同一分布族。

本章将两者结合，把贝叶斯推断视为一套流程：写下看到数据之前的信念，写下数据的生成方式，再由贝叶斯定理得出看到数据之后应有的信念。我们对两个模型精确地执行这一流程。第一个是偏向连续取值的硬币，此时 Beta 分布代替了此前的网格。第二个是直线，继而是平面，其系数带有高斯不确定性。在此过程中会遇到贝叶斯优化器向模型提出的几个问题：报告哪一个值；下一次观测会是什么；数据更支持两个模型中的哪一个；没有公式可用时如何处理。

直线模型比表面上看起来更重要。它的先验和后验都是两个权重上的高斯分布；@sec-function-space 为直线增加大量权重，把这个先验变成整个函数上的先验，@sec-gp-regression 再对后验做同样的处理。下一部分的高斯过程，就是本章的线性模型在特征个数趋于无穷时的极限。

## 先验、似然与后验 {#sec-prior-likelihood-posterior}

@sec-bayes-rule 针对取有限个值的未知量，给出了贝叶斯定理四个组成部分的名称。本书中的未知量大多是连续的：硬币的偏向可取 $[0, 1]$ 中任意值，此外还有直线的斜率、函数的值。定理的形式不变，只是以密度代替概率，以积分代替求和：

$$
p(\theta \given \D) = \frac{p(\D \given \theta)\, p(\theta)}{p(\D)},
\qquad
p(\D) = \int p(\D \given \theta)\, p(\theta)\, \dd\theta.
$$ {#eq-bayes-rule}

其中 $\theta$ 代表全部未知量，可以是向量；$\D$ 代表数据。先验 $p(\theta)$ 与似然 $p(\D \given \theta)$ 共同构成**模型**（model）。后验 $p(\theta \given \D)$ 是模型得出的结论；证据 $p(\D)$ 是模型在看到数据之前赋予这些数据的概率。

### 模型是生成数据的程序 {#sec-bayes-generative}

编写软件的读者或许会发现，把模型看作模拟数据的程序最容易理解。先验规定如何抽取未知量，似然规定在给定未知量时如何抽取数据。以硬币为例：

1. 从先验中抽取偏向 $\theta$。
2. 对每次抛掷 $i = 1, \dots, n$，以概率 $\theta$ 得到正面。

对带噪声的直线，每个观测对应一个输入 $x_i$：

1. 从先验中抽取截距和斜率。
2. 对每个输入 $x_i$，计算直线在该处的值，再加上独立的高斯噪声。

贝叶斯推断把这样的程序倒过来运行：已知输出，推断隐藏的抽样结果可能取哪些值，以及每种取值的概率。把模型写成模拟器还提供了一种实用的检验，可以检查先验是否表达了建模者的本意：用从先验中抽取的值运行程序，观察它生成的假数据。如果假数据明显荒谬，说明先验有误，之后再多的计算也无法弥补。这类模拟称为先验预测检验，@gabry2019visualization 说明了如何将其作为建模的常规步骤。

两个程序的第二步都假设：一旦 $\theta$ 已知，各观测相互独立（@sec-independence）。于是整个数据集的似然可以分解为

$$
p(\D \given \theta) = \prod_{i=1}^n p(y_i \given \theta),
\qquad
\log p(\D \given \theta) = \sum_{i=1}^n \log p(y_i \given \theta),
$$ {#eq-bayes-iid}

实现时使用对数之和，以免下溢。正如 @sec-prob-coin 所示，独立性还允许数据逐个到达：前 $n$ 个观测之后的后验，可作为第 $n + 1$ 个观测的先验，且到达顺序不影响结果。

### 闭式解与共轭 {#sec-bayes-conjugacy}

连续贝叶斯推断的难点在于证据中的积分。对大多数先验与似然的组合，这个积分没有解析公式；维数稍高时，也无法在网格上计算。贝叶斯定理的比例形式 @eq-prob-proportional 可以绕开这个积分：只要能认出右边（似然乘以先验）的形状属于某个已知分布，归一化常数也就随之已知。@sec-gaussian-sums 正是这样做的：高斯先验与高斯似然之积具有高斯分布的形状，因此无需计算任何积分，即知后验是高斯分布。

具有这种性质的先验有专门的名称。若每个后验仍属于同一先验族，则称该先验族与此似然**共轭**（conjugate）。更新共轭模型只需更新少数几个参数，因此接下来两节能够展示随数据增加而变化的精确后验。

::: {.keyidea title="后验就是完整的答案"}
贝叶斯模型所能给出的一切都由后验计算而来：最佳猜测、合理取值的范围、预测、决策。概括总会丢弃信息，而最常被丢弃的，恰恰是优化器选择下一次查询所需的信息：模型的不确定程度。
:::

## 共轭的例子 {#sec-beta-binomial}

回到硬币，现在其偏向 $\theta$ 可以是 $[0, 1]$ 中的任意数。这正是 Thompson 在 1933 年提出的问题：给定迄今观察到的成功与失败，应如何看待一个未知的成功概率？这样的一个概率超过另一个的可能性有多大[@thompson1933likelihood]？如今，人们对在线实验中按钮的点击率（@sec-app-online），以及多臂赌博机中的每个选项，提出的仍是同一个问题；多臂赌博机问题是指在成功率未知的少数几个选项（“臂”）之间反复做出选择（@sec-bandits）。

### Beta 分布 {#sec-bayes-beta-dist}

我们需要 $[0, 1]$ 上的一族密度，它既要足够灵活以表达先验，又要在硬币的更新下封闭。在某个特定的抛掷序列中出现 $h$ 次正面、$t$ 次反面，其似然为 $\theta^h(1 - \theta)^t$。与之形状相同的先验，乘以似然后仍保持这一形状。这就是 **Beta 分布**（Beta distribution）：

$$
\mathrm{Beta}(\theta;\, \alpha, \beta) = \frac{\theta^{\alpha - 1}(1 - \theta)^{\beta - 1}}{B(\alpha, \beta)},
\qquad \alpha, \beta > 0,
$$ {#eq-bayes-beta}

其中 $B(\alpha, \beta) = \int_0^1 \theta^{\alpha - 1}(1 - \theta)^{\beta - 1}\,\dd\theta$ 是使面积为一的常数，即 Beta 函数。$\alpha = \beta = 1$ 时密度是平的，即均匀先验。$\alpha$ 与 $\beta$ 相等且较大时，密度呈以二分之一为中心的鼓包；两者不等时，鼓包向一侧倾斜。Beta 分布的均值以及众数（$\alpha, \beta > 1$ 时）为

$$
\E[\theta] = \frac{\alpha}{\alpha + \beta},
\qquad
\operatorname{mode}[\theta] = \frac{\alpha - 1}{\alpha + \beta - 2},
$$

方差为 $\alpha\beta / \big((\alpha + \beta)^2(\alpha + \beta + 1)\big)$，随 $\alpha + \beta$ 增大而减小。

::: {.derivation title="Beta 先验与抛硬币共轭"}
1. 由 @eq-bayes-iid，$h$ 次正面和 $t$ 次反面的似然为 $p(\D \given \theta) = \theta^h(1 - \theta)^t$。
2. 由贝叶斯定理的比例形式和 @eq-bayes-beta，$p(\theta \given \D) \propto \theta^h(1 - \theta)^t \cdot \theta^{\alpha - 1}(1 - \theta)^{\beta - 1}$。常数 $B(\alpha, \beta)$ 不含 $\theta$，可以略去。
3. 把指数相加，得 $\theta^{\alpha + h - 1}(1 - \theta)^{\beta + t - 1}$。
4. 这正是参数为 $\alpha + h$ 和 $\beta + t$ 的 @eq-bayes-beta 的形状。常数只是使面积为一的那个数，因此密度由形状决定，于是 $p(\theta \given \D) = \mathrm{Beta}(\theta;\, \alpha + h,\, \beta + t)$。
5. 证据是两个常数之比：$p(\D) = B(\alpha + h, \beta + t) / B(\alpha, \beta)$。
:::

更新就是计数：正面次数加到 $\alpha$ 上，反面次数加到 $\beta$ 上。

### 以数据为单位的先验强度 {#sec-bayes-pseudo-counts}

这种更新方式提示了先验参数的一种解读。$\mathrm{Beta}(\alpha, \beta)$ 先验的效果，相当于已经看到了 $\alpha$ 次正面和 $\beta$ 次反面，因此可以用两个有明确含义的数来描述它：均值 $m_0 = \alpha / (\alpha + \beta)$，即预期的偏向；强度 $n_0 = \alpha + \beta$，即它相当于多少次抛掷。经过 $n = h + t$ 次真实抛掷后，后验均值为

$$
\E[\theta \given \D] = \frac{\alpha + h}{n_0 + n}
= \frac{n_0}{n_0 + n}\, m_0 \;+\; \frac{n}{n_0 + n}\, \frac{h}{n}.
$$ {#eq-bayes-shrinkage}

后验均值是先验均值与观测到的正面比例的加权平均，权重分别与先验强度和抛掷次数成正比。抛掷次数少时先验占主导，次数多时数据占主导。像这样把估计值拉向先验值，称为**收缩**（shrinkage）；@eq-bayes-shrinkage 精确给出了收缩的程度，即假想数据与真实数据之比。

在下图中可以设置这两部分，并观察后验的变化。

```{figure}
//| figure: bayes-beta
//| label: fig-bayes-beta
//| fig-cap: "Beta-二项模型（@eq-bayes-beta 与 @eq-bayes-shrinkage）。先验（虚线）由均值和以抛掷次数计的强度 $n_0$ 设定；点线是已观测抛掷的似然，缩放为单位面积，以便共用坐标轴；后验（实线）为 $\mathrm{Beta}(\alpha + h, \beta + t)$，阴影为其中心 95% 可信区间。可用按钮增加抛掷，或用滑块设置次数。"
```

**从均匀先验开始**。取默认的 $n_0 = 2$ 和均值 0.5，先验为平坦的 $\mathrm{Beta}(1, 1)$。此时后验与似然形状完全相同，点线与实线重合：在平坦先验下，贝叶斯定理只是把似然归一化。

**增强先验**。把 $n_0$ 调到 50。此时 7 次正面、3 次反面几乎无法使后验偏离 0.5：先验相当于数据五倍的抛掷次数。再把正面次数调到 70、反面次数调到 30，观察数据如何逐渐占据主导。

**设置错误而自信的先验**。把先验均值设为 0.2，$n_0 = 50$，数据设为 70 次正面、30 次反面。后验落在两者之间，更靠近数据。与数据冲突的自信先验需要大量观测才能纠正，这是先验应当如实反映未知之处的一个理由。

**观察区间收缩**。保持正反面次数之比不变，把两者都乘以四，可信区间约缩窄一半。比例的后验标准差按 $1/\sqrt{n}$ 下降，因此数据增至四倍，精度提高到两倍。

图中阴影所示的 95% **可信区间**（credible interval）是包含 95% 后验概率的范围。许多人以为置信区间回答的是这样的问题：给定这些数据，$\theta$ 可能位于何处？可信区间回答的正是这个问题。

### 硬币的预测 {#sec-bayes-coin-predicts}

后验还能回答关于未来的问题：下一次抛掷正面朝上的概率是多少？对正面概率 $\theta$ 在后验上求平均即可：

$$
\Prob(\text{next flip is heads} \given \D) = \int_0^1 \theta\, p(\theta \given \D)\,\dd\theta
= \E[\theta \given \D] = \frac{\alpha + h}{\alpha + \beta + n}.
$$ {#eq-bayes-coin-predictive}

在均匀先验下，上式为 $(h + 1)/(n + 2)$，即 Laplace 继承法则。3 次抛掷都是正面时，它给出 $4/5$，而不是观测比例 $3/3$ 所暗示的必然。在后验上求平均是预测的一般方法，@sec-predictive 将再次讨论。

这一模型在后文有两种反复出现的用法。Thompson 采样（@sec-thompson 将其推广到函数）最初正是针对这一情形提出的规则：从每个选项的 Beta 后验中抽取一个偏向，尝试抽样值最大的选项[@thompson1933likelihood]。此外，这一模型对每个观测只需要一个“是”或“否”，因此凡是数据为二元回答的场合，它都会再次出现。例如，2024 年的一个对话式推荐系统为每个物品维护一个 Beta 后验，用户每回答一次就更新一次，更新幅度由语言模型根据该回答对这一物品的支持程度确定[@austin2024bayesian]。

## 点估计及其局限 {#sec-point-estimates}

实践中常常需要一个单独的数：硬币的偏向、直线的斜率、学习率的最佳设置。把后验化为单个值有三种标准方法。在讨论它们共同丢失了什么之前，先分别了解每种方法的做法。

**最大似然估计**（maximum likelihood estimate，MLE）忽略先验，选取使数据出现概率最大的值：$\hat\theta_{\text{MLE}} = \argmax_\theta p(\D \given \theta)$。对硬币，它就是正面比例 $h/n$。对带高斯噪声的直线，最大化似然等价于最小化残差平方和，因此最大似然估计就是最小二乘拟合（@exr-bayes-ridge）。

**最大后验估计**（maximum a posteriori estimate，MAP）选取后验的峰值：$\hat\theta_{\text{MAP}} = \argmax_\theta p(\D \given \theta)\,p(\theta)$。对硬币，它是后验众数 $(\alpha + h - 1)/(\alpha + \beta + n - 2)$。写成对数形式，最大后验估计最大化对数似然与对数先验之和，对数先验相当于惩罚项。若直线的权重服从高斯先验，惩罚项就是权重向量长度平方的某个倍数，此时最大后验估计就是岭回归，即带 $L_2$ 惩罚的最小二乘拟合；许多软件工程师接触过的正则化就是这种做法。

**后验均值**（posterior mean）$\E[\theta \given \D]$ 对后验求平均。它是期望平方误差最小的估计，对硬币即为 @eq-bayes-shrinkage。用加一平滑避免零计数的读者，实际上已经用过均匀先验下的后验均值。

```{figure}
//| figure: bayes-beta
//| label: fig-bayes-estimates
//| fig-cap: "均匀先验下，3 次正面、0 次反面之后硬币偏向的三种点估计。最大似然估计位于 1，对应一枚永远不会反面朝上的硬币；在平坦先验下，最大后验估计与之重合；后验均值为 0.8。阴影所示的 95% 区间向下延伸到约 0.4。调整先验和次数，可以看到这些估计时而分开，时而汇合。"
heads: 3
tails: 0
estimates: true
```

**读出 3 次正面之后的最大似然估计**。按照这一估计，硬币总是正面朝上，以它为基础的模型会不惜一切押注不会出现反面。后验的结论则温和得多：偏向很可能较高，但约 0.4 至 1 之间的任何值都是合理的。

**给先验一定权重**。把先验强度设为 4、均值设为 0.5，即 $\mathrm{Beta}(2, 2)$。最大后验估计离开边界，移到 0.8，后验均值移到 0.71。先验相当于两次假想的正面和两次假想的反面，可以防止因数据过少而过度自信。

**增加数据，观察估计如何汇合**。60 次正面、20 次反面时，三种估计彼此相差不超过 0.02。数据足够多时，选用哪种估计已无关紧要，先验也是如此。

### 点估计无法做到的事 {#sec-bayes-point-limits}

点估计更深层的问题不在于可能出错，而在于它无法表明自己可能错多少。2 次抛掷中出现 1 次正面的硬币，与 1,000 次抛掷中出现 500 次正面的硬币，最大似然估计都是 0.5。任何人都会更信任后一个估计，后验则记录了其中的原因：在均匀先验下，前者的后验为 $\mathrm{Beta}(2, 2)$，几乎与先验本身一样宽，而后者的后验标准差约为 0.016。

在优化中，缺失的这部分信息正是问题的关键。设已试过两个学习率：第一个运行了 3 次，平均验证准确率为 0.80；第二个只运行了 1 次，准确率为 0.78。按照点估计，第一个更好，第二个应当放弃。按照后验，第二个完全可能更好：仅凭一次带噪声的运行，它的准确率仍不确定；在那里再运行一次，所获得的信息会比第一个的第 4 次运行更多。在利用看似最好的选项与探索不确定的选项之间作出选择，是贝叶斯优化的核心权衡（@sec-explore-exploit），只有保留不确定性的模型才能做出这种选择。@sec-acquisition 中的每个采集函数，都是同时依据后验均值与后验散布的规则。

## 贝叶斯线性回归 {#sec-blr}

硬币只有一个未知数。目标函数则是输入与输出之间的未知关系，其中最简单的是直线。设在输入 $x_i$ 处观测到带噪声的值 $y_i$，将其建模为

$$
y_i = w_1 + w_2 x_i + \varepsilon_i,
\qquad \varepsilon_i \sim \N(0, \sigma_n^2),
$$

其中 $w_1$ 为截距，$w_2$ 为斜率，噪声是标准差为 $\sigma_n$ 的独立高斯噪声。问题是：经过几次观测之后，应当如何看待这条直线，进而如何看待它在尚未尝试的输入处的取值？

### 模型 {#sec-bayes-blr-model}

把权重写成向量 $\vw = (w_1, w_2)^\T$，把各权重所乘的量（这里是常数 1 和输入 $x$）写成**特征**（features）向量 $\boldsymbol{\phi}(x) = (1, x)^\T$，于是直线的值为 $f(x) = \boldsymbol{\phi}(x)^\T\vw$。对 $n$ 个观测，以各特征向量为行，构成 $n \times 2$ 的**设计矩阵**（design matrix）$\boldsymbol{\Phi}$，观测值则排成向量 $\vy$。模型为：

$$
\vw \sim \N(\mathbf{0}, \mSigma_p),
\qquad
\vy \given \vw \sim \N\!\left(\boldsymbol{\Phi}\vw,\; \sigma_n^2\mI\right).
$$ {#eq-bayes-blr-model}

先验协方差 $\mSigma_p$ 表示预期的权重大小；图中取 $\sigma_p^2\mI$，即各权重独立、标准差为 $\sigma_p$。本节的结论都不依赖于特征取 $1$ 和 $x$，任何一组固定的特征均可，这正是 @sec-function-space 的出发点。

尚无数据时，权重上的先验已经是直线上的先验。$\vw$ 的每个样本对应一条直线，由 @eq-gauss-affine，直线在 $x$ 处取值的方差为 $\boldsymbol{\phi}(x)^\T\mSigma_p\boldsymbol{\phi}(x) = \sigma_p^2(1 + x^2)$。因此先验直线从 $x = 0$ 附近（那里只有截距在变化）向外呈扇形散开，离该处越远，散布越大，按二次方增长。

### 后验 {#sec-bayes-blr-posterior}

先验和似然关于 $\vw$ 都是高斯形式，因此后验也是高斯分布，可用 @eq-gauss-recipe-nd 的方法求出。

::: {.derivation title="权重上的后验"}
1. 由贝叶斯定理，$\log p(\vw \given \vy) = \log p(\vy \given \vw) + \log p(\vw) + \text{const}$，其中常数为对数证据，与 $\vw$ 无关。
2. @eq-bayes-blr-model 的对数似然为 $-\frac{1}{2\sigma_n^2}\lVert \vy - \boldsymbol{\Phi}\vw \rVert^2 + \text{const}$，对数先验为 $-\tfrac12\vw^\T\mSigma_p^{-1}\vw + \text{const}$。
3. 展开范数的平方：$\lVert \vy - \boldsymbol{\Phi}\vw \rVert^2 = \vy^\T\vy - 2\vy^\T\boldsymbol{\Phi}\vw + \vw^\T\boldsymbol{\Phi}^\T\boldsymbol{\Phi}\vw$。第一项不含 $\vw$。
4. 整理关于 $\vw$ 的二次项和一次项：$\log p(\vw \given \vy) = -\tfrac12\vw^\T\left(\sigma_n^{-2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + \mSigma_p^{-1}\right)\vw + \sigma_n^{-2}\vy^\T\boldsymbol{\Phi}\vw + \text{const}$。
5. 这正是 @eq-gauss-recipe-nd 的形式，精度为 $\mA = \sigma_n^{-2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + \mSigma_p^{-1}$，一次项系数为 $\mathbf{h} = \sigma_n^{-2}\boldsymbol{\Phi}^\T\vy$。所以后验是协方差为 $\mA^{-1}$、均值为 $\mA^{-1}\mathbf{h}$ 的高斯分布。
:::

合写为一式：

$$
\vw \given \vy \;\sim\; \N\!\left(\bar{\vw},\; \mA^{-1}\right),
\qquad
\mA = \frac{1}{\sigma_n^2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + \mSigma_p^{-1},
\qquad
\bar{\vw} = \frac{1}{\sigma_n^2}\mA^{-1}\boldsymbol{\Phi}^\T\vy.
$$ {#eq-bayes-blr-posterior}

这就是高斯过程教科书中的权重空间结果[@rasmussen2006gaussian, 第 2.1 节]，其形式与 @sec-gauss-product 中的结果相同。后验精度 $\mA$ 等于先验精度加上数据项，数据项是对各观测求和 $\sigma_n^{-2}\sum_i \boldsymbol{\phi}(x_i)\boldsymbol{\phi}(x_i)^\T$：每个观测只沿自身的特征向量方向增加精度，在其他方向上不增加。后验均值是惩罚系数为 $\sigma_n^2/\sigma_p^2$ 的岭回归拟合（@exr-bayes-ridge）。与 @sec-gaussian-conditioning 一样，后验协方差与观测值 $\vy$ 无关，只取决于观测的位置。

同一后验也可以通过另一途径得到。$\vy$ 是 $\vw$ 的线性映射加上独立噪声，因此 $(\vw, \vy)$ 联合服从高斯分布，可以用 @eq-gauss-conditional 以 $\vy$ 为条件。这一途径需要对 $n \times n$ 矩阵求逆，每个观测对应一行；而 @eq-bayes-blr-posterior 只需对 $2 \times 2$ 矩阵求逆，每个权重对应一行。两种结果一致，因为有一个恒等式能把这种形式的 $n \times n$ 矩阵之逆化为 $2 \times 2$ 矩阵之逆（Woodbury 恒等式，@sec-id-woodbury）。@sec-function-space 采用 $n \times n$ 的途径，因为权重个数趋于无穷时，这一途径依然可行。

### 同一后验的两种视角 {#sec-bayes-two-views}

下图在两个空间中展示后验。左侧是数据空间，每个观测是一个点，每个权重向量是一条直线。右侧是权重空间，每个权重向量是一个点：椭圆表示 $(w_1, w_2)$ 上的先验和后验，画法与 @fig-gauss-2d 中的二维高斯分布相同。左侧每条紫色直线是从后验中抽取的一个样本，右侧对应的紫色点表示同一样本。

```{figure}
//| figure: bayes-line
//| label: fig-bayes-line
//| fig-cap: "数据空间（左）与权重空间（右）中的贝叶斯线性回归（@eq-bayes-blr-posterior）。点击左图可添加观测，点击已有的点可将其删除。每条紫色直线是从后验中抽取的一个样本，在权重空间中对应一个紫色点。阴影带是直线取值的 95% 区间带，虚线是新的带噪声观测的 95% 区间带。虚线椭圆为先验，阴影椭圆为 Mahalanobis 距离 1 和 2 处的后验。数据仅作示意。"
```

**清空数据**。此时后验即先验：在权重空间中是一个圆，在数据空间中是一簇在 $x = 0$ 处最窄的扇形直线。

**添加一个点**。在权重空间中，圆收缩为细长的椭圆。一个观测确定了直线在一个输入处的值，即截距与斜率的某一个组合，另一个组合仍不受约束：在数据空间中，抽取的直线都从该点附近穿过，并绕它转动（@exr-bayes-one-point）。

**在远离第一个点的地方添加第二个点**。椭圆缩成一小团，各直线几乎重合。相距较远的两个点即可确定一条直线。

**再把两个点放得很近**。斜率仍不确定，椭圆仍然细长，直线在这对点的两侧呈扇形散开。观测的位置与观测的数量同样重要。

**调高噪声**。每个观测增加的精度减少，后验向先验靠拢，直线可以更自由地偏离数据点。

### 平面与更多输入 {#sec-bayes-plane}

本书中的目标函数通常有多个输入，直线于是变为平面。有两个输入时，特征为 $\boldsymbol{\phi}(\vx) = (1, x_1, x_2)^\T$，权重包括一个截距和两个斜率，记作 $\vw = (w_0, w_1, w_2)^\T$。这里从零开始编号，使斜率 $w_j$ 与对应输入 $x_j$ 的下标一致；@eq-bayes-blr-posterior 依然适用，只是精度矩阵变为 $3 \times 3$。变化在于数据能够确定哪些量，下图直观地展示了其中的几何结构。

```{figure}
//| figure: bayes-plane
//| label: fig-bayes-plane
//| fig-cap: "两个输入的贝叶斯线性回归。点击右侧俯视图，可在该处评估一个隐藏平面，噪声标准差为 0.1；俯视图的阴影表示 $f$ 的后验标准差，模型越不确定，颜色越深。三维视图显示立在竖杆上的观测、后验均值平面（蓝色）和从后验中抽取的五个平面（紫色）；拖动视图或使用按钮可以旋转。隐藏平面和初始输入仅作示意。"
```

**观察初始数据**。三个输入都位于对角线 $x_1 = x_2$ 上。抽取的平面都从这三个点附近穿过，却像铰链上的门一样绕对角线摆动；俯视图沿对角线颜色较浅，两个远角颜色较深。数据确定了沿对角线方向的斜率，对横跨对角线方向的倾斜却毫无约束；读数给出了这一方向。

**在俯视图上点击靠近远角的位置**。只需一个不在对角线上的观测，摆动即告停止。各平面迅速合拢，俯视图各处颜色变浅。

**清空俯视图，再评估随机输入**。三个处于一般位置（不在同一条直线上）的输入，就足以在噪声范围内确定一个平面。

一般规律可由 $\mA$ 的结构得出。每个观测只沿其特征向量增加精度，因此数据只约束权重空间中由输入张成的方向。有 $d$ 个输入时，线性模型有 $d + 1$ 个权重；要约束所有方向，至少需要 $d + 1$ 个观测，且这些观测的输入不能全部落在某个更低维的仿射子空间中。用线性模型拟合六个参数的调参问题，需要七次分布良好的评估，模型才不会在某个方向上仍与先验同样不确定。高斯过程可以弯曲，所需的评估要多得多；这一数目如何随维度增长，是 @sec-high-dimensions 讨论的主题之一。

高维线性模型并非只是教学工具。若特征本身由神经网络学习得到，基于这些特征的贝叶斯线性回归可以充当代理模型，曾用于大规模地为图像识别与图像描述模型调参，其代价与观测数呈线性关系[@snoek2015scalable]。更值得注意的是，2026 年的一项研究发现，对输入做一次几何变换后，贝叶斯线性回归在 60 至 6,000 维的任务上与领先的高维贝叶斯优化方法表现相当[@doumont2026we]。

### 计算方法 {#sec-bayes-blr-code}

设有 $M$ 个特征和 $n$ 个观测。计算 @eq-bayes-blr-posterior 时，构造 $\mA$ 的代价为 $O(nM^2)$，分解它的代价为 $O(M^3)$。代价只随观测数线性增长，上面两项研究利用的正是这一性质。相比之下，高斯过程的代价为 $O(n^3)$（@sec-gp-computation）。

::: {.code title="NumPy"}
```python
import numpy as np

def blr_posterior(Phi, y, prior_var, noise_var):
    """Posterior mean and covariance of w for y = Phi w + noise, w ~ N(0, prior_var I)."""
    M = Phi.shape[1]
    A = Phi.T @ Phi / noise_var + np.eye(M) / prior_var   # 后验精度
    L = np.linalg.cholesky(A)
    mean = np.linalg.solve(L.T, np.linalg.solve(L, Phi.T @ y / noise_var))
    Linv = np.linalg.solve(L, np.eye(M))
    cov = Linv.T @ Linv                                     # A^{-1}
    return mean, cov

x = np.array([-0.8, -0.3, 0.2, 0.7]); y = np.array([-0.5, 0.0, 0.35, 0.9])
Phi = np.column_stack([np.ones_like(x), x])              # 特征 (1, x)
mean, cov = blr_posterior(Phi, y, prior_var=1.0, noise_var=0.09)
```
:::

## 预测分布 {#sec-predictive}

权重的后验只是手段，而非目的。优化器需要的是两种信念：目标函数在某个尚未尝试的输入处取何值，以及在该处会观测到什么值。两者的求法与硬币下一次抛掷的预测相同：对每个可能的参数值给出的预测，在后验上求平均，

$$
p(y_* \given x_*, \D) = \int p(y_* \given x_*, \vw)\, p(\vw \given \D)\, \dd\vw.
$$ {#eq-bayes-predictive}

这就是**后验预测分布**（posterior predictive distribution）。它同时考虑了两个不确定性来源：参数并不确切已知；即使已知，新的观测也会带有新的噪声。

对线性模型，这个积分无需实际计算。潜在值 $f_* = \boldsymbol{\phi}(x_*)^\T\vw$ 是 $\vw$ 的高斯后验的线性映射，由 @eq-gauss-affine 知它服从高斯分布；观测又加上了独立噪声，由 @eq-gauss-sum，方差中要再加上噪声方差：

$$
f_* \given \D \sim \N\!\left(\boldsymbol{\phi}_*^\T\bar{\vw},\; \boldsymbol{\phi}_*^\T\mA^{-1}\boldsymbol{\phi}_*\right),
\qquad
y_* \given \D \sim \N\!\left(\boldsymbol{\phi}_*^\T\bar{\vw},\; \boldsymbol{\phi}_*^\T\mA^{-1}\boldsymbol{\phi}_* + \sigma_n^2\right),
$$ {#eq-bayes-blr-predictive}

其中 $\boldsymbol{\phi}_* = \boldsymbol{\phi}(x_*)$。第一个方差反映模型对直线的不确定性；第二个方差再加上一次新测量的噪声，这正是 @sec-gp-noise 中所作的区分。

```{figure}
//| figure: bayes-line
//| label: fig-bayes-predictive
//| fig-cap: "用直线拟合聚集在中部的 4 个观测所得的预测分布（@eq-bayes-blr-predictive）。阴影带覆盖关于直线取值的 95% 信念；虚线带在此基础上再加上一次新观测的噪声。区间带在数据中心处最窄，离开中心后逐渐张开，因为直线的方差随距离按二次方增长。数据仅作示意。"
points: "-0.2:0.1,-0.1:0.3,0:0.2,0.1:0.4"
samples: false
```

**观察区间带的形状**。区间带在数据中心处最窄，向两侧变宽，形如领结。直线在数据附近被固定，却可以自由转动，因此其不确定性随离中心的距离增大。

**比较两条区间带**。在数据附近，虚线带比阴影带宽得多：那里新观测的不确定性主要来自测量噪声。在远处两条区间带趋于一致，因为此时直线本身的不确定性占主导。

**把点分散开**。在两端附近添加观测，领结随之变平，因为相距较远的输入确定了斜率。

一种诱人的捷径是只用一条拟合好的直线来预测，即把点估计 $\bar{\vw}$ 代入似然。由此得到的**插入式预测**（plug-in predictive）处处方差为 $\sigma_n^2$，相当于声称确切知道这条直线，即使在远离所有数据的 $x = 10$ 处也是如此。在优化器最需要意识到自身无知的地方，它恰恰过于自信。

领结形状还揭示了线性模型的局限。在两簇数据之间，虽然没有任何观测，直线仍被两侧的数据簇固定，因此在那里的不确定性很小。对直线来说这是正确的；对未知的目标函数则不然，函数在空隙中可能有任何形态。用于优化的模型，其不确定性应当在所有缺少数据的地方增大，而不仅仅在远离数据中心的地方增大。@sec-function-space 给线性模型足够多的特征，使其能够弯曲，从而做到这一点。

## 模型证据 {#sec-evidence}

到目前为止，每个模型都附带一些选择：硬币的先验均值与强度；直线的先验宽度 $\sigma_p$ 与噪声 $\sigma_n$；特征本身。这类数用于设定模型，不属于模型推理的未知量 $\theta$，称为模型的**超参数**（hyperparameters）。（机器学习用同一个词指训练过程的设置，见 @sec-cost-of-evaluation。两种用法都指比被学习的量高一个层次的设置。）数据应如何为这类选择提供依据？比较各模型最佳拟合时的似然、选出最高者，这种做法行不通，因为更灵活的模型总能拟合得至少同样好，连噪声也一并拟合。

贝叶斯定理在更高一个层次上给出了答案。把模型 $\mathcal{M}$ 本身视为未知量，其在看到数据之后的概率为

$$
p(\mathcal{M} \given \D) \propto p(\D \given \mathcal{M})\, p(\mathcal{M}),
\qquad
p(\D \given \mathcal{M}) = \int p(\D \given \theta, \mathcal{M})\, p(\theta \given \mathcal{M})\, \dd\theta.
$$ {#eq-bayes-evidence}

决定结果的量是 @eq-bayes-rule 中的证据，也称**边际似然**（marginal likelihood），即模型在看到数据之前赋予这些数据的概率，对模型的先验取平均。证据奖励的不是模型所能达到的最佳拟合，而是模型平均预测的拟合程度。

### 自动的 Occam 剃刀 {#sec-bayes-occam}

这种平均自然地偏好简单模型。所有可能数据集上的概率之和必须为一。灵活的模型能解释许多不同的数据集，因而把概率摊得很薄，分给每个可解释数据集的概率都很少。僵硬的模型则把概率集中在较少的数据集上。数据落在僵硬模型的预测范围内时，僵硬模型胜出；否则灵活模型胜出。这就是 Occam 剃刀，只是在这里它是概率论的推论，而非经验法则[@mackay2003information, 第 28 章]。

::: {.example #ex-bayes-fair-coin title="这枚硬币公平吗？"}
比较同一枚硬币的两个模型。$\mathcal{M}_0$ 假定硬币公平，即恰有 $\theta = 1/2$；$\mathcal{M}_1$ 假定偏向未知，取均匀先验。对一个共 $n$ 次抛掷、含 $h$ 次正面的特定序列，$\mathcal{M}_0$ 赋予的概率为 $2^{-n}$；由 @sec-bayes-beta-dist 中推导的第 5 步，$\mathcal{M}_1$ 赋予的概率为 $B(h + 1, t + 1) = h!\,t!/(n + 1)!$。

10 次抛掷中有 5 次正面时，公平硬币模型的证据为 $1/1024 \approx 9.8 \times 10^{-4}$，偏向未知模型的证据为 $5!\,5!/11! \approx 3.6 \times 10^{-4}$：数据以约 2.7 倍的比值支持公平硬币。灵活的模型把概率分给了并未发生的一边倒的结果。10 次抛掷中有 9 次正面时，公平硬币模型的证据不变，灵活模型的证据则升至 $9!\,1!/11! = 1/110 \approx 9.1 \times 10^{-3}$，数据以约 9.3 倍的比值支持偏向未知的模型。
:::

线性模型的证据有闭式解。观测是高斯权重的线性映射加上独立的高斯噪声，由 @eq-gauss-affine 和 @eq-gauss-sum 知观测服从高斯分布 $\vy \sim \N(\mathbf{0},\, \mK_y)$，其中 $\mK_y = \boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T + \sigma_n^2\mI$；对数证据就是这个高斯分布在观测值 $\vy$ 处的对数密度：

$$
\log p(\vy) = \underbrace{-\tfrac12\vy^\T\mK_y^{-1}\vy}_{\text{data fit}}
\;\underbrace{-\;\tfrac12\log\lvert\mK_y\rvert}_{\text{complexity}}
\;-\;\tfrac{n}{2}\log 2\pi.
$$ {#eq-bayes-log-evidence}

两项的作用方向相反。先验越宽，$\mK_y$ 越大；对于需要大权重的数据，这会改善数据拟合项，但行列式也随之增大，这是把概率摊到更多数据集上的代价。把 $\boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T$ 换成核矩阵，同一公式就是 @sec-marginal-likelihood 中为拟合高斯过程超参数而最大化的边际似然。

```{figure}
//| figure: bayes-line
//| label: fig-bayes-evidence
//| fig-cap: "直线模型对 4 个观测的对数证据（@eq-bayes-log-evidence），数值见权重空间下方的读数。对这组数据，对数证据在先验标准差约为 0.5 至 0.7 时达到峰值，向两侧下降：窄的先验无法达到数据所需的斜率，宽的先验则把概率浪费在数据已排除的直线上。数据仅作示意。"
points: "-0.8:-0.5,-0.3:0,0.2:0.35,0.7:0.9"
```

**扫描先验宽度**。把 $\sigma_p$ 从 0.1 调到 4，观察对数证据：它从约 $-4.8$ 升至 $-2.2$ 附近的最大值，再降到约 $-4.9$。在峰值处选取 $\sigma_p$ 的做法称为**第二类最大似然**（type II maximum likelihood），也称经验贝叶斯：针对先验的设置最大化证据，而不是针对权重最大化似然。

**同时调整噪声**。噪声水平明显过小时，数据拟合项的惩罚很重；明显过大时，每个数据集看似都合理，却没有一个概率较高。当噪声恰好能解释数据点在直线周围的散布时，证据最高。

证据也有一些已知的弱点。即使后验几乎不受先验宽度影响，证据仍然依赖于先验宽度：把本已模糊的先验再加宽一倍，拟合的直线几乎不变，证据却会降低。此外，对大量超参数最大化证据可能导致过拟合，即拟合了小数据集中的偶然特征；@sec-fitting-hyperparameters 将针对高斯过程讨论这种失效情形。尽管如此，证据仍是本书为模型设定超参数的标准工具。

## 没有闭式解时 {#sec-no-closed-form}

共轭对只是例外。Beta 先验适用于硬币，是因为似然是 $\theta$ 与 $1 - \theta$ 的幂；高斯先验适用于直线，是因为噪声服从高斯分布。改变其中任何一个，后验就不再属于原来的分布族。

本书最关心的是比较。一个人表示选项 $A$ 优于选项 $B$ 时，一种常用模型（详见 @sec-thurstone）把这一回答的概率设为 $\Phi\big((f_A - f_B)/\sigma\big)$，即把 @eq-gauss-phi 中的标准正态分布函数作用于效用之差。（此处 $\sigma$ 是差的噪声；@sec-thurstone 将其写为单个选项噪声的 $\sqrt{2}$ 倍。）若对差 $\Delta = f_A - f_B$ 取高斯先验，一次回答之后的后验正比于

$$
\N(\Delta;\, 0, s^2)\;\Phi(\Delta/\sigma),
$$

即高斯密度乘以一条 S 形曲线。乘积是偏斜的：保留了先验的上尾，切掉了下尾。它不是高斯分布；在对许多选项作出许多回答之后，其归一化积分的维数与选项个数相同。

处理这类后验的方法有四类，@sec-approx-inference 将比较这几类方法。**Laplace 近似**（Laplace approximation，@sec-laplace）用以后验峰值为中心的高斯分布代替后验，协方差取自峰值处的曲率。**期望传播**（expectation propagation，@sec-ep）与**变分推断**（variational inference，@sec-vi）按其他意义上的匹配选取高斯分布。**采样**（sampling）方法（@sec-mcmc）直接从后验中抽取样本，以样本平均代替每一个积分。前三种方法把问题重新归结为高斯分布，使 @sec-gaussian 的结论全部再次适用。@chu2005preference 的高斯过程偏好模型采用的就是 Laplace 近似，@sec-part-preferences 的许多内容都建立在这一模型之上。

还缺少一样工具。本章主张优化器应在能学到最多的地方评估，却没有说明如何度量学到的内容。@sec-information 将给出度量的单位。

## 习题 {#sec-bayes-exercises}

::: {.exercise #exr-bayes-pseudo-counts}
一枚硬币的先验为 $\mathrm{Beta}(2, 2)$，10 次抛掷中观测到 8 次正面。求后验及其均值、众数，以及最大似然估计。该先验相当于多少次抛掷？它在后验均值中的权重是多少？

::: {.solution}
后验为 $\mathrm{Beta}(2 + 8, 2 + 2) = \mathrm{Beta}(10, 4)$，均值为 $10/14 \approx 0.714$，众数为 $9/12 = 0.75$，而最大似然估计为 $8/10 = 0.8$。先验相当于 $n_0 = 4$ 次抛掷，均值为 0.5，因此由 @eq-bayes-shrinkage，后验均值为 $\frac{4}{14}(0.5) + \frac{10}{14}(0.8) = 0.143 + 0.571 = 0.714$：先验的权重为 $4/14 \approx 0.29$。
:::
:::

::: {.exercise #exr-bayes-one-point}
考虑直线模型，取 $\mSigma_p = \mI$，只在输入 $x_1$ 处有一个观测，于是 $\boldsymbol{\Phi}$ 为行向量 $\boldsymbol{\phi}_1^\T = (1, x_1)$。证明后验精度 $\mA$ 在权重空间的某个方向上保持先验方差不变。这是哪个方向？它对 @fig-bayes-line 中的直线意味着什么？

::: {.solution}
此时 $\mA = \mI + \sigma_n^{-2}\boldsymbol{\phi}_1\boldsymbol{\phi}_1^\T$。对任何与 $\boldsymbol{\phi}_1$ 正交的 $\mathbf{u}$，有 $\mA\mathbf{u} = \mathbf{u}$，故 $\mathbf{u}$ 是特征值为 1 的特征向量，该方向上的后验方差等于先验方差 1。沿 $\boldsymbol{\phi}_1$ 方向，特征值更大，为 $1 + \lVert\boldsymbol{\phi}_1\rVert^2/\sigma_n^2$，因此方差缩小。受约束的组合是 $\boldsymbol{\phi}_1^\T\vw = w_1 + w_2 x_1$，即直线在 $x_1$ 处的值。不受约束的方向 $\mathbf{u} = (-x_1, 1)^\T$ 使斜率改变一个单位、截距改变 $-x_1$，即让直线绕点 $x = x_1$ 旋转。抽取的直线绕观测点转动，原因正在于此。
:::
:::

::: {.exercise #exr-bayes-ridge}
证明当 $\mSigma_p = \sigma_p^2\mI$ 时，后验均值 @eq-bayes-blr-posterior 使岭回归目标函数 $\lVert\vy - \boldsymbol{\Phi}\vw\rVert^2 + \lambda\lVert\vw\rVert^2$ 取最小值，其中 $\lambda = \sigma_n^2/\sigma_p^2$。当 $\sigma_p \to \infty$ 时，后验均值变成什么？

::: {.solution}
令目标函数的梯度为零，得 $-2\boldsymbol{\Phi}^\T(\vy - \boldsymbol{\Phi}\vw) + 2\lambda\vw = \mathbf{0}$，所以 $\vw = (\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + \lambda\mI)^{-1}\boldsymbol{\Phi}^\T\vy$。后验均值为 $\bar{\vw} = \sigma_n^{-2}(\sigma_n^{-2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + \sigma_p^{-2}\mI)^{-1}\boldsymbol{\Phi}^\T\vy$；在逆矩阵内外同乘 $\sigma_n^2$，即变为 $(\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + (\sigma_n^2/\sigma_p^2)\mI)^{-1}\boldsymbol{\Phi}^\T\vy$，与上式相同。后验是高斯分布，其均值即众数，因此这也是最大后验估计。当 $\sigma_p \to \infty$ 时，$\lambda \to 0$，后验均值趋于最小二乘解 $(\boldsymbol{\Phi}^\T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\T\vy$，即最大似然估计，前提是 $\boldsymbol{\Phi}^\T\boldsymbol{\Phi}$ 可逆。
:::
:::

::: {.exercise #exr-bayes-evidence}
在均匀先验下，求 3 次抛掷全为正面这一特定序列的证据，并与 @ex-bayes-fair-coin 中的公平硬币模型比较。数据支持哪个模型？比值是多少？连续出现多少次正面，偏向未知的模型才会以 10 倍的比值胜出？

::: {.solution}
偏向未知的模型给出 $B(4, 1) = 3!\,0!/4! = 1/4$，公平硬币给出 $2^{-3} = 1/8$，因此数据以 2 倍的比值支持偏向未知的模型。连续 $n$ 次正面时，比值为 $\frac{n!/(n+1)!}{2^{-n}} = \frac{2^n}{n + 1}$，$n = 6$ 时为 $64/7 \approx 9.1$，$n = 7$ 时为 $128/8 = 16$。需要连续 7 次正面才能达到 10 倍的比值。一连串正面确实是硬币不公平的证据，但不像直觉以为的那样有压倒性，因为灵活的模型必须把概率分摊到每一种可能的偏向上。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @gelman2013bayesian 第 2 章讨论单参数模型，从二项分布及其 Beta 先验讲起；全书是应用贝叶斯建模（包括先验预测检验）的标准参考书。
- @bishop2006pattern 第 2.1 节讲 Beta 分布和硬币，第 3 章讲贝叶斯线性回归及其预测分布、用于模型比较的证据，记号与本书相同。
- @rasmussen2006gaussian 第 2.1 节从权重空间的视角推导贝叶斯线性回归，随后过渡到高斯过程，@sec-function-space 走的正是这一步。
- @mackay2003information 第 28 章通过证据解释 Occam 剃刀，书中的图使这一论证广为人知。
- @thompson1933likelihood 提出了根据数据比较两个未知成功概率的问题，并随之提出了如今以他的名字命名的采样规则。
