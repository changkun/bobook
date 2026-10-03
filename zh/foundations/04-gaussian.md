---
status: done
synopsis: "全书赖以运转的分布：它在一维和多维中的形状；高斯分布经线性映射后为何仍是高斯分布，以及如何由此得到采样方法；借助 Schur 补推导出的条件化公式，高斯过程回归直接沿用这一公式。"
sources: ["textbooks", "Bishop 2006, sec. 2.3", "Rasmussen and Williams 2006, app. A"]
---

# 高斯分布 {#sec-gaussian}

前两章分别建立了一套语言和一套工具。@sec-probability 给出了不确定量的推理规则：分布、用于边际化的加法规则、用于条件化的乘法规则。@sec-linear-algebra 则给出了同时涉及许多量时这些规则所对应的矩阵：协方差矩阵及其 Cholesky 因子、分块矩阵。本章把两者结合于一个分布，即高斯分布，本书其余内容都以它为基础。

单独一个分布之所以值得用一整章来讲，是由贝叶斯优化器使用信念的方式决定的。优化器对未知函数持有信念，每次评估后更新信念，再依据信念决定下一步的评估位置。每一步都是对概率分布的一次运算，多数分布的这些运算是没有闭式解的积分。而对高斯分布，每种运算只需几行线性代数就能精确求解：高斯分布经线性映射仍是高斯分布，略去一部分坐标仍是高斯分布，以一部分坐标为条件仍是高斯分布。高斯过程（@sec-gp-definition）是函数值上的高斯分布，因此高斯过程回归和贝叶斯优化之所以可以计算，正是依靠这三条封闭性。

本章先讨论单变量情形，此时公式一目了然；再推广到多变量，此时分布的形状由协方差矩阵决定；然后依次推导这三种运算。最后一种运算即条件化，是本章的主要结果，@sec-gp-regression 将直接沿用。其后一节区分两种容易混淆的高斯分布组合方式，为 @sec-bayesian-inference 的贝叶斯更新作准备。

## 一维情形 {#sec-gaussian-1d}

设某次训练将采用新的学习率，尚未开始运行。根据类似训练的经验，预期验证准确率约为 0.82，向上或向下偏离超过 0.06 的情况很少。这一信念有中心，有散布范围，对向上和向下的偏差同等看待，且大偏差比小偏差少见。高斯分布（也称正态分布）是用两个数表达这类信念的标准方式。

随机变量 $X$ 服从均值为 $\mu$、方差为 $\sigma^2 > 0$ 的高斯分布，记作 $X \sim \N(\mu, \sigma^2)$，是指其密度为

$$
\N(x;\, \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right).
$$ {#eq-gauss-density-1d}

式中分号把求密度的点 $x$ 与参数 $\mu$、$\sigma^2$ 隔开。如 @sec-continuous 所述，密度不是概率，概率是密度曲线下的面积。

这个公式可以由内向外解读。指数 $-(x - \mu)^2 / 2\sigma^2$ 是一条开口向下的抛物线，在 $x = \mu$ 处取峰值。指数函数把抛物线变为钟形：在峰值处等于 1，随抛物线取负值而迅速下降。前面的系数对钟形作缩放，使总面积为一。取对数可以消去指数函数：

$$
\log \N(x;\, \mu, \sigma^2) = -\frac{(x - \mu)^2}{2\sigma^2} - \tfrac12 \log(2\pi\sigma^2).
$$

高斯密度是二次函数的指数。其逆命题是本章大部分推导的依据：任何密度，只要其对数是 $x$ 的开口向下的二次函数，就是高斯分布，且参数可以直接从系数读出。若

$$
\log p(x) = -\tfrac12 \alpha x^2 + \beta x + \text{const}, \quad \alpha > 0,
\qquad\text{then}\qquad
p(x) = \N\!\left(x;\, \beta/\alpha,\; 1/\alpha\right).
$$ {#eq-gauss-recipe}

只需配方即可看出这一点：$-\tfrac12\alpha x^2 + \beta x =
-\tfrac12\alpha(x - \beta/\alpha)^2 + \beta^2/2\alpha$，最后一项是常数，可并入归一化常数。
系数 $\alpha$ 是方差的倒数，称为**精度**（precision）。后文要证明某个分布是高斯分布时，都会这样做：先证明其对数密度是二次的，再套用这一公式。

两个参数的含义与其名称一致。均值是期望值 $\E[X] = \mu$，方差是偏差平方的期望 $\Var[X] = \E[(X - \mu)^2] = \sigma^2$（@sec-expectation）。标准差 $\sigma$ 是方差的平方根，与 $x$ 的单位相同，因此直观思考时用标准差。上述关于准确率的信念约为 $\N(0.82, 0.03^2)$：标准差为 0.03，偏差达到其两倍的情况很少见。

### 标准正态分布 {#sec-gauss-standard}

任何高斯分布都可由同一个参考分布经平移和伸缩得到，这个参考分布就是标准正态分布 $Z \sim \N(0, 1)$。若 $X \sim \N(\mu, \sigma^2)$，则 $Z = (X - \mu)/\sigma$ 服从标准正态分布；反之，$X = \mu + \sigma Z$。值 $z = (x - \mu)/\sigma$ 表示 $x$ 比均值高出多少个标准差，称为其 z 分数。标准正态分布有专用记号，全书通用：

$$
\phi(z) = \frac{1}{\sqrt{2\pi}}\, e^{-z^2/2},
\qquad
\Phi(z) = \int_{-\infty}^{z} \phi(t)\,\dd t.
$$ {#eq-gauss-phi}

其中 $\phi$ 是密度，$\Phi$ 是累积分布函数：$\Phi(z)$ 是 $Z \le z$ 的概率。$\Phi$ 无法用初等函数表示，程序库借助误差函数 $\Phi(z) = \tfrac12\left(1 + \operatorname{erf}(z/\sqrt{2})\right)$ 计算，可达到浮点数的全部精度。

经过标准化，关于 $X$ 的任何概率问题都可化为关于 $\Phi$ 的问题：

$$
\Prob(a \le X \le b) = \Phi\!\left(\frac{b - \mu}{\sigma}\right) - \Phi\!\left(\frac{a - \mu}{\sigma}\right).
$$ {#eq-gauss-interval}

有几个数值值得记住。区间 $\mu \pm \sigma$ 包含 68.3% 的概率，$\mu \pm 1.96\sigma$ 包含 95%，$\mu \pm 2\sigma$ 包含 95.4%，$\mu \pm 3\sigma$ 包含 99.7%。尾部衰减很快：无论朝哪个方向，偏离均值超过五个标准差的概率约为 $5.7 \times 10^{-7}$。

::: {.example #ex-gauss-beat-incumbent title="新的训练能否胜过目前最好的结果？"}
沿用对新训练准确率 $f$ 的信念 $f \sim \N(0.82, 0.03^2)$，并设目前最好的一次训练达到 0.85。由 @eq-gauss-interval，新训练表现更好的概率为

$$
\Prob(f > 0.85) = 1 - \Phi\!\left(\frac{0.85 - 0.82}{0.03}\right) = 1 - \Phi(1) \approx 0.159.
$$

尽管信念的中心比最好结果低 0.03，新训练仍有约六分之一的机会胜出。若在每个候选输入处用高斯过程后验计算这个数，得到的就是称为改进概率（@sec-pi）的采集函数。期望改进（@sec-ei）同样由 $\phi$ 和 $\Phi$ 构成。
:::

### 为什么选择高斯分布 {#sec-gauss-why}

读者也许会问，为什么默认采用这条钟形曲线，而不是别的分布。理由有三条，性质各不相同。

第一条来自定理。中心极限定理指出：设有许多相互独立的随机量，方差有限，且没有哪一个占主导，则无论各项服从什么分布，其和经标准化后都近似服从高斯分布[@blitzstein2019introduction]。测量噪声往往是许多微小扰动（如热涨落、时序抖动、舍入）之和，因而接近高斯分布。一个经典的演示：取十二个 $[0, 1]$ 上相互独立的均匀随机数，求和后减去 6，所得量均值为 0、方差为 1，其直方图已经很难与 $\phi$ 区分。

第二条是原则。在实数轴上均值和方差给定的所有分布中，高斯分布的熵最大；熵度量分布的分散程度，@sec-entropy 将给出其精确定义[@cover2006elements, 第 12 章]。如果只愿意确定中心和散布范围，高斯分布就是不作任何额外假设的选择。

第三条是计算方便，这也是本书在前两条理由不成立时仍采用高斯分布的原因：本章其余部分的每种运算都有闭式解。方便是选择模型的理由，却不能证明世界服从高斯分布。真实的量可能偏斜、有界或重尾，极端值远比上面的 $5.7 \times 10^{-7}$ 所暗示的常见。两个选项之间的比较是 @sec-part-preferences 的核心观测，它完全不服从高斯分布；@sec-approx-inference 说明如何用高斯分布近似由此得到的后验，使本章的工具依然适用。

## 多维情形 {#sec-gaussian-nd}

贝叶斯优化器的信念从来不只涉及单个数。它同时对目标函数在许多输入处的值持有信念，而且这些信念相互关联：若学习率为 0.010 时准确率很高，那么 0.011 时准确率很可能也高。一组彼此独立的一维高斯分布无法表达这种关联，需要的是值向量上的联合分布，它记录每一对值如何共同变化。

先看两个独立的坐标。若 $x_1 \sim \N(0, \sigma_1^2)$ 与 $x_2 \sim \N(0, \sigma_2^2)$ 相互独立，则联合密度是两个密度之积（@sec-independence），而指数函数相乘即指数相加：

$$
p(x_1, x_2) \propto \exp\!\left(-\frac{x_1^2}{2\sigma_1^2} - \frac{x_2^2}{2\sigma_2^2}\right).
$$

指数为常数的地方密度也为常数，即在曲线 $x_1^2/\sigma_1^2 + x_2^2/\sigma_2^2 = r^2$ 上。这些曲线是轴沿坐标方向的椭圆，$\sigma_1 = \sigma_2$ 时为圆。为了使两个坐标相互关联，允许指数中的二次式含有交叉项 $x_1 x_2$。此时椭圆发生倾斜，较大的 $x_1$ 会使较大的 $x_2$ 更可能或更不可能出现，取决于倾斜的方向。这些信息全部记录在一个矩阵中。

这个矩阵就是 @sec-prob-covariance 中的协方差矩阵 $\mSigma$。对均值向量为 $\vmu$ 的随机向量 $\vx = (x_1, \dots, x_d)^\T$，其元素为 $\Sigma_{ij} = \Cov[x_i, x_j] = \E[(x_i - \mu_i)(x_j - \mu_j)]$，对角线上是方差。协方差矩阵对称半正定（@sec-positive-definite），把某个元素除以相应的两个标准差，即得相关系数 $\rho_{ij} = \Sigma_{ij} / \sqrt{\Sigma_{ii}\Sigma_{jj}}$。

::: {.definition #def-gauss-mvn title="多元高斯分布"}
随机向量 $\vx \in \R^d$ 服从均值为 $\vmu \in \R^d$、协方差矩阵为正定矩阵 $\mSigma \in \R^{d \times d}$ 的高斯分布，记作 $\vx \sim \N(\vmu, \mSigma)$，是指其密度为

$$
\N(\vx;\, \vmu, \mSigma) = \frac{1}{(2\pi)^{d/2}\, \lvert\mSigma\rvert^{1/2}}
\exp\!\left(-\tfrac12 (\vx - \vmu)^\T \mSigma^{-1} (\vx - \vmu)\right),
$$ {#eq-gauss-density}

其中 $\lvert\mSigma\rvert$ 是 $\mSigma$ 的行列式。
:::

式中各部分都与一维情形对应。$d = 1$ 且 $\mSigma = [\sigma^2]$ 时，此式即 @eq-gauss-density-1d。指数仍是二次的，现在是向量 $\vx - \vmu$ 的二次型，协方差的逆 $\mSigma^{-1}$ 相当于一维中的 $1/\sigma^2$。协方差的逆称为**精度矩阵**（precision matrix），下面的若干推导用它表述最为自然。归一化因子中的行列式度量分布所占的体积（@sec-determinants），相当于一维中的 $\sigma$：分布越分散，峰越低，以保证总概率为一。

配方公式 @eq-gauss-recipe 可以直接推广到多维。若某个密度满足

$$
\log p(\vx) = -\tfrac12 \vx^\T \bm{\Lambda} \vx + \mathbf{h}^\T \vx + \text{const}
$$

其中 $\bm{\Lambda}$ 为正定矩阵，$\mathbf{h}$ 为向量，则 $p$ 是精度为 $\bm{\Lambda}$ 的高斯分布，即

$$
\vx \sim \N\!\left(\bm{\Lambda}^{-1}\mathbf{h},\; \bm{\Lambda}^{-1}\right).
$$ {#eq-gauss-recipe-nd}

配方的做法与前面相同，展开右边即可验证：

$$
-\tfrac12\vx^\T\bm{\Lambda}\vx + \mathbf{h}^\T\vx
= -\tfrac12(\vx - \bm{\Lambda}^{-1}\mathbf{h})^\T\bm{\Lambda}(\vx - \bm{\Lambda}^{-1}\mathbf{h})
+ \tfrac12\mathbf{h}^\T\bm{\Lambda}^{-1}\mathbf{h}.
$$

最后一项与 $\vx$ 无关。

### 形状 {#sec-gauss-shape}

指数中的量有专门的名称。$\vx$ 到 $\vmu$ 的 **Mahalanobis 距离**（Mahalanobis distance）定义为

$$
r(\vx) = \sqrt{(\vx - \vmu)^\T \mSigma^{-1} (\vx - \vmu)},
$$ {#eq-gauss-mahalanobis}

即多元情形的 z 分数。一维时它等于 $\lvert x - \mu\rvert / \sigma$，即偏离均值的标准差个数。一般而言，Mahalanobis 距离在每个方向上都以分布自身的散布为单位度量距离：一个点即使按普通距离离均值很远，只要位于分布较宽的方向上，按 Mahalanobis 距离仍可能很近。密度只通过 $r$ 依赖于 $\vx$，因此其等高线是 $r$ 为常数的点集：二维中为椭圆，更高维中为椭球。

这些椭圆的朝向如何确定？将协方差作特征分解 $\mSigma = \mathbf{U}\bm{\Lambda}_{\mathrm{e}}\mathbf{U}^\T$，其中 $\mathbf{U}$ 的各列是标准正交的特征向量 $\mathbf{u}_i$，$\bm{\Lambda}_{\mathrm{e}}$ 的对角线上是特征值 $\lambda_i$（@sec-eigen）。在旋转后的坐标 $\mathbf{y} = \mathbf{U}^\T(\vx - \vmu)$ 下，二次型变为 $\sum_i y_i^2/\lambda_i$，不含交叉项。因此每条等高线的轴都沿特征向量方向，Mahalanobis 距离为 $r$ 的椭圆半轴长为 $r\sqrt{\lambda_i}$。特征值是沿各主方向的方差，其乘积为 $\lvert\mSigma\rvert$。多元高斯分布在某个旋转后的坐标系中是与坐标轴对齐的钟形，这个坐标系由特征向量给出。

设两个坐标的标准差为 $\sigma_1, \sigma_2$，相关系数为 $\rho$，则协方差矩阵及其行列式为

$$
\mSigma = \begin{bmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{bmatrix},
\qquad
\lvert\mSigma\rvert = \sigma_1^2\sigma_2^2(1 - \rho^2).
$$ {#eq-gauss-2d-cov}

下图画出了均值为零时的这一情形。

```{figure}
//| figure: gauss-2d
//| label: fig-gauss-2d
//| fig-cap: "均值为零、协方差为 @eq-gauss-2d-cov 的二维高斯分布。阴影椭圆是 Mahalanobis 距离为 1、2、3 的等高线；虚线是特征向量轴；上方和右侧的条带分别显示 $x_1$ 与 $x_2$ 的边际密度。滑块用于设置两个标准差和相关系数。“样本”与“条件化”两个视图分别对应 @sec-gaussian-linear 和 @sec-gaussian-conditioning。"
```

**把 $\rho$ 设为零，并让两个标准差相等**。椭圆变为圆。分布在各个方向上都相同，特征向量可以指向任意方向。

**把 $\rho$ 调向 0.95**。椭圆沿对角线收窄，细如针状。两个标准差都为 1 时，特征值为 $1 + \rho$ 和 $1 - \rho$，前者趋于 2，后者与行列式趋于零：分布集中在一条直线附近，一旦知道 $x_1$，$x_2$ 也就几乎确定。$\rho = \pm 1$ 时，协方差矩阵奇异，@eq-gauss-density 的密度不存在，Cholesky 分解也会失败。在对角线上加一个小的抖动项，正是为了修复这类浮点问题（@sec-gp-computation）。

**让 $\rho$ 取负值**。椭圆向另一侧倾斜：此时较大的 $x_1$ 伴随较小的 $x_2$。

**调节 $\rho$ 时观察两个条带**。条带保持不变，原因见 @sec-gaussian-marginal。

每个椭圆包含多少概率？比一维的数值所暗示的要少。二维中，Mahalanobis 距离为 $r$ 的椭圆内的概率为 $1 - e^{-r^2/2}$：$r = 1$ 以内为 39%，$r = 2$ 以内为 86%，$r = 3$ 以内为 99%。要包含 95% 的概率，需要 $r \approx 2.45$，而不是 1.96。维度越高，差距越大。

::: {.aside title="高维中的概率集中在何处"}
对 $d$ 维标准正态分布，Mahalanobis 距离的平方 $r^2 = \sum_i z_i^2$ 是 $d$ 个独立标准正态变量的平方和。每一项均值为 1、方差为 2，因此 $r^2$ 的均值为 $d$、标准差为 $\sqrt{2d}$，$r$ 本身与 $\sqrt{d}$ 的差距约在一个单位以内。在 100 维中，几乎全部概率都位于距均值约 10 个单位的一层薄壳内，而峰值附近（密度最高处）几乎不含概率。一维中典型值位于均值附近，这一直觉在高维中不再成立；这种几何性质也是高维贝叶斯优化困难的原因之一（@sec-high-dimensions）。
:::

高斯分布还有一个特有的性质。$\mSigma$ 为对角矩阵时，二次型没有交叉项，密度分解为一维密度之积，各坐标相互独立。因此，对于高斯向量，不相关即意味着独立。其他分布则不然（@sec-independence）：@exr-gauss-not-joint 给出两个变量，各自服从高斯分布且互不相关，却并不独立，原因在于它们不服从联合高斯分布。

## 线性映射与采样 {#sec-gaussian-linear}

两个实际问题会引出同一个结果。第一，若 $f(\vx_1)$ 与 $f(\vx_2)$ 联合服从高斯分布，二者的差或平均值服从什么分布？@sec-part-preferences 中，每当一个人比较两个选项，都要用到这个差。第二，随机数生成器产生的是相互独立的标准正态随机数，如何把它们变成协方差任意的 $\N(\vmu, \mSigma)$ 的样本？凡是要画出从模型中抽取的函数，都需要这样的样本；Thompson 采样（@sec-thompson）同样需要，这一规则从模型中抽取一个可能的目标函数，在该样本取最大值处评估。

两个问题的答案都来自同一条封闭性。设 $\vx \sim \N(\vmu, \mSigma)$ 为 $d$ 维向量，$\mA$ 为 $m \times d$ 矩阵，$\mathbf{c}$ 为 $\R^m$ 中的向量，则

$$
\vy = \mA\vx + \mathbf{c} \;\sim\; \N\!\left(\mA\vmu + \mathbf{c},\; \mA\mSigma\mA^\T\right).
$$ {#eq-gauss-affine}

高斯分布经线性映射仍是高斯分布。均值像点一样随映射变换，协方差则夹在矩阵与其转置之间。

有一点需要注意。@def-gauss-mvn 中的密度要求协方差正定，而只有当 $\mA$ 的任何一行都不是其余各行的线性组合时，$\mA\mSigma\mA^\T$ 才正定。若这一条件不满足（例如 $\mA$ 的行数多于列数），$\vy$ 在下面第 6 步的意义下仍是高斯的，但局限于一个更低维的仿射子空间，在 $\R^m$ 上没有密度。两种情况下，均值和协方差的公式都成立。

::: {.derivation title="高斯分布的线性映射"}
1. 由期望的线性性（@sec-expectation），$\E[\vy] = \mA\,\E[\vx] + \mathbf{c} = \mA\vmu + \mathbf{c}$。
2. 减去均值，得 $\vy - \E[\vy] = \mA(\vx - \vmu)$。
3. 由协方差矩阵的定义，
   $\Cov[\vy] = \E\big[(\vy - \E[\vy])(\vy - \E[\vy])^\T\big]
   = \E\big[\mA(\vx - \vmu)(\vx - \vmu)^\T\mA^\T\big]$。
4. $\mA$ 是常数，可以移到期望之外：$\Cov[\vy] = \mA\,\E\big[(\vx - \vmu)(\vx - \vmu)^\T\big]\mA^\T = \mA\mSigma\mA^\T$。
5. $\vy$ 不只是具有这一均值和协方差的某个分布，它确实是高斯分布，这一点还需要再论证。当 $\mA$ 为可逆方阵时，把 $\vx = \mA^{-1}(\vy - \mathbf{c})$ 代入 @eq-gauss-density，所得指数是 $\vy$ 的二次函数，由 @eq-gauss-recipe-nd 知 $\vy$ 是高斯的。
6. 对一般的 $\mA$（例如构成差的单行矩阵），采用一个等价的定义：向量是高斯的，当且仅当其各坐标的任一线性组合都服从一维高斯分布[@blitzstein2019introduction]。$\vy$ 各坐标的线性组合也是 $\vx$ 各坐标的线性组合，因而是高斯的，所以 $\vy$ 也是高斯的。
:::

### 用 Cholesky 因子采样 {#sec-gauss-sampling}

采样就是沿着有用的方向运用这一映射。设 $\vz \sim \N(\mathbf{0}, \mI)$ 是由 $d$ 个独立标准正态随机数组成的向量，任何数值计算库都提供这样的随机数（由均匀随机数构造它们的经典方法是 @box1958note 的变换）。任取满足 $\mL\mL^\T = \mSigma$ 的矩阵 $\mL$，令

$$
\vx = \vmu + \mL\vz.
$$ {#eq-gauss-sample}

由 @eq-gauss-affine，$\vx$ 服从均值为 $\vmu$、协方差为 $\mL\mI\mL^\T = \mSigma$ 的高斯分布。$\mL$ 通常取 @sec-cholesky 中的 Cholesky 因子，即对角线为正的下三角矩阵：它对任何正定的 $\mSigma$ 都存在，计算一次需要 $O(d^3)$ 次运算，此后每抽取一个样本只需一次三角矩阵与向量的乘法[@rasmussen2006gaussian, 附录 A.2]。其他平方根同样可用，例如由特征分解得到的 $\mathbf{U}\bm{\Lambda}_{\mathrm{e}}^{1/2}$。它会把给定的 $\vz$ 映射到不同的点，但这些点的分布不变。

二维时，@eq-gauss-2d-cov 的 Cholesky 因子可以直接写出，再乘以 $\vz$，便可看出每个坐标的构成：

$$
\mL = \begin{bmatrix} \sigma_1 & 0 \\ \rho\sigma_2 & \sigma_2\sqrt{1 - \rho^2} \end{bmatrix},
\qquad
\begin{aligned}
x_1 &= \sigma_1 z_1, \\
x_2 &= \rho\sigma_2 z_1 + \sigma_2\sqrt{1 - \rho^2}\, z_2.
\end{aligned}
$$ {#eq-gauss-chol-2d}

展开乘积即可验证 $\mL\mL^\T = \mSigma$。这个公式还从生成机制上说明了相关性的含义。坐标 $x_2$ 一部分来自驱动 $x_1$ 的同一个随机数 $z_1$，另一部分来自新的随机数 $z_2$；其方差中有 $\rho^2$ 的比例来自共享部分。$\rho = 0$ 时两个坐标毫无共享，$\rho = \pm 1$ 时二者完全共享。

```{figure}
//| figure: gauss-2d
//| label: fig-gauss-samples
//| fig-cap: "用 Cholesky 因子采样（@eq-gauss-sample）。灰点是标准正态向量 $\vz$ 的样本；蓝点是同一批样本经映射 $\mL\vz$ 后的位置，其协方差由滑块设置。橙色箭头标出六个样本从 $\vz$ 到 $\mL\vz$ 的移动。虚线椭圆是 Mahalanobis 距离为 2 的等高线。“新样本”按钮重新抽取一组样本。"
view: samples
```

**观察箭头**。所有灰点都由同一个矩阵移动。$\mL$ 是下三角矩阵，因此 $x_1 = \sigma_1 z_1$ 只依赖于 $z_1$；$\sigma_1 = 1$ 时映射不改变第一个坐标，箭头都是竖直的。$\rho$ 为正时，映射给第二个坐标加上 $\rho\sigma_2 z_1$，使左侧的点下移、右侧的点上移，并把 $z_2$ 缩小为原来的 $\sqrt{1 - \rho^2}$ 倍。圆形点云于是变为倾斜的椭圆。

**比较样本相关系数与 $\rho$**。读数给出蓝点的相关系数，它与 $\rho$ 相差百分之几，换一组样本，误差也随之改变：这是采样噪声，$n$ 个点时其标准差约为 $(1 - \rho^2)/\sqrt{n}$。

**把 $\rho$ 设为零，并让两个标准差不相等**。此时 $\mL$ 是对角矩阵，只沿坐标轴拉伸点云。

对高斯过程，$\vx$ 存放输入网格上的函数值，每个网格点对应一个坐标；后验样本之所以只能在几千个点的网格上抽取，正是受分解的立方级代价所限（@sec-gp-posterior-samples）。随着输入维度增加，网格很快就不再可行：每个轴取 20 个点，直线上是 20 个值，正方形上是 400 个，三维立方体中是 8,000 个，其协方差矩阵有 6,400 万个元素。输入超过两三维时，样本改在几千个分散的候选点上抽取，而不在网格上抽取。

本节开头提出的另一个问题，是两个函数值之差。

::: {.example #ex-gauss-difference title="两个函数值之差"}
设 $A = f(\vx_1)$ 与 $B = f(\vx_2)$ 联合服从高斯分布，均值为 $\mu_A, \mu_B$，方差为 $v_A, v_B$，协方差为 $c$。差 $D = A - B$ 是由单行 $(1, -1)$ 给出的映射，由 @eq-gauss-affine 知它服从均值为 $\mu_A - \mu_B$ 的高斯分布，方差为

$$
\begin{bmatrix} 1 & -1 \end{bmatrix}
\begin{bmatrix} v_A & c \\ c & v_B \end{bmatrix}
\begin{bmatrix} 1 \\ -1 \end{bmatrix}
= v_A + v_B - 2c.
$$

协方差以负号进入方差。两个值强正相关时（例如光滑高斯过程下两个相邻的输入），二者之差的不确定性远小于任一值本身的不确定性。模型可以确信两个相似选项中哪一个更好，却不确定其中任何一个究竟有多好。@sec-thurstone 在这个差的基础上建立人类比较的模型，@sec-eubo 则用它为查询对打分。
:::

## 边际化 {#sec-gaussian-marginal}

高斯过程描述无穷多个函数值，而计算机只能存放有限多个。要使这种做法成立，关于少数几个值的信念，就不能取决于还选择追踪了其他哪些值。用 @sec-joint-marginal-conditional 的术语来说，需要的是子向量的边际分布，加法规则通过对其余变量积分得到它。对多数联合密度，这个积分是计算的难点；对高斯分布则毫无代价。

把向量分为两块：保留的坐标 $\mathbf{a}$ 与舍弃的坐标 $\mathbf{b}$，并相应地划分均值和协方差：

$$
\begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix}
\sim \N\!\left(
\begin{bmatrix} \vmu_a \\ \vmu_b \end{bmatrix},\;
\begin{bmatrix} \mSigma_{aa} & \mSigma_{ab} \\ \mSigma_{ab}^\T & \mSigma_{bb} \end{bmatrix}
\right).
$$ {#eq-gauss-partition}

对角块 $\mSigma_{aa}$ 和 $\mSigma_{bb}$ 是各块内部的协方差，$\mSigma_{ab}$ 是 $\mathbf{a}$ 的坐标与 $\mathbf{b}$ 的坐标之间的协方差。$\mathbf{a}$ 的边际分布为

$$
\mathbf{a} \sim \N(\vmu_a, \mSigma_{aa}).
$$ {#eq-gauss-marginal}

对高斯分布做边际化，就是读出一个子块。利用上一节的结果，证明只需一行：保留 $\mathbf{a}$、舍弃 $\mathbf{b}$ 是矩阵为 $[\mI \;\; \mathbf{0}]$ 的线性映射（单位块旁接一个零块），由 @eq-gauss-affine 得均值 $[\mI \;\; \mathbf{0}]\vmu = \vmu_a$，协方差 $[\mI \;\; \mathbf{0}]\mSigma[\mI \;\; \mathbf{0}]^\T = \mSigma_{aa}$。加法规则所需的积分由此一次性完成；也可以通过配方直接计算[@bishop2006pattern, 第 2.3.2 节]。

有两个推论在后文很重要。第一，边际分布具有一致性：$\mathbf{a}$ 的分布既不依赖于 $\mathbf{b}$ 包含多少其他坐标，也不依赖于包含哪些坐标。高斯过程正是依靠这一点才得以良定义，因为其定义只规定有限个函数值的联合分布（@sec-gp-definition）。第二，边际分布舍弃了交叉协方差 $\mSigma_{ab}$，也就舍弃了两块之间相互提供的全部信息。在 @fig-gauss-2d 中，图上方和右侧的条带就是两个边际分布；移动相关系数滑块会旋转并压扁联合密度，两个条带却都不变。许多联合分布具有相同的边际分布。

$x_2$ 的边际分布回答的问题是：忽略 $x_1$ 时，对 $x_2$ 应持什么信念？下一节回答另一个问题：知道 $x_1$ 之后，对 $x_2$ 应持什么信念？

## 条件化 {#sec-gaussian-conditioning}

本书其余内容都以这一运算为基础。贝叶斯优化器已在几个输入处评估了目标函数，需要据此得到对目标函数在其他各处取值的信念。若对所有这些输入处函数值的先验信念是联合高斯分布，问题就化为：观测到高斯向量的一部分坐标之后，其余坐标服从什么分布？与边际分布不同，答案必须用到观测值。

### 钟形曲面的切片 {#sec-gauss-slice}

先看二维。由乘法规则，给定 $x_1 = a$ 时 $x_2$ 的条件密度为

$$
p(x_2 \given x_1 = a) = \frac{p(a, x_2)}{p(a)}.
$$

分子是联合密度沿竖直线 $x_1 = a$ 的取值，即钟形曲面的一个切片。分母不依赖于 $x_2$，它只是把切片重新缩放到面积为一。因此条件密度的形状就是切片的形状。沿着切片，联合密度的指数是 $x_2$ 的二次函数，因为固定二元二次式中的一个变量后，剩下的是另一变量的二次式。所以切片本身也是钟形，由 @eq-gauss-recipe 知其为高斯分布。

计算这个二次式，即可得到切片的均值和方差。设协方差为 @eq-gauss-2d-cov，均值为 $\mu_1, \mu_2$，则

$$
x_2 \given x_1 = a \;\sim\; \N\!\left(\mu_2 + \rho\,\frac{\sigma_2}{\sigma_1}(a - \mu_1),\;\; \sigma_2^2(1 - \rho^2)\right).
$$ {#eq-gauss-cond-2d}

@exr-gauss-cond-2d 将由下面的一般公式推出这一结果。式中每一部分都有明确的含义。观测通过其 z 分数 $(a - \mu_1)/\sigma_1$ 进入公式。$x_2$ 的均值偏离 $\mu_2$ 的量，以 $x_2$ 自身的标准差为单位，等于该 z 分数的 $\rho$ 倍：若 $x_1$ 比其均值高一个标准差，则预测 $x_2$ 比其均值高 $\rho$ 个标准差。方差缩小为原来的 $1 - \rho^2$ 倍，这正是 $x_2$ 的方差中 $x_1$ 无法解释的比例，而且与 $a$ 完全无关。

```{figure}
//| figure: gauss-2d
//| label: fig-gauss-condition
//| fig-cap: "条件化即切片（@eq-gauss-cond-2d）。橙色线标出观测值 $x_1$；在图上拖动或使用按钮可以移动它。右侧条带比较 $x_2$ 的边际密度（蓝色）与给定观测时的条件密度（橙色），切片上的橙色粗线标出条件均值及其 95% 区间。品红色虚线描出各观测值对应的条件均值。"
view: condition
```

**把切片从左拖到右**。条件密度沿虚线滑动，宽度保持不变。宽度只取决于相关系数，与观测值无关。

**把 $\rho$ 调向 $\pm 0.95$**。条件密度急剧收窄，读数显示消除的方差比例 $\rho^2$ 超过 90%。$\rho = 0$ 时没有消除任何方差，条件分布等于边际分布：与 $x_2$ 不相关的观测不提供关于它的任何信息。

**比较虚线与椭圆**。条件均值线不是椭圆的长轴，而是比长轴更平缓。它穿过每个椭圆的最左点和最右点，因为沿一条竖直切片，密度在切片恰好与某个椭圆相切处最高。

这一现象由来已久。Francis Galton 注意到，身材异常高的父母，其子女平均而言不像父母那样异常，他将这一效应称为向平庸回归（regression toward mediocrity）[@galton1886regression]；统计学术语“回归”（regression）即源于此。在两个标准差相等时的 @eq-gauss-cond-2d 中，$x_2$ 的预测偏差是 $x_1$ 观测偏差的 $\rho$ 倍，只要 $\lvert\rho\rvert < 1$，就更接近均值。并没有什么力量把子女拉回均值。这种收缩只是对相关的高斯分布做条件化的结果。

### 一般公式 {#sec-gauss-conditioning-general}

同样的推理适用于任意维数，只需以分块代替数。按 @eq-gauss-partition 把向量分为已观测块 $\mathbf{a}$ 与未观测块 $\mathbf{b}$，则

$$
\mathbf{b} \given \mathbf{a} \;\sim\; \N\!\left(
\vmu_b + \mSigma_{ab}^\T \mSigma_{aa}^{-1} (\mathbf{a} - \vmu_a),\;\;
\mSigma_{bb} - \mSigma_{ab}^\T \mSigma_{aa}^{-1} \mSigma_{ab}
\right).
$$ {#eq-gauss-conditional}

与二维情形对照：矩阵 $\mSigma_{ab}^\T\mSigma_{aa}^{-1}$ 对应 $\rho\sigma_2/\sigma_1 = \Sigma_{12}/\Sigma_{11}$，减去的项 $\mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab}$ 对应 $\rho^2\sigma_2^2 = \Sigma_{12}^2/\Sigma_{11}$。

这个公式中的维数容易误读，下面看一个实际规模的例子。某机器学习模型有六个超参数待调，已完成 40 次训练。在贝叶斯优化器中，$\mathbf{a}$ 存放 40 个观测到的验证误差，$\mathbf{b}$ 存放尚未尝试的配置（例如 1,000 个）处的未知误差，因此 $\mSigma_{aa}$ 为 $40 \times 40$，$\mSigma_{ab}$ 为 $40 \times 1000$。数字六在任何地方都没有出现：高斯分布定义在函数值上，每个配置对应一个坐标，输入空间的维数只通过核函数为各对配置指定的协方差发挥作用（@sec-kernels）。条件化的代价随评估次数增长，而不随超参数个数增长，尽管维数越高，确定函数所需的评估越多（@sec-high-dimensions）。@snoek2012practical 正是用这一计算为隐 Dirichlet 分配、结构化支持向量机和卷积网络调参，结果达到或超过了人类专家选定的设置；@sec-cs-classifier 在真实数据上完整演示了这样一个调参问题。

推导要用到 @sec-block-matrices 中的一个结论，这里用 @eq-gauss-partition 的记号重述如下。块 $\mSigma_{aa}$ 的 Schur 补为

$$
\mathbf{S} = \mSigma_{bb} - \mSigma_{ab}^\T \mSigma_{aa}^{-1} \mSigma_{ab},
$$ {#eq-gauss-schur}

而精度矩阵 $\bm{\Lambda} = \mSigma^{-1}$ 的各块为

$$
\begin{bmatrix} \bm{\Lambda}_{aa} & \bm{\Lambda}_{ab} \\ \bm{\Lambda}_{ab}^\T & \bm{\Lambda}_{bb} \end{bmatrix}
=
\begin{bmatrix}
\mSigma_{aa}^{-1} + \mSigma_{aa}^{-1}\mSigma_{ab}\mathbf{S}^{-1}\mSigma_{ab}^\T\mSigma_{aa}^{-1} & -\mSigma_{aa}^{-1}\mSigma_{ab}\mathbf{S}^{-1} \\
-\mathbf{S}^{-1}\mSigma_{ab}^\T\mSigma_{aa}^{-1} & \mathbf{S}^{-1}
\end{bmatrix}.
$$ {#eq-gauss-block-inverse}

将 $\mSigma$ 与该矩阵相乘，逐块验证乘积为单位矩阵，即可检验这一公式[@petersen2012matrix]。推导只需要最下面一行：$\bm{\Lambda}_{bb} = \mathbf{S}^{-1}$ 与 $\bm{\Lambda}_{ab}^\T = -\mathbf{S}^{-1}\mSigma_{ab}^\T\mSigma_{aa}^{-1}$。

::: {.derivation title="借助 Schur 补对高斯分布做条件化"}
1. 由乘法规则，$p(\mathbf{b} \given \mathbf{a}) = p(\mathbf{a}, \mathbf{b}) / p(\mathbf{a})$。固定 $\mathbf{a}$ 时，$p(\mathbf{a})$ 是常数，因此条件密度作为 $\mathbf{b}$ 的函数，正比于联合密度 @eq-gauss-density。
2. 记 $\mathbf{u} = \mathbf{a} - \vmu_a$，$\mathbf{v} = \mathbf{b} - \vmu_b$。联合密度的对数等于 $-\tfrac12 Q$ 加一个常数，其中 $Q = (\vx - \vmu)^\T\bm{\Lambda}(\vx - \vmu)$，各块如 @eq-gauss-block-inverse 所示。
3. 逐块展开 $Q$：$Q = \mathbf{u}^\T\bm{\Lambda}_{aa}\mathbf{u} + 2\,\mathbf{v}^\T\bm{\Lambda}_{ab}^\T\mathbf{u} + \mathbf{v}^\T\bm{\Lambda}_{bb}\mathbf{v}$。两个交叉项 $\mathbf{u}^\T\bm{\Lambda}_{ab}\mathbf{v}$ 与 $\mathbf{v}^\T\bm{\Lambda}_{ab}^\T\mathbf{u}$ 相等，因为二者都是数，且互为转置。
4. 只保留含 $\mathbf{v}$ 的项；给定 $\mathbf{a}$ 时，其余各项都是常数。于是 $\log p(\mathbf{b} \given \mathbf{a}) = -\tfrac12\mathbf{v}^\T\bm{\Lambda}_{bb}\mathbf{v} - \mathbf{v}^\T\bm{\Lambda}_{ab}^\T\mathbf{u} + \text{const}$。
5. 这正是 @eq-gauss-recipe-nd 关于变量 $\mathbf{v}$ 的形式，精度为 $\bm{\Lambda}_{bb}$，$\mathbf{h} = -\bm{\Lambda}_{ab}^\T\mathbf{u}$。（$\bm{\Lambda}_{bb}$ 正定，因为正定矩阵的每个对角块都正定。）所以给定 $\mathbf{a}$ 时，$\mathbf{v}$ 服从协方差为 $\bm{\Lambda}_{bb}^{-1}$、均值为 $-\bm{\Lambda}_{bb}^{-1}\bm{\Lambda}_{ab}^\T\mathbf{u}$ 的高斯分布；$\mathbf{b} = \vmu_b + \mathbf{v}$ 的协方差与之相同，均值平移 $\vmu_b$。
6. 代入 @eq-gauss-block-inverse 的最下面一行。协方差为 $\bm{\Lambda}_{bb}^{-1} = \mathbf{S}$，即 Schur 补。
7. 均值的偏移量为
   $-\mathbf{S}\,(-\mathbf{S}^{-1}\mSigma_{ab}^\T\mSigma_{aa}^{-1})\,\mathbf{u}
   = \mSigma_{ab}^\T\mSigma_{aa}^{-1}(\mathbf{a} - \vmu_a)$。
8. 综上，$\mathbf{b} \given \mathbf{a}$ 服从均值为 $\vmu_b + \mSigma_{ab}^\T\mSigma_{aa}^{-1}(\mathbf{a} - \vmu_a)$、协方差为 $\mSigma_{bb} - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab}$ 的高斯分布，即 @eq-gauss-conditional。
:::

这一推导沿用 @bishop2006pattern 第 2.3.1 节的做法，@sec-id-conditioning 将这一结果与相关恒等式汇集在一起。由推导可得两点。其一，条件协方差就是 Schur 补 @eq-gauss-schur 本身。其二，第 5 步表明条件精度就是联合精度的块 $\bm{\Lambda}_{bb}$：条件化读出精度矩阵的一个块，正如边际化读出协方差矩阵的一个块。

::: {.keyidea title="高斯分布的条件化"}
观测高斯向量的一部分之后，其余部分仍服从高斯分布。其余部分的均值发生偏移，偏移量是观测值与其期望之差的线性函数；协方差则会缩小，缩小的量取决于观测了哪些坐标，而与观测值无关。
:::

要点的后半句解释了为什么高斯过程的不确定性取决于评估的位置，而不取决于评估的结果（@sec-gp-conditioning）。另一种推导可以使 Schur 补显得不那么神秘。

::: {.aside title="另一种推导：减去观测所能预测的部分"}
定义残差 $\mathbf{r} = \mathbf{b} - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mathbf{a}$，即从 $\mathbf{b}$ 中减去由 $\mathbf{a}$ 得到的最佳线性预测后剩下的部分。

1. $(\mathbf{a}, \mathbf{r})$ 是 $(\mathbf{a}, \mathbf{b})$ 的线性映射，由 @eq-gauss-affine 知二者联合服从高斯分布。
2. 交叉协方差为
   $\Cov[\mathbf{r}, \mathbf{a}] = \Cov[\mathbf{b}, \mathbf{a}] - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\Cov[\mathbf{a}, \mathbf{a}]
   = \mSigma_{ab}^\T - \mSigma_{ab}^\T = \mathbf{0}$。
3. 对联合高斯的向量，不相关即意味着独立（@sec-gaussian-nd），因此观测 $\mathbf{a}$ 不提供关于 $\mathbf{r}$ 的任何信息。
4. 于是给定 $\mathbf{a}$ 时，$\mathbf{b} = \mathbf{r} + \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mathbf{a}$ 等于一个已知向量加上一个分布未变的高斯变量。该分布的均值为 $\vmu_b - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\vmu_a$；展开 $\Cov[\mathbf{r}]$ 可知其协方差为 $\mSigma_{bb} - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab} = \mathbf{S}$。
5. 把已知向量加到均值上，即得 @eq-gauss-conditional。

条件协方差就是 $\mathbf{b}$ 中无法由 $\mathbf{a}$ 预测的那部分的协方差。
:::

一个简单的数值例子即可展示高斯过程回归的缩影。

::: {.example #ex-gauss-two-inputs title="高斯过程的缩影"}
两个输入 $\vx$ 与 $\vx'$ 相距很近。按照先验，目标函数值 $f(\vx)$ 和 $f(\vx')$ 各自服从 $\N(0, 1)$，相关系数为 0.8，光滑核函数通常为相邻输入赋予这样的相关系数（@sec-kernel-trick）。评估得到 $f(\vx) = 1.2$。由 @eq-gauss-cond-2d，取 $\sigma_1 = \sigma_2 = 1$，$\rho = 0.8$，得

$$
f(\vx') \given f(\vx) = 1.2 \;\sim\; \N(0.8 \times 1.2,\; 1 - 0.8^2) = \N(0.96,\; 0.36).
$$

对未评估输入的信念向观测值移动了 80% 的距离，标准差从 1 降到 0.6。若观测值为 $-1.2$，均值将移到 $-0.96$，标准差仍为 0.6。@sec-gp-conditioning 由核函数给出相关系数，对所有未评估的输入同时做同样的计算。
:::

### 计算方法 {#sec-gauss-conditioning-code}

代码中计算 @eq-gauss-conditional 时，使用 $\mSigma_{aa}$ 的 Cholesky 因子和三角求解，从不显式求逆（@sec-cholesky）。设 $\mL\mL^\T = \mSigma_{aa}$，令 $\mathbf{V} = \mL^{-1}\mSigma_{ab}$，$\vw = \mL^{-1}(\mathbf{a} - \vmu_a)$。由于 $\mSigma_{aa}^{-1} = \mL^{-\T}\mL^{-1}$，均值的偏移量为 $\mathbf{V}^\T\vw$，减去的协方差为 $\mathbf{V}^\T\mathbf{V}$。

::: {.code title="NumPy"}
```python
import numpy as np

def condition(mu, Sigma, ia, ib, a):
    """Mean and covariance of x[ib] given x[ia] = a, for x ~ N(mu, Sigma)."""
    Saa = Sigma[np.ix_(ia, ia)]
    Sab = Sigma[np.ix_(ia, ib)]
    Sbb = Sigma[np.ix_(ib, ib)]
    L = np.linalg.cholesky(Saa)
    V = np.linalg.solve(L, Sab)          # L^{-1} Sigma_ab
    w = np.linalg.solve(L, a - mu[ia])   # L^{-1} (a - mu_a)
    return mu[ib] + V.T @ w, Sbb - V.T @ V
```
均值为零、`Sigma` 取核矩阵时，这个函数就是 @alg-gp-regression 中的高斯过程预测器，只是变量名不同。
:::

当观测坐标几乎决定了其余坐标时，减法 $\mSigma_{bb} - \mathbf{V}^\T\mathbf{V}$ 会抵消掉大部分有效数字，结果可能略不对称，或出现极小的负特征值。再次分解之前（例如为了抽取样本），先将其对称化并加上一个小的抖动项，即可修复这一问题。

## 和与积 {#sec-gaussian-sums}

另有两种组合高斯分布的运算，二者都以两个高斯分布为输入、得到一个高斯分布，因而容易混淆。两个独立高斯随机变量相加，描述由两个不确定部分构成的量，例如函数值加上测量噪声。两个高斯密度相乘，描述关于同一个量的两条独立证据，这正是贝叶斯定理对高斯先验和高斯似然所做的运算。前者使不确定性增大，后者使其减小。

### 独立高斯变量之和 {#sec-gauss-sum}

若 $X \sim \N(\mu_1, \sigma_1^2)$ 与 $Y \sim \N(\mu_2, \sigma_2^2)$ 相互独立，则 $(X, Y)$ 联合服从协方差为对角矩阵的高斯分布，而 $X + Y$ 是由单行 $(1, 1)$ 给出的线性映射。由 @eq-gauss-affine，

$$
X + Y \sim \N\!\left(\mu_1 + \mu_2,\; \sigma_1^2 + \sigma_2^2\right).
$$ {#eq-gauss-sum}

相加的是方差，而不是标准差。标准差分别为 3 和 4 的两个独立误差，相加后标准差为 5，而不是 7。若 $X$ 与 $Y$ 相关，同一映射给出的方差为 $\sigma_1^2 + \sigma_2^2 + 2\Cov[X, Y]$。书中有两种用法反复出现。观测 $y = f(\vx) + \varepsilon$ 带有独立噪声 $\varepsilon \sim \N(0, \sigma_n^2)$，其方差为 $\Var[f(\vx)] + \sigma_n^2$，即 @sec-gp-noise 中的预测方差。$n$ 次独立测量（每次方差为 $\sigma^2$）的平均值，方差为 $\sigma^2/n$，因此标准差按 $1/\sqrt{n}$ 下降。

### 高斯密度之积 {#sec-gauss-product}

现在取同一变量 $x$ 上的两个高斯密度，将其逐点相乘。乘积未经归一化，但形状是高斯的：两个二次指数之和仍是二次的。

::: {.derivation title="两个高斯密度之积"}
取密度 $\N(x;\, \mu_1, \sigma_1^2)$ 与 $\N(x;\, \mu_2, \sigma_2^2)$。

1. 指数函数相乘即指数相加：$-\tfrac12\left[(x - \mu_1)^2/\sigma_1^2 + (x - \mu_2)^2/\sigma_2^2\right]$。
2. 按 $x$ 的幂次整理：$-\tfrac12 x^2$ 的系数是 $1/\sigma_1^2 + 1/\sigma_2^2$，$x$ 的系数是 $\mu_1/\sigma_1^2 + \mu_2/\sigma_2^2$。
3. 由 @eq-gauss-recipe，乘积正比于精度为 $1/\sigma^2 = 1/\sigma_1^2 + 1/\sigma_2^2$、均值为 $\mu = \sigma^2\left(\mu_1/\sigma_1^2 + \mu_2/\sigma_2^2\right)$ 的高斯密度。
4. 不含 $x$ 的项是 $-\tfrac12\left[\mu_1^2/\sigma_1^2 + \mu_2^2/\sigma_2^2 - \mu^2/\sigma^2\right]$。通分后，方括号化简为 $(\mu_1 - \mu_2)^2/(\sigma_1^2 + \sigma_2^2)$。
5. 两个归一化因子的乘积为 $1/(2\pi\sigma_1\sigma_2)$。由于 $\sigma^2(\sigma_1^2 + \sigma_2^2) = \sigma_1^2\sigma_2^2$，它等于 $\big[1/\sqrt{2\pi\sigma^2}\big]\big[1/\sqrt{2\pi(\sigma_1^2 + \sigma_2^2)}\big]$。
6. 合并各因子：$\N(x;\, \mu_1, \sigma_1^2)\,\N(x;\, \mu_2, \sigma_2^2) = \N(\mu_1;\, \mu_2, \sigma_1^2 + \sigma_2^2)\;\N(x;\, \mu, \sigma^2)$。
:::

$d$ 维时，以 @eq-gauss-recipe-nd 代替 @eq-gauss-recipe，同样的步骤给出

$$
\N(\vx;\, \vmu_1, \mSigma_1)\,\N(\vx;\, \vmu_2, \mSigma_2)
= Z\;\N(\vx;\, \vmu, \mSigma),
\qquad
\mSigma = \left(\mSigma_1^{-1} + \mSigma_2^{-1}\right)^{-1},\;\;
\vmu = \mSigma\left(\mSigma_1^{-1}\vmu_1 + \mSigma_2^{-1}\vmu_2\right),
$$ {#eq-gauss-product}

其中常数 $Z = \N(\vmu_1;\, \vmu_2,\, \mSigma_1 + \mSigma_2)$[@rasmussen2006gaussian, 附录 A.2]。

这一结果可以从精度的角度解读。精度相加，因此乘积比任一因子都窄。均值是两个均值的加权平均，权重与精度成正比，因此更尖锐的密度对均值的牵引更强。常数 $Z$ 是原始乘积曲线下的面积：两个密度一致时它较大，两者的质量分布在不同位置时它很小。

这正是高斯先验与高斯测量下的贝叶斯定理。设对某个量 $\theta$ 的先验信念为 $\N(\mu_0, \sigma_0^2)$，观测为 $y = \theta + \varepsilon$，噪声 $\varepsilon \sim \N(0, \sigma_n^2)$。由于公式关于 $y$ 和 $\theta$ 对称，似然 $\N(y;\, \theta, \sigma_n^2)$ 作为 $\theta$ 的函数，就是密度 $\N(\theta;\, y, \sigma_n^2)$。后验正比于先验乘以似然，因此是高斯分布，精度为 $1/\sigma_0^2 + 1/\sigma_n^2$，均值介于先验均值与观测值之间。常数 $Z = \N(y;\, \mu_0, \sigma_0^2 + \sigma_n^2)$ 是观测值在先验下的密度，@sec-evidence 称之为模型证据。注意，它恰好是 @eq-gauss-sum 中和 $\theta + \varepsilon$ 的分布：本节的两种运算在此交汇。@sec-prior-likelihood-posterior 将展开这一观点，@sec-blr 将其从单个数推广到权重向量。

```{figure}
//| figure: gauss-combine
//| label: fig-gauss-combine
//| fig-cap: "组合两个高斯分布的两种方式。“变量之和”显示独立的 $X$ 与 $Y$ 之和 $X + Y$ 的密度（@eq-gauss-sum），比两个输入都宽。“密度之积”显示两条密度曲线逐点相乘并归一化的结果（@eq-gauss-product），比两者都窄，均值偏向更尖锐的输入。虚线是原始乘积，其面积为 $Z$。可用滑块设置均值和标准差，或在峰值附近拖动以移动该输入。"
```

**保持输入不变，切换两种运算**。和以 $\mu_1 + \mu_2$ 为中心，比任一输入都宽；积位于 $\mu_1$ 与 $\mu_2$ 之间，比任一输入都窄。

**在积中把一个输入调得很宽**。$\sigma_1$ 为 2.5 时，乘积几乎与第二个输入重合：模糊的先验几乎不改变精确测量提供的信息。

**把两个均值拉开**。归一化乘积的宽度不变，因为精度与均值无关；但原始乘积向零下沉，$Z$ 也随之变小。两个互相矛盾的自信密度，会在两者都几乎没有质量的区域给出一个自信的折中。$Z$ 很小，是先验与测量相互冲突的警示信号。

::: {.pitfall title="两个高斯变量之积不服从高斯分布"}
@eq-gauss-product 相乘的是密度函数。两个高斯随机变量相乘是另一种运算，结果也不同：若 $X$ 与 $Y$ 是独立的标准正态变量，其积 $XY$ 的密度在零附近无界增长，尾部也比任何高斯分布都重。高斯随机变量对加法和线性映射封闭；高斯密度对乘法封闭。阅读推导时须区分两者。
:::

## 在代码中使用高斯分布 {#sec-gauss-code}

养成三个习惯，可以避免大多数数值问题。

**使用对数密度**。高维时，密度 @eq-gauss-density 是许多小因子之积，会发生下溢。对 $d = 1000$ 维的标准正态分布，即使均值处的密度 $(2\pi)^{-500}$ 也约为 $10^{-399}$，低于双精度浮点数所能表示的最小正数。对数密度则是若干大小适中的数之和。可用 $\mSigma$ 的 Cholesky 因子 $\mL$ 计算：令 $\vw = \mL^{-1}(\vx - \vmu)$，二次型即为 $\vw^\T\vw$，而 $\log\lvert\mSigma\rvert = 2\sum_i \log L_{ii}$（@sec-determinants）。

::: {.code title="NumPy"}
```python
import numpy as np

def gaussian_logpdf(x, mu, Sigma):
    L = np.linalg.cholesky(Sigma)
    w = np.linalg.solve(L, x - mu)       # L^{-1} (x - mu)
    d = len(mu)
    return -0.5 * w @ w - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)
```
:::

**切勿构造 $\mSigma^{-1}$**。本章凡是含逆矩阵的公式，都应像上面的条件化代码那样，用 Cholesky 分解和三角求解计算，理由见 @sec-linalg-why-not-invert。

**核对参数化方式**。记号 $\N(\mu, \sigma^2)$ 的第二个参数是方差，但多数程序库接受的是标准差：NumPy 的 `random.normal(loc, scale)`、SciPy 的 `stats.norm(loc, scale)` 与 PyTorch 的 `Normal(loc, scale)` 都要求传入 $\sigma$。在应传标准差处传入方差，是一种不报错却很常见的错误。多元版本接受协方差矩阵，有时也接受其 Cholesky 因子，例如 PyTorch 的 `MultivariateNormal(loc, scale_tril=L)`。

## 习题 {#sec-gauss-exercises}

::: {.exercise #exr-gauss-compare}
值 $f(\vx_1)$ 与 $f(\vx_2)$ 联合服从高斯分布，均值分别为 0.3 和 0.1，标准差都是 0.2，相关系数为 0.75。计算 $f(\vx_1) > f(\vx_2)$ 的概率。再取相关系数为 0 重新计算，并解释两者的差别。

::: {.solution}
由 @ex-gauss-difference，$D = f(\vx_1) - f(\vx_2)$ 的均值为 $0.2$，方差为 $0.04 + 0.04 - 2 \times 0.75 \times 0.04 = 0.02$，因此标准差为 $0.141$，$\Prob(D > 0) = \Phi(0.2/0.141) = \Phi(1.41) \approx 0.92$。相关系数为 0 时，方差为 $0.08$，标准差为 $0.283$，概率为 $\Phi(0.71) \approx 0.76$。正相关意味着两个值的误差倾向于同向，在差中部分抵消，因此二者的大小顺序比任一值本身更确定。
:::
:::

::: {.exercise #exr-gauss-cond-2d}
由 @eq-gauss-conditional 推导 @eq-gauss-cond-2d。然后证明：一般情况下，观测 $\mathbf{a}$ 不会增大任何线性组合 $\vw^\T\mathbf{b}$ 的方差。

::: {.solution}
取 $\mathbf{a} = x_1$，$\mathbf{b} = x_2$，则 $\mSigma_{aa} = \sigma_1^2$，$\mSigma_{ab} = \rho\sigma_1\sigma_2$，$\mSigma_{bb} = \sigma_2^2$。均值为 $\mu_2 + (\rho\sigma_1\sigma_2/\sigma_1^2)(a - \mu_1) = \mu_2 + \rho(\sigma_2/\sigma_1)(a - \mu_1)$，方差为 $\sigma_2^2 - \rho^2\sigma_1^2\sigma_2^2/\sigma_1^2 = \sigma_2^2(1 - \rho^2)$。一般地，$\vw^\T\mathbf{b}$ 的方差从 $\vw^\T\mSigma_{bb}\vw$ 降为 $\vw^\T\mSigma_{bb}\vw - \vw^\T\mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab}\vw$。令 $\mathbf{q} = \mSigma_{ab}\vw$，减去的量为 $\mathbf{q}^\T\mSigma_{aa}^{-1}\mathbf{q} \ge 0$，因为正定矩阵的逆也是正定的。当且仅当 $\mathbf{q} = \mathbf{0}$，即 $\vw^\T\mathbf{b}$ 与每个观测到的坐标都不相关时，减少量为零。
:::
:::

::: {.exercise #exr-gauss-not-joint}
设 $X \sim \N(0, 1)$，$S$ 与 $X$ 独立，以各二分之一的概率取 $+1$ 或 $-1$。令 $Y = SX$。证明 $Y$ 服从标准正态分布，且 $X$ 与 $Y$ 不相关，但二者既不独立，也不服从联合高斯分布。

::: {.solution}
$\N(0, 1)$ 是对称的，$-X$ 与 $X$ 同分布，因此 $\Prob(Y \le y) = \tfrac12\Prob(X \le y) + \tfrac12\Prob(-X \le y) = \Phi(y)$。协方差为 $\E[XY] = \E[S]\,\E[X^2] = 0 \times 1 = 0$。二者不独立，因为 $\lvert Y\rvert = \lvert X\rvert$：知道 $X$ 之后，$Y$ 只剩两个可能的值。二者不服从联合高斯分布，因为线性组合 $X + Y = (1 + S)X$ 以二分之一的概率等于零，其余情况下服从 $\N(0, 4)$；而联合高斯变量的线性组合应当服从高斯分布（@sec-gaussian-linear）。只有对联合高斯的变量，不相关才意味着独立。
:::
:::

::: {.exercise #exr-gauss-repeat}
量 $\theta$ 的先验为 $\N(0, \sigma_0^2)$，对其测量 $n$ 次，$y_i = \theta + \varepsilon_i$，噪声 $\varepsilon_i \sim \N(0, \sigma_n^2)$ 相互独立。用 @eq-gauss-product 求 $\theta$ 的后验。$\sigma_0 \to \infty$ 时结果如何？$\sigma_0 = 1$ 时后验方差是多少？

::: {.solution}
每次测量贡献一个似然因子 $\N(\theta;\, y_i, \sigma_n^2)$。把先验依次乘以这 $n$ 个因子，精度相加：后验精度为 $1/\sigma_0^2 + n/\sigma_n^2$，后验均值为 $\left(\sum_i y_i/\sigma_n^2\right)\big/\left(1/\sigma_0^2 + n/\sigma_n^2\right)$。当 $\sigma_0 \to \infty$ 时，先验精度趋于零，均值趋于样本平均值 $\bar{y}$，方差趋于 $\sigma_n^2/n$，即熟知的平均值标准误。$\sigma_0 = 1$ 时，方差为 $1/(1 + n/\sigma_n^2) = \sigma_n^2/(n + \sigma_n^2)$。同一个数将在 @exr-noise-floor 中再次出现，那里对高斯过程在同一输入处评估 $n$ 次。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @bishop2006pattern 第 2.3 节用配方推导分块高斯分布的条件分布和边际分布，本章采用的也是这一路线；该节进而讨论 @sec-bayesian-inference 中的线性高斯模型。
- @rasmussen2006gaussian 附录 A 汇集了高斯过程所需的高斯与矩阵恒等式，包括高斯密度之积和用 Cholesky 因子采样。
- @murphy2022probabilistic 第 3 章讨论多元高斯分布和线性高斯系统，附有大量例题。
- @blitzstein2019introduction 是一本浅显易懂的概率教材，书中借助线性组合定义多元正态分布，@sec-gaussian-linear 采用的正是这一定义。
- @petersen2012matrix 以简明的参考手册形式列出了分块求逆与高斯恒等式。
- @galton1886regression 是为向均值回归命名的论文，条件化一图展示的正是这一效应。
