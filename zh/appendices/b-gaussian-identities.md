---
status: done
synopsis: "书中推导所依赖的恒等式，逐条给出陈述、证明及其在书中的用途：分块求逆与 Schur 补、Woodbury 恒等式、高斯分布的条件化与线性高斯对、高斯密度的乘积，以及求两个高斯变量期望最大值的 Clark 公式。"
sources: ["Rasmussen and Williams 2006, app. A", "Petersen and Pedersen 2012", "Bishop 2006, sec. 2.3", "Clark 1961"]
---

# 矩阵与高斯恒等式 {#sec-gaussian-identities}

本书各章反复用到少数几条恒等式。每一条都在首次需要时引入，推导通常结合该章的例子展开。本附录以一般形式将它们汇集于此，每条附一个简短、便于核对的证明，并注明书中何处用到它。

五节内容层层递进，分块求逆是根本：Woodbury 恒等式是以两种方式解读的分块求逆，高斯分布的条件化是作用于协方差矩阵的分块求逆，两个高斯密度的乘积则是换了形式的条件化。最后一节讨论两个高斯值的期望最大值，与其他各节相互独立，为 @sec-acquisition 与 @sec-pbo 中的采集函数服务。

全文约定：向量均为列向量，$\mI$ 为尺寸相应的单位矩阵，凡求逆的矩阵均假定可逆。

## 分块求逆与 Schur 补 {#sec-id-block}

若矩阵的行与列各分为两组，则可以逐组求逆。写成分块形式，

$$
\mathbf{M} = \begin{bmatrix} \mA & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix},
$$ {#eq-id-blocks}

其中 $\mA$ 的尺寸为 $p \times p$，$\mathbf{D}$ 的尺寸为 $q \times q$。矩阵不必对称。$\mA$ 在 $\mathbf{M}$ 中的 **Schur 补**（Schur complement）为

$$
\mathbf{S} = \mathbf{D} - \mathbf{C}\mA^{-1}\mathbf{B},
$$ {#eq-id-schur}

即消去第一组之后 $\mathbf{D}$ 余下的部分，正如对两个方程消元后，会在 $2 \times 2$ 矩阵的一角留下 $d - cb/a$。

::: {.theorem #thm-id-block title="分块求逆"}
若 $\mA$ 与 $\mathbf{S}$ 可逆，则

$$
\begin{aligned}
\mathbf{M}^{-1} &=
\begin{bmatrix} \mathbf{P} & \mathbf{Q} \\ \mathbf{R} & \mathbf{S}^{-1} \end{bmatrix}, \\
\mathbf{P} &= \mA^{-1} + \mA^{-1}\mathbf{B}\,\mathbf{S}^{-1}\mathbf{C}\mA^{-1}, \\
\mathbf{Q} &= -\mA^{-1}\mathbf{B}\,\mathbf{S}^{-1}, \\
\mathbf{R} &= -\mathbf{S}^{-1}\mathbf{C}\mA^{-1},
\end{aligned}
$$ {#eq-id-block-inverse}

并且 $\det\mathbf{M} = \det\mA \cdot \det\mathbf{S}$。
:::

::: {.proof}
用两个三角矩阵消去非对角块，

$$
\mathbf{E} = \begin{bmatrix} \mI & \mathbf{0} \\ -\mathbf{C}\mA^{-1} & \mI \end{bmatrix},
\qquad
\mathbf{F} = \begin{bmatrix} \mI & -\mA^{-1}\mathbf{B} \\ \mathbf{0} & \mI \end{bmatrix}.
$$

1. 左乘 $\mathbf{E}$，即从第二块行中减去 $\mathbf{C}\mA^{-1}$ 左乘第一块行的结果。左下块变为 $\mathbf{C} - \mathbf{C}\mA^{-1}\mA = \mathbf{0}$，右下块变为 $\mathbf{D} - \mathbf{C}\mA^{-1}\mathbf{B} = \mathbf{S}$。
2. 再右乘 $\mathbf{F}$，即从第二块列中减去第一块列右乘 $\mA^{-1}\mathbf{B}$ 的结果；右上块由此消去，其余各块不变。于是
   $$
   \mathbf{E}\,\mathbf{M}\,\mathbf{F} = \begin{bmatrix} \mA & \mathbf{0} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}.
   $$
3. $\mathbf{E}$ 与 $\mathbf{F}$ 都可逆：将各自的非对角块变号，即得其逆矩阵。对第 2 步的等式两边求逆，得 $\mathbf{F}^{-1}\mathbf{M}^{-1}\mathbf{E}^{-1} = \operatorname{diag}(\mA^{-1}, \mathbf{S}^{-1})$，即以这两块为对角块的分块对角矩阵，从而 $\mathbf{M}^{-1} = \mathbf{F}\operatorname{diag}(\mA^{-1}, \mathbf{S}^{-1})\,\mathbf{E}$。
4. 将乘积展开。$\mathbf{F}\operatorname{diag}(\mA^{-1}, \mathbf{S}^{-1})$ 的上块行为 $(\mA^{-1},\, -\mA^{-1}\mathbf{B}\mathbf{S}^{-1})$，下块行为 $(\mathbf{0},\, \mathbf{S}^{-1})$；再右乘 $\mathbf{E}$，即把第二块列右乘 $-\mathbf{C}\mA^{-1}$ 后加到第一块列上，由此得到 @eq-id-block-inverse 的四个块。
5. $\mathbf{E}$ 与 $\mathbf{F}$ 是对角元全为 1 的三角矩阵，行列式为 1；乘积的行列式又等于行列式的乘积（@sec-determinants）。于是由第 2 步得 $\det\mathbf{M} = \det\mA\cdot\det\mathbf{S}$。
:::

第一组并无特殊之处。改为先消去第二组，利用 $\mathbf{D}$ 的 Schur 补 $\mathbf{T} = \mA - \mathbf{B}\mathbf{D}^{-1}\mathbf{C}$，同样的论证给出同一逆矩阵的第二种表达式：

$$
\begin{aligned}
\mathbf{M}^{-1} &=
\begin{bmatrix} \mathbf{T}^{-1} & \mathbf{Q}' \\ \mathbf{R}' & \mathbf{P}' \end{bmatrix}, \\
\mathbf{P}' &= \mathbf{D}^{-1} + \mathbf{D}^{-1}\mathbf{C}\,\mathbf{T}^{-1}\mathbf{B}\mathbf{D}^{-1}, \\
\mathbf{Q}' &= -\mathbf{T}^{-1}\mathbf{B}\mathbf{D}^{-1}, \\
\mathbf{R}' &= -\mathbf{D}^{-1}\mathbf{C}\,\mathbf{T}^{-1},
\end{aligned}
$$ {#eq-id-block-inverse-2}

且 $\det\mathbf{M} = \det\mathbf{D}\cdot\det\mathbf{T}$。

**书中用途**。@sec-linalg-schur 逐步推导了对称情形 $\mathbf{C} = \mathbf{B}^\T$，并说明对称矩阵 $\mathbf{M}$ 正定当且仅当 $\mA$ 与 $\mathbf{S}$ 都正定。条件化后的高斯分布以 Schur 补为协方差，原因正在于 @eq-id-block-inverse 的右下块为 $\mathbf{S}^{-1}$（@sec-id-conditioning）。取第二组为单个坐标，即得 @eq-kern-loo 中的留一公式；为已分解的矩阵添加一行一列，即得 @sec-linalg-cholesky-update 的 Cholesky 更新。

## Woodbury 恒等式 {#sec-id-woodbury}

$\mathbf{M}^{-1}$ 的两种表达式必须逐块相等。比较二者的左上块，便得到本附录中最有用的恒等式，它给出矩阵加上一个低秩矩阵之后其逆的变化。

::: {.theorem #thm-id-woodbury title="Woodbury 恒等式"}
设 $\mathbf{Z}$ 为 $n \times n$ 矩阵，$\mW$ 为 $m \times m$ 矩阵，$\mathbf{U}$ 与 $\mathbf{V}$ 为 $n \times m$ 矩阵。则

$$
\begin{aligned}
&\left(\mathbf{Z} + \mathbf{U}\mW\mathbf{V}^\T\right)^{-1}
= \mathbf{Z}^{-1} - \mathbf{Z}^{-1}\mathbf{U}\,\mathbf{N}^{-1}\mathbf{V}^\T\mathbf{Z}^{-1}, \\
&\text{where } \mathbf{N} = \mW^{-1} + \mathbf{V}^\T\mathbf{Z}^{-1}\mathbf{U}.
\end{aligned}
$$ {#eq-id-woodbury}
:::

::: {.proof}
取 $\mA = \mathbf{Z}$、$\mathbf{B} = -\mathbf{U}$、$\mathbf{C} = \mathbf{V}^\T$、$\mathbf{D} = \mW^{-1}$，对相应的分块矩阵应用 @sec-id-block 的结果。

1. $\mA$ 的 Schur 补为 $\mathbf{S} = \mW^{-1} + \mathbf{V}^\T\mathbf{Z}^{-1}\mathbf{U} = \mathbf{N}$，由 @eq-id-block-inverse，逆矩阵的左上块为 $\mathbf{P} = \mathbf{Z}^{-1} - \mathbf{Z}^{-1}\mathbf{U}\,\mathbf{N}^{-1}\mathbf{V}^\T\mathbf{Z}^{-1}$。
2. $\mathbf{D}$ 的 Schur 补为 $\mathbf{T} = \mathbf{Z} + \mathbf{U}\mW\mathbf{V}^\T$，由 @eq-id-block-inverse-2，逆矩阵的左上块为 $\mathbf{T}^{-1}$。
3. 矩阵的逆是唯一的，因此这两个块相等。
:::

该恒等式也称为矩阵求逆引理[@rasmussen2006gaussian, 附录 A.3]，其价值在于矩阵的尺寸。左边需要对 $n \times n$ 矩阵求逆；右边在已知 $\mathbf{Z}^{-1}$ 时，只需对 $m \times m$ 矩阵 $\mathbf{N}$ 求逆。当 $\mathbf{Z}$ 易于求逆（例如为对角矩阵）且 $m$ 远小于 $n$ 时，计算代价从 $O(n^3)$ 降到 $O(nm^2)$。

由同一分块矩阵还可以得到三个相关结果。

**秩一情形**。取 $m = 1$、$\mW = 1$，并以向量 $\mathbf{u}, \mathbf{v}$ 代替 $\mathbf{U}, \mathbf{V}$，中间的逆即为一个数，由此得到 Sherman-Morrison 公式：

$$
\left(\mathbf{Z} + \mathbf{u}\mathbf{v}^\T\right)^{-1}
= \mathbf{Z}^{-1} - \frac{\mathbf{Z}^{-1}\mathbf{u}\,\mathbf{v}^\T\mathbf{Z}^{-1}}{1 + \mathbf{v}^\T\mathbf{Z}^{-1}\mathbf{u}}.
$$ {#eq-id-sherman-morrison}

**行列式**。对同一分块矩阵，令 @sec-id-block 中的两个行列式公式相等，即得矩阵行列式引理[@rasmussen2006gaussian, 附录 A.3]：

$$
\det(\mathbf{Z} + \mathbf{U}\mW\mathbf{V}^\T)
= \det\mathbf{Z}\,\det\mW\,\det\mathbf{N}.
$$ {#eq-id-det-lemma}

**推移恒等式**。对任意 $n \times m$ 矩阵 $\mathbf{X}$ 与 $m \times n$ 矩阵 $\mathbf{Y}$，

$$
\left(\mI + \mathbf{X}\mathbf{Y}\right)^{-1}\mathbf{X} = \mathbf{X}\left(\mI + \mathbf{Y}\mathbf{X}\right)^{-1},
$$ {#eq-id-push-through}

这是因为 $\mathbf{X}(\mI + \mathbf{Y}\mathbf{X}) = (\mI + \mathbf{X}\mathbf{Y})\mathbf{X}$，两边分别左乘、右乘相应的逆，便把这两个因子移到了等式另一侧。

::: {.example #ex-id-blr title="权重空间与函数空间一致"}
@sec-bayes-blr-posterior 在权重空间中求出了含 $M$ 个特征的线性模型的后验：协方差为 $\mA_w^{-1}$，其中 $\mA_w = \mSigma_p^{-1} + \sigma_n^{-2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi}$ 是 $M \times M$ 矩阵；均值为 $\bar{\vw} = \sigma_n^{-2}\mA_w^{-1}\boldsymbol{\Phi}^\T\vy$。@sec-gp-noise 则在函数空间中用 $n \times n$ 矩阵 $\mK_y = \boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T + \sigma_n^2\mI$ 做预测。两者的预测必须相同，上述恒等式表明确实如此。

**协方差**：取 $\mathbf{Z} = \mSigma_p^{-1}$、$\mathbf{U} = \mathbf{V} = \boldsymbol{\Phi}^\T$、$\mW = \sigma_n^{-2}\mI$，应用 @eq-id-woodbury：

$$
\mA_w^{-1} = \mSigma_p - \mSigma_p\boldsymbol{\Phi}^\T\mK_y^{-1}\boldsymbol{\Phi}\mSigma_p.
$$

在等式两边用新输入的特征 $\boldsymbol{\phi}_*$ 从左右两侧相乘。左边是权重空间中的预测方差 $\boldsymbol{\phi}_*^\T\mA_w^{-1}\boldsymbol{\phi}_*$；右边是 $k(\vx_*, \vx_*) - \vk_*^\T\mK_y^{-1}\vk_*$，其中核函数为 @eq-fs-induced-kernel 的 $k(\vx, \vx') = \boldsymbol{\phi}(\vx)^\T\mSigma_p\boldsymbol{\phi}(\vx')$，$\vk_* = \boldsymbol{\Phi}\mSigma_p\boldsymbol{\phi}_*$。这正是 @eq-gp-noisy。

**均值**：由定义，$\mA_w\mSigma_p\boldsymbol{\Phi}^\T = \boldsymbol{\Phi}^\T + \sigma_n^{-2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T = \sigma_n^{-2}\boldsymbol{\Phi}^\T\mK_y$。左乘 $\mA_w^{-1}$、右乘 $\mK_y^{-1}$，得 $\sigma_n^{-2}\mA_w^{-1}\boldsymbol{\Phi}^\T = \mSigma_p\boldsymbol{\Phi}^\T\mK_y^{-1}$，即一个推移恒等式。于是 $\boldsymbol{\phi}_*^\T\bar{\vw} = \vk_*^\T\mK_y^{-1}\vy$，同样是 @eq-gp-noisy。

权重空间要对 $M \times M$ 矩阵求逆，函数空间则要对 $n \times n$ 矩阵求逆[@rasmussen2006gaussian, 第 2.1.2 节]。特征少而观测多时，权重空间的计算代价更低；特征有无穷多个时，只能采用函数空间（@sec-kernel-trick）。
:::

**书中用途**。除上例之外，@eq-id-sherman-morrison 还为 @exr-noise-floor 提供了第二种解法（@exr-id-sherman）。低秩近似使高斯过程能够处理大数据集（@sec-gp-computation），这类方法以 $m$ 个诱导点代替 $n$ 个观测，应用的正是 @eq-id-woodbury 与 @eq-id-det-lemma[@quinonero2005unifying]。

## 高斯分布的条件化 {#sec-id-conditioning}

将高斯向量分为两块，沿用 @eq-gauss-partition 的记号：

$$
\begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix}
\sim \N\!\left(
\begin{bmatrix} \vmu_a \\ \vmu_b \end{bmatrix},\;
\begin{bmatrix} \mSigma_{aa} & \mSigma_{ab} \\ \mSigma_{ab}^\T & \mSigma_{bb} \end{bmatrix}
\right).
$$

有两种运算可将其化为关于其中一块的陈述，结果都是高斯分布。

::: {.theorem #thm-id-conditioning title="高斯分布的边际分布与条件分布"}
$\mathbf{a}$ 的边际分布为 $\N(\vmu_a, \mSigma_{aa})$。给定 $\mathbf{a}$ 时 $\mathbf{b}$ 的条件分布是高斯分布，

$$
\begin{aligned}
\mathbf{b} \given \mathbf{a} &\sim \N\!\left(\vmu_{b \mid a},\, \mSigma_{b \mid a}\right), \\
\vmu_{b \mid a} &= \vmu_b + \mSigma_{ab}^\T\mSigma_{aa}^{-1}(\mathbf{a} - \vmu_a), \\
\mSigma_{b \mid a} &= \mSigma_{bb} - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab}.
\end{aligned}
$$ {#eq-id-conditional}

若用精度矩阵 $\bm{\Lambda} = \mSigma^{-1}$（按同样方式分块）表示，条件分布的精度为 $\bm{\Lambda}_{bb}$，均值为 $\vmu_b - \bm{\Lambda}_{bb}^{-1}\bm{\Lambda}_{ab}^\T(\mathbf{a} - \vmu_a)$。
:::

::: {.proof}
边际分布对应保留 $\mathbf{a}$、丢弃 $\mathbf{b}$ 的线性映射；高斯变量经线性映射后仍为高斯变量，均值与协方差随之映射（@eq-gauss-affine）。

再看条件分布。给定 $\mathbf{a}$ 时，$\mathbf{b}$ 的密度作为 $\mathbf{b}$ 的函数正比于联合密度，而联合密度的对数为 $-\tfrac12(\vx - \vmu)^\T\bm{\Lambda}(\vx - \vmu)$ 加一个常数。合并含 $\mathbf{b}$ 的项，余下一个二次型，其精度为 $\bm{\Lambda}_{bb}$，均值即上文以精度形式给出的均值。对 $\mSigma$ 应用 @thm-id-block，Schur 补为 $\mathbf{S} = \mSigma_{bb} - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab}$，精度矩阵的块为 $\bm{\Lambda}_{bb} = \mathbf{S}^{-1}$ 与 $\bm{\Lambda}_{ab}^\T = -\mathbf{S}^{-1}\mSigma_{ab}^\T\mSigma_{aa}^{-1}$。代入即得协方差 $\mathbf{S}$ 与均值偏移 $\mSigma_{ab}^\T\mSigma_{aa}^{-1}(\mathbf{a} - \vmu_a)$。@sec-gauss-conditioning-general 完成了每一步的计算，并给出了不借助精度矩阵的另一种证明。
:::

@eq-id-conditional 的三个特点在全书中反复出现：条件均值是观测块的线性函数；条件协方差完全不依赖于观测值；条件协方差等于先验协方差减去一个半正定矩阵，因此观测只会减少不确定性。

本书的多数模型并非以联合高斯分布的形式给出，而是由高斯先验与观测构成，观测等于未知量的线性函数加高斯噪声。下面的结果实现两种形式之间的转换。

::: {.theorem #thm-id-linear-gaussian title="线性高斯对"}
设 $\mathbf{a} \sim \N(\vmu, \mSigma)$，$\mathbf{b} \given \mathbf{a} \sim \N(\mathbf{H}\mathbf{a} + \mathbf{c},\, \mathbf{R})$。则 $\mathbf{a}$ 与 $\mathbf{b}$ 服从联合高斯分布，且

$$
\begin{aligned}
\E[\mathbf{b}] &= \mathbf{H}\vmu + \mathbf{c}, \\
\Cov[\mathbf{b}] &= \mathbf{H}\mSigma\mathbf{H}^\T + \mathbf{R}, \\
\Cov[\mathbf{a}, \mathbf{b}] &= \mSigma\mathbf{H}^\T,
\end{aligned}
$$ {#eq-id-lg-joint}

给定 $\mathbf{b}$ 时 $\mathbf{a}$ 的后验是高斯分布，且

$$
\begin{aligned}
\E[\mathbf{a} \given \mathbf{b}] &= \vmu + \mathbf{G}\,(\mathbf{b} - \mathbf{H}\vmu - \mathbf{c}), \\
\Cov[\mathbf{a} \given \mathbf{b}] &= \mSigma - \mathbf{G}\,\mathbf{H}\mSigma, \\
\mathbf{G} &= \mSigma\mathbf{H}^\T\left(\mathbf{H}\mSigma\mathbf{H}^\T + \mathbf{R}\right)^{-1}.
\end{aligned}
$$ {#eq-id-lg-posterior}
:::

::: {.proof}
记 $\mathbf{b} = \mathbf{H}\mathbf{a} + \mathbf{c} + \mathbf{e}$，其中 $\mathbf{e} \sim \N(\mathbf{0}, \mathbf{R})$ 与 $\mathbf{a}$ 独立。$(\mathbf{a}, \mathbf{e})$ 的两部分是相互独立的高斯变量，因而服从联合高斯分布；$(\mathbf{a}, \mathbf{b})$ 是它的线性映射，所以同样服从联合高斯分布。其矩由线性性得出：$\Cov[\mathbf{a}, \mathbf{b}] = \Cov[\mathbf{a}, \mathbf{H}\mathbf{a}] = \mSigma\mathbf{H}^\T$；$\Cov[\mathbf{b}] = \mathbf{H}\mSigma\mathbf{H}^\T + \mathbf{R}$，因为 $\mathbf{b}$ 的两个独立部分的协方差相加。后验即互换两块角色后的 @eq-id-conditional。
:::

**书中用途**。在观测输入上取 $\mathbf{H} = \mI$、$\mathbf{R} = \sigma_n^2\mI$，@eq-id-lg-joint 即为带噪声的高斯过程回归背后的联合分布（@sec-gp-noise），其中间一式即边际似然中的协方差 $\mK_y$（@sec-marginal-likelihood）。取 $\mathbf{H} = \boldsymbol{\Phi}$，则得到贝叶斯线性回归（@sec-blr），即 @ex-id-blr 中的函数空间形式。矩阵 $\mathbf{G}$ 称为增益，它把观测中出乎预期的部分转化为对先验均值的修正。

## 高斯密度的乘积 {#sec-id-products}

同一变量上的两个高斯密度相乘，结果仍呈高斯形状，只差一个常数因子。记 $\N(\vx;\, \vmu, \mSigma)$ 为均值为 $\vmu$、协方差为 $\mSigma$ 的高斯密度在 $\vx$ 处的值。

::: {.theorem #thm-id-product title="两个高斯密度的乘积"}
$$
\N(\vx;\, \vmu_1, \mSigma_1)\;\N(\vx;\, \vmu_2, \mSigma_2)
= Z\;\N(\vx;\, \vmu, \mSigma),
$$ {#eq-id-product}

其中 $\mSigma = \left(\mSigma_1^{-1} + \mSigma_2^{-1}\right)^{-1}$，$\vmu = \mSigma\left(\mSigma_1^{-1}\vmu_1 + \mSigma_2^{-1}\vmu_2\right)$，$Z = \N(\vmu_1;\, \vmu_2,\, \mSigma_1 + \mSigma_2)$。
:::

::: {.proof}
将这一乘积视为一个模型。令 $\vx \sim \N(\vmu_1, \mSigma_1)$，$\vy \given \vx \sim \N(\vx, \mSigma_2)$。

1. 联合密度为 $p(\vx)\,p(\vy \given \vx) = \N(\vx;\, \vmu_1, \mSigma_1)\,\N(\vy;\, \vx, \mSigma_2)$。高斯密度对自变量与均值的依赖只通过二者之差，所以 $\N(\vy;\, \vx, \mSigma_2) = \N(\vx;\, \vy, \mSigma_2)$。在 $\vy = \vmu_2$ 处，联合密度就是 @eq-id-product 的左边。
2. 同一联合密度也可以按另一顺序分解为 $p(\vy)\,p(\vx \given \vy)$。由 @thm-id-linear-gaussian，取 $\mathbf{H} = \mI$、$\mathbf{c} = \mathbf{0}$、$\mathbf{R} = \mSigma_2$，得 $p(\vy) = \N(\vy;\, \vmu_1, \mSigma_1 + \mSigma_2)$，而 $p(\vx \given \vy)$ 是高斯分布，协方差为 $\mSigma_1 - \mSigma_1(\mSigma_1 + \mSigma_2)^{-1}\mSigma_1$，均值为 $\vmu_1 + \mSigma_1(\mSigma_1 + \mSigma_2)^{-1}(\vy - \vmu_1)$。
3. 在 $\vy = \vmu_2$ 处，第一个因子就是常数 $Z$。
4. 由 @eq-id-woodbury，取 $\mathbf{Z} = \mSigma_1^{-1}$、$\mathbf{U} = \mathbf{V} = \mI$、$\mW = \mSigma_2^{-1}$，第 2 步中的协方差等于 $(\mSigma_1^{-1} + \mSigma_2^{-1})^{-1} = \mSigma$。
5. 由第 4 步，$\mSigma\mSigma_1^{-1} = \mI - \mSigma_1(\mSigma_1 + \mSigma_2)^{-1}$；又因 $\mSigma(\mSigma_1^{-1} + \mSigma_2^{-1}) = \mI$，有 $\mSigma\mSigma_2^{-1} = \mSigma_1(\mSigma_1 + \mSigma_2)^{-1}$。所以第 2 步中的均值在 $\vy = \vmu_2$ 处为 $\mSigma\mSigma_1^{-1}\vmu_1 + \mSigma\mSigma_2^{-1}\vmu_2 = \vmu$。
:::

由这一证明可以理解结果的三个部分。精度相加，是因为关于同一个量的两份独立证据合并在了一起。均值是两个均值以精度为权重的加权平均。常数 $Z$ 是第一个高斯分布经第二个高斯分布模糊之后，在第二个高斯分布中心处的概率密度：两个密度重叠时该值较大，不重叠时极小。@sec-gauss-product 用配方法推导了一维情形，并配有图示。

::: {.example #ex-id-bumps title="一个变量上的两个鼓包"}
@sec-fs-rbf-limit 需要计算两个未归一化鼓包之积 $e^{-(x - c)^2/2s^2}\,e^{-(x' - c)^2/2s^2}$ 对 $c$ 的积分。作为 $c$ 的函数，每个鼓包都是 $\sqrt{2\pi s^2}$ 乘以方差为 $s^2$ 的高斯密度，两者的中心分别为 $x$ 与 $x'$。由 @eq-id-product，乘积为

$$
2\pi s^2 \cdot \N(x;\, x',\, 2s^2) \cdot \N\!\left(c;\, \tfrac{x + x'}{2},\, \tfrac{s^2}{2}\right).
$$

最后一个因子对 $c$ 的积分为 1，所以积分值为 $2\pi s^2\,\N(x;\, x', 2s^2) = s\sqrt{\pi}\; e^{-(x - x')^2/4s^2}$，与该节用配方法求得的结果相同。径向基函数核正是两个鼓包之积中的常数 $Z$。
:::

**书中用途**。高斯先验与高斯似然下的贝叶斯定理即为 @eq-id-product，其中 $Z$ 为模型证据（@sec-evidence）。期望传播（@sec-ep）依据同一规则逐个乘、除高斯因子。

## 两个高斯变量的期望最大值 {#sec-id-clark}

有几个采集函数需要计算两个不确定量中较大者的期望值：期望改进比较一个不确定的值与一个已知的值（@sec-ei），最优选项期望效用比较效用的两个不确定的值（@sec-eubo）。当这两个量服从联合高斯分布时，该期望有闭式解。

::: {.theorem #thm-id-clark title="两个高斯变量的期望最大值"}
设 $A$ 与 $B$ 服从联合高斯分布，均值为 $\mu_A, \mu_B$，方差为 $v_A, v_B$，协方差为 $c$。记 $\delta = \mu_A - \mu_B$，$s^2 = v_A + v_B - 2c$ 为 $A - B$ 的方差。若 $s > 0$，则

$$
\begin{aligned}
\E[\max\{A, B\}] ={}& \mu_A\,\Phi(\alpha) + \mu_B\,\Phi(-\alpha) \\
&+ s\,\phi(\alpha), \qquad \alpha = \delta / s,
\end{aligned}
$$ {#eq-id-clark}

其中 $\phi$ 与 $\Phi$ 分别是标准正态分布的密度与分布函数。若 $s = 0$，期望为 $\max\{\mu_A, \mu_B\}$。
:::

::: {.proof}
1. 对任意两个数，$\max\{A, B\} = B + \max\{A - B, 0\}$。
2. $D = A - B$ 是高斯向量的线性映射，因而服从高斯分布（@eq-gauss-affine），均值为 $\delta$，方差为 $\Var[A] + \Var[B] - 2\Cov[A, B] = s^2$。
3. 若 $s > 0$，记 $D = \delta + sZ$，其中 $Z$ 服从标准正态分布。仅当 $Z > -\delta/s$ 时 $\max\{D, 0\}$ 不为零，因此 $\E[\max\{D, 0\}] = \int_{-\delta/s}^{\infty}(\delta + sz)\,\phi(z)\,\dd z$。第一部分为 $\delta\,(1 - \Phi(-\delta/s)) = \delta\,\Phi(\delta/s)$。第二部分：由 $\phi'(z) = -z\,\phi(z)$，$z\,\phi(z)$ 的原函数为 $-\phi(z)$，$z\,\phi(z)$ 从 $-\delta/s$ 到无穷的积分为 $\phi(-\delta/s) = \phi(\delta/s)$。于是 $\E[\max\{D, 0\}] = \delta\,\Phi(\delta/s) + s\,\phi(\delta/s)$。
4. 由期望的线性性与第 1 步，$\E[\max\{A, B\}] = \mu_B + \delta\,\Phi(\delta/s) + s\,\phi(\delta/s)$。由于 $\mu_B + \delta\,\Phi(\delta/s) = \mu_A\Phi(\delta/s) + \mu_B(1 - \Phi(\delta/s))$，且 $1 - \Phi(t) = \Phi(-t)$，这就是 @eq-id-clark。
5. 若 $s = 0$，$D$ 等于常数 $\delta$，$\max\{A, B\} = B + \max\{\delta, 0\}$ 的期望为 $\max\{\mu_A, \mu_B\}$。
:::

该公式出自 @clark1961greatest。这篇论文对相关系数任意的两个联合正态变量给出了精确结果，并通过反复应用该结果，给出多于两个变量时的近似。第 3 步正是期望改进背后的计算，@sec-ei 逐步完成了这一计算（@eq-acq-positive-part）。

该公式可以逐项解读。$\Phi(\delta/s)$ 是 $A$ 超过 $B$ 的概率，所以前两项是两个均值的加权平均，权重为相应变量较大的概率。第三项是由于不知道哪个较大而获得的额外收益，它对不确定性的依赖只通过 $s$。书中用到由此得出的四个推论。

- **从不低于较优均值**。$\max\{D, 0\} \ge D$ 且 $\max\{D, 0\} \ge 0$，所以 $\E[\max\{D, 0\}] \ge \max\{\delta, 0\}$，进而 $\E[\max\{A, B\}] \ge \max\{\mu_A, \mu_B\}$。
- **随 $s$ 递增**。将第 3 步的结果对 $s$ 求导，来自 $\Phi$ 的项与来自 $\phi'$ 的项相互抵消，恰好余下 $\phi(\delta/s) > 0$。关于两者之差的不确定性越大，价值总是越高。
- **相关性只通过 $s$ 起作用**。$A$ 与 $B$ 正相关会降低 $s$，从而降低期望最大值；负相关则使之升高。同涨同落的两个选项，几乎没有选择的余地。
- **均值相等**。$\delta = 0$ 时公式化为 $\mu + s/\sqrt{2\pi}$：额外收益约为 $0.4\,s$。

下图展示最大值的完整分布，@eq-id-clark 给出的正是该分布的均值。

```{figure}
//| figure: id-clark
//| label: fig-id-clark
//| fig-cap: "两个联合高斯值的最大值。细曲线：$A$ 与 $B$ 的密度。阴影：$\max\{A, B\}$ 的精确密度。竖直实线为其均值，即 @eq-id-clark；虚线为两个均值中较优者。读数以阴影密度的数值积分检验该公式，并给出超出较优均值的额外收益。"
```

**按下“均值相等”**。额外收益为 $s/\sqrt{2\pi}$：标准差分别为 1 与 0.5、不相关时，$s = 1.12$，额外收益为 0.446。

**把“相关系数 ρ”调向 0.95**。额外收益随 $s$ 一同缩小。两个标准差相等且相关系数接近 1 时，两个值同步变化，其差几乎为常数，最大值的期望几乎恰好等于较优均值。

**把相关系数调向 $-0.95$**。此时一个值高时另一个值低，最大值几乎总是远高于两个均值，额外收益达到最大。

**按下“一个确定的选项”**。$B$ 近乎常数时，阴影密度就是将 $A$ 的密度中低于 $B$ 的部分全部堆积到 $B$ 处所得的密度，其均值为 $\mu_B$ 加上 $A$ 相对于 $\mu_B$ 的期望改进。期望改进正是 @eq-id-clark 在两个值之一已知时的特例。

**书中用途**。@eq-id-clark 就是 @eq-eubo-closed，即一对选项的 EUBO 闭式解，其中 $A$ 与 $B$ 为两个选项的后验效用。只有两个输入参与时，它也是知识梯度（@sec-knowledge-gradient）。此时两个值是同一标准正态变量的线性函数，因而完全相关，定理依然适用，只需取 $s$ 为二者斜率之差的绝对值。

## 习题 {#sec-id-exercises}

::: {.exercise #exr-id-sherman}
令 $\mathbf{1}$ 为由 $n$ 个 1 组成的向量。用 @eq-id-sherman-morrison 计算 $(\sigma^2\mI + \mathbf{1}\mathbf{1}^\T)^{-1}\mathbf{1}$，并由此求出：在单位幅度的核下，于 $x_0$ 处做 $n$ 次带噪声观测之后，$x_0$ 处的后验方差（与 @exr-noise-floor 相同）。

::: {.solution}
取 $\mathbf{Z} = \sigma^2\mI$、$\mathbf{u} = \mathbf{v} = \mathbf{1}$，则 $\mathbf{Z}^{-1}\mathbf{1} = \mathbf{1}/\sigma^2$，$\mathbf{1}^\T\mathbf{Z}^{-1}\mathbf{1} = n/\sigma^2$。于是

$$
\begin{aligned}
(\sigma^2\mI + \mathbf{1}\mathbf{1}^\T)^{-1}\mathbf{1}
&= \frac{\mathbf{1}}{\sigma^2} - \frac{\mathbf{1}\,(n/\sigma^2)}{\sigma^2\,(1 + n/\sigma^2)} \\
&= \frac{\mathbf{1}}{\sigma^2}\cdot\frac{1}{1 + n/\sigma^2}
= \frac{\mathbf{1}}{\sigma^2 + n}.
\end{aligned}
$$

所有核函数值都是 1，所以 $\mK = \mathbf{1}\mathbf{1}^\T$，$\vk(x_0) = \mathbf{1}$，@eq-gp-noisy 的后验方差为 $1 - \mathbf{1}^\T\mathbf{1}/(\sigma^2 + n) = \sigma^2/(\sigma^2 + n)$。
:::
:::

::: {.exercise #exr-id-det}
用 @eq-id-det-lemma 计算 $n$ 个观测时 $\sigma^2\mI + \mathbf{1}\mathbf{1}^\T$ 的行列式，并由此求出 @eq-kern-lml 中的复杂度项 $-\tfrac12\log\lvert\mK_y\rvert$，其中核为单位幅度、长度尺度无穷长。再与长度尺度很短的情形比较，此时 $\mK_y = (1 + \sigma^2)\mI$。

::: {.solution}
取 $\mathbf{Z} = \sigma^2\mI$、$\mathbf{U} = \mathbf{V} = \mathbf{1}$、$\mW = 1$：$\det(\sigma^2\mI + \mathbf{1}\mathbf{1}^\T) = \sigma^{2n}\,(1 + n/\sigma^2) = \sigma^{2(n-1)}(\sigma^2 + n)$。复杂度项为 $-\tfrac12\left[(n - 1)\log\sigma^2 + \log(\sigma^2 + n)\right]$。噪声小时，该项是很大的正数：$n = 7$、$\sigma = 0.1$ 时为 $-\tfrac12\left[6 \times (-4.61) + 1.95\right] = 12.8$。长度尺度很短时，行列式为 $(1 + \sigma^2)^n$，该项为 $-\tfrac{n}{2}\log(1 + \sigma^2) = -0.03$。长的长度尺度所受的复杂度惩罚小得多，因为它预期所有观测几乎相等，而这只占数据集空间中很薄的一片。除非观测确实几乎相等，否则它会在数据拟合项上受到惩罚。
:::
:::

::: {.exercise #exr-id-max-iid}
设 $A$ 与 $B$ 是相互独立的标准正态变量。（a）用 @eq-id-clark 求 $\E[\max\{A, B\}]$。（b）不经积分，证明 $\E[\max\{A, B\}^2] = 1$，并求最大值的方差。（c）在 @fig-id-clark 中核对这两个数。

::: {.solution}
（a）$\delta = 0$，$s^2 = 2$，所以期望为 $s\,\phi(0) = \sqrt{2}/\sqrt{2\pi} = 1/\sqrt{\pi} \approx 0.564$。
（b）$\max\{A, B\}^2 + \min\{A, B\}^2 = A^2 + B^2$，其期望为 2。$(-A, -B)$ 与 $(A, B)$ 同分布，且 $\min\{A, B\} = -\max\{-A, -B\}$，所以 $\min\{A, B\}^2$ 与 $\max\{A, B\}^2$ 的期望相同，都等于 1。方差为 $1 - 1/\pi \approx 0.682$：两次抽样的最大值，其变异性比任何一次单独的抽样都小。
（c）把两个均值都设为 0，两个标准差都设为 1，相关系数设为 0。读数为 0.564，阴影密度明显比两条细曲线窄。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @petersen2012matrix 是免费的矩阵恒等式参考手册，收录了本附录中的每一条恒等式，以及 @sec-kern-gradient 用到的求导法则。
- @rasmussen2006gaussian 的附录 A 列出了高斯过程回归中用到的高斯与矩阵恒等式，记号与该书其余部分一致。
- @bishop2006pattern 第 2.3 节以教科书的篇幅推导了边际分布、条件分布与线性高斯对。
- @golub2013matrix 是矩阵计算的标准参考书，书中也解释了为何分解优于显式求逆。
- @clark1961greatest 是关于联合正态变量最大值的原始论文。
