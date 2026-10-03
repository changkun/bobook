---
status: done
synopsis: "向量、作为空间映射的矩阵、正定矩阵、特征向量、Cholesky 分解、行列式与分块矩阵：高斯过程所需的线性代数。每个概念都配有一幅图，并借助一个三维协方差矩阵揭示二维图中看不到的现象。"
sources: ["textbooks", "Strang 2016", "Golub and Van Loan 2013", "Rasmussen and Williams 2006, app. A"]
---

# 不确定性的线性代数 {#sec-linear-algebra}

@sec-prob-covariance 介绍了一张表，记录每一对不确定量之间的协方差，即协方差矩阵。目标函数在已评估的每个输入、以及正在考虑的每个输入处都有一个值，贝叶斯优化器为这些值保存这样一张表，因此 50 次评估之后，这张表至少有 50 行。信念的每一次更新、每一次预测、对模型自身设置的每一次拟合，都是在这张表上的计算。本章讨论的就是这些计算。

线性代数是一门很大的学科，本章只选取本书后文用到的部分。本章把矩阵看作移动空间中各点的映射，因为这一图景能够解释几个关键性质：哪些表可以是协方差矩阵，协方差赋予一团不确定性什么样的形状，如何完全不求逆矩阵而解出高斯过程的方程，以及行列式衡量的是什么。大部分直觉可以由二维的图建立，但有几个事实要到三维才会出现，因此本章还将让读者亲手转动一个三维协方差。

这些内容的作用在下一章 @sec-gaussian 中显现：这里的每个结果都会成为关于高斯分布的结论；在 @sec-gp-regression 中，这些结论又转化为一个算法。

## 向量 {#sec-vectors}

**向量**（vector）是一组有序的数。向量用粗体小写字母表示，按惯例写成列的形式：

$$
\vx = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} \in \R^d.
$$

符号 $\R^d$ 表示由 $d$ 个实数组成的所有这类列表的集合，$d$ 称为向量的**维度**（dimension）。写成 $\vx^\T = (x_1, \dots, x_d)$（读作“$\vx$ 转置”），列就变成了行。

向量在本书中有两种用法，区分二者可以避免一个常见的混淆。第一种用法中，向量是搜索空间中的一个点：一组超参数配置、一套外骨骼的设置、一个照片滤镜的参数。例如，@snoek2012practical 同时调节图像分类器训练过程的九个设置，因此他们尝试的每个配置都是 $\R^9$ 中的一个向量（@sec-ard 会再讨论他们的发现）。第二种用法中，向量保存目标函数在一组输入处的值，$\vf = (f(\vx_1), \dots, f(\vx_n))^\T$。它的维度是输入的个数，而不是超参数的个数，并且每评估一次就增加一维。高斯过程的协方差矩阵就位于这第二种空间中。

向量按分量相加；向量乘以一个数，即每个分量都乘以这个数。从几何上看，$\vx + \vy$ 是把 $\vy$ 的箭头接在 $\vx$ 的箭头末端，$2\vx$ 则是方向相同、长度加倍的箭头。

### 内积、长度与角度 {#sec-linalg-inner-products}

两个维度相同的向量逐个分量相乘再求和，得到它们的**内积**（inner product），又称点积：

$$
\vx^\T\vy = \sum_{i=1}^d x_i y_i.
$$ {#eq-linalg-inner}

这一记号读作一行乘以一列，这正是 @sec-matrices 将介绍的矩阵乘法规则。长度和距离都由内积而来。向量的**范数**（norm）即其长度，$\lVert\vx\rVert = \sqrt{\vx^\T\vx}$，也就是 $d$ 维空间中的勾股定理；两点之间的距离是二者之差的范数，$\lVert\vx - \vy\rVert$。

内积还能衡量角度。对任意两个向量，

$$
\vx^\T\vy = \lVert\vx\rVert\,\lVert\vy\rVert \cos\theta,
$$

其中 $\theta$ 是两者的夹角。内积为零的两个向量是**正交的**（orthogonal），即互相垂直。除以两个长度就得到 $\cos\theta$：方向相同时为 1，互相垂直时为 0，方向相反时为 $-1$。@sec-prob-covariance 的相关系数就是这个余弦，只是计算对象换成了中心化的随机变量，而不是数的列表。

距离对贝叶斯优化很重要。核函数决定目标函数在两个输入处的值相关程度有多强（@sec-kernels），而最常用的核函数只通过两个输入之间的距离依赖于它们。按照径向基函数（RBF）核 $k(\vx, \vx') = \exp\!\left(-\lVert\vx - \vx'\rVert^2 / 2\ell^2\right)$，相近的输入取值相似，相距遥远的输入互不相关，“相近”的含义则由长度尺度 $\ell$ 决定。

### 高维空间中的距离 {#sec-linalg-distances}

我们对距离的直觉来自二维和三维，到了更高的维度，这种直觉会产生误导。在单位立方体 $[0, 1]^d$ 中均匀随机地撒点。每个坐标对两点间距离的平方平均贡献 $1/6$，因此典型距离按 $\sqrt{d/6}$ 增长：二维中约为 0.58，六维中为 1.0，五十维中为 2.9。距离的离散程度却不随之增长。在一次用 200 个随机点所做的模拟中，从二维到一百维，两两距离的标准差始终保持在 0.24 左右。结果，对一个典型的点，它到最近邻的距离与到最远邻的距离之比，在二维中是 0.04，在六维中升至 0.22，在五十维中升至 0.65，在一百维中升至 0.74。

因此，在高维空间中，每个点都远离其他所有点，而且所有点之间的距离都相差无几。对于五十维问题，长度尺度适合二维问题的核函数会认为每一对点都互不相关，模型从一次评估中得不到关于其他任何评估的信息。这是贝叶斯优化在高维中困难的根源之一，种种补救办法，例如随维度增长的长度尺度、为每个输入分别设定长度尺度（@sec-ard），也都从这里出发（@sec-high-dimensions）。

## 作为变换的矩阵 {#sec-matrices}

**矩阵**（matrix）是一张矩形的数表，用粗体大写字母表示。$m \times n$ 矩阵 $\mA$ 有 $m$ 行、$n$ 列，$A_{ij}$ 是第 $i$ 行第 $j$ 列的元素。这张表是矩阵的存储格式，而矩阵的含义在于它所定义的映射。

### 矩阵与向量的乘积 {#sec-linalg-matvec}

$m \times n$ 矩阵乘以 $n$ 维向量，得到 $m$ 维向量，其第 $i$ 个分量是第 $i$ 行与该向量的内积：

$$
(\mA\vx)_i = \sum_{j=1}^n A_{ij} x_j.
$$

同一个乘积还有第二种读法，也是应当在头脑中想象的那一种。把 $\mA$ 的各列记作 $\mathbf{a}_1, \dots, \mathbf{a}_n$，则

$$
\mA\vx = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \dots + x_n \mathbf{a}_n.
$$ {#eq-linalg-columns}

向量 $\mathbf{e}_1 = (1, 0, \dots, 0)^\T$ 选出第一列：$\mA\mathbf{e}_1 = \mathbf{a}_1$。可见矩阵的各列记录了各坐标方向被映射到何处；按照 @eq-linalg-columns，其他每个向量都按其坐标移动：$\vx$ 含有多少 $\mathbf{e}_1$，就取多少 $\mathbf{a}_1$，依此类推。

这类映射是**线性的**（linear）：它把和映射为和，把倍数映射为倍数，$\mA(\vx + \vy) = \mA\vx + \mA\vy$，$\mA(c\vx) = c\,\mA\vx$。在平面上，线性映射把直线映射为直线，保持原点不动，并把正方形网格变为平行四边形网格。单位圆，即所有长度为一的向量构成的集合，变为一个椭圆。

```{figure}
//| figure: linalg-transform
//| label: fig-linalg-transform
//| fig-cap: "把 2×2 矩阵看作平面的映射。品红色和绿色的箭头是 $\mA$ 的两列，即两个坐标方向的像；拖动箭头末端可以改变矩阵。虚线圆和虚线正方形是单位圆和单位正方形，蓝色椭圆和黄色平行四边形是它们的像，浅蓝色的线是网格的像。黑色箭头是单位圆上的一个向量 $\vx$ 及其像 $\mA\vx$；可以沿圆周拖动 $\vx$。读数给出行列式和特征值，其含义见 @sec-eigen 和 @sec-determinants。"
```

可以尝试以下操作：

- **拖动 $\mA\mathbf{e}_1$ 的末端。** 只有第一列改变，整个网格随之移动：每个点的移动量，是它的第一个坐标乘以这一改变量。
- **依次选择预设：旋转、剪切、拉伸。** 旋转使圆转动而形状不变；拉伸以不同的倍数缩放两个轴；剪切使一个轴沿另一个轴滑动，把正方形变为面积不变的斜平行四边形。
- **选择预设：奇异。** 两列指向同一方向，椭圆坍缩成一条线段，整个平面被压扁到一条直线上。这条直线上的每个点都来自无穷多个点，因此这个映射无法撤销。

### 复合、单位矩阵与逆矩阵 {#sec-linalg-composition}

先作用 $\mathbf{B}$、再作用 $\mA$，仍是一个线性映射，其矩阵是乘积 $\mA\mathbf{B}$，各列是 $\mA$ 作用于 $\mathbf{B}$ 各列的结果。逐个元素来看，$(\mA\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$，即 $\mA$ 的第 $i$ 行与 $\mathbf{B}$ 的第 $j$ 列的内积。顺序很重要：先拉伸再旋转，与先旋转再拉伸并不相同，因此一般而言 $\mA\mathbf{B} \ne \mathbf{B}\mA$。按这种方式相乘两个 $n \times n$ 矩阵需要 $n^3$ 次乘法，$i$、$j$、$k$ 的每种组合各一次。

**单位矩阵**（identity matrix）$\mI$ 对角线上为一、其余为零，它使每个向量保持原位。如果另一个矩阵 $\mA^{-1}$ 能撤销方阵 $\mA$ 的作用，即 $\mA^{-1}\mA = \mA\mA^{-1} = \mI$，则称这个方阵**可逆**（invertible）。矩阵可逆，当且仅当它不把空间压扁：没有非零向量被映射为零，因而没有两个点映射到同一个像。@fig-linalg-transform 中的奇异预设正是相反的情形。乘积求逆要颠倒顺序，$(\mA\mathbf{B})^{-1} = \mathbf{B}^{-1}\mA^{-1}$，因为最后施加的映射必须最先撤销。

### 转置 {#sec-linalg-transpose}

**转置**（transpose）$\mA^\T$ 把矩阵沿对角线翻转，使 $(\mA^\T)_{ij} = A_{ji}$，$m \times n$ 矩阵变为 $n \times m$ 矩阵。列向量是 $n \times 1$ 矩阵，其转置是 $1 \times n$ 的行，内积 $\vx^\T\vy$ 就是一行与一列的乘积。反过来的乘积 $\vx\vy^\T$，即一列乘以一行，是一个 $n \times n$ 矩阵，称为**外积**（outer product）。乘积的转置同样要颠倒顺序，$(\mA\mathbf{B})^\T = \mathbf{B}^\T\mA^\T$。

@sec-gp-regression 中各个量的形状都遵循这些规则。设有 $n$ 个已观测的输入和 $m$ 个待预测的输入，已观测输入之间的核矩阵 $\mK$ 是 $n \times n$ 的，已观测输入与新输入之间的矩阵 $\mK_*$ 是 $n \times
m$ 的；
后验均值 $\mK_*^\T\mK^{-1}\vy$ 把一个 $m \times n$ 矩阵、一个 $n \times n$ 矩阵与一个 $n$ 维向量相乘，为 $m$ 个新输入各给出一个预测。

## 对称矩阵与正定矩阵 {#sec-positive-definite}

并非每张方形数表都能作为协方差矩阵。本节找出其中的条件，结果表明它是几何性的：协方差矩阵不能让任何方向的方差为负。

### 二次型 {#sec-linalg-quadratic-forms}

矩阵等于其转置，即 $A_{ij} = A_{ji}$ 时，称为**对称的**（symmetric）。协方差矩阵是对称的，因为 $\Cov[x_i, x_j] = \Cov[x_j, x_i]$；核矩阵也是，因为 $k(\vx, \vx') = k(\vx', \vx)$。

对于对称矩阵 $\mA$，表达式 $\vw^\T\mA\vw$ 对每个向量 $\vw$ 给出一个数，称为**二次型**（quadratic form）。在二维中，取 $\mA = \begin{bmatrix} a & b \\ b & d \end{bmatrix}$，

$$
\vw^\T\mA\vw = a w_1^2 + 2b\, w_1 w_2 + d\, w_2^2,
$$

这是 $\vw$ 各分量的二次多项式。协方差矩阵与加权和相遇时，就会出现二次型，原因在于下面的恒等式。

::: {.derivation title="加权和的方差"}
设 $\vx$ 是均值为 $\vmu$、协方差矩阵为 $\mSigma$ 的随机向量，$\vw$ 是固定的权重向量。

1. 加权和是 $\vw^\T\vx = \sum_i w_i x_i$，由期望的线性性（@eq-prob-linearity），其均值为 $\vw^\T\vmu$。
2. 由方差的定义（@eq-prob-variance），$\Var[\vw^\T\vx] = \E\big[(\sum_i w_i (x_i - \mu_i))^2\big]$。
3. 把和的平方展开，得到所有两两乘积：$\E\big[\sum_i \sum_j w_i w_j (x_i - \mu_i)(x_j - \mu_j)\big]$。
4. 由期望的线性性，再由协方差的定义，这等于
   $\sum_i \sum_j w_i w_j\, \E[(x_i - \mu_i)(x_j - \mu_j)]
   = \sum_i \sum_j w_i w_j \Sigma_{ij}$。
5. 这个二重和就是二次型，所以 $\Var[\vw^\T\vx] = \vw^\T\mSigma\vw$。
:::

### 每个方向的方差都非负 {#sec-linalg-pd-definition}

方差不可能为负。因此由上面的推导，协方差矩阵必须对每个 $\vw$ 都满足 $\vw^\T\mSigma\vw \ge 0$。这一性质有专门的名称。

::: {.definition #def-linalg-pd title="正定矩阵与半正定矩阵"}
如果对每个向量 $\vw$ 都有 $\vw^\T\mA\vw \ge 0$，则称对称矩阵 $\mA$ 是**半正定的**（positive semidefinite）；如果对每个非零的 $\vw$ 都有 $\vw^\T\mA\vw > 0$，则称它是**正定的**（positive definite）。
:::

取 $\mA = \mSigma$，正定性的含义是：空间中每个方向的方差都为正，变量的任何加权组合都不是确切已知的。仅为半正定的协方差矩阵，存在某个方差为零的组合，即一个固定不变的组合。在高斯过程中，两个观测输入重合时就会出现这种情况：此时 $f(\vx)$ 与 $f(\vx')$ 是同一个随机变量，$f(\vx) - f(\vx')$ 恰好为零。下面的每个结果都假定严格正定，因此各种实现都会在对角线上加一点噪声或“抖动项”（@sec-gp-computation）。

二维中这一条件很容易检验。标准差为 $\sigma_1, \sigma_2$、相关系数为 $\rho$ 的协方差矩阵是正定的，当且仅当 $\sigma_1, \sigma_2 > 0$ 且 $-1 < \rho < 1$，而且这一范围内的每个相关系数都是可能的。从三维开始，这一条件就不够了。

### 无法共存的三个相关系数 {#sec-linalg-impossible}

假设三个量的方差都是 1，第一个与第二个的相关系数为 $0.8$，与第三个的相关系数也为 $0.8$，而第二个与第三个不相关：

$$
\mSigma = \begin{bmatrix} 1 & 0.8 & 0.8 \\ 0.8 & 1 & 0 \\ 0.8 & 0 & 1 \end{bmatrix}.
$$

每个相关系数都在 $-1$ 与 $1$ 之间，每一对单独来看都是有效的二维协方差。然而，取权重 $\vw = (-2, 1, 1)^\T$，得

$$
\vw^\T\mSigma\vw = 4 + 1 + 1 + 2\,(-2)(0.8) + 2\,(-2)(0.8) + 2\,(1)(0) = -0.4,
$$

即量 $x_2 + x_3 - 2x_1$ 的方差为负。任何三个随机变量都不可能具有这样的相关系数。直观地说，如果 $x_1$ 与 $x_2$ 紧密地共同变化，又与 $x_3$ 紧密地共同变化，那么 $x_2$ 与 $x_3$ 必然在一定程度上共同变化，不可能互不相关。二者必须相关到什么程度，由 @exr-linalg-range 求出。

这并非罕见的特例。在金融中，许多资产之间的相关系数表可能不是半正定的；如何把这样的表修正为最接近的有效相关矩阵，已有专门的文献 [@higham2002computing]。对高斯过程而言，这正是核函数不能是任意相似度分数的原因。一条看似自然的规则，例如“两个输入距离小于 1 时完全相关，否则不相关”，在输入 $0$、$0.6$ 与 $1.2$ 上就会失效：前两个完全相关，后两个完全相关，而第一个与最后一个不相关，这是同一类不可能的三元组。@sec-kernels 的核函数都经过构造，保证这种情况永远不会发生：一个函数是有效的核函数，当且仅当对任意选取的输入，其矩阵都是半正定的（@sec-kernel-trick）。

在下一节的 @fig-linalg-ellipsoid 中，读者可以尝试构造这样的矩阵，观察问题出在何处。

## 特征向量：矩阵拉伸的轴 {#sec-eigen}

仅从矩阵的元素，很难一眼看出它的作用。在 @fig-linalg-transform 中，大多数向量 $\vx$ 被 $\mA$ 既转动又拉伸。不过，有些方向只被拉伸：像 $\mA\vx$ 沿着 $\vx$ 本身的方向。找到这些方向，就能把这个映射描述为一组互相独立的拉伸。

::: {.definition #def-linalg-eigen title="特征向量与特征值"}
如果 $\mA\mathbf{u} = \lambda\mathbf{u}$，则称非零向量 $\mathbf{u}$ 是方阵 $\mA$ 的特征向量，对应的特征值为 $\lambda$。
:::

特征向量是矩阵不转动的方向；特征值是这一方向被拉伸的倍数，负的特征值使方向反转。在 @fig-linalg-transform 中，沿圆周拖动 $\vx$，直到 $\vx$ 与 $\mA\vx$ 对齐，读数就会给出特征值。默认矩阵在 $18°$ 附近（特征值为 1.4）和 $135°$ 附近（特征值为 0.6）出现对齐。打开**显示特征向量**，可以看到这两个方向。旋转预设根本没有实特征向量，因为它转动了每一个方向。

### 对称矩阵 {#sec-linalg-spectral}

一般矩阵的特征向量不一定互相垂直，默认矩阵就是一例。协方差矩阵只可能是对称矩阵，而对称矩阵的性质要好得多。

::: {.theorem #thm-linalg-spectral title="对称矩阵的谱定理"}
$d \times d$ 对称矩阵 $\mA$ 有 $d$ 个实特征值 $\lambda_1, \dots, \lambda_d$，以及一组标准正交的特征向量 $\mathbf{u}_1, \dots, \mathbf{u}_d$，即两两垂直的单位向量。把特征向量作为矩阵 $\mathbf{U}$ 的各列，把特征值放在矩阵 $\bm{\Lambda}$ 的对角线上，则

$$
\mA = \mathbf{U}\bm{\Lambda}\mathbf{U}^\T = \sum_{i=1}^d \lambda_i\, \mathbf{u}_i\mathbf{u}_i^\T.
$$
:::

证明特征值为实数所需的篇幅，超出了它在这里应占的分量 [@strang2016introduction]，但垂直性的证明很简短。

::: {.derivation title="对称矩阵的特征向量互相垂直"}
设 $\mA\mathbf{u} = \lambda\mathbf{u}$，$\mA\mathbf{v} = \mu\mathbf{v}$，且 $\lambda \ne \mu$。

1. 在第一个方程两边左乘 $\mathbf{v}^\T$：$\mathbf{v}^\T\mA\mathbf{u} = \lambda\, \mathbf{v}^\T\mathbf{u}$。
2. 对左边转置；它是单个数，转置后不变：$\mathbf{v}^\T\mA\mathbf{u} = \mathbf{u}^\T\mA^\T\mathbf{v} = \mathbf{u}^\T\mA\mathbf{v}$，这里用到了 $\mA^\T = \mA$。
3. 由第二个方程，$\mathbf{u}^\T\mA\mathbf{v} = \mu\, \mathbf{u}^\T\mathbf{v}$。
4. 所以 $\lambda\, \mathbf{v}^\T\mathbf{u} = \mu\, \mathbf{u}^\T\mathbf{v}$，又因为 $\mathbf{u}^\T\mathbf{v} = \mathbf{v}^\T\mathbf{u}$，得 $(\lambda - \mu)\, \mathbf{u}^\T\mathbf{v} = 0$。
5. 因为 $\lambda \ne \mu$，所以 $\mathbf{u}^\T\mathbf{v} = 0$。
:::

这个分解从右往左读，就是一组操作步骤。$\mathbf{U}^\T$ 转动空间，使特征向量与坐标轴对齐；$\bm{\Lambda}$ 把每个轴按对应的特征值拉伸；$\mathbf{U}$ 再把空间转回去。对称矩阵就是一组互相垂直的拉伸，仅此而已。在 @fig-linalg-transform 中选择对称预设，并打开特征向量：它们互相垂直，而且正是椭圆的轴。

### 特征值与正定性 {#sec-linalg-eigen-pd}

谱定理把正定性的定义转化为关于特征值的条件。

::: {.derivation title="正定即特征值全为正"}
设 $\mA = \mathbf{U}\bm{\Lambda}\mathbf{U}^\T$ 是对称矩阵，$\vw$ 是任意向量。

1. 代入分解式：$\vw^\T\mA\vw = \vw^\T\mathbf{U}\bm{\Lambda}\mathbf{U}^\T\vw$。
2. 记 $\mathbf{c} = \mathbf{U}^\T\vw$，即 $\vw$ 沿各特征向量的坐标，$c_i = \mathbf{u}_i^\T\vw$。于是 $\vw^\T\mA\vw = \mathbf{c}^\T\bm{\Lambda}\mathbf{c} = \sum_i \lambda_i c_i^2$。
3. 如果每个 $\lambda_i > 0$，那么只要某个 $c_i \ne 0$，这个和就为正；由于 $\mathbf{U}$ 可逆，每个非零的 $\vw$ 都满足这一点。
4. 反过来，取 $\vw = \mathbf{u}_j$，得到 $c_j = 1$，其余每个 $c_i = 0$，所以 $\vw^\T\mA\vw = \lambda_j$，它必须为正。
:::

对于协方差矩阵，这个推导还能说明更多。取长度为一的 $\vw$，即一个方向。它沿各特征向量的坐标满足 $\sum_i c_i^2 = 1$，所以方向 $\vw$ 上的方差是各特征值以 $c_i^2$ 为权重的加权平均。特征值就是沿各特征向量的方差，其他任何方向的方差都介于最小与最大的特征值之间。特征值最大的特征向量，就是不确定性最宽的方向。不确定性的形状是一个椭球，其轴沿各特征向量，半轴长为 $\sqrt{\lambda_i}$（即沿各轴的标准差）；@sec-gauss-shape 将针对高斯分布推导这一点。

### 三维中的协方差 {#sec-linalg-ellipsoid}

在二维中，椭圆概括了一切。在三维中，会出现两种平面无法展示的现象。@fig-linalg-ellipsoid 画出了一个三维协方差的椭球 $\{\vx : \vx^\T\mSigma^{-1}\vx = 1\}$，即距中心一个标准差的点的集合（这里的标准差在什么意义下成立，由 @sec-gauss-shape 精确说明），同时画出它的三根轴，以及它在立方体各面墙上的影子。旁边是三个成对视图：每次取三个坐标中的两个，画出 $\mSigma$ 中相应 $2 \times 2$ 子块的椭圆。

```{figure}
//| figure: linalg-ellipsoid
//| label: fig-linalg-ellipsoid
//| fig-cap: "把 3×3 协方差矩阵画成椭球。滑块设定三个相关系数和三个标准差；拖动立方体可以转动它。黑色线段是特征向量轴，半长为 $\sqrt{\lambda_i}$；后面几面墙上的灰色形状是椭球的影子；立方体旁边的三个面板是成对视图，与影子吻合。按下**不可能的三元组**之后，每个成对视图仍是有效的椭圆，但矩阵不是正定的：椭球不存在，有一个特征值为负（红色的方向），@sec-cholesky 的 Cholesky 分解也会失败。打开**显示样本**，会画出点 $\mL\vz$，其中 $\vz$ 是由均值为 0、方差为 1 的独立随机数组成的向量。这些数值仅为示意。"
```

可以尝试以下操作：

- **转动立方体。** 椭球有三根互相垂直的轴，即特征向量；从大多数角度看，它们都不与任何坐标轴对齐。读数列出各特征值，其平方根就是各轴的半长。
- **比较影子与成对视图。** 二者是相同的椭圆。协方差椭球在两个坐标所在平面上的影子，就是相应 $2 \times 2$ 子块的椭圆；因此，忽略高斯分布的一个变量，相当于删去它所在的行和列（@sec-gaussian-marginal）。
- **按下不可能的三元组。** 成对视图显示三个有效的椭圆，相关系数分别为 0.8、0.8 和 0。椭球消失，红色虚线标出方差会为负的方向，即特征值 $-0.13$ 对应的特征向量。现在慢慢调高 $\rho_{23}$：一旦 $\rho_{23}$ 超过 0.28（滑块上为 0.29），椭球就会重新出现，起初是一张扁平的圆盘，随 $\rho_{23}$ 增大而变厚。
- **按下链式。** 这里 $x_1$ 与 $x_2$ 相连，$x_2$ 与 $x_3$ 相连，相关系数都是 0.8，而 $\rho_{13} = 0.64$ 是两者之积。椭球变成沿对角线的一根雪茄，有一个特征值远大于其他特征值：大部分不确定性都集中在单一方向上。

第一个教训是，逐对检查并不能证明协方差矩阵有效；是否有效，是整张表的性质。第二个教训是，协方差可能在某个方向上几乎是扁平的，而这个方向无法通过任何单个变量察觉：一个接近奇异的矩阵，其成对视图未必有哪一个看起来是退化的。

### 核矩阵的特征值 {#sec-linalg-kernel-spectrum}

第二个教训正是高斯过程的常态。取 $[0, 1]$ 中等间距的 100 个输入，在长度尺度为 0.1 的径向基函数核下，核矩阵在精确算术中是一个 $100 \times 100$ 的正定矩阵。它最大的几个特征值是 23.9、21.2 和 17.5，但特征值衰减极快，100 个中只有 28 个超过 $10^{-10}$。在双精度算术中，最小的那些特征值淹没在舍入误差里；在我们用 NumPy 做的一次运行中，算出的最小特征值甚至略微为负，$-4 \times 10^{-15}$。相邻输入处的函数值相关性极强，以至于 100 维空间中的大多数方向几乎没有方差。在同一次运行中，NumPy 对这个矩阵的 Cholesky 分解失败了；在对角线上加 $10^{-6}$，即 @sec-gp-computation 的抖动项，分解即可成功。函数上的光滑先验实际只有远少于 100 个的独立不确定方向，这也是高斯过程能够从少数几次评估中学到光滑函数的部分原因。

最大特征值与最小特征值之比称为**条件数**（condition number），它衡量矩阵有多接近奇异。双精度数约有 16 位有效十进制数字，用条件数为 $10^k$ 的矩阵求解方程组，可能损失约 $k$ 位 [@golub2013matrix]。光滑核函数在间隔很近的输入上，核矩阵的条件数常常接近 $10^{16}$，此时有效数字已一位不剩，因此抖动项或观测噪声是数值上的必需，而不是建模上的选择。

恰好为零的特征值同样携带信息。在 @sec-comparison-graph 中，由一组成对比较构建的矩阵，其特征值可以统计出这样的方向有多少个：在这些方向上，这些比较对一个人的偏好没有提供任何信息。

## 不求逆而解方程组 {#sec-cholesky}

高斯过程回归的公式中满是逆矩阵：@eq-gp-pointwise 中的后验均值 $\vk(\vx)^\T\mK^{-1}\vy$ 与方差 $k(\vx, \vx) - \vk(\vx)^\T\mK^{-1}\vk(\vx)$。按字面理解，就是先求 $\mK$ 的逆，再相乘。数值计算的惯例是绝不这样做。本节解释原因，并说明应当怎样做。

### 为什么不求逆 {#sec-linalg-why-not-invert}

含有 $\mK^{-1}\vy$ 的公式，从来不需要 $\mK^{-1}$ 本身，需要的是线性方程组 $\mK\bm{\alpha} = \vy$ 的解向量 $\bm{\alpha}$。宁可求解也不求逆，理由有两个。其一，构造逆矩阵的代价更高，是下文所用分解的数倍。其二，算出的逆矩阵不够精确：乘以经过舍入的逆矩阵，比直接用分解求解损失更多的有效数字 [@golub2013matrix]。矩阵病态时（核矩阵常常如此），这一差别决定了答案是否可用。

### 三角方程组 {#sec-linalg-triangular}

有些方程组很容易求解。对角线以上的元素全为零的矩阵是**下三角的**（lower triangular）。以这样的矩阵构成的方程组 $\mL\vz = \mathbf{b}$，可以从上到下逐个求出分量：第一个方程只含 $z_1$，第二个只含 $z_1$ 和 $z_2$，依此类推。

$$
z_i = \frac{1}{L_{ii}}\Big(b_i - \sum_{k < i} L_{ik} z_k\Big), \qquad i = 1, \dots, n.
$$ {#eq-linalg-forward}

这种**前代**（forward substitution）约需 $n^2$ 次运算，而一般方法需要 $n^3$ 次。**上三角**（upper triangular）方程组的对角线以下为零，用同样的方式从下往上求解，即**回代**（back substitution）。求解对称正定方程组的整体策略，就是把它化为两个三角方程组。

### Cholesky 分解 {#sec-linalg-cholesky-def}

::: {.definition #def-linalg-cholesky title="Cholesky 分解"}
每个对称正定矩阵 $\mA$ 都可以唯一地写成

$$
\mA = \mL\mL^\T,
$$ {#eq-linalg-cholesky}

其中 $\mL$ 是对角元为正的下三角矩阵。$\mL$ 称为 $\mA$ 的 **Cholesky 因子**（Cholesky factor）。
:::

这个因子是矩阵的一种平方根。$1 \times 1$ 矩阵 $[a]$ 的因子是 $[\sqrt{a}]$，当且仅当 $a > 0$ 时存在。$2 \times 2$ 的情形展示了一般算法如何运作，以及它为什么需要正定性。

::: {.derivation title="2×2 矩阵的 Cholesky 因子"}
设 $\mA = \begin{bmatrix} a & b \\ b & d \end{bmatrix}$，求 $\mL = \begin{bmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{bmatrix}$。

1. 乘开得 $\mL\mL^\T = \begin{bmatrix} l_{11}^2 & l_{11}l_{21} \\ l_{11}l_{21} & l_{21}^2 + l_{22}^2 \end{bmatrix}$。
2. 比较左上角的元素：$l_{11}^2 = a$，所以 $l_{11} = \sqrt{a}$，这要求 $a > 0$。
3. 比较非对角元素：$l_{11}l_{21} = b$，所以 $l_{21} = b / \sqrt{a}$。
4. 比较右下角的元素：$l_{21}^2 + l_{22}^2 = d$，所以 $l_{22} = \sqrt{d - b^2/a}$，这要求 $d - b^2/a > 0$。
5. 这两个条件同时成立，当且仅当 $\mA$ 正定：$a$ 是方向 $\mathbf{e}_1$ 上的方差，而 $a\,(d - b^2/a) = ad - b^2$ 是行列式，即两个特征值之积（@sec-determinants）。
:::

第 4 步中的量 $d - b^2/a$，是 @sec-block-matrices 中 Schur 补的初次出现。对于 $a = \sigma_1^2$、$b = \rho\sigma_1\sigma_2$、$d = \sigma_2^2$ 的协方差矩阵，它等于 $\sigma_2^2(1 - \rho^2)$，即已知第一个变量之后，第二个变量剩余的方差，@sec-gaussian-conditioning 将说明这一点。

一般算法以同样的方式，每次填写 $\mL$ 的一列。

::: {.algorithm #alg-linalg-cholesky title="Cholesky 分解"}
输入：一个 $n \times n$ 对称矩阵 $\mA$。

1. 对每一列 $j = 1, \dots, n$：
2. 计算 $s = A_{jj} - \sum_{k < j} L_{jk}^2$。如果 $s \le 0$，停止：$\mA$ 不是正定的。
3. 令 $L_{jj} = \sqrt{s}$。
4. 对每一行 $i = j + 1, \dots, n$，令 $L_{ij} = \big(A_{ij} - \sum_{k < j} L_{ik}L_{jk}\big) / L_{jj}$。
:::

第 2 步使这一算法既是分解，也是检验：它成功当且仅当矩阵正定，而且在实践中是代价最低的正定性检验。在 @fig-linalg-ellipsoid 中，不可能的三元组在第三列失败，此处 $s$ 为负。这一算法约需 $n^3/3$ 次浮点运算 [@golub2013matrix]。

### 用因子求解 {#sec-linalg-cholesky-solve}

有了 $\mA = \mL\mL^\T$，方程组 $\mA\vx = \mathbf{b}$ 就拆成两个三角方程组。先用前代解 $\mL\vz = \mathbf{b}$，再用回代解 $\mL^\T\vx = \vz$。于是 $\mA\vx = \mL(\mL^\T\vx) = \mL\vz = \mathbf{b}$，正是所求。$\mL$ 一旦已知，每个新的右端项只需 $O(n^2)$。这就是 @alg-gp-regression：对核矩阵做一次分解，再通过三角求解得到权重 $\bm{\alpha}$ 和每个预测方差。

::: {.code title="NumPy 与 SciPy"}
```python
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

A = np.array([[4.0, 2.0], [2.0, 3.0]])
b = np.array([2.0, 4.0])

L = np.linalg.cholesky(A)                  # 下三角，L @ L.T == A
z = solve_triangular(L, b, lower=True)     # 前代
x = solve_triangular(L.T, z, lower=False)  # 回代
print(x)                                   # [-0.25  1.5 ]

c = cho_factor(A)                          # 同样的计算，打包成现成的函数
print(cho_solve(c, b))                     # [-0.25  1.5 ]
```
:::

### 实际的计算代价 {#sec-linalg-cost}

精确高斯过程能够处理多少个观测，取决于这一立方级的代价。@tbl-linalg-timing 给出了用 NumPy 以双精度测得的耗时，测量所用的是编写本章的笔记本电脑（一台 Apple M5 Pro）；在其他机器上，数字会有所不同，但增长速率不变。

::: {.table #tbl-linalg-timing title="用 NumPy 对 n × n 对称正定矩阵做分解与求逆所需的时间，以及存储该矩阵所需的内存（每个元素 8 字节）。在一台笔记本电脑上测量一次；重要的是比值，而非绝对数值。"}
| $n$    | Cholesky 分解 | 求逆   | 内存     |
|-------:|----------:|----------:|---------:|
| 500    | 0.4 ms    | 2.3 ms    | 2 MB     |
| 1,000  | 2.0 ms    | 10 ms     | 8 MB     |
| 2,000  | 14 ms     | 82 ms     | 32 MB    |
| 4,000  | 132 ms    | 625 ms    | 128 MB   |
:::

$n$ 翻倍，时间大约变为八倍，与 $n^3$ 的预测一致；求逆的代价约为分解的五倍。一次贝叶斯优化运行的观测很少超过几百个，因此单次分解的代价很低。真正使代价成倍增加的，是拟合核函数自身的设置（例如长度尺度），因为拟合的每一步都要重新分解矩阵（@sec-marginal-likelihood）。对于更大的数据集，GPyTorch 库用共轭梯度法代替分解，并在图形处理器上运行；共轭梯度法是一种迭代方法，只需要核矩阵与向量的乘积。据其作者报告，这把精确推断的渐近代价从 $O(n^3)$ 降到了 $O(n^2)$ [@gardner2018gpytorch]。

### 生成样本的平方根 {#sec-linalg-cholesky-samples}

Cholesky 因子还提供了一种由独立的随机性构造相关的随机性的方法。如果 $\vz$ 的各坐标相互独立、方差为 1，那么 $\mL\vz$ 的协方差是 $\mL\mL^\T = \mSigma$，这是 @sec-gaussian-linear 将推导的规则 $\Cov[\mA\vz] = \mA\Cov[\vz]\mA^\T$ 的推论。在 @fig-linalg-ellipsoid 中打开**显示样本**，可以看到 160 个这样的点：每个点都是 $\mL\vz$，其中 $\vz$ 是一次抽取的三个均值为 0、方差为 1 的独立随机数（标准正态随机数，即 @sec-gauss-standard 的钟形曲线），这些点合在一起填满了椭球的形状。由于 $\mL$ 是下三角的，第一个坐标只用到 $z_1$，第二个用到 $z_1$ 和 $z_2$，第三个三者都用到，因此每个新坐标都由前面各坐标的随机性加上自身的一份新的随机性构成。@sec-gp-posterior-samples 的后验样本就是这样抽取的。

## 作为体积的行列式 {#sec-determinants}

多元高斯分布的密度（@sec-gaussian-nd）和用于拟合核函数的边际似然（@sec-marginal-likelihood）中都含有行列式。行列式的含义是几何性的。

方阵的**行列式**（determinant）记作 $\det\mA$ 或 $\lvert\mA\rvert$，是映射 $\mA$ 对面积（二维）、体积（三维）以及一般 $d$ 维体积的放大倍数。它的符号记录了这一映射是否像镜子那样翻转了方向。对于 $2 \times 2$ 矩阵，

$$
\det \begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc,
$$

即两列张成的平行四边形的有向面积。这就是 @fig-linalg-transform 中的黄色平行四边形，即单位正方形的像；奇异预设把它压扁为零面积，反射预设则使行列式变为负。

从这幅图中可以得出三条性质。先后施加两个映射，它们的体积倍数相乘，所以 $\det(\mA\mathbf{B}) = \det\mA \cdot \det\mathbf{B}$，特别地，$\det(\mA^{-1}) = 1/\det\mA$。矩阵可逆，当且仅当其行列式不为零，即它不把空间压扁。对于对称矩阵 $\mA = \mathbf{U}\bm{\Lambda}\mathbf{U}^\T$，旋转 $\mathbf{U}$ 与 $\mathbf{U}^\T$ 保持体积不变，而 $\bm{\Lambda}$ 把第 $i$ 个轴拉伸 $\lambda_i$ 倍，所以

$$
\det\mA = \prod_{i=1}^d \lambda_i.
$$ {#eq-linalg-det-eigen}

因此，对于协方差矩阵，行列式是衡量不确定性总体大小的单个数，有时称为**广义方差**（generalized variance）。@fig-linalg-ellipsoid 的椭球半轴为 $\sqrt{\lambda_i}$，所以体积是 $\frac{4}{3}\pi\sqrt{\lambda_1\lambda_2\lambda_3} = \frac{4}{3}\pi
\sqrt{\det\mSigma}$。
相关性会使它缩小：对于方差为一、相关系数为 $\rho_{12}, \rho_{13}, \rho_{23}$ 的三个变量，

$$
\det\mSigma = 1 + 2\rho_{12}\rho_{13}\rho_{23} - \rho_{12}^2 - \rho_{13}^2 - \rho_{23}^2,
$$

它对于独立的变量等于 1，随椭球变扁而趋向 0，对于那个不可能的三元组则变为负数（$1 - 0.64 - 0.64 =
-0.28$）。
在 @sec-gp-information-gain 中，这类行列式衡量一组评估能揭示多少关于目标函数的信息，使行列式最大的那组评估，信息量也最大。

### 由 Cholesky 因子求对数行列式 {#sec-linalg-logdet}

三角矩阵的行列式是其对角元之积。（在 $2 \times 2$ 的公式中，$b = 0$ 时 $ad - bc$ 就是 $ad$。一般地，三角矩阵把第 $i$ 个坐标轴按其第 $i$ 个对角元拉伸，然后做剪切；剪切使点沿平行于其他轴的方向移动，不改变任何体积，正如 @fig-linalg-transform 的剪切预设所展示的。）对 $\mA = \mL\mL^\T$，由行列式的乘积法则得

$$
\log\det\mA = 2\sum_{i=1}^n \log L_{ii}.
$$ {#eq-linalg-logdet}

取对数并非多此一举。有许多极小特征值的核矩阵，其行列式远小于最小的正双精度数（约 $5 \times 10^{-324}$）。对于 @sec-linalg-kernel-spectrum 中 100 个输入的例子，加上使之可以分解的抖动项之后，算出的特征值之积会下溢为零，而 @eq-linalg-logdet 给出 $\log\det\mA \approx -1139$，这是一个大小适中的数，且一旦得到因子，计算它几乎没有额外代价。高斯对数密度和对数边际似然的每一种实现，都用这种方式计算行列式（@sec-gauss-code）。

## 分块矩阵与 Schur 补 {#sec-block-matrices}

高斯过程同时处理两组变量：已观测的函数值和待预测的函数值。因此它的矩阵自然地分成若干块，如 @eq-gp-joint 所示。本节建立这类分块的代数，最终得到 @sec-gaussian-conditioning 所需的那个公式。

### 矩阵的分块 {#sec-linalg-partitioned}

分块的对称矩阵形如

$$
\mathbf{M} = \begin{bmatrix} \mA & \mathbf{B} \\ \mathbf{B}^\T & \mathbf{D} \end{bmatrix},
$$ {#eq-linalg-block}

其中 $\mA$ 是 $p \times p$ 的，$\mathbf{D}$ 是 $q \times q$ 的，$\mathbf{B}$ 是 $p \times q$ 的。各块可以像数一样相乘，只是每个乘积都要保持顺序，因为矩阵乘法不满足交换律。对于协方差，$\mA$ 与 $\mathbf{D}$ 是各组内部的协方差，$\mathbf{B}$ 存放两组之间的协方差。

### 分块消元 {#sec-linalg-schur}

手工求解二元一次方程组时，用第一个方程从第二个方程中消去第一个未知数。用分块做同样的一步，就得到本节的核心对象。

::: {.definition #def-linalg-schur title="Schur 补"}
对于分块矩阵 @eq-linalg-block，若 $\mA$ 可逆，则 $\mA$ 的 Schur 补是

$$
\mathbf{S} = \mathbf{D} - \mathbf{B}^\T\mA^{-1}\mathbf{B}.
$$ {#eq-linalg-schur}
:::

对于元素为 $a, b, d$ 的 $2 \times 2$ 矩阵，$a$ 的 Schur 补就是 Cholesky 推导中的那个数 $d - b^2/a$。一般地，它是 $\mathbf{D}$ 去掉由第一组解释的部分之后剩余的部分。

::: {.derivation title="分块消元"}
1. 从第二个块行中减去第一个块行的 $\mathbf{B}^\T\mA^{-1}$ 倍。写成矩阵乘积，取 $\mathbf{E} = \begin{bmatrix} \mI & \mathbf{0} \\ -\mathbf{B}^\T\mA^{-1} & \mI \end{bmatrix}$，有 $\mathbf{E}\mathbf{M} = \begin{bmatrix} \mA & \mathbf{B} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}$，因为右下块变成了 $\mathbf{D} - \mathbf{B}^\T\mA^{-1}\mathbf{B}$。
2. 对列做同样的操作：右乘 $\mathbf{E}^\T$ 消去右上块，又因为 $\mA$ 是对称的，所以 $\mathbf{E}\mathbf{M}\mathbf{E}^\T = \begin{bmatrix} \mA & \mathbf{0} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}$。
3. $\mathbf{E}$ 是可逆的：把非对角块的符号取反，即得其逆。所以 $\mathbf{M} = \mathbf{E}^{-1}\begin{bmatrix} \mA & \mathbf{0} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}\mathbf{E}^{-\T}$。
:::

由此得出三个推论，本书后文都会用到。

**行列式。** $\mathbf{E}$ 是对角块为单位矩阵的分块三角矩阵，所以它的行列式为 1，并且 $\det\mathbf{M} = \det\mA \cdot \det\mathbf{S}$。

**正定性。** 对任意向量 $\vw$，令 $\mathbf{v} = \mathbf{E}^{-\T}\vw$；则 $\vw^\T\mathbf{M}\vw = \mathbf{v}_1^\T\mA\mathbf{v}_1 + \mathbf{v}_2^\T\mathbf{S}\mathbf{v}_2$，其中 $\mathbf{v}_1$ 与 $\mathbf{v}_2$ 是 $\mathbf{v}$ 的两部分。所以 $\mathbf{M}$ 正定，当且仅当 $\mA$ 与它的 Schur 补 $\mathbf{S}$ 都正定。这就是不可能的三元组的精确表述：考虑 $x_1$ 之后，$x_2$ 与 $x_3$ 的剩余部分仍必须构成有效的协方差（@exr-linalg-range）。

**逆矩阵。** 对第 3 步逐块求逆，利用 $(\mathbf{E}^{-1})^{-1} = \mathbf{E}$，得到 $\mathbf{M}^{-1} = \mathbf{E}^\T \begin{bmatrix} \mA^{-1} & \mathbf{0} \\ \mathbf{0} & \mathbf{S}^{-1} \end{bmatrix} \mathbf{E}$，乘开得

$$
\mathbf{M}^{-1} =
\begin{bmatrix}
\mA^{-1} + \mA^{-1}\mathbf{B}\mathbf{S}^{-1}\mathbf{B}^\T\mA^{-1} & -\mA^{-1}\mathbf{B}\mathbf{S}^{-1} \\
-\mathbf{S}^{-1}\mathbf{B}^\T\mA^{-1} & \mathbf{S}^{-1}
\end{bmatrix}.
$$ {#eq-linalg-block-inverse}

逆矩阵的右下块是 Schur 补的逆。@sec-gaussian-conditioning 将用到这一点：取 $\mA$ 为已观测值的协方差、$\mathbf{D}$ 为未观测值的协方差，证明条件化之后的协方差就是 Schur 补 $\mathbf{D} - \mathbf{B}^\T\mA^{-1}\mathbf{B}$，即先验协方差减去观测所解释的部分。@sec-gp-conditioning 是同一个公式，只是各块换成了核矩阵；@sec-id-block 与 @sec-id-woodbury 把它与相关的恒等式汇集在一起 [@petersen2012matrix]。

### 增加一个观测 {#sec-linalg-cholesky-update}

从分块的角度，还可以看出如何低成本地更新 Cholesky 因子，而贝叶斯优化器每次评估之后都需要这样做。假设对目前已观测的 $n$ 个输入有 $\mK = \mL\mL^\T$，现在来了一个新输入，它与旧输入的协方差为 $\vk$，自身方差为 $\kappa$。新的核矩阵及其因子具有如下分块形式

$$
\begin{bmatrix} \mK & \vk \\ \vk^\T & \kappa \end{bmatrix}
= \begin{bmatrix} \mL & \mathbf{0} \\ \mathbf{l}^\T & l_{\ast} \end{bmatrix}
\begin{bmatrix} \mL^\T & \mathbf{l} \\ \mathbf{0}^\T & l_{\ast} \end{bmatrix}.
$$

比较各块，得 $\mL\mathbf{l} = \vk$，即一次代价为 $O(n^2)$ 的前代；以及 $l_{\ast}^2 = \kappa - \mathbf{l}^\T\mathbf{l}$。由于 $\mathbf{l}^\T\mathbf{l} = \vk^\T\mL^{-\T}\mL^{-1}\vk = \vk^\T\mK^{-1}\vk$，新对角元的平方是 $\kappa - \vk^\T\mK^{-1}\vk$，即 $\mK$ 的 Schur 补，它也是高斯过程在新输入处的后验方差（@eq-gp-pointwise）。只要核函数的设置保持不变，把因子扩大一行只需 $O(n^2)$，而不必从头花费 $O(n^3)$。如果没有观测噪声，而新输入又与某个旧输入重合，$l_{\ast}$ 就为零，分解随之失败：矩阵已经变为奇异，原因见 @sec-linalg-pd-definition。

## 习题 {#sec-linalg-exercises}

::: {.exercise #exr-linalg-cholesky}
设 $\mA = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$。（a）手算它的 Cholesky 因子。（b）用它求 $\det\mA$，并用 $ad - bc$ 检验。（c）用前代与回代解 $\mA\vx = (2, 4)^\T$。

::: {.solution}
（a）由 $2 \times 2$ 的推导，$l_{11} = \sqrt{4} = 2$，$l_{21} = 2/2 = 1$，$l_{22} = \sqrt{3 - 1} = \sqrt{2}$，所以 $\mL = \begin{bmatrix} 2 & 0 \\ 1 & \sqrt2 \end{bmatrix}$。

（b）$\det\mA = (l_{11}l_{22})^2 = (2\sqrt2)^2 = 8$，而 $4 \cdot 3 - 2 \cdot 2 = 8$。

（c）前代：$z_1 = 2/2 = 1$，$z_2 = (4 - 1 \cdot 1)/\sqrt2 = 3/\sqrt2$。回代，其中 $\mL^\T = \begin{bmatrix} 2 & 1 \\ 0 & \sqrt2 \end{bmatrix}$：$x_2 = (3/\sqrt2)/\sqrt2 = 1.5$，$x_1 = (1 - 1.5)/2 = -0.25$。检验：$4(-0.25) + 2(1.5) = 2$，$2(-0.25) + 3(1.5) = 4$。这正是 @sec-linalg-cholesky-solve 中的代码示例打印出的数。
:::
:::

::: {.exercise #exr-linalg-range}
三个变量的方差都为一，且 $\rho_{12} = \rho_{13} = 0.8$。利用第一个变量的 Schur 补，求出使协方差矩阵正定的所有 $\rho_{23}$ 值，并解释这个 Schur 补的含义。

::: {.solution}
按 $\mA = [1]$、$\mathbf{B} = (0.8, 0.8)$、$\mathbf{D} = \begin{bmatrix} 1 & \rho_{23} \\ \rho_{23} & 1 \end{bmatrix}$ 分块。于是

$$
\mathbf{S} = \mathbf{D} - \mathbf{B}^\T\mathbf{B}
= \begin{bmatrix} 0.36 & \rho_{23} - 0.64 \\ \rho_{23} - 0.64 & 0.36 \end{bmatrix}.
$$

由于 $\mA = [1]$ 是正定的，整个矩阵正定当且仅当 $\mathbf{S}$ 正定；对于对角元为正的 $2 \times 2$ 矩阵，这意味着 $\det\mathbf{S} = 0.36^2 - (\rho_{23} - 0.64)^2 > 0$，所以 $\lvert\rho_{23} - 0.64\rvert < 0.36$，即 $0.28 < \rho_{23} < 1$。这正是 @fig-linalg-ellipsoid 中椭球重新出现之处。$\mathbf{S}$ 是去掉由 $x_1$ 解释的部分之后 $x_2$ 与 $x_3$ 的协方差（@sec-gaussian-conditioning）：两者各自保留方差 $1 - 0.8^2 = 0.36$，而二者剩余的协方差 $\rho_{23} - 0.64$ 必须有效，即绝对值至多为 0.36。
:::
:::

::: {.exercise #exr-linalg-two-inputs}
两个输入的核函数值为 $\rho$，所以它们的核矩阵是 $\mK = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}$。求它的特征值、特征向量以及条件数。当 $\rho = 0.9999$ 时（光滑核函数下两个非常接近的输入就是如此），用 $\mK$ 求解可能损失多少位十进制数字？

::: {.solution}
$\mK(1, 1)^\T = (1 + \rho)(1, 1)^\T$，$\mK(1, -1)^\T = (1 - \rho)(1, -1)^\T$，所以特征值是 $1 \pm \rho$，对应的特征向量是 $(1, 1)/\sqrt2$（两个值的平均）与 $(1, -1)/\sqrt2$（两个值之差）。条件数是 $(1 + \rho)/(1 - \rho)$，当 $\rho = 0.9999$ 时为 $1.9999/0.0001 \approx 20{,}000$，所以十六位数字中大约会损失四位。较小的特征值是两个函数值之差的先验方差：两个相近的输入，其函数值几乎没有相差的余地，而这正是矩阵接近奇异的方向。
:::
:::

::: {.exercise #exr-linalg-update}
在 @sec-linalg-cholesky-update 中，假设观测带有噪声，因此要分解的矩阵是 $\mK + \sigma_n^2\mI$。此时新的对角元 $l_{\ast}^2$ 是多少？证明即使新输入与某个旧输入重合，它也不可能再为零。

::: {.solution}
新矩阵的角上是 $\kappa + \sigma_n^2$，原来的块是 $\mK + \sigma_n^2\mI$，所以 $l_{\ast}^2 = \kappa + \sigma_n^2 - \vk^\T(\mK + \sigma_n^2\mI)^{-1}\vk$。量 $\kappa - \vk^\T(\mK + \sigma_n^2\mI)^{-1}\vk$ 是带噪声时 $f$ 在新输入处的后验方差，即 @eq-gp-noisy 中的 $\sigma^2(\vx)$，所以 $l_{\ast}^2 = \sigma^2(\vx) + \sigma_n^2$，也就是在该处做一次新的带噪声测量的方差。由于 $\sigma^2(\vx) \ge 0$，有 $l_{\ast}^2 \ge \sigma_n^2 > 0$：观测噪声使矩阵保持正定，这就是带噪声的模型很少需要抖动项的原因。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @strang2016introduction 耐心地介绍了作为映射的矩阵、特征值与正定矩阵，与本章一样侧重几何。
- @golub2013matrix 是讲解这些计算如何在浮点运算中完成的标准参考书，内容包括三角方程组、Cholesky 分解、运算量与条件数。
- @rasmussen2006gaussian 的附录 A 收录了高斯过程回归所用的矩阵恒等式和基于 Cholesky 分解的计算。
- @petersen2012matrix 是一本紧凑的恒等式手册，包括分块求逆与行列式的公式，可用于核对推导。
- @sanderson2016essence 是一套动画视频，展示矩阵在平面上和空间中的作用，也就是 @fig-linalg-transform 背后的图景。
- @higham2002computing 讨论如何修正无效的相关矩阵，即 @sec-linalg-impossible 在实际中的一面。
