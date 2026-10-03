---
status: done
synopsis: "从赌博机的角度讨论如何从对决中学习：偏好不满足传递性时“最优选项”的含义；经典算法及其理论保证；2021 至 2026 年的核化界及其假设与单位；以及至今无人证明的下界。"
sources: ["Yue et al. 2012", "Sui et al. 2018", "Zoghi et al. 2014", "Wu and Liu 2016", "Kirschner and Krause 2021", "Pásztor et al. 2024", "Kayal et al. 2025"]
---

# 对决赌博机与比较的理论 {#sec-dueling-bandits}

@sec-pbo 构建了从对决中学习的循环，并讨论了如何选择下一对选项，但没有追问：选择配对的规则最好能达到什么程度？这一问题属于赌博机理论，@sec-regret 已针对普通评估作过介绍：衡量算法的标准是遗憾（regret），即其选择与最优选择相比累计的差距；遗憾增长缓慢的算法就是好算法。本章针对比较提出同样的问题。

这一问题在比较的情形下多了一层普通赌博机所没有的曲折。以数值为反馈时，“最优选项”就是取值最大者；以对决为反馈时，最优选项是获胜者，而“胜过谁”并不总能自洽：偏好可以成环，如同石头、剪刀、布。因此，本章先讨论“最优”的几种含义，再介绍在有限个选项中找出最优者的算法，然后转向偏好贝叶斯优化所处的连续核化设定（2021 至 2026 年的界均出自这一设定），最后讨论尚未证明的问题。

## 对决赌博机 {#sec-dueling-bandits-intro}

**对决赌博机**（dueling bandit）问题最初是针对搜索引擎提出的。设想一套企业内网搜索系统内置 $K$ 个排序函数，需要为新客户找出其中最好的一个。请用户为结果列表打分并不可靠，但一种称为**交错实验**（interleaving）的技巧可以把两个排序器的结果合并为一个列表，再根据用户的点击推断其更偏好哪个排序器。这样，系统收到的每一次查询都是两个排序器之间的一次对决；系统每展示一次较差排序器的结果，就要付出相应的代价[@yue2012karmed; @yue2009interactively]。

形式化地，设有 $K$ 个选项，与 @sec-bandits 一样称为臂（arm）。在每一轮 $t$，算法选出两条臂 $a_t$ 与 $b_t$（可以相同），并观察哪一条获胜。臂 $i$ 以未知概率 $P_{ij}$ 胜过臂 $j$，且 $P_{ji} = 1 - P_{ij}$，$P_{ii} = 1/2$。这些概率构成的 $K \times K$ 矩阵 $\mathbf{P}$ 包含了问题的全部信息，但算法始终看不到它，只能看到自己所选对决的结果。为方便起见，以二分之一为基准度量每个概率：$\Delta_{ij} = P_{ij} - 1/2$，$i$ 倾向于胜过 $j$ 时为正。

当偏好来自某个效用时，例如 @eq-pbo-probit 或 @sec-bradley-terry 的 Bradley-Terry 模型，$P_{ij}$ 是效用差的链接函数，本节的一切都很简单：效用最高的臂胜过其他每一条臂。对决赌博机的表述则不假设效用存在，而是直接从矩阵出发。这种表述更一般，也正因如此，下一个问题的答案不止一个。

### “最优”可以指什么 {#sec-duel-winners}

最自然的定义要求某条臂胜过其他所有臂。

::: {.definition #def-duel-condorcet title="Condorcet 赢家"}
若对每个 $j \ne i$ 都有 $P_{ij} > 1/2$，则称臂 $i$ 为 **Condorcet 赢家**（Condorcet winner）。
:::

这一名称来自投票理论：在投票理论中，Condorcet 赢家是在一对一投票中胜过其他每一位候选人的候选人。Condorcet 赢家至多一个，也可能不存在：若 A 胜过 B，B 胜过 C，C 又胜过 A，则没有任何一条臂能胜过其他所有臂。另外三个较弱的定义总能给出答案[@sui2018advancements]。

::: {.definition #def-duel-copeland title="Copeland 赢家"}
臂 $i$ 的 **Copeland 得分**（Copeland score）是它胜过的其他臂的数目 $\#\{j \ne i : P_{ij} > 1/2\}$。得分最高的臂称为 **Copeland 赢家**（Copeland winner）。
:::

::: {.definition #def-duel-borda title="Borda 赢家"}
臂 $i$ 的 **Borda 得分**（Borda score）是它胜过另一条臂的平均概率 $\frac{1}{K - 1}\sum_{j \ne i} P_{ij}$，即与均匀随机抽取的对手对决时获胜的概率。得分最高的臂称为 **Borda 赢家**（Borda winner）。
:::

::: {.definition #def-duel-vonneumann title="von Neumann 赢家"}
**von Neumann 赢家**（von Neumann winner）是臂上的概率分布 $\boldsymbol{\pi}$，满足：从 $\boldsymbol{\pi}$ 中抽取的臂平均而言以至少二分之一的概率胜过任意一条固定的臂，即对每个 $j$ 都有 $\sum_i \pi_i P_{ij} \ge 1/2$。
:::

这几个定义回答的问题略有不同。Copeland 赢家只计胜场、不计差距，因此总是存在，并且在 Condorcet 赢家存在时与之重合，但仍可能输给某些臂。Borda 赢家考虑差距，因此即使 Condorcet 赢家存在，两者也可能不同。例如，一条臂以微弱优势胜过所有臂，另一条臂以微弱劣势输给它、却大胜其余各臂，前者的平均胜率可能反而低于后者[@sui2018advancements; @urvoy2013generic; @jamieson2015sparse]。von Neumann 赢家不是单独一条臂，而是臂的混合，即如下零和博弈的最优策略：双方各选一条臂，收益为 $P_{ij} - 1/2$。von Neumann 的极小极大定理保证这一混合存在；Condorcet 赢家存在时，von Neumann 赢家把全部权重放在 Condorcet 赢家上[@dudik2015contextual; @sui2018advancements]。

其中两个定义已在本书中以其他名称出现过。@gonzalez2017preferentialb 最大化的**软 Copeland**（soft-Copeland）得分（@sec-dueling-formulation）是对均匀随机对手获胜的平均概率，因此名称虽含 Copeland，实为 Borda 得分的连续版本。Borda 计数则出现在 @sec-query-aggregation：把隐藏偏好各不相同的人合并到同一个 Bradley-Terry 模型中，拟合所得的效用按 Borda 计数为选项排序[@siththaranjan2024distributional]。

下图可用于构建不满足传递性的循环赛，并观察各个定义如何分道扬镳。

```{figure}
//| figure: duel-winners
//| label: fig-duel-winners
//| fig-cap: "认定五条臂中最优者的四种方式。矩阵以 Bradley-Terry 模型为起点，效用为 1、0.7、0.45、0.2 与 0（尺度为 3）；循环强度 c 在 logit 尺度上为 A、B、C 三者加入石头剪刀布分量，使 B 对 A、C 对 B、A 对 C 的获胜概率上升。左：每一对中由获胜者指向落败者的箭头，胜负越悬殊箭头越粗；三元循环中的箭头为品红色。右：行臂胜过列臂的概率。下：各概念的得分，赢家高亮显示（von Neumann 赢家高亮的是混合中的臂）。点击单元格可颠倒该对的胜负。矩阵仅作示意。"
```

可尝试以下操作：

- **把“循环强度 c”设为 0**。此时矩阵来自一个效用，A 胜过所有臂，同时是 Condorcet 赢家、Copeland 赢家、Borda 赢家与 von Neumann 赢家（@exr-duel-utility）。
- **把 c 调到约 1 以上**。此时 B 胜过 A，C 胜过 B，而 A 仍胜过 C。Condorcet 赢家消失；A、B、C 的 Copeland 得分相同；A 因大胜 D 与 E 而保持最高的 Borda 得分；von Neumann 赢家变为这三条臂的混合。在默认值 1.5 处，B 的权重最大（其中的规律见 @exr-duel-rps）。
- **把 c 调回 0，并切换到“险胜的冠军”**。A 以 0.55 的概率胜过每一条臂，B 以 0.9 的概率胜过 C、D 与 E。A 是 Condorcet 赢家与 Copeland 赢家，Borda 赢家却是 B：平均胜率体现了 B 的大幅领先。
- **翻转底部的一对**，令 E 胜过 D。顶部没有任何变化：只有强臂之间的胜负关系出现矛盾时，这些定义才会产生分歧。

### 传递性与遗憾 {#sec-duel-transitivity}

偏好矩阵具有多少结构，由**随机传递性**（stochastic transitivity）条件刻画。这类条件把“若 A 胜过 B、B 胜过 C，则 A 胜过 C”推广到概率。对满足 $\Delta_{ij} \ge 0$ 与 $\Delta_{jk} \ge 0$ 的臂，**强随机传递性**要求 $\Delta_{ik} \ge \max\{\Delta_{ij}, \Delta_{jk}\}$，**中等随机传递性**要求 $\Delta_{ik} \ge \min\{\Delta_{ij}, \Delta_{jk}\}$，**弱随机传递性**只要求 $\Delta_{ik} \ge 0$。另有一个独立的条件，即**随机三角不等式**（stochastic triangle inequality）：对按 $i \succ j \succ k$ 排列的臂，要求 $\Delta_{ik} \le \Delta_{ij} + \Delta_{jk}$；强传递性并不蕴含这一条件[@bengs2021preference]。凡是“效用加单调链接”形式的模型，包括 Bradley-Terry 模型与 Thurstone 模型，都满足强随机传递性，因为获胜概率随效用差增大；这些链接在差为正时又是凹的，所以也满足三角不等式（推断，依据定义；@exr-duel-utility）。

真实偏好是否违反传递性，尚有争议。@chau2022inconsistent 在斜对称的偏好函数（可以表示循环）上放置高斯过程，发现它在变色龙争斗、NFL 比赛与一个引文图上比 @chu2005preference 的效用模型更准确，并据此认为违反传递性的情形很常见。但这些数据集本来就预期存在不可传递性，其中也没有一个是单个人的设计偏好（@sec-obs-extensions）。2026 年的一项结果考察许多标注者评判语言模型回答的情形，证明偏好能用单个奖励函数表示，当且仅当其中不含 Condorcet 循环；还证明在群体的 Luce 模型下，这种循环存在的概率以指数速度趋于 1[@liu2026statistical]。单个人评判设计时，本书仍以效用为工作假设。

有了最优臂的定义，遗憾的定义也随之确定。若存在 Condorcet 赢家（记为臂 1），则 $a_t$ 与 $b_t$ 之间一次对决的遗憾定义为赢家胜过这两条臂时多出的胜率，

$$
r_t = \Delta_{1 a_t} + \Delta_{1 b_t}, \qquad R_T = \sum_{t=1}^{T} r_t,
$$ {#eq-duel-regret}

这是 @yue2012karmed 的表述，可以理解为：与展示出的两个排序器相比，更偏好最优排序器的用户所占的比例。赢家与自身对决不产生代价，因此学习者找到赢家后便可不再付出代价。不存在 Condorcet 赢家时，遗憾以某个 Copeland 赢家为基准度量：记归一化的 Copeland 得分为 $\zeta_i$（$i$ 胜过的其他臂所占的比例），一次对决的代价为 $\max_i \zeta_i - (\zeta_{a_t} + \zeta_{b_t})/2$[@zoghi2015copeland; @wu2016double]。

对决赌博机在一个具体方面比普通赌博机更难。算法要为所拉动的臂付出代价，却只能观察到这两条臂之间的相对优劣，无法观察到其中任何一条与未知的最优臂相比如何。要得知两者都很差，需要安排其他臂对之间的对决[@sui2018advancements]。

## 算法 {#sec-dueling-algorithms}

对决赌博机算法分为两种风格[@sui2018advancements]。多数是**非对称的**（asymmetric）：先选一条参照臂，即当前的冠军或可能的赢家，再选一个挑战者与之较量。也有一些是**对称的**（symmetric）：同一学习者的两个副本各选一条臂，如同博弈中的两个参与者。

### Interleaved Filter {#sec-duel-if}

第一个算法 Interleaved Filter[@yue2012karmed]类似一场有卫冕冠军的擂台赛。它随机选定一个候选，让候选依次与其余每一条臂对决。凡以高置信度输给候选的臂都被淘汰；一旦某条臂以高置信度胜过候选，该臂即成为新的候选。只剩一条臂时，算法在余下的运行中让它与自身对决。该算法假设各臂之间存在全序，且强随机传递性与三角不等式均成立。在这些假设下，其 IF2 版本的期望遗憾为 $(K/\varepsilon_{1,2}) \log T$ 阶，其中 $\varepsilon_{1,2} = \Delta_{12}$ 是最好的两条臂之间的差距；@yue2012karmed 还证明，任何算法在某些问题上都要承受 $(K/\varepsilon)\log T$ 阶的遗憾，其中 $\varepsilon$ 是最优臂的最小差距。IF2 在常数因子以内是最优的。

### 相对上置信界 {#sec-duel-rucb}

相对上置信界（RUCB）[@zoghi2014relative]将 UCB1 的乐观原则（@sec-regret-ucb1）推广到对决，并且只需要一个假设：Condorcet 赢家存在。它记录 $i$ 胜过 $j$ 的次数 $W_{ij}$，并为每个获胜概率构造乐观估计。

::: {.algorithm #alg-duel-rucb title="RUCB（相对上置信界）"}
输入：$K$ 条臂，探索参数 $\alpha > 1/2$。

1. 对迄今已对决 $n_{ij} = W_{ij} + W_{ji} > 0$ 次的每一对，令 $U_{ij} = W_{ij}/n_{ij} + \sqrt{\alpha \ln t / n_{ij}}$；从未对决过的一对，令 $U_{ij} = 1$；另令 $U_{ii} = 1/2$。
2. **冠军**：在乐观估计下能胜过其他每一条臂的臂（对所有 $j$ 有 $U_{cj} \ge 1/2$）中选一条，记为 $c$。若不存在这样的臂，则任选一条。
3. **挑战者**：选 $d = \argmax_j U_{jc}$，即乐观估计下最有可能胜过冠军的臂。若没有任何臂有望胜过冠军，挑战者可能就是 $c$ 本身。
4. 让 $c$ 与 $d$ 对决，更新计数，然后重复。
:::

冠军是仍有可能成为 Condorcet 赢家的臂；挑战者是最有可能证明冠军并非 Condorcet 赢家的臂。RUCB 具有有限时间遗憾界，为 $K \log T$ 阶加上一个按 $K^2$ 增长的常数[@zoghi2014relative; @sui2018advancements]。

可能达到的最优速率已有精确结论。@komiyama2015regret 对所有算法证明了一个渐近下界：该下界是对次优臂的求和，每条次优臂贡献 $\log T$，权重为能暴露该臂的最廉价对决的遗憾，再除以 Bernoulli 分布之间的 Kullback-Leibler 散度（@sec-kl）。他们还给出了渐近达到这一下界的算法 RMED[@sui2018advancements]。@saha2022versatile 首先达到了相对于 Condorcet 赢家的最优有限时间阶 $\sum_i \log T / \Delta_i$，并称这解决了一个长期存在的问题。

### 两次 Thompson 采样 {#sec-duel-dts}

Thompson 采样（@sec-regret-thompson）同样可以自然地推广到对决。双重 Thompson 采样（D-TS）[@wu2016double]为每个获胜概率 $P_{ij}$ 维护一个 Beta 后验，以 Beta(1, 1) 为初始，每轮采样两次。第一次对整个矩阵采样，在乐观 Copeland 得分最高的臂中，选出采样 Copeland 得分最高的臂。第二次采样选出挑战者：在尚未确知会输给第一条臂的臂中，最有可能胜过它的那一条。D-TS 以 Copeland 赢家为目标，因此无论 Condorcet 赢家是否存在都适用。对一般的 Copeland 问题，其遗憾为 $K^2 \log T$ 阶；Condorcet 赢家存在时，它的一个简化版本可达到 $K \log T + K^2 \log\log T$[@wu2016double]。Copeland 赢家也有专门的乐观算法，在温和假设下遗憾界为 $K \log T$ 阶[@zoghi2015copeland]，此外还有一个渐近最优的算法[@komiyama2016copeland]。

下图在 @fig-duel-winners 的矩阵上运行 RUCB 与双重 Thompson 采样的简化版本，以随机配对作对照，绘出累积 Copeland 遗憾。

```{figure}
//| figure: duel-race
//| label: fig-duel-race
//| fig-cap: "在 @fig-duel-winners 的五臂矩阵上比较 RUCB（@alg-duel-rucb 的简化形式，从乐观候选中均匀抽取冠军）、双重 Thompson 采样（Copeland 版本）与均匀随机配对。曲线为 10 次运行平均的累积 Copeland 遗憾。循环强度 c 为 0 时，A 是 Condorcet 赢家，两种学习规则的曲线都趋于平缓；c 超过约 1 后不再存在 Condorcet 赢家，以其存在为前提的 RUCB 以恒定速率累积遗憾。随机配对的曲线超出绘图区后截断，图中标出其最终值。仅作示意，探索参数 α = 0.51。"
```

可尝试以下设置。默认设置下 A 是 Condorcet 赢家，两条曲线都像对数函数一样逐渐趋缓，双重 Thompson 采样最终更低：2000 次对决后约为 30，RUCB 约为 70，随机配对则约为 1000。把“循环强度 c”调到 1.5，此时 A、B、C 共享 Copeland 赢家的头衔；数据足够多之后，即使按乐观估计，也没有任何一条臂能胜过其他每一条臂，于是 RUCB 的冠军步骤在多数轮次中退化为随机选臂：其遗憾沿直线增长，2000 次对决时超过 200。双重 Thompson 采样的曲线则继续趋缓。再在循环强度为 0 时选择“险胜的冠军”：赢家以 0.55 的概率胜过所有臂，差距为 0.05，两种算法在 2000 次对决之后仍在付出数以百计的遗憾。上面各个界中的 $1/\Delta$ 反映的正是这一效应：差距越小，分出胜负所需的对决越多。

### Sparring 与最初的人在回路应用 {#sec-duel-sparring}

对称风格把两条臂视为两个参与者。SelfSparring[@sui2017multi]从同一个 Thompson 采样后验中抽取对决的每一条臂（多元对决时则抽取若干条臂中的每一条），因此算法是在与自身对决；在臂上放置高斯过程先验后，它还能在相似的臂之间共享信息。该方法的理论不如其应用充分。它假设“近似线性”（approximate linearity），即获胜概率近似为效用差的线性函数，作者认为这比强随机传递性更严格；对相互独立的臂，它证明了算法收敛到最优臂，并具有渐近最优速率 $O(K\ln T/\Delta)$；作者还指出，有限时间保证需要更精细的分析，而核化版本的分析尚付阙如。

这些算法已在人在回路的场景中实际运行。CorrDuel 是面向大量相关选项的对决赌博机，曾在一项实际进行的临床试验中选择脊髓刺激参数，作者称这是在线学习算法首次用于脊髓损伤治疗[@sui2017correlational]。CoSpar 借助 SelfSparring 的后验采样调节外骨骼步态，并加入共同主动反馈，使用户也能提出改进建议[@tucker2020preference]；@sec-cs-exoskeleton 沿这一研究路线展开。

### 走出有限臂的列表 {#sec-duel-beyond-arms}

有两类扩展通向连续问题。对连续凸问题，最早的对决赌博机论文 @yue2009interactively 提出了基于对决的梯度下降，@kumagai2017regret 则对强凸且光滑的代价证明了 $\sqrt{T\log T}$ 阶的遗憾，在对数因子以内最优。对定义在 $d$ 个特征上的线性效用，@saha2021optimal 考虑选择子集并观察其中赢家的设定，给出了遗憾为 $\sqrt{dT}$ 阶（不计对数因子）的算法，以及与子集大小无关的匹配下界：更大子集中的赢家并无帮助，这是 @sec-query-batch 的结论在有限维中的对应。

## 核化对决 {#sec-kernelized-dueling}

偏好贝叶斯优化是有无穷多条臂的对决赌博机：连续定义域中的每一个点都是一条臂，效用 $f$ 是光滑的。对普通评估，@sec-gp-bandits 的分析处理过同样的跨越：用最大信息增益 $\gamma_T$（@def-regret-gamma）代替臂的数目，并假设 $f$ 属于某个核的**再生核 Hilbert 空间**（reproducing kernel Hilbert space，RKHS），即由核函数鼓包构建的函数空间，再以范数界 $\lVert f\rVert_k \le B$ 限制 $f$ 的粗糙程度（@sec-regret-other-settings；这一范数界假定了什么，见 @sec-ka-norm-bound）。核化对决赌博机采取同样的两步。

从一次对决中学到的是差 $h(\vx, \vx') = f(\vx) - f(\vx')$。定义在成对输入上的函数需要定义在成对输入上的核，@sec-pref-model 已推导过这个核：当 $f \sim \GP(0, k)$ 时，差 $h$ 是成对输入上的高斯过程，其协方差即偏好核（那里用 $g$ 表示效用）。赌博机文献称之为**对决核**（dueling kernel）：

$$
k^D\big((\vx, \vx'), (\vy, \vy')\big) = k(\vx, \vy) - k(\vx, \vy') - k(\vx', \vy) + k(\vx', \vy').
$$ {#eq-duel-kernel}

给 $f$ 加一个常数不改变 $h$，对决核也无法察觉这个常数，这正是 @sec-pref-identifiability 的平移不变性。下文的分析用 $k^D$ 或 $k$ 的信息增益表述速率，两者以相同的速率增长[@pasztor2024bandits; @kayal2025bayesian]。若 $k$ 的特征函数在输入分布下的均值为零（例如圆周上的平稳核），对决核的每个特征值恰为 $k$ 的某个特征值的两倍；在区间上这一条件不成立，两者并不严格对应（@sec-ka-mercer）。

### 第一个核化界 {#sec-duel-kk}

@kirschner2021bias 给出了一个信息导向采样规则，并称之为第一个具有累积遗憾保证的高效核化对决赌博机算法。其反馈模型是定量的：一次对决返回 $f(\vx_1) - f(\vx_2) + \xi$，即效用差加上次高斯噪声（尾部不比高斯分布更重的噪声）。该模型涵盖二元回答，但仅限于“二元回答也是一种有界的带噪声观测”这一意义。设 $f$ 属于范数至多为 $B$ 的 RKHS，且 $k(\vx, \vx) \le 1$，则把每次对决的两个点都计入时，遗憾为 $\sqrt{T\beta_T(\gamma_T + \log 1/\delta)}$ 阶，$T$ 轮后约为 $\gamma_T\sqrt{T}$。他们的出发点是稳健性：在贝叶斯优化中，若两次评估带有共同的偏差（例如被调节的系统发生漂移），取差即可抵消该偏差，即使偏差无界，这个界仍保持次线性。该模型既不是 Bradley-Terry 模型，也不是概率单位模型，因此这个界不必付出下文非线性链接所带来的代价（推断，依据所述反馈模型）。更早的一项结果把对决与直接评估结合了起来[@xu2020zeroth]。

### Bradley-Terry 模型下的界，2024 至 2026 年 {#sec-duel-bt-bounds}

有四项结果分析了本书用于比较的模型：回答服从 Bernoulli 分布，其概率是效用差的逻辑函数，$\Prob(\vx \succ \vx') = \operatorname{sigmoid}\big(f(\vx) - f(\vx')\big)$，其中 $\operatorname{sigmoid}(a) = 1/(1 + e^{-a})$，即 @eq-cmp-logit 中 $\tau = 1$ 时的链接。这些结果都用核化逻辑回归为 $f$ 构造置信集，再通过乐观、淘汰或采样选择配对，区别在于所作的假设以及计算遗憾所用的单位。以下四段供阅读这些论文的读者参考；初读时可直接跳到 @tbl-duel-rates，本章其余部分所用的内容都在该表中。

**POP-BO**[@xu2024principledb]是乐观算法，以上一个点为参照乐观地选择下一个点。以效用计，其遗憾为 $O(\sqrt{\beta_T\gamma_T T})$，其中置信宽度 $\beta_T$ 本身按 $\sqrt{T}$ 乘以某个覆盖数对数的平方根增长；覆盖数是逼近函数类中每个函数所需的函数个数。对线性核与平方指数核，这给出 $T^{3/4}$ 乘以多对数因子；对 Matérn 核，结果仅在光滑度 $\nu$ 超过 $(d/4)\big(3 + d + \sqrt{d^2 + 14d + 17}\big)$ 时成立，这一阈值是 $d^2$ 阶的。作者把多出的因子（约为 $T^{1/4}$）解读为偏好反馈的代价，理由是标量评估蕴含偏好，反之则不然。

**最大最小下置信界算法（MaxMinLCB）**[@pasztor2024bandits]把选择一对点视为领导者与跟随者之间的博弈：领导者所选的点即使面对跟随者的最优反应也应表现良好，双方都以下置信界评判。它以偏好概率计算遗憾：一次对决的代价为 $\big(\Prob(\vx^\star \succ \vx_t) + \Prob(\vx^\star \succ \vx'_t) - 1\big)/2$，两个点都最优时为零。其定理 6 表明：以至少 $1 - \delta$ 的概率，对所有 $T$ 同时有 $R_T \le C_3\,\beta_T\sqrt{T\gamma_T} = O(\gamma_T\sqrt{T})$，其中 $\beta_T$ 按 $\sqrt{\gamma_T}$ 增长，常数 $C_3 = (8 + 2\kappa)/\sqrt{\log(1 + 4/(\lambda\kappa))}$ 含有 @sec-duel-link 的链接斜率常数 $\kappa$ 与核化逻辑回归的正则化权重 $\lambda$。作者在摘要中称该保证“速率最优”（rate-optimal）；但它是 $O(\gamma_T\sqrt{T})$ 而非 $O(\sqrt{\gamma_T T})$，比已知最好的速率高出一个 $\sqrt{\gamma_T}$ 因子，所以“速率最优”至多是相对于 GP-UCB 类分析而言（推断）。此外，该分析把选择限制在可能的最大值点集合之内。

**多轮偏好反馈学习算法（MR-LPF）**[@kayal2025bayesian]采用标量贝叶斯优化中的分批路线。它至多运行 $\lceil\log_2\log_2 T\rceil + 1$ 轮，每轮长度递增。在一轮之内，它在存活的候选中为不确定性最大的配对安排对决，且不查看回答；一轮结束时，若某个候选即使按乐观估计，胜过另外某个候选的机会也低于二分之一，即将其淘汰。它假设 $f$ 属于范数至多为 $B$ 的 RKHS，链接为逻辑函数，且候选集 $\X$ 有限，大小为 $|\X|$。其定理 4.1 在 $T \ge T_0$ 时成立，这一预热长度不依赖于 $T$，具体取值见论文附录；该定理可化简为

$$
R_T = \tilde O\!\left(\sqrt{\gamma_T\, T \log(|\X|/\delta)}\right),
$$ {#eq-duel-mrlpf}

这里遗憾以偏好概率计，$\tilde O$ 隐去了对数因子。链接斜率常数只在第一轮出现，因此不进入主导项。这一速率与最好的标量结果同阶，在上界层面否定了 POP-BO 的解读：$T^{1/4}$ 来自 POP-BO 基于覆盖数的置信宽度，而非偏好反馈本身（推断）。作者指出，与之匹配的标量下界假设高斯噪声，而 Bradley-Terry 模型对应的是 Gumbel 噪声，因此他们只把这一比较作为紧性的非正式论证，而不是证明。

**偏好反馈下的 Thompson 采样（PF-TS）**[@lazzaro2026finiteb]是用于偏好的 Thompson 采样：两个独立的后验样本各自相对于一个共同的锚点求最大值。以至少 $1 - 2\delta$ 的概率，其以偏好概率计的遗憾为 $\tilde O(\beta_T\sqrt{T\gamma_T})$，其中 $\beta_T = O(\sqrt{\gamma_T + \log(1/\delta)})$，即 $\tilde O(\gamma_T\sqrt{T})$；作者指出，这与 @chowdhury2017kernelized 对标量 Thompson 采样给出的界一致。连续定义域通过离散化处理，并假设核函数已知。在一维 Ackley 函数上（300 轮，30 次运行），它的累积遗憾低于 MR-LPF 与 POP-BO，与 MaxMinLCB 相当；PF-TS 与 MR-LPF 两篇论文有两位共同作者。

第五条路线以神经网络代替核函数。神经对决赌博机[@verma2025neural]只要随机传递性成立，即适用于 Bradley-Terry、Thurstone 及其他链接。其界以平均效用遗憾计，依赖于一个有效维度与链接的最小斜率，作者预计它弱于对应的标量神经网络结果。

上述分析针对的都是频率派估计量（即带置信集的核化逻辑回归）以及为分析而构造的算法。实践中实际运行的流程是 Laplace 近似的高斯过程后验配合 EUBO（@sec-pbo-loop），但没有一项结果分析这一流程；它的保证是 @sec-eubo-theory 中的一步贝叶斯最优性与有限定义域上的一致性（推断）。

## 速率并列比较 {#sec-dueling-rates}

@tbl-duel-rates 将各项核化结果与其对应的标量结果并列。有两个标量参照值得注意：GP-UCB 与 GP-TS 的分析给出 $O^*(\gamma_T\sqrt{T})$[@chowdhury2017kernelized]；一个分批纯探索算法在 $O(\log\log T)$ 批之内达到 $O^*(\sqrt{\gamma_T T})$，对若干种核函数接近最优[@li2022gaussian]。这里 $O^*$ 隐去对数因子。包含有限臂与神经网络结果的完整表格见 @tbl-theory-rates。

::: {.table #tbl-duel-rates title="核化对决界：反馈、遗憾单位、主要假设、速率，以及链接斜率常数 κ 是否乘在主导项上。"}
| 结果 | 反馈 | 遗憾单位 | 主要假设 | 速率 | κ 是否在主导项中 | 标量对应结果 |
|---|---|---|---|---|---|---|
| @kirschner2021bias | 效用差加次高斯噪声 | 效用，两个点都计 | RKHS 范数 $\le B$ | $\approx \gamma_T\sqrt{T}$ | 否（无链接） | GP-UCB |
| POP-BO[@xu2024principledb] | 逻辑链接 | 效用 | 紧定义域；Matérn 核需要 $d^2$ 阶的 $\nu$ | $O(\sqrt{\beta_T\gamma_T T})$，约为 $T^{3/4}$ | 经由置信集 | 弱于 GP-UCB |
| MaxMinLCB[@pasztor2024bandits] | 逻辑链接 | 偏好概率 | RKHS 范数 $\le B$ | $O(\gamma_T\sqrt{T})$ | 是 | GP-UCB |
| MR-LPF[@kayal2025bayesian] | 逻辑链接 | 偏好概率 | 有限 $\X$；$T \ge T_0$；分批 | $\tilde O(\sqrt{\gamma_T T\log \lvert\X\rvert})$ | 仅第一轮 | 分批纯探索 |
| PF-TS[@lazzaro2026finiteb] | 逻辑链接 | 偏好概率 | 离散化定义域；核函数已知 | $\tilde O(\gamma_T\sqrt{T})$ | 经由 $\beta_T$、$\gamma_T$ | GP-TS |
:::

由此可见，偏好理论几乎逐项重现了标量理论：乐观方法与 Thompson 采样达到 $\gamma_T\sqrt{T}$，与 GP-UCB、GP-TS 相同；分批淘汰达到 $\sqrt{\gamma_T T}$，与其标量原型相同（推断）。完全序贯的偏好算法能否达到 $\sqrt{\gamma_T T}$，仍是未解决的问题。对标量反馈，能达到这一速率的序贯算法已经存在[@salgia2021domain]；COLT 2021 上作为开放问题提出的，是 GP-UCB 本身能否达到它[@vakili2021open]，该问题已得到部分解决[@whitehouse2023sublinear]。

两族结果之间相差的 $\sqrt{\gamma_T}$ 因子在高维时影响最大。对 $d$ 维中光滑度 $\nu > 1/2$ 的 Matérn 核，不计对数因子时 $\gamma_T$ 按 $T^{d/(2\nu + d)}$ 增长[@vakili2021information]（即 @tbl-regret-gamma 的最后一行），因此 $\gamma_T\sqrt{T}$ 按 $T^{1/2 + d/(2\nu + d)}$ 增长。对常用的 Matérn 5/2 核，指数为 $1/2 + d/(5 + d)$，在 $d = 5$ 时达到 1：从五维起，这些序贯算法的界不再表明遗憾比 $T$ 增长得慢，也就不再提供任何信息。$\sqrt{\gamma_T T}$ 的指数为 $(\nu + d)/(2\nu + d)$，在任何维度上都小于 1，并且等于 @scarlett2017lower 中标量下界的指数（推断，由我们根据所述速率计算；@exr-duel-exponents）。下图绘出了两者。

```{figure}
//| figure: theory-rates
//| label: fig-duel-rates
//| fig-cap: "核化速率的并列比较。上：一维定义域上各个界的形状，所有常数与链接因子都取 1，γ_T 用贪心法估计；曲线高低仅作示意，且各结果的遗憾单位不同，只有增长趋势可以比较。下：T^a 的指数 a 随维度 d 的变化，依据已发表的 Matérn 核 γ_T 的阶，忽略对数因子（该阶针对 ν > 1/2 给出；选择 Matérn 1/2 时在边界上套用同一公式）。指数高于 1 时，界的增长快于 T，不提供任何保证。@sec-theory-rates 中也有同一幅图。"
```

### 链接的斜率 {#sec-duel-link}

若干个界中出现的常数 $\kappa$ 衡量链接函数可以平坦到什么程度。好得多的选项与差得多的选项对决，几乎总是较好的一方获胜，因此回答几乎是确定的，却几乎不包含关于它究竟好多少的信息。在逻辑曲线平坦之处，效用差的大幅变化只引起回答的微小变化，学习因而很慢。设分析必须容许的效用差范围为 $[-D, D]$，该常数就是链接在这一范围上最小斜率的倒数：

$$
\kappa = \sup_{\lvert a\rvert \le D} \frac{1}{\operatorname{sigmoid}'(a)}, \qquad \frac{1}{\operatorname{sigmoid}'(a)} = 2 + e^{a} + e^{-a}.
$$ {#eq-duel-kappa}

该常数随范围呈指数增长。在零附近，逻辑函数的斜率为 $1/4$，故 $\kappa$ 至少为 4；若效用可以位于 $[-5, 5]$ 中的任何位置，效用差可达 10，$\kappa$ 超过 22,000[@kayal2025bayesian]。因此，主导项中含有 $\kappa$ 的界，即使对 $T$ 的依赖看起来不错，在实际的范围下也可能失去意义。

标量逻辑赌博机最先遇到同样的问题。@faury2020improved 证明，早先 $\kappa\sqrt{T}$ 阶的保证可以改进到 $\sqrt{T}$ 阶，$\kappa$ 只出现在二阶项中；@abeille2021instance 证明了 $d\sqrt{T/\kappa}$ 阶的问题相关下界，并给出了匹配的上界：在链接平坦之处，回答可以预测，问题甚至可能更容易。对决方面，@di2025nearly 在线性效用与 sigmoid 链接下把 $\kappa$ 移出了主导项，MR-LPF 对核函数做到了这一点；MaxMinLCB、PF-TS 与神经对决赌博机的主导项中仍保留这一常数。

### 遗憾的单位 {#sec-duel-units}

表中混用了两种单位，二者不可互换。**效用遗憾**（utility regret）计算 $f(\vx^\star) - f(\vx_t)$，即损失了多少效用。**偏好概率遗憾**（preference-probability regret）计算 $\Prob(\vx^\star \succ \vx_t) - 1/2$，即换成最优选项时多出的胜率。后者会饱和：极差的选项与较差的选项都几乎必输，二者的代价都接近 $1/2$，而它们的效用损失可以相差任意多。

::: {.derivation title="两种单位如何比较"}
用 $\operatorname{sigmoid}$ 表示逻辑链接，令 $a = f(\vx^\star) - f(\vx) \in [0, D]$ 为效用差距。

1. $\operatorname{sigmoid}(0) = 1/2$，所以偏好概率遗憾为 $\operatorname{sigmoid}(a) - 1/2 = \int_0^a \operatorname{sigmoid}'(u)\,\dd u$。
2. 斜率 $\operatorname{sigmoid}'$ 在 0 处最大，等于 $1/4$，并在 $[0, \infty)$ 上递减。以 $1/4$ 作为被积函数的上界，得 $\operatorname{sigmoid}(a) - 1/2 \le a/4$。
3. 斜率递减，故 $\operatorname{sigmoid}$ 在 $[0, \infty)$ 上是凹的，位于连接 $(0, 1/2)$ 与 $(D, \operatorname{sigmoid}(D))$ 的直线之上：$\operatorname{sigmoid}(a) - 1/2 \ge a\,c_D$，其中 $c_D = \big(\operatorname{sigmoid}(D) - 1/2\big)/D$。因子 $c_D$ 总小于 $1/(2D)$，$D \ge 4$ 时已与之接近。
4. 因此 $a\,c_D \le \operatorname{sigmoid}(a) - 1/2 \le a/4$；$a \to 0$ 时趋近上界，$a = D$ 时取到下界。差距小时，一个单位的偏好概率遗憾约相当于 4 个单位的效用遗憾；差距达到最大时，则相当于 $1/c_D$ 个单位，约为 $2D$。（若以被积函数的最小值 $\operatorname{sigmoid}'(D) = 1/\kappa$ 作为下界，其中 $\kappa$ 如 @eq-duel-kappa，则得到更粗糙的 $\operatorname{sigmoid}(a) - 1/2 \ge a/\kappa$。该不等式成立，但远不够紧：$D = 10$ 时它容许 22,028 倍的因子，而最坏情况只有 20 倍。）
:::

所谓偏好算法与标量贝叶斯优化同阶，通常是拿偏好概率遗憾与标量效用遗憾相比。在最优点附近差距较小，这种比较在因子 4 以内是公平的；远离最优点时，这一因子随差距增大，在最大差距处约为 $2D$（推断）。

::: {.pitfall title="对速率的三种误读"}
**误读：MaxMinLCB 以 $O(\sqrt{\gamma_T T})$ 达到速率最优**。其定理 6 给出的是 $O(\gamma_T\sqrt{T})$，比 $\sqrt{\gamma_T T}$ 高出一个 $\sqrt{\gamma_T}$ 因子，且常数中含有 $\kappa$[@pasztor2024bandits]。

**误读：MR-LPF 证明了比较与评估具有同样的样本效率**。它证明的是一个上界，与最好的标量上界同阶，且有以下限定：以偏好概率为单位，针对有限候选集，在预热期 $T_0$ 之后成立，所用的分批算法在每一轮结束之前都不理会回答；其紧性只有非正式的论证[@kayal2025bayesian]。这一结果并不涉及每次查询所含的信息量，这一点也没有任何下界能够定论。

**误读：有了高斯过程先验，SelfSparring 只需要 $O(d)$ 而不是 $O(K)$ 个样本**。2018 年的一篇综述称，高斯过程先验把样本复杂度从 $O(K)$ 降到 $O(d)$[@sui2018advancements]，但 SelfSparring 的论文并未证明这样的定理，并且说明其核化版本的分析尚付阙如[@sui2017multi]。这只是一个猜想。
:::

## 缺失了什么 {#sec-dueling-lower-bounds}

下界刻画的是：任何算法在某个问题上至少要付出多少遗憾；据此可以判断一个上界能否改进（@sec-regret-lower-bounds）。有限条臂之间的对决，下界是已知的，即上文 @yue2012karmed 与 @komiyama2015regret 的界，并有算法与之匹配。线性效用的下界也已知：在以赢家为反馈、选择子集的设定下为 $\sqrt{dT}$ 阶，与子集大小无关[@saha2021optimal]。至于采用 Bradley-Terry 链接或概率单位链接的核化问题（即偏好贝叶斯优化所提出的问题），截至 2026 年 9 月，我们没有找到与算法无关的下界（@sec-theory-lower）。

标量问题则有下界。对 Matérn 核，任何算法在某个 RKHS 范数有界的函数上都要承受至少 $T^{(\nu + d)/(2\nu + d)}$ 阶的累积遗憾，并且至少需要 $(1/\varepsilon)^{2 + d/\nu}$ 次评估才能找到 $\varepsilon$ 最优点[@scarlett2017lower]。通往比较的唯一桥梁是 @kayal2025bayesian 的非正式论证：若把两次带噪声的评估只保留大小关系，变成一次比较，则这次比较所含的信息不会多于那两次评估，因此在相应的噪声下，偏好下界应至少为标量下界的一半。标量下界假设高斯噪声，而 Bradley-Terry 模型对应 Gumbel 噪声，所以这一论证并不构成证明。

缺失的下界有三个，按其缺失对理论的制约程度排列（推断）：非线性链接下核化对决的下界；刻画斜率常数 $\kappa$ 必然以何种方式进入核化偏好遗憾的下界；排序与集合选择的核化下界，它将表明 @sec-query-batch 中有限臂的发现（只有比赢家更丰富的回答才有帮助）能否推广。

那么，一次比较是否比一个数值更昂贵？坦率地说，这取决于设定（推断）。对有限条臂和线性效用，对决的速率在常数与链接因子以内与数值奖励的速率一致。对核函数，在 MR-LPF 的假设下，最好的上界同阶，但没有下界能说明这是否紧。此外，一个二元回答至多携带一比特信息，一个带噪声的数值所携带的信息也有限，因此速率相等并不意味着每次查询的信息量相等。

::: {.frontier title="已定、有争议与缺失"}
**已定**。有限条臂时，相对于 Condorcet 赢家的最优对数遗憾与匹配的下界均已知，Copeland 赢家也有 $K\log T$ 阶的算法。线性效用时，$\sqrt{dT}$ 阶是最优的。Kirschner 与 Krause（2021）给出了第一个关于对决的核化累积界，采用“差加噪声”的反馈模型；在逻辑链接下，POP-BO 给出的速率约为 $T^{3/4}$，MaxMinLCB 与 PF-TS 约为 $\gamma_T\sqrt{T}$，MR-LPF 约为 $\sqrt{\gamma_T T}$，最后一项针对有限候选集，且在预热期之后成立。

**有争议**。MR-LPF 的速率是否紧，作者只给出了非正式论证。比较是否与评估同样高效，只在特定假设下对上界成立。分批且阶最优的方法，与序贯但差一个 $\sqrt{\gamma_T}$ 因子的方法相比孰优孰劣：唯一的直接比较出自有重叠的作者，且限于低维。

**缺失**。非线性链接下的核化下界，以及 $\kappa$ 在其中的作用；遗憾为 $\sqrt{\gamma_T T}$ 的完全序贯偏好算法；核函数下排序与集合选择的界；对实践中所用的“Laplace 近似加 EUBO”流程的任何频率派分析。
:::

@sec-pbo-theory 完整介绍了这一理论，包括漂移、污染、反应时，以及 @sec-query-aggregation 背后的可识别性结果。

## 习题 {#sec-duel-exercises}

::: {.exercise #exr-duel-utility}
设效用 $u_1 > u_2 > \dots > u_K$，链接 $s$ 连续、严格递增且满足 $s(-a) = 1 - s(a)$，并有 $P_{ij} = s(u_i - u_j)$。（a）证明臂 1 是 Condorcet 赢家、Copeland 赢家与 Borda 赢家，且 von Neumann 赢家把全部权重放在臂 1 上。（b）证明强随机传递性成立。

::: {.solution}
（a）由对称性，$s(0) = 1/2$；又 $s$ 严格递增，故对每个 $j \ne 1$ 有 $P_{1j} = s(u_1 - u_j) > 1/2$：臂 1 是 Condorcet 赢家，胜 $K - 1$ 场，Copeland 得分达到可能的最高值。再看 Borda 得分，把臂 1 与任意一条臂 $k$ 比较。对每条第三方的臂 $j$，有 $s(u_1 - u_j) > s(u_k - u_j)$；在两者的直接对决中，$s(u_1 - u_k) > 1/2 > s(u_k - u_1)$。臂 1 平均值中的每一项都大于臂 $k$ 平均值中的对应项，所以臂 1 的 Borda 得分更高。最后看 von Neumann 赢家：臂 1 上的点质量对每个 $j$ 给出 $\sum_i \pi_i P_{ij} = P_{1j} \ge 1/2$，因而符合条件；它也是唯一的，因为放在其他任何一条臂 $i$ 上的权重平均而言都会输给臂 1。

（b）若 $\Delta_{ij} \ge 0$ 且 $\Delta_{jk} \ge 0$，则 $u_i \ge u_j \ge u_k$。于是 $u_i - u_k \ge u_i - u_j$ 且 $u_i - u_k \ge u_j - u_k$，再由 $s$ 递增得 $\Delta_{ik} \ge \max\{\Delta_{ij}, \Delta_{jk}\}$。
:::
:::

::: {.exercise #exr-duel-rps}
三条臂构成一个循环，方向与 @fig-duel-winners 相同：B 以概率 $1/2 + a$ 胜过 A，C 以概率 $1/2 + b$ 胜过 B，A 以概率 $1/2 + c$ 胜过 C，其中 $a, b, c > 0$。（a）证明不存在 Condorcet 赢家，且三条臂的 Copeland 得分相同。（b）证明 von Neumann 赢家在 A、B、C 上的权重为 $\boldsymbol{\pi} = (b, c, a)/(a + b + c)$。（c）取 $a = 0.15$，$b = 0.18$，$c = 0.46$，哪条臂的权重最大？权重遵循什么规律？

::: {.solution}
（a）每条臂恰好胜过另外两条臂中的一条、输给另一条，所以没有哪条臂能同时胜过其余两条，每条臂的 Copeland 得分都是 1。

（b）记收益为 $M_{ij} = P_{ij} - 1/2$，条件是对每一列 $j$ 有 $\sum_i \pi_i M_{ij} \ge 0$。非零收益为 $M_{BA} = a$、$M_{CB} = b$、$M_{AC} = c$ 及其相反数。A 列：$\pi_B M_{BA} + \pi_C M_{CA} = a\pi_B - c\pi_C$。B 列：$\pi_A M_{AB} + \pi_C M_{CB} = -a\pi_A + b\pi_C$。C 列：$\pi_A M_{AC} + \pi_B M_{BC} = c\pi_A - b\pi_B$。取 $\boldsymbol{\pi} \propto (b, c, a)$，三列分别为 $ac - ca = 0$、$-ab + ba = 0$ 与 $cb - bc = 0$：每一列都打平；权重均为正，归一化后和为 1。

（c）权重为 $(0.18, 0.46, 0.15)/0.79 \approx (0.23, 0.58, 0.19)$，B 的权重最大。每条臂的权重等于它未参与的那场对决的差距：B 的权重是 A 胜过 C 的差距。在舍入误差以内，这些正是 @fig-duel-winners 默认设置下的差距与权重。
:::
:::

::: {.exercise #exr-duel-exponents}
对 $d$ 维中光滑度为 $\nu$ 的 Matérn 核，取 $\gamma_T \propto T^{d/(2\nu + d)}$ 并忽略对数因子。（a）证明 $\sqrt{\gamma_T T}$ 中 $T$ 的指数等于标量下界的指数 $(\nu + d)/(2\nu + d)$。（b）证明 $\gamma_T\sqrt{T}$ 增长得至少与 $T$ 一样快，当且仅当 $d \ge 2\nu$。（c）对 Matérn 5/2，MaxMinLCB 与 PF-TS 的界从哪个维度起不再是次线性的？

::: {.solution}
（a）$\sqrt{\gamma_T T} \propto T^{\frac12(1 + d/(2\nu + d))} = T^{(2\nu + 2d)/(2(2\nu + d))} = T^{(\nu + d)/(2\nu + d)}$。

（b）$\gamma_T\sqrt{T}$ 的指数为 $1/2 + d/(2\nu + d)$，当 $d/(2\nu + d) \ge 1/2$，即 $2d \ge 2\nu + d$，亦即 $d \ge 2\nu$ 时，该指数至少为 1。

（c）$\nu = 5/2$ 时为 $d = 5$：从五维起，这些界至少线性增长，不提供任何保证，而 $\sqrt{\gamma_T T}$ 在任何维度上都保持次线性。
:::
:::

::: {.exercise #exr-duel-kappa}
对逻辑链接 $\operatorname{sigmoid}(a) = 1/(1 + e^{-a})$，验证 $1/\operatorname{sigmoid}'(a) = 2 + e^{a} + e^{-a}$，并计算效用差最大可达 2、6、10 时的 $\kappa$。利用 @sec-duel-units 中的推导，差距为 10 时，偏好概率遗憾最多会把效用遗憾低估多少？

::: {.solution}
$\operatorname{sigmoid}'(a) = e^{-a}/(1 + e^{-a})^2 = 1/\big((1 + e^{-a})(1 + e^{a})\big) = 1/(2 + e^{a} + e^{-a})$。所以 $\kappa$ 在 2 处约为 $2 + 7.39 + 0.14 = 9.5$，在 6 处约为 $2 + 403.4 = 405$，在 10 处约为 $2 + 22026.5 = 22{,}028$。差距为 10 时，$\operatorname{sigmoid}(10) - 1/2 \approx 0.49995$，而效用差距为 10：比值约为 20，即推导中的上端 $1/c_D = D/(\operatorname{sigmoid}(D) - 1/2) \approx 2D$，远低于 $\kappa = 22{,}028$。无论选项差到什么程度，它在偏好概率遗憾中的代价都几乎是同样的 $1/2$。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @yue2012karmed 定义了 K 臂对决赌博机问题及其遗憾，提出了 Interleaved Filter，并给出了匹配的下界。
- @sui2018advancements 综述了各种算法（IF、RUCB、MergeRUCB、RMED、D-TS、Sparring、SelfSparring）与几种替代的赢家定义；@bengs2021preference 是篇幅更长的综述，按对偏好矩阵所作的假设组织内容。
- @zoghi2014relative 与 @wu2016double 分别是 RUCB 与双重 Thompson 采样的原始论文；@zoghi2015copeland 讨论 Copeland 赢家。
- @kirschner2021bias、@pasztor2024bandits、@kayal2025bayesian 与 @lazzaro2026finiteb 给出了核化的界，每篇都精确说明了各自的反馈模型与遗憾单位。
- @scarlett2017lower 给出了标量下界，任何偏好下界都要与之比较；@faury2020improved 解释了逻辑赌博机中的链接斜率常数。
