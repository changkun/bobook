---
status: done
synopsis: "从比较中学习的已证结论：有限臂与线性对决赌博机的结果；2021 至 2026 年的核化遗憾界，及其链接函数、假设与遗憾单位；EUBO 的决策论结果；缺失的下界；观测模型的理论；可识别性；漂移、污染、反应时与停止。"
---

# 理论：从对决赌博机到核化偏好优化 {#sec-pbo-theory}

找到最优选项需要多少次比较？一次比较提供的信息能否与一个数值相当？@sec-dueling-bandits 介绍了对决的理论及其核化界；本章完整报告这一理论，给出每个结果的假设，并指出证明止于何处。相关理论分为四层：有限臂对决赌博机、线性与情境对决赌博机、连续凸对决优化、核化偏好赌博机。只有最后一层对应偏好贝叶斯优化本身，其结果也出现得最晚。

截至 2026 年 9 月的研究现状可以概括为三句话。一个分批算法的核化上界已与阶最优的标量贝叶斯优化同阶，但任何核化偏好问题都还没有下界。实践中使用的流程，即 Laplace 近似加 EUBO，只具有一步贝叶斯最优性与有限定义域上的一致性。可识别性与聚合理论则表明，当人与人之间存在差异，或回答依赖于隐藏情境时，成对数据是最弱的反馈形式。

::: {.note title="本章使用的记号"}
- $T$：轮数（查询次数）。$K$：有限问题中臂（选项）的数目。$d$：输入或特征的维度。
- $\gamma_T$：核函数在 $T$ 次观测后的最大信息增益（@def-regret-gamma、@tbl-regret-gamma）。
- $B$：效用在核函数的**再生核 Hilbert 空间**（reproducing kernel Hilbert space，RKHS）中的范数上界，衡量函数相对于核函数的粗糙程度（@sec-regret-other-settings、@sec-ka-rkhs）。
- $\kappa$：链接函数在相关效用范围内最小斜率的倒数（@sec-theory-link-slope）。
- $\tilde O$：忽略对数因子的增长阶。$\Omega$：阶的下界。
- **两种遗憾单位。** 效用遗憾累加 $f(\vx^\star) - f(\vx_t)$。偏好概率遗憾累加 $\Prob(\vx^\star \succ \vx_t) - 1/2$，并对一次查询的两个点取平均；两者为何不能互换，见 @sec-duel-units。
:::

## 有限臂与线性对决赌博机 {#sec-theory-finite}

**2017 年以前。** @sec-dueling-algorithms 中的有限臂结果是这一理论的基石。Interleaved Filter 的期望遗憾为 $O(K \log T / \Delta_{\min})$，在强随机传递性与随机三角不等式下与其下界相匹配 [@yue2012karmed]；RMED 达到了一个渐近下界，该下界用 Bernoulli 分布之间的 Kullback-Leibler 散度（@sec-kl）表示 [@komiyama2015regret]。对决赌博机梯度下降在连续凸空间上的期望遗憾为 $T^{3/4}$ 阶 [@yue2009interactively]；情境对决赌博机引入了 **von Neumann 赢家**（von Neumann winner），即选项上的一种随机选择，它以至少二分之一的概率胜过任何单个选项 [@dudik2015contextual]。

**2017 年及以后。** González 等人的论文 [@gonzalez2017preferentialb] 既没有遗憾定理，也没有收敛定理（@sec-history-2017）。SelfSparring [@sui2017multi] 假设“近似线性”（approximate linearity），即获胜概率是效用差的近似线性函数；作者称这一要求比强随机传递性更严格。其定理 1 证明独立臂版本收敛到最优臂，定理 2 给出渐近最优的无遗憾速率 $O(K \ln(T)/\Delta)$。核化多对决尚无分析；2018 年一篇综述称高斯过程先验能把样本复杂度从 $O(K)$ 降到 $O(d)$ [@sui2018advancements]，但这只是猜想（@sec-duel-units）。Winner Stays [@chen2017dueling] 的**弱遗憾**（weak regret）为 $O(N^2)$，与 $T$ 无关，其中 $N$ 为臂数；按弱遗憾的定义，只要所展示的两个选项中有一个是最优选项，这一轮就记为零代价。@bengs2021preference 的综述按各结果对成对获胜概率矩阵所作的假设，整理了遗憾结果与**可能近似正确**（probably approximately correct，PAC）样本复杂度结果。后来，@saha2022versatile 首先在以 Condorcet 赢家为基准时达到了最优的 $O(\sum_i \log T / \Delta_i)$。

**连续凸对决。** @kumagai2017regret 在代价函数强凸且光滑时，用随机镜像下降得到 $O(\sqrt{T \log T})$ 的遗憾，并借助凸优化中的下界论证该结果在对数因子以内最优；摘要没有说明对链接函数的假设。@saha2021duelingb 给出了每对选项只产生一个带噪声比较比特时的查询复杂度，并证明了非平稳在线凸情形下的不可能性；他们 2025 年的论文处理了一般的转移函数 [@saha2025dueling]。@blum2024dueling 在单调对手下证明了 $\Omega(d)$ 的下界。

**线性与情境对决。** 在 @saha2021optimal 的设定中，每轮提供 $K$ 个带情境特征的项目，学习者从中选出大小为 $q$ 的子集，并观察到一个带噪声的赢家。他们给出了最优的 $\tilde O(\sqrt{dT})$ 算法与相匹配的 $\Omega(\sqrt{dT})$ 下界，且该下界与子集大小 $q$ 无关：来自更大子集的赢家反馈并无帮助。随后出现了高效算法与一般链接函数的结果 [@saha2022efficient; @bengs2022stochastic; @di2024variance]，Feel-Good Thompson 采样达到了接近极小极大的 $\tilde O(d\sqrt{T})$ [@li2024feel]。**Borda 遗憾**（Borda regret）以一个选项对其他所有选项的平均获胜概率来衡量该选项；@wu2024borda 对 Borda 遗憾证明了 $\Omega(d^{2/3} T^{2/3})$ 的下界与相匹配的上界。@di2025nearly 在有 $C$ 个标签被对手翻转时得到 $\tilde O(\kappa d\sqrt{T} + \kappa dC)$（该文用 $\kappa$ 表示链接函数的最小斜率，即本书所用之量的倒数，因此原文的界是除以这个量），并给出几乎匹配的下界；对 sigmoid 链接，他们把 $\kappa$ 从主项中去掉了。@sekhari2023contextual 则证明，仅用比较查询即可达到与观察奖励的标准情境赌博机相当的遗憾。Saha 的 $\sqrt{dT}$ 与 Feel-Good Thompson 采样的 $d\sqrt{T}$ 之间的差别来自臂集合不同（前者每轮 $K$ 个项目，后者为大的或连续的集合），两者并不矛盾（推断）。

**基于偏好的强化学习。** 在从轨迹偏好中学习策略的问题上（@sec-adj-pbrl），已有结果从渐近的贝叶斯无遗憾 [@novoseller2020dueling] 与第一个有限时间分析 [@xu2020preference]，发展到一般函数逼近 [@chen2022human] 与第一个贝叶斯简单遗憾保证 [@agnihotri2026best]。@zhu2023principled 证明，在 Plackett-Luce 模型下，完整的 $K$ 元最大似然估计量与把排序拆成成对比较的估计量都收敛，前者渐近地更有效。

## 第一个核化结果 {#sec-theory-kernelized}

2021 年以前已有一个基于核函数的结果。@xu2020zeroth 同时允许直接查询与对决，其 COMP-GP-UCB 在 $T$ 次直接查询后的简单遗憾为 $O(\Phi/\sqrt{T})$；其中 $\Phi$ 是在定义域的一部分上计算的信息增益，这一部分是经过比较之后仍可能包含最优点的候选区域。第一个关于累积遗憾的核化对决界由 @kirschner2021bias 给出（@sec-duel-kk），其适用范围取决于所依赖的假设：

1. **反馈。** 论文的式 2 是定量的对决反馈，$d_t = f(\vx_t^1) - f(\vx_t^2) + \xi_t$，其中 $\xi_t$ 是方差代理为 $\rho^2$ 的次高斯噪声。该模型只在有界噪声属于次高斯噪声的意义上涵盖二元反馈。
2. **函数类。** $f$ 属于已知的 RKHS，范数至多为 $B$，且 $k(\vx, \vx) \le 1$。
3. **遗憾。** 每次对决中两个点的效用差距之和。
4. **定理 1。** 遗憾为 $O\big(\sqrt{T\, \beta_{T,\delta}\,(\gamma_T + \log 1/\delta)}\big)$，对 $T$ 轮（论文中记作 $n$）约为 $\gamma_T\sqrt{T}$。
5. **定理 2。** 在最优点唯一的有限定义域上，遗憾为 $O\big(\Delta_{\min}^{-1}\beta(\gamma_T + \log(T/\delta))\big)$；对线性核为 $O(\Delta_{\min}^{-1} d^2 \log(T)^2)$，对径向基函数（平方指数）核为 $O(\Delta_{\min}^{-1} \log(T)^{2d+2})$。

这一结果不涵盖 Bradley-Terry 链接或概率单位链接下的 Bernoulli 结果，也不涵盖这类链接带来的代价 $\kappa$；Bradley-Terry 链接下的核化遗憾分析始于 2024 年的 POP-BO（推断，依据定理所陈述的反馈模型）。

## Bradley-Terry 界，2024 至 2026 年 {#sec-theory-bt}

有四个算法的分析针对偏好贝叶斯优化实际面临的设定：回答服从 Bernoulli 分布，其概率是效用差的逻辑函数（@sec-bradley-terry），效用属于某个 RKHS。@sec-duel-bt-bounds 已介绍过这些算法；本节给出定理本身，以及其中的常数与条件。

**POP-BO。** @xu2024principledb 假设定义域紧致，$f$ 属于某个 RKHS，反馈为 Bernoulli 反馈，$\Prob = \operatorname{sigmoid}(f(\vx) - f(\vx'))$，即 @eq-cmp-logit 的逻辑函数。算法在似然比置信集内乐观地选择，并以上一轮的点为参照点。

- **定理 5.2。** 效用遗憾 $R_T = O\big(\sqrt{\beta_T \gamma_T T}\big)$，其中 $\beta_T = O\big(\sqrt{T \log(T\, \mathcal{N}(\mathcal{B}_f, 1/T, \lVert\cdot\rVert_\infty)/\delta)}\big)$，$\mathcal{N}$ 是函数类的覆盖数，即覆盖该函数类所需的小球个数。对线性核与径向基函数核，由此得到 $T^{3/4}$ 乘以多对数因子（其定理 5.5）。对 Matérn 核，指数大于 $3/4$，且论文只在 $\nu > (d/4)\big(3 + d + \sqrt{d^2 + 14d + 17}\big)$ 时给出该界，即光滑度参数 $\nu$ 须达到 $d^2$ 的量级。
- **定理 5.4。** 所报告解的差距为 $O\big(\sqrt{\beta_T \gamma_T}/\sqrt{T}\big)$。
- **注 5.6。** 作者提出，偏好反馈大约要多付出一个 $T^{1/4}$ 因子，依据的直觉是：数值评估蕴含偏好，反之则不然。

后来的论文常把 POP-BO 的速率简写为 $\tilde O((\gamma_T T)^{3/4})$；引用时应注明其单位是效用遗憾。

**MaxMinLCB。** @pasztor2024bandits 把选择一对选项的问题表述为 **Stackelberg 博弈**（Stackelberg game），即一方先作出承诺、另一方随后应对的博弈，并为核化逻辑估计量构造了偏好置信序列。**定理 6**：以至少 $1 - \delta$ 的概率，对所有 $T$ 有

$$
R_T \le C_3\, \beta_T \sqrt{T \gamma_T} = O(\gamma_T \sqrt{T}),
$$ {#eq-theory-maxmin}

其中 $\beta_t = 4LB + 2L\sqrt{(2\kappa/\lambda)(\gamma_t + \log 1/\delta)}$，$\kappa = \sup_{\lvert a\rvert \le B} 1/\operatorname{sigmoid}'(a)$，$C_3 = (8 + 2\kappa)/\sqrt{\log(1 + 4/(\lambda\kappa))}$；$s$ 是链接函数，$L$ 是其 Lipschitz 常数（斜率的上界），$\lambda$ 是正则化参数。这里的遗憾是偏好概率遗憾；摘要中的“速率最优”（rate-optimal）至多在相对于 GP-UCB 类分析的意义上成立（推断；@sec-duel-bt-bounds）。@kayal2025bayesian 将其概括为 $\tilde O(\gamma_T \kappa^2 \sqrt{T})$。

**MR-LPF。** @kayal2025bayesian 的多轮偏好反馈学习算法假设：$f$ 属于已知核函数的 RKHS，范数至多为 $B$；核函数以 1 为界；只采用 Bradley-Terry（逻辑）链接；候选集 $\X$ **有限**。算法运行 $R \le \lceil \log_2 \log_2 T\rceil + 1$ 轮，各轮长度为 $N_1 = \sqrt{T}$ 与 $N_r = \sqrt{N_{r-1} T}$；每轮之内按最大核方差选择点对，每轮结束时，凡是对某个对手获胜概率的上置信界低于二分之一的点，都予以淘汰。

- **定理 4.1。** 存在一个与 $T$ 无关的常数 $T_0$（见其附录 B），使得对所有 $T \ge T_0$，以至少 $1 - \delta$ 的概率有 $R_T \le 2CR\, \beta_{(R)}(\delta) \sqrt{\gamma_{4\lambda}(T)}\,(\sqrt{T} + 1)$，其中 $\beta_{(r)}(\delta) = L\big(B + \sqrt{(\kappa_r/\lambda)\log(2R|\X|/\delta)}\big)$，$|\X|$ 是候选集 $\X$ 的大小（论文中记作 $N_\X$），$\kappa_1 = \kappa$，当 $r > 1$ 时 $\kappa_r = 6$。简化后为 $\tilde O\big(\sqrt{\gamma_T T \log(|\X|/\delta)}\big)$。
- **$\kappa$ 的去向。** 它只在第一轮出现，因而不进入主项。论文指出，效用取值于 $[-5, 5]$ 时，$\kappa$ 可以超过 22,000。
- **推论 4.5。** 找到满足 $\Prob(\vx^\star \succ \hat\vx) - 1/2 \le \varepsilon$ 的解所需的比较次数，对线性核为 $\tilde O(d \log(1/\delta)/\varepsilon^2)$，对径向基函数核为 $\tilde O(\log(1/\delta)/\varepsilon^2)$，对 Matérn 核为 $\tilde O\big(\log(1/\delta)/\varepsilon^{2 + d/\nu}\big)$，与标量反馈下阶最优的样本复杂度同阶。
- **紧性。** 作者指出，Scarlett 等人的下界假设高斯噪声，而 Bradley-Terry 对应 Gumbel 噪声，因此两者不能严格地形式比较；他们只把这一比较作为紧性的非正式论证，并论证偏好反馈的下界应至少为标量下界的一半。

凡称 MR-LPF“与标量贝叶斯优化相匹配”，都应同时说明它与标量设定的六处不同（常见的误读见 @sec-duel-units）：遗憾单位是偏好概率；$T_0$ 是否隐含对 $\kappa$ 或 $e^B$ 的依赖，尚无人核查；$\log |\X|$ 来自联合界，连续定义域因此需要离散化论证；算法是分批的，在一轮之内不自适应；最优性来自非正式的比较；每次查询涉及两个点，两者都计入遗憾。一个由机器生成评审意见的网站声称，该文定理 4.7 所用的一个 Loewner 序不等式在 $\lambda$ 较小时可能不成立 [@pith2026arxiv]；这一说法未经同行评审，也未得到人类来源的证实，而 PF-TS 的论文把 MR-LPF 的速率作为正确结果加以引用。

**PF-TS。** @lazzaro2026finiteb 分析了偏好反馈下的 Thompson 采样：抽取两个独立的后验样本，分别相对于一个共同的锚点求最大值，得到一次查询的两个点。**定理 1**：以至少 $1 - 2\delta$ 的概率，$R_T = \tilde O\big(\beta_T\sqrt{T\gamma_T}\big)$，其中 $\beta_T = O\big(\sqrt{\gamma_T + \log(1/\delta)}\big)$，即以偏好概率遗憾计为 $\tilde O(\gamma_T\sqrt{T})$；$\kappa$ 通过岭项 $\lambda\kappa$ 进入 $\beta_T$ 与 $\gamma_T$。论文称这个界与 Chowdhury 与 Gopalan 2017 年为标准 Thompson 采样建立的界相匹配，而后者在标量贝叶斯优化中本身并非阶最优。连续定义域必须离散化为 $(B G w d T^2)^d$ 个点，核函数假定已知；实验对象是一维 Ackley 函数，以及一个包含三种金属 63 种组成的催化剂数据集。在 Ackley 函数上，PF-TS 的累积遗憾低于 MR-LPF 与 POP-BO（@sec-duel-bt-bounds），但到 300 轮的时域终点时，MR-LPF 的瞬时遗憾仍有竞争力；这两篇论文有共同作者（Vakili、Shiu）。

**神经对决赌博机。** @verma2025neural 对链接函数 $\mu$ 假设 $\kappa_\mu = \inf \mu'(f(\vx) - f(\vx')) > 0$；只要随机传递性成立，其结果就适用于 Bradley-Terry 噪声、Thurstone 噪声与指数噪声。平均效用遗憾为 $\tilde O\big((\sqrt{d_{\text{eff}}}/\kappa_\mu + B\sqrt{\lambda/\kappa_\mu})\sqrt{T d_{\text{eff}}}\big)$，其中 $d_{\text{eff}}$ 是由所有成对情境差构造的有效维度；网络宽度必须是 $T$ 等量的多项式。作者预计这个界弱于标量神经赌博机的界。@oh2026neural 给出了 $\tilde O\big(d\sqrt{\sum_t \sigma_t^2} + \sqrt{dT}\big)$，并把宽度要求降到 $\tilde\Omega(T^6)$。

## 速率比较 {#sec-theory-rates}

@tbl-theory-rates 汇总了连续或核化偏好优化的主要结果，并列出标量贝叶斯优化的参照：GP-UCB 类与 GP-TS 类分析的 $O^*(\sqrt{T}\gamma_T)$ [@chowdhury2017kernelized]；分批纯探索（BPE）在 $O(\log\log T)$ 批之内达到 $O^*(\sqrt{T\gamma_T})$，对若干种核函数接近最优 [@li2022gaussian]；对 Matérn 核，累积遗憾的下界 $\Omega(T^{(\nu + d)/(2\nu + d)})$ 与简单遗憾样本复杂度的下界 $\Omega((1/\varepsilon)^{2 + d/\nu})$ [@scarlett2017lower]。这里的 $O^*$ 与 $\tilde O$ 一样隐去对数因子。

::: {.table #tbl-theory-rates title="连续与核化偏好优化的速率：反馈、遗憾单位、假设及对应的标量结果。"}
| 结果 | 反馈与链接函数 | 遗憾 | 主要假设 | 速率 | 主项中是否含 $\kappa$ | 对应的标量结果 |
|---|---|---|---|---|---|---|
| SelfSparring [@sui2017multi] | 多对决；近似线性链接 | 有限臂强遗憾 | 独立臂 | 渐近 $O(K\ln T/\Delta)$；核版本没有界 | 不适用 | 与有限臂赌博机渐近同阶 |
| Kumagai [@kumagai2017regret] | 带噪声的比较；强凸光滑代价 | 对决遗憾 | 强凸、光滑 | $O(\sqrt{T\log T})$ | 摘要未说明 | 在凸优化下界的意义上，于对数因子以内最优 |
| Xu 等 [@xu2020zeroth] | 对决加直接查询 | 简单遗憾 | RKHS | $O(\Phi/\sqrt{T})$ | 不适用 | GP-UCB 类，信息增益在基于比较的约束集上计算 |
| Kirschner 与 Krause [@kirschner2021bias] | 效用差加次高斯噪声（线性链接） | 两个点效用差距之和 | 范数 $\le B$ | $O(\sqrt{T\beta_T(\gamma_T + \log 1/\delta)})$，约为 $\gamma_T\sqrt{T}$ | 否 | 与 GP-UCB 形式相同 |
| POP-BO [@xu2024principledb] | 逻辑 | 效用；参照点为上一个点 | 紧致定义域；Matérn 核要求 $\nu$ 达到 $d^2$ 量级 | $O(\sqrt{\beta_T\gamma_T T})$，约为 $T^{3/4}$ 乘以多对数因子 | 通过置信集 | 弱于 GP-UCB |
| MaxMinLCB [@pasztor2024bandits] | 逻辑；一处脚注称分析可能推广到其他对称递增链接 | 偏好概率 | 范数 $\le B$ | $O(\gamma_T\sqrt{T})$ | 是（约为 $\kappa^2$） | 与 GP-UCB 同阶 |
| 神经对决赌博机 [@verma2025neural] | 一般链接 | 平均效用 | 网络宽度为多项式 | $\tilde O((\sqrt{d_{\text{eff}}}/\kappa_\mu)\sqrt{Td_{\text{eff}}})$ 及其他项 | 是 | 作者预计弱于 NeuralUCB |
| MR-LPF [@kayal2025bayesian] | 逻辑 | 偏好概率 | 有限 $\X$；$T \ge T_0$；分批 | $\tilde O(\sqrt{\gamma_T T\log |\X|})$ | 仅第一轮 | 与 BPE 同阶；紧性只有非正式论证 |
| PF-TS [@lazzaro2026finiteb] | 逻辑 | 偏好概率 | 连续定义域需离散化；核函数已知 | $\tilde O(\gamma_T\sqrt{T})$ | 通过 $\beta_T$ 与 $\gamma_T$ | 与 GP-TS 同阶 |
| qEUBO [@astudillo2023qeubob] | 逻辑或常数似然 | 贝叶斯简单遗憾 | 有限 $\X$；$q = 2$；差距条件或常数似然条件 | $o(1/n)$ | 不适用 | 贝叶斯的有限定义域结果，不能与频率派速率相比较 |
:::

偏好理论几乎逐项重现了标量贝叶斯优化的结果：乐观算法与 Thompson 采样达到 $\gamma_T\sqrt{T}$，分批淘汰达到 $\sqrt{\gamma_T T}$（推断；@sec-dueling-rates）。完全序贯的偏好算法能否达到 $\sqrt{\gamma_T T}$，仍是未解决的问题。标量设定中有一个相关问题，于 COLT 2021 提出 [@vakili2021open]：GP-UCB 本身能否达到这一速率。更精巧的标量算法已经达到 [@salgia2021domain]；@whitehouse2023sublinear 部分解决了这一问题，给出了 Matérn 核下 GP-UCB 的次线性界，在对数因子以内为 $T^{(\nu + 2d)/(2\nu + 2d)}$ 阶，仍高于下界；Matérn 核信息增益的改进速率来自 @vakili2021information，成立条件为 $\nu > 1/2$。两类算法相差的因子 $\sqrt{\gamma_T}$ 有多重要，取决于 $\gamma_T$ 增长的快慢，而后者又取决于核函数（@tbl-regret-gamma）；@fig-theory-rates 直观地展示了这一点。

```{figure}
//| figure: theory-rates
//| label: fig-theory-rates
//| fig-cap: "核化速率并列比较，即 @sec-dueling-rates 中的图。上图：一维定义域上各个界的形状，其中 $\gamma_T$ 由 @sec-regret-info-gain 的贪心规则在 300 个输入的网格上计算，正则化为 0.25，所有常数与链接因子 $\kappa$ 均取 1；曲线高度仅作示意，各界单位也不同，只有增长趋势可以比较。虚线 $T$ 表示从不改进的学习者的增长。下图：$T^a$ 中的指数 $a$ 随维度的变化，由我们根据已发表的 $\gamma_T$ 阶（@tbl-regret-gamma）算出，忽略对数因子；该阶的成立条件是 $\nu > 1/2$ [@vakili2021information]，选择 Matérn 1/2 时是在边界上套用同一公式。"
```

可以尝试以下几点：

- **默认设置（Matérn 5/2，$T = 1000$）。** 右侧的括号标出 $\gamma_T\sqrt{T}$ 与 $\sqrt{\gamma_T T}$ 之间相差的因子 $\sqrt{\gamma_T}$。所有常数取 1 时，在这一时域内 $\gamma_T\sqrt{T}$ 仍高于直线 $T$，而 $\sqrt{\gamma_T T}$ 远低于它：即使不计常数，多出的 $\sqrt{\gamma_T}$ 也决定了一个界是否具有实际意义。
- **切换到平方指数核。** $\gamma_T$ 按 $\log T$ 的幂增长，两类算法在渐近意义上只差多对数因子；POP-BO 的 $T^{3/4}$ 也随之出现。渐近地看，它是三者中增长最快的，但在图示的时域内，它低于 $\gamma_T\sqrt{T}$，接近 $\sqrt{\gamma_T T}$：增长的阶与实际时域内的大小可能并不一致。
- **切换到 Matérn 1/2，并加大长度尺度。** 核函数粗糙时，$\gamma_T$ 在一维中的增长几乎与 $\sqrt{T}$ 相当，因此 $\gamma_T\sqrt{T}$ 的指数在 $d = 1$ 时已经为 1。加大长度尺度会在固定时域内降低 $\gamma_T$，但不改变指数。
- **选 Matérn 5/2 查看下图。** $\gamma_T\sqrt{T}$ 的指数为 $1/2 + d/(5 + d)$，在 $d = 5$ 时达到 1；$\sqrt{\gamma_T T}$ 的指数为 $(2.5 + d)/(5 + d)$，在所有维度上都低于 1（@exr-duel-exponents）。POP-BO 的 Matérn 结果即使在 $d = 1$ 时也要求 $\nu > 2.41$，因此对 Matérn 5/2 只适用于一维。

## 链接斜率与遗憾单位 {#sec-theory-link-slope}

常数 $\kappa$ 是链接函数在分析所允许的效用差范围内最小斜率的倒数。对逻辑链接，它随这一范围指数增长，因为逻辑曲线在远离零处几乎是平的：MR-LPF 的作者指出，效用取值于 $[-5, 5]$ 时，它可以超过 22,000（@exr-duel-kappa）。@sec-duel-link 介绍了标量逻辑赌博机如何把 $\kappa$ 移出主项 [@faury2020improved; @abeille2021instance]；@sec-duel-units 说明了效用遗憾与偏好概率遗憾为何只在差距较小时成比例，比例因子为 $1/4$（@exr-theory-units）。这里补充两点。其一，在对决问题中，@di2025nearly（针对 sigmoid 链接）与 MR-LPF（针对核函数）把 $\kappa$ 从主项中去掉了，而 MaxMinLCB、PF-TS 与神经对决赌博机仍把它保留在主项的常数中；除神经方法的 $\kappa_\mu$ 外，没有任何核化结果针对一般的非逻辑链接给出显式的斜率依赖，MaxMinLCB 也只在一处脚注中提到其分析可能推广到其他对称递增的链接。其二，POP-BO 在 2024 年提出的直觉，即偏好要多付出 $T^{1/4}$ 的因子，在上界层面已被 MR-LPF 否定：这一差距来自 POP-BO 基于覆盖数的置信宽度，而非偏好反馈本身（推断）。

## 决策论结果 {#sec-theory-decision}

qEUBO [@astudillo2023qeubob] 是偏好贝叶斯优化中主要的贝叶斯决策论结果。查询 $X = (\vx_1, \dots, \vx_q)$ 的**一步贝叶斯最优值**（one-step Bayes optimal value）为 $V_n(X) = \E_n\big[\max_{\vx} \E_{n+1}[f(\vx)] - \max_{\vx}\E_n[f(\vx)] \mid X_{n+1} = X\big]$，即对 $X$ 再获得一个回答，预期能使最好的后验均值提高多少；停止于 $N$ 时，推荐点为 $\argmax_{\vx} \E_N[f(\vx)]$；带噪声的似然为 $L_i(f(X); \lambda) = \exp(f(\vx_i)/\lambda) / \sum_j \exp(f(\vx_j)/\lambda)$，$\lambda = 0$ 表示回答无噪声。四个定理及其条件见 @sec-acqf-eubo。对这些定理有三点解读（推断）：

- **$o(1/n)$ 为何这么快。** 这些条件把问题变成了几乎必然存在正效用差距的有限识别问题。这一结果不涉及连续定义域，也没有说明速率如何依赖于选项数或维度，并且不能与 $T^{-1/2}$ 阶的频率派简单遗憾相比较。
- **充分条件排除了什么。** 几乎必然成立的差距界排除了边际分布连续的普通高斯过程先验；而效用不同时获胜概率恒等于常数 $a > 1/2$ 的似然，也不是实践中使用的概率单位似然或逻辑似然。这一结果最好理解为 qEUBO 具有一致性的证据与 qEI 不具一致性的证明，而不是实际偏好贝叶斯优化的速率。
- **与 2026 年的批评并不矛盾。** 一步最优性不蕴含多步最优性或渐近最优性，因此与两篇预印本报告的过度利用和病态问题并不冲突（@sec-acqf-failures）；在连续定义域上，qEUBO 既没有这类结果，也没有频率派遗憾界。

::: {.keyidea title="理论分析的算法并非实践中的流程"}
理论论文分析的是建立在频率派核估计量之上的淘汰算法、乐观算法或 Thompson 采样算法。实践中使用的则是 Laplace 近似下的高斯过程后验加 EUBO 类采集函数，这一流程只具有一步贝叶斯最优性与有限定义域上的一致性。没有论文分析过这一实践流程的频率派遗憾，我们也没有找到连续定义域上高斯过程偏好采集的贝叶斯遗憾界（例如基于信息比的界）（推断）。需要理论保证的实践者必须运行经过分析的算法；使用默认设置时，应当把理论理解为对问题本身的陈述，而不是对所用方法的陈述。
:::

## 下界 {#sec-theory-lower}

下界刻画的是：对一类问题，任何算法都必须在其中某个问题上承受多少遗憾（@sec-regret-lower-bounds）。没有下界，就不能称一个上界是最优的。

**已确立阶最优性的情形。** 对有限臂对决赌博机，@saha2022versatile 达到了实例相关的 $\sum_i \log T/\Delta_i$，RMED 则渐近地达到这一下界 [@komiyama2015regret]。对抗、分批与弱遗憾等变体，以及 Condorcet 赢家与 Copeland 赢家的识别，也都有下界 [@saha2021adversarial; @saha2021dueling; @agarwal2022batched; @saad2024weak; @haddenhorst2021identification; @bengs2024identifying]；线性与情境情形则有 @sec-theory-finite 中的 $\Omega(\sqrt{dT})$ 下界、Borda 下界与污染下界。

**一次查询携带多少信息。** @sec-query-batch 报告了 Plackett-Luce 模型下有限臂情形的答案：对 $n$ 条臂，来自 $k$ 元子集的赢家反馈的最优样本复杂度为 $O((n/\varepsilon^2)\ln(1/\delta))$，与成对比较相同，而 top-$m$ 排序反馈能使之降为原来的 $m$ 分之一 [@saha2019pac]。同一批作者还给出了相匹配的实例相关界 [@saha2020pac]，以及 top-$m$ 反馈下 $O((n/m)\ln T)$、完整排序下 $O((n/k)\ln T)$ 的阶最优遗憾 [@saha2019combinatorial]。在符号反馈的凸优化中，$m$ 路 argmin 反馈带来的增益为 $\min\{\log m, d\}$ 阶 [@saha2024faster]；对线性 Plackett-Luce 模型，@lee2025preference 得到 $\tilde O\big((d/T)\sqrt{\sum_t 1/\lvert S_t\rvert}\big)$，其中 $S_t$ 是第 $t$ 轮展示的子集，由此可以证明更大的子集有帮助；而如果反馈只按过去的经验表现给臂排序，就不可能得到对数阶的实例相关遗憾 [@maran2024bandits]。“来自更大子集的赢家反馈没有帮助”与“更大的子集有帮助”看似矛盾，区别在于反馈类型：每轮要获得更多信息，反馈就不能只有赢家。对偏好贝叶斯优化而言，这预示着：请人“从 $K$ 个中选一个”的界面，在最坏情况的阶上并不优于成对对决，而排序界面可以更好；这一预测在核化偏好贝叶斯优化中尚未检验（推断）。

**核化情形：没有下界。** 在我们的检索范围内（PMLR 2017 至 2025 年、NeurIPS 2017 至 2024 年，以及对 2025 与 2026 年文献的浏览），我们**没有找到 Bradley-Terry 链接或概率单位链接下核化偏好反馈的与算法无关的下界**；有针对性的检索只找到了标量核函数下界与有限臂对决下界。MR-LPF 的近似最优性依赖于与 Scarlett 等人高斯噪声下界的非正式比较。任何偏好下界都须与标量核函数下界 [@scarlett2017lower; @cai2021lower] 以及 @iwazaki2025near 的时变核函数下界相比较。@sec-dueling-lower-bounds 列出了缺失的三类下界，并讨论了这一缺失对“比较是否比数值更昂贵”这一问题意味着什么；在核函数情形中，唯一确定的结论是：在带预热期的分批、有限定义域设定中，比较至多多付出常数因子与对数因子（推断）。

## 观测模型的理论 {#sec-theory-obs}

**偏斜高斯过程定理及其出处。** Benavoli、Azzimonti 与 Piga 的三篇论文需要区分：第一篇是分类论文，发表于 *Machine Learning* 第 109 卷（2020 年），引入了偏斜高斯过程 [@benavoli2020skew]；第二篇是 GECCO 2021 Companion 论文（arXiv 2008.06677），包含偏好贝叶斯优化的后验定理 [@benavoli2021preferentialb]；第三篇发表于 *Machine Learning* 第 110 卷（2021 年），证明了与正态似然、仿射概率单位似然及其乘积的共轭性 [@benavoli2021unified]。因此，把偏好后验定理归于“Machine Learning 2020”是错误的，“Machine Learning 2021”也只适用于一般的共轭性结论。

**统一偏斜正态**（unified skew-normal，SUN）分布是多元高斯分布的推广：将多元高斯密度乘以一个正态分布函数，使其倾斜（见 @sec-skew-gp）。下述定理中，$\Phi_m$ 是 $m$ 个独立标准正态变量的分布函数；$\Omega = D_\Omega \bar\Omega D_\Omega$ 把协方差矩阵分解为由标准差构成的对角矩阵 $D_\Omega$ 与相关矩阵 $\bar\Omega$。

::: {.theorem #thm-theory-skewgp title="偏好后验是偏斜高斯过程（Benavoli、Azzimonti 与 Piga，GECCO 2021 Companion）"}
设 $f \sim \GP(\xi, \Omega)$，关于 $n$ 个输入处的取值 $f(X)$ 有 $m$ 个观测，其似然为仿射概率单位似然 $p(W \mid f(X)) = \Phi_m(W f(X))$，其中 $W$ 是 $m \times n$ 的数据矩阵。

1. （定理 1。）$f(X)$ 的后验是统一偏斜正态分布 $\mathrm{SUN}_{n,m}$，其偏斜参数为 $\Delta = \bar\Omega D_\Omega W^\T$、$\gamma = W\xi$ 与 $\Gamma = W\Omega W^\T + I_m$。
2. （定理 2。）$f$ 的后验是偏斜高斯过程，均值函数为 $\xi$，协方差函数为 $\Omega$，偏斜函数为 $\Delta(\vx, X) = \Omega(\vx, X) W^\T$。
3. （推论 1。）对 Chu 与 Ghahramani 的似然 $\prod_k \Phi\big((f(\mathbf{v}_k) - f(\mathbf{u}_k))/(\sqrt2\,\sigma)\big)$，为保证可识别性取 $\sigma^2 = 1/2$，令 $W_{ij} = V_{ij} - U_{ij}$ 即得后验；其中 $V$ 与 $U$ 分别标记每次比较中被偏好与被拒绝的输入。
:::

对参数化的概率单位回归，@durante2019conjugate 此前已证明其与统一偏斜正态分布的共轭性。该定理给出的是概率单位模型的精确贝叶斯推断，而不是一致性结果或速率结果。它解释了 Laplace 近似与期望传播为何会误报对决概率，这一点对依赖预测获胜概率的采集函数很重要（@sec-obs-inference）；该定理不适用于逻辑链接（推断）。@wu2026knowledge 的扩展偏斜正态前瞻后验正是以此为基础。

**后验一致性。** **后验一致性**（posterior consistency）指后验随数据积累而集中到真实函数上；**收缩速率**（contraction rate）刻画集中的快慢。对于 Chu-Ghahramani 类型的高斯过程偏好模型，我们没有找到这类定理。最接近的结果有：POP-BO、MaxMinLCB 与 MR-LPF 中核化逻辑估计量的频率派置信集；qEUBO 在有限定义域上的贝叶斯一致性；SelfSparring 独立臂版本的渐近收敛。

**随机传递性。** 上述遗憾定理对获胜概率所作的正则性假设各不相同，其定义见 @sec-duel-transitivity：强、中等与弱随机传递性，以及随机三角不等式；强传递性并不蕴含随机三角不等式 [@bengs2021preference]。@yue2012karmed 需要强随机传递性与三角不等式；SelfSparring 的近似线性比强随机传递性更严格；神经对决的结果只要随机传递性成立即成立；@suk2023we 的跟踪结果需要强随机传递性与三角不等式的交集。任何“效用加单调链接”的模型，无论是 Bradley-Terry 模型还是概率单位模型，在每一时刻都同时满足这两个条件（推断，由定义推出；@exr-theory-sst）。@chau2022inconsistent 对可排序性假设提出了质疑；他们的猜想适用范围有多大，见 @sec-obs-extensions。

## 可识别性与聚合 {#sec-theory-identifiability}

若一个量的不同取值产生不同的数据分布，从而在原则上足够多的数据能将它们区分开，就称这个量是**可识别的**（identifiable）。一组结果表明，当回答依赖于模型观测不到的因素时，成对数据是最弱的反馈。

**隐藏情境。** @siththaranjan2024distributional 研究的设定是：有限个选项、无限多数据、均匀抽取的点对、L2 正则化的 Bradley-Terry 损失；每个回答都可能依赖于模型观测不到的**隐藏情境**（hidden context），即回答者是谁、处于何种状态：

- **定理 3.1。** Bradley-Terry 偏好学习按 **Borda 计数**（Borda count）隐式地聚合隐藏情境：学到的效用满足 $\hat u(a) > \hat u(b)$ 当且仅当 $\mathrm{BC}(a) > \mathrm{BC}(b)$，其中 $\mathrm{BC}(a)$ 是 $a$ 胜过一个随机对手的平均概率。
- **定理 3.2。** 若隐藏情境噪声在各选项之间独立同分布，且其差值的支撑集包含零的某个邻域，则学到的顺序与期望效用的顺序相同。
- **命题 3.3。** 多数偏好可以与期望效用一致，而 Bradley-Terry 却不一致。
- **定理 3.4。** 任何使用无限比较数据的确定性方法，都不能总是恢复期望效用，即使只要求在相差一个单调变换的意义下恢复也不行。

作者指出，标注者因此有虚报偏好的动机。@an2026differential（预印本）也指出，Bradley-Terry-Luce 损失对应于 Borda 计数。对单用户的偏好贝叶斯优化而言，这意味着：若一个人的回答依赖于未建模的情境，如疲劳、表述框架或顺序，高斯过程效用恢复出的就是一种 Borda 型聚合，而不是平均效用（推断；算例见 @exr-theory-hidden）。

**异质人群。** @chidambaram2026direct（AISTATS 2026；arXiv 2405.15065 与 2510.15716 是同名的不同版本）针对**随机系数 logit**（random-coefficient logit）模型证明了三个结果。在该模型中，每个用户都有各自的偏好权重 $\beta$：

- **引理 4.1。** 若每个用户只做一次二元比较，则即使用户无限多，类型分布也不可识别：$\beta$ 与 $-\beta$ 各占一半的混合在任何位置都给出概率 0.5。
- **定理 4.2**（重述 Fox 等人 2012 年的结果）。若各阶矩满足 Carleman 条件，特征的支撑集包含零附近的一个开集，$\beta$ 与特征独立，且至少有 3 个选项，则类型分布非参数可识别，即使数据只是三个选项的不完备排序也是如此。
- **引理 4.3。** 特征差矩阵满秩时，同一用户所做的大量多样的二元比较可以识别该用户的 $\beta$。

**环、偏序与情境效应。** @liu2026statistical 证明了三点：偏好能用奖励模型表示，当且仅当不存在 Condorcet 环（即按多数意见 $a$ 胜过 $b$、$b$ 胜过 $c$、$c$ 又胜过 $a$ 的情形）；在 Luce 模型下，Condorcet 环出现的概率以指数速度趋于 1；Nash 人类反馈学习得到混合策略，当且仅当没有哪个回答被多数认为优于其他所有回答。@drago2025theoretical 证明，构造与偏好偏序相容且维度最小的多目标效用是 NP 困难的。@depeuter2024preference 的出发点是一种带情境效应的偏好选择认知模型；他们使用该模型的易处理替代形式，在大规模人类数据上的推断优于 Bradley-Terry 的各种变体。@cao2026provably 则针对 Plackett-Luce 子集选择模型证明，仅从查询中学习会遇到平移不变性障碍，需要赌博机反馈作为锚。

综合来看（推断）：以单个固定效用上的遗憾衡量，成对比较在阶上并不天然比数值更昂贵；但在识别异质群体、在隐藏情境下恢复期望效用、处理成环的偏好这些问题上，成对数据是最弱的反馈，排序反馈优于只给出赢家的反馈。所有偏好贝叶斯优化遗憾界都以单一效用为前提，这一假设受到双重质疑：一是 Chau 等人的经验猜想，二是 Liu 等人的渐近结果。这对查询设计意味着什么，见 @sec-query-aggregation 与 @sec-many-users。

## 漂移、污染、反应时与停止 {#sec-theory-drift}

真实的人有四个特点：偏好会改变，回答会出错，作答需要时间，会话必须结束。每一点都已有一些理论，但大多不在核化设定之内。

**漂移。** 有限臂情形的理论已经成熟。@saha2022optimal 针对对抗性的偏好序列给出了 $O(\sqrt{KT})$ 的静态遗憾，对 $S$ 次有效切换给出了 $\tilde O(\sqrt{SKT})$ 的动态遗憾，对连续变化量 $V_T$ 给出了 $\tilde O(V_T^{1/3}K^{1/3}T^{2/3})$ 的动态遗憾，且都有相匹配的下界；ANACONDA [@kleinebuening2023anaconda] 能适应未知的切换次数；平稳分段 [@kolpaczki2022non]（预印本）与高维 Bradley-Terry 模型中的变点 [@li2022detecting] 也各有结果。@suk2023we 证明，在 Condorcet 类或强随机传递性类下，不可能以 $O(\sqrt{KLT})$ 的速率适应“显著变化”（significant shifts，$L$ 为显著变化的次数）；在常见的类中，使之可行的最大一类是强随机传递性与三角不等式的交集。@liu2026online 证明，若反馈按瞬时效用排序，次线性外部遗憾一般不可能实现，而当效用序列的总变差为次线性时则成为可能；@son2025right 给出了未知漂移下直接偏好优化的界。对于标量核函数，@iwazaki2025near 给出了非平稳核化赌博机的第一个与算法无关的下界；@bogunovic2016time 的定理 4.1 表明，在其 Markov 模型中，若每步变化量 $\varepsilon$ 固定，任何算法的累积遗憾都是 $\Omega(T\varepsilon)$。由于“效用加链接”的模型在每一时刻都满足强随机传递性与三角不等式，在核化偏好贝叶斯优化中跟踪漂移的障碍是技术性的，即缺少核函数加链接函数情形的动态遗憾分析，而非已知的不可能性（推断）；偏好贝叶斯优化中也没有漂移效用的模型（@sec-obs-extensions），因而没有动态遗憾保证。

**污染与偏差。** @agarwal2021stochastic 给出的遗憾与涉及 Condorcet 赢家的被污染比较个数呈线性关系，并证明这种线性依赖是必要的；@saha2022versatile 在被污染的 Condorcet 设定中只多付出加性的 $2C$。在线性情形中，@di2025nearly 给出了 $\tilde O(\kappa d\sqrt{T} + \kappa dC)$，其中 $\kappa$ 乘在污染项上；@oh2026robust 给出了 $\tilde O(d(\sqrt{T} + C + D))$，其中 $D$ 度量延迟。已知或未知的评价者偏差 [@tang2025tackling]，以及基于人类反馈的强化学习中被污染的点对 [@bukharin2024robust; @mandal2025corruption]，也都已有研究。在核函数情形中，唯一的稳健性结果仍是 Kirschner 与 Krause 2021 年的线性链接偏差模型；标量参照是 @bogunovic2020corruption。Siththaranjan 等人的结果意味着标注者有理由虚报，因此引出真实反馈的机制也很重要；一篇 AISTATS 2026 论文用 Vickrey-Clarke-Groves 机制实现了这一点 [@landolt2026eliciting]，我们只核实了其题名与发表会议。

**反应时。** 在人类信号通道中，只有反应时有理论分析，且仅针对线性效用。@li2024enhancing 使用 EZ 扩散模型，即描述选择过程中证据如何累积的一种简化漂移扩散模型（@sec-neuro-ddm），并从理论与实验两方面表明，对于偏好强烈的查询，反应时能补充选择所含的信息。@benkert2026time（工作论文，2026 年版本）证明，二元选择频率只能识别潜在偏好分布上的一个点，加上单调的反应时函数后则能在多个点上识别。@shvartsman2024response 的高斯过程反应时模型没有理论保证。

**停止。** 与偏好贝叶斯优化中带保证的停止规则最接近的结果如下。@haddenhorst2021testification 把识别 Condorcet 赢家与检验其是否存在结合起来，使学习者可以停止并拒绝作答，同时给出了期望样本复杂度的下界和一个在对数因子以内最优的算法。@shukla2024preference 针对由锥确定顺序的向量奖励，给出了一个下界与相匹配的偏好感知 Track-and-Stop 算法。固定置信度的识别也有各自的停止规则 [@bengs2024identifying; @saha2019pac; @saha2020pac]。Bıyık 等人的参数化规则，以及尚未移植到成对似然上的标量规则，见 @sec-hd-stopping。

## 已定、有争议与缺失 {#sec-theory-status}

::: {.frontier title="已定、有争议与缺失"}
**已定。** 有限臂对决赌博机的实例最优对数遗憾及相匹配的下界 [@komiyama2015regret; @saha2022versatile]。线性与情境对决在链接常数与对数因子以内的极小极大速率 [@saha2021optimal; @li2024feel]。2024 年以前的形式化保证：SelfSparring（2017 年）、Kumagai（2017 年）、Xu 等人（2020 年）、Kirschner 与 Krause（2021 年）、qEUBO（2023 年）。Kirschner 与 Krause 2021 年的结果是第一个关于累积遗憾的核化对决界，采用“差加噪声”模型。Bradley-Terry 链接下的核化上界：POP-BO 约为 $T^{3/4}$（效用遗憾）；MaxMinLCB 与 PF-TS 为 $\gamma_T\sqrt{T}$；MR-LPF 为 $\sqrt{\gamma_T T}$（分批、有限定义域、有预热期）；后三者以偏好概率遗憾计。概率单位偏好后验是偏斜高斯过程 [@benavoli2021preferentialb]。来自更大子集的赢家反馈在阶上没有增益，top-$m$ 排序带来 $m$ 倍增益（有限臂与线性模型）[@saha2019pac; @saha2021optimal]。

**有争议。** MR-LPF 的最优性：作者自己称紧性论证是非正式的，一份未经证实的机器评审也质疑了其定理 4.7 中的一个不等式。“比较与数值一样样本高效”：这是分批、有限定义域、逻辑链接设定中上界阶的相等，而非信息量的相等，且尚无定论。序贯与分批：MR-LPF 阶最优但分批，MaxMinLCB 与 PF-TS 序贯但损失 $\sqrt{\gamma_T}$ 因子，两类结果并存；唯一的直接比较来自 PF-TS 的论文，该文与 MR-LPF 有共同作者，且实验是低维的。

**缺失**，按对偏好贝叶斯优化理论的限制程度排列（推断）：Bradley-Terry 链接或概率单位链接下、显式给出 $\kappa$ 依赖的核化下界；阶最优的完全序贯算法；核化界中的排序似然或多选项似然；对实践中所用近似后验（Laplace 近似、期望传播）的分析，而非对精确估计量或频率派估计量的分析；核函数情形中漂移、污染与反应时的理论；连续定义域上 qEUBO 类规则的贝叶斯遗憾；把贝叶斯推荐 $\argmax_{\vx}\E_N f(\vx)$ 与某种保证联系起来的停止规则；高斯过程偏好模型的后验一致性；以及对决 Thompson 采样（DTS）与幻觉信念的遗憾界、KernelSelfSparring 为无遗憾算法的证明。
:::

## 习题 {#sec-theory-exercises}

::: {.exercise #exr-theory-hidden}
回答某个成对问题的人中，一半属于类型 1，对选项 $(a, b, c)$ 的效用为 $(10, 1, 0)$；另一半属于类型 2，效用为 $(0, 2, 1)$。每个人都按自己的效用确定性地作答，模型不知道回答者是谁。（a）计算每个选项的期望效用，以及每个选项胜过其他每个选项的概率。（b）计算 Borda 计数，即一个选项胜过均匀选取的另一选项的平均概率。（c）按 @sec-theory-identifiability 所述 Siththaranjan 等人的定理 3.1，Bradley-Terry 学习恢复出什么顺序？它与期望效用一致吗？

::: {.solution}
（a）期望效用分别为：$a$ 为 $5$，$b$ 为 $1.5$，$c$ 为 $0.5$，故 $a \succ b \succ c$。类型 1 偏好 $a$ 甚于 $b$，类型 2 偏好 $b$ 甚于 $a$，故 $\Prob(a \succ b) = 1/2$；同理 $\Prob(a \succ c) = 1/2$；两种类型都偏好 $b$ 甚于 $c$，故 $\Prob(b \succ c) = 1$。
（b）$\mathrm{BC}(a) = (1/2 + 1/2)/2 = 0.5$，$\mathrm{BC}(b) = (1/2 + 1)/2 = 0.75$，$\mathrm{BC}(c) = (1/2 + 0)/2 = 0.25$。
（c）学到的效用按 Borda 计数给选项排序，即 $b \succ a \succ c$，而期望效用把 $a$ 排在首位：类型 1 从 $a$ 获得的巨大收益从不在二元回答中显现，因为二元回答只记录偏好的方向。这些概率已是无限数据下的极限，再做更多同样的比较也不会改变学到的顺序；这正是定理 3.4 所描述的情形。在一个人的会话中，这些“类型”可以是情绪、表述框架或疲劳状态（推断）。
:::
:::

::: {.exercise #exr-theory-units}
在逻辑链接下，设某个查询与最优点的效用差距为 $g$，比较两种遗憾单位：效用遗憾 $g$ 与偏好概率遗憾 $\operatorname{sigmoid}(g) - 1/2$。分别计算 $g = 0.1$ 与 $g = 4$ 时的两者。在什么条件下，可以放心地把以一种单位给出的速率与以另一种单位给出的速率相比较？

::: {.solution}
当 $g = 0.1$ 时：$\operatorname{sigmoid}(0.1) - 1/2 \approx 0.0250$，接近 $g/4 = 0.025$，因为 $\operatorname{sigmoid}$ 在零处的斜率是 $1/4$。当 $g = 4$ 时：$\operatorname{sigmoid}(4) - 1/2 \approx 0.482$，而 $g/4 = 1$；偏好概率遗憾在 $1/2$ 处饱和，此时按比例换算的值约为它的 2 倍，即大差距被低估；$g$ 越大，低估越严重。只有当求和中起主导作用的差距较小时，两种单位才成比例；因此，一种单位下的速率只有在这一近似下、以因子 $1/4$ 换算，才能转为另一种单位，跨论文比较时应当说明这一点。
:::
:::

::: {.exercise #exr-theory-sst}
设效用为 $u$，链接 $F$ 严格递增，满足 $F(0) = 1/2$ 与 $F(-a) = 1 - F(a)$，且 $\Prob(i \succ j) = F(u_i - u_j)$。证明强随机传递性成立。再证明：若 $F$ 在 $[0, \infty)$ 上是凹函数，则随机三角不等式也成立。

::: {.solution}
记 $a = u_i - u_j$，$b = u_j - u_k$。若 $\Delta_{ij} \ge 0$ 且 $\Delta_{jk} \ge 0$，则由于 $F$ 递增且 $F(0) = 1/2$，有 $a, b \ge 0$。于是 $u_i - u_k = a + b \ge \max\{a, b\}$，又因为 $F$ 递增，$\Delta_{ik} = F(a + b) - 1/2 \ge \max\{F(a), F(b)\} - 1/2$，这就是强随机传递性。对三角不等式，令 $G(x) = F(x) - 1/2$，则 $G(0) = 0$，且 $G$ 在 $[0, \infty)$ 上是凹的。满足 $G(0) = 0$ 的凹函数在该区间上是次可加的：在 $0$ 与 $a + b$ 之间应用凹性，得 $G(a) \ge \tfrac{a}{a + b}G(a + b)$ 与 $G(b) \ge \tfrac{b}{a + b}G(a + b)$，两式相加得 $G(a) + G(b) \ge G(a + b)$，即 $\Delta_{ik} \le \Delta_{ij} + \Delta_{jk}$。逻辑链接与概率单位链接都递增、对称，且在 $[0, \infty)$ 上为凹，因此本书的模型在每一时刻都满足这两个条件；@sec-theory-drift 中的推断即以此为依据。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @bengs2021preference 是对决赌博机的综述，按对成对获胜概率所作的假设组织内容；@sui2018advancements 是篇幅更短的早期综述。
- @kirschner2021bias、@xu2024principledb、@pasztor2024bandits、@kayal2025bayesian 与 @lazzaro2026finiteb 给出了核化结果；阅读每个定理时，都应连同其反馈模型与遗憾单位一起看。
- @scarlett2017lower 与 @vakili2021information 给出了标量下界与信息增益速率，每个偏好结果都须与之比较。
- @astudillo2023qeubob 代表贝叶斯决策论一方；其条件值得仔细推敲。
- 回答依赖于隐藏情境时 Bradley-Terry 学习恢复出的是什么，@siththaranjan2024distributional 阐述得最为清楚。
