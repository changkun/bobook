---
status: done
synopsis: "查询选择规则从启发式到决策理论的演变、多个研究组各自独立发现的失效模式、查询的各种形式与偏好贝叶斯优化的各类问题变体，以及已发表的比较为何不能简单合并：每项比较都在各自的维度与噪声水平下进行。"
---

# 采集函数、查询形式与问题扩展 {#sec-acquisition-frontier}

采集函数是决定下一步询问什么的规则（@sec-acq-definition）。
在偏好贝叶斯优化中，它选出一对或一组选项展示给人；@sec-choosing-pairs 与 @sec-eubo 已介绍主要的候选规则。
本章依据 2017 年至 2026 年 9 月的研究文献逐一考察这些规则：各规则的来历、已证明的性质、表现好坏的条件，以及除“A 还是 B？”之外查询还能采取哪些形式。

采集函数的设计经历了三个阶段。2017 至 2021 年以启发式规则与信息论规则为主；2022 与 2023 年转向决策理论，研究者证明最优选项期望效用（expected utility of the best option，EUBO）及其多选项形式 qEUBO 在回答无噪声时具有一步贝叶斯最优性；2024 至 2026 年出现了以频率派遗憾为导向的核化对决算法（@sec-pbo-theory），EUBO 这一默认方法也受到批评并得到修补。
有两个发现贯穿下文。
其一，不同研究组报告的失效模式相互印证。
其二，各项实证比较的设定彼此不可比；预算小且噪声接近实际时，随机查询有时并不落后。

## 2017 至 2021 年：启发式与信息 {#sec-acqf-early}

**2017 年以前的两个规则。** @brochu2007active 的交互式贝叶斯优化把已查询点中后验均值最大者作为第一个选项，把相对于它期望改进最大的点作为第二个选项（@sec-ei）；@astudillo2023qeubob 后来指出，两个选项的 qEUBO 若强制包含当前最优点，就恰好退化为这一规则。
**贝叶斯分歧主动学习**（Bayesian active learning by disagreement，BALD）由 @houlsby2011bayesian（预印本）为高斯过程分类器提出。它选择回答与模型参数之间互信息最大的查询（@sec-mutual-information），即各个可能的模型分歧最大的查询。
@benavoli2021preferentialb 的对决信息增益把 BALD 推广到偏好。

**González 等人的三个规则。** @gonzalez2017preferentialb 在对决空间上提出了三个采集函数（@sec-dueling-formulation）。
**纯探索**（pure exploration，PE）选择对决结果方差最大的一对。
**Copeland 期望改进**（Copeland expected improvement，CEI）计算 Condorcet 赢家软 Copeland 值的一步前瞻改进；Condorcet 赢家指以大于二分之一的概率胜过其他每个选项的选项。
**对决 Thompson 采样**（dueling Thompson sampling，DTS）取偏好函数的一个连续 Thompson 样本，据此计算软 Copeland 得分，以得分最大的点为第一个点（@sec-thompson）；第二个点纯粹用于探索，取与第一个点对决时结果最不确定的点。
在他们的一维与二维函数实验中（@tbl-acqf-studies），DTS 始终是最好的策略；对决赌博机基线 Sparring 需要约 4,000 次迭代，才能接近 DTS 在 200 次对决中达到的水平。

**多方对决与容易回答的问题。** SelfSparring 把**多方对决**（multi-dueling，每轮比较多个选项）归约为普通的赌博机问题，用 Thompson 采样求解；其核版本 KernelSelfSparring 加入高斯过程先验，使信息能在选项之间共享[@sui2017multi]。
常用的**体积消除**（volume removal）目标按一个回答预期能排除多大一部分奖励参数空间来为查询打分；@byk2019asking 证明（其定理 1），这一目标的全局最优点是由 $K$ 个相同选项组成的平凡查询。
改用信息增益后，所选查询偏向人能有把握回答的问题，在模拟与一项用户研究中学习也更快。
这项工作属于采用参数化奖励的基于偏好的奖励学习，不属于高斯过程偏好贝叶斯优化（@sec-adj-pbrl）。

**2020 与 2021 年：新规则集中涌现。** 投影偏好贝叶斯优化[@mikkola2020projective]为其投影查询设计了五个规则，包括投影期望改进与偏好坐标下降。
@benavoli2021preferentialb 以当前赢家 $\vx_r$ 为参照定义了三个规则：**对决上置信界**（dueling upper confidence bound），取 $f(\vx) - f(\vx_r)$ 的 95% 可信区间上端；对决 Thompson 采样；EIIG，即改进概率的对数加上 $k$ 倍的对决信息增益，取 $k = 0.1$ 或 $0.5$。
@nguyen2021top 提出**多项预测熵搜索**（multinomial predictive entropy search，MPES），称之为第一个面向偏好观测下贝叶斯优化的信息论采集函数（@sec-entropy-search）。它对一次查询的全部输入联合优化，但必须枚举所有可能的回答，因而只适用于较小的查询集。
@siivola2021preferential 将批量期望改进（qEI）与批量 Thompson 采样改用于批次赢家反馈，即由人指出一组选项中最好的一个。

**区分两种不确定性。** @fauvel2021efficient（预印本）把对决结果的不确定性分为**认知**（epistemic）部分与**偶然**（aleatoric）部分：前者可以通过更多数据消除；后者是人自身的随机性，数据再多也无法消除。
他们的**最大不确定挑战**（maximally uncertain challenge，MUC）以后验均值最大的点为擂主，以使 $\Phi(f(\vx_1) - f(\vx))$ 的认知方差最大的点为挑战者；该规则有闭式解，也有批量版本。
同年出现了第一个具有累积遗憾保证的核化对决算法[@kirschner2021bias]，其反馈模型在 @sec-theory-kernelized 中考察。

## 2022 至 2023 年：EUBO 与决策论转向 {#sec-acqf-eubo}

**偏好探索中的 EUBO。** EUBO 最早出现于**偏好探索贝叶斯优化**（Bayesian optimization with preference exploration，BOPE）。在这一设定中，每次实验产生多个结果，决策者对这些结果的效用通过比较来学习。
@lin2022preferenceb 用 EUBO 选择展示给决策者的两个结果向量，并证明它是一步贝叶斯最优的偏好探索策略（@sec-eubo-theory）。
有些结果无法实现，两个变体针对的正是这一点：EUBO-ζ 从结果模型的一个后验样本生成结果，EUBO-f̃ 只比较很可能实现的结果。
在不加限制的结果集上，EUBO 往往过度探索无法实现的结果。
论文还得出结论：BALD 的蒙特卡洛版本 BALD-f̃ 是强而快速的基线。

**qEUBO。** @astudillo2023qeubob 把 EUBO 推广为

$$
\operatorname{qEUBO}_n(\vx_1, \dots, \vx_q) = \E_n\!\left[\max\{f(\vx_1), \dots, f(\vx_q)\}\right],
$$ {#eq-acqf-qeubo}

其中 $q$ 是一次查询中展示给人的选项数，人从中选出最好的一个；$f$ 是潜在效用（@sec-eubo 中记作 $g$）。
qEUBO 不是由若干独立查询组成的批量。
@sec-eubo-theory 已概述论文的四个定理；各定理的适用范围取决于其条件：

1. **定理 1。** 回答无噪声时，qEUBO 具有一步贝叶斯最优性，且等价于知识梯度（@sec-knowledge-gradient）。
2. **定理 2。** 在尺度为 $\tau$ 的逻辑（多项 logit）噪声下（@eq-cmp-logit；论文记作 $\lambda$），qEUBO 最大化点的一步价值至少为无噪声一步最优值减去 $\tau\, W\!\left((q - 1)/e\right)$。其中 $W$ 是 Lambert W 函数，即 $w \mapsto w e^w$ 的反函数；这一保证随 $q$ 增大放松得多快，由 @exr-acqf-lambert 计算。
3. **定理 3。** 在有限定义域上，若 $q = 2$ 且满足另外一些技术条件，qEUBO 的贝叶斯简单遗憾为 $o(1/n)$。贝叶斯简单遗憾指最优效用与推荐选项效用之差的期望，按先验取平均（@sec-regret-definitions）。
   充分条件举例：逻辑似然，且先验保证对所有 $\vx \neq \vy$ 几乎必然有 $\delta \le \lvert f(\vx) - f(\vy)\rvert \le \Delta$；或非退化的高斯过程先验，且只要 $f(\vx_1) \neq f(\vx_2)$，似然就等于常数 $a > 1/2$。
4. **定理 4。** 在满足同样假设的某些实例上，qEI 的贝叶斯简单遗憾对每个 $n$ 都高于常数 $R > 0$，即 qEI 不具有渐近一致性。

由此可得两点解读（推断）。
其一，EUBO 的一步最优性最早在 BOPE 中得到证明；qEUBO 的贡献在于将其推广到有噪声与 $q > 2$ 的情形，并给出有限定义域上的一致性。
其二，$o(1/n)$ 速率所依赖的条件（有限定义域，且效用差距有界地远离零）把问题变成了在有限多个选项中识别最优者，因此这一速率不能与 @sec-theory-rates 中连续定义域上的速率相比较。
软件方面，BoTorch 0.10.0（2024 年 2 月）加入了 qEUBO[@botorch2026changelog]，其关于偏好的教程用 `PairwiseGP` 与解析 EUBO 运行这一循环[@botorch2026pairwise]。

**幻觉信念。** 同年，@takeno2023practicalc 提出了 @sec-choosing-pairs 中的幻觉信念（HB）。它以当前赢家为对决的第一个点；再从截断后验中抽取潜在比较值的一个样本，称为**幻觉**（hallucination），用这个样本拟合一个高斯过程，并在其上应用期望改进或上置信界。
他们此前观察到，标准采集函数用于偏好高斯过程时会反复选择相似的对决：一次对决携带的信息很少，偏好模型的方差几乎不下降。
@ignatenko2025preference 后来从数据收集的极小极大视角出发，提出**剩余系统不确定性**（remaining system uncertainty），作为无需基准真值的性能度量。

## 2024 至 2026 年：噪声、精确知识梯度与摊销 {#sec-acqf-recent}

**2024 与 2025 年。** @ozaki2024multi 在多目标设定中用基于互信息的主动学习采集函数选择偏好查询（见 @sec-acqf-query-forms）。
@sinaga2024anchor（2024 年首次发布，2026 年版本收录于 ProbML 2026 的论文集轨道）提出一种风险规避的采集函数，在效用与比较的回答难度之间权衡，并证明其风险调整后的 EUBO 在相差一个加性常数的意义下仍具有一步贝叶斯最优性。
POP-BO[@xu2024principledb]与最大最小下置信界算法（MaxMinLCB）[@pasztor2024bandits]都是带遗憾界的乐观算法（@sec-theory-bt）；2025 年，偏好摊销黑箱优化（PABBO）[@zhang2025pabbob]用预训练的 Transformer 策略直接输出查询对。

**2026 年：检视默认方法。** 2026 年的方法研究大多是在检视默认流程，主要见于两篇预印本，下文在失效模式部分介绍[@wu2026knowledge; @shao2026adaptive]。
其他工作则改造经典思想。
@erarslan2025consecutive（预印本）把最大值熵搜索改造用于一种生产成本约束：每次比较都必须包含一个已经生产出来的候选。
@haltia2026elicitation（预印本）用成本感知的信息价值在直接评估与成对查询之间做选择。据作者报告，其性能接近两条单一来源轨迹的凸包；查询代价高或噪声大时，该方法回退到标准贝叶斯优化。
局部偏好贝叶斯优化[@menn2026local]（预印本）把信赖域搜索（TuRPBO）与导数引导的局部搜索（GIPBO、PrefSQP）移植到成对反馈上（@sec-hd-local）。
偏好反馈下的 Thompson 采样（PF-TS）[@lazzaro2026finiteb]抽取两个独立的后验样本，分别相对于一个共同的锚点求最大值，由此选出一对选项。

## 有记录的失效模式 {#sec-acqf-failures}

@sec-pbo-failure-modes 已预先介绍过这些失效。
本节给出各篇论文提出的机制，以及观察到每种失效时的条件。

**改编的期望改进屡屡停滞。** 四个研究组各自独立报告了同一现象。
González 等人发现 CEI 过度利用，Brochu 等人的交互式贝叶斯优化表现不佳[@gonzalez2017preferentialb]。
Fauvel 与 Chalk 发现 Brochu 等人的期望改进仅略优于随机查询，并将其归因于一种常见的病态：该采集函数反复选中相同的对决成员[@fauvel2021efficient]。
Takeno 等人发现，期望传播配合期望改进常因过度利用而停滞[@takeno2023practicalc]。
Astudillo 等人发现 qEI 往往在运行后期停滞（实验对象为 7 维 Alpine1 函数，初始数据包含与一个已知好点的比较），并在定理 4 中证明了其机制：当前最优点（后验均值最大的点）的值一旦已估计得相当精确，qEI 便不愿把它纳入查询，因而只能了解其他选项之间的相对优劣[@astudillo2023qeubob]。
这类规则会停止检验一个已充分了解的当前最优点（推断；@sec-choosing-pairs）。

**EUBO 过度利用，其流程出现病态。** 在 BOPE 不加限制的结果集上，EUBO 过度探索了无法实现的结果[@lin2022preferenceb]。
在单目标偏好贝叶斯优化中，EUBO 的查询向估计的最大值处坍缩；@sec-pbo-failure-modes 报告过这一现象，@sec-pbo-dims 在模拟中复现了它。
这一现象的出处 @wu2026knowledge（预印本）还给出了机制：在高斯过程先验与概率单位似然下，一步前瞻后验服从**扩展偏斜正态**（extended skew-normal）分布，即高斯分布的一种偏斜变体，其均值有闭式解；EUBO 只是精确知识梯度的下界，仅当概率单位噪声趋于零时才与之近似相等。
他们的测试案例是二维 Levy 函数；其摘要也承认，有一个案例表明知识梯度在某些情况下存在局限。
KappaSharp（同为预印本）报告，EUBO 的查询会产生孤立的比较对，导致 Hessian 矩阵秩亏[@shao2026adaptive]（@sec-obs-graphs）。
POP-BO 的作者还报告，在从高斯过程中采样得到的实例上，qEUBO 报告的解比他们的略好，但其累积遗憾是 POP-BO 的 2.5 倍以上[@xu2024principledb]。

**Thompson 采样随维度增大而过度探索。** DTS 在一维与二维中是最好的规则[@gonzalez2017preferentialb]，但在 34 个函数上，基于 Thompson 采样的规则表现平平，其中 KernelSelfSparring 在批量设定中较弱，原因被归于它独立地选择批内成员[@fauvel2021efficient]；在 4 维与 6 维 Hartmann 函数上，Thompson 采样过度探索[@takeno2023practicalc]。
凡是在四维及以上测试过 Thompson 采样类规则的研究，都发现它不如另一个规则（@fig-acqf-evidence-map）。
上述论文都没有单独分离出其中的机制；一种可能的解释是，后验样本的最大值往往落在后验最不确定之处，而远离所有观测的区域在定义域中所占比例随维度增大迅速上升（推断；@exr-acqf-coverage）。

**幻觉信念的优劣取决于噪声。** Takeno 等人的结果是在噪声方差 $10^{-4}$ 下得到的，而他们恰恰表明，Laplace 近似在这一区间失效最严重。
其附录 G.4 显示，幻觉信念与最大不确定挑战或二元期望改进结合时表现相对较差，作者怀疑原因是过度探索[@takeno2023practicalc]。
在逻辑噪声下，POP-BO 的作者报告幻觉信念陷入局部最优：它过于相信随机的偏好反馈，在抽取 Thompson 样本时把这些反馈当作硬约束[@xu2024principledb]。
换言之，幻觉信念的优势取决于噪声水平（推断）。

**其他代价。** MPES 必须枚举可能的回答。在 qEUBO 论文的 4 至 7 维实验中，MPES 每次迭代耗时 12.7 至 24.8 秒，qEUBO 约为 7 至 12 秒，且 MPES 的结果不如 qEUBO[@astudillo2023qeubob]。

## 采集函数的比较 {#sec-acqf-table}

@tbl-acqf-compare 汇总了主要规则。
“有利证据”与“有记录的失效”两栏中的每一项都注明了观察时的维度与噪声，取自报告该项发现的研究（各项研究的完整设定见 @tbl-acqf-studies）。
频率派遗憾界的细节见 @sec-pbo-theory。

::: {.table #tbl-acqf-compare title="偏好贝叶斯优化的主要采集规则及各项发现的观察条件。"}
| 规则（提出者，发表场所） | 理论 | 有利证据 | 有记录的失效 |
|---|---|---|---|
| PE[@gonzalez2017preferentialb] | 无 | 1 维与 2 维，每维 33 个网格点；未说明噪声 | 维度增大时变差 |
| CEI[@gonzalez2017preferentialb] | 无 | 只在 1 维 Forrester 函数上运行 | 过度利用；计算代价过高 |
| DTS[@gonzalez2017preferentialb] | 单目标下没有遗憾界；多目标版本渐近一致[@astudillo2025preferential] | 在 1 维与 2 维网格上最好；未说明噪声 | 从 4 维起过度探索（4 维与 6 维 Hartmann 函数，噪声方差 $10^{-4}$）[@takeno2023practicalc]；在 34 个函数上的九个规则中排第四（概率单位噪声，方差为 1）[@fauvel2021efficient] |
| KernelSelfSparring[@sui2017multi] | 独立臂版本渐近无遗憾；核版本只是猜想 | 没有专门的证据 | 批量设定中落后，因为批内成员独立选择[@fauvel2021efficient] |
| 改编的期望改进与 qEI[@brochu2007active; @siivola2021preferential] | qEI 不具渐近一致性（qEUBO 论文定理 4） | 批量实验中与批量 Thompson 采样没有明显差异（至多 4 维，效用噪声标准差 0.05） | 停滞、过度利用、选出相同的对决成员（四个研究组；1 至 7 维，噪声从 $10^{-4}$ 至逻辑噪声） |
| MUC[@fauvel2021efficient] | 无 | 在 34 个函数上按 Borda 排名并列第一（概率单位噪声，方差为 1） | 与 HB 结合时相对较差；在期望传播下于 Bukin 与 Ackley 函数上停滞 |
| 对决上置信界、对决 Thompson 采样、EIIG[@benavoli2021preferentialb] | 无 | 对决上置信界在 34 个函数上并列第一（概率单位噪声，方差为 1）[@fauvel2021efficient] | 在同一比较中 EIIG 排第七 |
| MPES[@nguyen2021top] | 无 | 在 1 至 3 维与 SUSHI 上始终最好；未说明噪声 | 必须枚举回答；在 4 至 7 维中不如 qEUBO 且最慢（逻辑噪声）[@astudillo2023qeubob] |
| BALD[@houlsby2011bayesian; @lin2022preferenceb; @botorch2026changelog] | 无 | 在 BOPE 中有竞争力但略差（10% 的错误选择） | 没有专门的记录 |
| EUBO、qEUBO[@lin2022preferenceb; @astudillo2023qeubob] | 无噪声时一步贝叶斯最优，等价于知识梯度；逻辑噪声下有加性常数保证；在有限定义域上 $q = 2$ 时贝叶斯简单遗憾为 $o(1/n)$ | 在 4 至 7 维、中等逻辑噪声、150 次查询下，除一个问题外均最好 | 查询向估计的最大值处坍缩（2 维 Levy 函数，概率单位噪声；预印本）[@wu2026knowledge]；Hessian 矩阵秩亏（预印本）[@shao2026adaptive]；累积遗憾更高（6 维 Ackley 函数与高斯过程样本，逻辑噪声）[@xu2024principledb] |
| HB[@takeno2023practicalc] | 无（列为未来工作） | 在至多 6 维的 12 个函数上、噪声方差 $10^{-4}$ 时总体最好 | 在逻辑噪声下陷入局部最优[@xu2024principledb]；与 MUC 结合时过度探索 |
| 精确知识梯度[@wu2026knowledge] | EUBO 是它的下界 | 一个 2 维 Levy 函数的案例研究（概率单位噪声） | 其摘要承认它在某些情况下有局限 |
| POP-BO、MaxMinLCB、MR-LPF、PF-TS[@xu2024principledb; @pasztor2024bandits; @kayal2025bayesian; @lazzaro2026finiteb] | 见 @sec-theory-rates | 全部为低维实验（1 至 6 维，逻辑噪声） | PF-TS 的论文报告 MR-LPF 在 $T = 300$ 时累积遗憾更高（1 维 Ackley 函数，3 维催化剂问题）[@lazzaro2026finiteb] |
| PABBO[@zhang2025pabbob] | 无 | 在多数任务上排第一或第二（1 维、2 维、6 维、HPO-B、Candy、Sushi；无噪声） | 维度固定；默认评测无噪声；在 6 维 Hartmann 函数上较弱 |
:::

表中的条件用图表示更便于比较。
@fig-acqf-evidence-map 按噪声模型与所测试的输入维度排布每项比较研究；选择一个规则，即可看到哪些研究涉及它、结果如何。

```{figure}
//| figure: acqf-evidence-map
//| label: fig-acqf-evidence-map
//| fig-cap: "采集规则各项发现的观察条件。每行代表一项研究，按噪声模型分组，横向覆盖该研究测试的输入维度（对数刻度）；虚线段表示该研究还测试了更低的维度，但未逐一列出；右侧一列的方块表示研究未给出维度的任务或真实数据任务。选择一个规则后，涉及该规则的研究按发现对它有利、好坏参半或不利着色，并列出各项发现及其条件。发现与条件取自 @tbl-acqf-compare 与 @tbl-acqf-studies 所引的研究；判定所用的颜色是本章的归纳。"
```

可以留意以下几点：

- **Thompson 采样**（默认视图）：两项有利发现都在 1 至 3 维，即使在这一范围内，它也不如 MPES。
- **幻觉信念**：在最上面一行（回答近乎无噪声）有利，在逻辑噪声一行不利。
- **EUBO 与 qEUBO**：在 4 至 7 维、逻辑噪声下有利；2024 至 2026 年测量累积遗憾或专门考察坍缩的研究则给出好坏参半或不利的结果。
- **空白区域**：20 维以上，只有局部偏好贝叶斯优化的预印本比较过不同规则。

## 查询形式 {#sec-acqf-query-forms}

查询不必是一对选项。
@sec-query-design 介绍了查询的各种形式，并报告了主要形式的证据。
对于更大的选项集合（@sec-query-batch），qEUBO 的实验发现 $q = 4$ 明显优于 $q = 2$，Siivola 等人则发现更大的批量只带来微小增益；两项研究所用的采集函数与噪声不同[@astudillo2023qeubob; @siivola2021preferential]。
对于画廊与投影（@sec-gallery-projective），在 2 至 20 维中，每个投影变体都胜过每个成对变体[@mikkola2020projective]；序列画廊的平面搜索在 5 至 20 维中胜过线搜索，6 名被试平均经过 5.36 次迭代即感到满意[@koyama2020sequential]。
只用成对比较时，随着参数增多，40 次比较所能缩小的差距比例迅速下降，@fig-pbo-dims 显示了下降的速度。
@benavoli2023choice 让人从一个集合中选出若干个彼此不可比的选项。
本节补充其余形式，并讨论综合各项证据能得出什么结论。

**共同主动反馈、序数标签与机器人专用形式。** CoSpar[@tucker2020preference]在 SelfSparring 的后验采样中加入共同主动反馈：用户既比较各次试验，也提出改进建议。
LineCoSpar[@tucker2020human]把计算限制在一条随机直线上（@sec-query-linecospar），为 6 名身体健全的被试调节了 6 个步态参数。
ROIAL[@li2021roial]在感兴趣区域内把序数标签与偏好结合起来，该区域的设定旨在保证安全与舒适。
@sec-cs-exoskeleton 通过一个案例研究介绍这一方向。

**对假设结果的偏好与改进请求。** @astudillo2020multib 让决策者比较属性向量。BOPE 则交替进行两个阶段：一个阶段由人比较从结果模型中采样的结果向量，这些结果可能只是假设的；另一个是实验阶段[@botorch2026bope]。
@ozaki2024multi 增加了**改进请求**（improvement request）：决策者指出希望所展示结果的哪个目标得到改进；效用取权重不确定的 Chebyshev 标量化，即各目标的加权最坏情况组合。

**弱偏好、有效性标签、数值与语言。** @byk2019asking 的“差不多”（About Equal）回答、C-GLISp 的“更好”“更差”“相近”、有效性标签与崩溃报告，都是对观测模型的扩展（@sec-obs-extensions）。
@xu2020zeroth 同时允许直接查询与对决（COMP-GP-UCB）。
对于有限个选项，@wang2025fusing 证明，高效算法在每个选项上只需付出两种遗憾中的较小者：一种来自奖励反馈，一种来自对决反馈。
Social BO[@adachi2025bayesian]（预印本）证明，在温和的理性公理下，仅凭带噪声的群体反馈无法达成不受社会影响的共识；该方法把廉价的公开投票与昂贵的私下投票结合起来。
自然语言也成为标签的来源。PEBOL[@austin2024bayesian]以自然语言推理作为独立条目上的似然；语言在回路优化（LILO）[@kobalczyk2026lilo]让语言模型把自由文本反馈转换为成对标签，供配合 qEUBO 的 `PairwiseGP` 使用，实验采用模拟决策者，没有以人为对象的研究（@sec-llm-in-loop）。
语言模型生成的标签彼此相关且带有偏差，并非独立的概率单位噪声（推断）。

**查询形式的证据能得出什么结论。** 证据指向两个方向。
能让人的每次操作携带更多信息的形式（投影、平面、$q = 4$），在提出它们的论文中都胜过成对比较；而唯一一项直接比较批次赢家与完整排序的研究发现，两者差别不大。
要求连续回答的形式（滑块、投影、共同主动建议）会引入其他噪声，即运动与知觉精度以及认知负荷带来的噪声；所引论文都只用一个笼统的噪声项来处理（推断）。
截至 2026 年 9 月，我们没有找到在相同采集函数与预算下比较成对比较、批次赢家、完整排序与滑块的受控人类研究；除成对比较与多选项集合外，也没有找到为其他形式专门推导的采集函数，滑块、平面与投影查询使用的都是改编的期望改进或随机子空间。
在这样的研究出现之前，查询形式最好从人的角度选择，即人能可靠而迅速地回答什么（@sec-query-form-evidence、@sec-hci-feedback-forms）；更丰富的形式应当在同一系统内与成对比较对照检验（推断）。

## 问题扩展 {#sec-acqf-extensions}

偏好贝叶斯优化的扩展方向与普通贝叶斯优化相同：多个目标、约束、情境、高维、混合输入、多保真度与停止。
@tbl-acqf-extensions 概括了各个方向，表后各段说明其中的要点。

::: {.table #tbl-acqf-extensions title="偏好贝叶斯优化的扩展、各自的证据与主要空白。"}
| 扩展 | 代表性工作（按时间顺序） | 证据类型 | 主要空白 |
|---|---|---|---|
| 多个目标与对结果的偏好 | Astudillo 与 Frazier 2020[@astudillo2020multib]；BOPE 2022[@lin2022preferenceb]；Ozaki 等 2024[@ozaki2024multi]；PUB-MOBO 2025[@ip2025user]；Astudillo 等 2025[@astudillo2025preferential]；Wang 等 2025[@wang2025bayesian]；Huber 等 2025[@huber2025bayesian]；Active-MoSH 2026[@chen2026interactive] | 模拟的决策者；工程基准 | 以人为对象的研究很少 |
| 约束 | StageOpt 2018[@sui2018stagewise]；Benavoli 等 2021[@benavoli2021preferentialb]；C-GLISp 2022[@zhu2022c]；Kwon 等 2022[@kwon2022physically]；Iwai 等 2025[@iwai2025constrainedb] | 控制器校准；11 名设计者 | 2025 年之后没有新的约束论文 |
| 安全 | StageOpt 2018[@sui2018stagewise]；ROIAL 2021[@li2021roial]；Cosner 等 2022[@cosner2022safety]；CrashPBO 2026[@menn2026preferential] | 脊髓刺激；一台四足机器人；三个机器人平台 | 未指出 |
| 情境 | Khan 等 2025[@khan2025efficient]；Wang 等 2025[@wang2025personalized]；Coutinho 等 2025、2026[@coutinho2025accelerated; @coutinho2026efficient] | 模拟建筑；一份负迁移的报告 | 多任务高斯过程尚未结合偏好似然检验 |
| 跨用户迁移 | Granley 等 2023[@granley2023human]；Meta-PO 2025[@li2025efficient]；PABBO 2025[@zhang2025pabbob] | Meta-PO 有 36 人 | 层级效用先验很少 |
| 高维 | 序列线搜索 2017[@koyama2017sequentialb]；投影偏好贝叶斯优化 2020[@mikkola2020projective]；LineCoSpar 2020[@tucker2020human]；序列画廊 2020[@koyama2020sequential]；qEUBO 2023[@astudillo2023qeubob]；局部方法 2026[@menn2026local]；GimmBO 2026[@liu2026gimmbo] | 至多 102 维的模拟比较 | 维度缩放先验与稀疏先验尚未检验 |
| 混合输入与类别输入 | 分段仿射代理模型 2025[@zhu2025global] | 基准 | 没有采用类别核的高斯过程偏好方法 |
| 多保真度 | Theiner 等 2026[@theiner2026efficient]；引出增强的贝叶斯优化 2026[@haltia2026elicitation] | 预印本或会议论文 | 2025 年年中之前没有 |
| 停止 | Bıyık 等 2019，定理 3[@byk2019asking]；序列画廊的“满意”按钮[@koyama2020sequential]；Ignatenko 等 2025[@ignatenko2025preference] | 参数模型 | 没有针对成对高斯过程的停止规则 |
:::

**多个目标。** @astudillo2025preferential 考虑每个目标都只能通过偏好观测的情形，提出了**对决标量化 Thompson 采样**（dueling scalarized Thompson sampling，DSTS）：从后验中采样，施加随机的 Chebyshev 标量化，再运行对决 Thompson 采样。
他们证明了该方法渐近一致，称这是偏好贝叶斯优化中对决 Thompson 采样的第一个收敛保证，同时指出，即使在单目标偏好贝叶斯优化中，DTS 的遗憾界也仍然未知；DSTS 在四个合成函数以及模拟的外骨骼与自动驾驶任务上表现最好。
@abdolshah2019multi（NeurIPS 2019）所说的“偏好”是各目标之间的重要性次序，而不是对设计的成对反馈；表中其他工作沿袭了 @sec-multi-objective 的方向。

**约束与安全。** StageOpt[@sui2018stagewise]在未知安全约束下优化效用，把扩展安全区域与最大化效用分开进行。
其约束满足与收敛保证是针对数值观测给出的。附录中的一个变体从偏好反馈中学习效用，安全函数仍接收数值测量；该变体没有自己的收敛定理，已在临床上用于脊髓刺激。
约束偏好贝叶斯优化[@iwai2025constrainedb]提出了 EUBOC：仿照约束期望改进[@gardner2014bayesian]，用约束得到满足的概率（以高斯过程建模）为 EUBO 加权。该方法在一项由 11 名专业设计者参与的横幅广告研究中接受了评估，以预测点击率作为约束。
处理约束的方式有两种：从人提供的标签中学习可行性（C-GLISp、Benavoli 等人的有效性标签、崩溃反馈），或单独测量约束（约束偏好贝叶斯优化中的点击率、StageOpt 的安全信号）。
哪种方式合适，取决于约束能否无需人参与而直接观测（推断）。

**情境与迁移。** 情境可以通过由专家知识学得的离线效用引入[@khan2025efficient]，也可以通过情境变量引入，例如控制器调节中的室外温度[@wang2025personalized]（预印本）。
迁移的实现途径有摊销（PABBO）、存储早期用户的模型（Meta-PO）与在群体上训练的编码器（Granley 等人），而没有采用跨用户的多任务高斯过程（推断）；群体先验能带来多少收益，见 @sec-hci-population。

**高维、混合输入、多保真度与停止。** 2017 至 2020 年的高维方法都把每次查询限制在经过当前最优点的低维子空间中，从而把 $d$ 维的采集函数优化化为一维或二维优化，并使人的每次操作携带不止 1 比特的信息（推断）；各方法达到的维度见 @sec-hd-pbo-dims，@sec-hd-local 考察了 2026 年的局部方法，以及干扰其与 qEUBO 比较的长度尺度限制。
@zhu2025global 的分段仿射代理模型用混合整数线性规划处理已知的线性约束，以及数值与类别混合的变量；这是我们找到的唯一面向混合输入的偏好方法。序列画廊则把无法处理离散参数（如布局、字体或滤镜类型）列为局限之一。
仅有的多保真度偏好方法是 @theiner2026efficient 的方法与引出增强的贝叶斯优化[@haltia2026elicitation]，两者都来自 2026 年。
停止方面，唯一明确的最优规则假设了参数化的奖励模型[@byk2019asking]；序列画廊则在用户按下“满意”按钮时停止。标量贝叶斯优化在停止问题上有哪些认识、偏好停止规则需要具备什么，见 @sec-hd-stopping。

## 相互竞争的“首创”声明 {#sec-acqf-firsts}

偏好贝叶斯优化处于对决赌博机、基于偏好的强化学习、控制（GLISp 一脉）与人机交互的交汇处，一个研究社区中的“首创”声明常常忽视另一个社区更早的工作。
以下各项声明都只在其确切的设定内成立（推断）。

- **约束偏好贝叶斯优化**[@iwai2025constrainedb]声称首次引入了不等式约束。但在此之前，C-GLISp[@zhu2022c]已于 2022 年处理了未知约束，StageOpt[@sui2018stagewise]的一个变体也已于 2018 年在数值测量的安全约束下从偏好反馈中学习效用；该文也没有引用 Benavoli 等人 2021 年的有效性标签[@benavoli2021preferentialb]。这一声明对高斯过程代理模型上的不等式约束成立。
- **LineSpar**[@cheng2020preference]是一篇研讨会论文，自称是第一个高维的基于偏好的贝叶斯优化，但它晚于序列线搜索（2017 年），与投影偏好贝叶斯优化（ICML 2020）同期。
- **DSTS**[@astudillo2025preferential]声称是第一个多目标偏好贝叶斯优化框架，这与一篇关于选择函数的预印本[@benavoli2021choice]有所重叠；这一声明对只能通过偏好观测的潜在目标成立。
- **POP-BO**[@xu2024principledb]称现有方法缺乏累积遗憾保证或全局收敛保证，并附限定语“连续输入空间”（continuous input space），且没有引用 @kirschner2021bias。
- **MPES**[@nguyen2021top]自称是第一个面向偏好观测的信息论采集函数（arXiv 首个版本发布于 2020 年 12 月），而 EIIG 规则[@benavoli2021preferentialb]（2020 年 8 月）也使用了对决信息增益；两者都源自 BALD[@houlsby2011bayesian]（推断）。
- **EUBO 的一步贝叶斯最优性**最早在 BOPE 中得到证明[@lin2022preferenceb]；qEUBO 论文的声明适用于将这一结果推广到逻辑噪声与 $q > 2$ 的部分。

## 实证比较的现状 {#sec-acqf-empirical}

@tbl-acqf-studies 列出了主要的比较研究及其设定。
表中 qTS 为批量 Thompson 采样，qNEI 为带噪声批量期望改进，HPO-B 是由超参数优化任务构建的基准，Sobol 序列是一种拟随机的空间填充设计。

::: {.table #tbl-acqf-studies title="偏好采集规则的主要实证比较及其设定。"}
| 研究 | 比较对象 | 维度 | 预算与重复次数 | 噪声模型 | 主要结论 |
|---|---|---|---|---|---|
| González 等 2017[@gonzalez2017preferentialb] | PE、CEI、DTS、随机、交互式贝叶斯优化、Sparring | 1、2 | 5 + 200 次对决；重复 20 次 | 未说明 | DTS 最好 |
| Mikkola 等 2020[@mikkola2020projective] | 5 个投影规则，对比成对的随机变体与 DTS 变体 | 2、6、10、20 | 100 次查询；25 次初始化 | 较小的高斯噪声 | 每个投影变体都胜过每个成对变体 |
| Siivola 等 2021[@siivola2021preferential] | 批量期望改进、批量 Thompson 采样、随机；三种推断方法 | 6 个函数；$d \le 4$ 的真实数据 | 批量大小 2 至 6；重复 10 次 | 效用噪声标准差 0.05 | 采集函数之间的差异大于反馈类型之间的差异；在真实数据上只比基线略好 |
| Nguyen 等 2021[@nguyen2021top] | MPES、期望改进、DTS | 1 至 3；CIFAR-10 嵌入；SUSHI | 未说明 | 未说明 | MPES 始终最好 |
| Fauvel 与 Chalk 2021[@fauvel2021efficient] | 9 个规则 | 34 个函数 | 80 次迭代；重复 40 次 | 概率单位噪声，归一化后方差为 1 | MUC 与对决上置信界并列第一；Brochu 的期望改进第八；随机第九 |
| Lin 等 2022[@lin2022preferenceb] | EUBO-ζ、EUBO-f̃、BALD-f̃、随机及其他 | 多结果问题 | 分 3 个阶段共 75 次比较；重复 30 次 | 10% 的错误选择 | EUBO 的各个变体最好 |
| Astudillo 等 2023[@astudillo2023qeubob] | qEUBO、MPES、qTS、qEI、qNEI、随机 | 4 至 7 | 初始 $4d$ 次 + 150 次查询；重复 50 或 100 次 | 逻辑噪声，按最优的 1% 点对上 10%、20%、30% 的错误率校准 | $q = 2$ 时，除汽车驾驶室设计问题外，qEUBO 在所有问题上最好 |
| Takeno 等 2023[@takeno2023practicalc] | HB、Laplace 近似与期望传播配合期望改进、MUC、偏斜高斯过程采样规则 | 至多 6（12 个函数） | 初始 $3d$ 次；重复 10 次 | 噪声方差 $10^{-4}$ | HB 总体最好；未包括 qEUBO |
| Xu 等 2024，POP-BO[@xu2024principledb] | DTS、HB、qEUBO | 高斯过程样本；6 维 Ackley 函数 | 未说明 | 逻辑噪声 | qEUBO 报告的解略好，累积遗憾是 POP-BO 的 2.5 倍以上 |
| Zhang 等 2025，PABBO[@zhang2025pabbob] | qEUBO、qEI、qNEI、qTS、MPES、随机 | 1、2、6；HPO-B；Candy；Sushi | 重复 30 次 | 无噪声 | PABBO 第一或第二；随机常常胜过某些高斯过程基线 |
| Lazzaro 等 2026，PF-TS[@lazzaro2026finiteb] | MR-LPF、POP-BO、MaxMinLCB | 1 维 Ackley 函数；3 维催化剂问题 | $T = 300$；重复 30 次 | 逻辑噪声 | PF-TS 的累积遗憾显著低于 MR-LPF 与 POP-BO |
| Menn 等 2026，局部方法（预印本）[@menn2026local] | 局部方法、qEUBO、HB 配合期望改进、GLISp、Sobol 序列 | 至多 102 | 约 $10d$ 次比较，此前先做 $5d$ 次随机评估用于策略搜索 | 高斯噪声，取值范围的 10% | 局部方法在陡峭的最优点上更好 |
:::

qEUBO 实验的噪声校准取自其附录 C.2 与代码[@astudillo2023noise]。

**这些比较不能简单合并。** 各项研究的度量不同（@sec-sw-regret 列出了遗憾的五种定义），噪声模型不同（从近乎无噪声的概率单位噪声，到 10% 的翻转回答与中等的 Gumbel 噪声，@sec-sw-noise-models），后验推断方法也不同：Laplace 近似、期望传播、变分推断或偏斜高斯过程采样。
排名会随这些选择而颠倒，POP-BO 与 qEUBO 的比较即是一例（推断）。
Fauvel 与 Chalk 在 34 个函数上的 Borda 分析是覆盖面最广的单项比较，但它早于 qEUBO 与幻觉信念，全程使用期望传播，且只运行了 80 次迭代。
Takeno 等人的比较覆盖面次之，但其中的对决几乎无噪声，也没有纳入 qEUBO。
没有一项比较能同时做到覆盖面广、噪声符合实际、包含 2023 年及以后的采集函数。

**随机查询有时并不落后。** PABBO 的作者写道，随机策略常常胜过某些高斯过程基线[@zhang2025pabbob]。
在信噪比低的真实数据上，Siivola 等人发现所有方法都只是勉强胜过基线。他们写道：偏好观测本质上不如直接观测信息丰富，更大的批量并不能缓解这一点；对于噪声很大的数据，采用大批量的随机搜索可能是好的选择[@siivola2021preferential]。
BoTorch 的 BOPE 教程在单次重复中输出的候选效用为：EUBO-ζ $-0.473$，随机偏好探索 $-0.216$，真实效用 $-0.101$，随机实验 $-1.380$；教程也指出，单次重复中 EUBO-ζ 并不一定胜出[@botorch2026bope]。
这三方面的证据指向同一方向：预算小、噪声接近实际时，精心设计的采集函数与随机设计之间的差距可能很小，单次运行的结果说明力也很有限（推断）。

**来自人的证据。** 本章涉及的人类研究规模都很小，且没有一项是为比较采集函数而设计的：投影偏好贝叶斯优化的一名材料科学用户、序列画廊的 6 名被试、LineCoSpar 的 6 名、约束偏好贝叶斯优化的 11 名设计者与 Meta-PO 的 36 名。
**截至 2026 年 9 月，我们没有找到任何在相同界面与预算下把人随机分配到不同采集函数（例如 qEUBO、DTS 与 MUC）的研究。**
能对此作出定论的实验见 @sec-open-decisive。
在此之前，实践者应把采集函数在真人身上的排名视为未知，并在每次会话中保留一部分随机查询作为对照；鉴于上述证据，这样做代价很小（推断）。
偏好贝叶斯优化也没有一个在度量、噪声模型与预算上取得共识的基准：各篇论文按各自需要选用 Forrester、six-hump camel、Hartmann、Ackley、Sushi 与 Candy（@sec-sw-datasets）。

## 已定、有争议与缺失 {#sec-acqf-status}

::: {.frontier title="已定、有争议与缺失"}
**已定。** 在 qEUBO 论文构造的实例上，qEI 不具有渐近一致性[@astudillo2023qeubob]。
四个独立研究组的实验都观察到改编的期望改进出现停滞。
EUBO 的一步贝叶斯最优性在无噪声时成立，在逻辑噪声下有加性常数保证，最早在 BOPE 中得到证明[@lin2022preferenceb]。
投影与平面这类让每次操作携带更多信息的查询形式，在提出它们的论文所做的模拟中胜过成对比较[@mikkola2020projective; @koyama2020sequential]。

**有争议。** 每次查询展示更多选项的价值：qEUBO 的实验发现 $q = 4$ 明显优于 $q = 2$，Siivola 等人只发现微小增益，而两者的采集函数与噪声并不相同。
幻觉信念的优势：回答近乎无噪声时它最好，但据 POP-BO 的作者报告，在逻辑噪声下它会陷入局部最优。
EUBO 的坍缩与病态：证据来自 2026 年的两篇预印本，尚未经过同行评审，也未在人类数据上检验。
Thompson 采样类规则的排名：它随维度与设定而颠倒。

**缺失。** 在共享基准上、噪声与预算一致的大范围采集函数比较。
把人随机分配到不同采集函数的实验。
为滑块、平面与投影查询推导的采集函数。
适用于成对高斯过程模型且带有保证的停止规则。
DTS 与 HB 的遗憾界，以及有噪声时 qEUBO 在连续定义域上的保证（@sec-pbo-theory）。

**若须立即做出选择。** 证据支持在比较中纳入随机查询与简单基线，并报告维度、噪声与度量。
配合 `PairwiseGP` 的 qEUBO 是维护最完善的默认选择，但这一推荐的依据是软件生态，且是在偏好贝叶斯优化框架内部得出的，并非来自与更简单方法的直接比较（推断）。
:::

## 习题 {#sec-acqf-exercises}

::: {.exercise #exr-acqf-lambert}
qEUBO 论文的定理 2 给出，尺度为 $\tau$ 的逻辑噪声造成的损失以 $\tau\, W((q - 1)/e)$ 为界，其中 $W$ 是 $w \mapsto w e^w$ 的反函数。
通过求解 $w e^w = (q - 1)/e$，计算 $q = 2$ 与 $q = 4$ 时的 $W((q - 1)/e)$，并说明回答带噪声时，这个界对每次查询展示更多选项有何含义。

::: {.solution}
$q = 2$ 时，求解 $w e^w = 1/e \approx 0.368$：取 $w = 0.28$，得 $0.28 \cdot e^{0.28} \approx 0.37$，故 $W(1/e) \approx 0.28$。
$q = 4$ 时，求解 $w e^w = 3/e \approx 1.10$：取 $w = 0.60$，得 $0.60 \cdot 1.82 \approx 1.09$，故 $W(3/e) \approx 0.60$。
以噪声尺度 $\tau$ 为单位，相对于无噪声一步最优值的保证损失从 $q = 2$ 到 $q = 4$ 大约翻了一倍；而从四个选项中得到的无噪声回答，其价值至少不低于从两个选项中得到的回答。
这个界本身不能说明更多选项是否有益，只说明保证放松得很慢；这与 qEUBO 的实验在中等噪声下发现 $q = 4$ 优于 $q = 2$ 相一致。
:::
:::

::: {.exercise #exr-acqf-coverage}
下面用一种粗略的方法说明 Thompson 采样为何随维度增大而过度探索。
设单位立方体 $[0, 1]^d$ 中有 $n = 20$ 个观测点，后验在每个点距离 $r = 0.1$ 以内有把握，在其他地方不确定。
利用 $d$ 维球的体积 $\pi^{d/2} r^d / \Gamma(d/2 + 1)$，忽略重叠与边界，估计 $d = 1$ 和 $d = 6$ 时有把握区域占立方体的比例。
后验样本的最大值往往落在哪里？

::: {.solution}
$d = 1$ 时，“球”是长度为 $2r = 0.2$ 的区间，20 个点可以覆盖整个区间（估计值 $20 \times 0.2 = 4$ 超过 1；由于重叠，真实覆盖比例至多为 1）。
$d = 6$ 时，球的体积为 $\pi^3 r^6 / 3! \approx 31.0 \times 10^{-6} / 6 \approx 5.2 \times 10^{-6}$，20 个球约覆盖立方体的 $10^{-4}$。
六维立方体几乎处处不确定，后验样本有很多机会在从未探查过的某处取到高值，其最大值也往往落在那里。
这种行为按其构造就是探索；在六维中，它很少回到最好的区域进一步细化。
以上只是我们对机制的示意，并非所引论文的结果；那些论文报告了过度探索，但没有分离出其原因。
:::
:::

::: {.exercise #exr-acqf-pool}
根据 @tbl-acqf-studies，找出一项同时比较 qEUBO 与幻觉信念的研究。
它采用什么噪声模型、在哪些维度上进行、使用什么度量？
要判定这两个规则在人类噪声水平下孰优孰劣，一项比较需要具备哪些条件？

::: {.solution}
只有 POP-BO 的论文[@xu2024principledb]同时包含两者（另有 DTS），采用逻辑噪声，实验对象为高斯过程样本与 6 维 Ackley 函数，预算未说明。
该文报告幻觉信念陷入局部最优；qEUBO 报告的解比 POP-BO 的略好，但累积遗憾是 POP-BO 的 2.5 倍以上。
幻觉信念表现最好的 Takeno 等人的研究没有纳入 qEUBO，且噪声方差为 $10^{-4}$。
能下定论的比较应当固定一个按人的比较校准的噪声模型，在相同的代理模型、推断方法、预算与维度下运行两个规则，同时报告简单遗憾与累积遗憾，纳入随机查询，最好还在真人身上重复进行。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @astudillo2023qeubob 是 qEUBO 的基本文献；阅读其四个定理时应连同条件一起读，并参阅说明噪声校准方法的附录 C.2。
- @fauvel2021efficient 是覆盖面最广的规则比较（34 个函数，九个规则），对采集中认知不确定性与偶然不确定性之别的论述也最为清晰。
- @takeno2023practicalc 与 @xu2024principledb 对幻觉信念的评价不一；对照阅读，可以看出噪声水平在多大程度上决定结果。
- @wu2026knowledge 推导了概率单位似然下的精确知识梯度；要理解 EUBO 与精确知识梯度的等价关系为何在有噪声时不再成立，这是最好的读物。
- 关于比成对比较更丰富的查询，@mikkola2020projective 与 @koyama2020sequential 是两篇基本文献。
