---
status: done
synopsis: "从 2005 年的基线模型到 2026 年 9 月：该领域如何得名，工具与推断方法如何定型，决策论转向，以及理论迎头赶上、默认流程受到审视的那几年。交互式时间线标出每个里程碑所属的泳道与阶段。"
---

# 偏好贝叶斯优化的十年 {#sec-pbo-history}

@sec-part-preferences 按通常的讲法介绍了偏好贝叶斯优化：从比较中学习的高斯过程效用（@sec-gp-preference）、选择下一对选项的规则（@sec-pbo）、查询可以采取的形式（@sec-query-forms），以及从对决中学习的赌博机视角（@sec-dueling-bandits）。
这种讲法只是一张快照。
其中各个部分分别在特定的时间、由特定的社区为回答特定的问题而提出，有几部分后来受到了质疑。

本书这一部分报告 2017 年以来的研究确立了什么、有哪些争议、留下了哪些空白。
本章是这一部分的总览。
下面按时间顺序叙述这段历史，从该领域得名之前已有的模型一直讲到 2026 年的预印本，以便后续各章分别深入追踪一条线索：观测模型（@sec-observation-models）、采集函数（@sec-acquisition-frontier）、理论（@sec-pbo-theory）、高维问题（@sec-high-dimensions）以及软件与评测（@sec-software-evaluation）。

这段历史的走向出人意料。
2026 年大多数人运行的流程，即高斯过程先验加概率单位链接与 Laplace 近似，是 2005 年的模型。
十年间变化的，与其说是这套机制，不如说是研究者针对它提出的问题。

```{figure}
//| figure: hist-timeline
//| label: fig-hist-timeline
//| fig-cap: "偏好贝叶斯优化从 2005 年至 2026 年 9 月的历程，分为六条泳道。实心圆为经同行评审的论文（另有一个研讨会），空心圆为预印本，方块为软件发布。阴影带表示所选事件所处的阶段。2005 至 2016 年压缩显示；同一年内各点错开排列只为便于辨认，不表示日期。每个时期标出的核心问题是本章的推断，并非任何来源的论断。发布日期取自 BoTorch 与 Ax 的更新日志和软件包页面[@botorch2026changelog; @ax2026changelog; @ax2022platform; @optuna2026dashboard]；各篇论文在正文讨论之处引用。"
```

图中只列出本章讨论的里程碑，外加两个 BoTorch 版本（0.10.0 与 0.18），并未收录全部论文。
可以留意以下几点：

- **理论的来源。** 2021 年以前，理论泳道中的每一项保证都来自对决赌博机社区；面向连续定义域的核化结果始于 2021 年，2024 年起大量出现。
- **人何时进入研究。** 2022 年以前，“人与应用”泳道中只有两篇外骨骼论文；此后这条泳道逐渐充实，只有 2024 年空缺。
- **哪些尚未经过同行评审。** 空心圆全部位于 2026 年：对默认流程最新的批评都是预印本。
- **重演这十年。** 打开“隐藏之后的年份”，选中 2005 年的事件，然后反复按“下一个”，即可看到该领域逐步积累的过程。

## 2017 年以前 {#sec-history-before}

在该领域得名之前，三块基石已经存在，分别出自不同的社区。

第一块是模型。
@chu2005preference 在潜在效用上放置高斯过程先验（潜在效用是为人对每个选项的喜爱程度打分的函数），并通过**概率单位**（probit）似然将其与比较联系起来：$\vx$ 优于 $\vx'$ 的概率，等于标准正态分布函数 $\Phi$ 在缩放后效用差处的取值（@sec-thurstone）。
这一似然不是高斯形式，后验因而没有闭式解；他们改用 **Laplace 近似**（Laplace approximation），即以后验峰值为中心的高斯分布来代替后验（@sec-laplace）。
这就是 @sec-gp-preference 的模型，二十年后它仍是默认模型。

第二块是交互式系统。
@brochu2007active 采用主动偏好学习（由系统自行决定下一步展示哪些选项），让人从候选画廊中挑选，以此设计计算机图形学所需的材质。其流程正是 @sec-pbo-loop 的循环：展示选项，记录选择，更新模型，再决定下一步展示什么。

第三块是来自信息检索的问题表述。
@yue2009interactively 将检索系统（例如向用户学习的搜索引擎）的交互式优化表述为**对决赌博机**（dueling bandit）问题：学习者反复挑选两个选项，只能观察到哪一个获胜（@sec-dueling-bandits-intro）。
这一表述将在线学习的分析工具，尤其是遗憾界，引入了从比较中学习的研究。

三者各自独立发展：高斯过程偏好学习属于机器学习，画廊属于图形学与交互设计，对决赌博机属于在线学习。
此后十年的大部分历程，可以理解为这些社区缓慢而不完全的交汇。

## 2017 至 2019 年：如何提问 {#sec-history-2017}

**得名。** @gonzalez2017preferentialb 在**对决空间**（dueling space）上定义了这一问题（对决空间即所有输入对构成的集合），将其命名为“偏好贝叶斯优化”（preferential Bayesian optimization），并提出三个采集函数：纯探索、Copeland 期望改进与对决 Thompson 采样（@sec-dueling-formulation）。
这篇论文没有给出收敛理论，即没有说明随着查询增多，推荐选项以多快的速度接近最优选项。

**赌博机一侧的保证。** 对于密切相关的问题，对决赌博机社区早已有形式化结果。
这些结果所控制的量是**遗憾**（regret）：由于展示的是算法所选的选项而非最优选项，在全部查询上累计损失的效用（@sec-regret）。
SelfSparring 是一种让多个选项同时对决的方法，其渐近收敛性已得到证明[@sui2017multi]；@kumagai2017regret 在该领域得名的同一年，给出了连续空间上对决赌博机的遗憾界。
分阶段安全贝叶斯优化 StageOpt 处理未知的安全约束，其定理针对数值观测；它还有一个偏好变体，该变体本身没有收敛定理，已应用于脊髓刺激[@sui2018stagewise]（@sec-acqf-extensions）。

::: {.note title="常见说法核查"}
有时会读到这样的说法：在 2024 年的核化遗憾界出现之前，从比较中学习没有任何形式化保证。
这一说法有一点是对的：González 等人的高斯过程表述没有给出收敛速率。2018 年的一篇对决赌博机综述指出了这一点，称之为“一种没有收敛速率理论保证的纯贝叶斯优化方法”（a pure Bayesian optimization approach without theoretical guarantees on convergence rate）[@sui2018advancements]。
但若用来描述整个领域，这一说法并不成立。
SelfSparring 的渐近收敛结果[@sui2017multi]、Kumagai 的连续定义域遗憾界[@kumagai2017regret]，以及偏好反馈的一项临床应用（StageOpt 的应用[@sui2018stagewise]），都早于 2019 年。
准确的说法范围更窄：这一问题的赌博机表述有理论保证，实践者运行的高斯过程流程则没有；并且如 @sec-pbo-theory 所述，理论分析的算法与实践中使用的流程至今仍不相同。
:::

**改变问题的形式。** 在图形学中，@koyama2017sequentialb 另辟蹊径。
他们的**序列线搜索**（sequential line search）把每次查询变成一个滑块：众包工作者沿设计空间中的一条直线拖动，停在自己最喜欢的点上（@sec-line-search）。
这篇论文开启了人机交互领域的一类研究，其主要变量是查询的形式，而非选择查询的规则。

**这一时期的问题。** 这几年的核心问题是如何向人提问。
机器学习改进了选择比较的规则，人机交互则主要改变查询的形式，很少改动采集函数（推断）。
图中 2019 年没有里程碑，但研究仍在继续，例如允许人回答“差不多”的工作[@byk2019asking]（@sec-obs-extensions）。

## 2020 至 2021 年：工具与推断定型 {#sec-history-2020}

**默认实现。** 2020 年 4 月，BoTorch（Meta 基于 PyTorch 构建的开源贝叶斯优化库，@sec-bo-software）在 0.2.3 版中加入了 `PairwiseGP`，即用于成对比较数据的模型[@botorch2026changelog]。
它实现的是概率单位链接加 Laplace 近似，此后这一组合一直是偏好贝叶斯优化中使用最广的实现。也就是说，从 2020 年起该领域有了事实标准，而这一标准正是 2005 年的模型（@sec-obs-pairwisegp）。

**子空间中的查询。** 同年出现的一些方法把每次查询限制在低维子空间中，以处理更多维度：在投影偏好贝叶斯优化中，人沿空间中的一条直线选出最好的点[@mikkola2020projective]；**序列画廊**（Sequential Gallery）则以网格形式展示设计的一个二维平面[@koyama2020sequential]（@sec-gallery-projective）。
在机器人学中，CoSpar 根据穿戴者的偏好学习外骨骼的行走步态[@tucker2020preference]，LineCoSpar 将其扩展到 6 个步态参数，并在 6 名身体健全的被试身上做了测试[@tucker2020human]。这两篇论文开创了外骨骼应用这一研究方向：@sec-health-exo 追踪其进展，@sec-cs-exoskeleton 以案例形式完整演示。

**精确后验。** 2021 年，@benavoli2021preferentialb 证明，在概率单位偏好似然下，效用的精确后验是**偏斜高斯过程**（skew Gaussian process）：它与高斯过程一样是函数上的分布，但其边际分布是偏斜的，而非对称的（@sec-skew-gp）。
Laplace 近似这类高斯近似无法表示这种偏斜；自此，后验推断的质量成为争论的焦点（@sec-obs-inference）。

**第一个核化界，以及一种替代方案。** 同样在 2021 年，@kirschner2021bias 给出了**核化**（kernelized）对决反馈下的第一个累积遗憾界。
核化是指假设未知效用在某个核函数所定义的意义下光滑，因此这一结果涵盖连续定义域（@sec-kernelized-dueling）；他们的反馈模型是效用差加噪声，而非偏好模型中的概率单位链接或逻辑链接。
控制工程领域独立处理了同一问题：GLISp 将径向基函数代理模型（即若干凸起的加权和）拟合到观测到的偏好上，完全不使用概率模型[@bemporad2021global]（@sec-obs-surrogates）。

至 2021 年底，该领域已有一个标准实现，已知这一实现的推断存在弱点，连续定义域的理论也已起步。
实际使用的采集函数则大多仍是启发式规则。

## 2022 至 2023 年：决策论转向 {#sec-history-2022}

**EUBO。** 启发式规则让位于有原则的规则。
在偏好探索贝叶斯优化（Bayesian optimization with preference exploration，BOPE）中，系统在实验进行的同时，通过比较学习人对实验结果的偏好。
针对这一设定，@lin2022preferenceb 提出了最优选项期望效用（expected utility of the best option，EUBO；@sec-eubo），并证明它**一步贝叶斯最优**（one-step Bayes optimal）：若会话在再获得一个回答后即结束，任何其他查询都不能得到期望意义下更好的最终推荐（@sec-eubo-theory）。
@astudillo2023qeubob 将其推广为 qEUBO，用于一次展示多个选项的查询和带逻辑噪声的回答，并证明期望改进经改编的一种批量版本不具有渐近一致性。
这就是决策论转向。qEUBO 成为 BoTorch 偏好采集函数的基础；BoTorch、Ax 与 optuna-dashboard 在数月之内即发布了这些新规则（@sec-sw-software）。

**推断质量的测量。** @takeno2023practicalc 测量了 Laplace 近似与期望传播偏离精确偏斜后验的程度，并提出**幻觉信念**（hallucination believer）：从后验中抽取潜在效用的一个样本，将其视为测得的数据，再套用任一标准采集函数（@sec-choosing-pairs）。
这一方法后来成为 optuna-dashboard（Optuna 优化库的网页界面）中偏好采样器的基础[@optuna2026dashboard]。

**人成为研究问题。** 与此同时，人机交互研究开始测量优化回路对其中的人有何影响。
设计过程由多目标优化器主导时，新手设计者报告的能动感与归属感，低于由自己主导设计过程的新手；这项研究使用的是性能目标，而非比较[@chan2022investigating]（@sec-hci-agency）。
在一次为期 3 个月的现场部署中，多数评分循环从未进入优化阶段，或没有收敛[@ou2022human]（@sec-sw-simulated-users）。
专家比新手迭代更多，最终却更不满意[@ou2023impact]（@sec-hci-expertise）。

**相邻领域壮大。** 2023 年，直接偏好优化（DPO）将 **Bradley-Terry** 似然置于大语言模型与人类偏好对齐的核心。在这一似然下，一个选项胜过另一个选项的概率是二者效用差的逻辑函数（@sec-bradley-terry）。DPO 直接用人的比较拟合语言模型，而不像基于人类反馈的强化学习（RLHF）那样借助奖励模型[@rafailov2023direct]。
同年，ICML 研讨会 *The Many Facets of Preference-Based Learning* 将对决赌博机、RLHF、社会选择与优化汇集在一起[@icml2023many]。
@sec-llms 讨论这两个领域之间的往来。

## 2024 至 2026 年：噪声、理论与审视 {#sec-history-2024}

最近三年可分为两个阶段。
2024 与 2025 年，理论、高维诊断、语言模型与正当性问题同时推进。
2026 年，研究集中于默认流程的缺陷及其更简单的替代方案。

### 2024 与 2025 年：理论、维度、语言模型与正当性 {#sec-hist-2024}

**Bradley-Terry 链接下的界。** 随着乐观算法 POP-BO[@xu2024principledb]、最大最小下置信界算法（MaxMinLCB）[@pasztor2024bandits]与多轮偏好反馈学习算法（MR-LPF）的提出，Bradley-Terry 链接下有了核化遗憾上界，即保证累积遗憾的增长不快于给定速率的结果；其中 MR-LPF 的上界与标量反馈同阶[@kayal2025bayesian]。
上界同阶并不说明一次比较与一个数值携带同样多的信息，只说明两者的保证同阶。
@sec-theory-rates 列出了每个速率及其假设。

**摊销优化器。** 偏好摊销黑箱优化（PABBO）预先在大量合成任务上训练神经网络，使其直接提出下一对选项；这样，一次查询只需一次前向传播，无需拟合模型、优化采集函数[@zhang2025pabbob]。
它是唯一面向成对偏好的摊销优化器（@sec-obs-surrogates）。

**高维问题的重新诊断。** 在以数值为观测的普通贝叶斯优化中，一系列论文将人们熟知的高维失效归因于先验与初始化，而非方法本身[@hvarfner2024vanilla; @xu2025standard; @papenmeier2025understanding]。
补救办法是**维度缩放先验**（dimension-scaled prior）：核函数长度尺度上的一种先验，其典型值随输入个数增大（@sec-lengthscale-priors）。
2024 年 9 月，BoTorch 0.12.0 将大多数模型改用这类先验，却明确排除了 `PairwiseGP`[@botorch2026changelog]；@sec-hd-diagnosis 考察这一缺口。

**语言模型与正当性。** 大语言模型开始用于对话式偏好引出[@austin2024bayesian]与对齐数据的主动收集[@dwaracherla2024efficient]。人机交互研究则转向群体先验，即把早期用户的偏好迁移给新用户[@li2025efficient]，以及设计者与优化器之间以自然语言进行的协作[@niwa2025cooperative]。
对齐研究还将偏好优化同样面临的一种担忧加以形式化：从偏好中学习的系统，可能改变其所测量的偏好。
@carroll2024aib 比较了面向可变偏好的八种对齐概念，发现每一种要么会奖励系统对人施加不当影响，要么过度规避风险；@williams2025targeted 发现，基于用户反馈优化的学习器，会学会专门针对最易受影响的用户（@sec-econ-endogenous）。

### 2026 年：受到审视的默认流程 {#sec-hist-2026}

**理论。** 偏好反馈下的 Thompson 采样（PF-TS）是完全序贯的算法，其上界为 $\gamma_T \sqrt{T}$ 阶[@lazzaro2026finiteb]；相比之下，MR-LPF 是带有有限候选集和预热期的分批算法，上界为 $\sqrt{\gamma_T T}$ 阶[@kayal2025bayesian]。
其中 $\gamma_T$ 是核函数的**最大信息增益**（maximum information gain），对光滑的核函数增长很慢（@sec-gp-information-gain）。

**默认流程的缺陷。** 2026 年的两篇预印本考察了由 `PairwiseGP`、Laplace 近似与 EUBO 组成的流程：EUBO 的查询向估计的最优选项坍缩[@wu2026knowledge]；它选出的成对选项与先前查询没有共同输入，致使 Laplace 似然的 Hessian 矩阵秩亏[@shao2026adaptive]。
@sec-pbo-failure-modes 已介绍过这两点。
观测模型扩展到了可能失败的实验[@menn2026preferential]；局部偏好贝叶斯优化（同为预印本）将该方法推进到约 100 维[@menn2026local]（@sec-hd-local）。

**更简单的替代方案与人。** 在普通的高维贝叶斯优化中，对输入做球面映射、再结合贝叶斯线性回归的方法（@sec-blr），在 60 至 6,000 维的任务上达到了最先进水平[@doumont2026we]。
Ax 加入了偏好优化，以及由语言模型将自由文本反馈转换为比较的试验[@ax2026changelog; @kobalczyk2026lilo]（@sec-llm-in-loop）。
2026 年采用强对照条件的人类研究，大多没有发现效应，或发现人手动调节的效果相差无几。
用于交互设备原型制作的成本感知贝叶斯优化，以约 67% 的成本达到相同性能，最终质量没有差异[@langerak2026cost]。
在一项有 12 名被试、以性能为目标的研究中，从用户模型学到的先验只在第二、三次迭代时有帮助[@liao2026efficient]。
11 名健康成年人用拇指杆遥控器自行调节外骨骼助力，在约 10.9 分钟内使代谢消耗降低 16.6%，与算法调节所报告的结果相当[@schafer2026user]。
由此，更简单的方法（包括人的手动调节）成为新方法必须与之比较的对照（@sec-hci-baselines）。
Benavoli 与 Azzimonti 的教程也已正式发表，内容是用高斯过程从偏好与选择中学习[@benavoli2026tutorial]；但该领域至今仍没有专门的综述（@sec-sw-infrastructure）。

## 核心问题如何转移 {#sec-hist-questions}

若把这十年看作一连串问题，其间经历了三次转向（推断）。

2017 至 2021 年，问题是：如何向人提问，又如何从回答中推断？
对决表述、序列线搜索、画廊与投影、`PairwiseGP` 和偏斜高斯过程，都回答了这一问题的某个方面。

2022 至 2025 年，问题变为：采集函数是否有原则，是否有理论保证？
EUBO 与 qEUBO 以一步贝叶斯最优性回答了前一半；POP-BO、MaxMinLCB 与 MR-LPF 以 Bradley-Terry 链接下的遗憾界回答了后一半。

2025 至 2026 年，问题同时变为四个：观测模型是否正确？人是否按模型假设的方式回答？更简单的方法是否已经足够？系统是否会改变它所测量的偏好？
EUBO 的坍缩、秩亏的 Hessian 矩阵、强对照下未发现效应的研究、在高维中胜出的线性模型，以及对齐研究中关于影响的结果，都属于这一阶段。
最后这次转向，就是从文献角度看到的本书主旨：瓶颈已经从算法转移到测量（@sec-syn-bottleneck）。

各时期在 2025 年有所重叠：第二个问题尚在解答之中，第三个问题已经提出，因此 @fig-hist-timeline 在这一年同时标出两者。

::: {.keyidea title="机制未变，问题在变"}
2026 年的默认流程，即高斯过程先验加概率单位链接与 Laplace 近似，就是 2005 年的模型。
十年间变化的是针对它提出的问题：先是如何提问，继而是查询的选择是否有原则、有保证，最后是描述人的模型究竟是否正确。
:::

后续各章分别承接其中一条线索。
@sec-observation-models 考察描述人如何回答的模型是否正确；@sec-acquisition-frontier 考察查询规则是否经得起检验；@sec-pbo-theory 梳理究竟证明了什么；@sec-high-dimensions 分析方法在何处失效、为何失效；@sec-software-evaluation 讨论软件默认做了什么、方法之间如何比较。
@sec-part-humans 讨论关于人的问题，@sec-part-perspectives 则从根本上探讨偏好是什么。

## 发表与社区 {#sec-history-community}

2023 年以后研究数量有所增长，但基数很小。arXiv 上摘要同时包含 *preferential*、*Bayesian* 与 *optimization*（或 *optimisation*）的条目，2023 年有 4 篇，2024 年 5 篇，2025 年 13 篇，2026 年前 9 个月 12 篇[@arxiv2026preferential]；这一计数会遗漏使用其他措辞（例如“dueling”或“human feedback”）的论文（推断）。
这些研究分散于机器学习、控制、机器人学与人机交互的各类发表场所，各社区之间很少相互引用：2021 年发表于 *Journal of Machine Learning Research* 的对决赌博机综述，就没有引用 González 等人 2017 年的论文[@bengs2021preference]。
@tbl-sw-groups 按学科列出了主要的研究组；证明遗憾界的研究组不做人类研究，做人类研究的研究组很少改动采集函数（推断）。
更完整的叙述见 @sec-sw-community。

## 已定、有争议与缺失 {#sec-hist-status}

::: {.frontier title="已定、有争议与缺失"}
**已定。** 默认流程在这十年间没有改变：@chu2005preference 的高斯过程、概率单位与 Laplace 模型，自 2020 年 4 月起由 BoTorch 的 `PairwiseGP` 实现[@botorch2026changelog]。
回答无噪声时，EUBO 与 qEUBO 具有一步贝叶斯最优性[@lin2022preferenceb; @astudillo2023qeubob]。
从比较中学习的形式化保证，在对决赌博机一侧远早于 2024 年就已存在[@sui2017multi; @kumagai2017regret]；Bradley-Terry 链接下的核化界则始于 2024 年[@xu2024principledb]。
BoTorch 在 2024 年改用维度缩放先验时，明确排除了偏好模型[@botorch2026changelog]。

**有争议。** 2026 年报告的两项缺陷，即 EUBO 向当前最优点坍缩与 Hessian 矩阵秩亏，在人的任务上是否带来任何代价；两者都只有预印本为据[@wu2026knowledge; @shao2026adaptive]。
对照足够强时，偏好优化是否胜过更简单的替代方案：2026 年的受控人类研究发现优势不大或没有优势[@langerak2026cost; @liao2026efficient; @schafer2026user]，而在标量反馈的高维问题中，线性模型达到了最先进水平[@doumont2026we]。

**缺失。** 专门的偏好贝叶斯优化综述。
社区之间的交流：2021 年 JMLR 上的对决赌博机综述没有引用为该领域命名的论文[@bengs2021preference]。
在相同界面与预算下把被试随机分配到不同采集函数的研究，以及以预注册终点指标与专家手动调节相比较的研究（@sec-open-decisive）。
:::

## 延伸阅读 {#further-reading .unnumbered}

- @gonzalez2017preferentialb 为该领域命名；宜与 @chu2005preference 对照阅读，该领域后来采用的是后者的模型，而非对决表述。
- @sui2018advancements 与 @bengs2021preference 是两篇对决赌博机综述；合起来读，可以看出赌博机社区了解什么、没有引用什么。
- @lin2022preferenceb 与 @astudillo2023qeubob 标志着决策论转向。
- @benavoli2026tutorial 是论述偏好与选择的高斯过程模型最完整的一份文献。
- BoTorch 的更新日志[@botorch2026changelog]逐版记录了实践者实际能够运行的内容，可作一部简明的历史来读。
