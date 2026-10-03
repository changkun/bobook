// A timeline of preferential Bayesian optimization from the 2005 baselines to
// September 2026. Events sit in six lanes (models and inference, queries and
// acquisition, theory, people and applications, software, neighboring fields)
// against the phases of the field's history. The mark encodes the evidence:
// a filled circle for a peer-reviewed paper or an event, a hollow circle for
// a preprint, a square for a software release. Selecting an event (click,
// the list, or Previous / Next) shows what it was, where it appeared, why it
// mattered, and the central question of the period; "Hide later years" fades
// everything after the selected year, so stepping through the list replays
// the decade. Within a year, positions are spread for legibility and are not
// dates.

import { defineFigure, tr, type Lang, type State, type Text } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { tpl } from "./lib/format.ts";

type Lane = "model" | "query" | "theory" | "people" | "software" | "field";
type Kind = "paper" | "preprint" | "software";

interface Ev {
  id: string;
  y: number;
  lane: Lane;
  kind: Kind;
  pick: Text; // short name for the list
  title: Text;
  venue: Text;
  why: Text;
}

// The Chinese edition of each event: [pick, title, venue, why]. Venues that
// are names (ICML 2017, JMLR 2021) stay as they are.
const ZH: Record<string, [string, string, string, string]> = {
  chu2005: ["高斯过程偏好模型", "Chu 与 Ghahramani：采用概率单位链接与拉普拉斯近似的高斯过程偏好学习", "ICML 2005", "基线模型。借助 BoTorch 的 PairwiseGP，它在 2026 年仍是默认模型。"],
  brochu2007: ["偏好画廊", "Brochu 等人：用于材质设计的画廊式主动偏好学习", "NeurIPS 2007", "早期的交互式设计工具，从人的选择中学习，并决定下一步展示什么。"],
  yue2009: ["对决赌博机", "Yue 与 Joachims：把检索系统的交互式优化表述为对决赌博机问题", "ICML 2009", "把对决赌博机的表述带入交互式优化：学习者只能看到两个选项中哪一个获胜。"],
  gonzalez2017: ["PBO 得名（González）", "González 等人：对决表述、这一名称与三个采集函数", "ICML 2017", "问题表述与这一名称的起点。它没有给出收敛理论。"],
  koyama2017: ["序列线搜索", "Koyama 等人：序列线搜索，每次众包查询是一个滑块", "SIGGRAPH 2017", "开启了人机交互中以查询形式为主要变量的研究方向。"],
  sui2017: ["SelfSparring", "Sui 等人：用于多方对决赌博机的 SelfSparring，具有渐近收敛性", "UAI 2017", "从比较中学习的形式化保证早在 2024 年之前就已存在。"],
  kumagai2017: ["连续空间对决界", "Kumagai：连续空间上对决赌博机的遗憾界", "NeurIPS 2017", "在偏好贝叶斯优化这一名称出现的同一年，给出了连续定义域上对决的遗憾界。"],
  stageopt2018: ["StageOpt", "Sui 等人：StageOpt，未知安全约束下的安全贝叶斯优化，以及一个从偏好中学习效用的变体", "ICML 2018", "偏好反馈的一项临床应用：脊髓刺激；其安全性与收敛定理针对的是数值观测。"],
  suisurvey2018: ["赌博机综述（IJCAI）", "Sui 等人：一篇对决赌博机综述称偏好贝叶斯优化是没有收敛速率保证的纯贝叶斯优化方法", "IJCAI 2018", "指出了理论上的空白，2021 至 2026 年的核化界开始填补它。"],
  mikkola2020: ["投影偏好贝叶斯优化", "Mikkola 等人：投影偏好贝叶斯优化", "ICML 2020", "把每次查询限制在一个低维子空间中，是 2020 年应对更高维度的一组方案之一。"],
  koyama2020: ["序列画廊", "Koyama 等人：序列画廊，从二维的选项画廊中选择", "SIGGRAPH 2020", "另一种低维查询：穿过设计空间的一个平面，以网格形式展示。"],
  cospar2020: ["CoSpar", "Tucker 等人：CoSpar，基于偏好的外骨骼步态优化", "ICRA 2020", "开启了外骨骼应用这一研究方向。"],
  linecospar2020: ["LineCoSpar", "Tucker 等人：LineCoSpar，沿直线进行的更高维外骨骼步态优化", "IROS 2020", "低维查询使更多步态参数得以调节。"],
  botorch023: ["BoTorch 0.2.3：PairwiseGP", "BoTorch 0.2.3 加入 PairwiseGP", "软件发布，2020 年 4 月", "概率单位链接加拉普拉斯近似成为事实上的默认实现。"],
  benavoli2021: ["偏斜高斯过程后验", "Benavoli 等人：概率单位偏好似然下的精确后验是偏斜高斯过程", "GECCO 2021 Companion", "高斯近似的误差成为争论的焦点。"],
  glisp2021: ["GLISp", "Bemporad 与 Piga：GLISp，一种非概率的径向基函数代理模型", "Machine Learning 2021", "控制工程在没有概率模型的情况下处理同一问题。"],
  kirschner2021: ["核化对决界", "Kirschner 与 Krause：核化对决的累积遗憾，反馈为效用差加噪声", "ICML 2021", "核化偏好反馈的第一个累积遗憾界。"],
  bengs2021: ["赌博机综述（JMLR）", "Bengs 等人：基于偏好的在线学习与对决赌博机综述", "JMLR 2021", "它没有引用 González 等人 2017 年的论文，可见各个社区之间很少互相引用。"],
  cglisp2022: ["C-GLISp", "Zhu 等人：C-GLISp，未知约束下基于偏好的全局优化", "IEEE Transactions on Control Systems Technology 2022", "控制方向的研究扩展到未知约束与控制器校准。"],
  lin2022: ["BOPE 中的 EUBO", "Lin 等人：在偏好探索（BOPE）中，EUBO 是一步贝叶斯最优的", "AISTATS 2022", "采集函数的决策论转向。"],
  chan2022: ["能动感与归属感", "Chan 等人：由优化器主导时，设计者的能动感与归属感下降", "CHI 2022", "人的因素成为研究问题。"],
  ou2022: ["现场中的循环", "Ou 等人：在一次现场部署中，多数循环从未进入优化阶段，或没有收敛", "Mensch und Computer 2022", "真实的循环与模拟的循环表现不同。"],
  botorch063: ["BoTorch 0.6.3：EUBO", "BoTorch 0.6.3 加入解析 EUBO、学习得到的目标与 BOPE 教程", "软件发布，2022 年 3 月", "决策论采集函数进入默认库。"],
  botorch065: ["BoTorch 0.6.5：logit", "BoTorch 0.6.5 加入逻辑链接似然，并把偏好贝叶斯优化教程改用 EUBO", "软件发布，2022 年 7 月", "概率单位链接与逻辑链接从此并存可用。"],
  ax026: ["Ax 0.2.6：成对模型", "Ax 0.2.6 是软件包中首次包含 PairwiseModelBridge 的版本", "软件发布，2022 年 8 月", "偏好模型进入 Meta 的实验管理层。"],
  takeno2023: ["幻觉信念", "Takeno 等人：测量高斯近似的误差，并提出幻觉信念规则", "ICML 2023", "optuna-dashboard 中偏好采样器的基础。"],
  astudillo2023: ["qEUBO", "Astudillo 等人：面向多选项查询与逻辑噪声的 qEUBO；批量期望改进不具一致性", "AISTATS 2023", "BoTorch 偏好采集函数的基础。"],
  ou2023: ["回路中的专业水平", "Ou 等人：专家比新手迭代更多，最终却更不满意", "IUI 2023", "人是谁，会改变循环的运行方式。"],
  botorch090: ["BoTorch 0.9.0：BALD", "BoTorch 0.9.0 加入成对的贝叶斯分歧主动学习采集函数", "软件发布，2023 年 8 月", "默认库中一种基于信息的 EUBO 替代方案。"],
  optuna2023: ["optuna-dashboard 采样器", "optuna-dashboard 0.13.0b1 提供带网页界面的偏好高斯过程采样器", "软件发布，2023 年 9 月", "Takeno 等人的方法带着面向人的界面到达实践者手中。"],
  dpo2023: ["DPO", "Rafailov 等人：直接偏好优化", "NeurIPS 2023", "Bradley-Terry 似然成为对齐语言模型的核心，偏好研究的规模随之扩大。"],
  ws2023: ["ICML 偏好研讨会", "The Many Facets of Preference-Based Learning：对决赌博机、RLHF、社会选择与优化同聚一个研讨会", "ICML 2023 研讨会", "把从偏好中学习的几个社区聚到一起；我们没有找到专门讨论偏好贝叶斯优化本身的研讨会。"],
  popbo2024: ["POP-BO", "Xu 等人：POP-BO，在 Bradley-Terry 链接下具有核化遗憾界的乐观算法", "ICML 2024", "Bradley-Terry 链接下的连续定义域理论。"],
  maxmin2024: ["MaxMinLCB", "Pásztor 等人：MaxMinLCB，按最大最小下置信界选择成对选项", "NeurIPS 2024", "Bradley-Terry 链接下的连续定义域理论。"],
  botorch0100: ["BoTorch 0.10：qEUBO", "BoTorch 0.10.0 加入 qEUBO", "软件发布，2024 年 2 月", "此版本之后，BoTorch 的偏好模块只有修复。"],
  botorch0120: ["BoTorch 0.12：先验", "BoTorch 0.12.0 把多数模型改为维度缩放的长度尺度先验，PairwiseGP 除外", "软件发布，2024 年 9 月", "标量贝叶斯优化的高维修正没有覆盖偏好模型。"],
  hvarfner2024: ["维度缩放先验", "Hvarfner 等人：使用随维度缩放的先验，标准贝叶斯优化在高维中也能奏效", "ICML 2024", "高维失效被重新诊断为先验与初始化的问题。"],
  carroll2024: ["会改变的偏好", "Carroll 等人：当偏好可以改变时，每一种对齐目标要么奖励不当影响，要么过于保守", "ICML 2024", "由系统引起的偏好改变成为一个形式化问题。"],
  austin2024: ["大语言模型引出偏好", "Austin 等人：用于对话式偏好引出、以语言模型做采集的贝叶斯优化", "RecSys 2024", "语言模型进入对话式偏好引出。"],
  dwaracherla2024: ["大语言模型的探索", "Dwaracherla 等人：为语言模型收集反馈时的高效探索", "ICML 2024", "语言模型进入对齐数据的主动收集。"],
  pabbo2025: ["PABBO", "Zhang 等人：PABBO，偏好摊销黑箱优化", "ICLR 2025", "唯一面向成对偏好的摊销优化器：由预先训练好的网络提出查询。"],
  iwai2025: ["约束偏好贝叶斯优化", "Iwai 等人：约束偏好贝叶斯优化，应用于横幅广告设计", "IJCAI 2025", "视觉设计方向的研究转向约束。"],
  kayal2025: ["MR-LPF", "Kayal 等人：MR-LPF，通过分批淘汰进行的多轮偏好反馈学习", "ICML 2025", "上界与标量反馈同阶。"],
  li2025: ["群体先验", "Li 等人：把早期用户的偏好作为先验迁移（Meta-PO）", "UIST 2025", "人机交互研究转向群体先验。"],
  niwa2025: ["自然语言协同设计", "Niwa 等人：通过自然语言交互进行协同设计优化", "UIST 2025", "人机交互研究转向自然语言协作。"],
  xu2025: ["标准高斯过程即可", "Xu 等人：高维贝叶斯优化只需标准高斯过程", "ICLR 2025", "对高维失效的重新诊断仍在继续。"],
  papenmeier2025: ["高维贝叶斯优化剖析", "Papenmeier 等人：理解高维贝叶斯优化", "ICML 2025", "对高维失效的重新诊断仍在继续。"],
  williams2025: ["定向操纵", "Williams 等人：在用户反馈上优化的学习器学会了针对最容易受影响的用户", "ICLR 2025", "用反馈训练的系统可能学会改变提供反馈的人。"],
  shao2026: ["秩亏的 Hessian 矩阵", "Shao 等人：在 EUBO 的查询下，拉普拉斯似然的 Hessian 矩阵变得秩亏", "arXiv 2026，预印本", "缺陷位于默认流程本身。"],
  crash2026: ["崩溃反馈", "Menn 等人：CrashPBO 把崩溃当作第二种结果", "IEEE Robotics and Automation Letters 2026", "观测模型扩展到会失败的实验。"],
  tutorial2026: ["高斯过程偏好教程", "Benavoli 与 Azzimonti：用高斯过程从偏好与选择中学习的教程", "Foundations and Trends in Machine Learning 2026", "偏好与选择模型的参考资料。截至 2026 年 9 月，仍没有专门的偏好贝叶斯优化综述。"],
  wu2026: ["精确知识梯度", "Wu 与 Gardner：概率单位似然下的精确知识梯度；EUBO 的查询向估计的最优处坍缩", "arXiv 2026，预印本", "一旦回答带有噪声，决策论的默认方法就受到质疑。"],
  local2026: ["局部偏好贝叶斯优化", "Menn 等人：局部偏好贝叶斯优化", "arXiv 2026，预印本", "把偏好贝叶斯优化推进到约 100 维，所用的模拟比较次数约为维度的十倍。"],
  lilo2026: ["LILO", "Kobalczyk 等人：LILO，由语言模型把自由文本反馈转换为成对标签", "ICML 2026", "语言模型以文本到比较的转换者身份进入主流软件。"],
  pfts2026: ["PF-TS", "Lazzaro 等人：偏好反馈下的汤普森采样，完全序贯算法的 γ_T √T 界", "AISTATS 2026", "序贯算法与分批算法现在相差一个因子，即信息增益的平方根。"],
  langerak2026: ["成本感知贝叶斯优化", "Langerak 等人：用于交互设备原型制作的成本感知贝叶斯优化", "CHI 2026", "以约 67% 的成本达到相同性能，最终质量没有差异。"],
  liao2026: ["来自用户模型的先验", "Liao 等人：在真实用户到来之前，从用户模型中学习先验", "CHI 2026", "只在第二、三次迭代时更好；从第六次起持平。"],
  schafer2026: ["手动自调", "Schäfer 等人：人用拇指摇杆自行调节外骨骼助力", "npj Biomedical Innovations 2026", "11 名健康成年人在约 10.9 分钟内使代谢消耗降低 16.6%，与算法调节相当。"],
  ax122: ["Ax 1.2：偏好", "Ax 1.2.2 与 1.2.3 加入偏好优化配置、BOPE 效用轨迹与语言模型抽象", "软件发布，2026 年 1 月与 2 月", "新的偏好功能落在 Ax 而不是 BoTorch 中。"],
  ax130: ["Ax 1.3.0：LILO、qEUBO", "Ax 1.3.0 加入 LILO 标注试验与自动 qEUBO 调度", "软件发布，2026 年 6 月", "语言模型标注进入主流软件。"],
  botorch018: ["BoTorch 0.18：先验未变", "BoTorch 0.18.0 修复 PairwiseGP 的状态处理；其长度尺度先验仍是 Gamma(2.4, 2.7)", "软件发布，2026 年 6 月", "默认偏好模型仍不考虑维度。"],
  doumont2026: ["线性模型，6,000 维", "Doumont 等人：球面输入映射加贝叶斯线性回归，在 60 至 6,000 维的任务上达到最先进水平", "AISTATS 2026", "高维上的成功并不需要复杂的代理模型。"],
};

const E = (id: string, y: number, lane: Lane, kind: Kind, pick: string, title: string, venue: string, why: string): Ev =>
  ({ id, y, lane, kind, pick: { en: pick, zh: ZH[id]?.[0] }, title: { en: title, zh: ZH[id]?.[1] }, venue: { en: venue, zh: ZH[id]?.[2] }, why: { en: why, zh: ZH[id]?.[3] } });

// Ordered by year, then lane, then as listed. Every event is cited in the
// chapter's prose or in the figure caption.
const EVENTS: Ev[] = [
  E("chu2005", 2005, "model", "paper", "GP preference model",
    "Chu and Ghahramani: Gaussian process preference learning with a probit link and the Laplace approximation",
    "ICML 2005", "The baseline model. Through BoTorch's PairwiseGP it is still the default in 2026."),
  E("brochu2007", 2007, "query", "paper", "Preference gallery",
    "Brochu et al.: gallery-style active preference learning for material design",
    "NeurIPS 2007", "An early interactive design tool that learns from a person's choices and decides what to show next."),
  E("yue2009", 2009, "theory", "paper", "Dueling bandits",
    "Yue and Joachims: interactive optimization of retrieval systems as a dueling bandit problem",
    "ICML 2009", "Brings the dueling bandit framing, in which the learner sees only which of two options wins, to interactive optimization."),
  E("gonzalez2017", 2017, "query", "paper", "PBO named (González)",
    "González et al.: the dueling formulation, the name, and three acquisition functions",
    "ICML 2017", "The starting point of the problem statement and of the name. It came without convergence theory."),
  E("koyama2017", 2017, "query", "paper", "Sequential line search",
    "Koyama et al.: sequential line search, one slider per crowdsourced query",
    "SIGGRAPH 2017", "Begins the human-computer interaction line in which the form of the query is the main variable."),
  E("sui2017", 2017, "theory", "paper", "SelfSparring",
    "Sui et al.: SelfSparring for multi-dueling bandits, with asymptotic convergence",
    "UAI 2017", "Formal guarantees for learning from comparisons existed well before 2024."),
  E("kumagai2017", 2017, "theory", "paper", "Continuous dueling bound",
    "Kumagai: a regret bound for dueling bandits over a continuous space",
    "NeurIPS 2017", "A regret bound for duels over a continuous domain, in the year the name PBO appeared."),
  E("stageopt2018", 2018, "theory", "paper", "StageOpt",
    "Sui et al.: StageOpt, safe Bayesian optimization with unknown safety constraints, and a variant that learns the utility from preferences",
    "ICML 2018", "A clinical application of preference feedback, spinal cord stimulation; its safety and convergence theorems are for numerical observations."),
  E("suisurvey2018", 2018, "theory", "paper", "Bandit survey (IJCAI)",
    "Sui et al.: a dueling bandit survey describes PBO as a purely Bayesian optimization method without guarantees on its convergence rate",
    "IJCAI 2018", "Names the gap in theory that the kernelized bounds of 2021 to 2026 began to fill."),
  E("mikkola2020", 2020, "query", "paper", "Projective PBO",
    "Mikkola et al.: projective preferential Bayesian optimization",
    "ICML 2020", "Restricts each query to a low-dimensional subspace, one of a group of 2020 answers to higher dimension."),
  E("koyama2020", 2020, "query", "paper", "Sequential Gallery",
    "Koyama et al.: Sequential Gallery, choosing from a two-dimensional gallery of options",
    "SIGGRAPH 2020", "Another low-dimensional query: a plane through the design space, shown as a grid."),
  E("cospar2020", 2020, "people", "paper", "CoSpar",
    "Tucker et al.: CoSpar, preference-based optimization of exoskeleton gaits",
    "ICRA 2020", "Begins the line of exoskeleton applications."),
  E("linecospar2020", 2020, "people", "paper", "LineCoSpar",
    "Tucker et al.: LineCoSpar, higher-dimensional exoskeleton gait optimization along lines",
    "IROS 2020", "Low-dimensional queries let more gait parameters be tuned."),
  E("botorch023", 2020, "software", "software", "BoTorch 0.2.3: PairwiseGP",
    "BoTorch 0.2.3 adds PairwiseGP",
    "Software release, April 2020", "The probit link with the Laplace approximation becomes the de facto default implementation."),
  E("benavoli2021", 2021, "model", "paper", "Skew GP posterior",
    "Benavoli et al.: the exact posterior under a probit preference likelihood is a skew Gaussian process",
    "GECCO 2021 Companion", "The error of Gaussian approximations becomes a point of dispute."),
  E("glisp2021", 2021, "model", "paper", "GLISp",
    "Bemporad and Piga: GLISp, a non-probabilistic radial basis function surrogate",
    "Machine Learning 2021", "Control engineering tackles the same problem without a probabilistic model."),
  E("kirschner2021", 2021, "theory", "paper", "Kernelized dueling bound",
    "Kirschner and Krause: cumulative regret for kernelized dueling, with feedback equal to a utility difference plus noise",
    "ICML 2021", "The first cumulative regret bound for kernelized preference feedback."),
  E("bengs2021", 2021, "theory", "paper", "Bandit survey (JMLR)",
    "Bengs et al.: a survey of preference-based online learning with dueling bandits",
    "JMLR 2021", "It does not cite González et al. 2017, a sign of how rarely the communities cite one another."),
  E("cglisp2022", 2022, "model", "paper", "C-GLISp",
    "Zhu et al.: C-GLISp, preference-based global optimization under unknown constraints",
    "IEEE Transactions on Control Systems Technology 2022", "The control line extends to unknown constraints and controller calibration."),
  E("lin2022", 2022, "query", "paper", "EUBO in BOPE",
    "Lin et al.: EUBO is one-step Bayes optimal in preference exploration (BOPE)",
    "AISTATS 2022", "The decision-theoretic turn in acquisition functions."),
  E("chan2022", 2022, "people", "paper", "Agency and ownership",
    "Chan et al.: when the optimizer leads, designers' sense of agency and ownership drops",
    "CHI 2022", "Human factors become a research question."),
  E("ou2022", 2022, "people", "paper", "Loops in the field",
    "Ou et al.: in a field deployment, most loops never entered optimization or did not converge",
    "Mensch und Computer 2022", "Real loops behave differently from simulated ones."),
  E("botorch063", 2022, "software", "software", "BoTorch 0.6.3: EUBO",
    "BoTorch 0.6.3 adds analytic EUBO, learned objectives, and a BOPE tutorial",
    "Software release, March 2022", "The decision-theoretic acquisition function reaches the default library."),
  E("botorch065", 2022, "software", "software", "BoTorch 0.6.5: logit",
    "BoTorch 0.6.5 adds a logistic-link likelihood and switches the PBO tutorial to EUBO",
    "Software release, July 2022", "Probit and logistic links are now available side by side."),
  E("ax026", 2022, "software", "software", "Ax 0.2.6: pairwise model",
    "Ax 0.2.6 is the first release whose package contains PairwiseModelBridge",
    "Software release, August 2022", "Preference models reach Meta's experiment-management layer."),
  E("takeno2023", 2023, "model", "paper", "Hallucination believer",
    "Takeno et al.: the error of Gaussian approximations measured, and the hallucination believer rule",
    "ICML 2023", "The basis of the preferential sampler in optuna-dashboard."),
  E("astudillo2023", 2023, "query", "paper", "qEUBO",
    "Astudillo et al.: qEUBO for queries with several options and logistic noise; batch expected improvement is not consistent",
    "AISTATS 2023", "The basis of BoTorch's preference acquisition functions."),
  E("ou2023", 2023, "people", "paper", "Expertise in the loop",
    "Ou et al.: experts iterate more than novices and end less satisfied",
    "IUI 2023", "Who the person is changes how the loop runs."),
  E("botorch090", 2023, "software", "software", "BoTorch 0.9.0: BALD",
    "BoTorch 0.9.0 adds a pairwise Bayesian active learning by disagreement acquisition function",
    "Software release, August 2023", "An information-based alternative to EUBO in the default library."),
  E("optuna2023", 2023, "software", "software", "optuna-dashboard sampler",
    "optuna-dashboard 0.13.0b1 offers a preferential Gaussian process sampler with a web interface",
    "Software release, September 2023", "Takeno et al.'s method reaches practitioners with an interface for people."),
  E("dpo2023", 2023, "field", "paper", "DPO",
    "Rafailov et al.: direct preference optimization",
    "NeurIPS 2023", "The Bradley-Terry likelihood becomes central to aligning language models, and preference research grows in scale."),
  E("ws2023", 2023, "field", "paper", "ICML preference workshop",
    "The Many Facets of Preference-Based Learning: dueling bandits, RLHF, social choice, and optimization in one workshop",
    "ICML 2023 workshop", "Brings several communities that learn from preferences together; we found no workshop devoted to PBO itself."),
  E("popbo2024", 2024, "theory", "paper", "POP-BO",
    "Xu et al.: POP-BO, an optimistic algorithm with a kernelized regret bound under the Bradley-Terry link",
    "ICML 2024", "Continuous-domain theory under the Bradley-Terry link."),
  E("maxmin2024", 2024, "theory", "paper", "MaxMinLCB",
    "Pásztor et al.: MaxMinLCB, pairs chosen by a max-min lower confidence bound",
    "NeurIPS 2024", "Continuous-domain theory under the Bradley-Terry link."),
  E("botorch0100", 2024, "software", "software", "BoTorch 0.10: qEUBO",
    "BoTorch 0.10.0 adds qEUBO",
    "Software release, February 2024", "After this release, BoTorch's preference module receives only fixes."),
  E("botorch0120", 2024, "software", "software", "BoTorch 0.12: priors",
    "BoTorch 0.12.0 switches most models to dimension-scaled lengthscale priors, excluding PairwiseGP",
    "Software release, September 2024", "Scalar BO's high-dimensional fix does not reach the preference model."),
  E("hvarfner2024", 2024, "field", "paper", "Dimension-scaled priors",
    "Hvarfner et al.: with priors that scale with dimension, standard Bayesian optimization works in high dimensions",
    "ICML 2024", "High-dimensional failure is re-diagnosed as a problem of priors and initialization."),
  E("carroll2024", 2024, "field", "paper", "Changing preferences",
    "Carroll et al.: when preferences can change, every alignment objective either rewards undue influence or is overly conservative",
    "ICML 2024", "Preference change caused by the system becomes a formal problem."),
  E("austin2024", 2024, "field", "paper", "LLM elicitation",
    "Austin et al.: Bayesian optimization with language-model acquisition for conversational preference elicitation",
    "RecSys 2024", "Language models enter conversational preference elicitation."),
  E("dwaracherla2024", 2024, "field", "paper", "Exploration for LLMs",
    "Dwaracherla et al.: efficient exploration when collecting feedback for language models",
    "ICML 2024", "Language models enter the active collection of alignment data."),
  E("pabbo2025", 2025, "query", "paper", "PABBO",
    "Zhang et al.: PABBO, preferential amortized black-box optimization",
    "ICLR 2025", "The only amortized optimizer for pairwise preferences: a network trained in advance proposes the queries."),
  E("iwai2025", 2025, "query", "paper", "Constrained PBO",
    "Iwai et al.: constrained preferential Bayesian optimization, applied to banner ad design",
    "IJCAI 2025", "The visual-design line turns to constraints."),
  E("kayal2025", 2025, "theory", "paper", "MR-LPF",
    "Kayal et al.: MR-LPF, multi-round learning from preference feedback by batched elimination",
    "ICML 2025", "An upper bound of the same order as for scalar feedback."),
  E("li2025", 2025, "people", "paper", "Population priors",
    "Li et al.: earlier users' preferences transferred as priors (Meta-PO)",
    "UIST 2025", "Human-computer interaction turns to population priors."),
  E("niwa2025", 2025, "people", "paper", "Natural-language design",
    "Niwa et al.: cooperative design optimization through natural-language interaction",
    "UIST 2025", "Human-computer interaction turns to collaboration in natural language."),
  E("xu2025", 2025, "field", "paper", "Standard GP suffices",
    "Xu et al.: a standard Gaussian process is all you need for high-dimensional Bayesian optimization",
    "ICLR 2025", "The re-diagnosis of high-dimensional failure continues."),
  E("papenmeier2025", 2025, "field", "paper", "High-dim BO analyzed",
    "Papenmeier et al.: understanding high-dimensional Bayesian optimization",
    "ICML 2025", "The re-diagnosis of high-dimensional failure continues."),
  E("williams2025", 2025, "field", "paper", "Targeted manipulation",
    "Williams et al.: learners optimized on user feedback learn to target the users most open to influence",
    "ICLR 2025", "A system trained on feedback can learn to change the people who give it."),
  E("shao2026", 2026, "model", "preprint", "Rank-deficient Hessian",
    "Shao et al.: with EUBO's queries, the Laplace likelihood Hessian becomes rank deficient",
    "arXiv 2026, preprint", "A flaw located in the default pipeline itself."),
  E("crash2026", 2026, "model", "paper", "Crash feedback",
    "Menn et al.: CrashPBO treats a crash as a second kind of outcome",
    "IEEE Robotics and Automation Letters 2026", "Observation models extend to experiments that fail."),
  E("tutorial2026", 2026, "model", "paper", "GP preference tutorial",
    "Benavoli and Azzimonti: a tutorial on learning from preferences and choices with Gaussian processes",
    "Foundations and Trends in Machine Learning 2026", "A reference for preference and choice models. As of September 2026 there is still no dedicated survey of PBO."),
  E("wu2026", 2026, "query", "preprint", "Exact knowledge gradient",
    "Wu and Gardner: the exact knowledge gradient under a probit likelihood; EUBO's queries collapse toward the estimated best",
    "arXiv 2026, preprint", "The decision-theoretic default is questioned once answers are noisy."),
  E("local2026", 2026, "query", "preprint", "Local PBO",
    "Menn et al.: local preferential Bayesian optimization",
    "arXiv 2026, preprint", "Takes PBO to about 100 dimensions, with simulated comparisons numbering about ten times the dimension."),
  E("lilo2026", 2026, "query", "paper", "LILO",
    "Kobalczyk et al.: LILO, a language model translates free-text feedback into pairwise labels",
    "ICML 2026", "Language models enter mainstream software as translators from text to comparisons."),
  E("pfts2026", 2026, "theory", "paper", "PF-TS",
    "Lazzaro et al.: Thompson sampling with preference feedback, a γ_T √T bound for a fully sequential algorithm",
    "AISTATS 2026", "Sequential and batched algorithms now differ by a factor of the square root of the information gain."),
  E("langerak2026", 2026, "people", "paper", "Cost-aware BO",
    "Langerak et al.: cost-aware Bayesian optimization for prototyping interactive devices",
    "CHI 2026", "The same performance at about 67% of the cost, with no difference in final quality."),
  E("liao2026", 2026, "people", "paper", "Priors from user models",
    "Liao et al.: priors learned from user models before real users arrive",
    "CHI 2026", "Better only at the second and third iterations; equal from the sixth on."),
  E("schafer2026", 2026, "people", "paper", "Manual self-tuning",
    "Schäfer et al.: people tune their own exoskeleton assistance with a thumb joystick",
    "npj Biomedical Innovations 2026", "11 healthy adults reached a 16.6% metabolic reduction in about 10.9 minutes, comparable to algorithmic tuning."),
  E("ax122", 2026, "software", "software", "Ax 1.2: preferences",
    "Ax 1.2.2 and 1.2.3 add a preference optimization configuration, a BOPE utility trace, and language-model abstractions",
    "Software release, January and February 2026", "New preference features land in Ax rather than in BoTorch."),
  E("ax130", 2026, "software", "software", "Ax 1.3.0: LILO, qEUBO",
    "Ax 1.3.0 adds LILO labeling trials and automatic qEUBO scheduling",
    "Software release, June 2026", "Language-model labeling reaches mainstream software."),
  E("botorch018", 2026, "software", "software", "BoTorch 0.18: same prior",
    "BoTorch 0.18.0 fixes PairwiseGP's state handling; its lengthscale prior is still Gamma(2.4, 2.7)",
    "Software release, June 2026", "The default preference model still ignores dimension."),
  E("doumont2026", 2026, "field", "paper", "Linear model, 6000 dims",
    "Doumont et al.: a spherical input mapping with Bayesian linear regression reaches the state of the art on 60- to 6000-dimensional tasks",
    "AISTATS 2026", "A complex surrogate is not necessary for high-dimensional success."),
];

const LANES: Array<{ id: Lane; label: Text; color: string }> = [
  { id: "model", label: { en: "Models, inference", zh: "模型与推断" }, color: C.model },
  { id: "query", label: { en: "Queries, acquisition", zh: "查询与采集" }, color: C.acq },
  { id: "theory", label: { en: "Theory", zh: "理论" }, color: C.c7 },
  { id: "people", label: { en: "People, applications", zh: "人与应用" }, color: C.c6 },
  { id: "software", label: { en: "Software", zh: "软件" }, color: C.c5 },
  { id: "field", label: { en: "Neighboring fields", zh: "相邻领域" }, color: C.ink3 },
];

// Phases of the chapter, as [first year, last year].
const PHASES = [
  { a: 2005, b: 2016, range: "2005–16", short: "05–16", name: { en: "baselines", zh: "基线" } },
  { a: 2017, b: 2019, range: "2017–19", short: "17–19", name: { en: "how to ask", zh: "如何提问" } },
  { a: 2020, b: 2021, range: "2020–21", short: "20–21", name: { en: "tools settle", zh: "工具定型" } },
  { a: 2022, b: 2023, range: "2022–23", short: "22–23", name: { en: "decision theory", zh: "决策论" } },
  { a: 2024, b: 2026, range: "2024–26", short: "24–26", name: { en: "theory, scrutiny", zh: "理论与审视" } },
];

// The central question of each period, an inference drawn from the record.
const QUESTIONS: Record<Lang, string[]> = {
  en: [
    "Before the name, the pieces lived in separate communities.",
    "How should we ask a person, and how should we infer from the answers?",
    "Are acquisition functions principled, and do they come with guarantees?",
    "Still: are acquisition functions principled and guaranteed? Increasingly also: is the observation model right?",
    "Is the observation model right? Do people answer as it assumes? Are simpler methods enough? Does the system change what it measures?",
  ],
  zh: [
    "在得名之前，各个组成部分分属不同的社区。",
    "应当如何向人提问，又如何从回答中推断？",
    "采集函数是否有原则，是否有理论保证？",
    "仍在问：采集函数是否有原则、有保证？也越来越多地问：观测模型对吗？",
    "观测模型对吗？人是否按它假设的方式回答？更简单的方法是否已经足够？系统是否改变了它所测量的东西？",
  ],
};
function question(y: number, lang: Lang = "en"): string {
  const Q = QUESTIONS[lang];
  if (y < 2017) return Q[0];
  if (y <= 2021) return Q[1];
  if (y <= 2024) return Q[2];
  if (y === 2025) return Q[3];
  return Q[4];
}

const labels = {
  en: {
    q: "Central question then (inference): ",
    paper: "peer-reviewed or event",
    preprint: "preprint",
    software: "software release",
    describe: "A timeline of preferential Bayesian optimization from 2005 to 2026: {n} events in six lanes. Selected: {pick}, {venue}. {why}",
  },
  zh: {
    q: "当时的核心问题（推断）：",
    paper: "同行评审论文或事件",
    preprint: "预印本",
    software: "软件发布",
    describe: "偏好贝叶斯优化从 2005 至 2026 年的时间线：六条泳道中共 {n} 个事件。当前选中：{pick}，{venue}。{why}",
  },
};

const params = {
  event: {
    kind: "choice", label: { en: "Event", zh: "事件" }, control: "select", default: "gonzalez2017",
    options: EVENTS.map((e) => ({ value: e.id, label: { en: `${e.y} · ${e.pick.en}`, zh: `${e.y} · ${e.pick.zh}` } })),
  },
  hideLater: { kind: "toggle", label: { en: "Hide later years", zh: "隐藏之后的年份" }, default: false },
} as const;

type P = { event: string; hideLater: boolean };

const indexOf = (id: string) => Math.max(0, EVENTS.findIndex((e) => e.id === id));
const phaseOf = (y: number) => PHASES.findIndex((ph) => y >= ph.a && y <= ph.b);

// Greedy word wrap by an estimated character budget.
function wrap(s: string, max: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const word of s.split(/\s+/)) {
    if (line && (line + " " + word).length > max) { out.push(line); line = word; }
    else line = line ? `${line} ${word}` : word;
  }
  if (line) out.push(line);
  return out;
}

// Wrap Chinese text, which has no spaces between words, to lines of at most
// `max` Latin-letter widths: a Chinese character counts 1.9, a run of Latin
// letters, digits, and symbols is kept whole, and a line never starts with
// closing punctuation.
function wrapZh(s: string, max: number): string[] {
  const toks = s.match(/[　-〿一-鿿＀-￯]|[^\s　-〿一-鿿＀-￯]+|\s+/g) ?? [];
  const width = (t: string) => labelWidth(t, 1, 1.9);
  const lines: string[] = [];
  let line = "", w = 0;
  for (const t of toks) {
    const tw = width(t);
    if (line && w + tw > max && !/^[，。；：、）？！”]$/.test(t) && !/^\s+$/.test(t)) { lines.push(line.trimEnd()); line = ""; w = 0; }
    if (!line && /^\s+$/.test(t)) continue;
    line += t; w += tw;
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines;
}

const wrapIn = (lang: Lang, s: string, max: number) => (lang === "zh" ? wrapZh(s, max) : wrap(s, max));

interface Layout {
  narrow: boolean;
  left: number;
  right: number;
  preL: number;
  preR: number;
  x0: number;
  col: number;
  head: number;
  rowH: number;
  x(y: number): number;
  bandX(a: number, b: number): [number, number];
}

function layout(w: number): Layout {
  const narrow = w < 480;
  const left = narrow ? 8 : 138;
  const right = w - 10;
  const preW = narrow ? 44 : 62;
  const gap = narrow ? 10 : 14;
  const preL = left, preR = left + preW;
  const x0 = preR + gap;
  const col = (right - x0) / 10;
  const x = (y: number) => (y < 2017 ? preL + 6 + ((y - 2005) / 11) * (preW - 12) : x0 + (y - 2017 + 0.5) * col);
  const bandX = (a: number, b: number): [number, number] => (a < 2017 ? [preL, preR] : [x0 + (a - 2017) * col, x0 + (b - 2016) * col]);
  return { narrow, left, right, preL, preR, x0, col, head: narrow ? 22 : 38, rowH: narrow ? 32 : 30, x, bandX };
}

// Position of every event: the year column, spread within the column when a
// lane has several events in one year, staggered vertically when crowded.
function positions(L: Layout): Array<{ x: number; dy: number }> {
  const groups = new Map<string, number[]>();
  EVENTS.forEach((e, i) => {
    const k = `${e.y}|${e.lane}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(i);
  });
  const out: Array<{ x: number; dy: number }> = EVENTS.map(() => ({ x: 0, dy: 0 }));
  for (const idx of groups.values()) {
    const n = idx.length;
    const crowded = L.narrow ? n >= 3 : n >= 4;
    const step = n > 1 ? Math.min((L.col * 0.72) / (n - 1), L.narrow ? 9 : 13) : 0;
    idx.forEach((i, k) => {
      const e = EVENTS[i];
      out[i] = { x: L.x(e.y) + (n > 1 ? (k - (n - 1) / 2) * step : 0), dy: crowded ? (k % 2 ? 4 : -4) : 0 };
    });
  }
  return out;
}

function mark(kind: Kind, x: number, y: number, r: number, color: string, extra: Record<string, string | number> = {}): string {
  if (kind === "software") return el("rect", { x: x - r * 0.9, y: y - r * 0.9, width: r * 1.8, height: r * 1.8, rx: 1, fill: color, stroke: C.paper, "stroke-width": 1.2, ...extra });
  if (kind === "preprint") return el("circle", { cx: x, cy: y, r: r - 0.6, fill: C.paper, stroke: color, "stroke-width": 1.8, ...extra });
  return el("circle", { cx: x, cy: y, r, fill: color, stroke: C.paper, "stroke-width": 1.2, ...extra });
}

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const e = EVENTS[indexOf(st.p.event)];
  return tpl(labels[lang].describe, { n: EVENTS.length, pick: tr(e.pick, lang), venue: tr(e.venue, lang), why: tr(e.why, lang) });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const Lb = labels[lang];
  const L = layout(st.w);
  const sel = indexOf(p.event);
  const se = EVENTS[sel];
  const pos = positions(L);
  const parts: string[] = [];
  const lanesTop = L.head;
  const lanesBottom = lanesTop + LANES.length * L.rowH;
  const laneY = (i: number) => lanesTop + i * L.rowH + (L.narrow ? 21 : L.rowH / 2);
  const selPhase = phaseOf(se.y);

  // phase bands: the selected event's phase is shaded; boundaries are rules
  PHASES.forEach((ph, i) => {
    const [a, b] = L.bandX(ph.a, ph.b);
    if (i === selPhase) parts.push(el("rect", { x: a, y: 0, width: b - a, height: lanesBottom, fill: C.panel, rx: 3 }));
    const cx = (a + b) / 2;
    parts.push(text(cx, 13, L.narrow ? ph.short : ph.range, { "text-anchor": "middle", "font-size": TYPE.small, class: i === selPhase ? "fig-t-strong fig-t-num" : "fig-t-muted fig-t-num" }));
    if (!L.narrow) parts.push(text(cx, 28, tr(ph.name, lang), { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
    if (i > 1) parts.push(el("line", { x1: a, x2: a, y1: lanesTop - 2, y2: lanesBottom, stroke: C.rule, "stroke-dasharray": "2 3" }));
  });

  // lanes
  LANES.forEach((ln, i) => {
    const top = lanesTop + i * L.rowH;
    if (i > 0) parts.push(el("line", { x1: L.narrow ? L.left : 8, x2: L.right, y1: top, y2: top, stroke: C.grid }));
    if (L.narrow) {
      parts.push(el("circle", { cx: L.left + 3.5, cy: top + 7.5, r: 3, fill: ln.color }), text(L.left + 10, top + 11, tr(ln.label, lang), { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
    } else {
      parts.push(el("circle", { cx: 12, cy: laneY(i), r: 3.5, fill: ln.color }), text(20, laneY(i) + 4, tr(ln.label, lang), { "font-size": TYPE.small, class: "fig-t-muted" }));
    }
  });

  // guide at the selected event
  const sx = pos[sel].x;
  parts.push(el("line", { x1: sx, x2: sx, y1: lanesTop, y2: lanesBottom + 4, stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "3 3" }));

  // events
  const r = L.narrow ? 3.6 : 4.8;
  const hits: string[] = [];
  EVENTS.forEach((e, i) => {
    const li = LANES.findIndex((l) => l.id === e.lane);
    const cy = laneY(li) + pos[i].dy;
    const faded = p.hideLater && e.y > se.y;
    parts.push(mark(e.kind, pos[i].x, cy, r, LANES[li].color, faded ? { opacity: 0.14 } : {}));
    hits.push(el("circle", { cx: pos[i].x, cy, r: L.narrow ? 6 : 7, fill: "transparent", "data-fig-hit": `ev-${e.id}` }));
  });
  const sli = LANES.findIndex((l) => l.id === se.lane);
  parts.push(el("circle", { cx: sx, cy: laneY(sli) + pos[sel].dy, r: r + 4, fill: "none", stroke: C.ink, "stroke-width": 1.8 }));

  // axis
  const ay = lanesBottom + 4;
  parts.push(el("line", { x1: L.preL, x2: L.preR, y1: ay, y2: ay, stroke: C.rule }), el("line", { x1: L.x0, x2: L.right, y1: ay, y2: ay, stroke: C.rule }));
  // axis break between the compressed early years and the decade
  const bx = (L.preR + L.x0) / 2;
  parts.push(el("path", { d: `M${bx - 5},${ay + 4}L${bx - 1},${ay - 4}M${bx + 1},${ay + 4}L${bx + 5},${ay - 4}`, stroke: C.ink3, "stroke-width": 1.2, fill: "none" }));
  for (const y of [2005, 2010, 2015]) parts.push(el("line", { x1: L.x(y), x2: L.x(y), y1: ay, y2: ay + 4, stroke: C.rule }));
  parts.push(text(L.x(2005) - 4, ay + 16, "2005", { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  for (let y = 2017; y <= 2026; y++) {
    parts.push(el("line", { x1: L.x(y), x2: L.x(y), y1: ay, y2: ay + 4, stroke: C.rule }));
    if (!L.narrow || (y - 2017) % 2 === 0) parts.push(text(L.x(y), ay + 16, String(y), { "text-anchor": "middle", "font-size": TYPE.small, class: y === se.y ? "fig-t-strong fig-t-num" : "fig-t-muted fig-t-num" }));
  }

  // legend of marks
  const ly = ay + 36;
  const legendItems: Array<[Kind, string]> = [["paper", Lb.paper], ["preprint", Lb.preprint], ["software", Lb.software]];
  let lx = L.narrow ? L.left : 8;
  let lrow = ly;
  for (const [k, lab] of legendItems) {
    const wv = labelWidth(lab, 6) + 26;
    if (lx + wv > L.right && lx > (L.narrow ? L.left : 8)) { lx = L.narrow ? L.left : 8; lrow += 18; }
    parts.push(mark(k, lx + 5, lrow - 4, 4.4, C.ink2), text(lx + 14, lrow, lab, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += wv;
  }

  // detail panel
  const px = L.narrow ? L.left : 8;
  const pw = L.right - px;
  let py = lrow + 16;
  parts.push(el("line", { x1: px, x2: L.right, y1: py, y2: py, stroke: C.rule }));
  py += 18;
  const lane = LANES[sli];
  parts.push(mark(se.kind, px + 5, py - 4, 4.4, lane.color), text(px + 15, py, `${tr(lane.label, lang)} · ${tr(se.venue, lang)}`, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const perChar = 6.5;
  const titleLines = wrapIn(lang, tr(se.title, lang), Math.floor(pw / (perChar + 0.4)));
  for (const t of titleLines) { py += 18; parts.push(text(px, py, t, { "font-size": TYPE.label, class: "fig-t-strong" })); }
  py += 4;
  for (const t of wrapIn(lang, tr(se.why, lang), Math.floor(pw / perChar))) { py += 17; parts.push(text(px, py, t, { "font-size": TYPE.body })); }
  py += 4;
  for (const t of wrapIn(lang, Lb.q + question(se.y, lang), Math.floor(pw / 6.0))) { py += 16; parts.push(text(px, py, t, { "font-size": TYPE.small, class: "fig-t-muted" })); }

  return svg(st.w, py + 10, describe(st), g({}, ...parts), g({}, ...hits));
}

const step = (p: P, d: number): P => ({ ...p, event: EVENTS[Math.min(EVENTS.length - 1, Math.max(0, indexOf(p.event) + d))].id });

export default defineFigure({
  name: "hist-timeline",
  title: { en: "A timeline of preferential Bayesian optimization, 2005 to 2026", zh: "偏好贝叶斯优化时间线，2005 至 2026 年" },
  labels,
  params,
  hint: { en: "Click an event or pick one from the list. With Hide later years on, Next replays the decade one event at a time.", zh: "点击一个事件，或从列表中选择。打开“隐藏之后的年份”后，按“下一个”会逐个事件重演这十年。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || !e.target?.startsWith("ev-")) return null;
    const id = e.target.slice(3);
    return EVENTS.some((x) => x.id === id) ? { ...p, event: id } : null;
  },
  actions: [
    { label: { en: "Previous", zh: "上一个" }, run: (p) => step(p as P, -1), enabled: (p) => indexOf(p.event) > 0 },
    { label: { en: "Next", zh: "下一个" }, primary: true, run: (p) => step(p as P, 1), enabled: (p) => indexOf(p.event) < EVENTS.length - 1 },
  ],
});
