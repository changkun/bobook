---
status: done
synopsis: "偏好贝叶斯优化仍在维护的软件及其默认设置；研究代码为何难以重新运行；方法如何借助模拟用户评测，度量的选择为何决定胜负；换成真人回答时有何变化；以及这一领域由谁研究、分布在哪些学科、研究数量有多少。"
---

# 软件、评测方法与研究社区 {#sec-software-evaluation}

前几章介绍了偏好贝叶斯优化各种方法的主张。本章转而讨论这些主张得出与使用的条件：实现这些方法的软件、软件替用户选定的默认设置、论文对不同方法的比较评测方式，以及从事这些工作的人。

截至 2026 年 9 月的状况可分三方面概括。第一，偏好贝叶斯优化只有一套仍在维护的通用软件栈，即 Meta 的 BoTorch 与 Ax，其余大多是随单篇论文发布的代码。第二，评测几乎完全依赖模拟用户，由其针对从标量优化沿用而来的合成测试函数回答比较。第三，研究社区规模很小，分散于机器学习、控制、机器人学和人机交互；这些领域在不同的会议和期刊上发表成果，在我们核查过的交叉引用中，彼此很少引用。

## 软件 {#sec-sw-software}

如今想运行偏好贝叶斯优化的读者需要选择软件，而可选范围比贝叶斯优化生态的规模所显示的要窄。下文的日期取自各项目的更新日志，即项目逐版本列出改动内容的文件；没有相应条目时，取 Python 软件包索引（PyPI）上的上传日期。两者可能相差一天。

### BoTorch {#sec-sw-botorch}

BoTorch 是 Meta 基于 PyTorch 构建的高斯过程模型与采集函数库，其偏好模型 `PairwiseGP` 已在 @sec-pairwisegp 与 @sec-obs-pairwisegp 中介绍。偏好功能于 2020 至 2024 年间陆续加入 [@botorch2026changelog]：

- 0.2.3（2020 年 4 月 27 日）加入了用于成对比较数据的 `PairwiseGP`。
- 0.3.2（2020 年 10 月）去掉了其噪声项，并默认加入 `ScaleKernel`。
- 0.6.3（2022 年 3 月 28 日）加入了 `LearnedObjective`、EUBO（最优选项期望效用，@sec-eubo）的解析形式，以及偏好探索贝叶斯优化（BOPE）的教程；在这一设定中，人针对实验结果而非设计表达偏好。
- 0.6.5（2022 年 7 月）加入了逻辑链接似然，并将偏好贝叶斯优化教程改用 EUBO。
- 0.9.0（2023 年 8 月 1 日）加入了成对 BALD（贝叶斯分歧主动学习，选择模型的各种合理假设对其答案分歧最大的查询）。
- 0.10.0（2024 年 2 月 26 日）加入了 qEUBO，即适用于含 $q$ 个选项的查询的 EUBO。

此后偏好代码只有修复：0.18.0（2026 年 6 月 3 日）修正了 `PairwiseGP` 管理存储状态的方式，并提高了 BALD 中混合熵的数值稳定性。最新版本为 0.18.1（2026 年 6 月 8 日），主分支的最后一次提交在 2026 年 9 月 30 日 [@botorch2026pypi]。偏好采集函数全部位于同一个源文件中 [@botorch2026preference]。

大多数新手运行的第一段代码是 BoTorch 的偏好贝叶斯优化教程，因此有必要了解它的设计。教程使用 4 维线性效用，在比较之前给效用加上标准差为 0.1 的高斯噪声，用 Kendall 秩相关评价模型（即两个排序中顺序一致的点对所占比例，重新缩放到 $-1$ 至 $1$ 之间），并且只用 3 次重复比较 EUBO 与随机查询 [@botorch2026pairwise]。

### Ax {#sec-sw-ax}

Ax 是 Meta 的实验管理平台，其模型由 BoTorch 提供。成对模型桥接 `PairwiseModelBridge` 最早见于 Ax 0.2.6（2022 年 8 月 17 日）的安装包 [@ax2022platform]。Ax 1.0.0 于 2025 年 5 月 8 日上传到 PyPI，配套论文为 @olson2025ax（AutoML 2025）。Ax 的偏好功能于 2026 年加入 [@ax2026changelog]：

- 1.2.2（2026 年 1 月；PyPI 上没有该版本的文件）加入了支持存储的 `PreferenceOptimizationConfig` 和 Kendall 秩相关诊断。
- 1.2.3（2026 年 2 月 19 日）加入了基于 `PairwiseGP` 的 BOPE 效用跟踪，以及面向语言模型的 `LLMProvider` 与 `LLMMessage` 抽象。
- 1.3.0（2026 年 6 月 4 日）加入了带标签新鲜度检查的 LILO 标注试验、偏好贝叶斯优化中 qEUBO 的自动分派、`ModelList` 中的 `PairwiseGP` 和效用排序图。语言在回路优化（LILO）借助语言模型把自由文本反馈转化为成对标签（@sec-llm-in-loop）。
- 1.3.1（2026 年 6 月 9 日）是最新版本，要求 Python 3.11 或更高版本 [@ax2026platform]。

Ax 的更新日志始于 1.1.0（2025 年 8 月），更早的历史依据安装包整理。主分支（最后一次提交在 2026 年 9 月 29 日）没有关于偏好、BOPE 或 LILO 的教程 [@ax2026tutorials]，LILO 的逻辑只体现在代码中 [@ax2026transition]。

### 所有支持偏好的软件包 {#sec-sw-packages}

@tbl-sw-packages 列出了我们找到的支持偏好的软件，各名称所附链接指向代码仓库或软件文档。HB-EI 与 HB-UCB 指 @sec-choosing-pairs 中的幻觉信念分别配合期望改进与上置信界。

::: {.table #tbl-sw-packages title="支持偏好的软件及其维护状态（截至 2026 年 9 月底）。"}
| 软件 | 代理模型与推断 | 采集函数 | 偏好功能 | 最新版本 | 状态 |
|---|---|---|---|---|---|
| BoTorch [@botorch2026license] | `PairwiseGP`：概率单位或逻辑似然，Laplace 近似，用 Laplace 边际似然确定超参数 | EUBO、qEUBO、成对 BALD、成对后验方差 | 成对数据模型、学习得到的目标函数、PBO 与 BOPE 教程 | 0.18.1（2026-06-08） | 活跃；偏好模块自 2024 年 2 月以来只有修复 |
| Ax | 调用 BoTorch 的 `PairwiseGP` | 自动分派 qEUBO（从 1.3.0 起） | 偏好优化配置、BOPE 效用跟踪、LILO 标注试验、效用排序图、Kendall 秩相关诊断 | 1.3.1（2026-06-09） | 活跃；没有偏好教程 |
| AEPsych [@meta2026aepsych] | `PairwiseProbitModel`，继承自 `PairwiseGP` | 我们没有核查 | 成对心理物理学实验 | 0.8.0（2025-04-11）；主分支为 0.8.0+dev | 2026-08-26（UTC）有提交 |
| optuna-dashboard [@optuna2026dashboard] | 依照 Takeno 等 2023：用期望传播拟合核函数超参数，对潜在效用做 Gibbs 采样 | 在采样得到的高斯过程上最大化对数期望改进 | 网页界面；用户从若干候选（教程中为 4 个）中标出最差的一个，系统记录由此蕴含的所有点对；自 0.13.0b1（2023 年 9 月）起可用 | 0.21.0（2026-09-10） | 活跃；不支持动态搜索空间 |
| OptunaHub plmbo [@ozaki2026plmbo] | 每个目标各用一个采用径向基函数核的 GPy 高斯过程；偏好权重用 NumPyro 的 Markov 链蒙特卡洛估计 | 我们没有核查 | 实现 @ozaki2024multi（AAAI 2024）中带主动偏好学习的多目标贝叶斯优化，比较结果在控制台中输入 | optunahub 0.5.0（2026-09-09）；插件声明 Optuna 3.6.1 | OptunaHub 没有纯偏好采样器 |
| pySequentialLineSearch [@koyama2025sequential]（C++，使用 Eigen 与 NLopt；实验性的 Python 绑定） | `PreferenceRegressor`：Bradley-Terry-Luce 似然，Matérn 5/2 核，最大后验超参数 | 期望改进（默认）、高斯过程上置信界 | 基于滑块的线搜索；成对比较演示 | 标签 v0.5（2025-11-11），`setup.py` 仍写 0.4；未发布到 PyPI | 低频维护；只在 macOS 10.15 上测试过 |
| GLIS、GLISp、C-GLISp [@bemporad2023glis]（Python 与 MATLAB） | 径向基函数代理模型（默认为逆二次型），配合逆距离加权；非概率模型 | 代理模型减去一个逆距离加权方差项和一个探索项，用粒子群优化求解 | 偏好取值于 $\{-1, 0, 1\}$（允许平局）；C-GLISp 增加了未知约束和“满意”标签 | glis 2.0.2（2023-06-19） | 2023-03-03 之后没有提交；混合变量的后继版本 PWAS/PWASp [@zhu2025pwas]，最后一次提交在 2025-01-23 |
| prefGP [@benavoli2026prefgp]（JAX 与 PyTorch） | 9 种偏好与选择似然；Laplace 近似、变分推断（完全与稀疏）、切片采样 | 无 | 只有偏好学习，没有优化循环 | 没有发布标签；未发布到 PyPI | 最后一次提交在 2026-07-14；锁定 botorch 0.9.3、gpytorch 1.11、jax 0.4.23、torch 2.1.2 |
| SkewGP [@benavoli2025skewgp] | 闭式偏斜高斯过程后验，可选 Laplace 近似 | Thompson 采样、上置信界、一种信息增益规则 | 偏好数据与混合数据 | 没有标签 | 最后一次提交在 2025-08-05（仅 README） |
| preferentialBO [@lab2023preferentialbo]（Python 3.9，GPy 1.10.0） | Gibbs 采样的偏好高斯过程；期望传播、Laplace 近似与 MCMC 基线 | HB-EI、HB-UCB 与论文中的 6 种基线规则 | Takeno 等（ICML 2023）的复现代码 | 没有标签 | 2023-07-26 之后没有提交 |
| qEUBO 论文代码 [@research2023qeubo; @astudillo2023qeuboc] | 论文中为带诱导点的变分高斯过程 | qEUBO 与基线 | 官方仓库有 5 个任务；作者个人的仓库有更多任务，包括 Animation 数据 | 没有标签 | 2023 年 3 月之后没有提交；没有依赖文件 |
| PABBO [@zhang2026pabbo] | 用强化学习预训练的 Transformer 策略 | 策略直接输出查询点对 | 摊销偏好优化 | 没有标签 | 最后一次提交在 2026-09-14；锁定 botorch 0.10.0 与 torch 的 CUDA 11.8 构建；需要 Weights and Biases 密钥 |
| Emukit 示例 [@emukit2026preferential]（GPy、Stan） | 期望传播、变分推断、MCMC | 批量比较规则 | 批量 PBO，只在 examples 目录中 | emukit 0.5.1（2026-02-22） | 示例无人维护；仍要求 Python 3.7、scipy 1.1.0 和 pystan |
:::

另有几种方法仅以单篇论文的代码形式存在：PPBO [@pml2022ppbo]（MIT，最后一次提交在 2022 年 9 月 13 日）；POLAR [@tucker2024polar]（MATLAB，BSD-3-Clause，最后一次提交在 2024 年 6 月 17 日，支持成对反馈、共同主动反馈与序数反馈）；LILO 的研究代码 [@research2026lilo]（MIT，最后一次提交在 2026 年 5 月 12 日）；DT-PBO，即以决策树为代理模型的偏好贝叶斯优化 [@thomasq992026dt]（MIT）。POP-BO [@predictepfl2024pop]、MR-LPF [@kayal2025bohf] 与 CrashPBO [@dsme2026crashpbo] 三个仓库没有许可证文件，因此适用默认的版权规定，未经许可不得复用。

### 不支持偏好的库 {#sec-sw-no-preference}

主要的贝叶斯优化与高斯过程库都没有偏好似然或偏好采集函数。2026 年 9 月底，我们在以下各库最新版本的源代码树中检索了“pairwise”“preference”与“dueling”三个词：GPyTorch 1.15.2 [@gpytorch2026pypi]、GPflow 2.11.1 [@gpflow2026pypi]、Trieste 4.6.0 [@labs2026trieste]、SMAC 2.4.1 [@automlorg2026smac]、HEBO 0.3.6 [@lab2024hebo]、Dragonfly 0.1.7 [@dragonfly2022opt]、GPyOpt 1.2.6 [@sheffieldml2023gpyopt] 与 Optuna 5.0.0 [@optuna2026pypi]。`PairwiseGP` 使用 GPyTorch 的核函数，但 Laplace 推断由其自行实现；“dueling”在 HEBO 中只出现于一个无关的强化学习子项目；Optuna 的偏好支持位于 optuna-dashboard 和 OptunaHub 中。GPyOpt 已于 2023 年 1 月 17 日归档，Dragonfly 实际上已无人维护。

### 软件格局的含义 {#sec-sw-landscape}

由此可得出三点判断（推断）。第一，实际可用的偏好贝叶斯优化依赖于一家机构：BoTorch 提供模型与采集函数，Ax 管理实验、BOPE 与 LILO，AEPsych 服务于心理物理学；自 2024 年 2 月加入 qEUBO 以来，新的偏好功能都落在 Ax 而非 BoTorch 中。第二，高斯过程的替代方案（GLISp、DT-PBO、PABBO）都很小众，其中只有 GLIS 能用 `pip` 安装。第三，在实践者可用的软件中，Ax 1.3.x 功能最完整且仍在维护，但这些功能缺少文档；我们没有核实 Ax 是否允许用户通过模型配置向 `PairwiseGP` 传入维度缩放先验。

## 忽视维度的默认先验 {#sec-sw-priors}

默认先验是大多数用户从未看到的建模决策。@sec-obs-pairwisegp 解读了 `PairwiseGP` 内部的各项选择（列于 @tbl-pref-botorch），@sec-hd-diagnosis 解释了固定尺度的长度尺度先验为何会随输入个数增加而失效。@tbl-sw-priors 汇总了我们考察过的各偏好软件包的默认设置，并以 BoTorch 的标量模型作为对照。除对照行外，没有一行随维度缩放。

::: {.table #tbl-sw-priors title="偏好软件与论文中的默认长度尺度设置。只有作为对照列入的 BoTorch 标量模型随维度缩放。"}
| 软件或论文 | 核函数 | 默认长度尺度设置 | 众数或初始值 | 是否随维度缩放 |
|---|---|---|---|---|
| BoTorch `PairwiseGP` [@botorch2026pairwisegp] | 径向基函数核，外层为必需的 `ScaleKernel` | Gamma(2.4, 2.7)，初始化在其众数处，下界 $10^{-4}$ | 约 0.52 | 否 |
| BoTorch 自 0.12.0 起的其他模型（对照）[@botorch2026gpytorch] | 取决于模型 | 对数正态，位置参数 $\sqrt{2} + \tfrac12\log d$，尺度参数 $\sqrt{3}$，下界 0.025 | $d = 10$ 时约 0.65，$d = 100$ 时约 2.05（推断） | 是 |
| optuna-dashboard 偏好采样器 [@optuna2026sampler] | Matérn 3/2，每个输入一个长度尺度 | Gamma(5, 10)；噪声先验 Gamma(5, 50) | 0.4 | 否 |
| sequential-line-search [@koyama2025preference] | Matérn 5/2，每个输入一个长度尺度 | 默认 0.5，最大后验先验，方差 0.25 | 0.5 | 否 |
| Koyama 等 2017 与 2020 年的论文 [@koyama2017sequentialb; @koyama2020sequential] | 2020 年论文中为 Matérn 5/2 | 对数正态，参数 $\mu = 0.5$、$\sigma^2 = 0.01$，几乎固定 | 约 0.5 | 否 |
| PABBO 的合成先验（训练与评估中相同）[@zhang2025pabbob; @zhang2025pabboc] | 径向基函数核与 Matérn 5/2、3/2、1/2 | 截断到 $[0.05, 2]$，中心在 1/3 附近 | 不适用 | 否 |
:::

没有一个偏好贝叶斯优化软件包说明其默认先验与维度的关系，因此凡在约 10 维以上使用这些软件包的人，都继承了标量贝叶斯优化在 2024 年已经放弃的固定尺度先验（推断；后果见 @fig-hd-lengthscale，应改传的设置见 @sec-hd-practice）。先验是否应当利用其他用户的数据（即群体先验，见 @sec-hci-population），是另一个问题：@tbl-sw-priors 中的默认设置都是弱先验，但维度一旦增大，它们便是尺度不当的弱先验（推断）。

## 复现研究代码 {#sec-sw-reproducibility}

无法重新运行的方法无从检验；无法重新运行的基线只能重新实现，而每次实现都略有差异。三个习惯能使研究代码可以重新运行：**发布标签**（release tag），即仓库中有名称的冻结版本；**依赖文件**（dependency file），列出代码所需的确切库版本；**许可证**（license），说明谁可以复用代码。

2026 年 9 月底，我们检查了 11 个研究仓库是否有发布标签，即 prefGP、SkewGP、preferentialBO、官方的 qEUBO 仓库、PABBO、PPBO、POLAR、GLIS、POP-BO、LILO 与 BO_toolbox，**结果 11 个都没有。** 依赖通常锁定在 2020 至 2024 年的版本（@tbl-sw-packages 列出了其中几个）：PPBO 锁定了 numpy 1.18.4 和已归档的 GPyOpt 1.2.6 [@pml2022ppbo]；两个 qEUBO 仓库都没有依赖文件 [@research2023qeubo; @astudillo2023qeuboc]。preferentialBO 的脚本调用了 NumPy 1.24 已移除的 `np.int`，因此只能在锁定的 numpy 1.23.2 下运行；其 README 还提示，在 Ubuntu 20.04 上尚未确认能完全复现论文结果（实验在 CentOS 6.9 上运行）[@lab2023preferentialbo]。

许可证也不统一。大多数仓库使用 MIT、BSD 或 Apache 许可证，但 PABBO 使用 AGPL-3.0，这是一种**著佐权**（copyleft）许可证，要求衍生软件（包括通过网络提供的软件）以相同许可证发布；AEPsych 的 Creative Commons Attribution-NonCommercial 4.0 许可证限制商业复用 [@meta2026aepsych]；另有三个 2024 至 2026 年的仓库根本没有许可证文件。每种方法还各带一套高斯过程实现（GPy、BoTorch 0.9 或 0.10、JAX、MATLAB），因此大多数论文都要重新实现基线；同一基线在一篇论文中显得很强、在另一篇中却很弱，这可能是原因之一（推断）。

::: {.pitfall title="运行偏好贝叶斯优化的研究代码"}
应做好锁定旧版本的准备。没有标签的仓库随时可能变动，没有依赖文件的仓库则只能靠猜测确定版本：应记下提交哈希，以及代码成功运行时所用的各库版本。在代码基础上开发之前，先检查许可证；上述仓库中有三个没有许可证，一个采用著佐权许可证。论文中的基线若是重新实现的，基线的设置便是结果的一部分（推断）。
:::

## 方法如何评测 {#sec-sw-evaluation}

比较优化器比表面上更难。单次运行取决于随机的初始设计和回答中的噪声，一种方法可能凭运气赢下一次运行，因此比较需要在多次运行上取平均（@sec-bo-loop）。遗憾界（@sec-regret）对一类函数成立，而非针对手头的具体函数，因此只有基准测试能说明各方法在实践中的排名，且受本节所述的种种限制。

### 沿用的测试函数与模拟用户 {#sec-sw-test-functions}

自 @gonzalez2017preferentialb 以来，几乎每篇偏好贝叶斯优化论文都用**模拟用户**（simulated user）评测方法。模拟用户是一个程序：对每次比较，它在两个选项处计算某个已知函数的值，加上噪声后据此作答。这些函数是标准的合成测试函数，通常为 1 至 8 维，由 González 等人从 Surjanovic 与 Bingham 维护的标量测试函数库中选取，后来的论文一直沿用。这些函数原本用于测试标量优化器，不具备 @sec-part-preferences 与 @sec-part-perspectives 所描述的任何人类效用性质，例如无差异阈值、漂移、不可传递性和锚定（推断）。@tbl-sw-evals 列出了代表性论文的测试问题与度量；其中大多数论文的噪声、预算与结论见 @tbl-acqf-studies；下文的列表汇总了噪声模型与度量。

::: {.table #tbl-sw-evals title="代表性偏好贝叶斯优化论文的评测方式：测试问题与主要度量。"}
| 论文 | 测试问题（维度） | 主要度量 |
|---|---|---|
| @gonzalez2017preferentialb，ICML 2017 | Forrester（1）；Six-hump camel、Gold-Stein、Levy（2）；每个维度 33 个点的网格；5 次初始对决，总预算 200 次对决 | 当前 Condorcet 赢家处的真实值 |
| @mikkola2020projective，ICML 2020 | Six-hump camel（2）、Hartmann（6）、Levy（10）、Ackley（20）；100 次查询 | 后验均值最大点处的真实目标函数值 |
| @siivola2021preferential，MLSP 2021 | SigOpt 库中的若干函数；4 个真实数据集，包括 Sushi 与 Candy（$d \le 4$） | 我们没有提取 |
| @fauvel2021efficient，arXiv 2021（预印本） | Surjanovic 与 Bingham 函数库中的 34 个函数，标准化为均值 0、方差 1 | 最终的最优值；Mann-Whitney U 检验与 Borda 得分 |
| BOPE：@lin2022preferenceb，AISTATS 2022 | 车辆安全问题（$d = 5$，$k = 3$ 个结果）、DTLZ2、OSY（$d = 6$，$k = 8$）、汽车驾驶室设计问题（$d = 7$，$k = 9$），配合若干效用 | 后验均值最大点处的真实效用 |
| qEUBO：@astudillo2023qeubob，AISTATS 2023 | Ackley（6）、Alpine1（7）、Hartmann（6）、汽车驾驶室设计问题（7）、Sushi（4）、Animation（5）；$4d$ 次初始查询，然后 150 次 | 后验均值最大点处的对数简单遗憾 |
| @takeno2023practicalc，ICML 2023 | 12 个函数（正文中 8 个，最高为 6 维 Hartmann） | 推荐点处的遗憾 |
| POP-BO：@xu2024principledb，ICML 2024 | 高斯过程样本、包括 6 维 Ackley 在内的标准测试函数、一个热舒适问题 | 累积遗憾与所报告解的次优程度 |
| MaxMinLCB：@pasztor2024bandits，NeurIPS 2024 | Ackley、Eggholder 等；Yelp 餐厅数据 | 以偏好概率计的累积遗憾 |
| PABBO：@zhang2025pabbob，ICLR 2025 | 高斯过程样本（1、2）、Forrester、Beale、Branin；附录中的 Ackley（6）与 Hartmann（6）；HPO-B、Candy、Sushi | 最佳查询点的简单遗憾 |
| MR-LPF：@kayal2025bayesian，ICML 2025 | 再生核 Hilbert 空间中的样本、Ackley（1）、Yelp（275 家餐厅，20 位用户） | 以偏好概率计的累积遗憾 |
| PF-TS：@lazzaro2026finiteb，AISTATS 2026 | Ackley（1），$T = 300$；三种金属的 63 种催化剂组成对应的产氢量，$T = 800$ | 以偏好概率计的累积遗憾 |
:::

表中，MaxMinLCB 是 Pásztor 等人的最大最小下置信界算法，PF-TS 是偏好反馈下的 Thompson 采样，HPO-B 是由 @sec-cs-classifier 中那类超参数优化任务构建的基准。2026 年的论文沿袭了同样的做法，例如预印本 KappaSharp 使用了 5 至 20 维的 11 个基准 [@shao2026adaptive]。

### 各论文自选的噪声模型 {#sec-sw-noise-models}

模拟用户如何出错，由每篇论文自行建模。我们至少找到了六类模型：

1. 在比较之前给效用加上高斯噪声，标准差为 0.01（Takeno 等人）、0.05（Siivola 等人）、0.1（BoTorch 教程）或函数值域的 10%（局部偏好贝叶斯优化 [@menn2026local]）。
2. 经过校准的逻辑噪声，使最好的 1% 的点中随机点对的回答出错率为 10%、20% 或 30%（qEUBO，见其附录中的研究）；对 6 维 Ackley 函数，三者分别对应逻辑尺度 0.0575、0.1416 与 0.2943 [@astudillo2023noise]。
3. 以固定概率翻转回答（BOPE 中为 10%）。
4. 默认没有噪声（PABBO）。
5. 标准化函数上的概率单位噪声（Fauvel 与 Chalk）。
6. 由语言模型模拟的决策者（LILO [@kobalczyk2026lilo]）。

按错误率校准比直接设定尺度更接近现实，因为它确定了模拟用户在关键比较上出错的频率；但噪声仍是同方差的（处处相同），且各次回答相互独立（推断）。两个模型即使在某一效用差上匹配到相同的错误率，在其他所有效用差上也都不一致，如 @fig-sw-noise 所示。

```{figure}
//| figure: sw-noise
//| label: fig-sw-noise
//| fig-cap: "模拟用户出错的三种方式，均匹配为效用差为 0.1 的比较有 10% 的时间回答错误：一是以固定概率翻转回答，即 BOPE 所用的方式；二是效用差上的逻辑噪声（Bradley-Terry 链接），即 qEUBO 所校准的方式；三是每个效用上的高斯噪声（概率单位链接），即 Takeno 等人、Siivola 等人和 BoTorch 教程所用的方式。纵轴取对数刻度，下限为 0.0001%。滑块调节匹配处的效用差与错误率，以及读数处的效用差（橙色线）。这些曲线以这组参数示意三种模型，并非任何一篇论文的设置，各论文的效用尺度互不相同；PABBO 默认无噪声，会落在图的下方。"
```

可以尝试以下几点：

- 在默认设置下，读取三条曲线在效用差 0.30 处的值。翻转模型仍有 10% 的时间出错，逻辑噪声为 0.14%，概率单位噪声为 0.006%：同样称为“10% 的噪声”，明显更差的选项获胜的可能性，在翻转模型下约为逻辑噪声下的 70 倍，为概率单位噪声下的一千多倍（概率单位的情形见 @exr-sw-flip 的手工计算）。
- 将读数处的效用差移到 0.05，此时的比较比匹配处更难分辨。逻辑噪声和概率单位噪声分别有 25% 和 26% 的时间出错，翻转模型仍为 10%：翻转模型对明确的选择更苛刻，对难分的选择更宽容。
- 将匹配处的错误率提高到 30%。此时在效用差 0.30 处，逻辑用户和概率单位用户分别有 7.3% 和 5.8% 的时间出错，彼此接近，翻转模型则出错 30%。回答噪声很大时，两种效用噪声模型几乎一致，与二者明显不同的是翻转模型。

### 遗憾的五种定义 {#sec-sw-regret}

**遗憾**（regret）是方法的结果与可达到的最佳值之间的差距（@sec-regret-definitions）。在偏好贝叶斯优化中，方法始终看不到效用值，因此论文必须规定把哪个点算作“结果”，而各论文的规定各不相同：

- 当前 **Condorcet 赢家**处的函数值，即获胜最多的点（González 等人）；
- **后验均值最大点处的简单遗憾**，该点即模型会推荐的点（qEUBO、BOPE；Takeno 等人采用其自己的推荐点）；
- **目前为止查询过的最佳点**的简单遗憾（PABBO）；
- 对每次比较的两个点求和、以偏好概率计的**累积遗憾**（MaxMinLCB、MR-LPF、PF-TS）；
- 以效用计的累积遗憾（POP-BO）。

论文还报告模型排序的质量（BoTorch 教程和 Ax 中采用 Kendall 秩相关）、每次迭代的计算时间（qEUBO、PABBO、Takeno 等人），以及一项用户实验中到达最近局部最小值所需的模拟步数（Mikkola 等人）。没有论文把达到某一阈值所需的比较次数作为主要度量，而这恰恰是回路中的人所付出的代价。

度量的选择会改变方法的排名。在 POP-BO 的论文中，在高斯过程样本实例上，qEUBO 报告的解略优于 POP-BO，累积遗憾却是 POP-BO 的 2.5 倍以上；在 6 维 Ackley 函数上，两种度量的结论又反了过来 [@xu2024principledb]。在最优点附近查询的方法，在以查询点计的遗憾上占优；模型校准良好的方法，在以后验均值计的遗憾上占优（推断）。读者可在 @fig-sw-metrics 中直接观察这一现象。

```{figure}
//| figure: sw-metrics
//| label: fig-sw-metrics
//| fig-cap: "同一组模拟运行，三种裁决。在贯穿全书的示例目标函数上，三种选择下一次对决的方式面对同一个模拟用户：EUBO；一个简单的当前最优点对挑战者规则（以模型当前的最佳猜测，对上乐观估计最高的点，乐观估计即后验均值加两个标准差），仅作示意，并非已发表的方法；随机点对。每次运行从一次随机对决开始，再提出 15 次对决。曲线为各次运行的均值，带宽为 ±1 个标准误。**推荐点**指后验均值最高的点的遗憾；**最佳已查询点**指目前为止展示给用户的最佳点的遗憾；**累积**指将每次对决中两个选项的平均遗憾逐次累加。设置为示意性的：一个输入、41 个候选点、长度尺度为 0.1 的径向基函数核、噪声为 0.1 的概率单位似然、在整个网格上做 Laplace 近似，模拟回答中加入所选大小的 Thurstone 噪声。这一逆转展示的是 POP-BO 观察结果背后的机制，而非对其的复现。"
```

可以尝试以下几点：

- 默认设置下（30 次运行，噪声 0.1），按**推荐点**度量，EUBO 排第一，当前最优点规则紧随其后，两者的带相互重叠，随机点对排在最后。
- 切换到**最佳已查询点**。此时随机点对获胜，遗憾接近 0.05，靠的是覆盖而非学习：16 个随机点对从仅 41 个候选点中抽取 32 次，平均约有 23 个候选点得到展示（@exr-sw-coverage）。在更大的空间或更高维度中，同一度量不会如此偏向随机点对（推断）。
- 切换到**累积**。当前最优点规则明显获胜：它提出的每一对都包含模型当前的最佳猜测，而该猜测很快接近最优点，因此每次比较有一半几乎没有代价（@exr-sw-incumbent）。
- 将**运行次数**设为 5，并多次按下**再运行一批**。按**推荐点**度量，赢家随批次而变化，主要在 EUBO 与当前最优点规则之间交替：重复次数少时，排名在一定程度上取决于运气。
- 改变噪声。噪声为 0.2 时，按**推荐点**度量，随机点对超过了当前最优点规则，另外两种度量下的顺序则保持不变。

### 统计与差异的大小 {#sec-sw-statistics}

统计报告普遍薄弱。各论文的重复次数从 3 次（BoTorch 教程）、10 次（Takeno 等人）、20 次、30 次直到 50 至 100 次（qEUBO）不等；不确定性的表示方式有 ±1、±1.96 或 ±2 个标准误、95% 置信区间或四分位距。只有 Fauvel 与 Chalk 对大量假设做了形式化检验：他们在 34 个函数上用 Mann-Whitney U 检验（一种不对分布作假设的检验，检验一种方法的结果是否倾向于大于另一种）比较每一对方法，并把胜场汇总为 Borda 得分（每胜过一种方法得一分）[@fauvel2021efficient]。后来的论文都没有采用这种做法。在确实测量了差异的研究中，复杂采集函数与随机查询之间的差距往往很小；相关证据以及现有比较为何难以相互印证，见 @sec-acqf-empirical。实践上的启示是：运行至少重复几十次，纳入随机查询，并检验主张所依赖的差异（推断）。

## 真实数据任务与缺失的数据集 {#sec-sw-datasets}

论文中所谓的“真实数据”任务，都把一群人的判断归并为单一的确定性效用。@tbl-sw-datasets 列出了目前使用的四个任务。

::: {.table #tbl-sw-datasets title="偏好贝叶斯优化评测中的真实数据任务，及其转化为确定性测试函数的方式。"}
| 任务 | 来源 | 内容 | PBO 论文如何使用 |
|---|---|---|---|
| Sushi | Kamishima 的寿司偏好数据 [@kamishima2026sushi] | 人对寿司种类的排序 | Siivola 等：100 种寿司的完整排序，带 4 个连续特征；PABBO：对用户取平均的五分制评分 |
| Candy | FiveThirtyEight 的在线成对投票 [@fivethirtyeight2017candy] | 2 个特征：糖分百分位与价格百分位 | 两篇论文都把投票转化为一个完整排序或胜率；Siivola 等统计为 86 种糖果，PABBO 为 85 种 |
| Yelp | 餐厅评分 | 275 家餐厅、20 位用户、32 维嵌入 | MR-LPF 与 MaxMinLCB；由评分而非比较构建 |
| Animation | qEUBO [@astudillo2023qeubob] | 来自 AEPsych 演示的类火焰粒子效果，有 5 个参数 | 作者“从人类用户那里收集了 100 次这样的成对比较”，据此拟合模型，并将其用作真值测试函数 |
:::

这些任务有利于找到群体最优点的方法，却掩盖了个人判断的不一致，而处理这种不一致正是偏好贝叶斯优化存在的意义（推断）。我们没有找到偏好贝叶斯优化的共享基准套件或排行榜，也没有找到专为评测偏好贝叶斯优化而设计的个体层面成对判断公开数据集。有了这样一个包含时间戳、呈现顺序、反应时和重复配对的数据集，各种方法便可在真实回答上做离线比较（@sec-rec-evaluation）。

## 复现中发现的问题 {#sec-sw-reproduction}

将论文与其代码对照检查，发现了以下问题。

- **qEUBO 的 Animation 任务。** 论文称拟合的模型为支持向量机 [@astudillo2023qeubob]。作者个人仓库中的真值有两个版本：`animation_runner.py` 使用支持向量机分类器，`animation2_runner.py` 则载入一个已拟合的 `PairwiseGP` [@astudillo2023qeuboc]。论文引用的官方仓库中没有 Animation 任务的运行脚本或数据 [@research2023qeubo]，两个仓库也都没有列出依赖。
- **BOPE 的维度。** 正文写作“DTLZ2 (d = 4, k = 8)”，一幅图的标注却是“DTLZ2 (d=8, k=4)”[@lin2022preferenceb]。
- **Candy 的规模。** 两篇论文分别报告 86 种和 85 种糖果 [@siivola2021preferential; @zhang2025pabbob]；FiveThirtyEight 的数据文件有 85 行 [@fivethirtyeight2017candy]。
- **Takeno 等人的代码**在当前环境中无法直接运行（@sec-sw-reproducibility）[@lab2023preferentialbo]。

这些问题单独看都不大；但合在一起意味着，目前若不联系作者，就无法逐一复现偏好贝叶斯优化的基准结果（推断）。

## 模拟用户与真实用户 {#sec-sw-simulated-users}

上述每个基准都假定，模拟用户可以充分代替真人，足以用来给方法排名。少数几项直接比较二者的研究都发现了明显差异。@tbl-sw-human 汇总了这些研究。

::: {.table #tbl-sw-human title="以真人替代通常假定的模拟用户的研究，及其发现的差异。"}
| 研究 | 场景 | 被试 | 与模拟用户的不同之处 |
|---|---|---|---|
| @schoinas2025evaluating，EMBC 2025 | 视网膜植入物编码器，模拟假体视觉 | 17 名视力正常者 | 只有约 50% 的试次与模拟智能体的选择相同；最终损失为 0.27，模拟中为 0.07 |
| @ou2022human，Mensch und Computer 2022 | 三维模型的 9 参数多边形简化 | 2 位艺术家，历时 3 个月；实验室中 20 人 | 现场有 11.9% 的序列得到满意结果，实验室中为 48.5%；判断不一致，损失厌恶 |
| @ou2023impact，IUI 2023 | 文本、照片与三维网格任务 | 60 人 | 停止取决于专业程度 |
| @colella2020human，UMAP 2020 | 一维函数，标量反馈 | 21 人 | 理解优化器的用户会有策略地给出带偏差的回答 |
| @chan2022investigating，CHI 2022 | 三维触摸交互设计，以测得的完成时间与空间误差为目标的多目标贝叶斯优化 | 40 名新手设计师 | 结果更好，但能动感和表达力更低 |
| @taddei2026bayesian，arXiv 2026（预印本） | 主动假肢的偏好优化 | 先模拟，再在 4 人身上试验（3 人用一种方法，1 人用另一种） | 一名被试的三次试验中，有一次的偏好估计在第 15 次迭代后仍在波动，可能源于疲劳 |
:::

主要结论由表中的数字体现，另有三个细节可作补充。其一，视网膜植入物研究检验的是 @granley2023human（NeurIPS 2023）只在模拟中评估过的一种优化；在主要条件下，17 名被试中仍有 16 名偏好优化后的结果，作者强调“用人类被试验证优化策略的重要性”[@schoinas2025evaluating]。其二，在 Ou 等人为期三个月的现场部署中，549 个评价序列中有 415 个在第一次迭代即告停止；作者的结论是，“使用偏好选择的优化缺乏处理不一致、相互矛盾的人类判断的机制”，并且“机器的输出反过来又通过启发式偏差和损失厌恶影响用户未来的输入”[@ou2022human]（@sec-hci-unstable、@sec-hci-expertise）。其三，理解优化器工作原理的用户会“有策略地给出带偏差的回答”[@colella2020human]，这与模拟用户关于反馈如实的假设相反；Mikkola 等人则报告，他们的方法能够区分人做出的选择与计算机程序做出的选择 [@mikkola2020projective]。

我们找到的模拟用户基准中，这些行为无一出现（推断）。模拟评测可以比较各算法在给定噪声模型下的行为，却无法预测它们面对真人时的排名（这一缺失的实验见 @sec-acqf-empirical）。在模拟器经过个体数据验证以前，用模拟用户得到的结果若要作为关于人的结论报告，应当先至少在几个真人身上检验（推断）；实验经济学为这类比较确立的规范可资借鉴（@sec-econ-replication）。

## 研究社区 {#sec-sw-community}

偏好贝叶斯优化的研究集中在少数几个研究组。本节先按学科勾勒这些研究组，再介绍一个领域的基础设施（综述、学位论文、研讨会、发表渠道），最后报告发表数量。

### 按学科划分的研究组 {#sec-sw-groups}

@tbl-sw-groups 按发表成果的学科排列各研究组；学科在很大程度上也对应着各组所提问题的类型。

::: {.table #tbl-sw-groups title="2017 至 2026 年偏好贝叶斯优化的研究组，按学科排列。"}
| 学科（典型发表渠道） | 研究组（机构） | 代表性工作 | 方向 |
|---|---|---|---|
| 机器学习方法（ICML、AISTATS、NeurIPS、ICLR、TMLR） | Frazier、Astudillo、Bakshy、Lin（康奈尔大学、Meta、加州理工学院） | [@astudillo2020multib]；BOPE [@lin2022preferenceb]；qEUBO [@astudillo2023qeubob]；[@astudillo2025preferential]；LILO [@kobalczyk2026lilo] | 决策论采集函数与结果上的偏好探索，近来转向由语言模型给出标签 |
| | Takeno、Nomura、Karasuyama（名古屋工业大学、CyberAgent AI Lab、理化学研究所） | [@takeno2023practicalc; @ozaki2024multi] | 偏斜高斯过程推断与幻觉信念；optuna-dashboard 的采样器实现了他们的方法 |
| | Benavoli、Azzimonti、Piga（都柏林圣三一学院、IDSIA） | [@benavoli2021preferentialb; @benavoli2026tutorial] | 偏好与选择似然、精确后验、prefGP 代码 |
| | Kaski（阿尔托大学） | PPBO [@mikkola2020projective]；批量 PBO [@siivola2021preferential]；PABBO [@zhang2025pabbob]；[@sinaga2024anchor] | 投影查询、批量、摊销、噪声模型 |
| 机器学习理论（ICML、NeurIPS、AISTATS） | Krause（苏黎世联邦理工学院） | [@kirschner2021bias]；MaxMinLCB [@pasztor2024bandits] | 对决赌博机理论 |
| | Vakili 及其合作者（MediaTek Research；伦敦大学学院与帝国理工学院的合作者） | MR-LPF [@kayal2025bayesian]；PF-TS [@lazzaro2026finiteb] | 遗憾理论；没有发布软件，没有人类研究 |
| | Jones（洛桑联邦理工学院） | POP-BO [@xu2024principledb] | 理论保证，应用为建筑热舒适 |
| 控制（ECC、ACC、IEEE Transactions on Control Systems Technology） | Bemporad、Piga（卢卡 IMT 高等研究院；IDSIA） | GLISp [@bemporad2021global]；C-GLISp [@zhu2022c]；PWAS 与 PWASp（代码 [@zhu2025pwas]） | 非概率代理模型与控制器标定 |
| 机器人学（ICRA、IROS、IEEE Robotics and Automation Letters） | Ames、Yue、Tucker、Novoseller（加州理工学院） | CoSpar [@tucker2020preference]；LineCoSpar [@tucker2020human]；POLAR [@tucker2022polar] | 个性化外骨骼步态（@sec-cs-exoskeleton） |
| | Trimpe（亚琛工业大学） | CrashPBO [@menn2026preferential]；局部 PBO [@menn2026local] | 2026 年进入该领域；调节机器人控制器 |
| 人机交互与图形学（SIGGRAPH、UIST、CHI、IUI、Mensch und Computer、UMAP） | Koyama（东京大学；产业技术综合研究所） | 序列线搜索 [@koyama2017sequentialb]；序列画廊（Sequential Gallery）[@koyama2020sequential]；约束偏好贝叶斯优化 [@iwai2025constrainedb] | 视觉设计的交互式优化（@sec-cs-photo），转向约束与语言模型辅助 |
| | Oulasvirta（阿尔托大学） | [@colella2020human; @chan2022investigating] | 人在回路优化的人类研究，大多使用评分或表现反馈，而不是比较 |
| | Butz、Buschek、Mayer（慕尼黑大学，媒体信息学） | [@ou2022human; @ou2023impact] | 创意工具中偏好贝叶斯优化的人类被试研究 |
| 生物医学工程（EMBC） | Fauvel、Chalk、Beyeler | [@fauvel2021efficient; @granley2023human; @schoinas2025evaluating] | 视网膜假体编码器的人在回路优化 |
:::

其他反复出现的贡献者包括 Mesbah 及其同事（KappaSharp，2026 年的一篇预印本）[@shao2026adaptive]、Wu 与 Gardner（面向偏好学习的知识梯度，2026 年的一篇预印本）[@wu2026knowledge]、控制领域的 Theiner、Hirt、Findeisen 及其同事（ECC 2025 与 ECC 2026）[@theiner2025exploiting; @theiner2026efficient]；我们没有核实他们所属的机构。纵观全表，各研究组正从方法转向应用，并转向以语言模型为中介，新加入者（Trimpe、Mesbah）则来自控制领域（推断）。证明遗憾界的研究组不做人类研究，做人类研究的研究组很少改动采集函数，构建软件的研究组则最接近决策论路线（推断）。

### 没有综述，九篇学位论文，没有专门的研讨会 {#sec-sw-infrastructure}

**没有专门的偏好贝叶斯优化综述。** 贝叶斯优化的一般参考资料给偏好的篇幅至多一页：Frazier 的教程 [@frazier2018tutorial] 中没有出现“preference”“pairwise”或“duel”等词，Garnett 的专著 [@garnett2023bayesian] 用约一段话提及优化人的偏好，@wang2023recent（ACM Computing Surveys 2023）的综述没有讨论偏好贝叶斯优化的章节。@bengs2021preference（JMLR 2021）的对决赌博机综述涵盖了用高斯过程建模的偏好概率，但既未引用 González 等人 2017 年的论文，也未引用 Brochu 等人的工作 [@brochu2010tutorial]。最接近参考书的是 @benavoli2026tutorial（Foundations and Trends in Machine Learning 2026）的教程，它涵盖偏好与选择的九种高斯过程模型，但对偏好贝叶斯优化只是简短提及。其余的教学材料是软件教程，以及 Koyama 与 Igarashi 撰写的书中章节 *Computational Design with Crowds* [@koyama2020computational]；这是一篇预印本，我们没有核实其正式发表信息。

**这一领域的综合工作大多见于博士学位论文。** 我们找到了至少九篇 2017 至 2025 年的博士学位论文，它们以偏好贝叶斯优化为中心，或以相当篇幅讨论这一主题（@tbl-sw-theses）。

::: {.table #tbl-sw-theses title="2017 至 2025 年以偏好贝叶斯优化为主题或大量涉及这一主题的博士学位论文。"}
| 作者 | 题名 | 机构 | 年份 |
|---|---|---|---|
| Yuki Koyama | *Computational Design Driven by Visual Aesthetic Preference* [@koyama2017computational] | 东京大学 | 2017 |
| Ellen Novoseller | *Online Learning from Human Feedback with Applications to Exoskeleton Gait Optimization* [@novoseller2021online] | 加州理工学院 | 2021（记录日期） |
| Eero Siivola | *Applications of human feedback in Gaussian processes* [@siivola2021applications] | 阿尔托大学 | 2021 |
| Tristan Fauvel | *Human-in-the-loop optimization of retinal prostheses encoders* [@fauvel2021human] | 索邦大学 | 2021 |
| Raul Astudillo | *Exploiting Composite Functions in Bayesian Optimization* [@astudillo2022exploiting] | 康奈尔大学 | 2022 |
| Maegan Tucker | *Enabling Robust and User-Customized Bipedal Locomotion on Lower-Body Assistive Devices via Hybrid System Theory and Preference-Based Learning* [@tucker2023enabling] | 加州理工学院 | 2023 |
| Petrus Mikkola | *Humans as Information Sources in Bayesian Optimization* [@mikkola2024humans] | 阿尔托大学 | 2024 |
| Mengjia Zhu | *Global and preference-based optimization using surrogate-based methods* [@zhu2024global] | 卢卡 IMT 高等研究院 | 2024 |
| Wenjie Xu | *Bayesian Optimization with Constraints, Structure and Human Feedback* [@xu2025bayesian] | 洛桑联邦理工学院 | 2025 |
:::

**最接近的研讨会面向整个偏好学习。** ICML 2023 研讨会 *The Many Facets of Preference-Based Learning* 涵盖对决赌博机、基于人类反馈的强化学习（RLHF，用于使语言模型与人类偏好对齐的方法）、社会选择与优化，其中有两篇论文讨论基于偏好的贝叶斯优化 [@icml2023many]。偏好贝叶斯优化的工作也见于 ICML 2019 Workshop on Human in the Loop Learning [@mccourt2019sampling]、NeurIPS 2022 Workshop on Gaussian Processes, Spatiotemporal Modeling, and Decision-making Systems [@takeno2022preferential] 以及 ProbML 2026；ProbML 2026 是与 ICML 同期举办、出版存档论文集的专题研讨会 [@probml2026symposium]。从 arXiv 评论和三次网络检索来看，2024 至 2026 年间 NeurIPS 和 ICML 上没有专门讨论偏好贝叶斯优化的研讨会。

### 发表数量 {#sec-sw-counts}

发表数量在增长，但基数很小。我们做了四次检索：两次针对 arXiv 摘要 [@arxiv2026preferential; @arxiv2026bayesian]，两次针对 Semantic Scholar [@scholar2026preferential]；每个来源各用一种窄检索式（短语）和一种宽检索式（偏好类词语加上贝叶斯优化）。各年数量见 @fig-sw-publications。

```{figure}
//| figure: sw-publications
//| label: fig-sw-publications
//| fig-cap: "2026 年 9 月对偏好贝叶斯优化论文所做的四次检索中，各年的条目数；菜单可在四次检索之间切换，纵轴刻度固定。2026 年的柱（斜线填充）只含 1 至 9 月。总数：46（arXiv，短语）、162（arXiv，宽泛）、43（Semantic Scholar，短语）与 135（Semantic Scholar，宽泛）。检索在两个方向上都有误差：短语检索漏掉了未使用这一短语的已知偏好贝叶斯优化论文（MaxMinLCB 与 POLAR 均不在两次 arXiv 检索的结果中），宽泛检索则混入了植物科学和流行病学中的无关结果。Semantic Scholar 的短语检索没有列出 2018 年和 2019 年。"
search: arxivPhrase
```

四次检索都显示，2023 至 2025 年间数量增长到原来的约 2.4 至 4.3 倍，而 2026 年前九个月的数量已接近 2025 年全年（分别为 12 条对 13 条、38 对 39、10 对 13、32 对 34）。但即使在 2025 与 2026 年，每年也只有 10 至 40 条。这一增长在时间上与语言模型对齐带给偏好反馈的关注相吻合（推断）。我们核查过的交叉引用很少：2021 年的对决赌博机综述没有引用 González 等人 2017 年的论文，2026 年的局部偏好贝叶斯优化论文也没有引用高维长度尺度先验方面的工作（@sec-hd-local）（推断）。

## 已定、有争议与缺失 {#sec-sw-status}

::: {.frontier title="已定、有争议与缺失"}
**已定。** 仍在维护的通用偏好贝叶斯优化软件只有 BoTorch 与 Ax，均来自 Meta；其他主要的贝叶斯优化与高斯过程库都不支持偏好。在我们核查过的偏好软件中，没有一个的默认长度尺度先验随维度缩放。研究代码没有发布标签，依赖过时，部分代码还没有许可证。评测依赖模拟用户和 1 至 8 维的合成函数，噪声模型、遗憾定义和重复次数都由各论文自行选定。既没有专门的偏好贝叶斯优化综述，也没有专门的研讨会；最接近的是 ICML 2023 上一个面向整个基于偏好的学习的研讨会。自 2023 年以来，发表数量从很小的基数开始增长。

**有争议。** 哪种采集函数最好：答案随维度、噪声和度量而变，qEUBO 与 POP-BO 相比报告的解更好、累积遗憾却更差，即是一例。用模拟用户得到的结果在多大程度上适用于人：现有证据指向明显差异，但相关研究规模都很小。

**缺失。** 共享的基准套件与排行榜。个体层面成对判断的公开数据集。在同一协议下重新运行多种方法的复现研究。随机分配采集函数的人类被试比较。Ax 偏好功能的教程和默认先验的文档。面向 R 或 Julia 且仍在维护的偏好贝叶斯优化软件包（未系统检索）。
:::

## 习题 {#sec-sw-exercises}

::: {.exercise #exr-sw-coverage title="随机点对为何在最佳已查询点度量上获胜"}
在 @fig-sw-metrics 中，每个随机查询是从 41 个候选点中均匀抽取的两个不同的点，一次运行提出 16 个这样的点对。（a）某个给定的候选点从未得到展示的概率是多少？（b）平均有多少个不同的候选点得到展示？（c）在示例目标函数上，有三个候选点的遗憾低于 0.12：最优点及其两个相邻点。其中至少一个得到展示的概率是多少？

::: {.solution}
（a）给定候选点不在某一个点对中的概率为 $39/41$，因此不在全部 16 个独立点对中的概率为 $(39/41)^{16} \approx 0.449$。（b）由期望的线性性，平均有 $41 \times (1 - 0.449) \approx 22.6$ 个不同的候选点得到展示。（c）三个给定候选点都不在某一个点对中的概率为 $\binom{38}{2}/\binom{41}{2} = (38 \cdot 37)/(41 \cdot 40) \approx 0.857$，都不在全部 16 个点对中的概率为 $0.857^{16} \approx 0.085$，因此至少一个得到展示的概率约为 0.915。随机点对几乎穷尽了只有 41 个点的空间；最佳已查询点这一度量奖励的是这种覆盖，却不能说明模型是否学到了最优点的位置。
:::
:::

::: {.exercise #exr-sw-incumbent title="当前最优点规则为何在累积遗憾上获胜"}
@fig-sw-metrics 中的累积遗憾逐次累加每次对决中两个选项的平均遗憾 $\tfrac12[r(\vx) + r(\vx')]$。假设从某一步起，当前最优点就是真正的最优点。对于总是包含当前最优点的规则和从不重复展示同一个点的规则，每次对决可能的最小遗憾分别是多少？这一度量奖励的是什么？

::: {.solution}
包含最优点的对决代价为 $\tfrac12[0 + r(\vx')] = \tfrac12 r(\vx')$，至多为挑战者遗憾的一半；若挑战者也是最优点，则为零。分散查询的规则要为两个新点付出完整的平均遗憾；在示例目标函数上，随机候选点的平均遗憾约为每个点 0.7。因此，累积遗憾奖励的是在每次查询中都保留当前最优点，即使挑战者几乎没有为模型带来新信息：它衡量的是人在过程中看到了什么，而非最终推荐有多好。正因如此，一种方法可能在一种度量上更好、在另一种度量上更差，POP-BO 的比较即是如此。
:::
:::

::: {.exercise #exr-sw-flip title="固定翻转率不是效用的噪声模型"}
无论两个选项是什么，BOPE 的模拟用户都有 10% 的时间选择较差的选项。若在每个效用上施加大小为 $\sigma$ 的概率单位噪声，效用差 $\Delta > 0$ 时的出错概率为 $\Phi(-\Delta/(\sqrt{2}\,\sigma))$。（a）$\sigma$ 取何值时，效用差 $\Delta = 0.1$ 的比较有 10% 的时间答错？（b）取这一 $\sigma$，效用差 $0.3$ 的比较答错的频率是多少？（c）这对比较分别使用这两种噪声模型的论文有何启示？

::: {.solution}
（a）需要 $\Delta/(\sqrt{2}\,\sigma) = \Phi^{-1}(0.9) \approx 1.2816$，故 $\sigma = 0.1/(\sqrt{2} \times 1.2816) \approx 0.055$。（b）此时 $\Delta/(\sqrt{2}\,\sigma) \approx 3.84$，$\Phi(-3.84) \approx 6 \times 10^{-5}$：几乎从不出错，翻转模型下则为 10%。（c）在概率单位噪声下，错误集中在难分的比较上；在固定翻转率下，明显更差的选项与略差的选项被选中的频率相同。后者对依赖少数几次决定性比较的方法更苛刻，因此同样称为“10% 的噪声”，难度可能大不相同，两种模型下的结果不能合并（推断）。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- `PairwiseGP` 的源代码 [@botorch2026pairwisegp] 和 BoTorch 的偏好贝叶斯优化教程 [@botorch2026pairwise] 实际上定义了实践中的偏好贝叶斯优化；在采信任何默认设置之前，应先阅读这两者。
- @fauvel2021efficient 是唯一一篇用形式化的多重比较评测多种采集函数的论文；其评测协议值得效仿。
- @xu2024principledb 在同一篇论文中展示了度量如何决定哪种方法显得更好。
- @schoinas2025evaluating 与 @ou2022human 是偏好循环中模拟用户与真人之间最清楚的直接比较。
- @benavoli2026tutorial 在模型方面最接近参考书，在优化方面则尚未达到。
