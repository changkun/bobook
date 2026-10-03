# 中文版术语表与写作规范

本文件供中文版的译者与审校使用，不进入构建（构建只渲染 `en/` 与 `zh/` 下 `book.yml` 列出的页面）。§1 规定标点、数字、引用与 Markdown 源文件的写法；§2 是按主题分组的术语对照；§3 列出保留英文的名称；§4 列出禁用写法，其中 §4.1 的表供 `tools/zh-check.ts` 机械检查；§5 给出书名、各部分与各章标题以及界面用语；§6 列出需要裁定的选择、同一英文词的两种译法、主编的裁定，以及翻译中新增的译名。

译名优先采用中文机器学习、统计学与心理学文献的通行译名，说明栏中的“标准用法”即指这类通行译名。同一概念有几种写法时只取一种，其余列入 §4。没有通行译名或需要权衡的选择列在 §6，以 §6.3 的裁定为准。

## 1. 写作规范

以下规则适用于中文版的正文、图注与表格。

### 1.1 空格

- 汉字与拉丁字母、汉字与阿拉伯数字之间加一个半角空格：“Matérn 核”“2020 年”“第 3 章”“20 维”。
- 全角标点两侧不加空格：写“高斯过程（Gaussian process）”，不写“高斯过程 （Gaussian process）”。
- 数字与 % 之间不加空格：“16.6%”。% 与后面的汉字之间加空格：“约 67% 的成本”。
- 行内公式与汉字之间加空格：“后验均值 $\mu_n(\vx)$ 随 $n$ 增大”。
- 统计量照英文写法保留等号两侧的空格：“N = 20”“d = 0.06”“p = 0.003”“p < 0.001”。

### 1.2 标点

- 正文用全角标点：，。：；？！（）、。汉字后不接半角逗号、分号、冒号或左括号。
- 括号一律用全角“（）”，括号里是拉丁字母也一样：“Matérn 核（Matérn 5/2）”。括号里英文全称与缩写之间用全角逗号，这是引入缩写的固定格式：“最优选项期望效用（expected utility of the best option，EUBO）”。公式内部的括号不变。
- 并列的词语用顿号“、”，并列的分句用逗号。
- 引号用弯引号“”，引号内再引用用‘’；不用直引号 "…"，也不用「」（见 §6）。按 CONVENTIONS.md，引号只用于与原文核对过的措辞。
- 书名号《》只用于中文书刊名；英文的论文题名、书名、期刊与会议名保留原样（§3）。
- 不用破折号。英文版不用 em dash，中文版也不用“——”，改用逗号、冒号、括号，或拆成两句。
- 省略号用“……”。

### 1.3 人名与引用

- 外国人名保留拉丁字母，不音译：“González 等人”“Chu 与 Ghahramani”。例外只有已经进入术语的译名：贝叶斯、高斯、Laplace、Markov、蒙特卡洛、Thompson、Pareto、Nash、Bernoulli、Gibbs、Shannon、Dirichlet、Newton、Weber、Fechner、Occam，以及“Mahalanobis 距离”。这些译名只用在术语里；这些人作为被引作者出现时，仍写拉丁字母。
- 三位及以上作者在叙述中写“X 等人”。表格、参考文献列表与括号内的引用写“X 等”。
- 两位作者写“X 与 Y”。列出三位作者时写“A、B 与 C”。
- 叙述式引用写“González 等人（2017）”，括号式引用写“（González 等，2017）”。引用由构建生成，正文只写 `@key` 与 `[@key]`，不手写作者与年份。构建目前用“和”与“等”，见 §6 第 4 条。
- 中文正文不出现“et al.”与连接作者的“and”。

### 1.4 年份、日期与数字

- 年份写“2020 年”；年份区间写“2017 至 2026 年”；月份写“2026 年 9 月”；日期写“2026 年 3 月 26 日”。
- 年代写“20 世纪 60 年代”，不写“1960 年代”（见 §6 第 17 条）。
- 数值区间用“至”：“10 至 20 维”。带单位或百分号的区间两端都写：“40% 至 50%”。英文的“to”“between … and …”都译成“至”。
- 测量值、样本量、实验次数用阿拉伯数字，数字与量词或单位之间加空格：“11 名被试”“3 个月”“549 个评价序列”“10.9 分钟”。不是测量结果的小数目用汉字：“两种方法”“三个问题”。
- 四位以上的整数用逗号分节：“1,225”“10,000”。不换算成“万”“亿”，以便与英文版逐一核对（见 §6 第 18 条）。
- 小数位与有效数字和英文版完全一致。“about”译为“约”；英文没有“about”时不加“约”。
- “percentage points”写“个百分点”；倍数写“约 10 倍”；“±”两侧加空格：“10.9 ± 0.9 分钟”。
- 时间单位用汉字（秒、分钟、小时、天、周、个月、年）；其余国际单位保留符号，与数字空一格：“12 dB”“300 ms”。
- 序号：章、节、图、表、式用阿拉伯数字（“第 3 章”“图 3.1”“式（3.4）”）；部分用汉字数字（“第二部分”）；行文中的小序数用汉字（“第一次迭代”“第二组”）。

### 1.5 术语与缩写

- 术语在英文版加粗定义的位置写“**中文**（English）”，之后只写中文。
- 一般概念的缩写（BO、GP、EI、PI、UCB、KG、TS、EP、MLE、MAP、LLM、HCI、JND、DDM）在中文正文中不用，写中文全称。数学式中的 `\EI`、`\PI`、`\UCB`、`\EUBO` 不变；GP-UCB、UCB1、qEI 这类算法名保留（§3）。
- 偏好贝叶斯优化：全书第一次出现写“偏好贝叶斯优化（preferential Bayesian optimization，PBO）”，此后正文写全称。这与英文版“首次出现后写 PBO”的约定不同，请主编裁定（§6 第 1 条）。
- 方法、算法、软件的名称与缩写保留英文（§3）。有中文名的，首次出现写“中文名（缩写）”，此后写缩写：首次写“最优选项期望效用（expected utility of the best option，EUBO）”，此后写 EUBO。
- qEUBO 是 EUBO 的多选项形式（一次查询展示 q 个选项），不称“批量形式”（CONVENTIONS.md）。
- “对决”在 `preferences/01` 首次出现时定义为一次成对比较，此后与“比较”可以互换。
- 英文版的“(inference)”写作“（推断）”，全角括号，放在句号之前：“……（推断）。”

### 1.6 不翻译的内容

- 数学：`$...$`、`$$...$$` 的全部内容、宏（`\vx`、`\mK`、`\EI`、`\operatorname{sigmoid}` 等）、公式标签，以及公式中 `\text{}` 里的文字，照英文版原样保留。
- 代码：反引号中的一切（标识符、文件名、参数名、命令），以及围栏代码块中的代码、字符串与输出。围栏代码块中的注释译成中文（见 §6 第 30 条）。
- 标识：`@sec-…`、`@fig-…`、`@eq-…`、`[@key]`、`{#sec-…}` 等交叉引用与引用语法，以及标题 id。
- 图：`{figure}` 围栏中 `//| figure:` 的模块名和各参数行不变，只翻译 `fig-cap`。图内界面文字由图模块的 `labels.zh` 提供，不在 Markdown 中翻译。英文版图中读者填写的输入标签“Answers”在 `labels.zh` 中写“回答”。
- 中文不用斜体表示强调。英文版的 *强调* 在中文里靠语序表达或直接去掉；需要突出的术语用加粗。斜体只留给拉丁字母的题名和变量名。

### 1.7 Markdown 源文件

- 中文段落不要在句中硬换行。markdown-it 把段内换行输出为换行符，浏览器会在两个汉字之间把它显示成一个空格。每段写成一行，或先确认构建会去掉汉字之间的软换行。

## 2. 术语对照

399 行，按主题分组。一行列出几个相关术语时，用分号分隔，中文与英文一一对应。说明栏中的“见 §6”指需要主编裁定的选择，“见 §6.2”指同一英文词在不同语境下的两种译法。

### 2.1 问题与总体框架

| English | 中文 | 说明 |
|---|---|---|
| Bayesian optimization | 贝叶斯优化 | 正文不用缩写 BO，见 §6 |
| preferential Bayesian optimization (PBO) | 偏好贝叶斯优化 | 全书首次出现附英文与 PBO；正文写全称，见 §6 |
| preferential optimization | 偏好优化 | 只指包括非贝叶斯方法在内的更大一类 |
| black box; black-box objective | 黑箱；黑箱目标函数 |  |
| objective (function) | 目标函数 | 上下文清楚时可简称“目标” |
| evaluation (of the objective); noisy evaluation | 评估；带噪声的评估 | 人给出的评判用“评价”，见 §6.2 |
| budget | 预算 |  |
| input; input space; domain | 输入；输入空间；定义域 | |
| search space; design space | 搜索空间；设计空间 | |
| dimension; number of inputs d | 维度；输入个数 d | “d 维”“20 维” |
| surrogate (model) | 代理模型 |  |
| acquisition function | 采集函数 |  |
| exploration; exploitation | 探索；利用 | |
| exploration-exploitation trade-off | 探索与利用的权衡 |  |
| query; option; candidate set | 查询；选项；候选集 | 向人提出的一次比较也称查询 |
| utility; latent utility | 效用；潜在效用 | |
| sequential (decision, design) | 序贯 | 序贯优化、序贯设计 |
| grid search; random search | 网格搜索；随机搜索 | |
| gradient descent; finite differences; evolution strategies | 梯度下降；有限差分；进化策略 | |
| hyperparameter; hyperparameter tuning | 超参数；超参数优化 | 泛称的设备或控制器调节用“调参”“调节” |
| human-in-the-loop optimization; person in the loop | 人在回路优化；人在回路 | |
| simulated user | 模拟用户 |  |
| oracle | 预言机 | 作为应答者的 oracle；统计学的 oracle rate 写“神谕速率”，见 §6 |
| benchmark; test function | 基准；测试函数 | |

### 2.2 概率与统计

| English | 中文 | 说明 |
|---|---|---|
| probability; frequency reading; belief reading | 概率；频率解释；信念解释 | 标准用法 |
| frequentist; Bayesian | 频率派；贝叶斯 |  |
| outcome; sample space; event; complement | 结果；样本空间；事件；对立事件 | |
| random variable | 随机变量 | 标准用法 |
| probability mass function | 概率质量函数 | 标准用法 |
| probability density (function) | 概率密度（函数） | 标准用法；“密度”可单用 |
| cumulative distribution function (CDF) | 累积分布函数 | 标准用法；Φ 称标准正态分布函数 |
| Bernoulli, binomial, categorical, uniform, exponential distribution | Bernoulli 分布、二项分布、类别分布、均匀分布、指数分布 | |
| joint distribution; conditional distribution | 联合分布；条件分布 | |
| marginal distribution; marginalize | 边际分布；边际化 | |
| sum rule; product rule; chain rule; law of total probability | 加法规则；乘法规则；链式法则；全概率公式 | 标准用法 |
| Bayes' rule | 贝叶斯定理 | 见 §6 |
| prior; likelihood; posterior; log-likelihood | 先验；似然；后验；对数似然 | |
| evidence; model evidence | 证据；模型证据 | “证据”在研究部分另指实证证据；作模型量时写“模型证据” |
| marginal likelihood; log marginal likelihood | 边际似然；对数边际似然 |  |
| base-rate neglect | 基础比率忽视 | 标准用法 |
| log-sum-exp; underflow | log-sum-exp；下溢 | |
| expectation; linearity of expectation | 期望；期望的线性性 | |
| mean; variance; standard deviation | 均值；方差；标准差 | |
| covariance; covariance matrix; correlation coefficient | 协方差；协方差矩阵；相关系数 | |
| law of total expectation; law of total variance | 全期望公式；全方差公式 | 标准用法 |
| aleatoric uncertainty; epistemic uncertainty | 偶然不确定性；认知不确定性 | |
| Monte Carlo (estimate); Markov chain Monte Carlo (MCMC) | 蒙特卡洛（估计）；Markov 链蒙特卡洛（MCMC） |  |
| unbiased; overfit | 无偏；过拟合 |  |
| independence; conditional independence; i.i.d. | 独立性；条件独立；独立同分布 | |
| noise; observation model | 噪声；观测模型 | |
| Gaussian distribution; normal distribution; standard normal | 高斯分布；正态分布；标准正态分布 | |
| z-score; error function; central limit theorem | z 分数；误差函数；中心极限定理 | 标准用法 |
| multivariate Gaussian; mean vector | 多元高斯分布；均值向量 | |
| precision; precision matrix | 精度；精度矩阵 | |
| Mahalanobis distance | Mahalanobis 距离 |  |
| completing the square; regression toward the mean | 配方；向均值回归 | 标准用法 |
| Beta distribution; Beta function | Beta 分布；Beta 函数 |  |
| Gamma distribution; gamma function Γ; log-normal | Gamma 分布；伽马函数；对数正态 | |
| conjugate prior | 共轭先验 | |
| pseudo-counts; prior strength; shrinkage | 伪计数；先验强度；收缩 | |
| credible interval | 可信区间 | 贝叶斯区间；不与置信区间混用 |
| confidence interval | 置信区间 | 频率派区间 |
| Laplace's rule of succession | Laplace 继承法则 | 标准用法 |
| maximum likelihood estimate (MLE) | 最大似然估计 |  |
| maximum a posteriori estimate (MAP); point estimate | 最大后验估计；点估计 | |
| posterior mean; posterior variance | 后验均值；后验方差 |  |
| ridge regression; regularization; add-one smoothing | 岭回归；正则化；加一平滑 | |
| Bayesian linear regression; weights; features; basis functions; design matrix | 贝叶斯线性回归；权重；特征；基函数；设计矩阵 |  |
| weight space; function space | 权重空间；函数空间 | 标准用法 |
| posterior predictive distribution; plug-in predictive | 后验预测分布；插入式预测 | |
| generative model; prior predictive check | 生成模型；先验预测检验 |  |
| Occam's razor; data-fit term; complexity penalty | Occam 剃刀；数据拟合项；复杂度惩罚 | 标准用法 |
| type II maximum likelihood; empirical Bayes; Bayes factor | 第二类最大似然；经验贝叶斯；贝叶斯因子 | |

### 2.3 信息论

| English | 中文 | 说明 |
|---|---|---|
| bit; nat | 比特；奈特 | 标准用法 |
| surprise (information content) | 意外度（信息量） | 正式名称为自信息，见 §6 |
| entropy; binary entropy; differential entropy | 熵；二元熵；微分熵 | |
| Kullback-Leibler divergence; KL divergence; relative entropy | Kullback-Leibler 散度；KL 散度；相对熵 | |
| cross-entropy | 交叉熵 | 标准用法 |
| Gibbs' inequality; Jensen's inequality | Gibbs 不等式；Jensen 不等式 | |
| mass-covering; mode-seeking | 覆盖质量的；寻找众数的 | 描述 KL 的两个方向 |
| conditional entropy; mutual information | 条件熵；互信息 | |
| data processing inequality | 数据处理不等式 | 标准用法 |
| information gain; expected information gain | 信息增益；期望信息增益 |  |
| maximum information gain γ_T | 最大信息增益 |  |
| Bayesian experimental design | 贝叶斯实验设计 | |
| Bayesian active learning by disagreement (BALD) | 贝叶斯分歧主动学习（BALD） |  |
| uncertainty sampling; submodularity | 不确定性采样；次模性 | 标准用法 |
| myopic; non-myopic | 短视；非短视 |  |

### 2.4 线性代数与数值计算

| English | 中文 | 说明 |
|---|---|---|
| vector; inner product; norm; orthogonal; orthonormal | 向量；内积；范数；正交；标准正交 | |
| matrix; linear map; identity matrix; transpose; outer product | 矩阵；线性映射；单位矩阵；转置；外积 | |
| inverse; invertible; singular | 逆矩阵；可逆；奇异 | |
| symmetric matrix; quadratic form | 对称矩阵；二次型 | 标准用法 |
| positive definite; positive semidefinite | 正定；半正定 | 标准用法 |
| jitter | 抖动项 | |
| eigenvector; eigenvalue; spectral theorem | 特征向量；特征值；谱定理 | |
| condition number; ill-conditioned | 条件数；病态 | |
| rank-deficient | 秩亏 |  |
| lower / upper triangular; forward / back substitution | 下三角 / 上三角；前代 / 回代 | 标准用法 |
| Cholesky factorization; Cholesky factor | Cholesky 分解；Cholesky 因子 | 人名不音译 |
| conjugate gradients | 共轭梯度法 | 标准用法 |
| determinant; log-determinant; generalized variance | 行列式；对数行列式；广义方差 |  |
| trace | 迹 | |
| block matrix; block inversion; Schur complement | 分块矩阵；分块求逆；Schur 补 | 标准用法 |
| Woodbury identity; Sherman-Morrison formula; matrix determinant lemma | Woodbury 恒等式；Sherman-Morrison 公式；矩阵行列式引理 | 标准用法 |
| Hessian | Hessian 矩阵 | |
| gradient; Newton's method; quasi-Newton (L-BFGS) | 梯度；Newton 法；拟 Newton 法（L-BFGS） | |
| graph Laplacian; weighted graph Laplacian | 图 Laplace 矩阵；加权图 Laplace 矩阵 | |
| algebraic connectivity; effective resistance; connected component | 代数连通度；有效电阻；连通分量 | |
| linear-Gaussian pair; gain | 线性高斯对；增益 | 标准用法 |
| big-O notation | 大 O 记号 | 标准用法 |

### 2.5 高斯过程

| English | 中文 | 说明 |
|---|---|---|
| Gaussian process (GP); Gaussian process regression | 高斯过程；高斯过程回归 | 正文不用缩写 GP |
| distribution over functions | 函数上的分布 |  |
| stochastic process; index set; sample path | 随机过程；指标集；样本路径 | |
| marginalization property (consistency) | 边际化性质（一致性） |  |
| mean function; covariance function | 均值函数；协方差函数 | 协方差函数与核函数同义 |
| kernel; kernel matrix; kernel trick | 核函数；核矩阵；核技巧 | 单独出现写“核函数”；复合词用“核”：Matérn 核、RBF 核 |
| positive semidefinite kernel; Mercer's theorem | 半正定核；Mercer 定理 | 标准用法 |
| stationary; non-stationary | 平稳；非平稳 |  |
| lengthscale | 长度尺度 |  |
| amplitude; output scale; signal variance | 幅度；输出尺度；信号方差 | |
| noise variance | 噪声方差 |  |
| RBF (radial basis function) kernel | 径向基函数核 | |
| squared exponential kernel | 平方指数核 | |
| Matérn kernel (Matérn 5/2) | Matérn 核（Matérn 5/2） | |
| periodic kernel; linear kernel; sums and products of kernels | 周期核；线性核；核的和与积 | |
| smoothness; mean-square differentiable | 光滑度；均方可微 | |
| Ornstein-Uhlenbeck process | Ornstein-Uhlenbeck 过程 |  |
| automatic relevance determination (ARD) | 自动相关性确定（ARD） | |
| fully Bayesian (treatment) | 全贝叶斯（处理） |  |
| dimension-scaled prior; vanishing gradients | 维度缩放先验；梯度消失 |  |
| leave-one-out (prediction); calibration; standardized residual | 留一（预测）；校准；标准化残差 | |
| standardize outputs; scale inputs | 标准化输出；缩放输入 | |
| posterior samples; drawing functions from a GP prior | 后验样本；从高斯过程先验中抽取函数样本 | 不写“采样函数”，见 §4 |
| latent value; latent function | 潜在值；潜在函数 | |
| latent variance; predictive variance | 潜在方差；预测方差 | f 的方差与 y 的方差，必须区分 |
| inducing points; random features; pathwise update | 诱导点；随机特征；路径式更新 |  |
| representer form; kernel ridge regression | 表示形式；核岭回归 | 标准用法 |
| reproducing kernel Hilbert space (RKHS) | 再生核 Hilbert 空间 |  |
| skew Gaussian process | 偏斜高斯过程 |  |
| skew-normal; unified skew-normal; extended skew-normal | 偏斜正态；统一偏斜正态；扩展偏斜正态 |  |
| elliptical slice sampling | 椭圆切片采样 |  |
| one-hot encoding; additive structure | 独热编码；加性结构 | |

### 2.6 贝叶斯优化与采集函数

| English | 中文 | 说明 |
|---|---|---|
| the Bayesian optimization loop | 贝叶斯优化循环 |  |
| initial design; space-filling; Latin hypercube; Sobol sequence | 初始设计；空间填充；拉丁超立方；Sobol 序列 |  |
| incumbent (point); best value observed f*_n | 当前最优点；当前最优值 | 见 §6 |
| improvement; improvement margin ξ | 改进；改进裕量 |  |
| probability of improvement (PI) | 改进概率 | 正文不写 PI；数学式中的 \PI 不变 |
| expected improvement (EI) | 期望改进 | 同上，正文不写 EI |
| upper confidence bound (UCB); lower confidence bound | 上置信界；下置信界 |  |
| confidence bound; exploration weight β | 置信界；探索权重 |  |
| Thompson sampling | Thompson 采样 |  |
| knowledge gradient (KG); exact knowledge gradient | 知识梯度；精确知识梯度 |  |
| entropy search; predictive entropy search; max-value entropy search | 熵搜索；预测熵搜索；最大值熵搜索 |  |
| one-step lookahead; one-step Bayes optimal (value) | 一步前瞻；一步贝叶斯最优（值） | |
| maximizing the acquisition function; multi-start gradient ascent | 采集函数优化；多起点梯度上升 |  |
| batch; parallel evaluation | 批量；并行评估 | qEI 写“批量期望改进” |
| kriging believer; constant liar | 克里金信念；常数说谎者 | 首次出现附英文；见 §6 |
| fantasy (fantasized observation) | 虚拟观测 | 见 §6 |
| noisy expected improvement (qNEI) | 带噪声期望改进 | qNEI 的首次写法见 §3.1 |
| constraint; black-box constraint; safe set | 约束；黑箱约束；安全集 | |
| multi-objective; Pareto front; Pareto optimal | 多目标；Pareto 前沿；Pareto 最优 | |
| hypervolume; hypervolume improvement; scalarization | 超体积；超体积改进；标量化 |  |
| multi-fidelity | 多保真度 | |
| cost-aware; expected improvement per second | 成本感知；每秒期望改进 |  |
| stopping rule | 停止规则 |  |
| Pandora's box; Gittins index | 潘多拉魔盒；Gittins 指数 | |
| warm start; cold start; transfer | 热启动；冷启动；迁移 |  |
| high-dimensional; effective dimension | 高维；有效维度 | |
| embedding; subspace | 嵌入；子空间 | |
| trust region | 信赖域 |  |
| local search; global search | 局部搜索；全局搜索 |  |
| heuristic | 启发式 |  |
| validation error; held-out data; cross-validation (5-fold) | 验证误差；留出数据；交叉验证（5 折） | |
| online experiments (A/B tests) | 在线实验（A/B 测试） |  |

### 2.7 遗憾、赌博机与理论

| English | 中文 | 说明 |
|---|---|---|
| regret (optimization sense); regret theory | 遗憾；遗憾理论 | 情绪意义的 regret 用“后悔”，见 §6.2 |
| instantaneous regret r_t; simple regret | 瞬时遗憾；简单遗憾 | |
| cumulative regret R_T | 累积遗憾 |  |
| Bayesian regret; worst-case regret | 贝叶斯遗憾；最坏情况遗憾 | |
| regret bound; upper bound; lower bound | 遗憾界；上界；下界 |  |
| sublinear; no-regret | 次线性；无遗憾 | |
| rate (of a bound) | 速率 |  |
| multi-armed bandit; arm; pull | 多臂赌博机；臂；拉动 | |
| horizon | 时域 |  |
| greedy; ε-greedy | 贪心；ε-贪心 | 见 §6 |
| optimism in the face of uncertainty | 面对不确定性时的乐观原则 | |
| instance-dependent; minimax | 实例相关；极小极大 | |
| kernelized | 核化 |  |
| Bayes optimal | 贝叶斯最优 |  |
| sample complexity; union bound; with high probability | 样本复杂度；联合界；以高概率 |  |
| probably approximately correct (PAC) | 可能近似正确（PAC） |  |
| polylogarithmic factor; logarithmic factor | 多对数因子；对数因子 | |
| warm-up phase; elimination | 预热期；淘汰 |  |
| finite-arm; linear bandits; contextual bandit | 有限臂；线性赌博机；情境赌博机 |  |

### 2.8 偏好模型与近似推断

| English | 中文 | 说明 |
|---|---|---|
| preference; preference relation; x ≻ x′ | 偏好；偏好关系；x 优于 x′ |  |
| preference learning | 偏好学习 |  |
| pairwise comparison | 成对比较 |  |
| duel; dueling | 对决 | 一次成对比较；定义后与“比较”互换 |
| psychophysics | 心理物理学 |  |
| Thurstone's law of comparative judgment; Case V | Thurstone 比较判断律；第五种情形 | |
| discriminal process; discriminal dispersion; scale value | 辨别过程；辨别离散度；量表值 | 标准用法 |
| Bradley-Terry model | Bradley-Terry 模型 | |
| Luce's choice axiom; Plackett-Luce model | Luce 选择公理；Plackett-Luce 模型 | |
| random utility model | 随机效用模型 |  |
| independence of irrelevant alternatives (IIA) | 无关选项独立性 |  |
| Gumbel noise; Gumbel distribution | Gumbel 噪声；Gumbel 分布 | |
| link function; link slope | 链接函数；链接斜率 |  |
| probit (link) | 概率单位（链接） | 不写 probit，见 §6 |
| logistic (link, function) | 逻辑（链接、函数） | |
| sigmoid; logit; multinomial logit; softmax | sigmoid 函数；logit；多项 logit；softmax | |
| noise scale σ; temperature τ | 噪声尺度；温度 |  |
| lapse rate | 失误率 |  |
| non-Gaussian likelihood | 非高斯似然 |  |
| Laplace approximation; mode | Laplace 近似；众数 |  |
| expectation propagation (EP); site; cavity distribution; moment matching | 期望传播；位点；空腔分布；矩匹配 |  |
| inverse Mills ratio | 逆 Mills 比 |  |
| variational inference; evidence lower bound (ELBO) | 变分推断；证据下界 |  |
| truncated Gaussian; orthant probability | 截断高斯；卦限概率 |  |
| posterior inference; approximate inference | 后验推断；近似推断 |  |
| identifiability; shift invariance | 可识别性；平移不变性 |  |
| comparison graph | 比较图 |  |
| ties; indifference; indifference threshold | 平局；无差异；无差异阈值 | |
| incomparability; incomplete preferences; incomplete answers | 不可比；不完备偏好；不完备回答 | |
| confidence (in an answer); preference strength | 把握度；偏好强度 | 回答者的把握；与统计的置信度区分 |
| ordinal labels; ratings; ranking; top-k ranking | 序数标签；评分；排序；top-k 排序 |  |
| set choice; batch winner; multi-option (query) | 集合选择；批次赢家；多选项（查询） | |
| absolute feedback; relative feedback; scalar feedback | 绝对反馈；相对反馈；标量反馈 | |

### 2.9 偏好贝叶斯优化、查询设计与对决赌博机

| English | 中文 | 说明 |
|---|---|---|
| dueling formulation; dueling space | 对决表述；对决空间 | |
| Condorcet winner | Condorcet 赢家 |  |
| Copeland winner; Copeland score; soft Copeland | Copeland 赢家；Copeland 得分；软 Copeland | |
| Borda winner; Borda score; Borda count | Borda 赢家；Borda 得分；Borda 计数 | |
| von Neumann winner | von Neumann 赢家 | |
| pure exploration; Copeland expected improvement | 纯探索；Copeland 期望改进 |  |
| dueling Thompson sampling; double Thompson sampling | 对决 Thompson 采样；双重 Thompson 采样 | |
| maximally uncertain challenge | 最大不确定挑战 |  |
| expected utility of the best option (EUBO) | 最优选项期望效用（EUBO） | 首次出现给中文名，之后写 EUBO |
| hallucination believer | 幻觉信念 |  |
| preference exploration; outcome | 偏好探索；结果 | BOPE 即偏好探索贝叶斯优化 |
| query design; query form | 查询设计；查询形式 | |
| sequential line search | 序列线搜索 |  |
| gallery; Sequential Gallery; slider | 画廊；序列画廊；滑块 |  |
| projective preferential BO | 投影偏好贝叶斯优化 |  |
| population prior | 群体先验 |  |
| dueling bandit; contextual dueling bandit | 对决赌博机；情境对决赌博机 |  |
| best-arm identification | 最优臂识别 |  |
| weak regret; strong regret; Borda regret; dueling regret | 弱遗憾；强遗憾；Borda 遗憾；对决遗憾 |  |
| preference-probability regret; utility regret; regret unit | 偏好概率遗憾；效用遗憾；遗憾单位 |  |
| relative upper confidence bound (RUCB) | 相对上置信界（RUCB） |  |
| stochastic transitivity (weak, moderate, strong); stochastic triangle inequality | 随机传递性（弱、中等、强）；随机三角不等式 |  |
| transitivity; intransitivity | 传递性；不可传递性 | |
| kernelized preference optimization | 核化偏好优化 |  |
| decision-theoretic (turn) | 决策论（转向） | 学科名 decision theory 写“决策理论” |
| information-theoretic | 信息论的 | |

### 2.10 研究前沿：观测模型、高维、软件与评测

| English | 中文 | 说明 |
|---|---|---|
| heteroscedastic; homoscedastic | 异方差；同方差 |  |
| structured noise; structured evaluation noise | 结构化噪声；结构化评价噪声 | |
| drift; preference drift; contamination | 漂移；偏好漂移；污染 |  |
| response time | 反应时 |  |
| hidden context; heterogeneous; aggregation | 隐藏情境；异质；聚合 |  |
| amortized; amortization | 摊销 |  |
| pretrained; in-context (surrogate) | 预训练；上下文（代理模型） | in-context 用“上下文”，contextual 用“情境” |
| prior-data fitted network (PFN); meta-learning | 先验拟合网络（PFN）；元学习 |  |
| failure mode; dimension ceiling | 失效模式；维度上限 |  |
| local PBO | 局部偏好贝叶斯优化 |  |
| reproduction; research code | 复现；研究代码 |  |
| evaluation methodology | 评测 | 方法论意义；单次函数评估用“评估” |
| placebo pairs; repeated pairs | 安慰剂对；重复配对 |  |
| test-retest; delayed retest | 重测；延迟重测 | |
| query-induced change; choice-induced preference change; revaluation | 查询引起的改变；选择引起的偏好改变；再评价 | |
| stable preference; lasting share | 稳定偏好；持久部分 |  |
| session; session length; fatigue; non-convergence | 会话；会话长度；疲劳；不收敛 |  |

### 2.11 人在回路：人机交互、健康与应用

| English | 中文 | 说明 |
|---|---|---|
| human-computer interaction (HCI); interactive design | 人机交互；交互式设计 |  |
| sense of agency | 能动感 | 哲学中的 agency 用“能动性”，见 §6.2 |
| ownership; expressiveness | 归属感；表达力 |  |
| workload; NASA Task Load Index (NASA-TLX) | 工作负荷；NASA 任务负荷指数 |  |
| Creativity Support Index | 创造力支持指数 |  |
| mixed-initiative | 混合主导 |  |
| novices; experts; individual differences | 新手；专家；个体差异 |  |
| explanation; trust; over-trust | 解释；信任；过度信任 | |
| field deployment | 现场部署 |  |
| exoskeleton; exosuit | 外骨骼；外骨骼服 |  |
| metabolic cost; indirect calorimetry | 代谢消耗；间接测热法 |  |
| gait cycle; peak torque | 步态周期；峰值力矩 | 见 §6.3 第 10 条 |
| prosthesis; transfemoral | 假肢；经股骨（截肢） |  |
| self-tuning; manual tuning | 自调；手动调节 |  |
| hearing aid | 助听器 |  |
| neuroprosthesis; retinal implant; spinal cord stimulation | 神经假体；视网膜植入物；脊髓刺激 | |
| built environment; thermal comfort; predicted mean vote (PMV) | 建成环境；热舒适；预测平均投票（PMV） |  |
| haptics; text entry; visualization design | 触觉；文本输入；可视化设计 |  |
| robot control; model predictive control (MPC) | 机器人控制；模型预测控制（MPC） |  |
| participant; between-subjects; within-subjects | 被试；被试间；被试内 | 实验与用户研究中统一用“被试” |
| crowdsourcing | 众包 |  |

### 2.12 相邻计算领域

| English | 中文 | 说明 |
|---|---|---|
| large language model (LLM) | 大语言模型 |  |
| reinforcement learning from human feedback (RLHF) | 基于人类反馈的强化学习（RLHF） |  |
| reward model; reward modeling | 奖励模型；奖励建模 |  |
| direct preference optimization (DPO) | 直接偏好优化（DPO） |  |
| reference policy; KL-regularized | 参考策略；KL 正则 |  |
| reward over-optimization; implicit reward | 奖励过度优化；隐式奖励 | |
| LLM judge | 大语言模型评判者 | |
| in-context learning | 上下文学习 |  |
| sycophancy; approval optimization | 谄媚；赞同优化 | |
| assistance game; AI alignment | 协助博弈；人工智能对齐 |  |
| preference-based reinforcement learning; inverse reinforcement learning | 基于偏好的强化学习；逆强化学习 |  |
| reward learning; trajectory segments; return; policy | 奖励学习；轨迹片段；回报；策略 |  |
| coactive feedback | 共同主动反馈 | |
| preference elicitation; active learning | 偏好引出；主动学习 |  |
| recommender system; feedback loop | 推荐系统；反馈回路 |  |
| filter bubble | 信息茧房 | 首次出现附英文，见 §6 |
| learning to rank; position bias; interleaving | 排序学习；位置偏差；交错实验 | |
| interactive evolutionary computation; quality-diversity | 交互式进化计算；质量多样性 |  |
| automated science; self-driving lab; high-throughput experimentation | 自动化科学；自驱动实验室；高通量实验 | |
| performative prediction; performative stability; induced preference shift | 表演性预测；表演性稳定；诱导偏好偏移 | |
| epistemic neural network | 认知神经网络 |  |
| agent | 智能体 |  |

### 2.13 心理学：判断与决策、心理物理学、社会心理学

| English | 中文 | 说明 |
|---|---|---|
| judgment and decision-making | 判断与决策 |  |
| psychometric function | 心理测量函数 |  |
| forced choice; two-alternative forced choice (2AFC) | 强制选择；二选一强制选择（2AFC） |  |
| just-noticeable difference (JND) | 最小可觉差 |  |
| Weber's law; Fechner's law; Stevens' power law | Weber 定律；Fechner 定律；Stevens 幂律 | |
| signal detection theory | 信号检测论 |  |
| range-frequency theory; assimilation; contrast | 范围频率理论；同化；对比 |  |
| context effects | 情境效应 |  |
| decoy effect; attraction effect; compromise effect; similarity effect | 诱饵效应；吸引效应；折中效应；相似性效应 |  |
| repulsion effect; distractor effect | 排斥效应；干扰项效应 | |
| order effects; recency; primacy; central tendency of judgment | 顺序效应；近因效应；首因效应；判断的趋中倾向 |  |
| preference reversal; procedure invariance | 偏好逆转；程序不变性 |  |
| constructive view; preference construction | 建构观；偏好建构 | |
| stable view; hierarchical view | 稳定观；层级观 | |
| preference discovery; noisy discovery | 偏好发现；带噪声的偏好发现 |  |
| preference scaffolding; preference crystallization; dormant preferences | 偏好支架；偏好结晶；休眠偏好 |  |
| ecological rationality; take-the-best | 生态理性；取最佳启发式 |  |
| satisficing; maximizers | 满意化；最大化者 |  |
| resource-rational analysis; mathematical psychology | 资源理性分析；数学心理学 | |
| choice overload | 选择过载 |  |
| loss aversion; prospect theory; reference dependence; endowment effect | 损失厌恶；前景理论；参照依赖；禀赋效应 |  |
| description-experience gap | 描述与经验差距 |  |
| decision fatigue; ego depletion | 决策疲劳；自我损耗 | |
| cognitive dissonance; free-choice paradigm; induced-compliance paradigm | 认知失调；自由选择范式；诱导服从范式 |  |
| spreading of alternatives | 选项分化 | 标准用法 |
| choice blindness | 选择盲 |  |
| mere exposure effect; processing fluency; feelings as information | 单纯曝光效应；加工流畅性；情感即信息 | |
| self-determination theory; hedonic adaptation | 自我决定理论；享乐适应 |  |
| moral psychology; sacred values | 道德心理学；神圣价值 | |
| heritability | 遗传力 | 见 §6 |
| circadian rhythm; neurodiversity | 昼夜节律；神经多样性 |  |
| test-retest reliability; intraclass correlation coefficient (ICC) | 重测信度；组内相关系数（ICC） |  |
| effect size; meta-analysis; preregistration | 效应量；元分析；预注册 | |
| publication bias; replication; multi-lab replication | 发表偏倚；复现；多实验室复现 |  |
| foot-in-the-door; question-behavior effect | 登门槛效应；提问-行为效应 | |
| regret aversion; regret (emotion) | 后悔厌恶；后悔 | |

### 2.14 神经科学与计算认知科学

| English | 中文 | 说明 |
|---|---|---|
| neuroeconomics; value-based decision; common currency | 神经经济学；基于价值的决策；共同货币 | |
| ventromedial prefrontal cortex; orbitofrontal cortex; dopamine | 腹内侧前额叶皮层；眶额皮层；多巴胺 | |
| neuromarketing; neuroforecasting | 神经营销；神经预测 | |
| sequential sampling model | 序贯抽样模型 | |
| drift-diffusion model (DDM); EZ-diffusion model | 漂移扩散模型；EZ 扩散模型 |  |
| drift rate; non-decision time | 漂移率；非决策时间 |  |
| efficient coding; divisive normalization | 有效编码；除法归一化 |  |
| predictive processing; free-energy principle | 预测加工；自由能原理 | |
| active inference; expected free energy | 主动推断；期望自由能 |  |
| psychopharmacology; addiction; motor learning; adaptation | 精神药理学；成瘾；运动学习；适应 | |

### 2.15 经济学、决策理论与社会选择

| English | 中文 | 说明 |
|---|---|---|
| behavioral economics | 行为经济学 |  |
| revealed preference | 显示偏好 |  |
| stated preference | 陈述偏好 |  |
| endogenous preferences; stochastic choice | 内生偏好；随机选择 | |
| rational inattention | 理性疏忽 |  |
| information economics; Bayesian persuasion; cheap talk; information design | 信息经济学；贝叶斯说服；廉价谈话；信息设计 | |
| social choice; mechanism design | 社会选择；机制设计 |  |
| strategy-proof; incentive compatible | 防策略；激励相容 | |
| Arrow's impossibility theorem; Condorcet paradox; single-peaked | Arrow 不可能定理；Condorcet 悖论；单峰 | |
| decision theory; expected utility | 决策理论；期望效用 | |
| multi-utility representation; lexicographic order | 多效用表示；字典序 | |
| compensatory; non-compensatory; limited consideration | 补偿式；非补偿式；有限考虑 | |
| discrete choice; conjoint analysis; part-worth; polyhedral methods | 离散选择；联合分析；部分价值；多面体方法 |  |
| contingent valuation; willingness to pay; hypothetical bias; scope insensitivity | 条件价值评估；支付意愿；假设偏差；范围不敏感 | |
| multi-criteria decision analysis (MCDA); swing weighting | 多准则决策分析；摆动赋权 |  |
| interactive multi-objective optimization; operations research | 交互式多目标优化；运筹学 |  |
| occasion noise (auditing and professional judgment) | 场合噪声（审计与专业判断） |  |
| experimental economics; game theory; Nash equilibrium | 实验经济学；博弈论；Nash 均衡 | |
| delay discounting | 延迟折扣 |  |

### 2.16 哲学、社会科学、自然科学与设计

| English | 中文 | 说明 |
|---|---|---|
| agency (philosophy); autonomy; manipulation | 能动性；自主；操纵 |  |
| transformative experience; meta-preferences; adaptive preferences | 转化性经验；元偏好；适应性偏好 | |
| incommensurability; parity | 不可通约性；对等 | |
| existentialism; phenomenology; 4E cognition; enactivism; pragmatism | 存在主义；现象学；4E 认知；生成主义；实用主义 |  |
| free will; illusionism; epistemology of verification | 自由意志；错觉论；验证的认识论 | |
| sociology of taste; cultural capital; habitus | 品味社会学；文化资本；惯习 |  |
| homophily; contagion; opinion dynamics; bounded confidence | 同质性；传染；舆论动力学；有界信任 | |
| WEIRD samples | WEIRD 样本 | |
| behavioral ecology; marginal value theorem | 行为生态学；边际值定理 | |
| second-order cybernetics; symmetry breaking | 二阶控制论；对称破缺 |  |
| differential privacy; randomized response | 差分隐私；随机响应 |  |
| Hodge decomposition; spectral graph theory | Hodge 分解；谱图理论 |  |
| design research; design fixation | 设计研究；设计固着 |  |
| lead user; jobs-to-be-done; Kano model | 领先用户；待完成任务；Kano 模型 | |
| prospect-refuge theory; biophilia | 瞭望庇护理论；亲生命性 | |
| sensory science; triangle test | 感官科学；三角检验 |  |
| empirical aesthetics; Wundt curve | 经验美学；Wundt 曲线 |  |
| health preference research; shared decision-making | 健康偏好研究；共同决策 | |

### 2.17 案例研究中的领域词

| English | 中文 | 说明 |
|---|---|---|
| classifier; gradient-boosted tree ensemble; support vector machine | 分类器；梯度提升树集成；支持向量机 |  |
| ligand; catalyst; solvent; base; yield | 配体；催化剂；溶剂；碱；产率 | |
| direct arylation; descriptors | 直接芳基化；描述符 | 标准用法 |
| photo enhancement | 照片增强 |  |

## 3. 保留英文的名称

这些名称在中文版中照英文原样书写，前后与汉字之间加空格。

### 3.1 方法与算法

有中文名的方法，首次出现按下表写，此后只写缩写或名称。

| 名称 | 首次出现的写法 |
|---|---|
| EUBO | 最优选项期望效用（expected utility of the best option，EUBO） |
| qEUBO | EUBO 的多选项形式 qEUBO（q 为一次查询展示的选项数） |
| BOPE | 偏好探索贝叶斯优化（BOPE） |
| POP-BO | 乐观算法 POP-BO |
| MaxMinLCB | 最大最小下置信界算法（MaxMinLCB） |
| PF-TS | 偏好反馈下的 Thompson 采样（PF-TS） |
| MR-LPF | 多轮偏好反馈学习算法（MR-LPF） |
| PABBO | 偏好摊销黑箱优化（PABBO） |
| PPBO | 投影偏好贝叶斯优化（PPBO） |
| LILO | 语言在回路优化（LILO） |
| StageOpt | 分阶段安全贝叶斯优化 StageOpt |
| GLISp | 以径向基函数为代理模型的 GLISp |
| DTS | 对决 Thompson 采样（DTS） |
| DSTS | 对决标量化 Thompson 采样（DSTS） |
| MPES | 多项预测熵搜索（MPES） |
| MUC | 最大不确定挑战（MUC） |
| CEI | Copeland 期望改进（CEI） |
| HB | 幻觉信念（HB） |
| BALD | 贝叶斯分歧主动学习（BALD） |
| PE；BPE | 纯探索（PE）；分批纯探索（BPE） |
| RUCB | 相对上置信界（RUCB） |
| qEI；qNEI | 批量期望改进（qEI）；带噪声批量期望改进（qNEI） |
| CMA-ES | 协方差矩阵自适应进化策略（CMA-ES） |
| DPO | 直接偏好优化（DPO） |
| RLHF | 基于人类反馈的强化学习（RLHF） |
| XPO | 探索性偏好优化（XPO） |
| Nash learning from human feedback | Nash 人类反馈学习 |
| PFN | 先验拟合网络（PFN） |
| PBGI | Pandora's Box Gittins 指数（PBGI） |

其余方法名直接写名称，不加中文：GP-UCB、GP-TS、UCB1、SelfSparring、KernelSelfSparring、CoSpar、LineCoSpar、LineSpar、ROIAL、POLAR、C-GLISp、TuRPBO、CrashPBO、CoExBO、GimmBO、MultiBO、BOgen、FontCraft、OptiCarVis、ProVoice、BlurDriving、AdaptiFont、PEBOL、OPEN、MAPLE、APOHF、APPO、IPO、PDO、SLiC、CheapVS、Duel-Evolve、MolSkill、ActiveUltraFeedback、BAL-PM、Meta-PO、SafeOpt、HOMI、KappaSharp、EIIG、LinESS、GIPBO、PrefSQP、NeuralUCB、TuRBO、SAASBO、BAxUS、REMBO、HEBO、SMAC、TPE、BOHB、FABOLAS、ParEGO、MAP-Elites、qNEHVI、qEHVI、LogEI、PPO、TabPFN、LLAMBO、BO-ICL、HIPE、EGO、HodgeRank、TrueSkill、PEBBLE、RIME、DemPref、APReL、NAUTILUS、ELECTRE、PROMETHEE、TOPSIS、SMAA。

MES、PES 在正文中写“最大值熵搜索”“预测熵搜索”，不用缩写。Sequential Gallery 与 sequential line search 是 Koyama 等人的方法名，但译作“序列画廊”与“序列线搜索”（§6.3 第 16 条），首次出现写“序列画廊（Sequential Gallery）”，此后写中文（§2.9）。

### 3.2 软件、平台与产品

BoTorch、Ax、PairwiseGP、GPyTorch、optuna-dashboard、Optuna、OptunaHub、AEPsych、PyTorch、NumPy、SciPy、scikit-learn、GPy、GPyOpt、GPflow、TensorFlow、JAX、NumPyro、MATLAB、NLopt、pySequentialLineSearch、prefGP、SkewGP、preferentialBO、SigOpt、PyPI、GitHub、arXiv、HoloLens、MuJoCo、Atalante、SoundSense Learn、Chatbot Arena。

软件中的类名、函数名与参数名放在反引号里，不翻译：`ScaleKernel`、`SingleTaskGP`、`PairwiseLaplaceMarginalLogLikelihood`、`PairwiseProbitLikelihood`、`PairwiseLogitLikelihood`、`qExpectedUtilityOfBestOption`、`AnalyticExpectedUtilityOfBestOption`、`LogExpectedImprovement`、`UpperConfidenceBound`、`get_covar_module_with_dim_scaled_prior`、`covar_module`、`HistGradientBoostingClassifier` 等。“optuna-dashboard”全小写，句首也不改大写。

### 3.3 模型

Transformer、GPT-3、GPT-4、GPT-4o、GPT-4o-mini、ChatGPT、Gemini、Llama、Claude、BART-large-MNLI、MusicVAE、VGG-16、AlphaGo、LoRA。泛称写“大语言模型”。

### 3.4 保留拉丁字母的人名术语

| 名称 | 中文写法 |
|---|---|
| Bradley-Terry | Bradley-Terry 模型；Bradley-Terry 链接 |
| Krippendorff | Krippendorff α 系数 |
| Mahalanobis | Mahalanobis 距离 |
| Thompson、Laplace、Markov、Chebyshev、Gibbs、Newton、Hilbert、Fourier、Bernoulli、Dirichlet、Pareto、Nash、Shannon、Weber、Fechner、Kano、Occam、Brown、Arrow | 见 §6.3 第 21 条 |
| Thurstone | Thurstone 模型；Thurstone 比较判断律 |
| Luce；Plackett-Luce | Luce 选择公理；Plackett-Luce 模型 |
| Condorcet | Condorcet 赢家；Condorcet 悖论 |
| Copeland | Copeland 赢家；Copeland 得分；软 Copeland |
| Borda | Borda 赢家；Borda 得分；Borda 计数 |
| von Neumann | von Neumann 赢家 |
| Matérn | Matérn 核；Matérn 5/2 |
| Cholesky | Cholesky 分解；Cholesky 因子 |
| Hessian | Hessian 矩阵 |
| Schur；Woodbury；Sherman-Morrison | Schur 补；Woodbury 恒等式；Sherman-Morrison 公式 |
| Mercer；Ornstein-Uhlenbeck | Mercer 定理；Ornstein-Uhlenbeck 过程 |
| Kullback-Leibler | Kullback-Leibler 散度；KL 散度 |
| Jensen | Jensen 不等式 |
| Gumbel | Gumbel 噪声；Gumbel 分布 |
| Beta；Gamma | Beta 分布；Gamma 分布（Γ 函数称伽马函数） |
| Sobol | Sobol 序列 |
| Gittins | Gittins 指数 |
| Hodge | Hodge 分解 |
| Arrow | Arrow 不可能定理 |
| Lai、Robbins | Lai-Robbins 下界 |
| Clark | Clark 公式 |
| Mills | 逆 Mills 比 |
| Kendall；Spearman | Kendall τ、Kendall 秩相关；Spearman 秩相关 |
| Cohen；Fleiss | Cohen's d、Cohen κ；Fleiss κ |
| Likert | Likert 量表 |
| Stevens | Stevens 幂律 |
| Wundt | Wundt 曲线 |
| Neyman | Neyman 正交 |
| Vickrey-Clarke-Groves | Vickrey-Clarke-Groves 机制 |
| Schelling | Schelling 焦点 |
| Hartmann、Branin、Ackley、Levy、Rosenbrock | 测试函数，写“6 维 Hartmann 函数” |

用汉字的例外见 §1.3；“Kano 模型”（Kano model）见 §6.3 第 21 条。

### 3.5 数据集与基准

HPO-B、YAHPO-Gym、LCBench、NATS-Bench、CIFAR-10、CIFAR-100、ImageNet16-120、SUSHI、MovieLens、BOPTEST、Mopta08、DTLZ2、ZDT3、OSY、Walker2D、LunarLander、MT-Bench、AlpacaEval、AlpacaFarm、UltraFeedback、PRISM、BIG-bench、LiveCodeBench、RouterBench、MixInstruct、MS MARCO、TREC、Planet49、OpenStreetMap、Many Labs 2（重复研究项目）。名称后可加“数据集”“基准”。

### 3.6 其他照原样保留的内容

- 论文题名、书名、期刊与会议名（ICML、NeurIPS、CHI、AISTATS、UIST、Psychological Science 等）与出版机构名。
- 代码与反引号中的一切；图模块名（`gp-posterior`、`pbo-oracle` 等，即 `src/figures/` 下的文件名）与图参数；交叉引用标识、引用键、文件路径、URL。
- 许可证名称：CC BY-NC-ND 4.0、BSD-3-Clause、MIT。
- 检索式中的原词：图 sw-publications 引用 arXiv 检索的原始检索词 preferential、Bayesian、optimization、optimisation，照原样保留，改译会误述检索。
- 统计符号：N、n、d、p、κ、τ。量表与指数首次出现给中文名：“NASA 任务负荷指数（NASA-TLX）”。
- 机构名：有通行中文名的大学与机构用中文（如“剑桥大学”“苏黎世联邦理工学院”“慕尼黑大学”），没有的保留英文（Meta、Google DeepMind、OMRON SINIC X）。

## 4. 禁用写法

### 4.1 机械检查清单

下表第一列的写法在中文版中不应出现。检查时对 `zh/` 下全部 Markdown 与图模块的 `labels.zh` 做固定字符串匹配，结果应为 0；本文件本身必然包含这些写法，检查时排除。

| 禁用写法 | 应写作 | 说明 |
|---|---|---|
| 拉普拉斯矩阵 | Laplace 矩阵 | §6.3 第 21 条 |
| 马氏距离 | Mahalanobis 距离 | §6.3 第 21 条 |
| 卡诺模型 | Kano 模型 | §6.3 第 21 条 |
| 布朗运动 | Brown 运动 | §6.3 第 21 条 |
| 阿罗不可能定理 | Arrow 不可能定理 | §6.3 第 21 条 |
| 汤普森 | Thompson | §6.3 第 21 条 |
| 拉普拉斯 | Laplace | §6.3 第 21 条 |
| 马尔可夫 | Markov | §6.3 第 21 条 |
| 切比雪夫 | Chebyshev | §6.3 第 21 条 |
| 伯努利 | Bernoulli | §6.3 第 21 条 |
| 牛顿 | Newton | §6.3 第 21 条 |
| 帕累托 | Pareto | §6.3 第 21 条 |
| 傅里叶 | Fourier | §6.3 第 21 条 |
| 吉布斯 | Gibbs | §6.3 第 21 条 |
| 韦伯 | Weber | §6.3 第 21 条 |
| 希尔伯特 | Hilbert | §6.3 第 21 条 |
| 纳什 | Nash | §6.3 第 21 条 |
| 香农 | Shannon | §6.3 第 21 条 |
| 奥卡姆 | Occam | §6.3 第 21 条 |
| 狄利克雷 | Dirichlet | §6.3 第 21 条 |
| 费希纳 | Fechner | §6.3 第 21 条 |
| 获取函数 | 采集函数 |  |
| 扭矩 | 力矩 | §6.3 第 10 条 |
| 塌缩 | 坍缩 | §6.3 第 18 条 |
| logit 链接 | 逻辑链接 | §6.3 第 14 条 |
| 贯穿目标函数 | 示例目标函数 | §6.3 第 11 条 |
| 贯穿全书的目标函数 | 贯穿全书的示例目标函数 | §6.3 第 11 条 |
| 采样函数 | 采集函数 | “从先验中抽取函数样本”也不写成“采样函数” |
| 置信上界 | 上置信界 |  |
| 置信下界 | 下置信界 |  |
| 现任最优 | 当前最优点（或当前最优值） |  |
| 代理函数 | 代理模型 |  |
| 决斗 | 对决 | 含“决斗赌博机” |
| 老虎机 | 赌博机 | 含“多臂老虎机” |
| 悔值 | 遗憾 |  |
| 最佳臂识别 | 最优臂识别 |  |
| 汤普森抽样 | 汤普森采样 |  |
| 探索与开发 | 探索与利用 |  |
| 探索-开发 | 探索与利用 |  |
| 探索和开发 | 探索与利用 |  |
| 边缘似然 | 边际似然 |  |
| 边缘分布 | 边际分布 |  |
| 边缘化 | 边际化 | 本书只有数学意义的 marginalize |
| 边缘概率 | 边际概率 |  |
| 长度刻度 | 长度尺度 |  |
| 平滑度 | 光滑度 |  |
| 马特恩 | Matérn |  |
| 高斯核 | 径向基函数核 | 本书不用 Gaussian kernel 这一名称 |
| 径向基核 | 径向基函数核 |  |
| 自动相关确定 | 自动相关性确定 |  |
| 楚列斯基 | Cholesky |  |
| 乔列斯基 | Cholesky |  |
| 海森矩阵 | Hessian 矩阵 |  |
| 黑塞矩阵 | Hessian 矩阵 |  |
| 海塞矩阵 | Hessian 矩阵 |  |
| 本征值 | 特征值 |  |
| 本征向量 | 特征向量 |  |
| 正半定 | 半正定 |  |
| 蒙特卡罗 | 蒙特卡洛 |  |
| 马尔科夫 | 马尔可夫 |  |
| 极大似然 | 最大似然 |  |
| 频率学派 | 频率派 |  |
| 贝叶斯法则 | 贝叶斯定理 |  |
| 贝叶斯规则 | 贝叶斯定理 |  |
| 变分推理 | 变分推断 |  |
| 贝叶斯推理 | 贝叶斯推断 |  |
| 近似推理 | 近似推断 |  |
| 后验推理 | 后验推断 |  |
| 贝塔分布 | Beta 分布 |  |
| 伽马分布 | Gamma 分布 | Γ 函数仍称伽马函数 |
| 伽马先验 | Gamma 先验 |  |
| 黑盒 | 黑箱 |  |
| 噪音 | 噪声 | 声学语境同样用“噪声” |
| 信任域 | 信赖域 |  |
| 超参数调优 | 超参数优化 |  |
| 潜在空间 | 潜空间 |  |
| 隐空间 | 潜空间 |  |
| 隐效用 | 潜在效用 |  |
| 潜效用 | 潜在效用 |  |
| 联系函数 | 链接函数 |  |
| 连接函数 | 链接函数 |  |
| 逻辑斯蒂 | 逻辑 |  |
| 逻辑斯谛 | 逻辑 |  |
| 罗吉斯蒂 | 逻辑 |  |
| 普罗比特 | 概率单位 |  |
| 配对比较 | 成对比较 |  |
| 顺序线搜索 | 序列线搜索 |  |
| 贪婪 | 贪心 | 含“ε-贪婪” |
| 可辨识性 | 可识别性 |  |
| 非传递 | 不可传递 |  |
| Condorcet 胜者 | Condorcet 赢家 |  |
| Copeland 胜者 | Copeland 赢家 |  |
| Borda 胜者 | Borda 赢家 |  |
| von Neumann 胜者 | von Neumann 赢家 |  |
| 孔多塞 | Condorcet |  |
| 科普兰 | Copeland |  |
| 博尔达 | Borda |  |
| 瑟斯顿 | Thurstone |  |
| 布拉德利 | Bradley |  |
| 强迫选择 | 强制选择 |  |
| 反应时间 | 反应时 |  |
| 最小可察觉差 | 最小可觉差 |  |
| 恰可察觉差 | 最小可觉差 |  |
| 偏好获取 | 偏好引出 |  |
| 偏好诱导 | 偏好引出 |  |
| 偏好反转 | 偏好逆转 |  |
| 发表偏差 | 发表偏倚 |  |
| 荟萃分析 | 元分析 |  |
| 遗传度 | 遗传力 |  |
| 词典序 | 字典序 |  |
| 大型语言模型 | 大语言模型 |  |
| 人类反馈强化学习 | 基于人类反馈的强化学习 |  |
| 自动驾驶实验室 | 自驱动实验室 |  |
| (推断) | （推断） | 半角括号 |

一个可用的检查命令（在仓库根目录执行）：

```bash
awk -F' [|] ' '/^### 4.1/{f=1;next} /^### 4.2/{f=0} f && /^[|] / && !/禁用写法|---/ {sub(/^[|] /,"",$1); print $1}' GLOSSARY.zh.md > /tmp/zh-forbidden.txt
grep -rnF -f /tmp/zh-forbidden.txt zh/ src/figures/
```

另外三项格式检查容易有误报（数学与代码中的半角符号），用于人工复核：

```bash
perl -CSD -ne 'print "$ARGV:$.: $_" if /\p{Han}"|"\p{Han}/' zh/*/*.md zh/*.md            # 直引号
perl -CSD -ne 'print "$ARGV:$.: $_" if /\p{Han}[,;:?!(]/' zh/*/*.md zh/*.md              # 汉字后接半角标点
perl -CSD -ne 'print "$ARGV:$.: $_" if /\p{Han}[A-Za-z0-9]|[A-Za-z0-9%]\p{Han}/' zh/*/*.md zh/*.md   # 缺空格
```

### 4.2 人工复核清单

下列写法在某些语境中是对的，不能机械禁止，审校时逐处判断。

| 写法 | 什么时候错 | 什么时候可以 |
|---|---|---|
| 后悔 | 指优化与赌博机中的 regret（应写“遗憾”） | 情绪与经济学：后悔厌恶、后悔理论 |
| 替代模型 | 指 surrogate（应写“代理模型”） | 指另一个候选模型（alternative model） |
| 两两比较 | 作名词指 pairwise comparison（应写“成对比较”） | 动词短语：“把所有选项两两比较” |
| 揭示偏好、显性偏好 | 指 revealed preference（应写“显示偏好”） | 动词短语“揭示……偏好”；“显性”指 explicit |
| 偏好启发 | 指 preference elicitation（应写“偏好引出”） | “偏好启发式方法”（倾向于用启发式） |
| 期望提升、提升概率 | 作 EI、PI 的名称（应写“期望改进”“改进概率”） | 一般动词短语：“提升……的概率” |
| 鲁棒 | 一般意义的 robust（应写“稳健”） | 固定术语“分布鲁棒优化” |
| 胜者 | Condorcet、Copeland、Borda、von Neumann 后面（应写“赢家”，见 4.1） | 指一次对决中被选中的选项时，写“获胜选项”更好 |
| 开发 | 指 exploitation（应写“利用”） | 软件开发、开发者 |
| 平滑 | 指 smoothness（应写“光滑度”） | 平滑处理、加一平滑、平滑箱形先验 |
| 推理 | 指 inference（应写“推断”） | 指 reasoning：大语言模型的推理 |
| 抽样 | 后验采样、Thompson 采样中的 sampling（应写“采样”） | 序贯抽样模型；调查抽样 |
| 评价 | 指对目标函数的一次 evaluation（应写“评估”） | 人给出的评分与评判；结构化评价噪声 |
| 置信度 | 指回答者对答案的把握（应写“把握度”） | 统计意义的置信水平 |
| 批量形式 | 用于 qEUBO（应写“多选项形式”） | qEI 作为期望改进的批量版本 |
| 被迫选择 | 作术语 forced choice（应写“强制选择”） | 一般叙述：“被迫在两者中选择” |
| 神谕 | 指作为应答者的 oracle（应写“预言机”） | 统计学的 oracle rate 写“神谕速率”；比喻“像神谕一样” |
| PBO、BO、GP、EI、UCB | 单独出现在中文正文中（若主编采纳 §6 第 1、2 条） | 方法名（GP-UCB、TuRPBO）、代码、数学式 |

## 5. 书名与界面用语

### 5.1 书名

- 书名：贝叶斯优化：从第一性原理到人类偏好（与 `zh/book.yml.draft` 一致）。
- 备选：贝叶斯优化：从基本原理到人类偏好。

“第一性原理”对应英文的 first principles，是字面译法，但在中文里常与物理学的第一性原理计算和商业口号联系在一起；“基本原理”更朴素，含义是“从最基本的概率与线性代数讲起”，与本书第一部分的内容一致。两者都准确，请主编选择（§6 第 27 条）。

### 5.2 部分标题

| 部分 | English | 中文 |
|---|---|---|
| 第一部分 | Foundations | 基础 |
| 第二部分 | Gaussian Processes | 高斯过程 |
| 第三部分 | Bayesian Optimization | 贝叶斯优化 |
| 第四部分 | Learning from Comparisons | 从比较中学习 |
| 第五部分 | Case Studies | 案例研究 |
| 第六部分 | The Research Frontier | 研究前沿 |
| 第七部分 | People in the Loop | 人在回路 |
| 第八部分 | Neighbors in Computing | 相邻计算领域 |
| 第九部分 | What Is a Preference? | 偏好是什么？ |
| 第十部分 | Synthesis | 综合 |

### 5.3 各章标题

| 文件 | English | 中文 |
|---|---|---|
| `index.md` | Bayesian Optimization | 贝叶斯优化 |
| `preface.md` | Preface | 前言 |
| `foundations/01-optimizing-the-unknown.md` | Optimizing What You Cannot Write Down | 优化写不出公式的函数 |
| `foundations/02-probability.md` | Probability as Bookkeeping for Uncertainty | 概率：为不确定性记账 |
| `foundations/03-linear-algebra.md` | The Linear Algebra of Uncertainty | 不确定性的线性代数 |
| `foundations/04-gaussian.md` | The Gaussian Distribution | 高斯分布 |
| `foundations/05-bayesian-inference.md` | Bayesian Inference | 贝叶斯推断 |
| `foundations/06-information.md` | Measuring Information | 度量信息 |
| `gp/01-distributions-over-functions.md` | Distributions over Functions | 函数上的分布 |
| `gp/02-gp-regression.md` | Gaussian Process Regression | 高斯过程回归 |
| `gp/03-kernels-and-hyperparameters.md` | Kernels and Hyperparameters | 核函数与超参数 |
| `bo/01-the-loop.md` | The Bayesian Optimization Loop | 贝叶斯优化循环 |
| `bo/02-acquisition-functions.md` | Acquisition Functions | 采集函数 |
| `bo/03-regret-and-bandits.md` | Regret, Bandits, and Guarantees | 遗憾、赌博机与理论保证 |
| `bo/04-bo-in-practice.md` | Bayesian Optimization in Practice | 贝叶斯优化实践 |
| `bo/05-applications.md` | Where Bayesian Optimization Works | 贝叶斯优化的用武之地 |
| `preferences/01-why-comparisons.md` | Why Ask for Comparisons | 为什么请人做比较 |
| `preferences/02-approximate-inference.md` | When the Posterior Is Not Gaussian | 后验不是高斯分布时 |
| `preferences/03-gp-preference-learning.md` | Gaussian Process Preference Learning | 高斯过程偏好学习 |
| `preferences/04-preferential-bo.md` | Preferential Bayesian Optimization | 偏好贝叶斯优化 |
| `preferences/05-query-design.md` | Designing the Question | 设计提问 |
| `preferences/06-dueling-bandits.md` | Dueling Bandits and the Theory of Comparisons | 对决赌博机与比较的理论 |
| `cases/01-tuning-a-classifier.md` | Tuning a Classifier | 为分类器调参 |
| `cases/02-chemical-reaction.md` | Optimizing a Chemical Reaction | 优化化学反应 |
| `cases/03-exoskeleton.md` | Tuning an Exoskeleton with a Person in the Loop | 人在回路中调节外骨骼 |
| `cases/04-photo-enhancement.md` | Enhancing a Photo by Comparison | 通过比较增强照片 |
| `frontier/01-a-decade-of-pbo.md` | A Decade of Preferential Bayesian Optimization | 偏好贝叶斯优化的十年 |
| `frontier/02-observation-models.md` | Observation Models, Surrogates, and Inference | 观测模型、代理模型与推断 |
| `frontier/03-acquisition-frontier.md` | Acquisition, Query Forms, and Problem Extensions | 采集函数、查询形式与问题扩展 |
| `frontier/04-theory.md` | Theory: From Dueling Bandits to Kernelized Preference Optimization | 理论：从对决赌博机到核化偏好优化 |
| `frontier/05-high-dimensions.md` | High Dimensions and the Changing Landscape of Bayesian Optimization | 高维问题与贝叶斯优化格局的变化 |
| `frontier/06-software-evaluation.md` | Software, Evaluation, and the Research Community | 软件、评测方法与研究社区 |
| `humans/01-interactive-design.md` | Interactive Design and Human-Computer Interaction | 交互式设计与人机交互 |
| `humans/02-body-and-health.md` | Wearable Robots, Health, and Assistive Technology | 可穿戴机器人、健康与辅助技术 |
| `humans/03-environments-science-industry.md` | Built Environments, Science, and Industry | 建成环境、科学与工业 |
| `neighbors/01-language-models.md` | Preferences and Large Language Models | 偏好与大语言模型 |
| `neighbors/02-adjacent-fields.md` | Reward Learning, Recommendation, Ranking, and Automated Science | 奖励学习、推荐、排序与自动化科学 |
| `perspectives/01-judgment-and-psychophysics.md` | Judgment, Decision, and Psychophysics | 判断、决策与心理物理学 |
| `perspectives/02-social-psychology.md` | Social, Affective, and Developmental Psychology | 社会、情感与发展心理学 |
| `perspectives/03-neuroscience.md` | Neuroscience and Computational Cognitive Science | 神经科学与计算认知科学 |
| `perspectives/04-economics.md` | Economics, Decision Theory, and Operations Research | 经济学、决策理论与运筹学 |
| `perspectives/05-philosophy.md` | Philosophy and Religious Traditions | 哲学与宗教传统 |
| `perspectives/06-social-sciences.md` | Social Sciences and the Humanities | 社会科学与人文学科 |
| `perspectives/07-natural-sciences.md` | Natural and Formal Sciences | 自然科学与形式科学 |
| `perspectives/08-design-and-senses.md` | Design, Sensory Science, Art, and Health | 设计、感官科学、艺术与健康 |
| `synthesis/01-what-a-comparison-measures.md` | What a Comparison Measures | 一次比较测量什么 |
| `synthesis/02-recommendations.md` | Building and Evaluating a Preferential Optimization System | 构建与评估偏好优化系统 |
| `synthesis/03-open-problems.md` | Open Problems and the Decisive Experiment | 未解决问题与判定实验 |
| `appendices/a-notation.md` | Notation | 记号 |
| `appendices/b-gaussian-identities.md` | Matrix and Gaussian Identities | 矩阵与高斯恒等式 |
| `appendices/c-minimal-implementation.md` | A Minimal Implementation | 最小实现 |
| `bibliography.md` | Bibliography | 参考文献 |

各章反复出现的小节标题：

| English | 中文 |
|---|---|
| The problem | 问题 |
| Exercises | 习题 |
| Further reading | 延伸阅读 |
| Settled, contested, missing | 已定、有争议与缺失 |
| Common claims, checked | 常见说法核查 |
| What it means for PBO / for preferential optimization | 对偏好贝叶斯优化的含义 / 对偏好优化的含义 |

### 5.4 编号对象与提示框

带编号的名称与 `src/pipeline/i18n.ts` 现有的中文设置一致；需要改动 i18n.ts 之处标在“依据”栏，并列入 §6。

| English | 中文 | 示例 | 依据 |
|---|---|---|---|
| Part | 部分 | 第二部分 | i18n.ts；部分用汉字数字，不加空格 |
| Chapter | 章 | 第 8 章 | i18n.ts |
| Section | 节 | 第 8.3 节 | i18n.ts |
| Appendix | 附录 | 附录 A | i18n.ts |
| Figure | 图 | 图 8.1 | i18n.ts |
| Equation | 式 | 式（8.4） | 全角括号（§1.2）；i18n.ts 现为“式 (8.4)”，见 §6 第 5 条 |
| Table | 表 | 表 8.1 | 标准用法 |
| Definition | 定义 | 定义 7.1 | 标准用法 |
| Theorem | 定理 | 定理 4.1 | 标准用法 |
| Lemma；Proposition；Corollary | 引理；命题；推论 | 引理 4.3 | 标准用法 |
| Example | 例 | 例 3.2 | i18n.ts；标准用法 |
| Exercise；Exercises（小节） | 习题 | 习题 3.1 | i18n.ts |
| Solution | 解答 | | i18n.ts |
| Algorithm | 算法 | 算法 2.1 | 标准用法 |
| Proof | 证明 | | 标准用法 |
| Derivation | 推导 | | 标准用法 |
| Key idea | 要点 | | i18n.ts |
| Note | 注 | | i18n.ts |
| Aside（标签 Going deeper，标记 optional） | 深入一步 | | i18n.ts；英文的“optional”标记写“选读” |
| Pitfall | 易错点 | | i18n.ts |
| Frontier（标签 Research status） | 研究现状 | | i18n.ts；这类提示框的标题“Settled, contested, missing”写“已定、有争议与缺失” |
| Recap | 回顾 | | i18n.ts |
| In code | 代码实现 | | i18n.ts |
| Further reading | 延伸阅读 | | 标准用法 |
| References | 参考文献 | | i18n.ts |
| Contents | 目录 | | i18n.ts |
| Bibliography | 参考文献 | | i18n.ts |
| Sources cited in Section 8.3 | 第 8.3 节引用的文献 | | i18n.ts |
| (inference) | （推断） | | §1.5 |

### 5.5 证据标注

| English | 中文 | 依据 |
|---|---|---|
| preprint | 预印本 | i18n.ts |
| working paper | 工作论文 | i18n.ts |
| workshop paper | 研讨会论文 | i18n.ts |
| software | 软件 | i18n.ts；另一种写法是“软件文档”，见 §6 第 6 条 |
| non-peer-reviewed | 非同行评审 | i18n.ts 现为“未经同行评审”，见 §6 第 6 条 |
| thesis | 学位论文 | i18n.ts |
| only title (and abstract) verified | 仅核实题名与出处 | 正文中说明只读到题名或摘要时用 |

## 6. 需要裁定的选择

### 6.1 判断依据

1. **偏好贝叶斯优化的缩写。** 建议全书首次出现写“偏好贝叶斯优化（preferential Bayesian optimization，PBO）”，此后正文写全称，以免中文行文夹杂过多拉丁字母缩写。代价：英文版 PBO 共 724 次，最密的章节是 `neighbors/02`（84）、`perspectives/08`（70）、`neighbors/01`（62）、`perspectives/06`（60）、`perspectives/04`（58）、`perspectives/07`（54）、`frontier/06`（53），这些章节写全称会比英文长。另一种做法是沿用英文版约定（每章首次写“偏好贝叶斯优化（PBO）”，此后写 PBO）；§2 与 §4 不受影响，只需改 §1.5 一条与 §4.2 最后一行。
2. **一般缩写写全称。** BO、GP、EI、UCB、KG、TS 同理写中文全称。英文版中 BO 104、GP 108、EI 81、UCB 79 次。
3. **弯引号。** 按要求用“”，这是中文出版物的标准用法；直引号与「」都不用。
4. **作者连接词。** 书面语中连接作者用“与”比“和”正式：两位作者写“X 与 Y”，三位作者写“A、B 与 C”。`src/pipeline/i18n.ts` 的中文设置现为 `and: "和"`、`etAl: "等"`，构建生成“Chu 和 Ghahramani（2005）”“González 等（2017）”。建议 `and` 改为“与”；叙述式引用用“等人”，括号式引用保留“等”，这需要 `cite.ts` 区分两种模式。本文件没有改动代码。
5. **式的括号。** `i18n.ts` 现为“式 (8.4)”，半角括号且有空格。建议改为“式（8.4）”，与 §1.2 一致。
6. **证据标注两处。** “non-peer-reviewed”：可写“非同行评审”，`i18n.ts` 用“未经同行评审”；两者都对，建议用较短的“非同行评审”。“software”：可标“软件文档”，`i18n.ts` 标“软件”；书中这一标注贴在软件条目本身（代码仓库、软件包）上，“软件”更准确，建议保留。
7. **Bayes' rule → 贝叶斯定理。** “贝叶斯定理”在中文机器学习教材中最通行；“贝叶斯公式”是概率论教材的叫法，也可接受。
8. **incumbent → 当前最优点；f*_n → 当前最优值。** “现任最优”是 incumbent 的字面译法，不用（§4.1）。
9. **probit → 概率单位。** probit 原是 probability unit 的缩写，“概率单位”是它的中文名。不少中文统计教材直接写 probit；本书正文写中文名（作者裁定）。
10. **logistic → 逻辑。** 写“逻辑链接”“逻辑函数”；名词审定的“逻辑斯谛”未采用（作者裁定）。
11. **link function → 链接函数。** 广义线性模型教材多写“联系函数”，这里取“链接函数”（作者裁定），并把“联系函数”列入 §4。
12. **filter bubble → 信息茧房。** 中文里常用“信息茧房”指这一现象；字面译法是“过滤气泡”，而“信息茧房”原本对应 Sunstein 的 information cocoons。本书取“信息茧房”，首次出现写“信息茧房（filter bubble）”。若主编更看重与英文概念一一对应，可改为“过滤气泡”。
13. **Weber's law → Weber 定律；Fechner's law → Fechner 定律。** 按心理学教材通行写法用译名。这是 §1.3“人名不音译”的例外。
14. **heritability → 遗传力。** 按遗传学通行名词用“遗传力”，把“遗传度”列入 §4。
15. **forced choice → 强制选择。** “强迫选择”列入 §4。2AFC 写“二选一强制选择”，心理物理学教材常见的“二择一迫选”未采用。
16. **greedy → 贪心。** 按计算机科学通行的“贪心算法”取“贪心”，“贪婪”列入 §4。
17. **年代写法。** 英文版有 1940s 至 2000s 共 9 处；建议写“20 世纪 60 年代”“21 世纪头十年”，不写“1960 年代”，这是国内出版物的通行写法。
18. **大数不换算。** 大数有两种通行写法（“10,000”与“1 万”）。为便于与英文版逐一核对，建议一律保留阿拉伯数字加逗号分节。
19. **surprise → 意外度（信息量）。** “自信息”是信息论的正式名称，但英文版这里用的是口语化的 surprise，“意外度”保留了这层意思；首次出现可写“意外度（又称自信息）”。
20. **kriging believer → 克里金信念；constant liar → 常数说谎者；fantasy → 虚拟观测。** 仿照“幻觉信念”（hallucination believer，§2.9）的构词译出，首次出现附英文。也可以直接保留英文名。
21. **RBF kernel → 径向基函数核。** 中文正文写全称“径向基函数”。英文版 RBF 出现 81 次；首次写“径向基函数（RBF）核”，此后写“径向基函数核”。图中标签受空间限制时可用“RBF 核”。
22. **Gaussian distribution → 高斯分布。** 中文概率教材多写“正态分布”，但本书始终说 Gaussian，且“高斯过程”与之相连，故用“高斯分布”；“standard normal”仍写“标准正态分布”。
23. **prior-data fitted network → 先验拟合网络。** 字面更完整的“先验数据拟合网络”未采用。
24. **oracle → 预言机。** “预言机”是计算机科学中 oracle 的常见译名（如回归预言机、偏好预言机）。本书 oracle 指应答者时写“预言机”；`perspectives/03` 中统计学的 oracle rates 写“神谕速率”；比喻义写“神谕”。
25. **mixed-initiative → 混合主导。** 人机交互文献也常写“混合主动”。
26. **preference elicitation → 偏好引出。** “偏好获取”“偏好诱导”列入 §4。
27. **副书名。** 见 §5.1。
28. **constructive / stable / hierarchical view → 建构观 / 稳定观 / 层级观。** “建构论”“层级论”多用作专名；作观点名时本书取“观”。哲学流派 constructivism 另写“建构主义”。
29. **lint 的取舍。** “两两比较”“揭示偏好”“期望提升”“提升概率”“鲁棒”在某些语境中正确，只列入 §4.2 人工复核，没有放进 §4.1。“边缘化”列入 §4.1，因为英文版只在数学意义上用 marginalize；若以后出现社会学意义的 marginalized，需要把它移到 §4.2。
30. **代码注释。** 建议把围栏代码块（附录 C 与各章“代码实现”提示框）中的注释译成中文，代码、字符串与输出不变；Python 3 对中文注释没有限制。若希望中英文版代码逐字相同、便于同步维护，也可以不译。

### 6.2 同一英文词的两种译法

| English | 语境 A | 语境 B |
|---|---|---|
| agency | 能动感：人机交互中的 sense of agency | 能动性：哲学中的 agency |
| regret | 遗憾：优化与赌博机 | 后悔：情绪与经济学（后悔厌恶、后悔理论） |
| sampling | 采样：后验采样、Thompson 采样、切片采样 | 抽样：序贯抽样模型、调查抽样 |
| evaluation | 评估：对目标函数的一次评估 | 评价：人的评分与评判；评测：方法论意义的 evaluation |
| context | 情境：情境赌博机、情境效应、隐藏情境 | 上下文：上下文学习、上下文代理模型 |
| smooth | 光滑：光滑度、Matérn 的光滑度参数 | 平滑：平滑处理、加一平滑 |
| batch | 批量：qEI、并行评估 | 批次：批次赢家；qEUBO 不称“批量形式” |
| confidence | 把握度：回答者对答案的把握 | 置信：置信区间、上置信界 |
| interval | 可信区间：贝叶斯后验区间 | 置信区间：频率派区间 |
| incumbent | 当前最优点：点 | 当前最优值：f*_n |
| kernel | 核函数：单独出现时 | 核：复合词（Matérn 核、核矩阵、核化） |
| Gaussian / normal | 高斯：高斯分布、高斯过程 | 正态：标准正态分布、偏斜正态 |
| decision theory | 决策理论：学科名 | 决策论：形容词用法（决策论转向、决策论规则） |
| inference | 推断：统计推断、（推断）标记 | 推理：reasoning |
| constructive | 建构观：关于偏好的观点 | 建构主义：哲学流派 |
| winner | 赢家：Condorcet 赢家等定义的对象 | 获胜选项：一次对决中被选中的选项 |
| question / query | 查询：系统向人提出的一次比较 | 提问：标题与日常叙述（“设计提问”） |

### 6.3 主编裁定（2026 年 10 月 2 日）

以下裁定覆盖上文 §1 至 §6.2 中与之不同的建议。

1. **偏好贝叶斯优化：** 采纳 §6 第 1 条。正文写全称；全书第一次出现写“偏好贝叶斯优化（preferential Bayesian optimization，PBO）”。表格单元格与图中标签空间不足时可写 PBO。
2. **一般缩写：** 采纳 §6 第 2 条。正文写中文全称（贝叶斯优化、高斯过程、期望改进、上置信界……）；数学式中的宏不变；表格与图中标签空间不足时可用缩写。
3. **引用格式：** 采纳 §6 第 4 条，构建已改：叙述式“Lin 等人（2022）”，括号式“（Lin 等，2022）”，两位作者“Chu 与 Ghahramani”。
4. **式的写法：** 采纳 §6 第 5 条，构建已改为“式（8.4）”；只写编号时为“（8.4）”。
5. **证据标注：** “非同行评审”；“软件”保留。构建已改。
6. **副书名：** 采用“从基本原理到人类偏好”（§5.1 的备选）。
7. **代码注释：** 译成中文，代码、字符串与输出不变。
8. **段内换行（§1.7）：** 不必把段落写成一行。构建会去掉两个汉字之间、以及全角标点两侧的段内换行；汉字与拉丁字母或数字之间的换行显示为一个空格，符合 §1.1。
9. **大数（修订 §6 第 18 条）：** 英文用 million、billion 这类数量词写的数，中文用“万”“亿”：2.2 million 写“220 万”，26 billion 写“260 亿”。英文用数字写出的精确数照写阿拉伯数字并逗号分节：“1,225”“40,000”。公式里的数不动；英文版中数字在公式内、数量词在公式外的写法已改为科学记数法（$31^6 \approx 9 \times 10^8$），两版一致。
10. **力矩（修订 §2.11）：** torque 写“力矩”（外骨骼与机器人文献的通行写法），peak torque 写“峰值力矩”。“扭矩”列入禁用。
11. **示例目标函数：** the running objective 在每章首次出现写“贯穿全书的示例目标函数”，此后写“示例目标函数”。“贯穿目标函数”“贯穿全书的目标函数”不用。
12. **汽车驾驶室设计问题：** qEUBO 论文的 Carcab（7 维 car cab design problem）写“汽车驾驶室设计问题”。
13. **thumbstick：** 写“拇指杆”，Schäfer 等人的设备写“拇指杆遥控器”。
14. **逻辑链接：** 经济学文献所说的 logit link 就是 logistic link，统一写“逻辑链接”。logit 作为模型名或变换名仍写 logit（多项 logit 模型）。
15. **Arrow 不可能定理：** 见第 21 条。
16. **序列画廊：** Sequential Gallery 按 §3.1 首次写“序列画廊（Sequential Gallery）”，此后写“序列画廊”。
17. **全书主旨的写法：** “The bottleneck has moved from algorithms to measurement”写“瓶颈已经从算法转移到测量”（前言与第六部分导言已如此）；“the hard part has moved to measurement”写“难点已经转移到测量上”。
18. **坍缩：** collapse 写“坍缩”（作者裁定），不写“塌缩”。
19. **文体：** 中文版用书面语，按国内优秀教材的行文：简洁、准确、克制；既不用翻译腔，也不用口语（“补上”“走一遍”“说了算”“不妨”等）。叙述中少用“你”，用“读者”或无主语句；图中的操作说明可以用“你”。“Some things to try”写“可以尝试以下几点”。
20. **Chebyshev 标量化：** 多目标优化中的 Chebyshev scalarization 写“Chebyshev 标量化”（作者裁定）。
21. **人名：** 外国人名一律保留拉丁字母，包括以人名命名的定理、不等式、方法与分布：Markov 不等式、Chebyshev 不等式、Hoeffding 不等式、Laplace 近似、Thompson 采样、Gibbs 采样、Newton 法、Hilbert 空间、Fourier 变换、Bernoulli 分布、Dirichlet 分布、Pareto 前沿、Nash 均衡、Shannon 熵、Arrow 不可能定理、Weber 定律、Kano 模型、Occam 剃刀、Mahalanobis 距离、Brown 运动。这是国内研究生数学教材（如张恭庆《泛函分析讲义》、程士宏《测度论与概率论基础》）的通行写法（作者裁定）。例外：贝叶斯（贝叶斯优化、贝叶斯推断）与高斯（高斯过程、高斯分布）是本书领域与书名的用语，保留汉字；蒙特卡洛是地名，欧氏距离为通行简称，亦保留。
22. 其余各条（§6 第 3、7 至 17、19 至 26、28、29 条）按本文件的建议执行。

### 6.4 翻译中新增的译名

译者在翻译中确定、本文件原先没有的译名。全书统一按此写；终稿前做一次一致性检查。

| English | 中文 | 首次确定于 |
|---|---|---|
| the running objective（贯穿全书的一维示例函数） | 贯穿全书的示例目标函数（首次）；此后“示例目标函数” | bo/05（zh5） |
| acceleration factor | 加速因子 | bo/05（zh5） |
| enhancement factor | 增强因子 | 同上，按同一构词 |
| tilted distribution（期望传播中乘入精确因子后的分布） | 倾斜分布 | preferences/02（zh5）；与 skew（偏斜，如偏斜高斯过程、偏斜正态）是两个不同的概念 |
| method of constant stimuli | 恒定刺激法 | preferences/01（zh5） |
| worth（Bradley-Terry 的参数） | 价值 | preferences/01（zh5） |
| half-normal distribution | 半正态分布 | preferences/02（zh5） |
| skewness | 偏度 | preferences/02（zh5） |
