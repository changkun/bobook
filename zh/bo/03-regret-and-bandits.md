---
status: done
synopsis: "优化器的评分方法：简单遗憾与累积遗憾、多臂赌博机及其经典算法、Lai-Robbins 下界、基于最大信息增益的 GP-UCB 遗憾界，以及这类保证在实践中的意义与局限。"
sources: ["Lai and Robbins 1985","Auer et al. 2002","Srinivas et al. 2010","Lattimore and Szepesvari 2020","Abbasi-Yadkori et al. 2011","Chowdhury and Gopalan 2017"]
---

# 遗憾、赌博机与理论保证 {#sec-regret}

@sec-acquisition 介绍了一系列采集函数，每一种都合理地回答了下一步该在哪里评估的问题。要在其中做出选择，或者提出新的采集函数，就需要有办法判断一个优化器优于另一个。通常的做法是绘制基准测试图：在若干测试函数上，画出已找到的最优值随评估次数的变化。后续章节也会使用这类图。本章要求更可靠的依据：一个对任何问题都有定义的分数，以及关于这个分数的结论，这些结论对某一明确类别中的每个问题都成立。

这个分数称为**遗憾**（regret）。遗憾的理论最初围绕一个比贝叶斯优化更简单的问题发展起来，即多臂赌博机：赌徒面对几台回报率未知的赌博机，反复选择拉动哪一台。赌博机问题只保留了 @sec-explore-exploit 中探索与利用权衡的最基本结构。在这一设定下，可以统计算法做了多少探索，证明最好的算法的探索次数只按对数增长，并证明任何算法都无法探索得更少。随后再把这套分析推广回函数：高斯过程把无穷多条相关的臂化为复杂度可以度量的问题，GP-UCB 算法的界就用这一复杂度表示。

即使跳过证明，也建议读者阅读最后一节。遗憾界是关于理想化问题的精确论断，对实践有一些有用的启示，对另一些问题则无从回答；借助本章的图，读者可以看到界与算法的实际表现可能相差多远。

## 为优化器打分 {#sec-regret-definitions}

设想两个优化器都在 @sec-bo-loop 中贯穿全书的示例目标函数上运行。第一个把大部分评估用在较高的峰附近；第二个在大部分预算内遍历整个定义域，最后推荐了同一个峰。哪一个更好？这取决于沿途的评估除了本身的花费之外是否还有其他代价，即关心的是整个旅程，还是只关心终点。两种遗憾分别把这两种回答精确化。

考虑在定义域 $\X$ 上最大化未知函数 $f$。与 @eq-loop-problem 相同，记最优值为 $f^\star = \max_{\vx \in \X} f(\vx)$，取到最优值的某个输入为 $\vx^\star$。优化器依次在 $\vx_1, \vx_2, \dots$ 处评估 $f$（可能带噪声），$T$ 次评估后推荐一个输入 $\hat\vx_T$，通常取观测到的最好输入，或后验均值的最大值点。

::: {.definition #def-regret-kinds title="遗憾"}
第 $t$ 次评估的**瞬时遗憾**（instantaneous regret）是所选输入相对最优值的差额：
$$
r_t = f^\star - f(\vx_t) \;\ge\; 0.
$$
$T$ 次评估后的**累积遗憾**（cumulative regret）是沿途各次差额之和；**简单遗憾**（simple regret）则只评价最终推荐：
$$
R_T = \sum_{t=1}^{T} r_t,
\qquad
s_T = f^\star - f(\hat\vx_T).
$$ {#eq-regret-defs}
:::

遗憾按所选输入处的真实 $f$ 计算，而不是按带噪声的观测值计算。优化器不知道 $f^\star$，因此永远无法算出遗憾。遗憾是供分析者使用的分数：在最大值已知的基准问题上可以计算，定理所界定的也正是这个量。

::: {.example #ex-regret-three title="终点相同，旅程不同"}
设 $f^\star = 1$。优化器 A 评估了三个输入，函数值分别为 $0.2$、$0.7$、$0.9$，瞬时遗憾分别为 $0.8$、$0.3$、$0.1$，因此 $R_3 = 1.2$；若推荐其中最好的输入，则 $s_3 = 0.1$。优化器 B 评估的三个输入，函数值都是 $0.9$。B 的简单遗憾同样是 $s_3 = 0.1$，累积遗憾却只有 A 的四分之一，即 $R_3 = 0.3$。
:::

每个 $r_t$ 都不超过 $f$ 的取值范围，所以累积遗憾至多随 $T$ 线性增长。从不学习的优化器，例如始终均匀随机查询的优化器，累积遗憾确实线性增长：每次评估的平均损失相同。若优化器的累积遗憾**次线性**（sublinearly）增长，即增长得比任何直线都慢，从而平均遗憾 $R_T / T$ 趋于零，则称该优化器是**无遗憾**（no-regret）的。形如 $R_T \le C\sqrt{T}$ 的界意味着平均遗憾按 $1/\sqrt{T}$ 下降；形如 $R_T \le C \log T$ 的界意味着后期几乎所有评估都用在接近最优的输入上。

累积遗憾的界同时界定了已访问的最好输入的简单遗憾。$T$ 个数中的最小值不超过它们的平均值，因此

$$
f^\star - \max_{t \le T} f(\vx_t) \;=\; \min_{t \le T} r_t \;\le\; \frac{R_T}{T}.
$$ {#eq-regret-simple-from-cumulative}

累积遗憾的界正是通过这种方式转化为优化的收敛速率[@srinivas2010gaussian]。这里有两点需要注意。第一，观测带噪声时，优化器不知道评估过的输入中哪一个的 $f$ 最大，因此访问过的最好输入不等于优化器能识别出的最好输入；最终报告什么，本身就是一个实际问题（@sec-practice-noise）。第二，反之不成立：均匀探索的优化器可以有很小的简单遗憾，同时累积遗憾线性增长。在下一节的赌博机问题中，这一差别十分明显。在固定的问题上，均匀探索的简单遗憾随预算呈指数下降；而保持累积遗憾较低的算法会尽早停止对接近最优的竞争者采样，其简单遗憾只按多项式速度下降[@bubeck2009pure; @lattimore2020bandit, 第 33 章]。

选用哪个分数，取决于由谁承担评估的代价。为模型做超参数优化或搜索新材料时，只有最终推荐会被采用；沿途的评估只是以算力或实验时间计量的成本，此时简单遗憾是合适的分数。在线实验中，每次评估都是展示给真实用户的一个产品变体；偏好研究中，每个选项都需要一个人去看、去穿戴或去听；此时过程本身很重要，合适的分数是累积遗憾。@sec-pbo-failure-modes 介绍了一种偏好方法，其最终答案更好，累积遗憾却是竞争方法的 2.5 倍以上[@xu2024principledb]。

最后一个区别在于界针对的函数范围。**频率派**（frequentist）界对某一明确类别中的每个函数都成立，例如具有给定光滑度的全部函数。**贝叶斯**（Bayesian）界则对从先验中抽取的函数在平均意义下成立，或以高概率成立。两种界在 @sec-gp-bandits 中都会出现。

::: {.keyidea title="终点与旅程"}
简单遗憾评价最终推荐，累积遗憾评价沿途的每一次评估。累积遗憾低，则已访问的最好输入的简单遗憾也低；反之不成立。
:::

## 多臂赌博机 {#sec-bandits}

要统计探索的次数，需要一个足够简单、便于计数的设定，**多臂赌博机**（multi-armed bandit）正是如此。设有 $K$ 个动作，因赌博机的拉杆而称为**臂**（arm）。拉动臂 $i$ 得到随机奖励，其分布固定但未知，均值为 $\theta_i$。玩家每轮拉动一条臂，共 $T$ 轮，目标是使总奖励最大。这类问题最早的规则由 Thompson 于 1933 年提出[@thompson1933likelihood]；1952 年，Robbins 将其作为实验序贯设计中的一个问题正式表述，并引入了遗憾的概念[@robbins1952some; @lattimore2020bandit, 第 4 章]。

赌博机可以看作做了两处简化的贝叶斯优化：定义域是 $K$ 个输入组成的有限集合，且这些输入互不相关，拉动臂 3 不提供关于臂 4 的任何信息。其余要素全部保留，包括噪声、预算，以及两种做法之间的权衡：继续尝试看似最好的臂，或检查其他可能更好的臂。

本节中每次奖励都相当于抛一次硬币：拉动臂 $i$ 以概率 $\theta_i$ 得 1，否则得 0。记最优臂的均值为 $\theta^* = \max_i \theta_i$，臂 $i$ 的**差距**（gap）为 $\Delta_i = \theta^* - \theta_i$，即每次拉动该臂而非最优臂时的平均损失。（本节中 $\theta_i$ 表示臂的平均奖励；符号 $\mu$ 仍专用于高斯过程的后验均值。）

赌博机算法的遗憾，就是以各臂均值代替 $f$ 的累积遗憾：$R_T = \sum_{t} (\theta^* - \theta_{a_t})$，其中 $a_t$ 为第 $t$ 轮拉动的臂。这一定义使用的是均值，而非实际的抛硬币结果，因此有时也称为**伪遗憾**（pseudo-regret）。记 $N_i(T)$ 为前 $T$ 轮中臂 $i$ 的拉动次数，将求和按臂分组，得到

$$
\E[R_T] = \sum_{i=1}^{K} \Delta_i \, \E[N_i(T)].
$$ {#eq-regret-decomposition}

这一分解把问题化为记账：遗憾等于每条较差臂的拉动次数按其差距加权之和。算法要保持低遗憾，就应少拉坏臂；但只有拉动一条臂，才能知道它是坏臂。整个领域研究的就是一个问题：拉动多少次才够。

在介绍算法之前，读者可以先亲自尝试。下图隐藏了五条臂的回报率，共有 50 次拉动机会。

```{figure}
//| figure: regret-bandits
//| label: fig-regret-you
//| fig-cap: "亲手拉动这些臂。每次拉动按该臂隐藏的概率得 1（实心方块）或 0（空心方块）。遗憾曲线的斜率会暴露最优臂，因此在揭晓各臂或用完全部 50 次拉动之前，曲线保持隐藏。此后，你的这次运行将与后面几节中各算法 20 次运行的平均值比较；这些算法的第一次运行面对的回报序列与你相同。“换一组臂”抽取一组新的臂。"
mode: you
```

多数读者在每条臂拉动两三次之后，就已有了偏爱的臂。读者是否回头拉过一条开局连输两次的臂？回报率在 0.15 至 0.6 之间时，最优臂连输两次的概率为 16%，因此这么早放弃一条臂，无异于一场赌博。下面的每个算法，都是针对这一决定的一条规则。

### 贪心与 ε-贪心 {#sec-regret-eps-greedy}

最简单的规则是**贪心**（greedy）：每条臂先各拉一次，之后始终拉动当前平均奖励最高的臂。贪心规则的失败方式颇具启发性。若最优臂第一次拉动恰好得 0，其平均值便为 0；只要某条较差臂的平均值为正，贪心规则就可能再也不拉动最优臂。这种情况以某个固定的正概率发生，一旦发生，遗憾将永远线性增长。

**ε-贪心**（epsilon-greedy）通过强制探索弥补这一缺陷：每轮以概率 $\varepsilon$ 拉动一条均匀随机选取的臂，否则拉动贪心臂。这样每条臂都会被拉动无穷多次，每个平均值都收敛到相应的均值。但这一补救要付出永不终止的代价。$\varepsilon$ 为常数时，每轮都有概率 $\varepsilon$ 用在均匀随机的臂上，每轮的期望遗憾因此增加 $\frac{\varepsilon}{K} \sum_i \Delta_i$，遗憾随之线性增长，斜率至少为此值（@exr-regret-eps-linear）。Auer、Cesa-Bianchi 与 Fischer 证明，若在第 $n$ 轮令 $\varepsilon$ 按 $\varepsilon_n = \min\{1, cK/(\Delta_0^2 n)\}$ 衰减，可以得到对数遗憾；但前提是 $\Delta_0$（原论文记作 $d$）是最优臂与次优臂之间差距的下界，而玩家并不知道这个值。在他们的实验中，没有任何一个 $c$ 值能在所试的全部奖励分布上都表现良好[@auer2002finite]。

### 置信界从何而来 {#sec-regret-concentration}

以固定比率强制探索，会不断拉动已知很差的臂。更好的规则只在关于某条臂的证据仍然薄弱时才探索它，为此需要一个数：拉动 $n$ 次后，这条臂的平均奖励与其均值可能相差多远？答案来自**集中不等式**（concentration inequalities），即平均值偏离均值超过给定距离的概率的界。本章的每个遗憾上界都以某个集中不等式为基础；选用哪个不等式，决定了据此构造的算法中的常数与对数项。

固定一条均值为 $\theta$ 的臂，设 $Y_1, \dots, Y_n$ 是它的 $n$ 个奖励，相互独立且都在 $[0, 1]$ 中，平均值为 $\hat\theta_n = \frac1n \sum_{s=1}^{n} Y_s$。问题是：对偏差 $a > 0$，**尾概率**（tail probability）$\Prob(\hat\theta_n \ge \theta + a)$ 可能有多大。下尾 $\Prob(\hat\theta_n \le \theta - a)$ 的处理方式相同，因此下面只讨论上尾。

**Markov 不等式**（Markov's inequality）。一个非负且均值很小的量，不可能经常取大值：如果它不小于 $c$ 的时间比例超过 $\E[Z]/c$，仅这些情形就会使其均值超过 $\E[Z]$。对 $Z \ge 0$ 与 $c > 0$，

$$
\Prob(Z \ge c) \le \frac{\E[Z]}{c}.
$$ {#eq-regret-markov}

证明如下：$Z$ 不小于 $c$ 乘以事件 $Z \ge c$ 的指示函数（事件发生时为 1，否则为 0），对两边取期望即得。平均值 $\hat\theta_n$ 非负，均值为 $\theta$，因此 $\Prob(\hat\theta_n \ge \theta + a) \le \theta/(\theta + a)$。对均匀硬币且 $a = 0.1$，这个界为 $0.83$，而且无论对多少个奖励取平均，它始终是 $0.83$。Markov 不等式只利用均值，而平均值的均值不随 $n$ 变化。

**Chebyshev 不等式**（Chebyshev's inequality）。补救的办法是把 Markov 不等式用于确实随 $n$ 缩小的量。偏差的平方 $(\hat\theta_n - \theta)^2$ 的均值为 $\Var[\hat\theta_n] = \Var[Y]/n$，因为独立项之和的方差等于各项方差之和（@sec-independence），而和除以 $n$ 会使方差除以 $n^2$。事件 $\hat\theta_n \ge \theta + a$ 蕴含 $(\hat\theta_n - \theta)^2 \ge a^2$，因此

$$
\Prob(\hat\theta_n \ge \theta + a) \le \frac{\Var[Y]}{n a^2}.
$$ {#eq-regret-chebyshev}

（Chebyshev 不等式同时界定两侧的尾部，因此也界定了每一侧。）对硬币，$\Var[Y] = \theta(1 - \theta)$，至多为 $\tfrac14$（@sec-prob-variance）。这个界现在按 $1/n$ 下降，但反过来用时代价很高。令右边等于目标失效概率 $\delta$，得宽度 $a = \sqrt{\Var[Y]/(n\delta)}$，它按 $1/\sqrt{\delta}$ 增长。赌博机算法需要非常小的失效概率，下一小节的算法在第 $t$ 轮要求小到 $t^{-4}$；宽度若按 $t^2$ 增长，每条臂都会永远留在考虑范围之内。实际情况要好得多。由中心极限定理，许多独立项的平均值近似服从高斯分布（@sec-gauss-why），而高斯分布的尾部随标准差个数 $c$ 按 $e^{-c^2/2}$ 下降，而不是按 $1/c^2$ 下降。

**Chernoff 方法**。把 Markov 不等式用于指数函数，就能体现这种行为[@lattimore2020bandit, 第 5 章]。对任意 $\lambda > 0$，事件 $\hat\theta_n - \theta \ge a$ 与事件 $e^{\lambda n(\hat\theta_n - \theta)} \ge e^{\lambda n a}$ 相同，而指数函数把平均值中的和变成积：$e^{\lambda n(\hat\theta_n - \theta)} = \prod_{s=1}^{n} e^{\lambda(Y_s - \theta)}$。独立因子之积的期望等于各因子期望之积，因为它们的联合分布可以分解（@def-prob-independence）。剩下的是对单个因子的界，提供这个界的条件有专门的名称。

::: {.definition #def-regret-subgaussian title="次高斯"}
称均值为零的随机变量 $Z$ 是 **$R$-次高斯**（sub-Gaussian）的，若对每个实数 $\lambda$ 都有
$$
\E\big[e^{\lambda Z}\big] \le e^{\lambda^2 R^2 / 2}.
$$
:::

均值为零、标准差为 $R$ 的高斯变量使上式取等号，因此这一条件的含义是：$Z$ 的尾部不比该高斯分布的尾部更重。有界变量也满足这一条件。由 **Hoeffding 引理**（Hoeffding's lemma），均值为零且始终落在区间 $[l, u]$ 中的变量是 $\tfrac12(u - l)$-次高斯的[@lattimore2020bandit, 第 5 章]。因此，$[0, 1]$ 中的奖励减去其均值是 $\tfrac12$-次高斯的；均值为零、绝对值从不超过 $\sigma$ 的噪声是 $\sigma$-次高斯的。（字母 $R$ 沿用下文所引论文的记法，与遗憾 $R_T$ 无关。）

在这一条件下，Chernoff 方法给出 **Hoeffding 不等式**（Hoeffding's inequality），这是 Hoeffding 针对有界随机变量之和证明的结果[@hoeffding1963probability]：对 $[0, 1]$ 中的奖励，

$$
\Prob(\hat\theta_n \ge \theta + a) \le e^{-2 n a^2}.
$$ {#eq-regret-hoeffding}

::: {.derivation title="用 Chernoff 方法证明 Hoeffding 不等式"}
1. 把 @eq-regret-markov 用于 $e^{\lambda n(\hat\theta_n - \theta)}$，取 $c = e^{\lambda n a}$，再利用上面的乘积，得 $\Prob(\hat\theta_n - \theta \ge a) \le e^{-\lambda n a} \prod_{s=1}^{n} \E\big[e^{\lambda(Y_s - \theta)}\big]$，对每个 $\lambda > 0$ 成立。
2. 每个 $Y_s - \theta$ 都是 $R$-次高斯的，因此每个因子至多为 $e^{\lambda^2 R^2/2}$，界变为 $\exp(-\lambda n a + n \lambda^2 R^2/2)$。
3. 指数作为 $\lambda$ 的函数是一条抛物线，在 $\lambda = a/R^2$ 处取最小值 $-n a^2/(2R^2)$。第 1 步对每个 $\lambda$ 都成立，对这一个当然也成立：$\Prob(\hat\theta_n - \theta \ge a) \le e^{-n a^2/(2R^2)}$。
4. $[0, 1]$ 中的奖励有 $R = \tfrac12$，代入即得 @eq-regret-hoeffding。
:::

反过来看，Hoeffding 不等式表明：以至少 $1 - \delta$ 的概率，平均值低于 $\theta + \sqrt{\ln(1/\delta)/(2n)}$。失效概率现在通过其对数进入宽度。把 $\delta$ 从 0.05 缩小到 $10^{-6}$，Hoeffding 区间的宽度变为原来的 2.1 倍，Chebyshev 区间则变为 224 倍。把同样的方法用于标准差为 1 的高斯变量 $Z$，得 $\Prob(Z \ge c) \le e^{-c^2/2}$；直接计算可将这个界减半[@srinivas2010gaussian, 引理 5.1]，因此两侧尾部的总概率至多为 $e^{-c^2/2}$。这就是 @sec-regret-gpucb-bound 中 GP-UCB 证明第 1 步所用的高斯界。

::: {.aside title="硬币情形下更紧的指数"}
Hoeffding 不等式只利用奖励的取值范围。对抛硬币，Chernoff 方法可以利用整个分布，给出 $\Prob(\hat\theta_n \ge \theta + a) \le e^{-n\,\mathrm{kl}(\theta + a,\, \theta)}$，其中 $\mathrm{kl}$ 是 @eq-info-kl-bernoulli 中两枚硬币之间的 Kullback-Leibler 散度[@lattimore2020bandit, 引理 10.3]。Pinsker 不等式 $\mathrm{kl}(p, q) \ge 2(p - q)^2$（@sec-regret-lower-bounds 还会用到）表明，这个界绝不弱于 $e^{-2na^2}$。对均匀硬币，两个指数几乎相同；对均值接近 0 或 1 的硬币，两者差别最大，因为这类硬币的方差很小。基于 $\mathrm{kl}$ 版本构造的上置信界 KL-UCB，对 Bernoulli 臂达到了 @sec-regret-lower-bounds 中的 Lai-Robbins 常数[@lattimore2020bandit, 定理 10.6]。
:::

**联合界**（union bound）。算法用到的不止一个区间：每一轮、每条臂各有一个，其分析需要所有区间同时成立，至少需要统计区间失效的次数。所需的工具是初等的：若干事件中至少有一个发生的概率，不超过各事件概率之和，

$$
\Prob(A_1 \text{ or } A_2 \text{ or } \cdots \text{ or } A_m) \le \sum_{j=1}^{m} \Prob(A_j),
$$ {#eq-regret-union}

因为任何一个事件发生的结果，在右边至少被计入一次。联合界对事件之间如何相互依赖没有任何要求。要使 $m$ 个区间以至少 $1 - \delta$ 的概率同时成立，只需给每个区间分配失效概率 $\delta/m$。利用 Hoeffding 不等式，宽度变为

$$
a = \sqrt{\frac{\ln(m/\delta)}{2n}} = \sqrt{\frac{\ln m + \ln(1/\delta)}{2n}},
$$ {#eq-regret-union-width}

因此区间的个数以加性的 $\ln m$ 出现在根号下。若用 Chebyshev 不等式，宽度会按 $\sqrt{m}$ 增长。正是指数型的尾部，使同时维持许多区间的代价可以承受。

赌博机算法中的对数正是由此而来。一个区间若要在时域 $T$ 内的每一轮都成立，需要 $m = T$，代价为 $\ln T$；每一轮为 $K$ 条臂各设一个区间，代价为 $\ln(KT)$。事先不知道时域时，可以把预算 $\delta$ 不均匀地分配，给第 $t$ 轮分配 $6\delta/(\pi^2 t^2)$。由于 $\sum_{t \ge 1} 1/t^2 = \pi^2/6$，各轮份额之和恰为 $\delta$；第 $t$ 轮的宽度中以 $\ln(\pi^2 t^2/(6\delta))$ 代替 $\ln(m/\delta)$，因此宽度按 $\sqrt{\ln t}$ 增长。GP-UCB 的 $\beta_t$ 就是这一构造，再对 $|\X|$ 个输入多取一次联合界（@sec-regret-gpucb-bound）。下一小节的 UCB1 在第 $t$ 轮把每个区间的失效概率设为 $t^{-4}$。由 $e^{-2na^2} = t^{-4}$ 解出 $a$，得 $a = \sqrt{2\ln t / n}$，即 @eq-regret-ucb1 中的加成项。指数 4 用于支付另一次联合：在第 $t$ 轮，参与比较的两条臂各自的拉动次数可以是不超过 $t$ 的任意值，次数组合约有 $t^2$ 种，而 $t^2 \cdot t^{-4} = t^{-2}$ 对所有轮次求和仍然有限。

**样本量由数据决定时**。Hoeffding 不等式讨论的是 $n$ 个奖励的平均值，其中 $n$ 在看到奖励之前就已固定。赌博机算法则根据已看到的奖励决定一条臂拉动多少次，开局不利的臂被拉动得更少。读者也许会问：每个奖励仍是如实抽取的，这是否有影响？确实有影响。抛一枚均匀硬币，一旦正面次数超过反面次数就停止。停止时的平均值总是大于二分之一，而且停止的可能性很大：100 次之内停止的概率为 0.92，1000 次之内为 0.97。对每个固定的 $n$，Hoeffding 不等式对前 $n$ 次的平均值依然成立；但对于看过抛掷结果之后才选定的 $n$，它不提供任何结论。

赌博机分析用联合界来弥补这一点。设想每条臂的奖励是开局之前就已抽好的一个列表，算法只决定每个列表读到第几项；这一模型赋予算法所见一切的概率与原问题相同[@lattimore2020bandit, 第 4.6 节]。对每个固定的 $n$，列表的前 $n$ 项是 $n$ 个独立的奖励，因此 Hoeffding 不等式对每个 $n$ 分别成立，再对 $n = 1, \dots, t$ 取联合界，就覆盖了算法实际达到的任何次数。这次联合就是上文的 $t^2$。它并非形式上的手续。若在每个次数上都使用单轮宽度 $\sqrt{\ln(1/\delta)/(2n)}$，取 $\delta = 0.05$，均匀硬币的区间在 1000 次拉动内至少失效一次的概率为 0.11，超过 $\delta$ 的两倍；在 10,000 次拉动内为 0.15，因为累计平均值的最大摆动缩小得比 $1/\sqrt{n}$ 稍慢[参见 @lattimore2020bandit, 习题 20.9]。若改用上文不均匀分配所得的宽度，1000 次拉动内的同一概率为 $8.5 \times 10^{-6}$：联合界是安全的，在这里还相当保守。（这些概率与上文均匀硬币的概率一样，都是逐次跟踪正面次数的分布精确算出的。）高斯过程的情形更难，因为每次评估都会改变各处的估计，权重又取决于算法选择在哪里观测；@sec-regret-self-normalized 给出处理这种情形的工具。

下图以硬币为例，把三个不等式并列比较。

```{figure}
//| figure: regret-concentration
//| label: fig-regret-concentration
//| fig-cap: "均值为 $\theta$ 的硬币抛 $n$ 次，平均值不小于 $\theta + a$ 的概率（实线，由二项分布精确算出），与三个界对照：Markov 不等式给出的 $\theta/(\theta + a)$，与 $n$ 无关；使用硬币自身方差的 Chebyshev 不等式给出的 $\theta(1 - \theta)/(na^2)$；以及 @eq-regret-hoeffding 中的 Hoeffding 界。两个坐标轴均为对数刻度，点线标出 0.05。读数给出标记的 $n$ 处的各个值，以及每条曲线从哪个 $n$ 起保持在 0.05 以下。“对 T 个平均值取联合界”大于 1 时，每条曲线都针对 $T$ 个必须同时低于 $\theta + a$ 的平均值，例如每轮一个或每条臂一个：每个界都乘以 $T$（@eq-regret-union），精确曲线则变为 $T$ 个独立平均值中至少有一个超过该线的概率。正面次数是整数，因此精确概率随 $n$ 呈锯齿状变化；几个 $n$ 落在同一像素上时，曲线取其中的最大值。"
```

可以尝试以下几点。

**查看默认设置**。对均匀硬币且 $a = 0.1$，抛 100 次平均值不小于 0.6 的概率为 0.028。Hoeffding 不等式给出的上界为 0.14，Chebyshev 不等式为 0.25，Markov 不等式为 0.83。读数给出每条曲线从多少次抛掷起保持在 0.05 以下：精确概率为 76 次，Hoeffding 界为 150 次，Chebyshev 界为 500 次，Markov 界则始终不能。

**把标记移到 $n = 1000$**。精确概率为 $1.4 \times 10^{-10}$，Hoeffding 界为 $2.1 \times 10^{-9}$，Chebyshev 界仍为 0.025。精确曲线与 Hoeffding 曲线以几乎相同的指数速率下降，两者之间的差距增长缓慢，从 $n = 100$ 时的约 5 倍增至 $n = 1000$ 时的 15 倍；Chebyshev 曲线在这样的坐标轴上是一条直线，只按 $1/n$ 下降。

**把“对 T 个平均值取联合界”设为 1000**。每个界都乘以 1000。Hoeffding 界现在从 496 次而不是 150 次起保持在 0.05 以下，多出 $\ln 1000/(2a^2) \approx 345$ 次；Chebyshev 界则从 500,000 次而不是 500 次起，是原来的一千倍。对 1000 个独立的平均值，精确概率需要 381 次。这就是 @eq-regret-union-width 中的 $\ln m$，只是从样本量的角度来看。

**把“硬币的均值 θ”调到 0.1**，并把“对 T 个平均值取联合界”恢复为 1。很少得奖的硬币方差为 0.09 而不是 0.25，其精确概率从 36 次起保持在 0.05 以下。利用方差的 Chebyshev 界改善到 180 次；Hoeffding 界只知道奖励落在 $[0, 1]$ 中，仍为 150 次，在 $n = 100$ 处给出 0.14，而精确值只有 0.002。同时利用方差的不等式（如 Bernstein 不等式）能弥补这一差距的大部分[@lattimore2020bandit, 习题 5.14]。

::: {.keyidea title="置信宽度中的两项代价"}
置信宽度要为两件事付出代价：一是平均值的尾部下降得多快，对有界奖励，Hoeffding 不等式把尾部界定为 $e^{-2na^2}$；二是有多少个区间必须同时成立，联合界把这一代价计为一个加性的对数项。UCB1 的加成项 $\sqrt{2 \ln t / N_i}$ 同时包含这两项代价。
:::

### 乐观：UCB1 {#sec-regret-ucb1}

更好的规则来自 @sec-ucb 中所说的面对不确定性时的乐观原则。对每条臂，根据其迄今为止的奖励，算出均值的合理上限，然后拉动合理上限最大的臂。经常被拉动的臂区间窄，合理上限接近其平均奖励。很少被拉动的臂区间宽，合理上限较为宽松，因此还会得到尝试。明显较差的臂，一旦区间收缩到最优臂的均值以下，就不再被拉动。

区间应取多宽？对取值于 $[0, 1]$ 的奖励，由 Hoeffding 不等式（@eq-regret-hoeffding），$n$ 个独立奖励的平均值 $\hat\theta_{i}$ 高估均值超过 $a$ 的概率至多为 $e^{-2na^2}$，低估的概率同样如此。在第 $t$ 轮选取宽度，使该概率等于 $t^{-4}$（理由见 @sec-regret-concentration），便得到 @auer2002finite 的 UCB1 规则：每条臂各拉一次之后，拉动

$$
a_t = \argmax_{i} \left( \hat\theta_i + \sqrt{\frac{2 \ln t}{N_i}} \right),
$$ {#eq-regret-ucb1}

其中 $N_i$ 为臂 $i$ 迄今的拉动次数，$\hat\theta_i$ 为其平均奖励。一条臂每被拉动一次，加成项按 $1/\sqrt{N_i}$ 缩小；臂被搁置时，加成项按 $\sqrt{\ln t}$ 增长。因此没有哪条臂会被永久放弃，而坏臂只会偶尔被重新拉动。

::: {.theorem #thm-regret-ucb1 title="UCB1（Auer、Cesa-Bianchi 与 Fischer，2002）"}
设有 $K > 1$ 条臂，各臂的奖励分布支撑在 $[0, 1]$ 上。对任意轮数 $T$，UCB1 的期望遗憾至多为
$$
8 \sum_{i:\,\Delta_i > 0} \frac{\ln T}{\Delta_i}
\;+\; \left(1 + \frac{\pi^2}{3}\right) \sum_{j=1}^{K} \Delta_j .
$$
:::

这一证明的梗概值得一读，因为同样的三个步骤还会在 @sec-gp-bandits 的高斯过程界中出现：一是以高概率成立的置信区间，二是论证乐观的代价至多为区间宽度，三是统计区间较宽的情形最多出现多少次。

::: {.derivation title="坏臂为什么约被拉动 ln T / Δ² 次"}
固定一条次优臂 $i$。若某条臂到第 $t$ 轮已被拉动 $n$ 次，记其加成项为 $a(n, t) = \sqrt{2 \ln t / n}$。以下是 @auer2002finite 中定理 1 证明的梗概。

1. 由 Hoeffding 不等式，$n$ 个奖励的平均值在给定方向上偏离均值超过 $a(n, t)$ 的概率至多为 $e^{-2n \cdot 2\ln t / n} = t^{-4}$。
2. 设臂 $i$ 已被拉动 $n \ge 8 \ln T / \Delta_i^2$ 次。对 $n$ 求解该不等式可知，对每个 $t \le T$ 都有 $a(n, t) \le \Delta_i / 2$。
3. 若两个区间都未失效，最优臂的指标至少为 $\theta^*$，臂 $i$ 的指标至多为 $\theta_i + 2a(n, t) \le \theta_i + \Delta_i = \theta^*$。因此臂 $i$ 不可能在比较中胜出，只有当两个区间之一失效时才会被拉动。
4. 到第 $t$ 轮，每条臂的拉动次数可以是不超过 $t$ 的任意值。由联合界（@eq-regret-union），把第 1 步的失效概率 $t^{-4}$ 对两条臂所有可能的拉动次数求和，结果至多为 $2t^2 \cdot t^{-4} = 2t^{-2}$；再对所有轮次求和，期望意义下至多多出 $2\sum_t t^{-2} = \pi^2/3$ 次拉动。
5. 综合以上各步，$\E[N_i(T)] \le 8 \ln T/\Delta_i^2 + 1 + \pi^2/3$。按 @eq-regret-decomposition 乘以 $\Delta_i$ 并对各臂求和，即得定理。
:::

这个界通过各臂的差距依赖于具体问题。远差于最优臂的臂很快被淘汰，贡献很小；与最优臂几乎一样好的臂各贡献 $\ln T / \Delta_i$，$\Delta_i$ 很小时这个量很大。@sec-ucb 的上置信界规则基于同一思路，只是以高斯过程的后验标准差代替 Hoeffding 宽度。

### Thompson 采样 {#sec-regret-thompson}

最古老的规则是贝叶斯式的：把每个未知均值 $\theta_i$ 视为带先验的随机量，随时更新其后验，每轮以某条臂为最优臂的概率拉动该臂。Thompson 的巧妙之处在于，这个概率无须计算：只要从每条臂的后验中各抽取一个合理的均值，再拉动抽取值最大的臂即可[@thompson1933likelihood]。

对抛硬币式的奖励，后验是 Beta 分布。从均匀先验出发，若一条臂赢了 $S_i$ 次、输了 $F_i$ 次，其后验为 $\mathrm{Beta}(1 + S_i, 1 + F_i)$，即 @sec-beta-binomial 中的共轭更新。后验均值接近该臂的平均奖励，离散程度随拉动次数增加而缩小。

::: {.algorithm #alg-regret-thompson title="Bernoulli 臂上的 Thompson 采样"}
输入：$K$ 条臂，时域 $T$。

1. 对每条臂令 $S_i \leftarrow 0$，$F_i \leftarrow 0$。
2. 对每一轮 $t = 1, \dots, T$，为每条臂独立抽取 $\tilde\theta_i \sim \mathrm{Beta}(1 + S_i, 1 + F_i)$。
3. 拉动 $a_t = \argmax_i \tilde\theta_i$，观测奖励 $y_t \in \{0, 1\}$。
4. 若 $y_t = 1$，将 $S_{a_t}$ 加一；否则将 $F_{a_t}$ 加一。
:::

探索来自随机抽取。拉动次数少的臂后验很宽，有时会抽出很高的值；拉动次数多且平均奖励低的臂则几乎不会。Thompson 的规则起初流传不广。2010 年前后，几个研究组重新发现了它，并在实验中发现其表现很强，它才流行起来，而当时还没有任何证明[@lattimore2020bandit, 第 36 章]。此后，@agrawal2012analysis 首次证明其期望遗憾按对数增长；@kaufmann2012thompson 与 @agrawal2013further 证明，对 Bernoulli 奖励，其首项常数达到最优，即下一节下界中的常数。这条规则的高斯过程版本从后验中抽取整个函数，在抽取的函数取最大值处评估，也就是 @sec-thompson 中的 Thompson 采样。

### 观察算法的表现 {#sec-regret-bandit-race}

下图在同一组臂上将三个算法各运行多次，画出各算法的平均累积遗憾，区间带覆盖居中 80% 的运行。每次运行中，三个算法面对相同的回报序列，因此差异来自规则本身，而非运气。

```{figure}
//| figure: regret-bandits
//| label: fig-regret-bandits
//| fig-cap: "ε-贪心（ε 为常数）、UCB1 与 Thompson 采样在 Bernoulli 臂上的累积遗憾，各臂均值见表；曲线为 20 次运行的平均值，区间带为各次运行的第 10 至第 90 百分位。表格给出各算法的拉动在各臂上的分布，以占全部轮次的比例表示。虚线是 @sec-regret-lower-bounds 中的 Lai-Robbins 速率 $c^* \ln t$，它是关于增长速度的渐近结论，并非每个 $t$ 处的下限。各臂均值仅作示意。"
```

可以尝试以下操作。

**打开“对数时间轴”**。在对数时间轴上，按 $\ln t$ 增长的遗憾是一条直线，线性增长的遗憾则急剧向上弯曲。Thompson 采样在几百轮后就稳定在一条直线上，ε-贪心则向上翘起。在这一时域内，UCB1 仍介于两者之间：差距为 0.1 时，其加成项较为保守，次优臂在数千轮内都仍在考虑范围之内，曲线要到更晚才变直。

**把“ε-贪心的 ε”设为 0**。此时即为贪心规则。区间带变宽：多数运行稳定在最优臂上，少数运行锁定在某条较差的臂上，不再离开。把“运行次数”设为 1，再按几次“换一组臂”，可以看到单次运行的结果。

**在默认时域下比较 UCB1 与 ε-贪心**。对这组臂，1,000 轮后 $\varepsilon = 0.1$ 的 ε-贪心遗憾低于 UCB1：UCB1 的加成项偏于保守，用额外的探索换取理论保证。把“差距 Δ”提高到 0.3，“轮数 T”提高到 5,000，ε-贪心的直线最终会超过 UCB1 的对数曲线。

**打开“显示 UCB1 保证”**。坐标轴会随之拉伸。默认设置下，$T = 1000$ 时 @thm-regret-ucb1 的界约为 1,100，是最差玩法（始终拉动最差的臂）损失量 450 的两倍多。这一保证成立，但在这一时域内不提供任何信息。

**观察 Thompson 采样**。Thompson 采样在此处遗憾最低，且在多数设置下，其曲线位于 Lai-Robbins 虚线下方。下一节解释为何这并不矛盾。

## 下界 {#sec-regret-lower-bounds}

UCB1 与 Thompson 采样的遗憾都按 $\ln T$ 增长。更聪明的算法能否做得更好，使遗憾以常数为界？@lai1985asymptotically 给出了否定的回答。

直观的解释与证据有关。算法要停止拉动一条较差的臂，必须确信这条臂并非暗中的最优臂。误弃最优臂会在剩余轮次中造成 $\Delta T$ 量级的损失，因此犯这种错误的概率必须控制在 $1/T$ 量级。要在这一水平上排除备择假设，所需证据量按 $\ln T$ 增长；而臂 $i$ 每被拉动一次，平均提供固定量的证据，即 **Kullback-Leibler 散度** $\mathrm{kl}(\theta_i, \theta^*)$。它是每次拉动在两个假设之间的平均对数似然比，两个假设分别为该臂回报率为 $\theta_i$ 和回报率为 $\theta^*$。对硬币而言，

$$
\mathrm{kl}(p, q) = p \ln\frac{p}{q} + (1 - p) \ln\frac{1 - p}{1 - q},
$$

当 $p = q$ 时其值为零；两枚硬币越容易区分，其值越大（@sec-kl 讨论一般情形下的散度）。所需证据量除以每次拉动提供的证据量，得到臂 $i$ 约需拉动 $\ln T / \mathrm{kl}(\theta_i, \theta^*)$ 次。

要把这一直觉变成定理，必须排除这样一类算法：它们在某个问题上碰巧表现很好，代价是在其他问题上表现极差。例如始终拉动臂 1 的规则，只要臂 1 是最优臂，其遗憾就为零。若一个算法在该类的每个赌博机上，遗憾都比 $T$ 的任何幂次增长得慢，即对每个 $a > 0$ 都有 $\E[R_T] / T^a \to 0$，则称该算法是**一致的**（consistent）。

::: {.theorem #thm-regret-lai-robbins title="Lai 与 Robbins（1985），Bernoulli 臂"}
对每个一致的算法和每个满足 $\theta^* < 1$ 的 Bernoulli 赌博机，
$$
\liminf_{T \to \infty} \frac{\E[R_T]}{\ln T} \;\ge\; c^*
\;=\; \sum_{i:\,\Delta_i > 0} \frac{\Delta_i}{\mathrm{kl}(\theta_i, \theta^*)}.
$$ {#eq-regret-lai-robbins}
:::

这是奖励分布属于参数族时一般结果的特例；Lattimore 与 Szepesvári 给出了其现代形式的陈述与证明[@lai1985asymptotically; @lattimore2020bandit, 定理 16.2]。

以下三点说明这一定理与前述算法的关系。第一，UCB1 的遗憾是对数级的，但并非最优。由 Pinsker 不等式，$\mathrm{kl}(p, q) \ge 2(p - q)^2$，因此 $c^*$ 的每一项至多为 $1/(2\Delta_i)$，而 @thm-regret-ucb1 中对应的项为 $8/\Delta_i$，至少大 16 倍。采用 Beta 后验的 Thompson 采样在极限意义下恰好达到 $c^*$[@kaufmann2012thompson; @agrawal2013further]。

第二，定理讨论的是极限，其中的 $\liminf$ 至关重要。定理断言遗憾与 $\ln T$ 之比不可能始终低于 $c^*$，并未断言对每个有限的 $T$ 都有 $\E[R_T] \ge c^* \ln T$；低阶项为负的算法可以在很长时间内位于这条曲线下方。@fig-regret-bandits 中的 Thompson 采样正是如此。依据这一定理，可以比较的是 $t$ 增大时对数时间轴上的斜率。

第三，这个界通过差距依赖于具体实例，差距缩小时界趋于无穷。但这并不意味着各臂几乎相等时遗憾会变大，因为拉动一条几乎同样好的臂代价很小。对所有实例取**最坏情况**（worst case）则是另一个量：对任意算法和任意 $T \ge K - 1$，都存在一个奖励服从高斯分布的 $K$ 臂赌博机，使该算法在其上的遗憾至少为 $\frac{1}{27}\sqrt{(K - 1)T}$[@lattimore2020bandit, 定理 15.2]。造成损失的差距随 $T$ 按 $\sqrt{K/T}$ 缩小：既大到足以产生影响，又小到难以察觉。实例相关的界按 $\ln T$ 增长，系数是依赖于问题的常数；最坏情况的界按 $\sqrt{T}$ 增长。两种视角在函数情形中都会再次出现。

## 从臂到函数 {#sec-gp-bandits}

在贝叶斯优化中，每个输入都是一条臂，连续定义域上有无穷多条。上述赌博机界都随臂数增长，或通过对差距的求和，或通过最坏情况中的 $\sqrt{K}$；$K$ 为无穷时，这些界不能说明任何问题。分析得以延续的关键在于，这些臂不再相互独立。按照高斯过程先验，相近的输入取值相近，因此评估一个输入也能获得其邻近输入的信息。于是需要用另一个量代替臂数，以度量实质上不同的臂有多少条，最大信息增益就是这样的量。

### 设定 {#sec-regret-gp-setting}

本节沿用 @srinivas2010gaussian 的分析。该分析假定 @sec-gp-noise 的模型完全成立：函数是从高斯过程中抽取的样本，$f \sim \GP(0, k)$，且 $k(\vx, \vx) \le 1$，从而先验标准差处处不超过 1。每次评估返回 $y_t = f(\vx_t) + \varepsilon_t$，噪声 $\varepsilon_t \sim \N(0, \sigma_n^2)$ 相互独立，方差已知。暂设定义域 $\X$ 为有限集，例如一个精细的网格（原论文将该集合记作 $D$），@sec-regret-other-settings 将放宽这一假设。奖励服从高斯分布的 $K$ 臂赌博机，就是核函数在对角线上为 1、其余位置为 0 的特例。

经过 $t - 1$ 次评估，后验均值为 $\mu_{t-1}(\vx)$，标准差为 $\sigma_{t-1}(\vx)$，按 @eq-gp-noisy 计算。GP-UCB 规则沿用 UCB1 的乐观原则，只是以后验代替 Hoeffding 区间：

$$
\vx_t = \argmax_{\vx \in \X} \; \mu_{t-1}(\vx) + \beta_t^{1/2}\, \sigma_{t-1}(\vx).
$$ {#eq-regret-gpucb}

其中后验均值相当于经验平均值，后验标准差相当于加成项，$\beta_t$ 决定允许乐观到几个标准差。这正是 @eq-loop-ucb 的规则，只是权重可以逐轮变化。与该式及 @srinivas2010gaussian 一致，$\beta_t$ 乘在方差上，因此其平方根乘在标准差上；有些教材和软件库则把标准差本身的乘子称为 $\beta$。

### 最大信息增益 {#sec-regret-info-gain}

$T$ 次带噪声的评估能提供多少关于 $f$ 的信息？@sec-gp-information-gain 已回答了这个问题，这里的界需要用到其中的三个事实。第一，在输入集合 $A$ 上的观测 $\vy_A$ 与函数之间的**互信息**（mutual information），即观测的不确定性中反映 $f$ 而非噪声的部分，为

$$
I(\vy_A; f) = \tfrac12 \log\det\!\left(\mI + \sigma_n^{-2} \mK_A\right),
$$ {#eq-regret-info}

其中 $\mK_A$ 是 $A$ 的核矩阵（@eq-info-gp-gain）。互信息只取决于在哪里评估，与观测到的值无关，因为高斯过程的后验方差不依赖于观测值（@sec-gp-conditioning）。第二，任意 $T$ 次评估所能提供的信息，上限就是这个量所能取到的最大值。

::: {.definition #def-regret-gamma title="最大信息增益"}
$T$ 次评估后的**最大信息增益**（maximum information gain）为
$$
\gamma_T = \max_{A \subset \X,\; |A| = T} \; \tfrac12 \log\det\!\left(\mI + \sigma_n^{-2} \mK_A\right).
$$ {#eq-regret-gamma}
:::

$\gamma_T$ 由核函数、定义域和噪声水平决定，在获得任何数据之前就已确定。

两个极端情形给出了它的范围。若全部 $T$ 次评估都位于同一输入，获得的信息为 $\tfrac12 \log(1 + T/\sigma_n^2)$，只按对数增长：如 @exr-noise-floor 所示，重复评估提供的信息越来越少。若核函数是对角的，如 $K$ 臂赌博机，则把评估均匀分配到各臂上最好，$\gamma_T$ 按 $\tfrac{K}{2}\log(1 + T/(K\sigma_n^2))$ 增长（@exr-regret-independent）。光滑的核函数介于两者之间：相近输入处的观测大多是冗余的，因此信息的增长比 $T$ 条独立的臂慢得多。较短的长度尺度、粗糙的核函数、较高的维度、较低的噪声，都会使定义域中更多的部分变得可以区分，从而增大 $\gamma_T$。

第三，这个最大值虽然无法精确计算，却很容易近似。在 $\vx$ 处新做一次评估，信息恰好增加 $\tfrac12 \log(1 + \sigma_n^{-2}\sigma_{t-1}^2(\vx))$（@eq-info-gain-chain）。因此，贪心规则每次都在后验方差最大处评估（即不确定性采样，@sec-info-sequential），所得信息至少为 $\gamma_T$ 的 $1 - 1/e \approx 0.63$[@srinivas2010gaussian]。@sec-bounds-and-practice 中的图正是用这种方法估计 $\gamma_T$ 的。

### GP-UCB 的界 {#sec-regret-gpucb-bound}

借助 $\gamma_T$，这个界的形式与赌博机界相仿，只是替换了其中的 $K$。

::: {.theorem #thm-regret-gpucb title="有限定义域上的 GP-UCB（Srinivas、Krause、Kakade 与 Seeger，2010）"}
设 $\X$ 有限，$\delta \in (0, 1)$，并令
$$
\beta_t = 2 \log\!\left(\frac{|\X|\, t^2 \pi^2}{6\delta}\right).
$$
若 $f$ 是从 $\GP(0, k)$ 中抽取的样本，其中 $k(\vx, \vx) \le 1$，噪声为 $\N(0, \sigma_n^2)$，则采用上述 $\beta_t$ 的 GP-UCB 以至少 $1 - \delta$ 的概率满足
$$
R_T \le \sqrt{C_1\, T\, \beta_T\, \gamma_T}
\quad \text{for all } T \ge 1,
\qquad C_1 = \frac{8}{\log(1 + \sigma_n^{-2})}.
$$ {#eq-regret-gpucb-bound}
:::

证明沿用 UCB1 证明梗概中的三个步骤，每一步都是初等的。

::: {.derivation title="平方根从何而来"}
以下各步对应 @srinivas2010gaussian 扩展版中的引理 5.1 至 5.4。

1. **置信**。给定数据，$f(\vx)$ 服从均值为 $\mu_{t-1}(\vx)$、标准差为 $\sigma_{t-1}(\vx)$ 的高斯分布；高斯变量偏离均值超过 $\beta^{1/2}$ 个标准差的概率至多为 $e^{-\beta/2}$（@sec-regret-concentration）。对 $|\X|$ 个输入和所有轮次使用联合界，并取定理中的 $\beta_t$，即可保证以至少 $1 - \delta$ 的概率，$|f(\vx) - \mu_{t-1}(\vx)| \le \beta_t^{1/2}\sigma_{t-1}(\vx)$ 对所有 $\vx$ 和 $t$ 同时成立。式中的 $\pi^2/6$ 即 $\sum_t 1/t^2$，用于把 $\delta$ 分摊到各轮。
2. **乐观的代价至多为宽度的两倍**。在上述事件成立时，由于 $\vx_t$ 使上界最大，有 $\mu_{t-1}(\vx_t) + \beta_t^{1/2}\sigma_{t-1}(\vx_t) \ge \mu_{t-1}(\vx^\star) + \beta_t^{1/2}\sigma_{t-1}(\vx^\star) \ge f(\vx^\star)$。减去 $f(\vx_t) \ge \mu_{t-1}(\vx_t) - \beta_t^{1/2}\sigma_{t-1}(\vx_t)$，得 $r_t \le 2\beta_t^{1/2}\sigma_{t-1}(\vx_t)$。
3. **本次运行获得的信息**。由 @eq-info-gain-chain，GP-UCB 实际所选输入获得的信息可以写成各轮之和：$I(\vy_T; f) = \tfrac12 \sum_{t} \log\!\left(1 + \sigma_n^{-2}\sigma_{t-1}^2(\vx_t)\right)$。这个量至多为 $\gamma_T$，即任意 $T$ 个输入所能达到的最大值。
4. **把方差换成信息**。记 $s^2 = \sigma_n^{-2}\sigma_{t-1}^2(\vx_t)$；由于 $\sigma_{t-1}^2(\vx_t) \le k(\vx_t, \vx_t) \le 1$，该量落在 $[0, \sigma_n^{-2}]$ 中。在这一区间上，凹函数 $\log(1 + s^2)$ 位于其弦的上方，因此 $s^2 \le C_2 \log(1 + s^2)$，其中 $C_2 = \sigma_n^{-2}/\log(1 + \sigma_n^{-2})$。将第 2 步两边平方并利用 $\beta_t \le \beta_T$，得 $r_t^2 \le 4\beta_T \sigma_n^2 s^2 \le C_1 \beta_T \cdot \tfrac12\log(1 + s^2)$，其中 $C_1 = 8\sigma_n^2 C_2 = 8/\log(1 + \sigma_n^{-2})$。对各轮求和并利用第 3 步，得 $\sum_t r_t^2 \le C_1 \beta_T \gamma_T$。
5. **Cauchy-Schwarz 不等式**。$R_T^2 = \left(\sum_t r_t\right)^2 \le T \sum_t r_t^2 \le C_1 T \beta_T \gamma_T$。开平方即得 @eq-regret-gpucb-bound。
:::

这一结果可以这样理解：$\beta_T$ 按 $\log T$（以及 $\log |\X|$）增长，而对实践中使用的核函数，$\gamma_T$ 次线性增长，因此 $R_T$ 按 $\sqrt{T}$ 乘以若干缓慢增长的因子增长。可见 GP-UCB 是无遗憾的；再由 @eq-regret-simple-from-cumulative，以至少 $1 - \delta$ 的概率，GP-UCB 评估过的最好输入与最大值之差不超过 $\sqrt{C_1 \beta_T \gamma_T / T}$。

有限定义域并不只是为了数学上的方便。在 @srinivas2010gaussian 的实验中，输入是 Intel Research Berkeley 某传感器网络中的 46 个温度传感器；第二项测试的输入是加利福尼亚州 I-880 高速公路某路段上的 357 个交通传感器，目标是找出最拥堵的位置。核矩阵并非由公式给出，而是传感器读数在记录数据前三分之二上的经验协方差；待优化的函数则取自其余三分之一数据中的快照。在温度数据上，GP-UCB 与期望改进都明显优于其他启发式方法，两者之间无显著差异；作者总结认为，GP-UCB 的表现至少不逊于那些没有遗憾界的现有方法。这个界还把两个研究传统联系起来。第 4 步表明，GP-UCB 的某一步只有在能提供信息时才可能代价高昂，因此优化器的遗憾受制于关于 $f$ 还有多少内容可学，而这正是实验设计中通用的度量（@sec-expected-information-gain）。

这个界的好坏取决于 $\gamma_T$ 增长的快慢。@tbl-regret-gamma 汇总了 $d$ 维定义域上的已知速率，这些速率由核函数特征值衰减的快慢决定（@sec-ka-infogain）；其中 $\nu$ 是 Matérn 核的光滑度参数（@sec-kernel-family）。

::: {.table #tbl-regret-gamma title="d 维有界闭（紧）定义域上，最大信息增益随评估次数 T 的增长"}
| 核函数 | $\gamma_T$ | 来源 |
|---|---|---|
| 线性核 | $O(d \log T)$ | @srinivas2010gaussian |
| 径向基函数（平方指数）核 | $O\big((\log T)^{d+1}\big)$ | @srinivas2010gaussian |
| Matérn 核，$\nu > 1$ | $O\big(T^{d(d+1)/(2\nu + d(d+1))} \log T\big)$ | @srinivas2010gaussian |
| Matérn 核，$\nu > 1/2$ | $O\big(T^{d/(2\nu + d)} (\log T)^{2\nu/(2\nu + d)}\big)$ | @vakili2021information |
:::

对径向基函数核，维度只出现在 $\log T$ 的指数上，因此界按 $\sqrt{T}(\log T)^{(d+2)/2}$ 增长（其中因子 $(\log T)^{(d+1)/2}$ 来自 $\gamma_T$，另一个因子 $(\log T)^{1/2}$ 来自 $\beta_T$）：即使在多维情形下，非常光滑的函数也能很快学到[@srinivas2010gaussian]。对 Matérn 核，最初的速率并不紧；@vakili2021information 于 2021 年给出的速率在相差对数因子的意义下与已知下界一致。

### 其他设定 {#sec-regret-other-settings}

有限定义域上的定理可以向三个方向推广，每个方向各有其假设。本小节概览相关结果，供日后在论文中遇到它们的读者参考。初读时可以跳过；后文唯一用到的概念是再生核 Hilbert 空间，@sec-kernelized-dueling 会再作解释，@sec-kernel-analysis 则有深入的讨论。

**连续定义域**。对 $d$ 维的紧凸定义域（例如箱形区域），@srinivas2010gaussian 证明了同样形式的界：$\beta_t$ 增加一个 $d \log t$ 量级的项，界本身增加一个加性常数。证明中随着 $t$ 增大把定义域离散得越来越细，这要求样本路径足够光滑，使相邻网格点上的函数值彼此接近。径向基函数核和 $\nu > 2$ 的 Matérn 核满足这一条件。粗糙的 Matérn 1/2 核不满足这一假设，作者猜想对它不存在这种形式的结果。

**固定函数**。上述定理是贝叶斯式的：对从先验中抽取的函数以高概率成立。频率派版本则要求对函数类中任意一个固定的函数给出保证。自然的函数类是该核函数的**再生核 Hilbert 空间**（reproducing kernel Hilbert space，RKHS）。这一函数空间由形如 @eq-gp-representer 的核函数鼓包之和构成，其范数 $\lVert f \rVert_k$ 衡量 $f$ 相对于该核函数的粗糙程度（@sec-ka-rkhs 构造了这一空间及其范数）。（高斯过程本身的样本路径比这更粗糙，范数为无穷大，因此两种设定互不包含。）若 $\lVert f \rVert_k^2 \le B$ 且噪声有界，则采用 $\beta_t = 2B + 300\gamma_t \log^3(t/\delta)$ 的 GP-UCB，其遗憾在相差对数因子的意义下为 $\sqrt{T}(\sqrt{B\gamma_T} + \gamma_T)$ 量级[@srinivas2010gaussian]。@chowdhury2017kernelized 改进了这一分析，并证明了 Thompson 采样的一种高斯过程版本的遗憾界。这类宽度从何而来，见 @sec-regret-self-normalized。

**下界**。在 RKHS 设定下，@scarlett2017lower 证明，对任何算法，Matérn 函数类中都存在某个函数，使该算法的累积遗憾至少为 $T^{(\nu + d)/(2\nu + d)}$ 量级。对径向基函数核，他们证明累积遗憾至少为 $\sqrt{T(\log T)^{d/2}}$ 量级；这与上界相符，差别仅在于上界中根号下 $\log T$ 的指数为 $2d + O(1)$ 而非 $d/2$。这些下界之于函数，正如 Lai 与 Robbins 的结果之于赌博机：它们表明该函数类迫使任何算法做多少探索。

把评估换成比较，同样的工具可以得出 @sec-kernelized-dueling 中核化对决赌博机的界，以及 @sec-pbo-theory 的理论；后者的一些基本下界目前仍然缺失（@sec-theory-lower）。

### 固定函数的置信界 {#sec-regret-self-normalized}

@thm-regret-gpucb 的置信步骤用到了 @sec-regret-concentration 中的两个工具。高斯变量偏离均值超过 $\beta^{1/2}$ 个标准差的概率至多为 $e^{-\beta/2}$；对 $|\X|$ 个输入和各轮取联合界，并给第 $t$ 轮分配 $\delta$ 中的 $6\delta/(\pi^2 t^2)$，就要求 $|\X|\, e^{-\beta_t/2} = 6\delta/(\pi^2 t^2)$。解出 $\beta_t$，即得定理中的 $\beta_t = 2 \log\!\big(|\X|\, t^2 \pi^2/(6\delta)\big)$。在那里，自适应地选择输入并无妨碍：给定已有的观测，据此选出的输入就是固定的，无论用什么规则选择，$f(\vx)$ 都服从均值为 $\mu_{t-1}(\vx)$、标准差为 $\sigma_{t-1}(\vx)$ 的高斯分布[@srinivas2010gaussian, 引理 5.1]。

@sec-regret-other-settings 中的频率派结果失去了这一支撑。在那里，$f$ 是满足 $\lVert f \rVert_k^2 \le B$ 的一个固定函数，唯一的随机性来自噪声。误差 $\mu_t(\vx) - f(\vx)$ 由两部分组成：先验把估计拉向零所造成的偏差，以及噪声项 $\varepsilon_1, \dots, \varepsilon_t$ 的加权和；而权重取决于算法选择在哪里观测，这又取决于先前的噪声。这不是权重事先固定的独立项之和，因此 Hoeffding 不等式不适用；在连续定义域上，也没有有限的输入列表可供取联合界。本小节介绍替代这两者的工具，并说明它如何给出随 $\gamma_t$ 与 $\log(1/\delta)$ 增长的宽度。与上面的概览一样，本小节初读时可以跳过。

**鞅**（martingale）。考虑累加和 $M_t = \sum_{s \le t} g_s \varepsilon_s$：每个权重 $g_s$ 可以依赖于第 $s$ 轮之前观测到的一切，但在抽取 $\varepsilon_s$ 之前就已确定；给定之前的一切，每个 $\varepsilon_s$ 的均值为零。这样的和称为鞅，好比赌徒在公平赌局中的资产，他根据历史决定每次下注的数额。下注是自适应的，赌局依然公平。Chernoff 方法在自适应的情形下依然有效。给定过去，$g_t$ 是一个固定的数，$\varepsilon_t$ 是 $R$-次高斯的（@def-regret-subgaussian），因此 $\E\big[e^{\lambda g_t \varepsilon_t} \given \text{past}\big] \le e^{\lambda^2 g_t^2 R^2/2}$。从最后一轮开始逐轮剥离，可以证明

$$
Z_t = \exp\!\Big(\lambda M_t - \tfrac12 \lambda^2 R^2 V_t\Big),
\qquad
V_t = \sum_{s \le t} g_s^2,
$$

对每个 $t$ 的期望都至多为 1。@eq-regret-hoeffding 的推导把期望按独立项分解，这里则按轮次分解，每一轮都以之前各轮为条件。结论还可以更强：$Z_t$ 始终非负，平均而言也不会逐轮向上漂移；对这样的过程，Markov 不等式有更强的形式，即**极大不等式**（maximal inequality）：$Z_t$ 在某一轮达到 $1/\delta$ 的概率至多为 $\delta$[@lattimore2020bandit, 定理 3.9]。于是，对所有轮次同时成立的界不再需要对轮次取联合界。

还剩两个问题。一是最佳的 $\lambda$ 依赖于随机的 $V_t$。二是高斯过程的估计不具有 $M_t$ 的形式：它赋予观测 $y_s$ 的权重依赖于第 $s$ 轮之后选择的输入。第一个问题的解决办法是不再选定 $\lambda$，而是对它求 $Z_t$ 的平均，这称为**混合方法**（method of mixtures）。

::: {.derivation title="一维情形下的混合方法"}
1. 对每个固定的 $\lambda$，$Z_t$ 始终非负，从 1 出发，且不向上漂移。这类过程对 $\lambda$ 的平均仍是这样的过程[@lattimore2020bandit, 引理 20.3]。
2. 对常数 $c > 0$，按 $\lambda \sim \N\big(0, 1/(cR^2)\big)$ 求平均。$\lambda$ 的密度为 $\sqrt{cR^2/(2\pi)}\,e^{-cR^2\lambda^2/2}$，因此 $\bar Z_t = \sqrt{cR^2/(2\pi)}\int \exp\!\big(\lambda M_t - \tfrac12\lambda^2R^2(V_t + c)\big)\,\dd\lambda$。与 @sec-gaussian-1d 一样对 $\lambda$ 配方，剩下 $\exp\!\big(M_t^2 / (2R^2(V_t + c))\big)$ 乘以一个高斯积分，该积分等于 $\sqrt{2\pi/(R^2(V_t + c))}$，因此 $\bar Z_t = \sqrt{c/(V_t + c)}\, \exp\!\big(M_t^2 / (2R^2(V_t + c))\big)$。
3. 由极大不等式，以至少 $1 - \delta$ 的概率，对每个 $t$ 都有 $\bar Z_t < 1/\delta$。取对数并整理，得 $M_t^2 < R^2 (V_t + c) \big(2 \log(1/\delta) + \log(1 + V_t/c)\big)$ 对每个 $t$ 成立。
:::

这一结果可以用标准差来解读。$R^2 V_t$ 相当于 $M_t$ 的方差，因此这个和保持在约一个标准差 $R\sqrt{V_t + c}$ 乘以 $\sqrt{2\log(1/\delta) + \log(1 + V_t/c)}$ 的范围之内。$2\log(1/\delta)$ 是每个 Chernoff 界都要支付的置信代价。$\log(1 + V_t/c)$ 是事先不知道方差会有多大的代价，它只按方差的对数增长。这类界称为**自归一化**（self-normalized）界：和以其自身累积的方差为尺度来度量。

第二个问题的解决办法是改用向量。用特征表示核函数，$k(\vx, \vx') = \boldsymbol{\phi}(\vx)^\T\boldsymbol{\phi}(\vx')$，即 @sec-bayes-blr-posterior 的权重空间观点，并把权重的先验协方差取为 $\mI$。前 $t$ 轮可以用两个量概括：权重的后验精度，以及沿接收噪声的输入的特征方向累加的噪声：

$$
\mA_t = \mI + \sigma_n^{-2} \sum_{s \le t} \boldsymbol{\phi}(\vx_s)\boldsymbol{\phi}(\vx_s)^\T,
\qquad
\mathbf{s}_t = \sum_{s \le t} \varepsilon_s\, \boldsymbol{\phi}(\vx_s).
$$

$\mathbf{s}_t$ 每一项的权重向量 $\boldsymbol{\phi}(\vx_s)$ 都在抽取对应的噪声之前确定，因此 $\mathbf{s}_t$ 是取向量值的鞅。用方向上的高斯分布代替上面推导中 $\lambda$ 的高斯分布求平均，就得到下面的界。

::: {.theorem #thm-regret-self-normalized title="自归一化界（Abbasi-Yadkori、Pál 与 Szepesvári，2011）"}
设特征只有有限个分量。假设每个输入 $\vx_s$ 都根据第 $s$ 轮之前的观测选择，且在给定之前一切的条件下，每个噪声项 $\varepsilon_s$ 都是 $R$-次高斯的。则对任意 $\delta \in (0, 1)$，以至少 $1 - \delta$ 的概率，
$$
\mathbf{s}_t^\T \mA_t^{-1} \mathbf{s}_t \le \sigma_n^2 R^2 \big(\log\det\mA_t + 2\log(1/\delta)\big)
\quad \text{for all } t \ge 0 \text{ at once.}
$$ {#eq-regret-self-normalized}
:::

这是 @abbasiyadkori2011improved 的定理 1，其中正则化参数取为 $\sigma_n^2$，因此原文中的矩阵为 $\sigma_n^2 \mA_t$。Lattimore 与 Szepesvári 用混合方法证明了 $R = 1$ 的情形[@lattimore2020bandit, 定理 20.4]，任意 $R$ 都可以通过缩放噪声化为这一情形。只有一个特征时，该定理就是上面推导中取 $c = \sigma_n^2$ 的情形。

对数行列式就是信息增益。由矩阵行列式引理（@eq-id-det-lemma），$\det \mA_t = \det(\mI + \sigma_n^{-2}\mK_t)$，其中 $\mK_t$ 是前 $t$ 个输入的核矩阵，因此 $\tfrac12 \log\det\mA_t$ 就是算法所选输入获得的信息 $I(\vy_t; f)$（@eq-regret-info），至多为 $\gamma_t$。联合界按输入逐个收取一个对数，这个界则按数据已测量的方向收费。

::: {.aside title="对数行列式计数的是什么"}
假设输入事先固定，噪声服从标准差为 $\sigma_n$ 的高斯分布，从而 $R = \sigma_n$。记 $\mA_t$ 的特征值为 $1 + \eta_1, 1 + \eta_2, \dots$。由于 $\E[\mathbf{s}_t\mathbf{s}_t^\T] = \sigma_n^4(\mA_t - \mI)$，@eq-regret-self-normalized 左边除以 $\sigma_n^4$ 后的均值为 $\sum_j \eta_j/(1 + \eta_j)$。每一项都在 0 与 1 之间：数据测量充分的方向接近 1，几乎未触及的方向接近 0，因此这个均值计数的是已测量的方向。右边除以 $\sigma_n^4$ 后为 $\sum_j \log(1 + \eta_j) + 2\log(1/\delta)$，而 $\log(1 + \eta) \ge \eta/(1 + \eta)$。这个界对每个已测量方向收取的代价略高于其平均份额，正是这点余量，换来了对自适应输入和所有 $t$ 同时成立的结论。
:::

在有限定义域上（如 @thm-regret-gpucb），具有有限个分量的特征总是存在的，该定理由此转化为固定函数的置信界。

::: {.derivation title="固定函数的置信界"}
取 $\boldsymbol{\phi}(\vx)$ 为 $\mK_\X^{1/2}$ 中对应于 $\vx$ 的列，其中 $\mK_\X$ 是整个定义域的核矩阵。则 $k(\vx, \vx') = \boldsymbol{\phi}(\vx)^\T\boldsymbol{\phi}(\vx')$，且 RKHS 中的每个函数都可写成 $f(\vx) = \boldsymbol{\phi}(\vx)^\T\vw$，其中 $\lVert \vw \rVert = \lVert f \rVert_k \le \sqrt{B}$。记 $\boldsymbol{\Phi}_t$ 为以 $\boldsymbol{\phi}(\vx_1)^\T, \dots, \boldsymbol{\phi}(\vx_t)^\T$ 为行的矩阵，则 @eq-bayes-blr-posterior 与 @eq-bayes-blr-predictive 给出的后验为 $\mu_t(\vx) = \boldsymbol{\phi}(\vx)^\T\bar\vw_t$，其中 $\bar\vw_t = \sigma_n^{-2}\mA_t^{-1}\boldsymbol{\Phi}_t^\T\vy_t$，以及 $\sigma_t^2(\vx) = \boldsymbol{\phi}(\vx)^\T\mA_t^{-1}\boldsymbol{\phi}(\vx)$。

1. **拆分误差**。代入 $\vy_t = \boldsymbol{\Phi}_t\vw + (\varepsilon_1, \dots, \varepsilon_t)^\T$ 与 $\boldsymbol{\Phi}_t^\T\boldsymbol{\Phi}_t = \sigma_n^2(\mA_t - \mI)$，得 $\bar\vw_t = \vw - \mA_t^{-1}\vw + \sigma_n^{-2}\mA_t^{-1}\mathbf{s}_t$，因此 $\mu_t(\vx) - f(\vx) = -\boldsymbol{\phi}(\vx)^\T\mA_t^{-1}\vw + \sigma_n^{-2}\boldsymbol{\phi}(\vx)^\T\mA_t^{-1}\mathbf{s}_t$。
2. **把输入与其余部分分开**。在内积 $\mathbf{u}^\T\mA_t^{-1}\mathbf{v}$ 下应用 Cauchy-Schwarz 不等式，对任意向量 $\mathbf{v}$，有 $|\boldsymbol{\phi}(\vx)^\T\mA_t^{-1}\mathbf{v}| \le \sigma_t(\vx)\sqrt{\mathbf{v}^\T\mA_t^{-1}\mathbf{v}}$。
3. **偏差**。$\mA_t$ 等于 $\mI$ 加上若干半正定项，因此 $\vw^\T\mA_t^{-1}\vw \le \lVert\vw\rVert^2 \le B$，第 1 步中的第一项至多为 $\sqrt{B}\,\sigma_t(\vx)$。
4. **噪声**。由 @eq-regret-self-normalized，第二项至多为 $\sigma_t(\vx)\,(R/\sigma_n)\sqrt{\log\det\mA_t + 2\log(1/\delta)}$。
5. **信息**。$\log\det\mA_t = 2I(\vy_t; f) \le 2\gamma_t$。

综合以上各步，以至少 $1 - \delta$ 的概率，对每个输入和每个 $t \ge 0$ 同时有
$$
|f(\vx) - \mu_t(\vx)| \le \Big(\sqrt{B} + \frac{R}{\sigma_n}\sqrt{2\big(\gamma_t + \log(1/\delta)\big)}\Big)\, \sigma_t(\vx).
$$ {#eq-regret-selfnorm-width}
:::

GP-UCB 在第 $t$ 轮使用 $t - 1$ 次观测后的后验，因此由 @eq-regret-selfnorm-width 可以得到一个有效的置信界，其中

$$
\beta_t^{1/2} = \sqrt{B} + \frac{R}{\sigma_n}\sqrt{2\big(\gamma_{t-1} + \log(1/\delta)\big)}.
$$ {#eq-regret-beta-frequentist}

第一项是偏差：范数大的函数可以远离先验的预期，但至多偏离 $\sqrt{B}$ 个后验标准差。第二项来自噪声。它随 $\gamma_{t-1}$ 增长，因为数据测量过的每个方向，都是噪声可能推动估计的方向；它也像每个 Chernoff 界一样随 $\log(1/\delta)$ 增长。输入的个数 $|\X|$ 没有出现：第 2 步一次覆盖了所有输入，而贝叶斯宽度需要对它们取联合界。正因如此，这类界可以推广到连续定义域。

这一宽度与 @sec-regret-other-settings 中的频率派结论一致。在那里，噪声的绝对值以 $\sigma_n$ 为界，模型的噪声方差为 $\sigma_n^2$[@srinivas2010gaussian, 定理 3]，因此 $R = \sigma_n$；把 @eq-regret-beta-frequentist 平方并利用 $(u + v)^2 \le 2u^2 + 2v^2$，得 $\beta_t \le 2B + 4\big(\gamma_{t-1} + \log(1/\delta)\big)$。@srinivas2010gaussian 的取法 $\beta_t = 2B + 300\gamma_t \log^3(t/\delta)$ 含有同样来自 $f$ 的范数的 $2B$ 项，噪声部分也是同样的信息增益，只是乘以 $300\log^3(t/\delta)$。他们的证明使用了 Freedman 不等式（Bernstein 不等式利用条件方差的鞅版本），并对各轮取联合界[@srinivas2010gaussian, 附录 B]；因子 300 与对数的立方都来自这条路线（推断）。

对一般的定义域，特征可能有无穷多个分量，@abbasiyadkori2011improved 的论证便不再成立。@chowdhury2017kernelized 证明了在这种情形下依然成立的自归一化界。在 $\lVert f \rVert_k^2 \le B$、且给定过去时噪声为 $R$-次高斯的条件下，他们的定理 2 表明，以至少 $1 - \delta$ 的概率，
$$
|\mu_{t-1}(\vx) - f(\vx)| \le \Big(\sqrt{B} + R\sqrt{2\big(\gamma_{t-1} + 1 + \log(1/\delta)\big)}\Big)\, \sigma_{t-1}(\vx)
$$
对时域 $T$ 内的每个输入和每一轮都成立，其中后验与 $\gamma_{t-1}$ 都以噪声方差 $1 + 2/T$ 代替 $\sigma_n^2$ 计算。（他们用 $B$ 表示范数本身，他们的 $\beta_t$ 是 $\sigma_{t-1}(\vx)$ 的乘子，即本书的 $\beta_t^{1/2}$。）多出的 1 用于抵偿略微放大的噪声方差，后者使对数行列式增加 $t\log(1 + 2/T)$，至多为 2。他们的算法 IGP-UCB 使用这一宽度，比 GP-UCB 的宽度窄，两者之比按 $\log^{3/2}(t/\delta)$ 增长。

@sec-regret-gpucb-bound 中 GP-UCB 证明的第 2 至 5 步，除了置信论断和 $k(\vx, \vx) \le 1$ 之外，没有用到关于 $f$ 的任何性质。在有限定义域上采用 @eq-regret-beta-frequentist 的宽度，这几步再次给出 $R_T \le \sqrt{C_1 T \beta_T \gamma_T}$，只是现在对每个满足 $\lVert f \rVert_k^2 \le B$ 的固定 $f$ 以至少 $1 - \delta$ 的概率成立。由于 $\beta_T \le 2B + 4(R/\sigma_n)^2\big(\gamma_T + \log(1/\delta)\big)$，对固定的 $\delta$，遗憾为 $\sqrt{T}\big(\sqrt{B\gamma_T} + \gamma_T\big)$ 量级，即 @sec-regret-other-settings 中所述的量级，而且不再隐含对数因子。用本书的记号，@chowdhury2017kernelized 把他们的界写为 $R_T = O\big(\sqrt{BT\gamma_T} + \sqrt{T\gamma_T(\gamma_T + \log(1/\delta))}\big)$。

::: {.keyidea title="信息增益是自适应的代价"}
输入根据先前的噪声选择时，固定函数的置信界必须在数据可能测量过的每个方向上都成立。自归一化界为此支付的代价是后验精度的对数行列式，即信息增益的两倍，因此宽度按 $\sqrt{\gamma_t + \log(1/\delta)}$ 增长，而不是随输入的个数增长。
:::

## 遗憾界对实践的意义 {#sec-bounds-and-practice}

@thm-regret-gpucb 给出的是一个具体数值，值得在所有假设都成立的问题上把它算出来：函数从 GP-UCB 所用的高斯过程中抽取，定义域为有限网格，算法知道真实的噪声水平。下图正是这样设置的，并在同一函数、同样的噪声下运行 GP-UCB 两次：一次采用定理中的 $\beta_t$，一次采用实践中常用的常数乘子。

```{figure}
//| figure: regret-gpucb
//| label: fig-regret-gpucb
//| fig-cap: "GP-UCB 在从其自身先验中抽取的函数（上）上运行，定义域由 160 个输入组成：一维时为网格；三维或六维时为立方体中一个固定的拉丁超立方样本，此时上方面板按各输入的第一个坐标绘制。函数下方的刻线标出各次运行的评估位置，刻线越高表示重复评估越多。下方面板以对数刻度显示累积遗憾，包括采用定理中 $\beta_t$（$\delta = 0.1$）的 GP-UCB、采用常数乘子 $\sqrt\beta$ 的 GP-UCB、均匀随机查询的期望遗憾，以及 @thm-regret-gpucb 的界。该界使用 $\gamma_T$ 的贪心估计，贪心估计只会低估 $\gamma_T$，因此真实的界至少与图中所画一样高。数值对应一个随机函数；按“换一个函数”可换成另一个。"
```

可以尝试以下操作。

**查看默认设置**。默认设置下，$T = 200$ 时这个界达到几百，比均匀随机查询产生的遗憾还大；而采用定理中 $\beta_t$ 的 GP-UCB 遗憾低于 20，采用 $\sqrt\beta = 2$ 的 GP-UCB 约为其一半。在这个问题上定理成立，但在这一时域内，它比不采用任何巧妙策略所得的平凡界还要弱。

**拖动“噪声标准差 σ_n”**。$\sigma_n$ 从 0.1 增至 1 时，$C_1$ 从 1.73 增大到 11.5，$\gamma_T$ 则下降，因为带噪声的评估提供的信息更少。界的变化幅度远小于这两个常数中的任何一个。两次运行受噪声的影响都比界更大。

**把“核函数”切换为“Matérn 1/2 核”**。粗糙函数的可区分区域多得多，$\gamma_T$ 的增长快数倍，两次运行都需要更长时间才能找到峰值。

**把“实用 √β”设为 0**。此时为纯利用：始终在后验均值最高处评估。从处处为零的均值出发，这条规则会反复评估第一个取值高于零的输入，因为其他输入看起来永远不会更好。除非该输入恰好位于峰值，否则遗憾将呈直线增长，默认函数正是这种情形。按“换一个函数”可以看到两种情形。另请注意，函数改变时界保持不变：界只取决于核函数、噪声和 $T$。

**切换到“3 维”，再切换到“6 维”**。定义域仍有 160 个输入，但现在散布于立方体中；长度尺度为 0.1 时，几乎没有哪两个输入近到足以相关。每个输入实际上都成了一条独立的臂：$\gamma_T$ 从约 42 跃升至三维时的约 350 和六维时的 380，接近 160 条独立臂的值，两次运行的损失也大得多。把“长度尺度 ℓ”提高到 0.5，信息增益又会下降，因为长度尺度越长，每次评估所能代表的立方体范围越大。维度增大时如何选择长度尺度，本身就是一个实际问题（@sec-high-dim-practice）。

**比较两个乘子**。读数显示 $\sqrt{\beta_T}$，$T = 200$ 时约为 6（@exr-regret-beta）。若后验设定正确，其误差远用不着六个标准差的乐观。正因如此，采用定理参数的运行在采用实用参数的运行稳定之后，仍会长时间继续探索。

### 遗憾界能说明什么 {#sec-regret-bounds-say}

本章的界确立了一点：在明确的假设下，黑箱优化确实能够实现次线性遗憾。这些界还指出了决定其速率的因素：对赌博机，是差距和臂数；对高斯过程，是最大信息增益，它由先验和噪声决定，与算法无关。@tbl-regret-gamma 中的速率对问题难度的排序是合理的：光滑的核函数比粗糙的容易，而在光滑度低的情形下，维度的危害最大。

这些证明还解释了某些设计选择为何重要。探索绝不能完全停止：贪心规则和常数 ε 规则的失败有其结构上的原因；置信参数 $\beta_t$ 缓慢增长，原因与 UCB1 的加成项含有 $\ln t$ 相同。乐观、后验采样与信息寻求是具有理论保证的机制，@sec-acquisition 中的多个采集函数正是基于它们构建的。

### 遗憾界不能说明什么 {#sec-regret-bounds-not-say}

**在实际时域内，常数很重要**。证明界时，只要能让证明成立，用什么常数都可以，这些界也很少是紧的。GP-UCB 的作者通过交叉验证（在拟合时留出的数据上逐一尝试各种缩放）发现，把 $\beta_t$ 缩小到定理取值的五分之一后，算法表现更好；他们也指出，自己并未优化界中的常数[@srinivas2010gaussian]。Auer 等人提出的变体 UCB1-TUNED 在他们几乎所有的实验中都明显优于 UCB1，但他们无法为其证明遗憾界[@auer2002finite]。@fig-regret-bandits 与 @fig-regret-gpucb 中的界都成立，但在实践者实际面对的时域内，这些界比朴素玩法的遗憾还大。

**假定模型正确**。@thm-regret-gpucb 假定核函数及其超参数、噪声水平均已知，且 $f$ 恰好抽取自这一先验；RKHS 版本则假定范数有已知的上界 $B$。实践中，超参数随数据的到来不断重新拟合（@sec-fitting-hyperparameters），算法也因此改变。@bull2011convergence 证明了固定先验下期望改进的收敛速率，并表明若用标准的序贯方法估计先验参数，该过程可能永远找不到最优点；改用其他估计量可以恢复这些速率。@berkenkamp2019noregret 给出了第一个无须知道超参数、可证明无遗憾的算法，其做法是缓慢扩大所考虑的函数类。

**所用的分数未必符合需要**。GP-UCB 的界针对累积遗憾。若只有最终推荐重要，@eq-regret-simple-from-cumulative 可以把它转化为保证，但为累积遗憾调校的算法在探索上可能过于谨慎[@bubeck2009pure]。@bull2011convergence 称期望改进也许是这一问题上最流行的方法，并在文中分析了无噪声评估下它的简单遗憾。在不同设定下、针对不同分数证明的界，无法用来为 @sec-acq-compare 中的采集函数排出高下（推断）。

**假定能精确求出采集函数的最大值点**。定理中的 $\vx_t$ 是 @eq-regret-gpucb 的精确最大值点。在连续定义域上，求这个最大值本身就是困难的多峰问题，只能用 @sec-acq-optimization 的方法近似求解[@srinivas2010gaussian]。本章的界都没有计入由此产生的误差。

**维度出现在指数上**。@fig-regret-gpucb 的三维和六维设定在小规模上展示了这一效应。径向基函数核的速率 $(\log T)^{d+1}$ 对 $T$ 而言温和，对 $d$ 而言却并不温和：$d = 10$、$T = 1000$ 时，$(\ln T)^{11}$ 约为 $1.7 \times 10^9$，因此除非 $O(\cdot)$ 中隐藏的常数极小，$\sqrt{T\gamma_T}$ 量级的界将比随 $T$ 线性增长的平凡界大几个数量级（推断）。在 RKHS 设定下，对 Matérn 核，把 GP-UCB 的 $\gamma_T\sqrt{T}$ 量级遗憾与 @vakili2021information 的速率结合，得到的指数为 $\tfrac12 + d/(2\nu + d)$；一旦 $d \ge 2\nu$，指数即达到 1，界也就不再提供任何信息（推断）。贝叶斯优化在实践中如何应对高维问题，是 @sec-high-dim-practice 与 @sec-high-dimensions 讨论的主题。

**界针对的是函数类，而非具体的函数**。最坏情况的界对函数类中的每个函数成立，贝叶斯界对从先验中抽取的大多数函数成立。实际面对的则是某个特定的函数，两种界都无法预测某种方法在这个函数上的排名。回答这个问题要靠基准测试，而基准测试也有其局限（@sec-sw-evaluation）。

可取的态度是把遗憾界视为设计原则和合理性检查，而不是预测。有保证的算法由不会永久陷入停滞的机制构成，其常数再根据经验调节，这些保证的提出者本人也是这样做的。下一章讨论这些经验性的决策。

## 习题 {#sec-regret-exercises}

::: {.exercise #exr-regret-eps-linear}
设有 $K$ 条臂，$\varepsilon > 0$ 为常数。证明 ε-贪心在 $T$ 轮后的期望遗憾至少为 $\frac{\varepsilon T}{K} \sum_i \Delta_i$（忽略开始时每条臂各拉一次的那一轮）。对 @fig-regret-bandits 的默认臂（均值为 $0.6$、$0.5$、$0.383$、$0.267$、$0.15$），取 $\varepsilon = 0.1$，计算这一斜率，并与图比较。

::: {.solution}
每一轮，算法以概率 $\varepsilon$ 拉动一条均匀选取的臂，其期望差距为 $\frac1K \sum_i \Delta_i$；以概率 $1 - \varepsilon$ 拉动贪心臂，其差距不小于 0。因此每轮的期望遗憾至少为 $\frac{\varepsilon}{K}\sum_i \Delta_i$，对 $T$ 轮求和即得该界。对默认的臂，各差距为 $0, 0.1, 0.217, 0.333, 0.45$，和为 $1.1$，因此斜率至少为每轮 $0.1 \times 1.1 / 5 = 0.022$，1,000 轮共计 22。图中 ε-贪心在 1,000 轮后的遗憾约为 60，多出的部分来自贪心臂并非最优臂的那些轮次：很少被探索的臂，其平均值仍然噪声较大。
:::
:::

::: {.exercise #exr-regret-pinsker}
两条 Bernoulli 臂的均值分别为 $0.6$ 和 $0.5$。计算 @eq-regret-lai-robbins 中的 Lai-Robbins 常数 $c^*$，以及 @thm-regret-ucb1 中 $\ln T$ 的系数。$T = 10^4$ 轮后，按这两个数，较差臂分别应被拉动多少次？

::: {.solution}
$\mathrm{kl}(0.5, 0.6) = 0.5\ln(0.5/0.6) + 0.5\ln(0.5/0.4) = 0.5(-0.1823 + 0.2231) = 0.0204$。由 $\Delta = 0.1$ 得 $c^* = 0.1/0.0204 = 4.90$，较差臂拉动次数的渐近值为 $\ln T/\mathrm{kl} = 9.21/0.0204 \approx 450$。UCB1 的系数为 $8/\Delta = 80$，约为 $c^*$ 的 16 倍；它给出的较差臂拉动次数上界为 $8\ln T/\Delta^2 + 1 + \pi^2/3 \approx 7{,}370 + 4$，而总预算只有 $10^4$ 次拉动，这个界几乎没有意义。Pinsker 不等式 $\mathrm{kl} \ge 2\Delta^2 = 0.02$ 在此几乎是紧的，因此 16 倍这一比值已接近最坏情况。
:::
:::

::: {.exercise #exr-regret-independent}
考虑 $K$ 条臂上的对角核函数（每个 $k(i, i) = 1$，其余元素均为 0），设 $T$ 是 $K$ 的倍数。证明 $\gamma_T = \frac{K}{2}\log\!\left(1 + \frac{T}{K\sigma_n^2}\right)$。据此，@thm-regret-gpucb 对 $R_T$ 随 $K$ 和 $T$ 的增长给出什么结论？与 @sec-regret-lower-bounds 中最坏情况的下界相比又如何？

::: {.solution}
臂相互独立时，$\mK_A$ 是分块对角矩阵：若臂 $i$ 被评估 $m_i$ 次，对应的块是元素全为 1 的 $m_i \times m_i$ 矩阵 $\mathbf{1}\mathbf{1}^\T$。$\mI + \sigma_n^{-2}\mK_A$ 中相应的块为 $\mI + \sigma_n^{-2}\mathbf{1}\mathbf{1}^\T$，它有一个特征值 $1 + m_i/\sigma_n^2$（特征向量为 $\mathbf{1}$），其余特征值都等于 1，因此行列式为 $1 + m_i/\sigma_n^2$（矩阵行列式引理的特例，@eq-id-det-lemma）。于是信息量为 $\frac12\sum_i \log(1 + m_i/\sigma_n^2)$，约束条件为 $\sum_i m_i = T$。对数函数是凹函数，各 $m_i$ 相等即 $m_i = T/K$ 时和最大，由此得到该公式。因此 $\sqrt{C_1 T \beta_T \gamma_T}$ 按 $\sqrt{KT}$ 乘以 $T$ 和 $K$ 的对数因子增长。最坏情况的下界为 $\frac{1}{27}\sqrt{(K-1)T}$，所以正如 @srinivas2010gaussian 所指出的，在这一特例中，GP-UCB 的界在相差对数因子的意义下是紧的。
:::
:::

::: {.exercise #exr-regret-beta}
按 @fig-regret-gpucb 的设定 $|\X| = 160$、$\delta = 0.1$、$T = 200$，计算定理中的 $\beta_T$ 与 $\sqrt{\beta_T}$。若把网格加密到 $|\X| = 16{,}000$ 个点，$\sqrt{\beta_T}$ 如何变化？这说明 $|\X|$ 起什么作用？

::: {.solution}
$|\X| T^2 \pi^2/(6\delta) = 160 \times 40{,}000 \times 9.8696/0.6 \approx 1.053 \times 10^8$，其自然对数为 $18.47$，因此 $\beta_T \approx 36.9$，$\sqrt{\beta_T} \approx 6.08$。网格加密 100 倍，$\beta_T$ 增加 $2\ln 100 \approx 9.2$，得到 $\beta_T \approx 46.1$，$\sqrt{\beta_T} \approx 6.79$。对 $|\X|$ 的依赖是对数的，但在有限网格上永远不会消失；这也是连续定义域版本需要单独论证的原因：无限加密网格会使 $\beta_t$ 变为无穷大。
:::
:::
::: {.exercise #exr-regret-sample-sizes}
奖励取值于 $[0, 1]$。要使一条臂的平均值超过其均值 0.1 或以上的概率至多为 0.05，按 Chebyshev 不等式（取最大可能的方差 $\tfrac14$）和按 Hoeffding 不等式，分别需要拉动多少次？再对概率 $10^{-4}$ 求解，以及对必须在 1000 轮中每一轮都成立、总失效概率为 0.05 的区间求解。把最后的答案与 @fig-regret-concentration 中“对 T 个平均值取联合界”设为 1000 时的读数比较。

::: {.solution}
由 Chebyshev 不等式，按 @eq-regret-chebyshev 需要 $n \ge \Var[Y]/(\delta a^2) = 0.25/(0.05 \times 0.01) = 500$。由 Hoeffding 不等式，按 @eq-regret-hoeffding 需要 $n \ge \ln(1/\delta)/(2a^2) = \ln 20/0.02 = 149.8$，即 150 次。对 $\delta = 10^{-4}$，Chebyshev 不等式需要 250,000 次拉动，Hoeffding 不等式需要 $\ln(10^4)/0.02 = 460.5$，即 461 次。$\delta$ 缩小为五百分之一，Chebyshev 不等式的答案扩大 500 倍，Hoeffding 不等式的答案只扩大约 3.1 倍。对 1000 轮，联合界给每一轮分配 $\delta = 5 \times 10^{-5}$：Chebyshev 不等式需要 500,000 次拉动，Hoeffding 不等式需要 $\ln(20{,}000)/0.02 = 495.2$，即 496 次，与图中读数一致。Hoeffding 不等式多需要的 $\ln 1000/0.02 \approx 345$ 次（两个答案都向上取整后为 346 次），来自 @eq-regret-union-width 中的 $\ln 1000$。
:::
:::

::: {.exercise #exr-regret-linear-width}
对 $d$ 维中的线性核 $k(\vx, \vx') = \vx^\T\vx'$，取特征 $\boldsymbol{\phi}(\vx) = \vx$，输入的长度至多为 1。证明对任意 $T$ 个输入有 $\log\det\mA_T \le d\log\!\big(1 + T/(d\sigma_n^2)\big)$，从而其信息增益至多为 $\tfrac{d}{2}\log\!\big(1 + T/(d\sigma_n^2)\big)$。由此，@eq-regret-beta-frequentist 对 $\beta_T$ 的增长给出什么结论？@thm-regret-gpucb 中的 $\log|\X|$ 在这里由什么代替？对 $d = 3$、$T = 1000$、$\sigma_n = 1$，计算信息增益的界。

::: {.solution}
$\mA_T = \mI + \sigma_n^{-2}\sum_s \vx_s\vx_s^\T$ 是 $d \times d$ 矩阵，其特征值 $\kappa_1, \dots, \kappa_d$ 均为正。行列式等于这些特征值之积，由几何平均与算术平均之间的不等式，至多为 $\big(\tfrac1d\sum_j \kappa_j\big)^d = (\tr\mA_T/d)^d$。迹为 $d + \sigma_n^{-2}\sum_s \lVert\vx_s\rVert^2 \le d + T/\sigma_n^2$，因此 $\log\det\mA_T \le d\log\!\big(1 + T/(d\sigma_n^2)\big)$，其一半即为信息增益的界，与 @eq-regret-selfnorm-width 推导的第 5 步相同。这就是 @tbl-regret-gamma 中的 $O(d\log T)$。无论定义域包含多少个输入，特征都只有 $d$ 个分量，因此自归一化界可以直接应用；又因为 $2\gamma_{T-1}$ 满足同样的界，@eq-regret-beta-frequentist 给出 $\beta_T^{1/2} \le \sqrt{B} + (R/\sigma_n)\sqrt{d\log\!\big(1 + T/(d\sigma_n^2)\big) + 2\log(1/\delta)}$，按 $\sqrt{d\log T}$ 增长：维度代替了 $\log|\X|$，定义域也可以是无限的。对 $d = 3$、$T = 1000$、$\sigma_n = 1$，这个界为 $1.5\log(334.3) = 8.72$ 奈特，而在 1000 个完全不相关的输入上评估所获得的信息为 $T \cdot \tfrac12\log 2 = 346.6$ 奈特。
:::
:::

## 延伸阅读 {#further-reading .unnumbered}

- @lattimore2020bandit 是赌博机理论的标准参考书：第 5 章讲集中不等式，第 7、8 章讲上置信界，第 20 章讲自归一化界与混合方法，第 15、16 章讲下界，第 33 章讲简单遗憾与纯探索，第 36 章讲 Thompson 采样。该书可在网上免费阅读。
- @auer2002finite 篇幅短小，易于阅读，内容包括上文概述的 UCB1 证明、衰减 ε-贪心规则，以及对两者的实验比较。
- @russo2018tutorial 是 Thompson 采样的实用教程，给出了 Bernoulli 臂之外的许多详细示例。
- @srinivas2010gaussian 提出了 GP-UCB、最大信息增益以及 @sec-gp-bandits 中的遗憾界；证明见其 arXiv 扩展版。
- @abbasiyadkori2011improved 针对线性赌博机证明了 @thm-regret-self-normalized 中的自归一化界；@chowdhury2017kernelized 将其推广到一般定义域上的核函数，并给出了 @sec-regret-self-normalized 中的置信宽度与遗憾界。
- @garnett2023bayesian 第 10 章综述贝叶斯优化的理论分析，包括期望改进和基于信息的策略的相关结果。
- @lai1985asymptotically 是下界的原始文献；对多数读者而言，先读 @lattimore2020bandit 中的现代表述更容易。
