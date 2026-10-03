---
synopsis: "一部关于贝叶斯优化与偏好贝叶斯优化的交互式教材，从其所依赖的概率论讲起，直至研究前沿，以及人的偏好究竟是什么。"
---

<div class="cover-top">
<div class="book-cover" role="img" aria-label="《贝叶斯优化：从基本原理到人类偏好》封面，作者欧长坤。高斯过程后验的一束曲线在六个观测处收拢；下方的弧线连接着一个人比较过的每一对选项。">
<img src="@@ROOT@@assets/cover-art.svg" alt="" width="400" height="600">
<div class="bc-text" aria-hidden="true"><span class="bc-author">欧长坤</span><span class="bc-title">贝叶斯优化</span><span class="bc-sub">从基本原理到人类偏好</span></div>
</div>
<div class="cover-text">

# 贝叶斯优化 {#sec-home}

<p class="book-subtitle">从基本原理到人类偏好</p>

有些函数评估代价高昂，又无法写出解析式：模型训练一整天后的准确率、穿戴外骨骼行走一分钟后的舒适度、只能由人判断的设计美观程度，都属此类。贝叶斯优化能以很少的评估次数为这类函数找到好的输入。它用概率模型刻画每个尚未尝试的输入可能得到的结果，并把每一次评估安排在模型预期收获最大之处。当唯一的测量手段是由人在两个选项中做出取舍时，同样的思路便成为偏好贝叶斯优化（preferential Bayesian optimization，PBO），而做出取舍的人本身也成为被研究系统的一部分。

本书从基础讲起，不假定读者还记得学校里的概率论与线性代数；在此之上依次讲解高斯过程及其核函数背后的分析、优化循环和基于比较的学习。随后通过四个真实问题完整演示这些方法，并梳理截至 2026 年的研究：哪些结论已得到证明，这些方法用于真人时效果如何，以及心理学、经济学、神经科学与哲学如何看待偏好是否预先存在、有待发现。本书的核心论点是：算法已趋成熟，难点已转向测量，即一次比较究竟测量了什么，以及提问本身如何影响回答者。

</div>
</div>

<div class="cover-live">
<div class="cover-fig">

```{figure}
//| figure: bo-loop
//| class: hero
//| bare: true
//| autoplay: true
//| fig-cap: "一个优化器在一个它看不见的函数（虚线）上工作。蓝色是它的信念，色带是它有多不确定，橙色是它下一步要看的地方。"
steps: 10
```

</div>
<div class="cover-live-text">

每一章都有可交互的图形、可逐步展开的推导，参考文献附在引用它们的小节之后。在部分图中，读者本人就是被优化的对象。

可以从 @sec-optimizing-the-unknown 开始阅读，也可以先读[前言](preface.html)，或从下方[目录](#contents)中选择感兴趣的部分。

本书在 [GitHub](https://github.com/changkun/bobook) 开源。若发现错误、与来源不符的数字或难以理解的段落，欢迎[提交 issue](https://github.com/changkun/bobook/issues)；每一页底部也有为该页提交问题的链接。

</div>
</div>

## 目录 {#contents .unnumbered}

<!--CONTENTS-->

## 三种读法 {#sec-reading-paths .unnumbered}

<div class="paths">
<div class="path">
<p class="path-name">从头开始</p>
<p class="path-who">初学概率，或需要复习。每种工具在使用之前都会先讲清楚。</p>

- [+@sec-optimizing-the-unknown]
- [+@sec-gaussian]
- [+@sec-gp-regression]
- [+@sec-acquisition]
- [+@sec-pbo]

</div>
<div class="path">
<p class="path-name">搭建系统</p>
<p class="path-who">具备一定机器学习基础，希望在有真人参与的场景中实施偏好优化。</p>

- [+@sec-bo-loop]
- [+@sec-gp-preference]
- [+@sec-pbo]
- [+@sec-cs-photo]
- [+@sec-recommendations]

</div>
<div class="path">
<p class="path-name">研究</p>
<p class="path-who">熟悉贝叶斯优化，希望了解该领域的研究现状与待解问题。</p>

- [+@sec-pbo-history]
- [+@sec-observation-models]
- [+@sec-hci]
- [+@sec-what-comparisons-measure]
- [+@sec-open-problems]

</div>
</div>

```{figure}
//| figure: book-map
//| class: map
//| fig-cap: "各章如何层层递进：每个部分占一行。点击某一章，可以看到它依赖哪些章、哪些章依赖它，并可打开这一章。"
```
