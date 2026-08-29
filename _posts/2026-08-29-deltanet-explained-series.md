---
title: "DeltaNet 详解：模型、并行算法与神经架构（三篇合译）"
description: "完整合译 DeltaNet Explained 三篇系列文章：从线性注意力与 Delta Rule，到序列维度并行的分块算法、WY 表示和现代 DeltaNet 神经架构。"
date: 2026-08-29 00:00:00 +0800
categories: [人工智能, 模型架构]
tags: [DeltaNet, 线性注意力, RNN, Delta Rule, 并行算法, Gated DeltaNet]
author: Songlin Yang
translator: Yong/ChatGPT
translation: true
translation_authorized: false
reprint: true
original_title: "DeltaNet Explained (Part I–III)"
original_author: "Songlin Yang"
original_url: "https://sustcsonglin.github.io/blog/2024/deltanet-1/"
original_urls:
  - title: "Part I — The Model"
    url: "https://sustcsonglin.github.io/blog/2024/deltanet-1/"
  - title: "Part II — The Algorithm"
    url: "https://sustcsonglin.github.io/blog/2024/deltanet-2/"
  - title: "Part III — The Neural Architecture"
    url: "https://sustcsonglin.github.io/blog/2024/deltanet-3/"
image: /assets/images/posts/deltanet-explained/delta-net-arch.png
math: true
---

本系列配套解读我们的 NeurIPS 2024 论文——[《Parallelizing Linear Transformers with the Delta Rule over Sequence Length》](https://arxiv.org/abs/2406.06484)。论文由 Songlin Yang 与 [Bailin Wang](https://berlino.github.io/)、[Yu Zhang](https://yzhang.site/)、[Yikang Shen](https://mitibmwatsonailab.mit.edu/people/yikang-shen/) 和 [Yoon Kim](https://people.csail.mit.edu/yoonkim/) 合作完成。你可以在[这里查看实现](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/delta_net.py)，在[这里查看演示幻灯片](https://people.csail.mit.edu/yoonkim/data/efficient_architectures_talk.pdf)。

## 系列目录

1. [第一篇：模型](#part-one)
2. [第二篇：算法](#part-two)
3. [第三篇：神经架构](#part-three)

---

## 第一篇：模型 {#part-one}

### 作为 RNN 的线性注意力

**符号约定：**本文使用大写粗体字母表示矩阵，小写粗体字母表示向量，普通小写字母表示标量。

#### 什么是线性注意力？

标准的 softmax 注意力机制虽然强大，却会带来随序列长度平方增长的复杂度。为了理解线性注意力如何解决这个问题，我们先从标准 softmax 注意力开始（假设只有一个注意力头）：

$$
\begin{aligned}
\mathrm{并行训练：} &&& \mathbf{O} = \mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V} &&\in \mathbb{R}^{L\times d} \\
\mathrm{迭代推理：} &&& \mathbf{o}_t = \sum_{j=1}^t \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j)}{\sum_{l=1}^t\exp(\mathbf{q}^\top_t \mathbf{k}_l)}\mathbf{v}_j &&\in \mathbb{R}^d
\end{aligned}
$$

其中：

- $L$ 表示序列长度；
- $d$ 表示注意力头的维度；
- $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O} \in \mathbb{R}^{L \times d}$ 分别表示 query、key、value 和输出矩阵；
- $\mathbf{M} \in \mathbb{R}^{L \times L}$ 是用于自回归建模的因果掩码，保证每个位置只能关注它之前的位置。

[线性注意力](https://arxiv.org/abs/2006.16236)所做的事情，就是直接移除 softmax 算子[^linear-attention-note]：

$$
\begin{aligned}
\mathrm{并行训练：} &&& \mathbf{O}= (\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V} &&\in \mathbb{R}^{L\times d} \\
\mathrm{迭代推理：} &&& \mathbf{o}_t = \sum_{j=1}^t (\mathbf{q}_t^\top \mathbf{k}_j) \mathbf{v}_j &&\in \mathbb{R}^d
\end{aligned}
$$

仅仅移除 softmax 并不会立刻降低计算复杂度，却会带来一个至关重要的数学性质：**线性**。特别是结合律，使我们可以重排计算，从而显著提高效率。在训练阶段，研究者已经开发出**分块并行（chunkwise parallel）**技术，利用这种线性，在保持硬件效率的同时实现次二次复杂度；这也构成了我们开源 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 库的基础。

在推理阶段，我们同样可以按下面的方式重排计算：

$$
\begin{aligned}
\mathbf{o}_t &= \sum_{j=1}^t \mathbf{v}_j(\mathbf{k}_j^\top \mathbf{q}_t), && \mathbf{k}_j^\top \mathbf{q}_t = \mathbf{q}_t^\top \mathbf{k}_j \in \mathbb{R} \\
&= \left(\sum_{j=1}^t\mathbf{v}_j\mathbf{k}_j^\top\right)\mathbf{q}_t && \text{根据结合律}
\end{aligned}
$$

定义状态矩阵 $\mathbf{S}_t = \sum_{j=1}^t\mathbf{v}_j\mathbf{k}_j^\top$，计算便可以写成：

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t\mathbf{k}_t^\top \in \mathbb{R}^{d\times d}, \qquad
\mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t \in \mathbb{R}^{d}.
$$

这个形式揭示出：线性注意力本质上是一个拥有矩阵值状态 $\mathbf{S}$ 的**线性 RNN**。该状态不断累积 key–value 外积，使状态规模能够高效地从 $\mathcal{O}(d)$ 扩展到 $\mathcal{O}(d^2)$。

<details>
<summary>为什么要扩展状态？</summary>

传统 RNN 的状态更新依赖昂贵的矩阵乘法，因此其隐藏维度往往与输入维度相同，或至少处于同一数量级。然而，RNN 完全依靠循环状态记忆全部历史；状态大小通常会成为保存足够信息的瓶颈，在检索任务中尤其如此。

自 Mamba 1 明确指出这一问题后，我们看到大量工作开始研究硬件高效的状态扩展。线性注意力这种基于外积的更新，已经证明是高效扩大状态的理想方式；Mamba 2 也采用了这一策略。在我们此前的 HGRN2 工作中，我们研究了不同的状态扩展方法，而基于外积的机制同时表现出了良好的性能与可扩展性。

</details>

使用这种方法后，我们只需要保存并更新 $\mathbf{S}_t$，不再需要维护此前所有的 key–value 对。这一优化显著提高了效率：自回归推理的时间复杂度从 $\mathcal{O}(L^2d)$ 降至 $\mathcal{O}(Ld^2)$，空间复杂度则从 $\mathcal{O}(Ld)$ 降至 $\mathcal{O}(d^2)$。它在以下两种场景中特别有优势：

- **长序列建模**：softmax 注意力的二次复杂度可能成为显著瓶颈；
- **生成阶段**：计算通常受内存带宽限制。当 $L \gg d$ 时，移除 KV Cache 可以显著降低推理延迟。

#### 没有免费的午餐：线性注意力的主要局限

遗憾的是，世上没有免费的午餐。线性注意力使用固定大小的状态矩阵，因而无法完美保存全部历史信息；精确检索也就格外困难。

更形式化地说，线性注意力实现的是一种 key–value 联想记忆，即 key 与 value 外积之和：$\mathbf{S} = \sum \mathbf{v}_i\mathbf{k}_i^\top$。假设所有 key 都被归一化为单位长度，当我们试图检索特定 key $\mathbf{k}_j$ 对应的 value 时，会得到：

$$
\begin{aligned}
\mathbf{S}\mathbf{k}_j
&= \sum_i \mathbf{v}_i (\mathbf{k}_i^\top \mathbf{k}_j) \\
&= \mathbf{v}_j + \underbrace{\sum_{i\neq j} (\mathbf{k}_i^\top \mathbf{k}_j)\mathbf{v}_i}_{\text{检索误差}}.
\end{aligned}
$$

为了尽量减小检索误差项，我们需要对所有 $i\neq j$ 都满足 $\mathbf{k}_i^\top \mathbf{k}_j = 0$——换言之，所有 key 都必须彼此**正交**。但这暴露出一个根本限制：在 $d$ 维空间里，最多只能存在 $d$ 个相互正交的向量。这也解释了为什么增大注意力头维度会有帮助：向量空间会有更多“空间”来存储不同的 key–value 对。

这一理论限制会直接反映在实践中：标准线性注意力在语言建模上的表现明显落后于 softmax 注意力。首要原因是记忆“过载”：在这个 key–value 联想记忆系统里，我们只能添加新的关联，却无法擦除已有信息。随着序列增长，“检索误差”不断累积，最终导致性能下降。正如 David Eagleman 在《Livewired: The Inside Story of the Ever-Changing Brain》中所说：

> “记忆的敌人不是时间，而是其他记忆。”

（感谢 Kazuki Irie 提供这条引文。）最近，GLA、Mamba 等带门控的线性注意力变体加入了**遗忘机制**，显著缩小了它们与标准注意力在语言建模任务上的差距。然而，这些模型在上下文检索和精确复制能力上仍然面对根本困难；近期工作已经从实验和理论两方面观察并证明了这些限制。

<details>
<summary>进一步了解线性注意力的门控变体</summary>

线性注意力和 RNN 关系密切，因此研究者自然会想到使用遗忘门来增强线性注意力；无论对非线性 RNN 还是线性 RNN，门控机制都展现了极强的效果：

$$
\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \mathbf{v}_t\mathbf{k}_t^\top.
$$

为了提高参数效率，不同模型会以不同方式对 $\mathbf{G}_t \in \mathbb{R}^{d\times d}$ 进行结构化参数化，其中常见的是外积结构：

Decaying Fast Weight：

$$\mathbf{G}_t = \boldsymbol{\beta}_t \boldsymbol{\alpha}_t^\top.$$

GLA：

$$\mathbf{G}_t = \mathbf{1} \boldsymbol{\alpha}_t^\top.$$

Mamba 1：

$$\mathbf{G}_t = \exp\!\left(-\left(\boldsymbol{\Delta}_t \mathbf{1}^\top\right) \odot \exp(A)\right).$$

Mamba 2：

$$\mathbf{G}_t = \gamma_t \mathbf{1}\mathbf{1}^\top.$$

相关总结可参见 GLA 论文的表 1。

</details>

### DeltaNet：使用 Delta Rule 的线性注意力

#### 什么是 Delta Rule？

[Delta Rule](https://en.wikipedia.org/wiki/Delta_rule) 是神经网络中一种基础的误差校正学习原则。它的核心思想非常简单：根据我们想要得到的结果（目标）与实际得到的结果（预测）之间的差值（delta），调整模型参数。

直观地说，可以想象你在教一个孩子瞄准靶心。如果射得太偏左，你会让他向右调整；如果太偏右，就向左调整。调整幅度取决于偏离目标的程度——这正是 Delta Rule 所表达的概念。

<details>
<summary>展开查看 Delta Rule 代码</summary>

```python
import numpy as np

def delta_rule(x, y, epochs=100, lr=0.1):
    """
    一个简单的 Delta Rule 实现
    x：输入特征（N 个样本 × D 个特征）
    y：目标值（N 个样本）
    """
    # 初始化权重
    w = np.zeros(x.shape[1])

    # 训练
    for _ in range(epochs):
        for i in range(len(x)):
            # 前向计算
            pred = np.dot(x[i], w)

            # 计算误差
            error = y[i] - pred

            # 更新权重
            w += lr * error * x[i]

    return w

# 使用示例
if __name__ == "__main__":
    # 生成玩具数据
    x = np.random.randn(100, 3)  # 100 个样本、3 个特征
    true_w = np.array([0.5, -0.2, 0.1])
    y = np.dot(x, true_w) + 0.1 * np.random.randn(100)

    # 训练
    w = delta_rule(x, y)
    print("真实权重：", true_w)
    print("学习到的权重：", w)
```

</details>

#### 什么是 DeltaNet？

[DeltaNet](https://arxiv.org/abs/2102.11174) 将这种误差校正原则应用到线性注意力中。它不是简单累加 key–value 外积，而是根据预测误差更新状态：

$$
\begin{aligned}
\mathbf{S}_{t}
&= \mathbf{S}_{t-1} - \beta_t(\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t)\mathbf{k}_t^\top \\
&= \mathbf{S}_{t-1} - \beta_t \mathbf{S}_{t-1} \mathbf{k}_t \mathbf{k}_t^\top + \beta_t \mathbf{v}_t \mathbf{k}_t^\top.
\end{aligned}
$$

将各组成部分拆开看，它与 Delta Rule 的对应关系就很清楚了：

- $\beta_t \in \mathbb{R}$ 相当于学习率；
- $\mathbf{k}_t \in \mathbb{R}^d$ 是输入数据；
- $\mathbf{v}_t \in \mathbb{R}^d$ 是目标；
- $\mathbf{S}_{t-1}\mathbf{k}_t \in \mathbb{R}^d$ 是当前预测。

后文还会重新讨论这一形式，并说明它如何从在线损失函数上的一次梯度下降中自然产生。

理解这个更新规则还有另一种直观方式。可以把 $\mathbf{S}_{t-1}\mathbf{k}_t$ 看作从记忆中检索当前 key $\mathbf{k}_t$ 所关联的“旧 value”。当同一个 key 出现了新关联的 value $\mathbf{v}_t$ 时，我们不会盲目覆盖，而是进行一次审慎更新：

$$
\begin{aligned}
\mathbf{v}_t^{\mathrm{new}} &= (1-\beta_t) \mathbf{v}_t^{\mathrm{old}} + \beta_t \mathbf{v}_t, \\
\mathbf{S}_t &= \mathbf{S}_{t-1}
- \underbrace{\mathbf{v}_t^{\mathrm{old}} \mathbf{k}_t^\top}_{\text{擦除}}
+ \underbrace{\mathbf{v}_t^{\mathrm{new}} \mathbf{k}_t^\top}_{\text{写入}}.
\end{aligned}
$$

$\mathbf{v}_t^{\mathrm{new}}$ 是旧 value 与当前 value 的学习型组合，由动态的 $\beta_t \in (0,1)$ 控制：当 $\beta_t=0$ 时，记忆内容保持不变；当 $\beta_t=1$ 时，旧的关联 value 会被新的 value 完全替换。

#### DeltaNet：强大的上下文学习 RNN

MQAR（Multi-Query Associative Recall，多查询联想回忆）是近期常用的合成基准，用于衡量次二次复杂度模型的上下文联想回忆能力。

MQAR 任务的形式如下：每个字母与一个数字关联，模型需要正确回忆查询序列中各个字母对应的数字。例如，给定输入：

`A 4 B 3 C 6 F 1 E 2 → A ? C ? F ? E ? B ?`

其格式由两部分组成：

1. 箭头前是 key–value 对，即字母及其对应数字；
2. 箭头后是查询序列，即需要回忆对应数字的字母。

该示例的正确输出是：

`4, 6, 1, 2, 3`

传统的门控卷积和循环模型通常在这项任务上表现不佳，但我们的实验显示，DeltaNet[^deltanet-overlooked] 展现出了格外强劲的性能：

<figure>
  <img src="/assets/images/posts/deltanet-explained/mqar-1.png" alt="DeltaNet 在最困难 MQAR 设置上的表现">
  <figcaption>原 Zoology 论文中最困难的设置。</figcaption>
</figure>

最初的结果令人十分振奋：DeltaNet 在 MQAR 上取得完美表现，超出了我们的预期。更有希望的是，MQAR 表现与真实语言建模任务中的“Associative-Recall-Hit”高度相关。联想回忆失败是次二次模型的主要错误来源，也在很大程度上解释了它们与 softmax 注意力之间的困惑度差距。因此，DeltaNet 在 MQAR 上的完美表现意味着它有很大的潜力。

我们还在 MAD 上进行了实验。MAD 是另一个用于检验新架构能力的综合基准，覆盖面比 MQAR 更广。结果如下：

| 模型 | 压缩 | 模糊回忆 | 上下文回忆 | 记忆 | 噪声回忆 | 选择性复制 | 平均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer | 51.6 | 29.8 | 94.1 | 85.2 | 86.8 | 99.6 | 74.5 |
| Hyena | 45.2 | 7.9 | 81.7 | 89.5 | 78.8 | 93.1 | 66.0 |
| Multihead Hyena | 44.8 | 14.4 | 99.0 | 89.4 | 98.6 | 93.0 | 73.2 |
| Mamba | 52.7 | 6.7 | 90.4 | 89.5 | 90.1 | 86.3 | 69.3 |
| GLA | 38.8 | 6.9 | 80.8 | 63.3 | 81.6 | 88.6 | 60.0 |
| DeltaNet | 42.2 | 35.7 | 100 | 52.8 | 100 | 100 | 71.8 |

这些结果表明，DeltaNet 具有很强的上下文回忆能力。这些合成任务的运行成本很低，却清楚表明 DeltaNet 很可能在规模化后继续表现良好。于是，我们开始集中精力开发 DeltaNet 的训练算法和 kernel 实现——毕竟，如果不能先证明某种架构具有潜力，就贸然扩大它的规模，很可能浪费大量时间和资源。

下一篇将介绍一种优雅的算法，它可以让 DeltaNet 在序列长度维度上并行。不过在此之前，我们先建立一些直觉，理解 DeltaNet 为什么特别适合上下文检索任务。

#### 与线性注意力相比，DeltaNet 为什么更擅长上下文检索？

DeltaNet 的更新规则可以通过梯度下降，在每个时间步 $t$ 依次最小化期望输出与预测输出之间的均方误差（MSE）推导出来[^ttt-connection]：

$$
\mathcal{L}_t(\mathbf{S}) = \frac{1}{2}\|\mathbf{S} \mathbf{k}_t - \mathbf{v}_t\|^2.
$$

使用梯度下降最小化该 MSE 损失，可以得到：

$$
\begin{aligned}
\mathbf{S}_t
&= \mathbf{S}_{t-1} - \eta_t \nabla \mathcal{L}_t(\mathbf{S}_{t-1}) \\
&= \mathbf{S}_{t-1} - \eta_t \left(\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t\right) \mathbf{k}_t^\top.
\end{aligned}
$$

令学习率 $\eta_t=\beta_t$，就会得到 DeltaNet 的更新规则。

相比之下，标准线性注意力使用的是线性损失函数：

$$
\mathcal{L}^\prime_t(\mathbf{S}) = -\langle \mathbf{S} \mathbf{k}_t, \mathbf{v}_t \rangle.
$$

线性注意力相应的更新规则为：

$$
\begin{aligned}
\mathbf{S}_t
&= \mathbf{S}_{t-1} - \eta_t \nabla \mathcal{L}_t^\prime(\mathbf{S}_{t-1}) \\
&= \mathbf{S}_{t-1} + \eta_t \mathbf{v}_t \mathbf{k}_t^\top.
\end{aligned}
$$

令 $\eta_t=1$，便恢复了标准线性注意力的更新。

由此，DeltaNet 在上下文检索上的优势就很清楚了：它会在每一步最小化 MSE，因此非常适合联想回忆之类的任务——对这些任务来说，降低大误差是准确检索的关键。

[^linear-attention-note]: 最初的线性注意力形式还会在 query 和 key 上加入特征映射，并带有归一化项；近期研究表明，这两个部分可能并非必需。
[^deltanet-overlooked]: 有趣的是，DeltaNet 最初就是为提高联想回忆性能而设计的，但在这项工作之前，它在很大程度上一直被忽视。
[^ttt-connection]: 这一形式揭示了它与 Test-Time Training（TTT）之间一个有趣的联系：在移除层归一化等非线性组件，并将 TTT 的 mini-batch 大小设为 1 时，DeltaNet 与 TTT-linear 在数学上等价。

---

## 第二篇：算法 {#part-two}

### DeltaNet 的并行扫描：一次失败的尝试

上一部分已经看到，DeltaNet 在这些诊断性合成任务上表现很好。那么，接下来只要把它扩大到现代语言模型的规模就可以了，对吗？事实并没有这么简单。原始 DeltaNet 被当作纯 RNN 处理，需要执行 $\mathcal{O}(L)$ 个串行步骤；面对拥有大规模并行计算能力的 GPU，这种方式效率很低。

因此，我们需要找到一种沿序列长度维度并行化 DeltaNet 的方法，以实现硬件高效的训练。本篇先讨论一种有趣但不实用的方案——并行扫描（parallel scan），然后再给出另一种实践中更加高效的并行算法。

#### 从 Delta 更新到矩阵乘法形式

从 DeltaNet 原始的状态更新方程开始：

$$
\mathbf{S}_{t} = \mathbf{S}_{t-1} - \beta_t(\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t)\mathbf{k}_t^\top.
$$

为了把它转换成矩阵乘法形式，逐步展开：

$$
\begin{aligned}
\mathbf{S}_{t}
&= \mathbf{S}_{t-1} - \beta_t(\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t)\mathbf{k}_t^\top \\
&= \mathbf{S}_{t-1} - \beta_t \mathbf{S}_{t-1} \mathbf{k}_t \mathbf{k}_t^\top + \beta_t \mathbf{v}_t \mathbf{k}_t^\top \\
&= \mathbf{S}_{t-1} (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\intercal) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top.
\end{aligned}
$$

为简化记号，定义：

- 转移矩阵 $\mathbf{M}_t = \mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\intercal$；
- 更新项 $\mathbf{X}_t = \beta_t \mathbf{v}_t \mathbf{k}_t^\top$。

状态更新于是变成：

$$
\mathbf{S}_{t} = \mathbf{S}_{t-1}\mathbf{M}_t + \mathbf{X}_t \in \mathbb{R}^{d\times d}.
$$

#### 定义结合算子

这一形式与经典资料 [*Prefix Sums and Their Applications*](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf) 中式（1.5）的一阶递推完全一致。在这个框架中，矩阵乘法（$\otimes$）和矩阵加法（$\oplus$）充当二元算子，并满足所需性质：

1. 矩阵加法满足结合律：$(A+B)+C=A+(B+C)$；
2. 矩阵乘法满足结合律：$(AB)C=A(BC)$；
3. 矩阵乘法对加法满足分配律：$A(B+C)=AB+AC$。

根据该框架，把每一步的状态对定义为：

$$
c_t = [\mathbf{M}_t, \mathbf{X}_t]
= [\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\intercal,\, \beta_t \mathbf{v}_t \mathbf{k}_t^\top].
$$

再定义用于合并这些状态对的结合算子 $\bullet$：

$$
c_i \bullet c_j = [\mathbf{M}_i\mathbf{M}_j,\, \mathbf{M}_j\mathbf{X}_i + \mathbf{X}_j].
$$

这个算子保留了更新中的时间依赖：合并两个时间步时，较早的更新项 $\mathbf{X}_i$ 必须先经过较晚的转移矩阵 $\mathbf{M}_j$ 变换，而较晚的更新项 $\mathbf{X}_j$ 保持不变。

#### DeltaNet 的并行扫描

<figure>
  <img src="/assets/images/posts/deltanet-explained/scan.png" alt="DeltaNet 并行扫描示意图">
  <figcaption>并行扫描示意图。</figcaption>
</figure>

有了这个结合算子，我们就能用并行扫描同时计算全部状态。算法分成两个阶段。

##### 向下扫描阶段（Sweep-Down）

首先并行合并相邻的状态对，得到部分结果。对步骤 0 和 1：

$$
c_1 = c_0 \bullet c_1
= [\mathbf{M}_0\mathbf{M}_1,\, \mathbf{M}_1\mathbf{X}_0 + \mathbf{X}_1].
$$

对步骤 2 和 3：

$$
c_3 = c_2 \bullet c_3
= [\mathbf{M}_2\mathbf{M}_3,\, \mathbf{M}_3\mathbf{X}_2 + \mathbf{X}_3].
$$

然后合并这两个结果：

$$
\begin{aligned}
c_{1:3} = c_1 \bullet c_3
= [&\mathbf{M}_0\mathbf{M}_1\mathbf{M}_2\mathbf{M}_3,\\
&\mathbf{M}_2\mathbf{M}_3(\mathbf{M}_1\mathbf{X}_0 + \mathbf{X}_1)
+ \mathbf{M}_3\mathbf{X}_2 + \mathbf{X}_3].
\end{aligned}
$$

##### 向上扫描阶段（Sweep-Up）

在这一阶段，使用部分结果计算中间状态：

$$
c_2 = c_1 \bullet c_2
= [\mathbf{M}_1\mathbf{M}_2,\, \mathbf{M}_2\mathbf{X}_1 + \mathbf{X}_2].
$$

这种并行化把 DeltaNet 的串行状态更新转换成并行计算，在保持数学等价的同时，将串行依赖链从 $\mathcal{O}(L)$ 步缩短到 $\mathcal{O}(\log L)$ 步。

#### DeltaNet 的并行扫描有什么问题？

尽管可以并行，DeltaNet 的并行扫描仍面对两个主要问题：计算复杂度和内存需求。

第一个问题是**时间复杂度**。如果把 $\mathbf{M}_t$ 当作稠密矩阵，矩阵乘法具有三次方开销，DeltaNet 的并行扫描复杂度会达到 $\mathcal{O}(L\log L\,d^3)$。乍看之下，我们似乎可以利用 $\mathbf{M}_t$ 的“单位矩阵加低秩项”结构加速。下面仔细推导一下。

两个相邻矩阵相乘时：

$$
\begin{aligned}
&(\mathbf{I}-\beta_0 \mathbf{k}_0 \mathbf{k}_0^\top)
(\mathbf{I} - \beta_1 \mathbf{k}_1 \mathbf{k}_1^\top) \\
&= \mathbf{I}(\mathbf{I} - \beta_1 \mathbf{k}_1 \mathbf{k}_1^\top)
- \beta_0 \mathbf{k}_0 \mathbf{k}_0^\top(\mathbf{I} - \beta_1 \mathbf{k}_1 \mathbf{k}_1^\top) \\
&= (\mathbf{I} - \beta_1 \mathbf{k}_1 \mathbf{k}_1^\top)
- \beta_0 \mathbf{k}_0 \mathbf{k}_0^\top
+ \beta_0\beta_1 \mathbf{k}_0 \mathbf{k}_0^\top \mathbf{k}_1 \mathbf{k}_1^\top \\
&= \mathbf{I} - \beta_1 \mathbf{k}_1 \mathbf{k}_1^\top
- \beta_0 \mathbf{k}_0 \mathbf{k}_0^\top
+ \beta_0\beta_1 \mathbf{k}_0 (\mathbf{k}_0^\top \mathbf{k}_1) \mathbf{k}_1^\top.
\end{aligned}
$$

利用单位矩阵加低秩项的结构，计算复杂度可以从 $\mathcal{O}(d^3)$ 降至 $\mathcal{O}(d^2)$：我们只需要计算向量内积 $(\mathbf{k}_0^\top \mathbf{k}_1)$ 和向量之间的外积。下一对矩阵同理：

$$
\begin{aligned}
&(\mathbf{I}-\beta_2 \mathbf{k}_2 \mathbf{k}_2^\top)
(\mathbf{I} - \beta_3 \mathbf{k}_3 \mathbf{k}_3^\top) \\
&= \mathbf{I} - \beta_3 \mathbf{k}_3 \mathbf{k}_3^\top
- \beta_2 \mathbf{k}_2 \mathbf{k}_2^\top
+ \beta_2\beta_3 \mathbf{k}_2 (\mathbf{k}_2^\top \mathbf{k}_3) \mathbf{k}_3^\top.
\end{aligned}
$$

然而，当我们继续合并这些结果、计算 $c_{1:4}$ 之类的更大跨度时，乘法会迅速变得复杂。我们需要计算：

$$
\begin{aligned}
&(\mathbf{I} - \beta_1 \mathbf{k}_1 \mathbf{k}_1^\top
- \beta_0 \mathbf{k}_0 \mathbf{k}_0^\top
+ \beta_0\beta_1 \mathbf{k}_0 (\mathbf{k}_0^\top \mathbf{k}_1) \mathbf{k}_1^\top)\\
&\quad\cdot
(\mathbf{I} - \beta_3 \mathbf{k}_3 \mathbf{k}_3^\top
- \beta_2 \mathbf{k}_2 \mathbf{k}_2^\top
+ \beta_2\beta_3 \mathbf{k}_2 (\mathbf{k}_2^\top \mathbf{k}_3) \mathbf{k}_3^\top).
\end{aligned}
$$

第一个括号中的每一项都必须与第二个括号中的每一项相乘。虽然每个矩阵最初只是 $\mathcal{O}(1)$ 个秩 1 项之和，但乘法会使项数呈二次增长。经过 $\log L$ 层并行扫描后，最终会得到 $\mathcal{O}(L^{\log c})$ 个项，其中 $c$ 是每个矩阵最初包含的项数。即便每一项仍然是秩 1，这种项数的指数式增长也使显式维护该结构变得不切实际。因此，将它们视作稠密矩阵、接受 $\mathcal{O}(d^3L\log L)$ 复杂度反而更加合理，尤其是考虑到现代硬件对稠密矩阵运算的高效支持。这就是并行扫描在理论上很吸引人，却在 DeltaNet 实际计算中面对严重困难的原因。

第二个主要问题是**空间复杂度**。并行扫描必须在每一步把所有中间的 $d\times d$ 矩阵写入高带宽内存（HBM）。对于拥有矩阵值状态的线性 RNN，这种物化开销高达 $\mathcal{O}(Ld^2)$，代价难以承受。循环计算可以避免这种物化[^recurrent-materialization]，但并行扫描似乎没有明显的绕过方式，除非所有状态都能装进 SRAM。Mamba 的硬件感知选择性扫描算法采用的就是这种方法，因此无需物化中间状态；但它会限制状态大小——状态太大就会耗尽共享内存。鉴于 I/O 成本支配了这类计算，并行扫描在实践中可能并不值得采用。

相关讨论可参见 [François Fleuret 的帖子](https://x.com/francoisfleuret/status/1793016689589625263)和[作者此前关于分块算法的讨论](https://x.com/SonglinYang4/status/1793029555277697379)。分块算法是另一种结合扫描，它的内存效率更高，并通过执行更多矩阵乘法来提高 Tensor Core 的利用率。因而，如果能为 DeltaNet 开发一种相对于 $d$ 保持二次复杂度、同时保留内存效率的分块训练算法，将非常有价值。

### DeltaNet 的分块算法

#### 线性注意力的分块并行形式

线性注意力之所以高效，是因为它可以使用向量来维护紧凑的状态表示，而不必物化完整矩阵。外积之和可以改写为矩阵乘法：

$$
\begin{aligned}
\sum_{i=1}^t \mathbf{v}_i \mathbf{k}_i^\top &= \mathbf{V}_t\mathbf{K}_t^\top, \\
\text{其中 } \mathbf{V}_t &= [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_t], \\
\mathbf{K}_t &= [\mathbf{k}_1, \mathbf{k}_2, \ldots, \mathbf{k}_t].
\end{aligned}
$$

现代 GPU 的 Tensor Core 对这种矩阵乘法做了高度优化。利用这一性质，我们不再保存全部中间隐藏状态，而是以大小为 $C$ 的固定间隔保存状态作为检查点。于是，只需保存 $\mathbf{S}_{0}, \mathbf{S}_{C}, \mathbf{S}_{2C}, \ldots, \mathbf{S}_{(n-1)C}$，其中 $n=\lceil L/C\rceil$。

记 $\mathbf{S}_{[i]} := \mathbf{S}_{iC} \in \mathbb{R}^{d\times d}$；对 $\square \in \{\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O}\}$，记 $\square_{[i]} = \square_{iC+1:(i+1)C} \in \mathbb{R}^{C\times d}$；对 $\square \in \{\mathbf{q},\mathbf{k},\mathbf{v},\mathbf{o},\mathbf{S}\}$，记 $\square_{[i]}^r = \square_{iC+r}$。对于分块 $i$ 内的任意位置 $r$，可以计算：

$$
\begin{aligned}
\mathbf{S}_{[i]}^r
&= \mathbf{S}_{[i]} + \sum_{t=1}^{r} \mathbf{v}_{[i]}^t \mathbf{k}_{[i]}^{t\top}, \\
\mathbf{o}_{[i]}^r
&= \mathbf{S}_{[i]}^r \mathbf{q}_{[i]}^r \\
&= \mathbf{S}_{[i]}\mathbf{q}_{[i]}^r
+ \sum_{t=1}^{r} \mathbf{v}_{[i]}^t (\mathbf{k}^{t\top}_{[i]} \mathbf{q}_{[i]}^r).
\end{aligned}
$$

写成矩阵形式：

$$
\begin{aligned}
\mathbf{S}_{[t+1]} &= \mathbf{S}_{[t]} + \mathbf{V}_{[t]}^\top \mathbf{K}_{[t]} &&\in \mathbb{R}^{d\times d}, \\
\mathbf{O}_{[t]} &= \mathbf{Q}_{[t]} \mathbf{S}_{[t]}^\top
+ (\mathbf{Q}_{[t]}\mathbf{K}_{[t]}^\top \odot \mathbf{M}) \mathbf{V}_{[t]} &&\in \mathbb{R}^{C\times d}.
\end{aligned}
$$

<figure>
  <img src="/assets/images/posts/deltanet-explained/chunk-linear-attn.png" alt="线性注意力的分块并行算法">
  <figcaption>线性注意力分块算法的可视化表示。</figcaption>
</figure>

当分块大小 $C$ 是 16 的倍数时，这种分块形式能够利用 Tensor Core，获得很高的硬件利用率；我们的开源库 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 已经实现了这一算法。

#### DeltaNet 的 WY 表示

前面的失败尝试说明，DeltaNet 转移矩阵的累积乘积似乎很难被紧凑表示，仿佛必须保存大量中间结果。幸运的是，我们还有办法：DeltaNet 的转移矩阵与 [Householder 矩阵](https://en.wikipedia.org/wiki/Householder_transformation)十分相似（当 $\beta_t=2$ 时），而 Householder 矩阵的累积乘积存在一种优雅的紧凑表示。

<figure class="figure-narrow">
  <img src="/assets/images/posts/deltanet-explained/householder.png" alt="Householder 反射变换">
  <figcaption>Householder 反射变换的可视化表示。</figcaption>
</figure>

这种表示称为 **WY 表示**，最早由 1985 年的一篇开创性论文提出。使用 DeltaNet 的记号，累积乘积可以写成：

$$
\prod_{i=1}^{t} (\mathbf{I} - \beta_i \mathbf{k}_i \mathbf{k}_i^\top)
= \mathbf{I} - \sum_{i=1}^t \mathbf{w}_i\mathbf{k}_i^\top.
$$

可以用数学归纳法证明。定义 $\mathbf{P}_n = \prod_{t=1}^n(\mathbf{I}-\beta_t\mathbf{k}_t\mathbf{k}_t^\top)$。当 $n=1$ 时等式显然成立；假设它对 $n-1$ 成立，则对 $n$ 有：

$$
\begin{aligned}
\mathbf{P}_n
&= \mathbf{P}_{n-1} (\mathbf{I} - \beta_n \mathbf{k}_n \mathbf{k}_n^\top) \\
&= \left(\mathbf{I} - \sum_{t=1}^{n-1} \mathbf{w}_t \mathbf{k}_t^\top\right)
   (\mathbf{I} - \beta_n \mathbf{k}_n \mathbf{k}_n^\top) \\
&= \mathbf{I} - \sum_{t=1}^{n-1} \mathbf{w}_t \mathbf{k}_t^\top
- \beta_n \mathbf{k}_n \mathbf{k}_n^\top
+ \left(\sum_{t=1}^{n-1} \mathbf{w}_t \mathbf{k}_t^\top\right)
  \beta_n \mathbf{k}_n \mathbf{k}_n^\top \\
&= \mathbf{I} - \sum_{t=1}^{n-1} \mathbf{w}_t \mathbf{k}_t^\top
- \underbrace{\left(\beta_n \mathbf{k}_n
- \beta_n \sum_{t=1}^{n-1}\mathbf{w}_t(\mathbf{k}_t^\top\mathbf{k}_n)\right)}_{\mathbf{w}_n}
  \mathbf{k}_n^\top \\
&= \mathbf{I} - \sum_{t=1}^{n} \mathbf{w}_t\mathbf{k}_t^\top.
\end{aligned}
$$

这个证明不仅说明了该表示的正确性，也给出了计算 $\mathbf{w}$ 向量的构造方法。

同样，可以用归纳法证明 $\mathbf{S}_n = \sum_{t=1}^{n} \mathbf{u}_t\mathbf{k}_t^\top$：

$$
\begin{aligned}
\mathbf{S}_n
&= \mathbf{S}_{n-1}(\mathbf{I}-\beta_n\mathbf{k}_n\mathbf{k}_n^\top)
+ \beta_n\mathbf{v}_n\mathbf{k}_n^\top \\
&= \left(\sum_{t=1}^{n-1}\mathbf{u}_t\mathbf{k}_t^\top\right)
   (\mathbf{I}-\beta_n\mathbf{k}_n\mathbf{k}_n^\top)
+ \beta_n\mathbf{v}_n\mathbf{k}_n^\top \\
&= \sum_{t=1}^{n-1}\mathbf{u}_t\mathbf{k}_t^\top
- \left(\sum_{t=1}^{n-1}\mathbf{u}_t\mathbf{k}_t^\top\right)
  \beta_n\mathbf{k}_n\mathbf{k}_n^\top
+ \beta_n\mathbf{v}_n\mathbf{k}_n^\top \\
&= \sum_{t=1}^{n-1}\mathbf{u}_t\mathbf{k}_t^\top
+ \underbrace{\left(\beta_n\mathbf{v}_n
- \beta_n\sum_{t=1}^{n-1}\mathbf{u}_t(\mathbf{k}_t^\top\mathbf{k}_n)\right)}_{\mathbf{u}_n}
  \mathbf{k}_n^\top \\
&= \sum_{t=1}^{n}\mathbf{u}_t\mathbf{k}_t^\top.
\end{aligned}
$$

观察这个外积之和的结构，可以发现它与线性注意力的更新形式非常接近。这种相似性为开发新的并行算法提供了路径。

#### DeltaNet 的分块并行形式

首先展开 DeltaNet 的递推：

$$
\begin{aligned}
\mathbf{S}_t
&= \mathbf{S}_{t-1}(\mathbf{I}-\beta_t\mathbf{k}_t\mathbf{k}_t^\top)
+ \beta_t\mathbf{v}_t\mathbf{k}_t^\top \\
&= \sum_{i=1}^t \beta_i(\mathbf{v}_i\mathbf{k}_i^\top)
   \left(\prod_{j=i+1}^t(\mathbf{I}-\beta_j\mathbf{k}_j\mathbf{k}_j^\top)\right).
\end{aligned}
$$

与线性注意力类似，我们可以使用检查点，以大小为 $C$ 的固定间隔保存状态。对分块 $i$ 内的任意位置 $r$：

$$
\begin{aligned}
\mathbf{S}_{[i]}^r
&= \mathbf{S}_{[i]}
\underbrace{\prod_{t=1}^{r}(\mathbf{I}-\beta_{[i]}^t\mathbf{k}_{[i]}^t\mathbf{k}_{[i]}^{t\top})}_{\text{分块内累积乘积：}\mathbf{P}_{[i]}^r} \\
&\quad+
\underbrace{\sum_{t=1}^{r}
\left(\beta_{[i]}^t\mathbf{v}_{[i]}^t\mathbf{k}_{[i]}^{t\top}
\prod_{s=t+1}^{r}(\mathbf{I}-\beta_{[i]}^s\mathbf{k}_{[i]}^s\mathbf{k}_{[i]}^{s\top})\right)}_{\text{分块内状态（累积乘积和）：}\mathbf{H}_{[i]}^r} \\
&= \mathbf{S}_{[i]}\left(\mathbf{I}-\sum_{t=1}^r\mathbf{w}_{[i]}^t\mathbf{k}_{[i]}^{t\top}\right)
+ \sum_{t=1}^r\mathbf{u}_{[i]}^t\mathbf{k}_{[i]}^{t\top}.
\end{aligned}
$$

$\mathbf{w}_{[i]}^t$ 和 $\mathbf{u}_{[i]}^t$ 使用 WY 表示计算，但计算从每个分块的第一个位置开始，而不是从整条序列的开头开始，因此各分块可以并行处理：

$$
\mathbf{w}_{[t]}^r = \beta_{[t]}^r
\left(\mathbf{k}_{[t]}^r - \sum_{i=1}^{r-1}\mathbf{w}_{[t]}^i
(\mathbf{k}_{[t]}^i)^\top\mathbf{k}_{[t]}^r\right),
$$

$$
\mathbf{u}_{[t]}^r = \beta_{[t]}^r
\left(\mathbf{v}_{[t]}^r - \sum_{i=1}^{r-1}\mathbf{u}_{[t]}^i
(\mathbf{k}_{[t]}^i)^\top\mathbf{k}_{[t]}^r\right).
$$

对于输出计算：

$$
\begin{aligned}
\mathbf{o}_{[i]}^r
&= \mathbf{S}_{[i]}^r\mathbf{q}_{[i]}^r \\
&= \mathbf{S}_{[i]}\mathbf{q}_{[i]}^r
- \sum_{t=1}^r\mathbf{S}_{[i]}\mathbf{w}_{[i]}^t
  (\mathbf{k}_{[i]}^{t\top}\mathbf{q}_{[i]}^r)
+ \sum_{t=1}^r\mathbf{u}_{[i]}^t
  (\mathbf{k}_{[i]}^{t\top}\mathbf{q}_{[i]}^r) \\
&= \mathbf{S}_{[i]}\mathbf{q}_{[i]}^r
+ \sum_{t=1}^r(\mathbf{u}_{[i]}^t-\mathbf{S}_{[i]}\mathbf{w}_{[i]}^t)
  (\mathbf{k}_{[i]}^{t\top}\mathbf{q}_{[i]}^r).
\end{aligned}
$$

合并后，可以写成矩阵乘法形式：

$$
\begin{aligned}
\mathbf{S}_{[i+1]}
&= \mathbf{S}_{[i]}(\mathbf{I}-\mathbf{W}_{[i]}^\top\mathbf{K}_{[i]})
+ \mathbf{U}_{[i]}^\top\mathbf{K}_{[i]} \\
&= \mathbf{S}_{[i]}
+ \left(\mathbf{U}_{[i]}-\mathbf{W}_{[i]}\mathbf{S}_{[i]}^\top\right)^\top
  \mathbf{K}_{[i]} &&\in \mathbb{R}^{d\times d}, \\
\mathbf{O}_{[i]}
&= \mathbf{Q}_{[i]}\mathbf{S}_{[i]}^\top
+ (\mathbf{Q}_{[i]}\mathbf{K}_{[i]}^\top\odot\mathbf{M})
  \left(\mathbf{U}_{[i]}-\mathbf{W}_{[i]}\mathbf{S}_{[i]}^\top\right)
  &&\in \mathbb{R}^{C\times d}.
\end{aligned}
$$

<figure>
  <img src="/assets/images/posts/deltanet-explained/delta-chunk.png" alt="DeltaNet 的分块并行算法">
  <figcaption>DeltaNet 分块算法的可视化表示。</figcaption>
</figure>

#### 从图论角度理解 UT 变换

分块并行形式把 DeltaNet 的大部分操作转换成了与线性注意力类似的高效矩阵乘法。不过，仍有一个关键计算瓶颈：更新向量 $\mathbf{U}_{[i]}$ 和 $\mathbf{W}_{[i]}$ 的递归构造。

这正是需要 **UT 变换**的原因：把递归计算重构成可以利用高效矩阵乘法的形式。下面从图论角度理解它。

在图论中，对一个带权有向图，邻接矩阵 $\mathbf{A}$ 表示节点间的直接连接：$\mathbf{A}[i,j]$ 是从节点 $j$ 指向节点 $i$ 的边权。当计算 $(\mathbf{I}-\mathbf{A})^{-1}$ 时，其中每个元素 $[i,j]$ 都表示从 $j$ 到 $i$ 的所有可能路径的权重之和。

重新观察递归更新方程：

$$
\mathbf{w}_{[t]}^r = \beta_{[t]}^r
\left(\mathbf{k}_{[t]}^r - \sum_{i=1}^{r-1}\mathbf{w}_{[t]}^i
(\mathbf{k}_{[t]}^i)^\top\mathbf{k}_{[t]}^r\right),
$$

$$
\mathbf{u}_{[t]}^r = \beta_{[t]}^r
\left(\mathbf{v}_{[t]}^r - \sum_{i=1}^{r-1}\mathbf{u}_{[t]}^i
(\mathbf{k}_{[t]}^i)^\top\mathbf{k}_{[t]}^r\right).
$$

它们构成一个带权有向图：

- 节点表示序列位置；
- 当 $i<r$ 时，有一条从位置 $i$ 指向位置 $r$ 的有向边，对应因果依赖；
- 边权 $-\beta_{[t]}^r\mathbf{k}_{[t]}^{i\top}\mathbf{k}_{[t]}^r$ 通过 key 相似度和学习率编码交互。

该图的邻接矩阵可以高效计算：

$$
\mathbf{A}_{[t]} = \operatorname{tril}
\left(-\operatorname{diag}(\boldsymbol{\beta}_{[t]})
\mathbf{K}_{[t]}\mathbf{K}_{[t]}^\top,-1\right).
$$

由于 $\mathbf{A}_{[t]}$ 是严格下三角矩阵，$\mathbf{I}-\mathbf{A}_{[t]}$ 也是对角线全为 1 的下三角矩阵。借助这一特殊结构，可以通过前向代入高效求逆：

$$
\mathbf{T}_{[t]} = (\mathbf{I}-\mathbf{A}_{[t]})^{-1}.
$$

这避免了通用矩阵求逆，显著提高了计算效率。得到能够刻画各位置之间全部累积影响路径的 $\mathbf{T}_{[t]}$ 后，再执行最终乘法：

$$
\mathbf{W}_{[t]} = \mathbf{T}_{[t]}
\operatorname{diag}(\boldsymbol{\beta}_{[t]})\mathbf{K}_{[t]}, \qquad
\mathbf{U}_{[t]} = \mathbf{T}_{[t]}
\operatorname{diag}(\boldsymbol{\beta}_{[t]})\mathbf{V}_{[t]}.
$$

这样就能应用这些累积影响，以硬件高效的形式计算更新。

#### 速度比较

我们使用 Triton 分别实现了 DeltaNet 的循环版本和分块并行版本。实验比较了不同序列长度 $L$ 和注意力头维度 $d_{\text{head}}$ 下的性能，模型维度固定为 $d=2048$。为确保不同配置间的比较公平，我们通过调整 batch size，把序列元素总数固定在 16,384。

<figure class="figure-medium">
  <img src="/assets/images/posts/deltanet-explained/speedup.png" alt="DeltaNet 循环与分块并行实现的速度比较">
  <figcaption>不同序列长度与注意力头维度下的速度比较。</figcaption>
</figure>

如图所示，分块并行方法始终快于循环基线。更重要的是，在两种条件下，这种性能优势会变得更加明显：序列更长，或者注意力头维度更大。要理解原因，需要分析分块方法所解决的循环实现的两个根本限制。

第一个限制与并行策略有关。循环实现逐步处理序列，主要依赖两个维度的并行来让 GPU 核心保持忙碌：batch 维度（同时处理多条序列）和注意力头维度（并行计算多个注意力头）。在序列长度适中、batch size 较大时，这种策略表现不错；但在现代训练场景中，它会遇到困难。如今的模型越来越多地使用更长序列或更多参数，通常必须缩小 batch size 才能提高内存效率。FlashAttention 2 论文特别强调了这种变化，并指出序列级并行对于训练至关重要。如果不能沿序列维度并行，循环实现会撞上一个根本瓶颈：当 batch size 与注意力头数的乘积较小时，无法提供足够的并行工作来充分利用现代 GPU，最终造成流式多处理器（SM）占用率低、速度不理想。

第二个限制与硬件利用率有关。现代 GPU 包含专门加速矩阵乘法的 Tensor Core；对于半精度计算，与 FLOP 数相同的其他操作相比，它们最高可以带来约 16 倍加速。循环实现虽然总 FLOP 数更少，却难以有效利用这些硬件加速器。注意力头维度越大，这个问题越严重；而需要大记忆容量的任务（例如上下文检索）往往恰恰需要较大的头维度。相比之下，分块实现重构了计算，使 Tensor Core 的使用最大化。尽管理论 FLOP 数更高，真实运行性能反而更好。

这项性能分析说明了现代硬件高效深度学习中的一个重要原则：原始 FLOP 数并不总能直接转化成墙钟时间。能否利用专用硬件加速器并维持较高的 GPU 利用率，通常比理论操作数更重要。分块实现之所以成功，正是因为它让计算形式与硬件现实相匹配。

最后，我们比较了 13 亿参数规模下 DeltaNet 与其他模型的训练吞吐量。

<figure class="figure-medium">
  <img src="/assets/images/posts/deltanet-explained/throughputs.png" alt="13 亿参数规模模型的训练吞吐量比较">
  <figcaption>13 亿参数规模下的训练吞吐量比较。</figcaption>
</figure>

DeltaNet 的吞吐量很有竞争力，只比 GLA（Gated Linear Attention，门控线性注意力）略慢。考虑到 DeltaNet 的转移矩阵表达能力更强，这一点性能差距是合理的权衡。

[^recurrent-materialization]: 关于循环计算如何避免中间状态物化，可参见 Katharopoulos 等人的线性注意力论文第 3.3.1 节。

---

## 第三篇：神经架构 {#part-three}

### DeltaNet 架构设计

<figure>
  <img src="/assets/images/posts/deltanet-explained/delta-net-arch.png" alt="现代 DeltaNet 神经架构">
  <figcaption>DeltaNet 架构。</figcaption>
</figure>

在最后一篇中，我们将介绍如何对 DeltaNet 架构进行现代化改造。在保留核心 Delta Rule 机制的同时，我们引入了几项架构改进，显著提升了模型性能。

从宏观上看，DeltaNet 遵循由 Llama 普及的现代 Transformer block 设计，让 token mixing 与 channel mixing 交替出现：DeltaNet 取代 self-attention 负责 token mixing，SwiGLU 则负责 channel mixing。我们的主要修改集中在 token mixing 层，共有三项：

1. 在 query 和 key 的处理中，以 $L_2$ 归一化和 SiLU 激活替换原始的 $L_1$ 归一化与 $1+\mathrm{ELU}$ 激活；
2. 在 query、key 和 value 的线性投影之后加入短卷积；
3. 在最终投影之前加入输出归一化。

完整处理流水线如下：

- **Query / Key：**Linear → ShortConv → SiLU → $L_2$Norm；
- **Value：**Linear → ShortConv → SiLU；
- **Beta：**Linear → Sigmoid；
- **Output：**Delta Rule(query, key, value, beta) → RMSNorm → Linear。

下面逐项说明这些修改为何对模型性能至关重要。

#### Query 和 Key 的归一化

key 向量归一化是 DeltaNet 架构的关键环节。这不仅是一个技术细节，而是模型稳定性和有效性的基础。考虑 DeltaNet 的核心方程：

$$
\mathbf{S}_{t} = \mathbf{S}_{t-1}
(\mathbf{I} - \beta_t \mathbf{k}_t\mathbf{k}_t^\top)
+ \mathbf{v}_t\mathbf{k}_t^\top.
$$

这个循环系统的稳定性取决于转移矩阵 $(\mathbf{I}-\beta_t\mathbf{k}_t\mathbf{k}_t^\top)$ 的特征值。该矩阵有一个很优美的谱结构：

- 沿 $\mathbf{k}_t$ 方向的特征值为 $1-\beta_t\|\mathbf{k}_t\|^2$；
- 与 $\mathbf{k}_t$ 正交的所有方向，特征值均为 1。

为了实现稳定更新，需要所有特征值的绝对值都不超过 1。给定 $0\leq\beta_t\leq1$，就要求 $\|\mathbf{k}_t\|^2\leq2$。原始 DeltaNet 使用 $L_1$ 归一化；我们发现 $L_2$ 归一化不仅实验表现更好，而且具有更直观的几何解释：当 $\beta_t=1$ 且 $\|\mathbf{k}_t\|_2=1$ 时，矩阵 $\mathbf{I}-\mathbf{k}_t\mathbf{k}_t^\top$ 会成为一个投影矩阵。它会选择性擦除 $\mathbf{k}_t$ 方向上的信息，同时保留其他全部方向。

<figure class="figure-medium">
  <img src="/assets/images/posts/deltanet-explained/projection.png" alt="投影矩阵擦除平行分量、保留正交分量的示意图">
  <figcaption>投影矩阵的几何作用。</figcaption>
</figure>

投影矩阵具有一个重要的几何效果：作用到任意向量上时，它会移除与 $\mathbf{k}_t$ 平行的分量，并保留所有正交分量。在 DeltaNet 中，这意味着每次更新都会“清理”状态，移除可能干扰当前 key 方向的分量。随着时间推进，这一操作有助于让不同 key 向量保持更清晰的分离，从而减少第一篇讨论的存储模式之间的干扰（即检索误差）。这一几何性质也解释了为什么直接符合投影解释的 $L_2$ 归一化，会比 $L_1$ 归一化带来更好的检索性能。

我们还发现，对 query 应用 $L_2$ 归一化也能提高模型性能。这与近期 self-attention 架构的发展趋势一致：QK-normalization 已经成为稳定并增强注意力机制的有效技术。

最后，当前设计可能存在一个限制：转移矩阵的特征值被约束为严格正值。近期一项很有启发性的工作说明，这可能限制模型的状态跟踪能力。幸运的是，论文提出的改进非常简单——把 beta 项改成 $\beta_t=2\beta_t$，转移矩阵就可以拥有负特征值。仅仅一行修改，便可能显著扩展 DeltaNet 的表示能力。感兴趣的读者可以阅读[这段进一步讨论](https://x.com/riccardograzzi/status/1860017064473428220)。

#### 输出归一化

在标准线性注意力中，每个位置的输出会除以注意力权重之和：

$$
\mathbf{o}_t =
\frac{\left(\sum_{i=1}^t \mathbf{v}_i \phi(\mathbf{k})_i^\top\right)
\phi(\mathbf{q})_t}
{\sum_{i=1}^t \phi(\mathbf{k})_i^\top\phi(\mathbf{q})_t},
$$

其中 $\phi$ 是正值特征映射。然而，Qin 等人的一项重要分析表明，这个归一化项可能导致无界梯度和训练不稳定。为了解决这一问题，他们建议移除分母，改为在最终投影之前对输出应用归一化。这项架构修改此后逐渐成为标准做法，RetNet、GLA、Mamba 2 等现代线性注意力模型都采用了它。

#### 激活函数的选择

<figure class="figure-narrow">
  <img src="/assets/images/posts/deltanet-explained/silu.png" alt="SiLU 激活函数曲线">
  <figcaption>SiLU 激活函数。</figcaption>
</figure>

原始 DeltaNet 使用 $1+\mathrm{ELU}$ 激活，而我们的实验显示 SiLU 能获得更好的性能；这与 Mamba 2、xLSTM 和 Lightning Attention 等近期架构的选择一致。传统线性注意力模型通常会选择能够通过正值特征映射保证注意力分数为正的激活函数，例如 ReLU、$1+\mathrm{ELU}$ 或指数函数；允许负值取得成功，与 Differential Transformer 的发现相呼应，也说明把注意力分数严格限制为正值可能没有必要，甚至会限制模型。

#### 短卷积

短卷积通常是 kernel window 小到 4 的 depthwise separable Conv1D。它已经成为近期次二次注意力模型中的关键组件，并以不同形式出现在 Mamba、xLSTM、MetaLA 等架构中。它可以看作 H3 早先提出的 “shift-SSM” 的推广，同时与 RWKV 中的 “token-shift” 密切相关。

至于短卷积为什么有效，我们认为原因是：它提供了一条“捷径”，让单层网络就能形成 induction head。这对上下文学习有帮助；大规模实验也发现，即便在 softmax 注意力中，短卷积依然有用。

#### 实验结果

有了并行算法和上述架构，我们终于可以把 DeltaNet 扩大到标准语言建模场景。评测覆盖三个主要指标：

- **语言建模：**WikiText 困惑度；
- **常识推理：**LAMBADA、PiQA、HellaSwag、WinoGrande、ARC-easy 和 ARC-challenge 的平均结果；
- **上下文检索：**FDA、SWDE 和 SQuAD 的平均结果。

下面比较不同架构的状态大小，其中 $H$ 表示层数，$d$ 表示模型维度：

| 架构 | 状态扩展倍数 | 总状态大小 | 实现细节 |
| --- | ---: | ---: | --- |
| **Mamba** | 16× | $64Hd$ | 把 value 投影扩展到 $2d$，使用 16× 扩展率；以 Mamba 层替换 FFN，使有效状态大小加倍 |
| **RetNet** | 512× | $512Hd$ | 把 value 投影扩展到 $2d$；query/key 头维度固定为 256 |
| **GLA** | 256× | $256Hd$ | query/key 头大小是 value 头的一半；每层维持 $4d^2$ 个参数 |
| **DeltaNet** | 128× | $128Hd$ | 整个架构始终使用 128 维注意力头 |

##### 主要结果（3.4 亿参数，150 亿 token）

| 模型 | Wiki 困惑度 ↓ | 常识平均分 ↑ | 检索平均分 ↑ | 状态大小 |
| --- | ---: | ---: | ---: | ---: |
| Transformer++ | 28.39 | 41.2 | 28.6 | N/A |
| RetNet（无卷积） | 32.33 | 41.0 | 14.6 | 512× |
| Mamba（有卷积） | 28.39 | 41.8 | 12.5 | 64× |
| GLA（无卷积） | 28.65 | 41.5 | 18.0 | 128× |
| DeltaNet（有卷积） | 28.24 | 42.1 | 22.7 | 128× |

DeltaNet 在保持合理状态大小的同时，在所有指标上都获得了有竞争力的表现。尤其值得注意的是，它在检索任务上表现强劲，支持了我们的假设：Delta Rule 机制能够提供有效的上下文检索能力。

##### 消融实验（3.4 亿参数，150 亿 token）

| 模型 | Wiki 困惑度 ↓ | 常识平均分 ↑ | 检索平均分 ↑ |
| --- | ---: | ---: | ---: |
| DeltaNet（完整模型） | 28.24 | 42.1 | 22.7 |
| 移除短卷积 | 29.08 | 41.4 | 18.6 |
| 使用 $L_1$-norm + $1+\mathrm{ELU}$ | 31.12 | 40.1 | 11.5 |
| 使用 $L_2$-norm + $1+\mathrm{ELU}$ | 28.03 | 42.1 | 21.8 |
| 使用 $L_2$-norm + ReLU | 28.75 | 40.9 | 21.0 |

消融实验揭示了几项重要发现。最突出的是，检索性能对归一化方式非常敏感：$L_2$ 归一化显著优于 $L_1$ 归一化，支持了前面关于投影性质的理论分析。短卷积同样是关键组件，这说明有效的基于位置的寻址可以有意义地补充 DeltaNet 基于内容的检索机制。激活函数的选择也会产生影响，但相对较小：SiLU 比 ReLU 和 $1+\mathrm{ELU}$ 有所提升，不过其作用不及归一化方式和短卷积明显。

### 混合模型：把 DeltaNet 与注意力结合起来

<figure>
  <img src="/assets/images/posts/deltanet-explained/hybrid.png" alt="DeltaNet 与滑动窗口注意力或全局注意力组成的混合架构">
  <figcaption>左：滑动窗口注意力与 DeltaNet 的混合模型。右：全局注意力与 DeltaNet 的混合模型。</figcaption>
</figure>

尽管 DeltaNet 的 Delta Rule 机制在检索任务上展现出潜力，它仍面对所有 RNN 架构共有的根本限制：状态大小固定。无论选择哪一种更新规则，这项约束都会为检索性能设置一个内在上限。为了突破这一限制，我们探索了有策略地组合 DeltaNet 与注意力机制的混合架构。

第一种方法参考 Griffin 和 Samba 等近期架构，以交错方式组合滑动窗口注意力和 DeltaNet。由于窗口大小固定，这种混合架构仍然保持次二次复杂度，但它也继承了纯 RNN 模型的类似理论限制。正如 Griffin 所说明的，固定上下文窗口会限制模型检索窗口范围之外的信息。

因此，我们转向第二种方法：用全局注意力增强 DeltaNet。如果用注意力替换大量 DeltaNet 层，推理效率会受到明显影响；所以我们参考 H3，只放置两个全局注意力层：一个位于第 2 层，另一个位于第 $N/2-1$ 层。严格来说，这使模型不再具有次二次复杂度；但由于注意力层用得很少，相比完整 Transformer，KV Cache 需求仍然大幅降低。

3.4 亿参数规模的结果证明了这些混合方法的有效性：

| 模型 | Wiki 困惑度 ↓ | 常识平均分 ↑ | 检索平均分 ↑ |
| --- | ---: | ---: | ---: |
| Transformer++ | 28.39 | 41.2 | 28.6 |
| DeltaNet | 28.24 | 42.1 | 22.7 |
| + 滑动窗口注意力 | 27.06 | 42.1 | 30.2 |
| + 全局注意力 | 27.51 | 42.1 | 32.7 |

随后，我们把实验扩大到 13 亿参数，并在 SlimPajama 上训练 1000 亿 token。结果进一步支持了前面的发现：

| 模型 | Wiki 困惑度 ↓ | 常识平均分 ↑ | 检索平均分 ↑ |
| --- | ---: | ---: | ---: |
| Transformer++ | 16.85 | 50.9 | 41.8 |
| DeltaNet | 16.87 | 51.6 | 34.7 |
| + 滑动窗口注意力 | 16.56 | 52.1 | 39.6 |
| + 全局注意力 | 16.55 | 51.8 | 47.9 |

滑动窗口注意力带来了显著提升，但在更大规模下仍无法完全达到 Transformer 的检索性能。相比之下，只增加两个全局注意力层就取得了惊人的结果，检索表现甚至超过了 Transformer 基线[^sparse-global-attention]。

最后，我们按照 PowerLM-3B 的设置，评测了一个使用 1 万亿 token 训练的 30 亿参数模型。这些结果表明，DeltaNet 在 RNN 架构中表现强劲，但与基于 Transformer 的模型相比仍略有差距：

| 模型 | ARC | HellaSwag | OBQA | PIQA | WinoGrande | MMLU | 平均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-3B | 59.1 | 73.6 | 43.4 | 77.5 | 69.2 | 54.1 | 62.8 |
| PowerLM-3B | 60.5 | 74.6 | 43.6 | 79.9 | 70.0 | 45.0 | 62.3 |
| DeltaNet-3B | 60.4 | 72.8 | 41.0 | 78.5 | 65.7 | 40.7 | 59.8 |
| RecurrentGemma-2B | 57.0 | 71.1 | 42.0 | 78.2 | 67.6 | 31.8 | 57.9 |
| RWKV-6-3B | 49.5 | 68.6 | 40.6 | 76.8 | 65.4 | 28.4 | 54.9 |
| Mamba-2.7B | 50.3 | 65.3 | 39.4 | 75.8 | 63.1 | 26.1 | 53.3 |

结果说明，DeltaNet 在不同规模下都很有效，但扩展到更大规模后，与 Transformer 架构之间仍然存在小幅差距。我们正在探索把 DeltaNet 与注意力机制结合起来的更大规模混合模型，敬请期待后续进展。

[^sparse-global-attention]: 近期工作表明，只在全部层中的一小部分（约 10%）使用全局注意力，也可以显著改善模型性能。

---

## 参考文献

以下条目完整汇总了三篇原文在正文中引用的工作；论文标题沿用正式英文名称。

<details>
<summary>展开查看三篇原文引用的 39 项参考文献</summary>

1. Blelloch, Guy E. (1990). *Prefix Sums and Their Applications*.
2. [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) (2024).
3. [An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887) (2024).
4. [Transformer Quality in Linear Time](https://proceedings.mlr.press/v162/hua22a.html) (2022).
5. Gu, Albert; Dao, Tri. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (2023).
6. [Hierarchically Gated Recurrent Neural Network for Sequence Modeling](https://papers.nips.cc/paper_files/paper/2023/hash/694be3548697e9cc8999d45e8d16fe1e-Abstract-Conference.html) (2023).
7. Joffrain, Thierry et al. [Accumulating Householder Transformations, Revisited](https://api.semanticscholar.org/CorpusID:15723171) (2006).
8. Poli, Michael et al. [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://api.semanticscholar.org/CorpusID:257050308) (2023).
9. Qin, Zhen et al. [Various Lengths, Constant Speed: Efficient Language Modeling with Lightning Attention](https://api.semanticscholar.org/CorpusID:270063820) (2024).
10. Shen, Yikang et al. [Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler](https://arxiv.org/abs/2408.13359) (2024).
11. Xu, Mingyu et al. [KV Shifting Attention Enhances Language Modeling](https://api.semanticscholar.org/CorpusID:274422840) (2024).
12. Arora, Simran et al. [Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff](https://arxiv.org/abs/2402.18668) (2024).
13. Beck, Maximilian et al. [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517) (2024).
14. Bischof, Christian H.; Van Loan, Charles. [The WY Representation for Products of Householder Matrices](https://api.semanticscholar.org/CorpusID:36094006) (1985).
15. Chou, Yuhong et al. [MetaLA: Unified Optimal Linear Approximation to Softmax Attention Map](https://openreview.net/forum?id=Y8YVCOMEpz) (2024).
16. De, Soham et al. [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427) (2024).
17. Dao, Tri. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023).
18. Grazzi, Riccardo et al. [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537) (2024).
19. [Hungry Hungry Hippos: Towards Language Modeling with State Space Models](https://openreview.net/forum?id=COZDy0WYGg) (2023).
20. Jelassi, Samy et al. [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/abs/2402.01032) (2024).
21. Katharopoulos, Angelos et al. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html) (2020).
22. Dao, Tri; Gu, Albert. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) (2024).
23. Mao, Huanru Henry. [Fine-Tuning Pre-trained Transformers into Decaying Fast Weights](https://aclanthology.org/2022.emnlp-main.697) (2022).
24. Olsson, Catherine et al. [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (2022).
25. [RWKV: Reinventing RNNs for the Transformer Era](https://aclanthology.org/2023.findings-emnlp.936) (2023).
26. Poli, Michael et al. [Mechanistic Design and Scaling of Hybrid Architectures](https://arxiv.org/abs/2403.17844) (2024).
27. [The Devil in Linear Transformer](https://aclanthology.org/2022.emnlp-main.473) (2022).
28. Qin, Zhen et al. [HGRN2: Gated Linear RNNs with State Expansion](https://api.semanticscholar.org/CorpusID:269043328) (2024).
29. Ren, Liliang et al. [Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522) (2024).
30. [Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html) (2021).
31. Sun, Yu et al. [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620) (2024).
32. Sun, Yutao et al. [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) (2023).
33. [The Unreasonable Effectiveness of the Forget Gate](https://arxiv.org/abs/1804.04849) (2018).
34. Wen, Kaiyue et al. [RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval](https://arxiv.org/abs/2402.18510) (2024).
35. Widrow, Bernard; Hoff, Marcian E. *Adaptive Switching Circuits* (1960).
36. Yang, Songlin; Zhang, Yu. [FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism](https://github.com/sustcsonglin/flash-linear-attention) (2024).
37. Yang, Songlin et al. [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) (2023).
38. Ye, Tianzhu et al. [Differential Transformer](https://arxiv.org/abs/2410.05258) (2024).
39. [Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927) (2023).

</details>

---

## 原文与相关资源

- Songlin Yang, [DeltaNet Explained (Part I): The Model](https://sustcsonglin.github.io/blog/2024/deltanet-1/)
- Songlin Yang, [DeltaNet Explained (Part II): The Algorithm](https://sustcsonglin.github.io/blog/2024/deltanet-2/)
- Songlin Yang, [DeltaNet Explained (Part III): The Neural Architecture](https://sustcsonglin.github.io/blog/2024/deltanet-3/)
- 论文：[Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)
- 实现：[flash-linear-attention / delta_net.py](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/delta_net.py)
- 演示：[Efficient Architectures for Long Sequence Modeling](https://people.csail.mit.edu/yoonkim/data/efficient_architectures_talk.pdf)
- 原系列参考文献：[2024-12-03-delta.bib](https://github.com/sustcsonglin/sustcsonglin.github.io/blob/master/assets/bibliography/2024-12-03-delta.bib)
