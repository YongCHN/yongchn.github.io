---
layout: post
title: "大语言模型强化学习全景：从 PPO、GRPO 到 Agentic RL 基础设施与收敛"
description: "一份面向研究与工程实践的 LLM 强化学习技术综述：统一解释 PPO/GRPO、主流训练框架、Agentic RL、Sandbox、全异步训练、R3、Partial Rollout，以及训练收敛的进展与挑战。"
date: 2026-09-15 10:00:00 +0800
categories: [人工智能, 大语言模型]
tags: [强化学习, RLHF, RLVR, PPO, GRPO, Agentic RL, RL Infra, 大语言模型]
author: Yong
translation: false
math: true
---

过去几年，大语言模型（Large Language Model，LLM）的强化学习经历了两次明显的范式跃迁。第一次是从“模仿高质量文本”转向“根据人类或 AI 的偏好优化行为”，以 InstructGPT 式 RLHF 为代表；第二次是从“优化一轮回答”转向“让模型在可验证环境中探索、推理和行动”，以 RLVR、长链推理与 Agentic RL 为代表。

这两次跃迁改变的不只是算法。训练对象从一个生成器变成了策略（policy），数据从静态语料变成了策略在线产生的轨迹（trajectory），训练系统也从单一的前向—反向流水线演进为由推理引擎、训练引擎、奖励服务、环境与沙箱共同组成的分布式闭环。

本文基于截至 **2026 年 9 月 15 日**公开的论文、技术报告、官方文档与开源实现，试图回答五个问题：

1. 什么是强化学习，LLM 强化学习与传统强化学习究竟哪里相同、哪里不同？
2. PPO、GRPO 的目标函数、工程代价与适用边界是什么？
3. 当前 LLM 强化学习算法与开源框架发展到了哪里？
4. Agentic RL 为什么要求新的 Sandbox、全异步、R3 与 Partial Rollout 基础设施？
5. 我们所说的“RL 收敛”到底是什么，当前取得了什么进展，又存在哪些未解决问题？

> **资料边界。** 闭源模型通常只公开原则、评测和有限训练信息，不能把“使用了大规模 RL”外推为某个具体算法或系统实现。本文对 OpenAI、Anthropic、Google 等模型只复述其公开披露；对算法细节的讨论主要依据论文和开源系统。

## 阅读导航

1. [从传统强化学习到大语言模型强化学习](#foundations)
2. [PPO、GRPO 与推理 RL 的算法谱系](#algorithms)
3. [模型训练范式与 RL 框架的进展](#frameworks)
4. [Agentic RL 与 RL Infra 的架构演进](#agentic-rl)
5. [RL 训练收敛：进展、判据与挑战](#convergence)
6. [结语](#conclusion)

## 核心结论

- LLM RL 没有脱离传统 RL：它仍在优化策略的期望累计回报，仍面对探索、信用分配、方差、分布漂移和奖励投机。
- 单轮 RLHF/RLVR 可以近似看成 contextual bandit，也可以在 token 粒度写成长时序 MDP；真正的 Agentic RL 则更接近长时程、部分可观测的 POMDP。
- PPO 的 critic 提供低方差优势估计，但要额外维护 value model；GRPO 用同一问题的组内相对奖励替代 critic，节省显存，却把代价转移到了多样本 rollout、组内方差和采样策略上。
- RL 系统的主瓶颈已经从“能否容纳四个模型”转为“长尾 rollout、权重同步、环境吞吐、策略陈旧度，以及训练—推理数值一致性”。
- Fully async、Partial Rollout 和 R3 都不是纯吞吐优化：它们会改变样本来自哪个策略、哪些 token 参与 loss，以及训练侧究竟在优化哪一个行为策略，因此也是算法正确性问题。
- “reward 收敛”不等于“能力收敛”。可靠判断至少要同时观察可验证成功率、策略熵、KL、pass@k 边界、训练—推理 log-prob 差异、长度分布、分难度泛化和安全回归。

---

## 一、从传统强化学习到大语言模型强化学习 {#foundations}

### 1.1 强化学习是什么

强化学习研究的是：一个智能体如何通过与环境交互，学习一套能够最大化长期回报的决策规则。经典形式是马尔可夫决策过程（Markov Decision Process，MDP）：

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\rho_0,\gamma),
$$

其中 $$\mathcal{S}$$ 是状态空间，$$\mathcal{A}$$ 是动作空间，$$P(s'\mid s,a)$$ 是状态转移，$$r(s,a)$$ 是奖励，$$\rho_0$$ 是初始状态分布，$$\gamma$$ 是折扣因子。策略 $$\pi_\theta(a\mid s)$$ 给出在状态 $$s$$ 下选择动作 $$a$$ 的概率，目标是：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}
\left[\sum_{t=0}^{T}\gamma^t r_t\right].
$$

与监督学习不同，RL 的训练目标通常没有逐样本的“标准动作”。智能体只能尝试动作、获得可能延迟且有噪声的奖励，再推断哪些决策值得增加概率。由此产生五个经典难题：探索与利用、长程信用分配、高方差、非平稳数据分布，以及错误奖励带来的策略投机。标准理论可参考 Sutton 与 Barto 的[《Reinforcement Learning: An Introduction》](http://incompleteideas.net/book/the-book-2nd.html)。

### 1.2 LLM 如何成为策略

给定提示 $$x$$，自回归模型生成序列 $$y=(y_1,\ldots,y_T)$$：

$$
\pi_\theta(y\mid x)=\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t}).
$$

因此可以建立三种互补视角：

| 视角 | 状态 | 动作 | 环境转移 | 奖励 |
|---|---|---|---|---|
| 单轮回答 / contextual bandit | 用户提示 $$x$$ | 整段回答 $$y$$ | 通常不显式建模 | 对整段回答打分 |
| token 级 MDP | 提示与已生成 token | 下一个 token | 把 token 追加到上下文 | 多数在结尾给出，KL 可逐 token 给出 |
| Agentic POMDP | 对话历史、工具观测、压缩记忆 | 文本、工具调用或环境动作 | 外部系统返回新观测 | 过程奖励、终局成功或成本 |

第一种视角解释了为什么许多 RLHF/RLVR 算法可以把整段 response 当作一个样本；第二种视角解释了 PPO ratio、KL 和 loss mask 为什么按 token 计算；第三种视角才覆盖浏览网页、操作终端、修改代码、调用 API 等长时程任务。

### 1.3 奖励来自哪里

LLM RL 的几种常见反馈不是不同的优化理论，而是不同的奖励来源：

- **RLHF**：收集人类偏好，训练 reward model（RM），再用 RM 分数优化策略。[InstructGPT](https://arxiv.org/abs/2203.02155)展示了经典的 SFT → RM → PPO 三阶段流程。
- **RLAIF**：偏好标签或批评由另一个模型产生。Anthropic 的[Constitutional AI](https://arxiv.org/abs/2212.08073)用一组原则生成自我批评与 AI 偏好；Google 的[RLAIF 研究](https://arxiv.org/abs/2309.00267)系统比较了 AI 与人类反馈。
- **RLVR**：奖励可被程序客观验证，例如数学答案、代码单元测试、形式化证明、棋局结果或环境任务完成度。它减少了 learned reward model 的主观误差，但没有自动解决奖励漏洞与分布外泛化。
- **混合奖励**：把正确性、格式、长度、工具成本、安全约束、风格偏好和 learned judge 组合起来。真实系统通常属于这一类。

### 1.4 相同的数学内核，不同的工程尺度

| 维度 | 传统深度 RL | LLM 强化学习 |
|---|---|---|
| 优化目标 | 最大化期望累计回报 | 相同；常再加参考策略 KL 与通用能力约束 |
| 策略 | 中小型神经网络 | 数十亿至数千亿参数的自回归 Transformer/MoE |
| 动作空间 | 离散动作或低维连续控制 | 每一步从大词表采样；工具调用还是结构化复合动作 |
| 轨迹长度 | 数十至数千环境步 | 单轮可有数万 token；Agent 可有数百轮交互 |
| 奖励 | 环境逐步或终局反馈 | 常为稀疏终局奖励、RM/critic 分数或 verifier 结果 |
| 初始策略 | 常从随机或较弱策略起步 | 从强预训练/SFT 模型起步，先验极强 |
| 数据 | 模拟器交互经常较便宜 | rollout、长上下文推理、工具与沙箱执行都很昂贵 |
| 正则化 | 熵、trust region 等 | 对 reference model 的 KL 尤其关键，以防语言能力漂移 |
| 部署一致性 | 训练与执行通常共用代码路径 | rollout 常用 vLLM/SGLang，训练用 FSDP/Megatron，容易数值不一致 |
| 主要风险 | 样本低效、探索失败、策略崩溃 | 再叠加 reward hacking、语言退化、长度偏置、安全回归和基础设施静默错误 |

最重要的差异是：LLM 在 RL 前已经从海量文本中获得了丰富行为分布。RL 往往不是从零发现解法，而是在原分布中重新分配概率质量，使正确、有用或可验证的轨迹更容易被一次采样命中。这既是 LLM RL 高效的原因，也是其能力上界争议的来源。

---

## 二、PPO、GRPO 与推理 RL 的算法谱系 {#algorithms}

### 2.1 从策略梯度开始

REINFORCE 使用 log-derivative trick 得到：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{\tau\sim\pi_\theta}
\left[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A_t\right],
$$

其中 $$A_t$$ 是优势函数：当前动作相对于某个 baseline 好多少。减去不依赖当前动作的 baseline 不改变梯度期望，却能显著降低方差。actor-critic 方法用一个 value model $$V_\phi(s_t)$$ 学习这个 baseline。

### 2.2 PPO：用受限更新换稳定性

[Proximal Policy Optimization（PPO）](https://arxiv.org/abs/1707.06347)的核心不是“直接最大化奖励”，而是限制新策略不要一步偏离采样数据对应的旧策略太远。定义 token 级概率比：

$$
r_t(\theta)=
\frac{\pi_\theta(y_t\mid x,y_{<t})}
{\pi_{\theta_{\mathrm{old}}}(y_t\mid x,y_{<t})},
$$

PPO-Clip 的代理目标是：

$$
L^{\mathrm{clip}}(\theta)=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

当优势为正时，PPO 防止新策略无限放大该动作概率；当优势为负时，防止概率下降过猛。它是 trust-region 思想的工程化近似，而不是严格保证每一步单调改进。

在 actor-critic PPO 中，常用 GAE 估计优势：

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t),
\qquad
\hat A_t=\sum_{l\ge 0}(\gamma\lambda)^l\delta_{t+l}.
$$

经典 LLM RLHF 还会加入相对于参考模型 $$\pi_{\mathrm{ref}}$$ 的 KL 惩罚：

$$
R(x,y)=r_{\mathrm{RM}}(x,y)
-\beta\sum_t \log
\frac{\pi_\theta(y_t\mid x,y_{<t})}
{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}.
$$

这使完整 PPO 系统通常同时容纳四种角色：actor、critic、reference model 和 reward model。模型可以共享初始化或部分参数，但前向、显存与调度成本仍然可观。[InstructGPT 附录](https://arxiv.org/pdf/2203.02155)也说明，稳定训练往往还需要预训练梯度混合、KL 控制、reward/advantage whitening、梯度裁剪和严格的 batch 语义。

PPO 的优点是优势估计细、适合逐步奖励和真正的长时序任务；缺点是系统重、超参数耦合多，critic 对超长 CoT 又容易产生 value bias。

### 2.3 GRPO：用组内比较替代 critic

[DeepSeekMath](https://arxiv.org/abs/2402.03300)提出 Group Relative Policy Optimization（GRPO）。对同一提示 $$q$$，旧策略采样 $$G$$ 个回答 $$\{o_1,\ldots,o_G\}$$，得到奖励 $$R_i$$，再做组内标准化：

$$
\hat A_i=
\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\varepsilon}.
$$

同一回答里的 token 通常共享序列级优势，再套用 PPO-Clip：

$$
L_{\mathrm{GRPO}}=
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\min\left(r_{i,t}\hat A_i,
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_i\right)
-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
\right].
$$

直觉上，同一道题天然形成一个小型竞赛：高于组内平均的答案被提高概率，低于平均的答案被压低。这样不再训练与 actor 同规模的 critic，显著节省显存和计算，因此特别适合数学、代码等有终局 verifier 的场景。

但“去掉 critic”不等于“免费”：

- 每个 prompt 必须生成多个回答，rollout 开销随组大小增长；
- 如果一组全对或全错，奖励方差接近零，几乎没有学习信号；
- 相对标准化使优势受同组样本和题目难度影响；
- 把序列奖励广播到每个 token，无法知道究竟是哪一步推理起作用；
- token 平均与标准差归一化会引入长度和难度偏置；
- 稀疏奖励下，GRPO 仍依赖 base/SFT 模型先产生至少少量成功轨迹。

### 2.4 PPO 与 GRPO：如何选择

| 问题 | PPO | GRPO |
|---|---|---|
| baseline | learned critic / value model | 同一 prompt 的组内奖励 |
| 额外模型成本 | 高 | 低，不需要 critic |
| rollout 成本 | 可每题较少样本 | 需要 group sampling |
| 逐步奖励与长程信用 | 更自然，可用 GAE | 默认把结果奖励广播给所有 token |
| 奖励尺度 | critic 可学习跨题价值 | 组内归一化天然消除部分尺度差异 |
| 零方差组 | 不必然失效 | 全对/全错时基本无梯度，需要动态采样 |
| 实现复杂度 | 高，超参数多 | 表面较简单，但 sampling 与 loss aggregation 很关键 |
| 典型场景 | 通用 RLHF、过程奖励、Agentic RL | 数学/代码 RLVR、长 CoT、critic 代价过高时 |

这不是绝对二选一。Agentic RL 可以使用 trajectory return 加 GRPO 式组内 baseline；推理 RL 也可以回到 value-based PPO。关键在于奖励密度、轨迹长度、每题可承受的采样数，以及是否有能力训练可信的 critic。

### 2.5 GRPO 之后：修正偏置、提高探索与适配 MoE

2025 年后出现的算法大多不是推翻策略梯度，而是在四个位置做文章：优势估计、ratio 粒度、clip 方式和采样策略。

- **RLOO / REINFORCE++**：用 leave-one-out 或 batch/group baseline 降低 REINFORCE 方差，保留 critic-free 的系统优势。[Back to Basics](https://arxiv.org/abs/2402.14740)和[REINFORCE++](https://arxiv.org/abs/2501.03262)是代表工作。
- **Dr. GRPO**：[《Understanding R1-Zero-Like Training》](https://arxiv.org/abs/2503.20783)指出，按回答长度平均会让长回答中的每个 token 承受不同隐式权重，组内标准差归一化也会产生题目难度偏置；其简化版本移除这些归一化，并提醒“回答越来越长”不必然等于推理能力涌现。
- **DAPO**：[DAPO](https://arxiv.org/abs/2503.14476)组合 asymmetric/decoupled clipping、动态采样、token-level policy-gradient loss 和 overlong reward shaping，在开放数据与 veRL 代码上给出了可复现的大规模训练配方。
- **VAPO**：[VAPO](https://arxiv.org/abs/2504.05118)没有放弃 value model，而是针对长 CoT 的 value bias、异质长度和稀疏奖励重新设计 value-based PPO，说明 critic 在信用分配充分复杂时仍有价值。
- **GSPO**：[Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)把重要性 ratio、clip 与优化粒度提升到 sequence 级。Qwen 团队报告它能缓解 token ratio 在长序列和 MoE 上的高方差，并服务于后续 Qwen3 训练。
- **CISPO**：[MiniMax-M1](https://arxiv.org/abs/2506.13585)提出 clip importance-sampling weight 而不是直接裁掉 token 更新，目标是保留有效梯度并提高 RL 效率。

这些名字容易制造一种错觉，好像算法胜负已经确定。实际上，各报告使用的 base model、数据、reward、采样预算、最大长度与基础设施都不同；脱离完整 recipe 比较单个 objective，结论通常并不可靠。

### 2.6 DPO 算不算强化学习

[Direct Preference Optimization（DPO）](https://arxiv.org/abs/2305.18290)从 KL 正则化的 RLHF 最优解出发，把隐式 reward 写成 policy 与 reference policy 的 log-ratio，最终直接在离线偏好对上做分类式优化。它与 RLHF 有共同的理论出发点，却通常没有在线环境交互、policy rollout 和显式 reward/value model。

因此更准确的说法是：DPO 属于 preference optimization，也是广义 alignment 技术；但在讨论本文的在线 RL infra 时，不应把一次离线 DPO 训练与 PPO/GRPO 的在线策略优化混为一谈。实践中两者经常串联：DPO/SFT 先提供稳定初始化，在线 RL 再针对当前策略产生的新轨迹继续改进。

---

## 三、模型训练范式与 RL 框架的进展 {#frameworks}

### 3.1 从偏好对齐到可验证推理与 Agent 能力

公开报告展现出一条清晰但并非线性的路线：

| 阶段 / 代表 | 公开的 RL 重点 | 应当如何解读 |
|---|---|---|
| [InstructGPT](https://arxiv.org/abs/2203.02155) | 人类偏好 RM + PPO + reference KL | 奠定通用助手 RLHF 标准流程 |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | AI feedback、原则驱动的偏好模型与 RL | 把监督扩展到可规模化的 AI 反馈 |
| [Llama 3](https://arxiv.org/abs/2407.21783) | SFT、rejection sampling、PPO 与 DPO 多轮迭代 | 说明工业 post-training 通常是混合流程 |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | 数学 verifier 与 GRPO | critic-free group-relative RL 成为重要分支 |
| [OpenAI o1](https://openai.com/index/learning-to-reason-with-llms/) | 大规模 RL 教会模型使用长 CoT，性能随训练和测试时计算增长 | 公开了 scaling 现象，未披露具体算法与基础设施 |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | R1-Zero 直接从 base 做 RL；R1 加 cold start 与多阶段 RL/SFT | 证明纯 RL 可诱导可见推理行为，也暴露可读性和语言混杂问题 |
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | 128K 长上下文 RL、online mirror descent、partial rollout、多模态 | 把长 CoT 与系统优化共同视为 RL scaling 轴 |
| [Seed1.5-Thinking](https://arxiv.org/abs/2504.13914) | 大规模 MoE reasoning RL、verifier 与训练 recipe | 展示模型、数据、reward、系统协同的重要性 |
| [Qwen3](https://arxiv.org/abs/2505.09388) / [GSPO](https://arxiv.org/abs/2507.18071) | thinking/non-thinking 融合、多阶段 RL、sequence-level policy optimization | 从“让模型会想”进入稳定扩展与模式控制 |
| [Magistral](https://arxiv.org/abs/2506.10910) | 自有模型与 infra 上的 scalable RL，研究纯 RL 与 reasoning language | 表明文本 RL 可在保持其他能力的同时强化推理 |
| [MiniMax-M1](https://arxiv.org/abs/2506.13585) | CISPO、长上下文、SWE sandbox RL | 将可验证软件工程环境并入大规模 RL |
| [Gemini 2.5](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf) | 人类与 critic 反馈的 RL、推理、多模态和 agent 能力 | 奖励由 data RM 与 rubric critic 等信号组合 |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) | Agentic、Reasoning、Coding 统一 post-training，配套 slime | Agent 场景开始成为基础模型训练的一等目标 |
| [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780) | 多教师 on-policy distillation、MoE 稳定训练与 R3 等系统技术 | RL loop 也成为在线蒸馏与多教师能力融合的基础设施 |

两点尤其值得强调。第一，工业模型几乎不是“只做一次 GRPO”得到的，而是 SFT、拒绝采样、蒸馏、偏好优化、RLVR、安全 RL 和数据刷新组成的多阶段循环。第二，报告中的 benchmark 数字不能直接归因于某个算法；base model、数据清洗、verifier、采样预算和测试时策略经常贡献更大。

### 3.2 开源框架：从 Trainer 到分布式闭环

当前框架大致可分为三类：便于算法实验的 trainer、面向大模型的分布式 RL runtime，以及面向真实交互的环境/agent runtime。

| 框架 | 主要定位 | 训练 / rollout 路径 | 适合场景 |
|---|---|---|---|
| [TRL](https://huggingface.co/docs/trl/index) | Hugging Face 生态的易用 post-training trainer | Transformers/Accelerate，并可接 vLLM | 单机到中等规模实验、算法原型、模型生态兼容 |
| [DeepSpeed-Chat](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) | 早期端到端 RLHF pipeline 与 Hybrid Engine | DeepSpeed training/generation | 理解经典三阶段 RLHF 与 ZeRO 扩展 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Ray + DeepSpeed/vLLM 的分布式 RLHF/Agentic RL | 训练与 rollout 可 colocate 或 disaggregate | PPO/GRPO/REINFORCE 系列、多节点与异步训练 |
| [veRL / HybridFlow](https://github.com/verl-project/verl) | 灵活表示 RL dataflow 与角色映射 | FSDP/Megatron + vLLM/SGLang | 大规模 RLVR、复杂 placement、算法与系统共同开发 |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | NVIDIA 栈的可扩展 post-training | PyTorch/FSDP/Megatron + vLLM/SGLang | 大模型/MoE、多节点、长上下文和生产级 recipe |
| [ROLL](https://github.com/alibaba/ROLL) | 多角色、异构调度的规模化 RL/Agentic RL | Ray + Megatron + SGLang/vLLM | 多任务 RLVR、Agent、多模态与大集群 |
| [slime](https://github.com/THUDM/slime) | 轻量、深度绑定 Megatron + SGLang 的 RL scaling | 显式 data buffer 与 server-based rollout | 需要定制生成、reward、工具和 GLM/DeepSeek/Qwen 大模型路径 |
| [AReaL](https://github.com/inclusionAI/AReaL) | fully asynchronous language reasoning RL | 独立 rollout 与 learner workers | 研究陈旧度控制、异步吞吐和 off-policy correction |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | 模块化全栈与长时程 Agent RL | 训练、推理、agent 环境解耦 | 多轮工具使用、真实环境与 fully async 实验 |

表中能力以各项目当前公开文档为准，变化很快，不应把“README 中支持”视作在所有模型、并行策略与异步模式下都完成了等价验证。

[HybridFlow/veRL 论文](https://arxiv.org/abs/2409.19256)总结了一个关键抽象：LLM RL 本身是 dataflow，但每个节点不是一个普通算子，而是一套分布式大模型程序；节点之间也不是简单张量边，而是跨设备重分片与多对多传输。其 hybrid-controller 和 3D-HybridEngine 试图同时解决可编程性、资源映射与 actor 在训练/生成布局之间的切换。

[OpenRLHF](https://arxiv.org/abs/2405.11143)强调 Ray、vLLM 与 DeepSpeed 的角色级调度；[NeMo-Aligner](https://arxiv.org/abs/2405.01481)及后续 NeMo RL 强调 Megatron 规模化；[RLHFuse](https://arxiv.org/abs/2409.13221)则进一步把 generation 和 training 拆成细粒度子任务，通过跨阶段与阶段内融合填补 GPU bubble。这些系统工作共同说明：训练吞吐不只是 kernel 快慢，而取决于整个闭环最慢、最不均匀的阶段。

### 3.3 为什么 rollout 成为主瓶颈

预训练每个 token 只需 teacher-forced 前向和反向；RL rollout 必须自回归生成，后一个 token 等待前一个 token，而且输出长度高度不均匀。推理 RL 还要求同一 prompt 采样多条长 CoT，Agentic RL 又要等待网页、测试、编译、API 或沙箱。因此常见问题包括：

- 少数超长轨迹拖住整个同步 batch；
- rollout GPU 等 learner，learner GPU 又等 rollout，形成 phase bubble；
- actor 在训练并行布局与推理布局之间频繁 reshard；
- 完整权重同步占用网络与暂停窗口，MoE 权重更大；
- reward/verifier 或环境响应慢，使昂贵 GPU 等待 CPU、I/O 或外部服务；
- 多轮上下文导致 prefill、KV cache 和轨迹存储快速膨胀；
- 训练引擎重算的 log-prob 与推理引擎实际采样概率不完全一致。

因此现代 RL infra 往往拆成下面的数据面：

```text
prompt/task ──> rollout / agent workers ──> trajectory buffer ──> learner
                    │                           │                   │
                    ├── inference engine        ├── reward/verifier │
                    └── sandbox/environment     └── logprob/metadata│
                                                                    │
                  weight/version service <──────────────────────────┘
```

上层控制面则负责资源编排、模型版本、弹性伸缩、超时、失败重试、checkpoint、指标和审计。把这两层分开，是从“训练脚本”走向“长期运行的 RL 平台”的标志。

---

## 四、Agentic RL 与 RL Infra 的架构演进 {#agentic-rl}

### 4.1 什么是 Agentic RL

在单轮数学 RLVR 中，环境几乎只在最后检查答案。Agentic RL 中，模型在每一轮观察环境、思考、选择工具、读取结果，再继续行动：

```text
任务 ─> 观察 o0 ─> 思考/动作 a0 ─> 环境 ─> 观察 o1 ─> ... ─> 终局验证
                         │                         │
                     tool call                tool result
```

环境真实状态 $$s_t$$ 往往不可完全见，模型只看到观测 $$o_t$$，因此更合适的形式是 POMDP。浏览器页面可能变化，代码仓库包含未读文件，工具会失败，历史还可能被摘要压缩。策略不仅要生成“正确答案”，还要学习何时搜索、验证、回退、停止和控制成本。

[RAGEN](https://arxiv.org/abs/2504.20073)在多轮环境中观察到 reward variance cliff、gradient spike 和 Echo Trap，并指出如果只有粗粒度终局奖励，agent 可能学到浅层捷径而非真正的推理。[WebAgent-R1](https://arxiv.org/abs/2505.16421)展示了异步在线交互和二元成功奖励可以显著提升 Web agent，但也强调 warm-up/behavior cloning 的重要性。[SWE-RL](https://arxiv.org/abs/2502.18449)与[Agent-RLVR](https://arxiv.org/abs/2506.11425)把可验证奖励扩展到真实软件演化和仓库级任务。

与单轮 RL 相比，Agentic RL 新增了几类难题：

- **长程信用分配**：最终测试失败，无法直接判断是第 2 轮检索、第 15 轮编辑还是第 30 轮验证有错；
- **动作边界**：自然语言 token、结构化 tool call 和环境动作需要正确 mask，不能把环境返回文本当作策略输出训练；
- **非平稳环境**：网站、依赖、数据库和其他 agent 都可能变化；
- **极端长尾**：简单任务数秒完成，编译、浏览或训练任务可能运行数十分钟；
- **高失败率**：容器崩溃、网络超时、工具异常和 verifier 故障不能混同为策略失败；
- **安全与隔离**：待训练策略会产生任意命令、代码和网络请求，且 RL 会主动搜索奖励漏洞。

这也是 token-in/token-out（TITO）逐渐成为基础设施原则的原因：agent harness 应把推理引擎实际接收和采样的 token ID 原样交给训练侧，而不是把多轮文本、工具调用与观测重新套模板、重新分词。否则一次看似无害的序列化差异，就可能让训练 loss 对应到另一条策略轨迹。

### 4.2 Sandbox Infra：环境不是一个 `reward()` 函数

一个可用于规模化 Agentic RL 的环境至少包含四个逻辑部件。NeMo Gym 的[公开定义](https://github.com/NVIDIA-NeMo/Gym)也是 dataset、agent harness、verifier 与 per-task state 的组合；[OpenEnv](https://github.com/meta-pytorch/OpenEnv)则用 Gymnasium 风格 API、容器化和 HTTP 服务来统一隔离环境。

| 部件 | 职责 | 关键要求 |
|---|---|---|
| Task package | 初始状态、任务描述、隐藏测试和资源 | 版本化、去污染、可分训练/验证/测试 |
| Agent harness | 消息模板、工具 schema、turn loop、context management | token-in/token-out 可追踪，动作 mask 正确 |
| Sandbox/environment | 执行代码、浏览网页、操作文件或服务 | 强隔离、快速 reset/snapshot、资源限额、网络策略 |
| Verifier/reward | 判断成功、过程进展、成本与安全 | 尽量确定、抗投机、可审计、区分 infra error |

生产级 Sandbox 还需要：

1. **安全边界**：容器或 microVM 隔离、非特权用户、seccomp/cgroup、文件系统与凭据隔离、出站网络 allowlist、执行超时和资源配额。
2. **可复现性**：固定镜像 digest、依赖锁定、数据快照、随机种子、虚拟时钟，以及每条轨迹对应的环境版本。
3. **生命周期效率**：预热池、copy-on-write snapshot、快速 reset、分层缓存和按任务亲和性调度；否则环境启动时间会吞掉 rollout 收益。
4. **可观测性**：记录原始 token ID、tool call、tool result、环境 diff、进程退出码、网络访问、verifier 细项和每阶段延迟。
5. **故障语义**：把 policy failure、invalid action、timeout、sandbox crash、verifier error 和平台中断分开；只有前几类在定义明确时才能进入奖励。

环境安全不是外围问题。RL 会系统性寻找所有能提高分数的路径，包括修改测试、读取隐藏答案、利用服务漏洞、伪造完成标记或让 verifier 超时。一个不安全的 sandbox 会把“越狱基础设施”误当成“agent 能力提升”。

### 4.3 从同步到 Fully Async

同步 RL 的逻辑最清楚：固定策略生成完整 batch，冻结这些数据，learner 更新若干步，再把新权重同步给所有 rollout workers。

```text
sync:       [ rollout v0 ][ train -> v1 ][ rollout v1 ][ train -> v2 ]
phase-async:[ rollout v0 --------][ rollout v1 --------]
                    [train v0][train v1]
fully async: rollout workers 持续产出；learner 持续消费；权重按版本独立刷新
```

[AReaL](https://arxiv.org/abs/2505.24298)把 generation 与 training 完全解耦，通过控制 workload 限制数据陈旧度，并使用 staleness-enhanced PPO；论文报告在其数学与代码设置中，相比同步系统最多取得 2.57 倍训练加速且保持最终性能。[Laminar](https://arxiv.org/abs/2510.12633)进一步提出 trajectory-level asynchrony 和 relay worker 参数服务，避免全局权重同步把所有 rollout 锁在同一个版本屏障上。

异步系统必须区分三个策略：

- $$\mu$$：真正生成该 token 的 behavior/rollout policy；
- $$\pi_{\mathrm{old}}$$：learner 侧定义 PPO trust region 的旧策略；
- $$\pi_\theta$$：正在更新的当前策略。

PPO ratio $$\pi_\theta/\pi_{\mathrm{old}}$$ 约束一次优化内的更新幅度，却不能自动校正样本其实来自 $$\mu$$。额外的 behavior correction 可写成：

$$
w_t=
\frac{\pi_{\mathrm{old}}(a_t\mid s_t)}
{\mu(a_t\mid s_t)}.
$$

直接使用长序列 importance sampling 会导致权重乘积爆炸或消失。工程中常采用 token-level truncated importance sampling（TIS）、sequence mask、V-trace 风格截断，或限制最大 policy lag。截断以可控偏差换取较低方差：

$$
\bar w_t=\min(w_t,c).
$$

所以 fully async 的正确目标不是“永不等待”，而是最大化有效样本吞吐，同时把版本延迟、importance weight 方差和环境分布漂移维持在经过验证的范围内。至少要记录 sample 的 model version、sampler log-prob、生成配置与环境版本；没有这些元数据，就无法判断 off-policy 程度。

### 4.4 Partial Rollout：不要让最长轨迹决定每一步时钟

Partial Rollout 至少有两种相关但不同的用法。

**第一种是 Kimi k1.5 的长轨迹续跑。** [Kimi k1.5](https://arxiv.org/abs/2501.12599)为每次 rollout 设固定输出 token 预算。超过预算的未完成轨迹保存到 replay buffer，下一迭代从已有前缀继续，旧前缀可以复用，只有当前新增片段需要新的 on-policy 生成；某些旧片段可从 loss 中排除。它让 128K 级 CoT 不必每次从头重生成，还可提前终止重复内容。

**第二种是 active partial rollout。** [APRIL](https://arxiv.org/abs/2509.18521)等工作关注同步 batch 的 straggler：并发启动更多请求，达到所需样本数后截断仍未完成的长尾请求，并在后续轮次继续或有策略地复用，从而提高 GPU 利用率。这里的重点不是单条 128K 轨迹本身，而是 batch makespan。

Partial rollout 会引入四个正确性问题：

- 前缀可能来自旧策略，后缀来自新策略，trajectory 不再由单一 policy 生成；
- 恢复时使用 token 前缀重新 prefill，必须保持 tokenizer、chat template 和工具消息完全一致；
- loss mask 要明确哪些 token 是 action、哪些只是旧前缀或环境 observation；
- 如果更容易截断失败或超长样本，会改变训练数据分布并产生隐式长度偏置。

正确实现必须保存 segment 边界、每段 policy version/log-prob、终止原因、KV/prefix 重建信息和可训练 token mask。Partial rollout 是一种调度策略，也是一种分段 off-policy 数据模型。

### 4.5 R3：MoE RL 的路由一致性

在 MoE 模型中，每层 router 为每个 token 选择 top-k experts。rollout 常在 SGLang/vLLM 上执行，训练重算则在 Megatron/FSDP 上执行；kernel、并行布局和浮点舍入的细小差异可能翻转接近决策边界的 expert 选择。经过多层传播后，即使模型权重相同，训练侧重算的隐藏状态和 log-prob 也可能明显偏离实际采样路径。

[Rollout Routing Replay（R3）](https://arxiv.org/abs/2510.11370)的做法是：

1. rollout 时记录每个生成 token、每个 MoE 层的 top-k expert 选择；
2. 数据管道把 routing metadata 与 token、log-prob 一起传给 learner；
3. 训练 forward 强制复用 rollout 的 expert identity；
4. router logits 和选中 expert 的 gate weight 仍由当前参数重算，使梯度继续流向 router。

R3 修复的是**离散计算路径不一致**，与 TIS 修复概率分布偏差并不相同。代价是额外的 routing metadata、推理引擎导出接口、跨流水线传输，以及训练 kernel 对 replay mask 的支持。NeMo RL 的[Router Replay 文档](https://docs.nvidia.com/nemo/rl/nightly/guides/router-replay.html)、veRL 的[实现说明](https://github.com/verl-project/verl/tree/main/examples/router_replay)和 ROLL 的[官方文档](https://alibaba.github.io/ROLL/docs/User%20Guides/Advanced%20Features/router_replay/)都体现了这种端到端要求。

该领域仍有开放争论：GSPO 报告 sequence-level ratio 足以稳定其 MoE 训练，并可能避免 routing replay；R3 论文则报告显式路由对齐优于 GSPO/TIS。两者实验设置不同，更稳妥的结论是：**loss 对 mismatch 的容忍度与消除 mismatch 是两种路线，是否需要 R3 应在具体模型、引擎、精度和训练长度上验证。**

### 4.6 一套面向 Agentic RL 的参考架构

```text
┌──────────────────────────── Control Plane ────────────────────────────┐
│ scheduler · model registry · version/staleness policy · autoscaling  │
│ checkpoint · retries · quotas · lineage · metrics · audit            │
└───────────────────────────────────────────────────────────────────────┘

 Task Store ──> Agent Orchestrator ──> Rollout Model Service
                    │       │                 │
                    │       ├── token IDs/logprobs/router trace (R3)
                    │       │
                    │       └── Sandbox/Environment Pool
                    │              ├── tools / browser / terminal
                    │              └── state snapshot / reset
                    v
              Trajectory Store ──> Reward & Verifier Farm
                    │                    │
                    └──── train mask + reward + provenance
                                         v
                                  Replay/Data Buffer ──> Learner
                                                           │
                                            Weight/Delta Service
                                                           │
                                            Rollout Model Service
```

这套架构有五条原则：环境执行与模型推理解耦；token ID 而不是重新编码后的文本是训练事实来源；每条轨迹带完整 provenance；同步、partial 与 fully async 共享同一版本语义；基础设施失败绝不能静默变成负奖励。

---

## 五、RL 训练收敛：进展、判据与挑战 {#convergence}

### 5.1 先定义“收敛”

对非凸、在线、策略不断改变数据分布的超大模型，几乎不存在实践可用的全局最优收敛证明。工程上至少要区分四层：

1. **优化收敛**：loss、梯度、KL、clip fraction 和 value error 不再剧烈发散；
2. **代理目标收敛**：训练 reward 或 verifier success 持续上升并进入平台期；
3. **任务能力收敛**：独立 held-out benchmark 的 pass@1、成功率和成本改善；
4. **泛化与安全收敛**：分布外任务、人类偏好、稳健性和安全不因代理目标优化而退化。

四层可能互相矛盾。reward 平滑上升但真实质量下降是 reward hacking；pass@1 上升但大 $$k$$ 的 pass@k 下降，可能是分布变尖而非能力边界扩展；平均成功率上升但只来自简单题，可能是 curriculum 或采样分布漂移；训练 policy 稳定但 rollout policy 崩溃，则可能是训练—推理 mismatch。

### 5.2 已经取得的进展

**第一，RL scaling 已从个案变成可重复观察。** OpenAI o1 报告性能随 train-time RL compute 和 test-time thinking compute 平滑提升；Kimi k1.5 报告上下文长度是新的扩展维度；DeepSeek-R1、DAPO、Seed1.5、Qwen3、Magistral、MiniMax-M1 等公开报告证明，多种 base model 与 recipe 都能通过可验证奖励显著提升推理任务表现。

**第二，开放复现改善了“只见曲线、不见配方”的局面。** DAPO 开放算法、数据和 veRL 代码；Dr. GRPO、Open-R1 等工作揭示 prompt template、长度归一化、采样温度和 base model 先验足以显著改变结论。社区开始把训练 recipe 当成研究对象，而不只比较算法名。

**第三，critic-free 与 value-based 两条路线都在进步。** GRPO/REINFORCE 系列降低了系统门槛，VAPO 则证明经过修正的 value model 在长 CoT 中仍能提供有效信用分配。研究焦点从“要不要 critic”转向“baseline 在什么粒度最可靠”。

**第四，系统吞吐和稳定性开始共同设计。** AReaL、Laminar、Partial Rollout 和现代框架减少同步 bubble；TIS、版本控制、token-in/token-out 和 R3 则尝试控制加速带来的 off-policy 与数值偏差。[Training-Inference Mismatch 的诊断研究](https://arxiv.org/abs/2605.14220)进一步表明，微小 log-prob 不一致可以独立触发训练崩溃，应当作为一等系统指标。

**第五，RL 后训练的规模规律开始被系统研究。** 一项覆盖多种模型规模和训练设置的[实证研究](https://arxiv.org/abs/2509.25300)观察到，在固定计算预算下，大模型少训练步往往优于小模型多训练步，高质量数据重复利用也可能有效。但这些规律目前主要来自数学推理设置，离跨领域、跨架构的普适 scaling law 仍有距离。

### 5.3 仍未解决的核心挑战

#### 挑战一：奖励是否代表真正目标

learned RM 是人类偏好的不完美代理，持续优化会触发 Goodhart's law。[Reward Model Overoptimization 的 scaling law](https://arxiv.org/abs/2210.10760)表明，proxy reward 与更接近真实目标的 gold reward 会在过度优化后背离。RLVR 虽然奖励可验证，也可能只验证最终答案、公开测试或格式，无法约束推理真实性、代码可维护性和隐藏副作用。

方向包括 verifier 隔离、隐藏测试、对抗测试、judge ensemble、不确定性惩罚、定期人类审计，以及把结果奖励与可验证过程奖励结合。2026 年的[Verifiable Process Rewards](https://arxiv.org/abs/2605.10325)展示了用符号或算法 oracle 提供 turn-level 稠密反馈的潜力，但它仍依赖中间步骤确实可验证。

#### 挑战二：稀疏奖励下的长程信用分配

一条 100 轮 agent 轨迹只得到 0/1，所有 action token 共用回报，梯度方差极高。PRM/critic 可以提供密集信号，却可能引入更难察觉的模型偏差；Monte Carlo rollout 能估计中间状态价值，但成本高。可验证过程奖励、hierarchical return、sub-trajectory advantage、counterfactual baseline 和任务进度函数是重要方向。

#### 挑战三：探索会耗尽

[Entropy Mechanism](https://arxiv.org/abs/2505.22617)观察到推理 RL 中策略熵在训练早期快速下降，并与性能平台期相伴。成功轨迹被反复强化后，模型可能只会一种解法，遇到新任务无法探索。简单增加 temperature 又会降低有效成功样本率。

需要同时管理 prompt 难度、组内成功比例、采样多样性、熵或 KL、失败轨迹价值，以及 curriculum。理想训练集应包含“当前策略偶尔能解、但尚未稳定掌握”的 frontier tasks，而不是全对或全错的题。

#### 挑战四：RL 是发现新能力，还是重排已有概率

[《Does RL Really Incentivize Reasoning Beyond the Base Model?》](https://arxiv.org/abs/2504.13837)发现，RLVR 模型在小 $$k$$ 的 pass@k 更好，但 base model 在足够大的 $$k$$ 下可达到相近甚至更高的覆盖，提示 RL 主要提高正确路径的采样效率，同时可能缩窄分布。这一结果并不证明 RL 永远不能发现新能力，却要求研究同时报告 pass@1 和大 $$k$$ 覆盖，而不是把一次采样性能全部解释成能力边界扩张。

更有希望的路线是：用 teacher、搜索、自博弈或工具反馈把当前分布外的新轨迹引入，再用 RL 巩固；或者用 boundary-aware curriculum 专门选择接近能力边界的任务。

#### 挑战五：算法中的隐式偏置

回答长度归一化、组内标准差、截断上限、EOS 处理、超长惩罚和 token/sequence aggregation 都会改变“一个样本对梯度有多大影响”。它们可能诱导更长回答、偏爱某类难度，或让被截断轨迹获得错误优势。Dr. GRPO 与 DAPO 的价值之一，就是把这些看似实现细节的偏置暴露出来。

#### 挑战六：异步带来的策略陈旧与高方差校正

Fully async 提高硬件利用率，却让轨迹来自不同旧版本。importance sampling 理论上可校正分布，长序列上却容易产生极端权重；截断和丢弃又引入偏差。真正的难题是联合选择 learner/rollout 资源比例、最大 lag、权重同步频率、ratio 粒度和 replay 年龄，使 wall-clock 收敛更快，而非只让 tokens/s 更高。

#### 挑战七：训练—推理不一致

即使 policy version 相同，训练引擎与 rollout 引擎仍可能因 BF16/FP16、量化、不同 attention/MoE kernel、padding、top-k 实现或 tokenizer/template 路径而产生不同 log-prob。TIS 可以减轻分布偏差，R3 对齐 MoE 路由，TITO 保留实际 token ID；但补丁叠加并不等于问题彻底解决。需要一个可 bitwise 或近似对齐的诊断基线，再逐项开启性能优化。

#### 挑战八：环境与评测本身不平稳

Agent 任务的网页、包仓库、外部 API 和数据会变化；同一策略在不同时间可能得到不同结果。若 verifier flaky、隐藏测试泄漏或 sandbox 版本漂移，训练曲线没有可比性。必须版本化环境，并用可重放 trace、固定快照和多次独立评测估计方差。

#### 挑战九：能力提升伴随遗忘与安全风险

在窄域 verifier 上持续训练可能损害通用对话、多语言、校准和安全拒答。推理更强的 agent 也更善于发现奖励与沙箱漏洞。reference KL、通用数据 replay、多目标约束、安全 curriculum 和独立 red-team eval 必须贯穿训练，而不能只在最终 checkpoint 补一次对齐。

### 5.4 一套更可信的收敛看板

| 指标 | 它回答的问题 | 危险信号 |
|---|---|---|
| train / held-out reward | 代理目标是否被学到、是否过拟合 | train 升而 held-out 降 |
| verifier pass@1 与 success@cost | 一次完成能力与成本是否改善 | reward 升但真实成功不升 |
| pass@k 曲线 | 分布变尖还是能力覆盖扩大 | pass@1 升、large-k 降 |
| policy entropy / top-token covariance | 探索是否枯竭 | 早期骤降并伴随平台期 |
| KL to reference / checkpoint | 策略漂移是否受控 | KL 突增或语言能力回归 |
| response/turn/tool length 分布 | 是否出现长度投机或死循环 | 错误回答变长、重复率升高 |
| zero-variance group 比例 | GRPO batch 是否有学习信号 | 大量全对或全错组 |
| clip fraction / gradient norm | 更新是否长期打在 trust-region 边界 | clip 饱和、gradient spike |
| critic loss / explained variance | value baseline 是否可信 | value loss 降但解释方差恶化 |
| policy lag / importance-weight ESS | 异步数据是否过旧 | ESS 崩溃、极端 ratio 增多 |
| sampler–trainer log-prob gap | 训推路径是否一致 | 同权重差异随长度累积 |
| MoE route match rate | R3 或路由一致性是否生效 | 层间 mismatch 快速放大 |
| infra error taxonomy | 失败来自模型还是平台 | timeout/crash 被记成任务失败 |
| OOD、人类偏好与 safety eval | 能力是否真正泛化且安全 | 窄域上涨、通用与安全退化 |

一个实用停止条件应是多目标的：held-out 成功率在多个评测窗口不再显著提升，探索指标尚未崩溃，KL/长度/成本处于预算内，训练—推理 mismatch 受控，且通用能力与安全套件没有不可接受回归。单看 reward plateau 既可能停得太早，也可能早已过拟合。

### 5.5 下一阶段值得关注的方向

- **从 outcome verifier 到可验证过程监督**：为真正可检查的中间动作提供稠密反馈，同时避免用另一个不透明模型替代真实目标。
- **能力边界驱动的 curriculum 与 self-play**：自动生成位于当前 frontier 的任务，让 RL 不只是压缩已有分布。
- **面向长轨迹的 off-policy 算法**：超越简单 token TIS，在可控偏差下提高 replay 与异步样本效率。
- **环境标准化**：统一 task、state、action、verifier、trace 与 provenance 接口，使训练和评测能跨框架复用。
- **算法—系统共同验证**：把 tokenization、log-prob、路由、精度和版本一致性纳入算法测试，而不再视为纯 infra 细节。
- **可恢复的全异步平台**：用 trajectory-level versioning、弹性沙箱、增量权重同步和明确的最大陈旧度服务目标（staleness SLA）支撑数周级训练。
- **能力与安全共同收敛**：在训练目标中显式处理副作用、权限、资源成本和不确定性，而不是事后过滤。

---

## 六、结语 {#conclusion}

LLM 强化学习的主线可以概括为：**以预训练模型为强先验，用在线产生的轨迹和可扩展反馈重新分配行为概率，再用系统工程把这个闭环扩展到更长上下文、更大模型和更真实的环境。**

PPO 代表了以 critic 和受限更新换稳定性的经典路线；GRPO 代表了以组内比较去掉 critic、让 RLVR 更容易扩展的路线。DAPO、Dr. GRPO、GSPO、VAPO 和 CISPO 展示的并不是一个已经终结的算法竞赛，而是对优势、长度、clip、ratio 和 value 的持续校正。

当目标从一道数学题扩展到一个真实任务时，Agentic RL 让 RL 重新成为完整意义上的序列决策问题。此时 Sandbox、environment、verifier、fully async、partial rollout、TITO 和 R3 不再是附属组件，它们共同定义了训练数据来自哪个策略、沿哪条计算路径产生、是否可以安全重放，以及梯度究竟在优化什么。

因此，下一代 RL Infra 最重要的指标也许不是峰值 tokens/s，而是 **verified useful trajectories per unit time**：单位时间内产生多少来源清晰、环境可信、策略偏差可控、能够真正改善 held-out 能力的轨迹。只有当算法目标、环境语义和系统执行三者一致时，“规模化 RL”才会转化为可持续的能力提升。

## 参考资料与延伸阅读

### 基础与算法

1. Sutton & Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html).
2. Schulman et al., [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
3. Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
4. Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155).
5. Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).
6. Ahmadian et al., [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback](https://arxiv.org/abs/2402.14740).
7. Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300).
8. Yu et al., [DAPO](https://arxiv.org/abs/2503.14476).
9. Liu et al., [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783).
10. Zheng et al., [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071).

### 模型与训练报告

11. Bai et al., [Constitutional AI](https://arxiv.org/abs/2212.08073).
12. Meta AI, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).
13. OpenAI, [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/).
14. DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948).
15. Kimi Team, [Kimi k1.5](https://arxiv.org/abs/2501.12599).
16. ByteDance Seed, [Seed1.5-Thinking](https://arxiv.org/abs/2504.13914).
17. Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).
18. Mistral AI, [Magistral](https://arxiv.org/abs/2506.10910).
19. MiniMax, [MiniMax-M1](https://arxiv.org/abs/2506.13585).
20. Google DeepMind, [Gemini 2.5 Technical Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf).
21. GLM Team, [GLM-4.5](https://arxiv.org/abs/2508.06471).
22. Xiaomi LLM Team, [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780).

### 系统、Agent 与收敛

23. Sheng et al., [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256).
24. Hu et al., [OpenRLHF](https://arxiv.org/abs/2405.11143).
25. Shen et al., [NeMo-Aligner](https://arxiv.org/abs/2405.01481).
26. Zhong et al., [RLHFuse](https://arxiv.org/abs/2409.13221).
27. Fu et al., [AReaL](https://arxiv.org/abs/2505.24298).
28. Wang et al., [RAGEN](https://arxiv.org/abs/2504.20073).
29. Wei et al., [WebAgent-R1](https://arxiv.org/abs/2505.16421).
30. Wei et al., [SWE-RL](https://arxiv.org/abs/2502.18449).
31. Zhou et al., [APRIL: Active Partial Rollouts](https://arxiv.org/abs/2509.18521).
32. Ma et al., [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers（R3）](https://arxiv.org/abs/2510.11370).
33. Sheng et al., [Laminar](https://arxiv.org/abs/2510.12633).
34. Gao et al., [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760).
35. Yue et al., [Does Reinforcement Learning Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837).
36. Cui et al., [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617).
37. Zhong et al., [Diagnosing Training Inference Mismatch in LLM Reinforcement Learning](https://arxiv.org/abs/2605.14220).
38. Zhang et al., [The Landscape of Agentic Reinforcement Learning for LLMs](https://arxiv.org/abs/2509.02547).
