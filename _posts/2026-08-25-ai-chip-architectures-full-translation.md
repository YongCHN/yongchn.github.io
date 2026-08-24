---
title: "AI 芯片架构：GPU、TPU、WSE、Trainium 与 LPU（全文译文）"
description: "一篇全面梳理主流 AI 芯片的长文：从内存墙出发，深入 NVIDIA、Google、AMD、Cerebras、AWS 与 Groq 的架构、扩展方式和软件栈。"
date: 2026-08-25 09:00:00 +0800
categories: [人工智能, 计算机体系结构]
tags: [AI芯片, GPU, TPU, WSE, Trainium, LPU]
author: Jacob Peake
translator: Yong
translation: true
translation_authorized: true
original_title: "AI Chip Architectures"
original_author: "Jacob Peake"
original_url: "https://www.jacobpeake.com/ai-chip-architectures"
image: /assets/images/posts/ai-chip-architectures.png
---

2018 年，在[国际计算机体系结构研讨会（ISCA）](https://en.wikipedia.org/wiki/International_Symposium_on_Computer_Architecture)上，**[John Hennessy](https://en.wikipedia.org/wiki/John_L._Hennessy)** 和 **[David Patterson](https://en.wikipedia.org/wiki/David_Patterson_(computer_scientist))** 发表了图灵奖演讲：**[《计算机体系结构的新黄金时代》](https://dl.acm.org/doi/10.1145/3282307)**。

20 世纪 80 年代，Hennessy 和 Patterson 开展其图灵奖获奖研究时，CPU 的单线程性能每年增长 52%。到 2018 年，随着**[摩尔定律](https://en.wikipedia.org/wiki/Moore%27s_law)**和**[登纳德缩放定律](https://en.wikipedia.org/wiki/Dennard_scaling)**走向终结，这一增长率已降至 3%。

业界需要领域专用架构（Domain-Specific Architecture，DSA）。两位学者在演讲中使用已经投入生产的 Google **[TPU v1](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)** 作为实例：在神经网络推理中，其吞吐量是 CPU 的 29 倍，能效则高出 80 倍。演讲最后预言：**“未来十年，我们将看到新型计算机体系结构的寒武纪大爆发。”**

这个预言已经成真。如今，数十种架构都在被认真研发：GPU、TPU、LPU、NPU、DPU、ASIC、晶圆级引擎、可重构数据流、神经形态、光子和模拟计算。其中，AI 计算尤其受到关注。

目前真正规模化部署的赢家包括：GPU（NVIDIA、AMD）、脉动阵列加速器（TPU、Trainium）、Cerebras 晶圆级引擎，以及 Groq LPU。

NVIDIA 显然处于领跑位置；AMD 紧随其后，分别获得了 [OpenAI](https://openai.com/index/openai-amd-strategic-partnership/) 和 [Meta](https://www.amd.com/en/newsroom/press-releases/2026-2-24-amd-and-meta-announce-expanded-strategic-partnersh.html) 各 6 GW 的采购承诺。TPU 用于训练 Gemini，并将以多达 100 万颗芯片[为 Anthropic 提供服务](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services)；Anthropic 还使用[超过 100 万颗 Trainium 芯片](https://techcrunch.com/2026/03/22/an-exclusive-tour-of-amazons-trainium-lab-the-chip-thats-won-over-anthropic-openai-even-apple/)运行 Claude。Cerebras [如今为 OpenAI 提供推理服务](https://openai.com/index/cerebras-partnership/)；Groq LPU 则通过一笔 200 亿美元的[人才收购交易被纳入 NVIDIA](https://www.datacenterdynamics.com/en/news/nvidia-builds-out-lpu-chip-team-following-20bn-groq-acquihire-announcement-rumored-for-gtc/)。

本文旨在综览这些不同路线，包括它们的设计哲学、体系结构、扩展方法（纵向扩展与横向扩展），以及软件栈（也就是如何为芯片编程）。

---

## 问题所在

AI 计算主要由**矩阵乘法**主导。Transformer 由一连串矩阵乘组成：Q/K/V 投影、注意力、输出投影、前馈网络（FFN），其间穿插归一化、激活和残差相加等逐元素操作。训练一个前沿模型需要执行约 `10^25` 次乘加运算（矩阵乘本身就是一连串乘加运算）。

矩阵乘的**形状**取决于具体负载。**训练**会让一批序列依次前向通过每一层，再反向传播损失并更新权重；同一时刻，会有数千个 token 流过同一个权重矩阵。**预填充（prefill）**是推理中摄取提示词的阶段：在生成第一个输出 token 之前，完整输入序列会一次性通过模型完成投影。训练和预填充都会让许多 token 复用同一个权重矩阵，因此每一层执行的是大型**矩阵-矩阵乘法**（GEMM），算术强度很高，通常受计算能力限制。

**解码（decode）**则是自回归的：模型每次只输出一个 token，每个 token 都取决于之前的所有 token；在 token N 生成之前，token N+1 无法开始。每一步只有一个 token 接受投影，因此每个矩阵乘都变成了**矩阵-向量乘法**（GEMV）。生成一个 token 需要完整遍历模型的全部权重，还要为注意力完整读取 KV Cache。与预填充相比，其算术强度会下降几个数量级。

推理系统会通过批处理 token 来恢复一部分算术强度，把 GEMV 重新提升为 GEMM：**连续批处理**把多个用户的解码步骤堆叠起来；**推测解码**为每个请求一次堆叠 K 个草拟 token，再用一次前向传播验证；**多 token 预测**则把同样的技巧直接内置在模型中。这些方法能提高矩阵乘单元的利用率，并提升每字节运算次数（Ops/B）。不过在连续批处理中，每个用户的请求仍需读取各自的 KV Cache，因此长上下文解码会从受权重带宽限制，转为受 KV 带宽限制。

这里的体系结构难题，就是怎样足够快地把数值搬到执行矩阵乘的位置。这就是所谓的**内存墙**：计算能力一直呈指数增长，内存带宽却没有。

每一种架构都为赢下数据搬运这场比赛提出了不同策略。理解一颗芯片，最终可以归结为四个问题：数据存放在哪里？数据如何移动到计算单元？计算单元本身是什么样子？以及在大规模系统中，芯片之间怎样通信？

---

## NVIDIA GPU

> **设计哲学**：NVIDIA GPU 是一种大规模并行处理器。其理念是：由主机 CPU 协调、通过 **[CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)** 暴露给开发者、拥有数千个线程的可编程芯片，才是运行可并行化负载的正确机器。每一代产品都会在可编程流式多处理器上增加新的加速原语，却不改变编程模型。同一颗芯片既能训练 Transformer、提供推理服务，也能渲染图形和运行科学模拟——这就是“加速计算”。

### 演进谱系

| 年份 | 架构与芯片 | 关键变化 |
| --- | --- | --- |
| 2006 | **[Tesla](https://en.wikipedia.org/wiki/Tesla_(microarchitecture)) G80** | 第一款支持 CUDA 的 GPU；统一着色器和 SIMT 执行模型。 |
| 2010 | **[Fermi](https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf) GF100** | 第一款真正的计算架构：统一 L1/L2 缓存、双 warp 调度器、IEEE-754 FP64。 |
| 2012 | **[Kepler](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf) K20、K40** | 引入 SMX、动态并行和 Hyper-Q；GPU 可以自行发起工作。 |
| 2014 | **[Maxwell](https://developer.nvidia.com/maxwell-compute-architecture) M40** | 重新设计 SM，每瓦性能约为 Kepler 的 2 倍。 |
| 2016 | **[Pascal](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf) P100** | NVLink 1.0、HBM2、原生 FP16 吞吐；第一款明确面向深度学习设计的 GPU。 |
| 2017 | **[Volta](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) V100** | 首次加入 Tensor Core；引入独立线程调度。 |
| 2018 | **[Turing](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf) T4** | 第二代 Tensor Core，支持 INT8/INT4；首次加入 RT Core。 |
| 2020 | **[Ampere](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) A100** | 第三代 Tensor Core，支持 TF32 和结构化稀疏；引入多实例 GPU 分区。 |
| 2022 | **[Hopper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) H100、H200、GH200** | 第四代 Tensor Core、FP8、Transformer Engine；HBM3、TMA、线程块集群和异步 `wgmma`。 |
| 2024 | **[Blackwell](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief) B100、B200、GB200** | 第五代 Tensor Core，支持 FP4；Tensor Memory（TMEM）、双裸片 Chiplet GPU、NVLink 5。 |
| 2025 | **[Blackwell Ultra](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) B300、GB300** | 中期更新：FP4 吞吐约提高 1.5 倍，288 GB HBM3e；针对长上下文推理调优。 |
| 2026 | **[Rubin](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference) Rubin、VR200、Rubin CPX** | HBM4、第三代 Transformer Engine、搭配 Vera CPU，通过 Rubin CPX 实现预填充解耦。 |
| 2027 | **[Rubin Ultra](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) Rubin Ultra** | 四裸片 GPU 封装，每个封装 1 TB HBM4e；部署在功耗 600 kW 的 NVL576 Kyber 机架中，每 GPU 达 100 PetaFLOPS FP4。 |

### 体系结构

一颗 NVIDIA GPU 由三部分组成：**面向吞吐量的核心、确保核心持续获得数据的深层内存层次，以及刚好足够让数千个线程同时处于执行状态的调度硬件**。这些核心称为**流式多处理器（SM）**，每个封装会复制 100 多个：V100 有 80 个，A100 有 108 个，H100 有 132 个，B200 有 148 个，B300 有 160 个，Rubin 有 224 个。

每个 SM 内部都采用同样的配方：四个 SM 子分区，每个子分区各自拥有 warp 调度器、分派单元、`16K × 32 bit` 寄存器文件、标量 CUDA Core 通道、用于超越函数的特殊函数单元，以及通往 SM 内 Tensor Core 的专用端口。四个子分区共享一块 L1/共享内存以及 TMA。线程每 32 个组成一个 warp，以 SIMT 锁步方式执行；每个子分区可同时驻留数十个 warp，调度器通过在它们之间切换来隐藏内存或算术停顿。

![Blackwell B200 单裸片平面图](/assets/images/posts/ai-chip-architectures-full/nvidia-gpu-die.png)

*图 1：Blackwell B200 单裸片平面图。GigaThread Engine 沿中央布置，把裸片分成左右两半；每一半都有自己的 L2 缓存带，上下两侧分布着 GPC 集群。HBM3e 堆栈经内存控制器排列在外缘；顶部是 NVLink 和较小的 PCIe Gen 6 主机链路，底部 NV-HBI 桥则连接镜像的第二颗裸片，共同组成完整封装。*

![流式多处理器内部结构](/assets/images/posts/ai-chip-architectures-full/nvidia-sm.png)

*图 2：一个流式多处理器的放大图。四个子分区各自拥有 warp 调度器、分派单元、寄存器文件和 Tensor Memory，共享下方的 L1/SMEM 与 TMA。*

#### 计算

CUDA Core 是最初的计算吞吐来源；在 AI 中，它们仍负责矩阵乘以外的所有工作，包括激活、残差相加、归一化和地址运算。但 Transformer 块约 99% 的 FLOPs 都来自矩阵乘，因此绝大多数计算吞吐由 Tensor Core 提供。

Tensor Core 会在小型矩阵块上执行**融合矩阵乘加**：`D = A · B + C`。完整矩阵乘被拆成许多输出块。为生成一个输出块，kernel 会沿共享内维度 K 前进：从左输入矩阵的一条行带中取 A，从右输入矩阵的一条列带中取 B，再把每次部分乘积累加到持续更新的累加器中。C 保存当前的部分和，D 是更新后的数值，并会被带入下一步。内循环完成后，D 就是完整输出矩阵中的一个成品块；许多这样的分块 MMA 共同构成完整矩阵乘。

矩阵块形状写作 **M × N × K**：M × N 是输出块大小，K 表示一条指令单次收缩的内维长度；矩阵乘 K 轴的其余部分由 kernel 内循环遍历。累加器在整个循环中保持不变：每次 MMA 的输出 D 会成为下一次 MMA 的输入 C，因此实际原地执行的是 `C ← A · B + C`。连续指令不断把部分乘积折叠进同一存储位置，直到完整遍历 K 轴。

V100 的第一代单元（每 SM 8 个）执行 warp 级 `16×16×16` FP16 MMA。A100 的第三代单元加入 TF32、BF16、FP64 矩阵乘和 2:4 结构化稀疏。H100 的第四代单元加入原生 FP8，并把抽象层级从一个 warp 提升为一个 **warp group**：128 个线程协同发起形状为 `64×256×16` 的异步 `wgmma`；矩阵乘在后台执行，同时发起它的 warp 加载下一个数据块。

B200 的第五代单元更进一步：它支持由一对 SM 分担操作数的 `256×256×16` **双 SM MMA**、原生 FP4，并为每个 SM 配备专用的 256 KB Tensor Memory（TMEM）scratchpad，用于保存累加器块，避免它们挤占寄存器文件。Rubin 的第六代单元继续提高 FP4 吞吐，加入原生 FP6，并配合第三代 Transformer Engine 在硬件中执行自适应 NVFP4 微块缩放；每个块的量化元数据都留在 Tensor Core 路径上，不必经过 CUDA Core。

六代产品始终不变的一点是：矩阵乘仍位于线程/warp 层次内部。但发起一条矩阵乘指令所需的线程数越来越少，而且“发起”本身逐渐与“执行”解耦。Volta 的 `mma.sync` 是 warp 集体同步指令：一个 warp 的全部 32 个线程共同执行，每个 lane 都在寄存器中持有 A、B 和累加器 D 的片段；指令完成之前，warp 会一直阻塞。

Hopper 的 `wgmma.mma_async` 把发起者扩大为 128 线程的 warp group，把 B 放入共享内存描述符（A 可由 kernel 选择放在寄存器或描述符中），并在发起后**立即返回**。矩阵乘在后台执行，warp group 同时排入下一个块；完成状态由 `wgmma.commit_group` 和 `wgmma.wait_group` 跟踪。

Blackwell 的 `tcgen05.mma` 完成了这次迁移：A 与 B 一同进入共享内存描述符（A 也可直接来自 TMEM），累加器 D 则落入 TMEM，而非寄存器文件。所有操作数都离开 lane 之后，就不再需要协调任何逐线程状态，因此只需**单个线程**发起指令并立即返回；完成信号通过 `mbarrier` 发出，消费者 warp 在其上等待。与此同时，其余 warp 乃至发起指令的线程都可以自由执行其他工作。

CTA-pair 变体把相同模型扩展到两个 SM：成对集群中每个 SM 各有一个线程发起彼此协调的 MMA，并在 SM 对之间共享操作数，在相同的异步/`mbarrier` 完成机制下组合出 `256×256×16` 双 SM 数据块；只是屏障被提升到集群层级，以确保两个 SM 步调一致。

矩阵乘在变大的同时，发起它的线程负担却越来越轻：一条最初要求 32 个 lane 锁步执行的指令，如今更接近由描述符驱动的单一命令。它仍从 warp 模型内部发出，却已经不再由 warp 亲自执行。

正是这种解耦，让 Transformer 注意力 kernel 能在 GPU 上高效运行。矩阵乘尚在执行时，warp 可以计算 softmax、应用掩码或预加载下一个数据块。现代注意力 kernel（FlashAttention-3、FA4）的共同结构，就是让矩阵乘与周围逐元素工作重叠；前提正是矩阵指令不能阻塞 warp。

#### 内存

片上采用**各级硬件管理缓存，再叠加软件提示**的层次结构。片外是 HBM：V100 配备 32 GB HBM2，H100 配备 80 GB HBM3，B200 配备 192 GB HBM3e，B300 配备 288 GB，Rubin 配备 288 GB HBM4。芯片级 L2 缓存位于 HBM 和 SM 之间：V100 为 6 MB、A100 为 40 MB、H100 为 50 MB、B200 为 60 MB。B200 的 L2 被拆成两个 30 MB 存储体，分别位于双裸片封装的两侧；利用位置感知的驻留控制，可以把热点数据块固定在更近的裸片上。

每个 SM 内有 256 KB 统一 L1/SMEM，kernel 启动时会在硬件管理的 L1 与程序员控制的 scratchpad 之间划分空间。每个 SM 还有约 256 KB 寄存器文件，分成四份供各子分区使用。

Blackwell 加入第五层存储：TMEM。每个 SM 配备 256 KB、专供 MMA 累加器使用、且只能由 Tensor Core 寻址的 TMEM，把操作数驻留带来的压力从通用寄存器文件中移走。

各层之间的数据移动也在逐步与 warp 解耦。Ampere 以前，加载一个数据块是同步过程：每个线程发起自己的全局加载，warp 阻塞到所有片段进入寄存器，再进行第二次复制，把数据送入共享内存。每个数据块都会让 warp lane 花时间做地址运算和等待。

Ampere 引入 `cp.async`：每个线程可发起绕过寄存器、直接从 HBM 到 SMEM 的异步复制。warp 把在途复制分组提交，只在消费者真正需要数据时等待。Hopper 又以专用 DMA 引擎 TMA 取代这一方式：一个线程提交多维数据块描述符（基地址、主维度、swizzle），引擎负责全部地址运算并把数据写入共享内存，完成状态通过 `mbarrier` 发出。

整个 warp 因而无需再发起加载和计算地址，kernel 只需把描述符放入队列。TMA 还支持**集群级多播**：一次 HBM 读取就能扇出到线程块集群内的所有 SM，把过去 N 次独立加载变成一次。Blackwell 再次扩展 TMA，允许直接加载到 TMEM，因此累加器块无需经 SMEM 暂存即可流入。每一代产品都让 warp 每处理一个数据块少做一件事。

#### Warp 专门化

Hopper 时代的编程惯用法是 **warp 专门化**：在同一个线程块中，一部分 warp 充当生产者，连续发起 TMA 加载；另一部分 warp 充当消费者，在新数据块到达后发起 `wgmma`。二者不再通过旧式 SM 级 `__syncthreads()` 同步，而是使用共享内存中的 `mbarrier`，以及与 TMA 完成事件绑定的异步事务屏障。这样可以在 warp 粒度上执行细粒度生产者/消费者握手，而不是让整个线程块一起同步。

现代注意力 kernel 的参考模式——FlashAttention-3、[CUTLASS](https://github.com/NVIDIA/cutlass) 乒乓 GEMM、Blackwell FA4 kernel——都使用同一套方法：TMA 驱动的生产者流水线经共享内存和 TMEM 向 `wgmma` 消费者流水线供数，双方通过 `mbarrier` 握手；线程块集群（Hopper 及以后）再把多个 SM 绑定为协同计算单元，使 Blackwell 的双 SM MMA 能自然构建在这套模式之上。

#### 数值格式

FP32 曾是默认格式。Volta 带来使用 FP32 累加的 FP16，以及让 FP16 能够用于训练的损失缩放技巧。Ampere 加入 TF32（FP32 的指数范围、FP16 的尾数精度，可直接替换 FP32 矩阵乘）、BF16，以及可令剪枝权重有效吞吐翻倍的 2:4 结构化稀疏。

Hopper 引入 E4M3 和 E5M2 两种原生 FP8，并配合 Transformer Engine 对每一层激活自动缩放，使其保持在 FP8 的动态范围内。Blackwell 再次把精度减半至 FP4，同时推出微缩放 MX 格式：数据块级共享指数可以挽回 FP4 损失的大部分精度；第二代 Transformer Engine 也把自动缩放流水线重新面向 FP4。Rubin 的第三代 Transformer Engine 加入 NVFP4（NVIDIA 收紧后的 FP4 变体）、原生 FP6，以及更激进的稀疏支持。

芯片布局本身如今也是数值格式故事的一部分。B100/B200/B300 都由两颗达到光罩极限的裸片组成，通过约 10 TB/s 的 NV-HBI 链路缝合，对软件呈现为一颗逻辑 GPU；封装上配有 8 个 HBM 堆栈。Rubin 延续双裸片 Chiplet 配方，约含 3360 亿个晶体管和 8 个 HBM4 堆栈。每一代产品都通过把位宽减半，再以粒度更细的缩放方案恢复精度——并且越来越多地通过在封装中键合更多硅片——换取约 2 倍的每瓦吞吐。

#### 五项押注

1. **可编程性。** 工作负载是移动的目标（注意力变体、新模型架构），因此让每个模块保持可编程，并允许开发者编写 [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)。即便是专用单元，也通过这一模型暴露，而不是作为固定功能模块。
2. **用大规模多线程隐藏延迟。** 延迟不可预测且依赖数据，因此不使用静态排程，而是通过大量超额线程隐藏延迟：每个 SM 最多驻留 64 个 warp，由硬件 warp 调度器每个周期挑选一个就绪 warp。
3. **用 warp 包裹矩阵乘。** 矩阵单元贡献了绝大部分计算吞吐，但它必须位于其他工作所使用的同一 warp/线程抽象之后。因此使用 `mma.sync → wgmma → tcgen05.mma` 包裹它，而不是暴露为固定功能流水线。这样，单个 kernel 就能在一次遍历中融合矩阵乘、softmax 和逐元素操作。
4. **异步内存层次。** 让内存层次显式化并由程序员管理，而不是完全隐式或由编译器针对已知延迟的 scratchpad 静态排程。保留 L2 缓存，同时把 SMEM 和 TMEM 暴露为具名 scratchpad，再叠加异步机制：TMA 执行批量复制，TMEM 保存矩阵乘累加器，`mbarrier` 完成生产者/消费者握手。整个层次由可编程 kernel 在软件中形成流水线。
5. **摊薄 SIMT 税。** 每一个用于 warp 调度器、寄存器文件或一致性缓存的晶体管，都无法再用于 MAC。NVIDIA 接受这项成本，并以两种方式摊薄它：一是让 Tensor Core 足够大，使 SIMT 机制的成本分摊到更多 MAC 上；二是使用 TMEM 等单元，以部分通用灵活性换取更高 MAC 密度。

### 扩展

扩展分为两个范围：**纵向扩展（scale-up）**和**横向扩展（scale-out）**。

**纵向扩展**把多颗 GPU 绑定到一个一致性内存域。任何 GPU 都能通过 NVLink 以纳秒级延迟直接加载或存储另一颗 GPU 的 HBM：使用同一地址空间，不必显式传输。

**横向扩展**在机架和集群层面把这些内存域连接起来。数据通过显式 RDMA 以微秒级延迟跨域：各域地址空间彼此分离，但一个集群可容纳数万颗芯片。

AI 基础设施会同时使用两者：张量并行、MoE 专家路由等带宽密集型集合通信留在纵向扩展域内；数据并行和流水线并行则跨越横向扩展网络。

#### 纵向扩展

纵向扩展栈由 NVLink 和 NVSwitch 组成。NVLink 在 GPU 之间实现缓存一致性互连，因此一颗 GPU 发出的加载或存储可以直接指向另一颗 GPU 的 HBM，由硬件处理地址转换和一致性。但 NVLink 本身是点到点的：一条链路只连接两颗芯片。NVSwitch 则是一颗专用交叉开关芯片，每颗 GPU 都与其相连；它负责路由，使所有 GPU 能以完整 NVLink 带宽同时彼此通信，实现无阻塞全互连。

两者共同定义了 HGX 八 GPU 基板：八个 H100 SXM 模块通过 PCIe Gen5 与 x86 主机（AMD EPYC 或 Intel Xeon）配对。Hopper 还提供搭配 Grace 的版本：GH200 Grace Hopper Superchip 通过 900 GB/s 的 NVLink-C2C 把一颗 Grace ARM CPU 与一颗 H100 键合，消除 PCIe 主机-设备跳转。模块可纵向扩展为 GH200 NVL2 对和机架级 GH200 NVL32。到了 Blackwell，CPU-GPU 配对成为默认形式。

GB200 模块通过 NVLink-C2C 把一颗 Grace 与两颗 B200 融合；NVL72 再把 36 个这样的模块连接成单一液冷纵向扩展域：72 颗 GPU、36 颗 Grace CPU、13.5 TB HBM 和 17 TB LPDDR5X，共同构成一个平坦的一致性地址空间。

Rubin 分两步推进。NVL144 于 2026 年推出，它是在同一 Oberon 级机架中的 Rubin 代际更新：包含 72 个 Rubin 封装；按照 NVIDIA 新的裸片计数惯例标为 144 颗 GPU，HBM4 与 NVLink 6 使单封装带宽翻倍。真正的机架级跃迁发生在 2027 年的 Rubin Ultra：NVL576 把 144 个四裸片 Rubin Ultra 封装装入新的 Kyber 机箱，在一个一致性域内形成 576 个 GPU 裸片。

![NVIDIA NVL72 纵向扩展](/assets/images/posts/ai-chip-architectures-full/nvidia-scale-up.png)

*图 3：NVL72。72 颗 Blackwell GPU 位于一排 NVSwitch ASIC 之下，后者组成无阻塞交叉开关，使任意 GPU 都能以完整 NVLink 带宽寻址其他 GPU 的 HBM。整套互连通过无源铜背板运行：约 5,184 根线缆盲插，提供约 130 TB/s 全互连带宽，相比光学方案节省约 20 kW 收发器功耗。*

如此高的密度靠**无源铜缆**维系。NVL72 的 NVLink 网络通过背板盲插 5,184 根线缆——每机架约有 2 英里线缆，不使用线内重定时器，SerDes 位于 GPU 和交换 ASIC 本身——在 72 颗 GPU 之间承载约 130 TB/s 全互连带宽。NVIDIA 估算，相比需要在每条链路上安装可插拔收发器的光学方案，铜缆每机架可节省约 20 kW。

铜缆让“把机架当成一颗 GPU”在经济上成为可能：距离短于 2 米时，它在功耗、成本和单位价格信号完整性方面仍然胜出；超过这个距离，数据位就必须转移到光纤上。

NVL144 仍位于 Oberon 机架内，封装数量与 NVL72 同为 72，所以铜缆继续有效：线缆不必变长，只需通过第六代 SerDes 以更高速度传输。Rubin Ultra 的 NVL576 则通过重塑机架延续同一铜缆界线：新的 Kyber 形态高度约为 Oberon 的两倍，把全部 576 个 GPU 裸片装入一个机箱。它的尺寸经过专门设计，即使拥有 144 个四裸片封装和数万根线缆，也能让每条 NVLink 路径保持在无源铜缆可及范围内。

#### 横向扩展

横向扩展栈来自 NVIDIA 对 Mellanox 的收购。与 NVLink 不同，横向扩展网络**不保持一致性**：各节点拥有独立地址空间，数据只能通过软件显式发起的 RDMA 跨越网络，通常被封装在 NCCL 的 all-reduce、all-to-all 等集合操作中。

参考集群是 DGX SuperPOD：八个 NVL72 机架经 Quantum-X800 InfiniBand 连接，在同一个调度器下组成 576 颗 Blackwell GPU；更大的训练集群继续平铺多个 SuperPOD。2026 年的 Rubin SuperPOD 保留同样的八机架布局，但改用 NVL144，使每个 SuperPOD 从 576 颗 GPU 增至 1,152 颗。2027 年 Rubin Ultra 把该配方扩大一个数量级：每个 Kyber 机架含 576 个 GPU 裸片，机架之间通过 Quantum-X Photonics CPO 连接，在同一调度器下容纳数千颗 GPU。

![NVIDIA DGX SuperPOD 横向扩展](/assets/images/posts/ai-chip-architectures-full/nvidia-scale-out.png)

*图 4：DGX SuperPOD。八个 NVL72 机架（共 576 颗 GPU）位于 Quantum-X800 InfiniBand 主干之下。每颗 GPU 通过 800 Gbps ConnectX-8 NIC 横向扩展；跨机架跳转经过 OSFP-RHS 可插拔光收发器，延迟从机架内 NVLink 的纳秒级提高到微秒级。*

每颗 GPU 都有自己的 ConnectX NIC 接入网络。Blackwell 节点使用每 GPU 800 Gbps 的 ConnectX-8，比单 GPU NVLink 带宽低一个数量级；延迟也从纳秒升至微秒。Rubin 改用每 GPU 1.6 Tbps 的 ConnectX-9；当单机架纵向扩展域从 72 颗增至 576 颗 GPU 时，单 GPU 横向扩展带宽也随之翻倍。每块 NIC 旁还配有 BlueField DPU，以 ARM 核心和加速器从主机 CPU 卸载存储、网络和安全任务。对于偏好以太网而非 InfiniBand 的客户，Spectrum-X 提供针对 AI 流量调优的无损以太网替代方案。

铜与光的分界发生在机架边界。NVL72 内部主干使用铜缆；链路一旦需要以 800 Gbps 跨机架，就会变为光学链路。200 G/通道的无源铜制 DAC 极限约为 1.5 至 2 米，远不足以跨机架，因此当前 SuperPOD 主干使用 OSFP-RHS 可插拔收发器，每个模块都包含自己的激光器、调制器、光电探测器和 DSP。若 SuperPOD 主干向数千颗 GPU 扇出，从光学角度看，就意味着数万个可插拔模块，单是收发器激光器便要消耗数十千瓦。

到了 Rubin，这一光学层被收进交换 ASIC。Quantum-X Photonics（InfiniBand）和 Spectrum-X Photonics（以太网）以**共封装光学**取代可插拔模块：激光器、调制器和光电探测器通过 TSMC COUPE 键合到交换芯片封装上。NVIDIA 宣称，相比 OSFP 可插拔方案，这可将激光器数量减少约 4 倍、链路功耗降低约 3.5 倍。曾经把 GPU 变成双裸片封装、把 HBM 堆叠到旁边的 Chiplet 逻辑，如今也出现在网络层：计算、内存和光子器件被垂直集成在同一衬底上。

NVLink Fusion 最近还开放了纵向扩展网络本身：第三方 CPU 和 XPU 可以加入 NVLink 域，让超大规模云厂商围绕 NVIDIA 互连构建半定制机架，而不必从头设计自己的缓存一致性网络。

### 软件

**[CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)** 是大规模并行处理器的自然编程模型。开发者编写 kernel（一段由每个线程各执行一次的代码），再把它发射到由线程块和 warp 组织的数千个线程上；由程序员决定线程共享什么、何时同步，以及每个线程负责问题的哪一部分。这正是该抽象十八年来几乎没有变化的原因，也是为什么 2007 年以来编写的 CUDA kernel 直到 Blackwell 上仍能编译运行。

这种连续性既是护城河，也是约束。每一代新硬件——Tensor Core、TMA、TMEM——都必须加入同一个 kernel/warp 模型，再通过 **[PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/)** 和 **[SASS](https://docs.nvidia.com/cuda/cuda-binary-utilities/)** 中的内在指令暴露，例如 `mma.sync`、`wgmma.mma_async`。NVIDIA 无法彻底重构 SM，因为已有太多代码依赖它；作为回报，每一笔 CUDA 软件投资都能跨代累积价值。

PTX 之上是一套构建了二十年的软件栈：**[cuBLAS](https://docs.nvidia.com/cuda/cublas/)** 和 **[cuDNN](https://developer.nvidia.com/cudnn)** 提供数学与 DNN 原语；**[CUTLASS](https://github.com/NVIDIA/cutlass)** 用模板化 C++ 编码数十年的 GEMM 经验；**[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** 提供分页注意力、运行中批处理和推测解码；再通过 **[PyTorch](https://pytorch.org/)**、**[Triton](https://triton-lang.org/)** 和 **[JAX](https://github.com/jax-ml/jax)** 接入框架。

**[FlashAttention](https://arxiv.org/abs/2205.14135)** 是现代 AI 最重要的算法重写之一。它对注意力进行分块，避免物化 `O(N²)` 矩阵。其四代版本（FA1 到 FA4）都针对最新 NVIDIA 硅片手工优化：FA3 面向 Hopper 的异步流水线，FA4 面向 Blackwell；移植到其他硬件往往会晚数月甚至数年。

这套软件栈的大部分并非由 NVIDIA 员工编写。真正的护城河并不只是 CUDA 本身，而是二十年来积累的第三方 kernel、库和工具，以及沿途学会这套 API 的数百万开发者。

NVIDIA 还会随硅片一起交付人的专业能力：把数十名自家工程师派驻前沿实验室和超大规模云团队，为每种新模型架构编写 kernel，并针对每一代新硅片调优。实验室下个月想训练的任何东西，往往都能比在其他平台上更快地在 NVIDIA 上获得良好性能。因此，离开 NVIDIA 不只是重写 kernel 和库，还意味着要重新训练整个工程团队的思维模型，并失去目前驻扎在公司内部的 NVIDIA 工程师。

---

## Google TPU

> **设计哲学**：**[TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)** 是一台**矩阵乘法机器**。它不追求成为能够运行任意大规模并行负载的可编程芯片，而是专注于一个原语：在大型[脉动阵列](https://en.wikipedia.org/wiki/Systolic_array)上执行稠密矩阵乘法，并让 **[XLA](https://openxla.org/xla)** 编译器提前规划每一个周期和每一个字节的内存。没有硬件调度器，没有缓存，也没有线程或 warp。每一代产品都继续扩大 Pod，通过 ICI 互连把数千颗芯片连接为一台一致的机器。TPU 无意渲染图形或运行科学模拟；它存在的目的，是以高于任何通用替代方案的每瓦效率训练并服务 Google 的负载，包括搜索、翻译、推荐和 Gemini。

### 演进谱系

| 年份 | 架构与芯片 | 关键变化 |
| --- | --- | --- |
| 2015 | **[TPU v1](https://arxiv.org/abs/1704.04760)** | 首款投入生产的深度学习 ASIC；仅支持经 PCIe 连接的 INT8 推理。 |
| 2017 | **[TPU v2](https://cacm.acm.org/research/a-domain-specific-supercomputer-for-training-deep-neural-networks/)** | 第一款支持训练的 TPU；MXU 从 INT8 转向 BF16，确立“双 TensorCore + HBM”结构。 |
| 2018 | **[TPU v3](https://cacm.acm.org/research/a-domain-specific-supercomputer-for-training-deep-neural-networks/)** | 第一款液冷 TPU；MXU 和 HBM 相比 v2 翻倍；Pod 扩至 1,024 颗芯片。 |
| 2020 | **[TPU v4](https://arxiv.org/abs/2304.01433) v4、v4i** | 首次采用可重构光路交换机 Palomar；加入 SparseCore；同时支持 BF16 和 INT8；Pod 扩至 4,096 颗芯片。 |
| 2023 | **[TPU v5](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer) v5e、v5p** | v5e 面向能效，v5p 面向性能；v5p 的 INT8 FLOPs 为 v4 的 3.3 倍，HBM 带宽为 2.2 倍；Pod 扩至 8,960 颗芯片。 |
| 2024 | **[Trillium](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus) v6e** | 首次采用 256×256 MXU；功耗相近时峰值 FLOPS 为 v5e 的 4.7 倍；用于训练 Gemini 2.0。 |
| 2025 | **[Ironwood](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/) v7** | 面向推理模型的推理阶段设计；加入原生 FP8；9,216 芯片 SuperPod 达到 42.5 ExaFLOPS FP8。 |
| 2026 | **[TPU v8](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/) 8t、8i** | 8t 面向训练，8i 面向推理；加入原生 FP4；8t 的 9,600 芯片 SuperPod 达到 121 ExaFLOPS FP4。 |

### 体系结构

一颗 TPU 就是**一台矩阵乘引擎，外围只配备足以持续为它供数的硬件**。计算单元称为 **TensorCore**：v2 以后的旗舰芯片每个封装配备两个，面向能效的芯片（v4i、v5e、v6e）则配备一个。

每个 TensorCore 内都有同一套五组件配方：一个或多个执行矩阵运算的 MXU、执行逐元素运算的 VPU、统筹全局的标量单元、处理跨 lane 归约的 XLU，以及相连的转置/置换单元；此外还有负责向 MXU 送入数据和排出结果的累加器队列。

从 v4 开始，每颗芯片还会在 TensorCore 外加入专用 SparseCore 数据流引擎：v4、v5p 和 Ironwood 每颗芯片 4 个，Trillium 每颗芯片 2 个。SparseCore 明确用于吸收脉动阵列并不擅长的嵌入查找负载。

所有模块都位于同一个 VLIW 发射平面上，由 Core Sequencer 每个周期填满一条 322 bit 指令束的全部八个功能槽。没有指令缓存未命中，没有 warp 调度器，没有乱序执行引擎，也没有分支预测器：编译器就是调度器，省下的硅片面积全部用于放置更多 MAC。

![TPU Ironwood / v8t 单封装平面图](/assets/images/posts/ai-chip-architectures-full/google-tpu-chip.png)

*图 5：TPU Ironwood / v8t 单封装平面图。两颗计算 Chiplet 跨越裸片间桥并排放置；每颗 Chiplet 含一个 TensorCore，顶部有两个 SparseCore 数据流引擎，两侧是 HBM3e 堆栈。ICI 端口沿上下边缘布置，用于构建 3D torus；右上角还有一个较小的 DCN NIC，负责横向扩展。*

![一个 TensorCore 的内部结构](/assets/images/posts/ai-chip-architectures-full/google-tpu-tensorcore.png)

*图 6：TensorCore 放大图。顶部标量单元每周期向八个功能槽发射一条 322 bit VLIW 指令束；VPU 通过二维向量 lane 执行逐元素运算；XLU 和转置/置换单元处理跨 lane 归约与布局重排；四个 256×256 MXU 执行脉动矩阵乘。累加器队列把部分和排入 VMEM——这个由软件管理的 scratchpad 负责向阵列供数并接收结果。*

#### TensorCore

MXU 就是脉动阵列。v1 配备一个 `256×256` INT8 推理阵列；v2 是第一代可训练 TPU，引入 `128×128` 单元，以 BF16 相乘、FP32 累加。从 v4 开始，INT8 以等效吞吐重新回到 MXU。此后每个 TensorCore 的 MXU 数量不断增加：v2 为 1 个，v3 为 2 个，v4/v5e/v5p 为 4 个。Trillium 又把阵列扩大到 `256×256`（每个阵列每周期有 65,536 个乘加单元），Ironwood、8t 和 8i 都保留这一形状。

计算 `C = A × B` 时，矩阵 B 的数值会预先加载，每个单元固定放置一个权重。这种**权重驻留（weight-stationary）**数据流正是 TPU 与其他输出驻留阵列的区别。激活值从左边缘进入，每周期向右传播一列，在每个单元中与驻留权重相乘；部分和则向下流入底部的累加器队列。

数据一旦进入阵列，就不再访问内存：每个权重被所有经过它的激活值重复使用，每个激活值沿一行复用 128 次（或 256 次）。数据复用直接接进硅片布线，不由缓存仲裁。计算中的主要成本并非乘法本身——它只消耗几皮焦——而是读写内存，单次访问的能耗可高出 100 至 1,000 倍；脉动阵列从结构上删除了这项成本。

代价是**阵列填充不足**：在 `256×256` 阵列上执行 `128×128` 矩阵乘会浪费 75% 的硅片。因此 [XLA](https://openxla.org/xla) 会执行分块、补齐和排程，使维度成为 128 的倍数（v6e 及以后为 256），模型代码也会围绕这些量级编写。

VPU 是次要计算引擎，却可能是微体系结构上更有趣的对象：每颗 TPU 都是一台二维向量机，而非一维 SIMD 机器。VPU 寄存器文件保存二维 VREG。在 v4/v5p 上，其形状为 `(8, 128)`：宽 128 个 lane、深 8 个 sublane；每个核心有 32 个（v4）或 64 个（v5p）寄存器，每个 `(lane, sublane)` 上有 4 个独立浮点 ALU。

lane 轴与脉动阵列输入宽度一致，因此在 Trillium 和 Ironwood 的 MXU 扩至 256 时，lane 数很可能也随之扩大到 256；Google 尚未公布 v5p 之后的 VPU 维度。sublane 轴使 VPU 能以每 X 个时钟执行一次矩阵乘的速度向 MXU 流送数据块，其中 X 是 sublane 维度。

现代 TPU 程序的大部分加速来自 **VPU/MXU 重叠**：MXU 在后台执行矩阵乘的同一批周期里，VPU 同时执行量化、layer norm、softmax、激活和偏置相加。跨 lane 归约——任何二维向量 ISA 都难以处理的情况——交给 XLU；它速度慢、成本高，是一个已知编译器热点。无法与二维形状对齐的布局变换则由专用转置/置换单元吸收，避免数据往返内存。

标量单元是最小的模块，却可能影响最大：它是一个单线程双发射整数 ALU，拥有 32 个 32 bit 寄存器、4 KiB 保存控制状态的 SMEM，以及一块存放程序的 Imem。它是唯一执行指令取出的模块。每个周期，标量单元取出一条 322 bit VLIW 指令束，在本地执行自己的两个标量槽（地址运算、循环计数器、分支、同步寄存器检查），并把剩余六个槽分派给芯片其他部分：2 个向量 ALU 槽（VPU）、2 个向量加载/存储槽（HBM↔VMEM DMA）、2 个矩阵槽（MXU 队列压入/弹出）。

模块间同步是显式的：同步标志记录 MXU 和 VPU 流水线是否繁忙，编译器插入屏障检查，而不是由硬件跟踪依赖关系。标量单元让 TensorCore 其余部分看起来像固定功能数据流：每个周期，由一个位置决定八件事情如何发生，也没有动态重排序缓冲区可以补救错误决策。

#### 内存

片上内存层次与计算侧遵循同一思想：**没有缓存，每一层都由软件管理**。片外为 HBM：v2/v5e 为 16 GB，v3/v4/v6e 为 32 GB，v5p 为 95 GB，Ironwood 为 192 GB，v8 代为 216 至 288 GB。片上则是显式寻址、手工叠放的多层 scratchpad。

最靠近计算的是 VMEM，即同时向 VPU 和 MXU 输入队列供数的向量 scratchpad：v4 为 32 MiB，v5e 为 128 MiB，面向推理的 v8i 更扩大到 384 MiB，目的正是把完整 KV Cache 放在片上。

VMEM 之上是 v4 引入的 CMEM，容量为 128 MiB。它是一块速度较慢、容量更大的 SRAM，在 HBM 与 VMEM 之间充当暂存区，吸收融合算子的中间值。标量单元有自己的 SMEM（v4 上用于控制状态的空间约 10 MiB）和很小的标量寄存器文件。

程序中的每个张量都会在编译时固定到其中一层。XLA 的缓冲区分配过程负责调度各层之间的 DMA，使数据在使用它的周期到来前刚好到达。硬件不预取、不淘汰，也不维护一致性。编译器做对时，阵列永不停顿；做错时，则没有备用路径。

#### SparseCore

TensorCore 外部、打破脉动阵列模式的模块是 v4 引入的 SparseCore。推荐和排序模型高度依赖嵌入查找——在巨型表中通过数十亿索引取值——其访问模式与稠密矩阵乘正好相反：不规则、间接，而且是全互连通信。`256×256` 脉动阵列完全不适合这种形状。

SparseCore 是一种数据流处理器，拥有 16 个计算块和专用 SPMEM scratchpad。它位于 TensorCore 旁边，负责 scatter、gather、分段归约原语，以及分片嵌入表产生的、依赖数据的 all-to-all 流量。它只占约 5% 的裸片面积和功耗，却能让嵌入密集模型加速 5 至 7 倍。v4 每颗芯片有 4 个 SparseCore，v5p 保持这一数量，Trillium 减为 2 个，Ironwood 又恢复到 4 个——双裸片布局中每颗 Chiplet 2 个。

v8i（Zebrafish）推理芯片完全移除 SparseCore，改在 I/O Chiplet 上放置集合通信加速引擎 CAE。它解决的是另一个问题——自回归解码中的集合归约——但理念相同：从主核心中切出一个小型加速器，专门吸收脉动阵列形状不适合的负载。

#### 数值格式

TPU v1 只支持 INT8 推理；v2 改以 BF16 作为标准训练格式：它拥有与 FP32 相同的动态范围、只占一半内存，也不需要损失缩放。v4 重新引入原生 INT8。Ironwood 随后加入原生 FP8（E4M3 和 E5M2），在相同面积内提供约两倍于 BF16 的吞吐。v8 又加入原生 FP4，并直接在 MXU 内执行块级缩放乘法，消除了 Ironwood 仍需承担的 VPU 反量化开销。

所有现代 TensorCore 都在硬件中支持**随机舍入**：由尾数较低位决定舍入概率，使低精度累加在长时间训练中的期望值保持不变。正是这类小细节，使 BF16/FP8 能缩小与 FP32 的精度差距。

芯片边界上是 ICI 端口本身：采用二维 torus 的 v2/v3/v5e/v6e 有 4 个端口；采用三维 torus 的旗舰 v4/v5p/v7/8t 有 6 个；旁边还有用于横向扩展的 DCN NIC。从芯片视角看，ICI 端口只是 Core Sequencer 能在 VLIW 指令束中寻址的另一组 DMA 引擎：发送远程张量与从 VMEM 传输到 HBM 属于同一指令类别，编译器把集合通信纳入它为计算和本地内存构建的同一份总时间表。

#### 五项押注

1. **脉动阵列。** 矩阵乘主导负载，因此把硅片面积投入脉动阵列。
2. **软件管理的 scratchpad。** 计算便宜、内存昂贵，因此在线路中复用数据，并以软件管理的 scratchpad 取代缓存。
3. **编译器排程。** 负载可以静态预测，因此把调度移入编译器：采用 VLIW 发射，不推测、不乱序，也不使用动态调度器。
4. **只为 MAC 使用硅片。** 功耗比峰值更重要，因此删除所有不执行乘加的晶体管：缓存标签、分支预测器、重排序缓冲区，一个不留。
5. **阵列外专用引擎。** 稠密矩阵乘阵列不适合某些真实负载（嵌入、集合通信），因此切出小型专用引擎 SparseCore、CAE，而不扭曲主核心迁就它们。

### 扩展

TPU 的纵向扩展思路与 NVIDIA 相反。NVLink + NVSwitch 让其他 GPU 的 HBM 看起来像本地内存——这是硬件管理的一致性地址空间；Google ICI 则使用**消息传递**。没有远程加载语义，没有缓存一致性，也没有交叉开关。每项多芯片操作都是由 [XLA](https://openxla.org/xla) 编译出的显式集合通信。纵向扩展域不是由交换网络连接，而是由 **torus** 连接：芯片直接与相邻芯片布线，并在边缘回绕；到了机架边界，再由光路交换机缝合。

**纵向扩展**通过 ICI 把芯片直接连接成二维或三维 torus。XLA 发出 SPMD 集合通信，紧密编排数千颗 TPU，使它们像一个程序一样运行。它不维护一致性，却能以低延迟提供巨大的二分带宽。

**横向扩展**通过数据中心网络把多个 Pod 连接起来：芯片数量远超单个 ICI 域所能容纳，但单芯片带宽更低。当前由 Virgo 处理东西向 TPU 流量（v8t 及以后），Jupiter 负责南北向流量；Multislice 和 Pathways 跨 Pod 编排 SPMD。

#### 纵向扩展

ICI 链路直接从 TPU 裸片引出：一个液冷机架内容纳 64 颗芯片，组成 `4×4×4` 立方体；立方体内部使用高速串行通道和直连铜缆，立方体之间使用光纤。单芯片 ICI 聚合带宽从 v2 的约 250 GB/s，提升到 Ironwood 的 1.2 TB/s 双向，再到 v8t 的两倍。

各代拓扑交替变化：面向能效的 v2、v3、v5e、v6e 使用二维 torus；旗舰 v4、v5p、v7、v8t 使用三维 torus。

没有 NVIDIA 对应物的是 **Palomar OCS**：位于立方体之间的 **3D-MEMS 光路交换机**。微型反射镜会实际旋转，把任意输入光纤映射到任意输出。一个 v4 SuperPod 使用 48 台 Palomar，把 64 个立方体（4,096 颗芯片）连成一个三维 torus；v5p 和 Ironwood 继续扩大同一方案。

重配置耗时为毫秒级，而非纳秒级，但这并不构成问题，因为 OCS 使用**电路交换**：作业开始时选定拓扑，让它运行一周，再为下一项负载重新配置。同一组件把三个问题合并解决：按负载重构拓扑——扭曲 torus 可使二分带宽提升多达 70%；按需切分子 Pod；以及**容错**——芯片失效时，OCS 通过光路换入备用立方体，使作业在不丢失 ICI 域的情况下继续运行。

![TPU Ironwood SuperPod](/assets/images/posts/ai-chip-architectures-full/google-tpu-scale-up.png)

*图 7：TPU Ironwood SuperPod。左侧为一个 64 芯片立方体（4×4×4），最近邻之间由直连铜缆构成三维 torus，每个面都有边缘回绕；右侧为 144 个立方体，由 Palomar OCS——可按负载重构拓扑的 3D-MEMS 光路交换机——缝合为一个一致的 ICI 域。*

这样一来，SuperPod 成为纵向扩展的基本单位：其作用相当于 NVIDIA NVL72，规模却大两个数量级。v4 包含 4,096 颗芯片，v5p 包含 8,960 颗；Ironwood（TPU v7）则由 144 个、每个 64 颗芯片的立方体组成，共 9,216 颗芯片，对外呈现为一个一致 ICI 域：拥有 1.77 PB HBM（约 68 PB/s）和 42.5 ExaFLOPS FP8。

TPU 8t（Sunfish）进一步扩大到 9,600 颗芯片、2 PB HBM（约 62 PB/s）和 121 ExaFLOPS FP4。TPU 8i（Zebrafish）拥有 1,024 颗芯片、约 295 TB HBM（8.8 PB/s）和约 10 ExaFLOPS FP4。

8i 放弃 torus，改用称为 **Boardfly** 的分层高基数拓扑：4 芯片环 → 8 基板组 → 最多 36 个由 OCS 连接的组，使 all-to-all 延迟减半。它专为 MoE 推理设计。三维 torus 适合最近邻集合通信——环形 all-reduce 可让每条链路每周期满载；MoE 专家路由却完全相反，是 all-to-all：每颗芯片都向其他芯片发送不同片段，往返延迟由跳数最长的一对芯片决定。1,024 芯片三维 torus 的直径为 16 跳；Boardfly 的“环 → 组 → OCS”层次把它压缩到 7 跳。

#### 横向扩展

截至 TPU v7，横向扩展都运行在同一套网络上：**Jupiter**。从 2022 年开始，其主干已通过 **Apollo OCS** 实现全光化；Apollo 与 Palomar 属于同一 3D-MEMS 系列，只是扩展到整栋数据中心。Google 从机架到数据中心主干的每一层都使用同一种原语——光路交换——这是其他厂商没有的架构标志。当前每栋建筑中的 Jupiter 可提供 13 Pb/s 二分带宽。

到了 TPU 8t，横向扩展拆成两套网络。东西向 TPU 到 TPU 流量转移到专用加速器网络 **Virgo**；Jupiter 保留南北向职责：存储访问、通用计算和跨站点扩展。

Virgo 是由高基数交换机构成的平坦两层无阻塞拓扑：任意两颗 TPU 之间最多经过两台交换机。一个 Virgo 集群可连接超过 134,000 颗 TPU 8t，二分带宽达 47 Pb/s；相比上一代 DCN，单芯片带宽提高 4 倍，空载延迟降低 40%。它还具有多平面故障隔离和亚毫秒遥测，调度器可在拖慢整个训练步骤之前终止落后节点。其架构收益在于，每一层都能独立演进：纵向扩展、东西向横向扩展和前端网络可以按不同节奏迭代，无需彼此重新布线。

![TPU 8t 横向扩展](/assets/images/posts/ai-chip-architectures-full/google-tpu-scale-out.png)

*图 8：TPU 8t 横向扩展。东西向 TPU 流量经过 Virgo——由高基数交换机组成的平坦两层无阻塞网络，任意 TPU 之间最多两次交换跳转，可连接超过 134,000 颗 TPU，二分带宽 47 Pb/s。存储、通用计算和跨站点等南北向流量仍走 Jupiter；自 2022 年起，Jupiter 主干通过 Apollo OCS 实现全光化。*

Ironwood 的单芯片横向扩展带宽约为 100 Gbps，v8t 约为其 4 倍，但仍比单芯片 ICI 低两个数量级。这一带宽差决定了分区方式：张量并行和 MoE 专家路由留在 ICI 内；数据并行和流水线并行跨横向扩展网络。

Google 的 Multislice 框架已经接入 [XLA](https://openxla.org/xla)，使单一 SPMD 程序可以跨越不同 Pod 中的多个 slice。编译器会发出分层集合通信：每个 slice 内执行环形 all-reduce，再在更高层跨 slice 归约。这正是隐藏 ICI/DCN 带宽差距的方法：尽可能让工作留在 slice 内的高速 ICI 上，只让跨 slice 的剩余部分支付慢速网络成本。

再上一层是 Pathways。NCCL + Slurm + Megatron 式调度器由多个控制器驱动 SPMD；Pathways 则由**一个**客户端驱动整项作业，并虚拟化多个“岛屿”——拥有各自 ICI 域、通过 DCN 相连的 Pod。它负责成组调度、弹性训练（slice 失效时由 OCS 重塑拓扑，Pathways 在新拓扑上从最近检查点恢复）和跨区域编排。Gemini Ultra 是第一个跨多个数据中心训练的前沿模型，Pathways 把它们缝合为一个同步 SPMD 作业。

这套哲学可以概括为：**编译器就是调度器，torus 就是拓扑，光学交换机则是从机架到数据中心每一层通用的可重构衬底。**

### 软件

CUDA 以 kernel 为驱动，TPU 软件栈则以编译器为驱动。在 GPU 上，开发者编写 kernel，框架把 kernel 串联起来，编译器的工作大多是局部的。在 TPU 上，开发者使用 [JAX](https://github.com/jax-ml/jax) 编写数值程序；其下方的一切都由 [XLA](https://openxla.org/xla) 负责：哪些操作融合，每个张量存放在哪里，怎样布局到二维向量寄存器上，HBM 到 VMEM 的 DMA 何时发出，322 bit VLIW 指令束如何排程，以及程序怎样分片到数千颗芯片。

硬件没有兜底机制：没有 warp 调度器、缓存或乱序执行引擎来掩盖一份糟糕的时间表。编译器就是整个系统。这个架构最核心的取舍是：**XLA 无需手工调优便能更接近理论上限，但要弥合最后的差距也更困难。**

编译路径是：`JAX → JAXpr → StableHLO → HLO → LLO → VLIW 指令束`。JAX 在 `jit` 下把 Python 函数追踪为带类型的函数式 IR（JAXpr），再降低为 StableHLO——所有前端如今都会发出的、由 OpenXLA 标准化并进行版本管理、包含约 100 种静态形状原语的操作集。

XLA 把 StableHLO 作为 HLO 输入，依次运行各项 pass：**操作融合**把逐元素、归约和矩阵乘合成一个 kernel，使中间值不进入 HBM；**布局分配**决定每个张量的二维分块，使其无需转置即可流入 MXU——这比一维 SIMD 机器困难得多，因为寄存器和脉动阵列输入都是二维的；**缓冲区分配**把每个张量固定在 VMEM、CMEM 或 HBM，并预先计算重叠窗口；之后执行 SPMD 分区，最后由 VLIW 调度器填满每条指令束的八个槽位。

HLO 再降低为 TPU 专用 IR LLO（Low-Level Optimizer），由 LLO 发出最终 VLIW 流。编译良好的程序会在每个周期的同一条指令束中，同时重叠 MXU 脉动执行、VPU 逐元素运算和 HBM↔VMEM DMA。

多芯片执行采用 SPMD：一份程序、分片数据、分层集合通信，由 GSPMD 发出。GSPMD 正被基于 MLIR 的继任者 Shardy 取代，后者会在 2026 年初成为默认方案。用户只需在少数关键张量上通过 Mesh + PartitionSpec 声明分片；编译器会把分片传播到图的其他部分，并在布局变化处插入 all-reduce、all-gather 和 reduce-scatter。

当编译器选择了错误的集合通信时，`shard_map` 允许用户进入**手工 SPMD**：编写带显式本地形状和显式集合操作的逐设备代码；它又能组合在 `jit` 内部，所以只需手工分区一个 kernel，而不必放弃其他部分的自动分区。这与 PyTorch 惯用法相反：FSDP 和 DeepSpeed 以运行时包装模型，在模块边界发起集合通信；GSPMD/Shardy 则把全图分区视作编译器问题。

Pallas 是“逃生通道”：JAX 的 kernel 编写语言，大体相当于 GPU 上的 Triton。Pallas kernel 使用 JAX 风格的 Python 编写，通过基于 MLIR 的 TPU 后端 Mosaic 降低到 LLO，再作为自定义操作嵌回 HLO。它之所以存在，是因为 XLA 并非总能为新型注意力变体、融合 MoE 分派或需要手工 VMEM 分块与 DMA 排程的操作综合出最优方案——例如 FlashAttention 级优化，其优势来自排程，而不是代数形式。

Pallas:Mosaic-GPU 还能用同一个前端面向 H100/Blackwell，因此 kernel 作者可以只写一次，再降低到任一硬件衬底。更上层的库统一采用 JAX：Flax NNX 用于模块，Optax 用于优化器，Orbax 用于异步分布式检查点，Grain 用于输入流水线，Tunix 用于后训练/强化学习，Qwix 用于量化。Google 的参考训练栈位于最上层，包括用于 LLM（含 DeepSeek-V3 级 MoE）的 MaxText 和用于 Flux、Wan 2.1 的 MaxDiffusion，二者都使用纯 JAX。Pathways 位于其下，以 `pathwaysutils` 暴露给用户，让一个 Python 客户端能驱动跨数千颗芯片和多个 Pod 岛屿的作业，同时不脱离 JAX 编程模型。

PyTorch 路径确实存在，但处于次要地位。`torch_xla` 使用 LazyTensor 机制：每个 PyTorch 操作记录到 HLO 图中，在下一道屏障处编译，编译产物按图形状哈希缓存。PyTorch/XLA 2.x 加入 GSPMD 式分片注解、通过 XLA 后端集成 `torch.compile`、JAX 桥，以及在 PyTorch/XLA 2.7 中加入 C++11 ABI 构建并显著加快追踪。

与 JAX 的差距仍然存在——JAX 原语能更自然地映射到 StableHLO，复杂并行策略的覆盖也更完善。因此，vLLM TPU（由 Cloud Next 2025 发布的 `tpu-inference` 插件驱动）会把每个模型——无论使用 JAX 还是 PyTorch 定义——统一降低到 `JAX → XLA` 路径。

2026 年 4 月发布的 TorchTPU 是 Google 的回应：它提供原生 PyTorch 体验，在 XLA 上支持 eager mode、`torch.distributed` 和 `torch.compile`，并计划取代 `torch_xla`。

与 CUDA 相比，TPU 生态是**中心化的，而非繁杂蔓生的**。框架以下几乎所有组件——XLA、JAX、Flax、Optax、Pallas、MaxText、Pathways、Shardy、Mosaic——都由 Google 自己开源，并与硅片同步演进。第三方 kernel 远少于 CUDA 数十年的积累；当负载形态古怪时，护城河较浅，而当负载像 Gemini 时，护城河则更深。

近期 Ironwood（v7）使用“协同设计 AI 栈”的说法，正是对这套模式的明确表述：芯片、ICI 网络、OCS、XLA、Pathways、Pallas、MaxText、vLLM 和 Pathways 作为一个产品共同发布；v8t/v8i 延续相同模式，统一由 `tpu-inference` 路径降低。

NVIDIA 侧的 Triton 和 `torch.compile` 正在缩小差距——kernel 驱动与编译器驱动正在趋同——但两端的哲学差异依然真实：**在 TPU 上，编译器是唯一重要的接口；在 GPU 上，编译器只是多个接口之一。**

---

## AMD GPU

> **设计哲学**：**[AMD Instinct GPU](https://www.amd.com/en/products/accelerators/instinct.html)** 的押注与 NVIDIA 不同。NVIDIA 每一代都扩展单个 SM **能够做什么**，AMD 则从 GCN（2012）开始一直保守地维持**计算单元（Compute Unit，CU）**，把投入转向封装：自 2021 年以来，每一代产品的 HBM 容量都追平或超过同期 NVIDIA 旗舰；推出首款采用 **3D 堆叠**的数据中心 GPU（CDNA 3）、首款一致性 **CPU+GPU APU**（MI300A），并押注**开放生态**（ROCm、HIP、OCP MX、UALink）。

### 演进谱系

| 年份 | 架构与芯片 | 关键变化 |
| --- | --- | --- |
| 2018 | **[Vega 20](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/specs/radeon-instinct-mi50-data-sheet.pdf) MI50、MI60** | 首款 7 nm GPU；FP64 向量吞吐为 FP32 的 1/2。Instinct 最后一代 GCN 家族产品，此后分化为 CDNA / RDNA。 |
| 2020 | **[CDNA](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf) MI100** | 首次加入 MFMA 矩阵核心；彻底移除图形固定功能硅片；原生支持 BF16。 |
| 2021 | **[CDNA 2](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf) MI210、MI250、MI250X** | 通过双 GCD 封装推出首款 MCM Instinct；全速 FP64 矩阵计算。 |
| 2023 | **[CDNA 3](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) MI300A、MI300X** | 首款 3D 堆叠 Chiplet GPU：XCD 通过 TSV 混合键合到 IOD；加入 FP8 和 Infinity Cache；MI300A 实现一致性 CPU+GPU APU；驱动 El Capitan。 |
| 2024 | **CDNA 3 更新版 MI325X** | 计算部分不变，升级 HBM3E：容量 256 GB、带宽 6.0 TB/s。 |
| 2025 | **[CDNA 4](https://www.amd.com/en/products/accelerators/instinct/mi350.html) MI350X、MI355X** | 原生支持采用 OCP MX 微缩放的 **FP4 / FP6**；每 CU FP64 约减半；首次明显从 HPC 转向 AI 密度。 |
| 2026 | **[CDNA Next](https://www.amd.com/en/blogs/2025/amd-advancing-ai-2025-mi400-helios-rack-scale-ai-platform.html) MI430X、MI440X、MI455X** | HBM4；Helios 机架（72 颗 MI455X，首发通过 UALoE、2027 年起原生 UALink）：AMD 首次正面回应 NVL72。 |

### 体系结构

术语对照：

| AMD | NVIDIA |
| --- | --- |
| Compute Unit（CU，计算单元） | Streaming Multiprocessor（SM，流式多处理器） |
| SIMD | SM 子分区 |
| SIMD lane | CUDA Core（FP32 ALU） |
| Wavefront（wave64） | Warp（warp32） |
| Matrix Core | Tensor Core |
| MFMA | `mma.sync` / `wgmma` / `tcgen05.mma` |
| VGPR / SGPR | 寄存器文件 |
| LDS（Local Data Share） | SMEM（共享内存） |
| Infinity Fabric | NVLink |

NVIDIA 的架构野心体现在每个 SM **内部**——每代都加入新的张量原语、异步机制和操作数存储；AMD 的野心则位于 CU **之间**，体现在能把多少个 CU 绑定成一个一致封装。CU 本身很保守：四组 16-lane SIMD、一个共享标量单元、64 KB Local Data Share、一个 L1 向量缓存、每个 SIMD 一组 VGPR 文件和 CU 共享的 SGPR 池，以及从 CDNA 1 开始、运行 MFMA 的 Matrix Core。

自 2012 年 GCN 以来，其形状没有发生实质变化；扩大的只是数量——MI100 为 120 个 CU，MI250X 为 220 个，MI300X 为 304 个，MI355X 为 256 个——以及把它们连接起来的封装方式。一组 64 线程 wavefront 会在 4 个周期内流过 16 条 SIMD lane；每个 SIMD 同时驻留许多 wavefront，调度器在它们之间切换以隐藏停顿。CU 内部没有什么奇异之处；CDNA 真正有趣的地方，全在 CU 之外。

![AMD Instinct MI355X（CDNA 4）封装平面图](/assets/images/posts/ai-chip-architectures-full/amd-gpu-chip.png)

*图 9：AMD Instinct MI355X（CDNA 4）封装平面图。8 颗 XCD（Accelerator Complex Die，每颗约 32 个活跃 CU）通过 TSMC SoIC 混合键合到下方两颗 IOD 基础裸片。IOD 承载 256 MB Infinity Cache（每颗 128 MB）、HBM PHY、Infinity Fabric 和 PCIe Gen 5。外围排列 8 组 12-Hi HBM3E，总容量 288 GB。*

![一个 AMD Compute Unit 的内部结构](/assets/images/posts/ai-chip-architectures-full/amd-cu.png)

*图 10：单个 Compute Unit 放大图。一个调度器在四个周期内，把 wave64 wavefront 分派到四组 SIMD16 向量引擎；每组 SIMD 旁都有一个执行矩阵乘的 Matrix Core（MFMA）。共享标量单元、三类寄存器文件（VGPR / SGPR / AGPR）、160 KB LDS scratchpad 和 32 KB L1 向量缓存构成完整配方——整体形状自 2012 年 GCN 以来一直延续。*

#### 计算

CU 内部，SIMD 与 Matrix Core 并行工作。四组 SIMD 处理所有逐元素操作：激活、归一化、残差、地址运算；Matrix Core 处理矩阵乘。这与 NVIDIA 的 CUDA Core / Tensor Core 分工相同，但矩阵抽象沿着完全不同的路线演进。

NVIDIA 的 Tensor Core 不断沿线程层级上移：Volta 由 32 线程 warp 驱动，Hopper 由 128 线程 warp-group 驱动，Blackwell 则由单个线程发起，还可选择跨两个 SM 的 cluster。AMD 的 Matrix Core 始终停留在原处。从 2020 年 MI100 到 2025 年 MI355X，每一代 MFMA 都以 wavefront 为作用域：一个 wave64 发出一条矩阵指令（`V_MFMA_*`），四组 SIMD 协同驱动它；操作数来自 wavefront 的寄存器文件——A、B 来自 VGPR，C、D 通常来自专用 AGPR 文件。指令变快、格式集合变宽，但发射者和作用域都没有改变。

CDNA 4 在供数端做出了一项让步：加入专用的 **LDS→MFMA 转置加载**，把操作数以 Matrix Core 所需布局直接交给它；其精神类似 NVIDIA TMA，但矩阵操作本身仍由 wave 发出。

吞吐数字直接讲述了格式演进。CDNA 1 在 2020 年以 FP32 / FP16 / BF16 / INT8 上市，每 CU 每周期分别达到 256 / 1,024 / 512 / 1,024 FLOPs，并与 A100 同期原生支持 BF16。CDNA 2 把 FP64 路径翻倍为每 CU 每周期 256 FLOPs 的全速矩阵运算——这是 AMD 独有的押注，也让 MI250X 进入 Frontier 超级计算机。CDNA 3 在 FP8 上达到与 H100 相当的 4,096 FLOPs（E4M3 + E5M2），加入 2:4 结构化稀疏，并提供等效 TF32 路径：截断尾数，以 FP64 矩阵速率执行 FP32 矩阵乘。

CDNA 4 再次翻倍，以 FP4 达到 16,384 FLOPs，并支持采用 OCP MX 块缩放的 FP6；同一条 MFMA 还可以混用 A/B 精度，例如 FP8 × FP4。与此同时，这一代把每 CU FP64 吞吐减半——AMD 首次选择以 HPC 密度换取 AI 密度，而不是二者兼得。

以 wavefront 为作用域的决定带来两项成本。

**分歧（divergence）。** 半空的 wave64 会浪费 32 条 lane，半空的 warp32 只浪费 16 条。对控制流大多一致的负载，这个代价很小；对不规则负载，影响则很明显。

**重叠（overlap）。** NVIDIA 采用异步、描述符驱动的矩阵乘，把“发射”与“执行”解耦：发射线程触发指令后即可继续，Tensor Core 在后台运行；当前一轮矩阵乘尚未结束时，warp 就能执行 softmax、应用 mask，或预加载下一块数据。AMD 的 wavefront 集体 MFMA 没有等效能力：发出矩阵乘的同一 wave 在等待期间无法同时做有意义的向量工作。可以让**不同** wavefront 相互重叠，但必须在软件中以显式 wavefront 屏障分阶段安排；这更加脆弱，也会占用更多 wave 槽位和寄存器。

影响大小取决于负载。**纯稠密 GEMM**（DGEMM、大批量训练内环）在矩阵乘期间没有其他有用工作可做；两种引擎都能饱和，异步能力收益很小。这恰好是 AMD 历来领先百亿亿次级 HPC 的负载：Frontier 使用 MI250X，El Capitan 使用 MI300A。

**Transformer 注意力**（FlashAttention-3、FA4）会交错执行矩阵乘、softmax、mask 和 KV Cache 读取，异步重叠就是 kernel 的整体结构。AMD 必须手工重建这条流水线，因此落后于 NVIDIA 的硬件级支持。**MoE 分派、分页注意力、推测解码**也属于同类：它们都是希望在矩阵乘旁并行执行的地址不规则工作。

NVIDIA 的矩阵指令抽象跨代移动得更远（warp → warp-group → 单线程异步 + cluster），AMD 并未跟进。

#### 内存

AMD 的通用内存层次比 NVIDIA **更少**，但拥有一层 NVIDIA 完全没有的巨大缓存。从 CU 向外依次是：64 KB LDS scratchpad（软件管理、32 个 bank，相当于 NVIDIA SMEM）、向量 L1（早期 CDNA 为 16 KB，MI300X 起为 32 KB），以及每颗 XCD 数 MB 的 L2。不过 L2 在不同 XCD 之间并不一致；一致性发生在 L2 的上一层。

这一层就是 **Infinity Cache**：MI300X 上为 256 MB，分布在四颗 IOD 上，16 路组相联；实测带宽约 12 TB/s，是 MI300X 5.3 TB/s HBM3 带宽的两倍以上。它最初出现在 RDNA 游戏 GPU 中，用于弥补较窄的 GDDR 总线；AMD 在 CDNA 3 上复用该 IP 服务 AI，而注意力的 KV 复用与权重复用恰好非常适合大容量末级缓存。NVIDIA 押注更大的 HBM 带宽（B200 为 8 TB/s，Rubin 随 HBM4 继续提升），AMD 则押注缓存。

片外 HBM 容量快速增长：MI100 / MI210 / MI250X / MI300X / MI325X / MI350X 依次为 32 / 64 / 128 / 192 / 256 / 288 GB。自 2021 年起，每一代都追平或超过同期 NVIDIA 旗舰。其判断是：推理负载越来越受容量限制，内存更多的芯片就会胜出。

#### 数值格式

格式演进遵循所有 AI 硅片共有的精度折半路线：FP32 → FP16 → FP8 → FP4；每进一步，就用粒度更细的缩放恢复精度。AMD 特有的维度是**开放性**。CDNA 4 的 FP4 和 FP6 使用 **OCP MX 块缩放乘法**：它与 Blackwell 的 MXFP4、TPU v8 的 MXU 使用同一种数值格式，但由 AMD、NVIDIA、Intel、Meta、Microsoft、Qualcomm、ARM 等共同组成的开放联盟制定，AMD 也是创始成员，而非由任何单一厂商所有。MI355X、B200 与 TPU v8 实际采用的是同一种格式。

CDNA 4 的转折值得单独强调：每 CU FP64 吞吐减半。MI300X 同时服务训练、HPC 和推理；MI355X 首先是一颗 AI 芯片。驱动 Frontier 的全速 FP64 矩阵押注并未消失，但已经不再承担全部重任。

#### Chiplet

封装正是 CDNA 开始不像 NVIDIA、转而成为另一种东西的地方。

CDNA 1 的 MI100 是单片式 7 nm 芯片。CDNA 2 的 MI250X 是 AMD 首款多芯片 GPU：两颗 Aldebaran GCD 并排放在 2.5D EFB 有机基板上，通过 4 条封装内 Infinity Fabric 链路连接，聚合带宽 400 GB/s，但在软件中仍呈现为两块独立 GPU。

CDNA 3 是改变一切的一步。8 颗 **XCD**（TSMC N5，每颗约 115 mm²）通过 **TSMC SoIC** 混合键合，以亚微米间距 **TSV**、不使用微凸点，3D 堆叠到下方 4 颗 **I/O Die**（TSMC N6）。IOD 承载 Infinity Cache、HBM3 PHY、Infinity Fabric 链路和 PCIe Gen 5；每颗 IOD 上方承载两颗 XCD、旁边连接两组 HBM。

4 颗 IOD 通过二分带宽 4.8 TB/s 的 **Infinity Fabric AP** 缝合，因此这个拥有 1530 亿个晶体管的封装，对 kernel 来说就是一块 GPU：缓存和地址空间在 IOD 层统一。NVIDIA 在 H100 之前一直使用单片设计，到了 B200 才通过 2.5D CoWoS-L 转向两颗光刻掩模极限尺寸裸片。AMD 提前一代、以更小的单裸片面积进入 3D 堆叠；两家公司在同一封装前沿做出了不同押注。

**MI300A APU** 又向前迈了一步：把 8 颗 XCD 中的 2 颗替换为 3 颗 Zen 4 **CCD**，保留 HBM、Infinity Cache 和 IOD，让 CPU 与 GPU 共享由 HBM3 支撑、具备硬件一致性的同一物理地址空间。这里没有主机—设备复制，没有固定页内存，路径中也没有 PCIe。Zen 4 核心和 CDNA 3 XCD 直接读取同一批页面。NVIDIA Grace-Hopper 通过 NVLink-C2C 连接的是**两个**封装，MI300A 却只有**一个**。由 11,039 个、每节点 4 颗 MI300A 构成的 **El Capitan**，就是这项设计得到正当性的部署证明。

在 CDNA 4 的 MI355X 中，8 颗 XCD 仍通过 SoIC 3D 堆叠到基础裸片，但 XCD 转向 TSMC N3P，每颗有 32 个活跃 CU（共 256 个；MI300X 为 304 个）。每 XCD 的 CU 数量下降，是为了给更大的 Matrix Core 和 160 KB LDS 腾出面积。MI300X 的 4 颗 IOD 合并为 2 颗，每颗在 TSMC N6 上做得宽一倍，上方承载 4 颗 XCD，旁边连接 4 组 HBM3E。

每颗 IOD 现在拥有 256 MB Infinity Cache 中自己的 128 MB 分片、一半 HBM PHY、相应的 Infinity Fabric 链路和 PCIe Gen 5。两颗 IOD 之间的 Infinity Fabric AP 二分带宽为 5.5 TB/s，比 CDNA 3 高约 15%；8 组内存改为 12-Hi HBM3E，在引脚数不变时达到 288 GB、8 TB/s，容量比 MI300X 高 50%。整个封装拥有 1850 亿个晶体管，对 kernel 仍呈现为一块 GPU。

#### 五项押注

1. **先 HPC，后 AI。** HPC 与 AI 在某个阶段是同一项押注，直到它们不再相同：CDNA 2 到 CDNA 3 持续提供全速 FP64 矩阵计算；当推理经济性明确偏向低精度后，在 CDNA 4 分岔。
2. **内存容量。** 自 2021 年起，每一代 HBM 容量追平或超过同期 NVIDIA 旗舰，并加入 256 MB 末级 Infinity Cache，吸收 H100 必须访问 HBM 才能获得的复用。
3. **较早采用 3D 堆叠。** 在 NVIDIA 之前，把计算单元 3D 堆叠到缓存与 I/O 之上：2023 年就以 TSMC SoIC 把 XCD 混合键合到 IOD，当时 NVIDIA 仍为单片设计。
4. **一致性 CPU+GPU。** MI300A APU 是已经交付的产品中最激进的 Chiplet 方案，El Capitan 则是证明。
5. **开放的纵向扩展互连。** 选择 UALink 和 OCP MX，而不是 NVLink 与专有 FP4。

### 扩展

内存押注带来了扩展层面的后果：8 颗 MI300X 拥有 1.5 TB HBM，8 颗 MI350X 拥有 2.3 TB，因此一个 405B 参数模型可以 FP8 完整放进单台 8-GPU 机器——权重、KV Cache，以及为更长上下文和更大批次保留的余量都包括在内；同一模型在 8×H100（640 GB）上则需要精细分片。

对 2024—2025 年的推理负载，AMD 无需在机架层让纵向扩展追平 NVL72，只要单机有竞争力即可。但在前沿**训练**中，它必须做到这一点，而 AMD 直到 2026 年才给出答案。

**纵向扩展**：通过 Infinity Fabric 把 GPU 绑定为一个一致内存域。到 MI355X 为止，规模止于 8-GPU OAM 机器（每 GPU 896 GB/s mesh）；Helios 则通过 UALink 扩大到 72-GPU 机架，首发时通过以太网隧道 UALoE，2027 年起原生支持。

**横向扩展**：通过以太网连接这些域，不使用 InfiniBand。Pensando NIC（Pollara 400、Vulcano 800）实现 Ultra Ethernet Consortium 的 UET RDMA 传输；Broadcom Tomahawk 6 提供交换芯片和 CPO。

#### 纵向扩展

截至 MI355X，AMD 的纵向扩展意味着通过 Infinity Fabric 连接的 **8-GPU OAM 平台**。每颗 MI300X 拥有 7 条 IF 链路——分别直连机器内每个对端——每条双向 128 GB/s，因此在全互连 all-to-all 拓扑中，每 GPU mesh 带宽为 896 GB/s。MI350X 把每条链路提高到 153.6 GB/s，每 GPU 约 1,075 GB/s，但仍保持 8-GPU 形状。

平台符合 OCP UBB 2.0，机械插槽与 NVIDIA HGX 基板相同，因此服务器厂商无需重新设计整机，就能在同一机箱中交付 AMD 或 NVIDIA。

截至 MI355X，AMD 尚未交付与 NVL72 对应的机架级系统。在 MI300X 集群上运行更大模型的客户，只能通过以太网跨多台 8-GPU 服务器扩展，为 NVIDIA 用户可以留在纵向扩展域内的通信支付横向扩展延迟。这是训练中真正重要的差距，也正是 **Helios** 要填补的空白。

![AMD Helios 机架级系统](/assets/images/posts/ai-chip-architectures-full/amd-scale-up.png)

*图 11：AMD Helios。72 颗 MI455X GPU 位于 Open Rack Wide 机箱中一排 UALink 交换机下方，组成一个一致的 UALink 内存域。首发时互连使用 UALoE（Infinity Fabric 经以太网隧道传输）作为过渡，等待 2027 年原生 UALink 交换芯片交付；每颗 GPU 对外连接一块 Pensando Vulcano 800 NIC。*

Helios 是 AMD 首个机架级纵向扩展域，将于 2026 年下半年随 MI455X 交付：每机架 72 颗 GPU、约 31 TB HBM4、1.4 PB/s 聚合 HBM 带宽、2.9 ExaFLOPS FP4 / 1.4 ExaFLOPS FP8、260 TB/s 纵向扩展带宽、43 TB/s 横向扩展带宽。

其形态采用 **Open Rack Wide（ORW）**——Meta 在 2025 年提交给 OCP 的双宽液冷设计——而非 AMD 私有机箱。基于 Meta 参考设计而不是从头设计机架，是 AMD 的有意押注：任何已标准化采用 ORW 的超大规模云厂商，都能部署 Helios，而无需为数据中心设施开展定制工程。

互连是 **UALink**（Ultra Accelerator Link）。这是 AMD 与 Apple、AWS、Cisco、Google、HPE、Intel、Meta、Microsoft 和 Synopsys 等共同创建的开放联盟标准。2025 年 4 月发布的 UALink 200G 1.0 定义 200 GT/s lane、每方向 800 Gbps，交换拓扑可扩展到每 Pod 1,024 个加速器。它承诺提供与 NVLink 相当、但不属于任何一家公司的缓存一致互连：任何厂商都能制造 UALink 交换机，任何加速器都能使用 UALink，标准归联盟所有，而非归销量最大的供应商所有。

问题在于：**原生 UALink 交换芯片要到 2027 年才能大规模出货**。Astera Labs Scorpio 以及 Auradine、Enfabrica、Xconn 的竞品，都瞄准 2026 年末至 2027 年部署。Helios 首发时以 **UALoE**（Infinity Fabric 经标准以太网隧道传输）过渡：在等待原生 UALink fabric 的同时保留编程模型；原生交换将在 2027 年随 MI500 到来。

所以首发 Helios 更接近一个高速、通过以太网隧道实现一致性的集群，而非 NVL72 真正的缓存一致 NVLink 域。它在时间线上做出了实质让步，代价换来的是在 2026 年下半年及时交付具有竞争力的产品。

#### 横向扩展

AMD 不提供 InfiniBand。其整个横向扩展栈都基于以太网，并锚定另一项开放标准：**Ultra Ethernet Consortium（UEC）**。

2025 年 6 月发布的 UEC 1.0 定义了 **Ultra Ethernet Transport（UET）**：一种运行于标准以太网之上的新 RDMA 传输，支持分组喷洒、基于 SACK 的选择性重传和现代拥塞控制。UET 不是 RoCEv2——后者把 InfiniBand 传输封装进以太网帧——而是为 AI 横向扩展 fabric 从头重构 RDMA 语义。AMD 与 Broadcom、Cisco、Meta、Microsoft 同为创始成员。策略与 UALink 相同：拥有标准，而不是独占实现。

![AMD 横向扩展网络](/assets/images/posts/ai-chip-architectures-full/amd-scale-out.png)

*图 12：AMD 横向扩展。Helios 机架通过基于开放 Ultra Ethernet（UEC）标准的普通以太网彼此通信。UET 以重新设计的 RDMA 语义取代 RoCEv2。每颗 GPU 配备 Pensando Vulcano 800 NIC（PCIe Gen 6、800 GbE、UEC 1.0），机架间交换采用配备共封装光学的 Broadcom Tomahawk 6。AMD 拥有 NIC 层，交换机和光学层则使用合作伙伴芯片。*

NIC 来自 AMD 于 2022 年收购的网络创业公司 **Pensando**。**Pollara 400** 是当前 AI NIC：400 GbE、P4 可编程、支持 UEC、PCIe Gen 5，与 MI300X / MI355X 搭配。**Vulcano 800** 将于 2026 年随 MI455X 交付：符合 UEC 1.0、采用 PCIe Gen 6、提供原生 UALink 接口，每 GPU 横向扩展带宽是 Pollara 的 8 倍。

**Salina 400** 则是前端 DPU：16 个 Arm Neoverse-N1 核心、双 400 GbE，负责存储、SDN 与防火墙，相当于 NVIDIA BlueField；它不同于 AI 后端 NIC。

不过交换芯片并非 AMD 所有。Helios 的 43 TB/s 横向扩展 fabric 通过 **Broadcom Tomahawk 6**：这是一颗 102.4 Tbps 以太网交换 ASIC，配备名为“Davisson”的共封装光学。AMD 没有自有 CPO，也没有自有交换 ASIC；光学层依赖合作伙伴芯片。NVIDIA 则拥有完整堆栈：InfiniBand、Spectrum-X Ethernet、ConnectX、BlueField、Quantum-X Photonics CPO，全都来自内部。

AMD 只拥有其中一层——通过 Pensando 获得 NIC + DPU——并押注开放标准与最佳合作伙伴芯片组合的演进速度会超过垂直整合。

行业方向已经向 AMD 靠拢。Dell'Oro 报告称，2025 年以太网承载的 AI 横向扩展 fabric 数量超过 InfiniBand 两倍；AWS、Microsoft、Meta、Oracle 和 xAI 都已为其基于 AMD 的 AI 集群标准化采用以太网。剩下的问题已不再是以太网能否在 RDMA 语义上追平 InfiniBand——UEC 正在弥合差距——而是 Helios 能否足够快地追上 NVL72 的机架级能力，从而赢得如今仍默认选择 NVIDIA 的前沿训练负载。

### 软件

**[ROCm](https://rocm.docs.amd.com/)** 是开源的 **[CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)** 对照物。NVIDIA 软件栈专有且垂直整合——cuBLAS、cuDNN、TensorRT-LLM 都以仅由 NVIDIA 维护的二进制 blob 交付；ROCm 则原生立足 GitHub，押注 PyTorch、Triton、vLLM、OCP MX 等开放标准，而非围墙式函数库集合。AMD 与 NVIDIA 的软件差距确实存在，但 AMD 的策略是通过开放社区弥合它，而不是从头建立一套平行的 CUDA 栈。

底层是 **HIP**，即 AMD 的 CUDA 兼容 C++ 运行时。**hipify** 可自动把 CUDA 源码转换成 HIP。大批量 HPC 代码——HACC、Laghos、QMCPack——开箱即用的移植率为 80% 至 95%，这是 CORAL-2 项目给出的数字。现代 AI kernel 的移植效果较差：任何依赖 Hopper 或 Blackwell 专用原语（TMA 描述符、`wgmma`、`tcgen05.mma`）的代码，都没有干净的 ROCm 对应物，必须手工重写。

HIP 之上是一层刻意与 NVIDIA 一一对应命名的函数库：**[rocBLAS](https://github.com/ROCm/rocBLAS)** 对应 cuBLAS；**[hipBLASLt](https://github.com/ROCm/hipBLASLt)** 对应 cuBLASLt；**[MIOpen](https://github.com/ROCm/MIOpen)** 对应 cuDNN；**[RCCL](https://github.com/ROCm/rccl)** 对应 NCCL；**Composable Kernel** 及其现代 ck-tile DSL 对应 CUTLASS；rocprofv3 / rocprof-sys / rocprof-compute 对应 Nsight 系列。

AMD 没有第一方 TensorRT-LLM 对应物。它的答案是扶持 **[vLLM](https://github.com/vllm-project/vllm)** 作为开源服务引擎，并提供插入其中的 AMD 专用算子 **AITER**；专门的 ROCm vLLM 持续集成，在 2026 年初把测试通过率从 37% 提高到 93%。

PyTorch 路径是一等公民。Eager-mode PyTorch 自 2018 年起就能运行在 ROCm 上；`torch.compile` 通过 Triton 降低，而 Triton 的 ROCm 后端——包含用于预编译数学 kernel 的 AOTriton——也已进入上游。这里没有 XLA 式中间 IR；ROCm 直接编译到 HIP / Triton / CK。

随着 Triton 成为 PyTorch 默认 kernel 路径，大量移植成本会自然消失：通过 `torch.compile` 运行的 kernel，无需修改源码即可同时工作在 CUDA 和 ROCm 上。这正是 AMD 开放战略背后的架构押注：Triton 的 Python DSL 成为跨厂商通用语，绕开建立 CUDA 等效 kernel 生态的必要性。

**FlashAttention** 是决定成败的案例。**FA2** 已通过 Composable Kernel 在 MI300X 上投入生产，PyTorch 在 ROCm 上默认使用 CK 或 AOTriton。**FA3** 为 Hopper 调优，目前通过 AITER + CK 获得部分支持，但 Dao-AILab 的标准实现仍仅支持 CUDA。2026 年 3 月发布、面向 Blackwell 的 **FA4** 完全没有 ROCm 移植。

Hazy Research 于 2025 年 11 月发布的 **[HipKittens](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)** 是 ThunderKittens 的 MI355X 移植；它声称只用约 500 行代码，就能让前向传播性能追平手工调优的 AITER。规律是：开源学术 kernel 通常会在 NVIDIA 版本之后数月，而非数年，补齐 AMD 的长尾。

生产部署已经验证这项策略。Microsoft Azure 的 **ND MI300X v5** 实例于 2024 年 5 月正式商用，OpenAI 在其上运行 GPT 推理；Meta 通过 Grand Teton 平台在 MI300X 上服务 Llama 3 / Llama 4；Oracle OCI 的 **BM.GPU.MI300X.8** 于 2024 年 9 月正式商用，MI355X 则在 2026 年跟进。这些都是超大规模云厂商的真实服务集群，不是试点。

客观差距依然存在。独立基准测试（Phoronix，2026 年 3 月）显示，在相同精度、同级芯片上的标准 PyTorch / vLLM / SGLang 负载中，ROCm 7.2 比对应 CUDA **慢 10% 至 25%**。ROCm 7 已达到**功能对等**，但尚未达到**性能对等**。

NVIDIA 最持久的护城河仍是 FlashAttention-4 这类利用 Blackwell 最新原语的研究代码长尾：它没有干净的 ROCm 对应物，只能等待手写 AITER kernel 或 HipKittens 一类社区移植。NVIDIA 会把工程师派驻前沿实验室；AMD 则通过 GitHub 交付 kernel。两种策略在通用负载——Llama 推理、注意力、稠密 Transformer 训练——上逐渐趋同，但新型研究代码的长尾仍会让 MI300X / MI355X 部署付出 NVIDIA 用户无需承担的工程成本。

---

## Cerebras WSE

**[Cerebras](https://www.cerebras.ai/)** 制造了**有史以来真正交付过的最大芯片**。其哲学是：内存墙是切割晶圆造成的。一家晶圆厂会在 300 mm 硅片上印制几十颗裸片，再把它们锯开；随后，整个行业又用最复杂的工程——HBM、NVLink、CoWoS、每机架 5,184 根铜缆——把这些碎片重新连起来，而可用带宽只有裸片内的一小部分。

Cerebras 直接跳过切割。**Wafer-Scale Engine（WSE，晶圆级引擎）**就是一整片硅：84 个光刻掩模区、46,225 mm²、900,000 个数据流核心；全部片上内存都使用 SRAM，距离计算单元只有一个周期。

### 演进谱系

| 年份 | 架构与系统 | 关键变化 |
| --- | --- | --- |
| 2019 | **[WSE-1](https://old.hotchips.org/hc31/HC31_1.13_Cerebras.SeanLie.v02.pdf) / CS-1** | 首款交付的晶圆级处理器：1.2 万亿晶体管、400,000 个核心、18 GB 晶圆上 SRAM。 |
| 2021 | **[WSE-2](https://8968533.fs1.hubspotusercontent-na2.net/hubfs/8968533/IEEE%20Micro%202023-03%20Hot%20Chips%2034%20Cerebras%20Architecture%20Deep%20Dive.pdf) / CS-2** | 7 nm：850,000 个核心、40 GB SRAM；**权重流式传输**把权重移出晶圆，存放到 MemoryX。 |
| 2023 | **[Condor Galaxy](https://www.hpcwire.com/aiwire//2023/08/30/cerebras-and-g42s-inception-unveil-jais-a-13b-parameter-arabic-llm-trained-on-condor-galaxy/) / CG-1** | 与 G42 构建 64 系统集群；训练 Jais 阿拉伯语 LLM 家族。 |
| 2024 | **[WSE-3](https://www.cerebras.ai/press-release/cerebras-announces-third-generation-wafer-scale-engine) / CS-3** | 5 nm：4 万亿晶体管、900,000 个核心、44 GB SRAM；每核心 FP16 SIMD 翻倍为 8-wide；集群规格可扩至 2,048 套系统。 |
| 2024 | **[推理服务](https://www.cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed)** | 不再流式传输，而把权重固定在 SRAM 中；实现业界独立测得的最快解码速度，并成为如今定义公司的业务转向。 |

### 体系结构

GPU 是一套层次结构：线程嵌在 warp 中、warp 嵌在 SM 中、裸片嵌在封装中、封装嵌在机架中；每一道边界都有自己的带宽、延迟和编程结构。所有由裸片构建的加速器都会继承某种版本的层次。

WSE 则是一张**平面**：900,000 个完全相同的核心在二维 mesh 上从一侧铺到另一侧，没有共享缓存、没有全局内存，在任意核心与其余 899,999 个核心之间也没有任何类别的边界。

每个核心都极小：WSE-2 上约 38,000 µm²，面积大约一半是 SRAM、一半是逻辑，峰值功耗 30 mW。核心包括 48 kB 本地 SRAM、16 个通用寄存器、六级流水线、4-wide FP16 FMAC SIMD（WSE-3 上为 8-wide），以及连接 fabric 的五端口路由器。

执行方式是**数据流**：核心保持空闲，直到一枚 **wavelet** 到达；wavelet 中的控制位选择要触发哪个 handler task；当张量操作数到达和排出时，8 个硬件**微线程**逐周期切换。这里没有 warp、warp 调度器、缓存未命中或重排序缓冲区：**数据何时到达，何时就是调度。**

![Cerebras WSE-3 晶圆与单个光刻掩模区](/assets/images/posts/ai-chip-architectures-full/cerebras-wafer-die.png)

*图 13：左侧为 WSE-3 晶圆：84 个光刻掩模区以 12×7 网格铺满 300 mm 晶圆内可容纳的最大方形，划片槽接缝保持完整；裸片边缘一排 12×100 GbE 是唯一对外通道。右侧放大一个光刻掩模区：统一的二维核心 mesh 以每裸片 2,880 GB/s 的金属连线跨越划片槽，软件看到的是一张没有接缝、包含 900,000 个核心的 fabric。*

![一个 Cerebras 核心](/assets/images/posts/ai-chip-architectures-full/cerebras-core.png)

*图 14：单个 Cerebras 核心放大图。拥有 24 种 color 静态路由的五端口 fabric 路由器，向运行 8 个微线程的数据流任务调度器供数；下方是 GPR、44 个张量描述符寄存器、分成 8 个单周期 bank 的 48 kB 本地 SRAM、FMAC SIMD 计算引擎，以及在发送端利用非结构化稀疏的零值过滤器。*

#### 晶圆

步进式光刻机每次曝光晶圆上的一个掩模区，每次约 850 mm²；所以所有传统芯片都受这个上限约束——这也解释了为什么 NVIDIA 的 B200 一触及极限就变成两颗裸片。

Cerebras 像 TSMC 其他客户一样，以 12×7 网格把同一颗约 550 mm² 的裸片印制 84 次；然后使用与 TSMC 联合开发的工艺，在通常会被锯开的、宽度不到 1 mm 的划片槽上方额外铺设高层金属。mesh 通过源同步并行接口跨越每条接缝，WSE-3 上每裸片带宽为 2,880 GB/s；整个跨裸片层只消耗约 97 W。对软件而言，接缝并不存在：它看到的只有一张统一 mesh、一颗芯片。

晶圆级计算过去曾被尝试过，却败在良率：单片晶圆计算机上任何一处缺陷都会毁掉整片晶圆，这也是 20 世纪 80 年代这一思路夭折的原因。Cerebras 的答案是粒度。H100 上一处缺陷会禁用整个约 6 mm² 的 SM；WSE 上同一缺陷只会禁用一颗 0.05 mm² 的核心。WSE-3 实际制造约 970,000 个核心，交付时启用 900,000 个：约 7% 的备用核心，加上冗余 fabric 链路，让硬件可以绕开所有缺陷，恢复完整的逻辑 mesh。

#### 核心

核心最不寻常的地方不是数据通路，而是“指令”到底是什么。16 个通用寄存器旁边有 **44 个数据结构寄存器（DSR）**，每个保存一份张量描述符：基地址、范围、步幅，最多四维。

指令通过 DSR 指定操作数，因此一条 FMAC 指令表达的是：“把到达的数据流与这个驻留张量相乘，再累加到那个张量”；硬件会持续流送元素，直至整个张量结束。乘法外围没有软件循环，每个元素也无需重新取指；循环就写在描述符里。NVIDIA 花了五代 Tensor Core 才逐步把矩阵乘推向一条描述符驱动命令；在 WSE 核心上，张量指令从一开始就只有这种形式。

顺序由 fabric 决定。一个 color 是编译时静态路由、并绑定 handler task 的虚拟通道，所以在某个 color 上发送 wavelet，**本身就是**调用目的核心上的代码：16 个控制位是调用，16 个数据位是参数。

**任务调度器**在核心的 8 个微线程中保存进行中的张量操作，并根据操作数是否就绪每周期切换。它与 warp 调度器在 64 个驻留 warp 间隐藏停顿属于同一种工作，只是这里用 8 个上下文就够了，因为要隐藏的延迟只是 SRAM bank 忙碌或相邻核心一跳，而不是一次 HBM 往返。

48 kB 本地 SRAM 围绕数据通路而非局部性组织：8 个单端口、每个 6 kB 的 bank，每周期提供两次 64 bit 读取和一次 64 bit 写入——正好读入两个由 4 个 FP16 元素组成的操作数，再写出一个结果，与 WSE-2 的 FMAC 宽度完全匹配。256 byte 软件管理缓存（WSE-3 为 512 B）把最热数据保留在流水线旁。

这就是整台机器命题的缩影：在每个核心上，内存带宽与计算能力精确匹配；整片晶圆把这种平衡重复 900,000 次。

#### 计算

晶圆上没有矩阵单元。NVIDIA、Google 和 AMD 都把 FLOPs 集中在专用矩阵乘引擎——Tensor Core、MXU、Matrix Core——主要区别只是如何向引擎供数。Cerebras 则用 fabric 拼装出矩阵乘。

一项 GEMM 会成为整片晶圆的协同编舞：每个到达的权重沿着一行保存激活值的核心广播；每个核心都把该权重与自己驻留的数据片执行一次乘加——即每个权重执行一次 AXPY；随后，部分和沿 mesh 归约。Tensor Core 从寄存器 tile 获得的数据复用、MXU 从固定布线获得的数据复用，WSE 都从几何结构中获得：激活值从不移动，在途操作数只有当前要相乘的那个值。

计算 FLOPs 时必须谨慎，因为 Cerebras 宣传的数字无法直接比较。WSE-3 标称的 **125 PFLOPS 是稀疏 FP16**：它假设在理想稀疏张量上，硬件跳过零值可获得约 8 倍收益。稠密 FP16 大约是 **15.8 PFLOPS**——由 900,000 核心 × 8-wide FMAC × 1.1 GHz 推算；Cerebras 没有公布官方稠密数字。

这是真实计算能力，但不是重点：按每瓦稠密 FLOPs 计算，晶圆输给同期每一种 GPU。WSE 从来不是一台 FLOPs 机器，而是一台**带宽机器**；这些 FLOPs 的存在，是为了跟上 SRAM。

跳过零值正是数据流发挥价值的地方。因为计算由到达的数据触发，所以零值不会触发任何动作：**零值在发送端被过滤**，接收核心根本看不到它，也不会浪费那个周期。这是逐元素粒度的非结构化稀疏——完整通用情况；NVIDIA 的 2:4 结构化稀疏只是其中一个受限样本。

但到目前为止，这仍是一项尚未真正发挥的选项。Cerebras 自己的稀疏预训练结果——[SPDF](https://arxiv.org/abs/2303.10464) 在 13 亿参数上达到 75% 稀疏，后续工作扩大到 67 亿参数——都由厂商撰写且规模低于 7B；没有任何已披露的旗舰客户模型使用稀疏训练。该硬件上最大的训练任务 Jais 2 仍是稠密模型。唯一能利用非结构化稀疏的硅片，尚未交付一个使用它的代表性模型。

#### 内存

内存层次只有一级：**分散在核心内部、每片 48 kB 的 44 GB SRAM，晶圆上除此之外什么都没有。**没有 HBM、没有 L2、没有淘汰策略；每个 byte 距离 FMAC 都只有一个周期。

标称带宽为 21 PB/s，但这个数字必须注明含义：它是 900,000 个本地 SRAM 端口的**总和**，属于晶圆内聚合带宽，不是一条点到点链路，也不能与 HBM 数字直接比较。更诚实的比较是每 FLOP 可获得多少 byte：晶圆每个稠密 FP16 FLOP 可供给约 1.3 byte，而 B200 从 HBM 获得的约为 0.002 byte。按这个维度，所有 GPU 和 TPU 都处于饥饿状态，WSE 是唯一平衡的机器。

解码本质上是纯带宽问题——每生成一个 token 都要完整读取一次权重——而晶圆最终恰好为这个阶段塑形。

另一面则是这一层的边缘。晶圆通往外界的连接只有 12×100 GbE，即 **1.2 Tb/s**，仅略高于一颗 Blackwell GPU 所连接的单块 ConnectX-8 NIC。晶圆上 SRAM 与晶圆外以太网之间相差**五个数量级**。NVIDIA 的层次会逐级下降，每一层只比上一层慢几倍；WSE 只有两级，中间却是一道悬崖。晶圆是一座孤岛，而同一个事实既是孤岛的超能力，也是它的牢笼。

这座岛也不会继续变大。领先制程上的 SRAM 密度实际上已经停止缩放：WSE-3 完整跨越一次制程节点、晶体管数增加 54%，SRAM 容量却只比 WSE-2 多 10%。逻辑仍会缩小，六晶体管 SRAM 单元却不会。架构最稀缺的资源，恰好是下一代工艺节点不再赠送的东西。

#### 权重流式传输

在晶圆上训练，会把其他架构视为理所当然的数据流彻底反转：GPU 或 TPU 让权重驻留、激活值流过；WSE 则让**激活值驻留、权重流过**。

主权重保存在集群旁边的 **MemoryX**——一台包含 DRAM 和闪存的设备。权重逐层流过晶圆，触发与 SRAM 中固定激活值的乘加，然后离开；反向传播时，梯度流回外部，优化器步骤则由 MemoryX 内的 CPU 执行。权重更新是 O(参数量) 的逐元素工作，没有复用，因此 CPU 级计算足以跟上。晶圆从不存储权重，“即使临时存放也不会”——这是 [Cerebras 自己的表述](https://www.kisacoresearch.com/sites/default/files/documents/cs_weight_streaming_white_paper_-_cerebras.pdf)。模型大小受 MemoryX 限制，而不是 44 GB；44 GB 限制的是激活值与批次。

它换来的是编程模型。一片晶圆能保存完整一层的激活值，因此没有张量并行、流水线并行或 FSDP 分片：70B 模型可以写成一个单设备程序，多系统扩展则通过 **SwarmX** 使用**纯数据并行**。SwarmX 是一棵广播/归约树，把一条权重流扇出到 N 片晶圆，再在返回路径上汇总梯度。主导 GPU 训练的那张“并行策略电子表格”，根本没有 Cerebras 这一页。

代价则体现在市场真实选择所揭示的规模上。规格表写着可扩至 2,048 台 CS-3，但已披露的最大集群只有 64 台（Condor Galaxy 3）。平台上已披露、从头训练的最大模型是 **Jais 2：700 亿参数、2.6 万亿 token**，由锚定客户 G42 与驻场 Cerebras 工程师共同训练。CS-1 问世七年以来，没有任何客户披露过 70B 以上模型。GPU 实验室通常公开、一般为 35% 至 45% 的模型 FLOPs 利用率（MFU），Cerebras 从未为任何训练任务披露。

#### 数值格式

数值格式一句话就能说完：**FP16 和 BF16，使用 FP32 累加**；此外 WSE-3 加入一条 16-wide 8-bit 整数路径，Hot Chips 资料将其标为定点数。没有 FP8、没有 FP4，也没有微缩放。

其他每家厂商都在每代把精度折半，再用块缩放买回准确率；Cerebras 仍坚持 16 bit 计算，并把它作为质量卖点——“原始 16-bit 权重”。矛盾显而易见：SRAM 容量是这套架构最稀缺的资源，而 8-bit 权重可以把一个模型所需的晶圆数减半。只支持 16 bit 究竟源于数值理念，还是数据通路路线图的空缺，仍是开放问题；Cerebras 的一手资料中没有任何内容表明晶圆支持浮点 8。

#### 五项押注

1. **不要切割晶圆。** 裸片边界是其他所有厂商都要缴纳的税：SerDes、中介层、HBM 堆栈、电缆、交换机。用金属连线缝合 84 个光刻掩模区，竞争系统中带宽最高的那道边界就根本不存在。
2. **SRAM 是唯一内存。** 以行业最激进的比例，用容量换带宽：44 GB，晶圆内聚合带宽 21 PB/s。让机器本身保持平衡，而不是用层次结构掩盖失衡。
3. **数据流核心，不设矩阵单元。** 900,000 个小核心由到达的 wavelet 触发，通过广播、FMAC 和 mesh 归约拼出矩阵乘；跳过零值不是特殊模式，而是免费结果。
4. **权重移动，激活值驻留。** 权重流式传输把模型大小（MemoryX）与晶圆内存（44 GB）解耦，并把集群扩展简化为纯数据并行。
5. **销售延迟，而非吞吐。** 晶圆能以比任何 HBM 系统更快的速度为每个 token 重新读取整个模型；把这种速度作为溢价产品销售，而不是竞争单 token 成本。

### 扩展

纵向扩展与横向扩展在这里有不同含义。NVIDIA 的纵向扩展问题——让 72 个封装表现得像一台设备——在 WSE 上由光刻直接解决：一致域从晶圆厂出来时就是一个整体。剩下的只有晶圆边缘之外的一切，而没有其他机器会如此猛烈、如此早地撞上自己的边缘。

**纵向扩展**：就是晶圆本身。900,000 个核心位于同一二维 mesh：32 bit 链路、单周期 hop、通过 24 种 color 静态路由、原生广播、214 Pbit/s 聚合 fabric 带宽。面积由 300 mm 晶圆尺寸固定为 46,225 mm²。

**横向扩展**：立刻进入以太网。每系统 12×100 GbE（1.2 Tb/s）。训练通过 SwarmX 扩展——经 RoCE 做数据并行广播/归约；推理则在层边界把模型分片到不同系统，采用流水线并行。

#### 纵向扩展

晶圆内部 fabric 没有 SerDes、电缆或收发器，每增加一条链路也没有边际成本：路由在编译时决定，每一跳只需一个周期，广播是 fabric 原生原语，而非交换机功能。

NVL72 要用 5,184 根铜缆和一整托盘 NVSwitch ASIC，为 72 颗 GPU 提供 130 TB/s all-to-all；WSE 的对应域只是一个由光刻形成的物体。

限制在于，这个域的大小是常数。NVIDIA 的纵向扩展域每代都在增长，三年内从 NVL72 走向 NVL576；晶圆从 2019 年开始就一直是 46,225 mm²，未来也会维持如此。300 mm 是行业实际采用的最大晶圆——450 mm 转型在十年前已经失败——所以 Cerebras 的纵向扩展路线图只能依靠下一代工艺节点提供多少密度；再也没有额外面积可拿。

#### 横向扩展

训练横向扩展依赖 SwarmX，而它只做一件事：复制。把权重流广播到 N 片晶圆，再在返回路径上归约梯度；批次随系统数量增长，模型大小则不会。标称 2,048 套系统、稀疏“256 ExaFLOPS”的上限从未真正建成；实际最大值为 64。

推理则完全放弃权重流式传输，因为算术不允许。每生成一个 token，如果要通过约 150 GB/s 的通道从 MemoryX 流送 70B 模型的 140 GB 权重，就会耗时约一秒。因此，推理会**把权重固定在 SRAM 中**，在层边界把模型分片到多片晶圆：Llama 70B“最少只需四台”CS-3，通过以太网执行流水线并行；每增加一片晶圆，就增加 44 GB 权重加 KV 容量，同时增加 23 kW 负载。

其速度真实存在，也经过独立验证。**Artificial Analysis** 在 2024 年 8 月发布时测得：Llama 3.1 8B 为每秒 1,850 token，70B 为每秒 446 token；随后 Llama 405B 达到每秒 969 token、首 token 延迟 240 ms；2025 年 Llama 4 Maverick 达到每秒 2,522 token，约为当时公开最佳 Blackwell 数字的 2.4 倍。厂商引用的峰值更高：采用推测解码的 70B 为每秒 2,100 token；GPT-OSS-120B 为每秒 3,000 token，而实时独立测量更接近 2,000。没有任何 GPU 提供商能在单用户解码速度上接近它。

经济性则是锋利的另一面。每片晶圆只有 44 GB，这意味着前沿规模模型会吞噬整个机群：[SemiAnalysis](https://newsletter.semianalysis.com/p/cerebras-faster-tokens-please) 估算，一个 1.6 万亿参数级模型约需 24 台 CS-3，而它在 GPU 上只占几座机架。每套系统的物料成本据分析师估算约 45 万美元，标价约 200 万至 300 万美元，但从未正式披露。

在解码期间，晶圆庞大的 FLOPs 大多空闲；Cerebras 拒绝披露批次大小，也从未公布单系统吞吐。同一开源模型的 API 单 token 价格大约是 GPU 提供商的 3 至 5 倍，Llama 405B 还被悄然从 API 下架；SemiAnalysis 将此解读为服务经济性未能成立。

固定 SRAM 也会为上下文定价：KV Cache 与权重共享同一 44 GB，长上下文会挤占容量，迫使每个副本使用更多系统。其 API 上限为 131K token，而前沿提供商已支持 256K 至 1M。系统也能服务 MoE——厂商宣称 Qwen3-235B 约每秒 1,500 token——但 MoE 恰好是这种内存形态最糟糕的情况：庞大的参数占用最昂贵的内存，同一时刻却只访问少数专家。

市场对它给出了坦率定价。Mistral Le Chat（约每秒 1,100 token）、Perplexity Sonar 和 Meta Llama API 都愿意为延迟付费。2026 年 1 月，OpenAI 签约采购**截至 2028 年共 750 MW 的 CS-3 容量**；签约时[报道金额超过 100 亿美元](https://www.cnbc.com/2026/01/14/cerebras-scores-openai-deal-worth-over-10-billion.html)，[此后增加到超过 200 亿美元](https://finance.yahoo.com/technology/ai/articles/cerebras-systems-openai-tout-20b-040208708.html)，这是晶圆级计算迄今获得的最大背书。这些容量上首个交付的旗舰是 **[GPT-5.6 Sol](https://openai.com/index/gpt-5-6/)**，于 2026 年 7 月发布，标称每秒 750 token。

### 软件

软件栈像 TPU 一样由编译器驱动，但入口窄得多：Cerebras 编译器是一个 **kernel 匹配器**，而不是通用代码生成器。

`cerebras.pytorch` 通过惰性张量，把训练步骤追踪为 Torch-MLIR 和图 IR；随后把子图与手写 kernel 库匹配，找不到匹配的操作则退回较慢的自动生成 kernel。与 GPU 标准相比，[文档列出的限制](https://training-api.cerebras.ai/en/rel-2.4.0/wsc/tutorials/cstorch-limitations.html) 十分严苛：只支持静态图；不支持动态形状、数据依赖控制流或训练步骤中途以 eager 方式访问张量；PyTorch 版本还固定在落后于上游的版本。最好的独立实践者记录来自荷兰国家计算中心 [SURF](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/112592526/Evaluation+Cerebras+CS-2)，其中报告了不受支持的层类型，也指出标准 PyTorch 代码不存在一比一移植路径。

这里也没有 kernel 逃生通道。面对新型注意力变体，CUDA 的答案是“自己写 kernel”，TPU 的答案是 Pallas，ROCm 的答案是 Triton；Cerebras ML 栈完全没有用户 kernel 路径。当匹配器严重失配时，只能由 Cerebras 工程师修复。

另一种独立的 SDK 语言 **CSL** 会暴露原始机器——task、wavelet、color——并取得过醒目的 HPC 成绩：例如 [TotalEnergies stencil 代码](https://arxiv.org/abs/2204.03775) 约为 A100 的 228 倍，以及使用 48 台 CS-2 的 Gordon Bell 奖决赛项目。但它是另一个孤立世界，与 PyTorch 流程并不相连。平台上的每一个旗舰模型——Jais、BTLM、Med42——都由驻场 Cerebras 员工参与共同开发。

这带来一种奇特的免疫力。GPU 时代最具代表性的 kernel 谱系 FlashAttention，是一种把注意力平铺到内存层次中的方案，而 WSE 根本没有可供平铺的层次。让 AMD 花数年时间追赶移植的那类优化，在这里完全不适用。

但免疫力与贫乏性来自同一个事实。CUDA 上不断复利增长的第三方 kernel 生态，在这里没有可以附着的表面；平台历史上的每一次 kernel 改进都只有同一个作者。

那么，晶圆最终处在什么位置？它确实拥有一个凭实力赢得的真实利基：batch-one 解码速度经过独立验证，且愿意把延迟看得比成本更重要的客户已经为它付费。利基周围则是坚硬边界：单 token 价格高 3 至 5 倍；问世七年，训练上限仍停在 70B；根据 2026 年 5 月 IPO 前后的 S-1 文件，2025 年约 86% 收入仍集中在两家与阿布扎比关联的客户；而它最稀缺的资源 SRAM 密度，恰好在模型继续增长时停止缩放。

Hennessy 和 Patterson 预言了一场寒武纪大爆发；WSE 是其中最极端的身体形态——它认定内存墙只是一种封装选择，并用 46,225 mm² 的硅片拒绝接受这项选择。

---

## AWS Trainium

打造 AWS **Nitro** 卡与 **Graviton** CPU 的 **Annapurna Labs** 团队，把 **Trainium** 设计成一名**快速追随者**。计算核心采用 TPU 已验证的方案——128×128 权重驻留脉动阵列、软件管理 scratchpad、全程序编译——甚至直接共享 Google 的 **[XLA](https://openxla.org/xla)** 编译器。横向扩展 fabric 则沿用已经承载 AWS 其他服务的 Nitro 卸载网络。

真正属于 Amazon 的部分很窄，却经过精心选择：在借鉴来的核心旁加装专用集合通信硅片，以及利用垂直整合，为一颗只需在 **AWS 内部**打败 NVIDIA 的芯片制定价格。

### 演进谱系

| 年份 | 架构与芯片 | 关键变化 |
| --- | --- | --- |
| 2015 | **[Annapurna Labs](https://en.wikipedia.org/wiki/Annapurna_Labs)** | Amazon 以约 3.5 亿美元收购这家以色列芯片创业公司，使其成为 AWS 内部硅片团队。 |
| 2018 | **[Graviton](https://en.wikipedia.org/wiki/AWS_Graviton) + Nitro** | Arm 服务器 CPU 与 DPU 卸载 fabric。 |
| 2019 | **[Inferentia](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inferentia.html) / NeuronCore-v1** | AWS 首款 ML 芯片，仅支持推理：4 个 NeuronCore、8 GB DRAM、三个固定功能引擎。 |
| 2022 | **[Trainium1](https://aws.amazon.com/blogs/aws/amazon-ec2-trn1-instances-for-high-performance-model-training-are-now-available/) / Trn1、v2** | 首款训练芯片：2 个 NeuronCore-v2、可编程 GPSIMD 引擎、32 GB HBM、NeuronLink 二维 torus。 |
| 2023 | **[Inferentia2](https://aws.amazon.com/blogs/machine-learning/aws-inferentia2-builds-on-aws-inferentia1-by-delivering-4x-higher-throughput-and-10x-lower-latency/) / v2** | 与 Trn1 共享 NeuronCore-v2；推理与训练产品线汇合到同一微体系结构。 |
| 2024 | **[Trainium2](https://aws.amazon.com/blogs/aws/amazon-ec2-trn2-instances-and-trn2-ultraservers-for-aiml-training-and-inference-is-now-available/) / Trn2、v3** | 8 个 NeuronCore-v3；首次真正加速 FP8；96 GB HBM3；64 芯片 UltraServer；驱动 Project Rainier。 |
| 2025 | **[Trainium3](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/) / Trn3、v4** | AWS 首款 3 nm 芯片（TSMC N3P）；支持 OCP MXFP8 / MXFP4；NeuronSwitch all-to-all fabric 取代 torus；144 芯片 UltraServer。 |

### 体系结构

另一段自有芯片故事属于 Google，而理解 Trainium 的最好方式，是把它视为在另一朵云中重建的 TPU 命题。底层押注相同：由软件管理 SRAM 向脉动阵列供数；编译器提前排程；没有缓存，也没有线程调度器。不同之处在于构成单元。

一颗 Trainium 只包含**少量 NeuronCore**——Trn1 为 2 个，Trn2 和 Trn3 为 8 个；每个 NeuronCore 也不是单体矩阵乘引擎，而是一组解耦的专用引擎：执行矩阵乘的 **Tensor Engine**（128×128 脉动阵列）、执行归约的 **Vector Engine**、执行逐点数学的 **Scalar Engine**，以及由 8 个 512 bit 向量处理器构成、可以运行其他工作的可编程 **GPSIMD Engine**。

外围则是数据搬运器：128 个 **DMA 引擎**、安排传输顺序的 **Sync Engine**，以及从 Trn2 开始专门负责集合通信的 **CC-Core**。这里没有 warp 或 wavefront；这些引擎构成静态排程的数据流流水线。真正承担设计重量的决定都围绕脉动阵列周围的部分，而不是阵列本身。

![AWS Trainium2 封装平面图](/assets/images/posts/ai-chip-architectures-full/aws-trainium-chip.png)

*图 15：AWS Trainium2 封装平面图。两颗计算裸片并排位于 CoWoS 中介层上，每颗裸片含 4 个 NeuronCore-v3、整颗芯片共 8 个；每颗裸片两侧分别连接两组 HBM3。中央 NeuronLink 模块承载封装内 die-to-die 链路和 chip-to-chip torus 端口；顶部较小的 PCIe / Nitro EFA 区域负责连接主机和横向扩展 fabric。*

![一个 NeuronCore-v3](/assets/images/posts/ai-chip-architectures-full/aws-trainium-neuroncore.png)

*图 16：一个 NeuronCore-v3 放大图。中央是 128×128 权重驻留 Tensor Engine，由分为 128 个分区的 SBUF State Buffer 供给操作数，部分和则排入小型 PSUM 累加器。Vector、Scalar 和可编程 GPSIMD 引擎共享同一 SBUF、在旁并行运行；128 个 DMA 引擎与一个 Sync Engine 从 HBM 分阶段载入 tile，一组 CC-Core 则驱动 NeuronLink 端口，让集合通信与计算同时执行。*

#### 计算

**Tensor Engine** 拥有矩阵乘 FLOPs，另外三种引擎负责其他一切。它是一张由 16,384 个处理单元组成的 128×128 网格，采用权重驻留方式：一个操作数 tile 载入阵列后固定不动（`LoadStationary`），另一个从中流过（`MultiplyMoving`），部分和落入 **PSUM**。PSUM 是一块小型累加 SRAM，引擎可以读—加—写，使 K 轴长度超过 128 的收缩运算逐段累积。

这正是每种矩阵乘加速器核心的 `D = A·B + C` tile MMA；不同在于 NVIDIA 把它包裹进 warp 层次，Google 从 VLIW 指令束发射它，Trainium 则把它暴露为针对具名 scratchpad 的两条显式指令。

三代芯片中，阵列物理尺寸始终固定在 128×128；变化的是每个单元能装入多少个乘积。Trn1 的 NeuronCore-v2 以 FP32 累加执行 BF16 / FP16；FP8 也只达到 BF16 速率，没有加速。Trn2 的 v3 对 FP8 进行双泵送，呈现出等效 256×128 阵列，成为首款真正在 8 bit 上获得 2 倍加速的 Trainium。Trn3 的 v4 把微缩放操作数打包，呈现等效 512×128 阵列，以 BF16 的 4 倍速率运行。物理乘加单元数量从未变化，只是数据通路给它们送入了更窄的数字。

其余三种引擎负责让阵列保持繁忙。**Vector Engine** 处理跨元素归约（layer norm、softmax、pooling）；**Scalar Engine** 处理一进一出的逐点操作（激活、GELU）；由 8 个完全可编程、运行 C 的向量处理器组成的 **GPSIMD Engine**，吸收无法映射到前三者的工作。

编译良好的一个步骤会重叠运行全部四种引擎：Tensor Engine 执行矩阵乘时，Vector Engine 同时处理上一个 tile 的 softmax，DMA 引擎则载入下一个 tile。这与 TPU 和 GPU 注意力 kernel 中高效的生产者/消费者重叠相同，只是这里使用独立物理引擎，而非独立 warp 或 VLIW 槽。

当一层可以干净地分解到四类引擎时，设计就会得到回报；Transformer 大多如此。边缘情况则要付出代价：无法适配任何专用引擎的算子会落入更慢的 GPSIMD 可编程路径，也最容易成为新型架构的瓶颈。这就是 Trainium 版本的长尾成本，而每种非 GPU 加速器都要承担它。

#### 内存

内存层次把计算哲学应用到存储：**三级内存全部由软件管理，任何位置都没有硬件缓存。**AWS 自己的文档明确对比指出，NeuronCore 不像 CPU 或 GPU 那样拥有缓存，“所有内存移动都在程序中显式表达”。

片外是 **HBM**：Trn1 为 32 GB，Trn2 为 96 GB HBM3，Trn3 为 144 GB HBM3e。片上、最靠近引擎的是主 scratchpad **State Buffer（SBUF）**：带宽约为 HBM 的 20 倍，划分为 128 个分区；每个 NeuronCore 容量在 v2 / v3 / v4 上依次为 24 / 28 / 32 MiB。阵列与 SBUF 之间还有专为矩阵乘输出设计的 2 MiB 累加器 **PSUM**。

数据沿 HBM → SBUF → Tensor Engine → PSUM → SBUF 移动，每一跳都由编译器发出；硬件不预取，也不淘汰。这与 Google VMEM 的押注完全相同：显式 scratchpad 要求编译器完美排程，没有缓存可以掩盖错误；它也与 NVIDIA 硬件管理的 L2、L1 正好相反。

Trainium 同时继承其上限与脆弱性：排程正确时，引擎永不停顿；排程错误时，没有后备路径。这套设计用相对充裕的 HBM 配合适中的峰值 FLOPs，因此按单位计算能力衡量，Trainium 比同级 NVIDIA 芯片携带更多内存。但在**绝对容量**上仍然落后：Trn2 的 96 GB 少于 H200 和 B200；2025 年的 Trn3 为 144 GB，也少于同期 192 GB B200 和 288 GB B300。

所以 AWS 在论证大模型服务经济性时真正使用的杠杆不是内存领先，而是**价格**：由自己制造、自己出租的硅片，能够按每单位计算和 HBM 的成本定价。

#### 数值格式

Trainium 与所有其他架构沿着同一条精度折半路线前进：FP32 → BF16 → FP8 → FP4，但有两项 Trainium 特有变化。

第一是**可配置 FP8**。它不像 Hopper 那样固定 E4M3 和 E5M2；Tensor Engine 接受可调指数偏置，支持 E5M2、E4M3 与 E3M4，让编译器能针对每个张量在范围和精度之间取舍。

第二是 Trn3 的 FP4 **不会带来额外吞吐**。OCP MXFP4 操作数在到达阵列前会向上转换为 MXFP8，因此 FP4 与 FP8 同速，只节省内存与带宽，不增加计算能力。两代都借助行业通用的精度恢复技巧：Trn3 开始使用微缩放块指数，每一代都在硬件中支持**随机舍入**。

有一个数字不应直接采信：AWS 宣传的稀疏峰值为 FP8 的 4 倍，但其架构页面自己写的是相对稠密 FP8 为 2 倍——4 倍指的是相对稠密 BF16。因此营销口径中的加速比与数据通路并不完全一致。

#### 硅片中的集合通信

GPU 上没有干净对应物的模块，是**集合通信核心**。分布式训练和推理会把大量墙钟时间花在集合操作上：每次梯度更新都需要 all-reduce，每个 MoE 层都需要 all-to-all。

GPU 会把这些集合作为 NCCL kernel 运行在执行数学计算的同一批 SM 上，因此通信与计算争用相同硅片，二者能否重叠要靠软件争取。Trainium 把功能切出为专用硬件：每颗 Trn2 拥有 20 个 **CC-Core**，直接连接 **NeuronLink** 端口，在 Tensor 与 Vector Engine 持续运行时，同时执行 all-reduce、all-gather、reduce-scatter 和 all-to-all。

这与 Google 加入 SparseCore、Cerebras 加入核心外零值过滤器是同一种动作：发现主引擎形状不适合某类负载，就在旁边投入少量面积制造专用模块，而不是从核心偷走周期。通信成为芯片**并发完成**的工作，而不是芯片必须暂停计算才能做的工作。

#### 五项押注

1. **云才是产品，芯片只是组件。** Annapurna 把芯片、服务器、机架、Nitro 网络和云 API 设计成一个栈，所以 Trainium 只需在 AWS 内部赢得性价比，从不需要在商用芯片规格表上取胜。
2. **借用计算命题，不重新发明。** 128×128 权重驻留阵列、软件管理 SBUF / PSUM scratchpad、全程序编译都是 TPU 的押注，甚至复用 Google OpenXLA；节省下来的投入用于网络与机架。
3. **集合通信属于硅片。** 专用 CC-Core 在硬件中把 all-reduce、all-to-all 与计算重叠，而不是把它们做成抢占矩阵单元 FLOPs 的 kernel。
4. **复用云自己的网络。** 横向扩展使用带 SRD 传输的 EFA——AWS 其余服务已经采用的 Nitro 卸载、分组喷洒式 RDMA；不使用 InfiniBand。
5. **让拓扑迁就负载。** Trn1 与 Trn2 复制 TPU 的 torus；当 MoE 流量超出最近邻网络能力后，Trn3 用 NeuronSwitch 交换式 all-to-all fabric 取代它。坦率地说，这仍在追随既有方案：先追随 Google，现在又追随 NVIDIA。

### 扩展

Trainium 的扩展方式继承 AWS 其他系统的分层：需要像一台机器协作的芯片，通过紧耦合 **NeuronLink** 域连接；再往外的一切则使用云的通用 **EFA** fabric。

纵向扩展域不像 NVLink 那样提供缓存一致共享内存。AWS 把 UltraServer 描述为汇聚的数 TB 内存，但底层仍是点到点链路上的消息传递；从思想上说，它更接近 TPU ICI，而不是 NVSwitch 交叉开关。

**纵向扩展**：NeuronLink 把芯片绑定成一台 UltraServer。到 Trn2 为止使用 torus——每个实例含 16 颗芯片，构成 4×4 二维 torus；每台 UltraServer 含 64 颗芯片，构成 4×4×4 三维 torus。Trn3 改用 NeuronSwitch all-to-all fabric。它使用消息传递，而非一致性 load/store。

**横向扩展**：通过以太网使用由 Nitro 卸载的 Elastic Fabric Adapter。SRD 传输把每条数据流喷洒到多条路径上，以可靠但乱序的方式交付；UltraCluster 通过 10p10u fabric 扩展到数十万颗芯片。

#### 纵向扩展

NeuronLink 是 Trainium 的芯片间 fabric，相当于 NVIDIA 的 NVLink、TPU 的 ICI。到 Trn2 为止，它像 TPU 一样把芯片连成 **torus**：一个 **trn2 实例**包含 16 颗芯片，构成 4×4 二维 torus，每芯片约 1.28 TB/s；**Trn2 UltraServer** 则把四个实例连成包含 64 颗芯片的 4×4×4 三维 torus，对外呈现为一个纵向扩展域，拥有 83 PetaFLOPS 稠密 FP8 和约 6 TB HBM。

第三个 torus 轴被刻意做窄：实例间环形链路每芯片约 256 GB/s，而实例内部为 1.28 TB/s。这正是 torus 的典型取舍：布线便宜、最近邻带宽巨大，但跨越网络直径时需要很多跳。AWS 把 64 芯片 UltraServer 对标 NVIDIA 72-GPU NVL72；二者聚合计算能力处于同一级别，但 torus 并非交叉开关，在非最近邻流量下表现会截然不同。

这项取舍正是 Trn3 放弃 torus 的原因。**NeuronSwitch-v1** 是一种交换式 **all-to-all** fabric，芯片间带宽大约翻倍；更重要的是，它把网络直径压平，让任意芯片通过一次交换 hop 抵达任意其他芯片。

Trn3 UltraServer 扩展到 144 颗芯片，提供 362 PetaFLOPS 稠密 FP8 和 20.7 TB HBM3e。背后的动机与 Google 为 MoE 推理转向高基数拓扑相同：专家路由是 all-to-all，恰好是 torus 的最坏情况；交换机把相距最远的一对芯片缩短为一次穿越。

Trainium 的互连路线图，就是行业演进的压缩重演：当负载偏向最近邻时采用 torus；当它不再如此时，转向交叉开关。

![Trn3 UltraServer 纵向扩展](/assets/images/posts/ai-chip-architectures-full/aws-trainium-scale-up.png)

*图 17：Trn3 UltraServer 放弃 Trn2 torus，转用 NeuronSwitch-v1——运行在 NeuronLink-v4（每芯片约 2 TB/s）上的交换式 all-to-all fabric。服务器内部，芯片通过一级（L1）NeuronSwitch 相连，任意芯片一跳可达；服务器之间，两台二级（L2）NeuronSwitch 把 144 芯片 UltraServer 连成一个 all-to-all 域，拥有 20.7 TB HBM3e 和 362 PetaFLOPS 稠密 FP8。扁平直径特别适合 MoE 和 all-to-all 集合通信，而 torus 会为此付出多跳代价。*

#### 横向扩展

横向扩展并非定制系统，而是 AWS 既有 fabric。每个 Trainium 实例都通过 **Elastic Fabric Adapter** NIC 接入数据中心网络——每个 Trn2 实例为 3.2 Tbps；传输协议是由 **Nitro** 卡卸载、而非在加速器上运行的 **SRD（Scalable Reliable Datagram）**。

SRD 是 AWS 从头设计的 RDMA 答案。它不同于 RoCE 或 InfiniBand 的单条有序流，而是把每条消息喷洒到最多 64 条并行路径上，可靠但乱序交付；重组工作上推到集合通信库，从而绕开一条拥塞路径造成的队头阻塞。这项传输最初面向 AWS 整朵云构建，后来被复用于加速器 fabric。

![AWS Trainium 横向扩展](/assets/images/posts/ai-chip-architectures-full/aws-trainium-scale-out.png)

*图 18：AWS Trainium 横向扩展。UltraServer 通过由 Nitro 卡卸载的 Elastic Fabric Adapter NIC，在标准以太网上而非 InfiniBand 上互连。SRD 把每条数据流喷洒到最多 64 条路径，可靠但乱序交付，避开队头阻塞。10p10u UltraCluster fabric——约 10 Pb/s、延迟低于 10 微秒——连接数十万颗芯片；Project Rainier 为 Anthropic 在美国多个数据中心部署约 500,000 颗 Trainium2。*

层次顶端是 **UltraCluster**。它由 **10p10u** 网络缝合——AWS 用这个名称表示一座数据中心内约 10 Pb/s 带宽、低于 10 μs 延迟——并可扩展到数十万颗芯片。

证明案例是 **Project Rainier**：约 50 万颗 Trainium2 分布在美国多个数据中心，2025 年末为 **Anthropic** 上线；到 2026 年初，Claude 已运行在超过 100 万颗芯片上。这是任何外部实验室对非 NVIDIA 训练平台做出的最大承诺。

它之所以存在，是因为端到端经济账能够成立。AWS 声称 Trainium2 的性价比比 Hopper 级 GPU 实例高 30% 至 40%——这是 AWS 自己的数字，比较对象是上一代 NVIDIA 而非 Blackwell；由于从 Nitro 卡到 API 的每一层都由 Amazon 所有，利润空间也由 Amazon 自己设定。

### 软件

Trainium 软件直接表明了它借鉴的对象：**[Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/)** 是一个**基于 TPU 同款 OpenXLA 基础、以编译器为先的软件栈**。

Neuron 编译器 `neuronx-cc` 接收 XLA HLO 图，把它降低成由 Neuron 运行时加载到 NeuronCore 的 **NEFF** 二进制。其前端 IR 来自 Google；Google 自己的 OpenXLA 公告也把 Trainium 与 TPU 并列为一等 PJRT 设备。

**torch-neuronx** 通过 PyTorch/XLA 的 LazyTensor 追踪运行 PyTorch：记录操作，在步骤边界编译整张图；**jax-neuronx** 则通过 StableHLO 降低 JAX。若把 kernel 驱动的 CUDA 放在一端、全程序 XLA 放在另一端，Trainium 几乎与 TPU 重合：编译器就是整个系统，而且大体就是同一个编译器。

二者的差异在逃生通道。面对新型注意力变体或融合 MoE 分派，单靠 XLA 不一定能综合出最优结果，因此 Neuron 提供 **NKI（Neuron Kernel Interface）**：一种 Python、tile 级 kernel 语言，直接暴露四种引擎与 SBUF / PSUM scratchpad。

它就是 Trainium 的 **Pallas** 或 **Triton**：当 kernel 的优势来自排程而不是代数表达时，用同一种 tile DSL 思路进入全程序编译器之下。更底层，集合通信库把 all-reduce 与 all-to-all 映射到 CC-Core 和 NeuronLink 拓扑，相当于 NCCL；**NeuronX Distributed** 则提供分片训练层。

与 CUDA——甚至与 TPU 软件栈——的差距在于成熟度，而不是设计。到 2024 年末，NKI、JAX 路径和分布式库仍处于 beta；移植后的模型只能运行在 AWS 上，没有跨厂商后备方案；vLLM 后端也落后于上游项目。

最明确的迹象来自锚定客户的工作方式：**Anthropic** 并不是简单通过 PyTorch 把目标设为 Trainium，而是与 Annapurna 深度合作、编写自己的底层 NKI kernel，并把修复贡献回 Neuron 栈。Trainium 在前沿已经可以生产使用，但在前沿场景中，它来自共同工程，而不是开箱即用：继承来的编译器很优秀，周围生态却还年轻。

---

## Groq LPU

**[Groq](https://groq.com/) LPU** 是一台**确定性**机器。其他所有芯片都会花费硅片去容忍不确定性：用缓存隐藏内存延迟，用调度器填补停顿，用仲裁器解决无法预测的争用。LPU 把这些全部删除。

它移除每一个**响应式**组件——没有缓存、分支预测器、仲裁器、重排序缓冲区，甚至没有片上交叉开关——把整个调度问题交给编译器，由后者把每条指令和每个 byte 安排到精确周期。剩下的是一颗在运行前就已知延迟的芯片。

TPU 把调度移入编译器，却保留 HBM 和动态网络；Groq 则删除最后两项非确定性来源：内存全部使用 SRAM，网络也要排程，因此数百颗芯片可以作为一个时钟精确的程序运行。

### 演进谱系

| 年份 | 架构与芯片 | 关键变化 |
| --- | --- | --- |
| 2016 | **[创立 Groq](https://en.wikipedia.org/wiki/Groq)** | Jonathan Ross——曾把 Google TPU 作为 20% 时间项目发起——离职创办公司，开发确定性推理芯片。 |
| 2020 | **[TSP](https://groq.humain.ai/wp-content/uploads/2024/02/2020-Isca.pdf) / GroqChip 1** | 首颗芯片（ISCA 2020，*Think Fast*）：单功能切片核心、14 nm、无 HBM、无缓存。 |
| 2022 | **[多处理器](https://dl.acm.org/doi/10.1145/3470496.3527405)** | ISCA 2022：软件排程网络通过编译后的 Dragonfly，把确定性时间表扩展到数千颗芯片。 |
| 2023 | **[Samsung 4 nm](https://www.prnewswire.com/news-releases/groq-selects-samsung-foundry-to-bring-next-gen-lpu-to-the-ai-acceleration-market-301900464.html)** | 宣布使用 Samsung SF4X 的第二代 LPU；最终没有交付，据报道流片失败。 |
| 2024 | **[LPU / GroqCloud](https://techcrunch.com/2024/03/01/ai-chip-startup-groq-forms-new-business-unit-acquires-definitive-intelligence/)** | TSP 更名为 Language Processing Unit；凭借创纪录的解码速度，公司从销售板卡转向销售 token。 |
| 2025 | **[NVIDIA 授权](https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale)** | NVIDIA 获得 LPU 技术非独家授权，并聘用 Ross 与大部分团队。 |
| 2026 | **[NVIDIA Groq 3 LPU](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) / LP30、LPX** | 技术在 GTC 2026 以 Rubin NVL72 旁的延迟协处理器重新出现，通过拆分注意力与 FFN 工作。 |

### 体系结构

其他所有架构都从**复制核心**出发：把一个 SM、TensorCore、CU 或数据流核心铺满裸片，再把工作分派给这些副本。LPU 反其道而行：它拿一颗传统核心，把它**拆开**。

指令控制、向量 ALU、矩阵单元、内存和网络分别成为一个**功能切片**——由相同硬件组成、贯穿芯片全高的一列；这些列在裸片上横向并排。每个切片内部同质，整颗芯片横向异质。

数据不会停在寄存器文件中等待发往执行单元，而是像流水线上的零件一样，向东或向西横向流过各个切片，每周期跨越一个寄存器；与此同时，VLIW 指令从控制切片向北发射，与数据汇合。数据通路中的任何部分都不响应运行时事件：编译器知道每个周期中每个操作数的位置，硬件只需推动时钟。

“流式传输”就是这套设计的身份。它最初名为 **Tensor Streaming Processor（TSP）**，一直沿用到 2024 年更名为 **Language Processing Unit**。

![Groq LPU 裸片平面图](/assets/images/posts/ai-chip-architectures-full/groq-chip.png)

*图 19：Groq LPU 平面图。裸片围绕中央 VXM 向量切片分为镜像的东西两个半区。从外向内依次为：边缘的 MXM 矩阵平面、SXM 交换切片、位于 VXM 两侧的 MEM SRAM 切片组。指令控制 ICU 沿南边缘排列，向北向所有切片发射 VLIW 指令束；操作数流则在切片间东西移动，每周期跨一个寄存器。垂直方向堆叠 320 条 lane，组织为 20 个 superlane。*

垂直轴是 SIMD 宽度。芯片纵向有 320 条 lane，组织成 20 个、每个 16 lane 的 **superlane**；第 21 个是为良率准备、出厂时熔断且对软件不可见的备用组。每个切片一次作用于全部 320 条 lane。

水平轴则是时间。每条 lane 有 64 个逻辑**流寄存器**，32 个向东、32 个向西；每次时钟跳动，流就沿对应方向前进一个切片，直至被消费或从裸片边缘离开。一个切片从经过的流中读出操作数、完成计算，再把结果写回流中，送往下一切片。

裸片围绕中央向量单元镜像成两个半区，因此某个数值只需产生一次，就可以被任一侧的切片消费。

#### 计算

LPU 与其他架构保持同样的分工：矩阵工作由专用单元完成，其他工作交给向量引擎；只是两者都被安排成流中的切片。

矩阵路径是 **MXM**：4 个独立的 320×320 乘加平面——每个半区 2 个——总计 409,600 个乘法器，输入 INT8 或 FP16，累加到 INT32 或 FP32。权重会安装到整个平面上，耗时不到 40 个周期；随后激活值流过，乘积持续累加。

在 900 MHz 下，它大约达到 **750 INT8 TOPS 与 188 FP16 TFLOPS**。不同寻常的是，这些数字没有稀疏性星号：TSP 完全拒绝跳过零值，因为依赖数据的跳过会让执行时间依赖数据，而确定性是它唯一绝不交易的属性。

向量路径位于裸片中央的 **VXM**：每条 lane 上有排列为 4×4 mesh 的 16 个 ALU，全芯片共 5,120 个 32 bit ALU，负责激活、归一化、量化和残差相加。

由于计算是**空间化**的，而非发往共享单元，一个操作数可以连续几个周期依次经过一串 VXM ALU，再直接进入 MXM 平面，完全无需访问内存。GPU kernel 要手工实现的算子融合，在这里就是切片的物理顺序。

第三种切片 **SXM** 处理直线数据流无法表达的移动：lane 位移、320-lane 置换、转置以及芯片间链路都在此完成。因此，跨 lane 重排数据是一等操作，而不是一次往返 SRAM 的搬运。

#### 内存

没有 HBM、没有 DRAM，也没有缓存。片上内存是 **MEM 切片**：88 个切片、东西半区各 44 个，共 230 MB SRAM；每个 byte 距离计算切片都只有一个周期，聚合带宽约 80 TB/s。

这就是全部层次：只有一级，平坦、由软件寻址，也没有任何会引入可变访问延迟的淘汰、预取或一致性机制。

由此产生的后果，是这套架构的决定性约束：230 MB 装不下一个模型。Llama-2 70B 的 FP16 权重为 140 GB，因此必须**跨数百颗芯片分片**，把权重散布到一整座甚至更多机架的聚合 SRAM 中；实际部署配置约为 576 颗 LPU。

GPU 把模型固定在少量封装的 HBM 中，让 token 流过模型；LPU 则把模型展开在集群 SRAM 中，让 token 流过集群。芯片数量由容量而非计算能力决定：权重必须能够装下。

这与 Cerebras 做出相同交换——只使用 SRAM、不使用 HBM——却从相反方向抵达：Cerebras 保留一颗巨型裸片，牺牲单片晶圆的容量；Groq 保持普通裸片尺寸，牺牲的是让模型永远无法装入一颗芯片。

#### 数值格式

数值格式代表一条没有走下去的路。其他每家厂商都在逐代把精度从 FP16 降到 FP8、再降到 FP4，并用块缩放买回准确率；TSP 一直停留在 **FP16 和 INT8**，使用 FP32 累加，从未在硅片中交付 FP8 或 FP4。

它唯一独特的数值方案是 **TruePoint**：把 320 元素点积融合成一次舍入，以 FP32 累加，因此 FP16 乘法器阵列能在归约时接近 FP32 准确率。Groq 报告称，相对 FP32 基准的最大误差约为 0.05%。

究竟 16 bit 是理念选择，还是从未获得低精度更新的数据通路，已经很难与第二代芯片未能交付这一事实分开。SRAM 容量是架构最稀缺的资源，8-bit 权重本可以把模型所需芯片数减半；这样一台受容量限制的机器完全有理由需要 FP8，却没有在硅片中获得它。

这与 Cerebras 只支持 16 bit 的数据通路面临同一个开放问题和同一种矛盾：最缺容量的供应商，却使用最宽的精度计算。

#### 确定性

其他每一种加速器都会隐藏延迟，LPU 则把它**显露**出来。ISA 携带每条指令的执行延迟，数据通路从设计上就是固定延迟，因此编译器能够提前计算每个结果出现的精确周期。

硬件中没有任何东西能打乱这份时间表：没有会未命中的缓存、会阻塞的仲裁器、会预测失败的分支，也没有需要回滚的推测执行。Groq 自己的测量给出了证明：BERT-Large 连续运行 24,240 次，所有结果都落在约 75 µs 的窄区间内；编译器预测延迟与实测值相差不到 2%。

这是 TPU 的直觉——把排程移进编译器、删除会反过来猜测的硬件——再向前一步。TPU 编译器排程的是一颗芯片；LPU 编译器排程的是一整套**系统**，因为确定性也贯穿网络。

它又正好与 Cerebras 相反。Cerebras 核心是**数据流**：任何操作数一到就触发执行；WSE 会对数据做出响应，LPU 则提前按时钟迎接数据。两台机器都删除了调度器，一台用“到达”取而代之，另一台用“时钟”取而代之。

#### 五项押注

1. **确定性胜过容错性。** 删除所有响应式组件——缓存、仲裁器、预测器、重排序缓冲区——让编译器掌控每一个周期。
2. **空间化功能切片。** 把核心拆成切片，让操作数流经各切片；融合直接体现为平面布局，数据复用存在于布线，而不是寄存器文件的搬运舞蹈中。
3. **SRAM 是唯一内存。** 无论容量代价多大都不用 HBM。以模型无法在片上存放为代价，换取单周期固定延迟访问，并接受一个模型必须横跨数百颗芯片。
4. **网络也要排程。** 让芯片自己充当路由器，逐周期编译通信，因此千芯片集群可以成为一个没有交换机、也不会拥塞的确定性程序。
5. **销售延迟，而非吞吐。** 针对 batch 1 的单用户每秒 token 数优化——这是 GPU 最不擅长的区间——把速度本身定价为产品，而不是竞争单 token 成本。

### 扩展

LPU 的扩展方式与这里其他所有架构都不同，因为它没有需要单独构建的纵向扩展 fabric：芯片本身就是交换机。每颗 LPU 最多拥有 16 条芯片间 **RealScale** 链路，板卡上对外暴露 11 条，同时充当计算端点与路由器。

把芯片彼此直接布线，集群就成为一套**无胶合逻辑多处理器**：没有 NIC、交换 ASIC 或机架顶交换机。因为确定性贯穿这些链路，整个集群也会运行在同一份编译期时间表上。

**纵向扩展**：节点包含 8 颗通过 RealScale C2C 全互连的 LPU，形成一个 Dragonfly group，对外呈现为单个高基数虚拟路由器。由软件排程、无交换机、无一致性。

**横向扩展**：沿用同一 fabric 向外延伸。Dragonfly 由节点组成：每机架 9 个节点，即 72 颗芯片，其中一个节点作为热备；规格可扩展到 10,440 颗芯片，每一跳仍遵循编译好的确定性时间表。

#### 纵向扩展

一个节点由 8 颗全互连 LPU 构成：每颗芯片的 7 条链路分别直连另外 7 颗，因此节点内任意芯片之间都只有一跳。每颗芯片剩余 4 条链路——全节点共 32 条——组合成 ISCA 论文所说的“32 端口虚拟路由器”，成为节点接入更大 fabric 的上行链路。

没有基板交换机，也没有一致性地址空间。远程操作数不是通过 load 取得，而是被**安排在某一周期到达**：源芯片在编译器选定的周期注入数据，目的芯片在落地周期消费它。

![Groq LPU 扩展网络](/assets/images/posts/ai-chip-architectures-full/groq-scale.png)

*图 20：8 颗 LPU 全互连形成一个节点——一个对外呈现为高基数虚拟路由器的 Dragonfly group；9 个节点组成 72 芯片机架，其中一个节点是热备。芯片就是路由器：没有 NIC，也没有交换机。编译器逐周期安排每次芯片间传输，即“Scheduled, Not Routed”；近同步链路通过每 256 周期交换一次 Hardware-Aligned Counter 保持锁步，并以 FEC 取代重传，避免重试扰乱时间表。一个 70B 模型会占用整座机架的 SRAM。*

#### 横向扩展

节点之外，各节点连接成 **Dragonfly**：9 个节点构成 72 芯片机架——第 9 个为热备，因此有 64 颗活跃芯片；规格上可扩展到 10,440 颗芯片，任意两颗之间少于 6 跳。

fabric 由**软件排程**：路由与流量控制都移到编译时，论文的表述非常直接——“scheduled, not routed（排程，而非路由）”。没有背压，也没有动态仲裁，因为编译器已经证明接收方届时就绪；链路使用前向纠错而非重传，因为重试会扰乱时间表。

让一整座机架中拥有独立时钟的芯片保持锁步，本身也是一个问题。链路是**近同步（plesiochronous）**的；fabric 通过一棵生成树，每 256 个周期交换一次 **Hardware-Aligned Counter**，维持全局共识时间；周期性 deskew 指令会让各芯片短暂停顿，重新对齐。

Groq 报告的收益是：面对大张量时，8 路 all-reduce 与 A100/NVSwitch 节点相当；面对小张量时则更快，因为排程 fabric 不需要支付动态网络的握手延迟。

成本写在内存押注的物理现实中。一个模型副本不是一台服务器，而是一座甚至八座机架。根据一份分析，约 576 颗芯片运行 Llama-2 70B 时，还要在 LPU 旁配置 144 颗主机 CPU 和 144 TB 主机内存；8-GPU 服务器只需两颗 CPU。

每颗芯片下方的晶圆成本很低——GlobalFoundries 14 nm，据报道低于 6,000 美元，而 H100 级芯片约 16,000 美元——但系统需要数百颗；在解码期间，当 SRAM 承担主要工作时，绝大部分庞大计算能力都处于空闲。

**[SemiAnalysis](https://newsletter.semianalysis.com/p/groq-inference-tokenomics-speed-but)** 直言：当优化目标是延迟时，LPU 可以赢得每 token 物料成本；一旦通过批处理追求每美元吞吐，就会以约一个数量级输给 GPU。这套架构竞争的不是成本，而是速度。

### 软件

编程模型最纯粹地表达了“**编译器就是机器**”：这里**没有 kernel**。

用户把来自 PyTorch、TensorFlow 或 ONNX 的模型交给 Groq 编译器；编译器先降低到一小组张量操作，再静态排程每条指令、每条数据流以及每次芯片间传输。没有人会编写 `wgmma` 或手工调优 tile，因为根本不存在可以针对其调优的动态硬件。

Groq 展示过一个例子：不到 10 人的团队用 4 天就让 LLaMA 跑起来；而同一模型在 GPU 上需要数月手写 kernel 调优。编译器周围的软件栈——profiler、runtime、用于模型导入的 `GroqFlow`——规模很小且闭源；随着公司停止销售板卡、转而销售 token，`GroqFlow` 已于 2025 年归档。

这项转向揭示了架构真正适合的用途。LPU 从结构上就是**只支持推理**——Ross 的说法是，训练是一场本地游戏，推理则是一场全球游戏——并在一件事情上保持不败：单用户解码延迟。

独立测量支持这一主张：**[Artificial Analysis](https://artificialanalysis.ai/providers/groq)** 的结果显示，在开放模型提供商中，Groq 的每秒 token 速度位居前列。它与其他目标并不匹配：模型无法装入一座 SRAM 机架；负载希望通过大批次提升每美元吞吐；或者静态时间表无法表达动态控制流。

系统能够服务 MoE，但数据依赖的专家路由与一个希望提前知道一切的编译器并不自然兼容，Groq 也很少公开如何协调二者。

故事的尾声是，这一切的买方成为了 NVIDIA。2025 年 12 月，NVIDIA 获得 LPU 技术的**非独家授权**，并聘用 Ross 与大部分团队。它并不是一次收购：根据 NVIDIA 自己的 10-K，没有产品、客户合同或股权易手；不过约 130 亿美元的交割付款让媒体普遍称其为收购。

在 GTC 2026 上，这项技术以 **NVIDIA Groq 3 LPU** 再度出现：由 256 颗仅使用 SRAM 的推理芯片组成一座机架，位于 Rubin NVL72 旁边，把 Transformer 拆分到两者之间——GPU 执行注意力，LPU 执行前馈网络和 MoE 层，由 Dynamo 协调交接。

AI 中最具确定性的架构，最终成为最具可编程性架构内部的一颗延迟协处理器。与此同时，GroqCloud 仍在原始 14 nm 硅片上提供 token。

---

## 横向比较

下列所有算力数字均为对应精度下的峰值；除非供应商没有公布口径，否则都指稠密计算。内存带宽采用各架构的原生层级：GPU、TPU 与 Trainium 使用 HBM，Cerebras 与 Groq 使用片上 SRAM 聚合带宽；这些数字**不能直接比较**。纵向扩展带宽也沿用各供应商自己的定义，可能指每芯片聚合、每机架聚合或真实二分带宽。

### 单芯片

| 公司 | 年份 | 芯片 | 加速器内存 | 内存带宽 | 旗舰稠密算力 | TDP | 纵向扩展带宽 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA | 2023 | H100 SXM5 | 80 GB HBM3 | 3.4 TB/s | 1.98 PetaFLOPS FP8 | 700 W | 900 GB/s |
| NVIDIA | 2024 | H200 SXM | 141 GB HBM3e | 4.8 TB/s | 1.98 PetaFLOPS FP8 | 700 W | 900 GB/s |
| NVIDIA | 2024 | B200 | 192 GB HBM3e | 8 TB/s | 4.5 PetaFLOPS FP8 / 9 PetaFLOPS FP4 | 1,000 W | 1.8 TB/s |
| NVIDIA | 2025 | B300 | 288 GB HBM3e | 8 TB/s | 7.5 PetaFLOPS FP8 / 15 PetaFLOPS FP4 | 1,400 W | 1.8 TB/s |
| NVIDIA | 2026 | Rubin | 288 GB HBM4\* | 约 13 TB/s\* | 约 17 PetaFLOPS FP8\* / 约 50 PetaFLOPS FP4\* | 约 1,500 W\* | 3.6 TB/s |
| NVIDIA | 2027 | Rubin Ultra | 1 TB HBM4e\* | 约 32 TB/s\* | 约 33 PetaFLOPS FP8\* / 约 100 PetaFLOPS FP4\* | 约 1,800 W\* | 3.6 TB/s |
| Google | 2023 | TPU v5p | 95 GB HBM2e | 2.8 TB/s | 0.46 PetaFLOPS BF16 | 未披露 | 1.2 TB/s |
| Google | 2025 | TPU Ironwood（v7） | 192 GB HBM3e | 7.4 TB/s | 4.6 PetaFLOPS FP8 | 未披露 | 1.2 TB/s |
| Google | 2026 | TPU v8t Sunfish | 216 GB HBM3e | 6.5 TB/s | 12.6 PetaFLOPS FP4 | 未披露 | 未披露 |
| AMD | 2023 | MI300X | 192 GB HBM3 | 5.3 TB/s | 2.6 PetaFLOPS FP8 | 750 W | 896 GB/s |
| AMD | 2024 | MI325X | 256 GB HBM3e | 6.0 TB/s | 2.6 PetaFLOPS FP8 | 1,000 W | 896 GB/s |
| AMD | 2025 | MI355X | 288 GB HBM3e | 8 TB/s | 10 PetaFLOPS FP8 / 20 PetaFLOPS FP4 | 1,400 W | 1,075 GB/s |
| AMD | 2026 | MI455X | 待定 | 待定 | 约 40 PetaFLOPS FP4\* | 待定 | 未披露 |
| Cerebras | 2021 | WSE-2 | 40 GB SRAM（晶圆上） | 20 PB/s（聚合） | 7.5 PetaFLOPS FP16 | 23 kW（系统） | 一致域即晶圆 |
| Cerebras | 2024 | WSE-3 | 44 GB SRAM（晶圆上） | 21 PB/s（聚合） | 约 15.8 PetaFLOPS FP16\* | 23 kW（系统） | 一致域即晶圆 |
| AWS | 2022 | Trainium1 | 32 GB HBM2e\* | 820 GB/s | 0.19 PetaFLOPS BF16 / FP8 | 未披露 | 未披露 |
| AWS | 2024 | Trainium2 | 96 GB HBM3 | 2.9 TB/s | 1.3 PetaFLOPS FP8 | 约 500 W\* | 1.28 TB/s |
| AWS | 2025 | Trainium3 | 144 GB HBM3e | 4.9 TB/s | 2.5 PetaFLOPS FP8 | 未披露 | 未披露 |
| Groq | 2020 | GroqChip（第一代 TSP / LPU） | 230 MB SRAM | 80 TB/s（片上聚合） | 0.188 PetaFLOPS FP16 | 215 W | 330 GB/s（11 链路板卡） |
| Groq | 2026 | NVIDIA Groq 3 LP30 | 500 MB SRAM | 150 TB/s（片上聚合） | 约 1.2 PetaFLOPS FP8\* | 未披露 | 2.5 TB/s |

### 单机架 / Pod

| 公司 | 年份 | 系统 | 芯片数 | 聚合稠密算力 | 加速器内存总量 | 纵向扩展 fabric 带宽 | 每芯片 NIC | 功耗 | 冷却 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA | 2023 | HGX H100 | 8 | 16 PetaFLOPS FP8 | 640 GB | 7.2 TB/s | 400 Gbps（CX-7） | 约 10 kW | 风冷 |
| NVIDIA | 2024 | HGX H200 | 8 | 16 PetaFLOPS FP8 | 1.1 TB | 7.2 TB/s | 400 Gbps | 约 10 kW | 风冷 |
| NVIDIA | 2024 | GB200 NVL72 | 72 | 360 PetaFLOPS FP8 / 720 PetaFLOPS FP4 | 13.4 TB | 130 TB/s | 800 Gbps（CX-8） | 约 120 kW | 液冷 |
| NVIDIA | 2025 | GB300 NVL72 | 72 | 540 PetaFLOPS FP8 / 1,100 PetaFLOPS FP4 | 20.7 TB | 130 TB/s | 800 Gbps | 约 120 kW | 液冷 |
| NVIDIA | 2026 | NVL144 | 144 | 约 1.2 ExaFLOPS FP8 / 约 3.6 ExaFLOPS FP4 | 约 21 TB | 约 260 TB/s\* | 1.6 Tbps（CX-9） | 约 200 kW\* | 液冷 |
| NVIDIA | 2027 | NVL576（Kyber） | 576 | 约 5 ExaFLOPS FP8 / 约 15 ExaFLOPS FP4 | 约 144 TB | 未披露 | 1.6 Tbps | 约 600 kW\* | 液冷 |
| Google | 2023 | TPU v5p Pod | 8,960 | 4.1 ExaFLOPS BF16 | 852 TB | 三维 torus | ICI 同时承担纵向与横向扩展 | 未披露 | 液冷 |
| Google | 2025 | TPU Ironwood Pod | 9,216 | 42.5 ExaFLOPS FP8 | 1.77 PB | 三维 torus | 光学 OCS | 约 10 MW\* | 液冷 |
| Google | 2026 | TPU v8t Sunfish Pod | 9,600 | 121 ExaFLOPS FP4 | 约 2 PB | Boardfly | 光学 OCS | 未披露 | 液冷 |
| AMD | 2023 | MI300X 8-GPU OAM | 8 | 21 PetaFLOPS FP8 | 1.5 TB | 7.2 TB/s | 400 Gbps | 约 10 kW | 风冷 |
| AMD | 2024 | MI325X 8-GPU OAM | 8 | 21 PetaFLOPS FP8 | 2.0 TB | 7.2 TB/s | 400 Gbps | 约 12 kW\* | 风冷 |
| AMD | 2025 | MI355X 8-GPU OAM | 8 | 80 PetaFLOPS FP8 / 160 PetaFLOPS FP4 | 2.3 TB | 8.6 TB/s | 400 Gbps | 约 16 kW\* | 液冷 |
| AMD | 2026 | Helios（MI455X） | 72 | 1.4 ExaFLOPS FP8 / 2.9 ExaFLOPS FP4 | 31 TB | 260 TB/s | 未披露 | 未披露 | 液冷 |
| Cerebras | 2024 | Condor Galaxy 3 | 64 片晶圆 | 约 1 ExaFLOPS FP16\* | 2.8 TB SRAM + MemoryX | 以太网树 | 1.2 Tb/s 以太网 | 约 1.5 MW\* | 液冷 |
| AWS | 2022 | Trn1 实例 | 16 | 3 PetaFLOPS BF16 | 512 GB | 二维 torus | 约 50 Gbps（EFA） | 未披露 | 风冷 |
| AWS | 2024 | Trn2 UltraServer | 64 | 83 PetaFLOPS FP8 | 6.1 TB | 三维 torus | 200 Gbps（EFAv3） | 未披露 | 风冷 |
| AWS | 2025 | Trn3 UltraServer | 144 | 362 PetaFLOPS FP8 | 20.7 TB | NeuronSwitch | 未披露 | 未披露 | 液冷 |
| Groq | 2022 | GroqRack | 64 颗活跃（安装 72 颗） | 12 PetaFLOPS FP16 | 14 GB SRAM | 3.2 TB/s 二分带宽 | RealScale，无单芯片 NIC | 未披露 | 风冷 |
| Groq | 2026 | NVIDIA Groq 3 LPX | 256 | 315 PetaFLOPS FP8 | 128 GB SRAM + 12 TB DDR5 | 未披露（聚合 C2C 为 640 TB/s） | 未披露 | 未披露 | 液冷 |

\* 表示由分析师推算、按时代推定，或从供应商聚合数字推导的数值；“未披露”表示供应商尚未公开该项规格。

### 这些数字说明了什么

- **单芯片 FP8 已经趋同。** B200 为 4.5 PF，Ironwood 为 4.6 PF，MI355X 为 10 PF，三者相差约 2 倍。单芯片军备竞赛已经十分接近，架构真正分岔的地方是机架和 Pod。
- **HBM 容量是 AMD 持久的胜点。** 2023—2025 年依次达到 192 → 256 → 288 GB，每一代都追平或击败 NVIDIA。NVIDIA 直到 2025 年末的 B300 才在 288 GB 追平；Rubin Ultra 则在 2026 年以每封装 1 TB 重新领先。
- **截至 2026 年，机架级纵向扩展是 NVIDIA 的胜点。** 2024—2025 年，GB200 / GB300 NVL72 是唯一实际交付的一致性机架级域；AMD 只在单机内纵向扩展，直到 Helios 才进入机架级。TPU 绕开了这个问题：其 torus 同时就是机架与集群。
- **TPU Pod 的芯片数量远超任何 NVIDIA 机架。** Ironwood Pod 拥有 9,216 颗芯片、42.5 ExaFLOPS FP8；NVL576 为 576 颗 GPU、约 5 ExaFLOPS FP8。TPU 以“固定单芯片速率 × 巨型 Pod”获得更大的系统聚合算力，代价是较低的单芯片带宽。
- **单芯片功耗快速上升。** Hopper 700 W → Blackwell / MI325X 1,000 W → B300 / MI355X 1,400 W → Rubin Ultra 约 1,800 W（分析师估算）。超过约 1,000 W 后，液冷成为必需；风冷实际上止步于 Hopper。
- **NVIDIA 横向扩展 NIC 带宽每代翻倍。** Hopper 的 CX-7 为 400 Gbps，Blackwell 的 CX-8 为 800 Gbps，Rubin 的 CX-9 为 1.6 Tbps。AMD 落后一代——Pollara 400 → Vulcano 800——反映了 Pensando 较小的装机基础与较晚的整合时间。
- **Cerebras 打破了表格坐标轴。** 它完全没有 HBM，而是 44 GB 晶圆上 SRAM，聚合带宽 21 PB/s；每个稠密 FLOP 可获得约 1.3 byte，而 GPU 各行接近 0.002。代价也写在同一行：总内存少于单颗 H200；每瓦稠密 FLOPs 落后于同期所有 GPU；纵向扩展列为空，因为一致域就是晶圆本身。
- **Trainium 竞争的是经济性，而不是规格表。** 单芯片指标落后——Trn2 的 1.3 PF FP8 约为 MI355X 的四分之一——但 Trn2 UltraServer 在 2024 年与 NVL72 同期实现 64 芯片机架级纵向扩展，只是采用消息传递 torus，而非一致性交叉开关；Trn3 又转向交换式 NeuronSwitch fabric。AWS 从 Nitro 卡到 API 掌控每一层，而一个锚定客户——Anthropic，使用超过 100 万颗 Trainium2——已经在前沿规模验证它。
- **Groq 用容量换取 SRAM 带宽，再以芯片数量扩大内存池。** 第一代 GroqRack 的 64 颗活跃芯片总共只有 14 GB；Groq 3 LPX 扩大到 256 颗芯片、128 GB SRAM，聚合 SRAM 带宽 40 PB/s。其 12 TB DDR5 层以及与 Rubin 配对的方式表明：LPU 补充的是大内存 GPU 机架，而不是取代它。
