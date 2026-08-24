---
title: "AI 芯片架构全景：从内存墙到 GPU、TPU 与数据流机器"
description: "读懂 NVIDIA GPU、Google TPU、AMD Instinct、Cerebras、AWS Trainium 与 Groq LPU 的关键，不是比较一张峰值算力表，而是理解数据放在哪里、怎样移动、由谁调度，以及如何跨芯片扩展。"
date: 2026-08-24 15:00:00 +0800
categories: [人工智能, 计算机体系结构]
tags: [AI芯片, GPU, TPU, LLM推理, 体系结构]
author: Yong
translation: false
image: /assets/images/posts/ai-chip-architectures.png
---

过去十年，AI 芯片从一个相对单一的市场变成了架构实验场：NVIDIA 和 AMD 继续扩展 GPU，Google 与 AWS 设计云内 ASIC，Cerebras 把一整片晶圆变成处理器，Groq 则把动态调度几乎全部交给编译器。

如果只比较“每秒多少 FLOPS”，这些方案很容易被压缩成一张排行榜。但峰值算力只是计算单元在理想条件下的上限。真实模型能跑多快，往往由另一些问题决定：权重和激活存放在哪里？一次计算要搬运多少字节？矩阵单元能否持续吃饱？多颗芯片之间交换张量需要付出多大代价？软件能否把这些资源组织起来？

这篇文章建立一套统一的观察框架，再用它分析六类已经进入实际部署的 AI 加速架构。

> **说明**：本文是独立撰写的原创综述，最初受到 Jacob Peake 的 [AI Chip Architectures](https://www.jacobpeake.com/ai-chip-architectures) 启发。文中的架构描述和参数以厂商技术文档与公开论文为依据，不是对该文章的翻译或改写。

## 一、先别问有多少 FLOPS：问四个问题

理解一颗 AI 芯片，可以从四个问题开始：

1. **数据在哪里？** 是在片外 HBM、片上缓存、软件管理的 SRAM，还是分散在成千上万个计算核心旁边？
2. **数据怎样移动？** 由线程发起加载、DMA 引擎搬运、编译器静态排程，还是沿着固定的数据流网络传播？
3. **计算怎样发生？** 是 SIMT 向量线程、脉动阵列、矩阵引擎、细粒度数据流核心，还是功能切片流水线？
4. **系统怎样扩展？** 多颗芯片通过全互连、环面网络、树形归约还是静态调度网络连接？

这四个问题比单个产品的营销名称更稳定。产品会迭代，数据、计算、调度和通信之间的矛盾不会消失。

## 二、Transformer 到底让芯片做什么

Transformer 的主要计算可以概括为矩阵乘法，穿插归一化、激活、残差连接、采样等逐元素操作。训练与推理看起来都在运行同一个模型，但它们对硬件的压力并不相同。

### 训练与预填充：更像大型矩阵乘矩阵

训练时，大量 token 以批次形式通过同一组权重；前向传播之后还要反向传播并更新参数。推理的预填充（prefill）阶段也会一次处理整段输入上下文。

这两类工作通常形成较大的 GEMM（General Matrix-Matrix Multiplication）。一块权重从内存取出后，可以与许多 token 对应的激活重复使用，因此每搬运一个字节能够完成较多运算，也就是**算术强度较高**。当批次和矩阵尺寸足够大时，瓶颈更可能落在矩阵计算单元，而不是内存带宽。

### 自回归解码：更像矩阵乘向量

解码（decode）阶段一次生成一个 token，后一 token 必须等待前一 token 的结果。批次较小时，每层权重在一次解码步中只被使用很少几次，却仍然需要从显存读取；注意力还要读取不断增长的 KV Cache。

于是问题从“能做多少乘加”变成“能多快把模型参数和 KV Cache 送到计算单元”。连续批处理可以把多个用户的解码步合并，提高权重复用率，但会增加调度复杂度，并可能影响单请求延迟。长上下文下，KV Cache 的容量和带宽又会逐渐取代权重，成为主要压力。

## 三、内存墙：算力增长得比数据移动更快

[Roofline 性能模型](https://digicoll.lib.berkeley.edu/record/136692/files/EECS-2008-134.pdf)用一个简单关系描述程序的性能上界：

```text
可达到的性能 ≤ min（峰值计算性能，内存带宽 × 算术强度）
```

如果一个算子每读取 1 字节只做少量运算，再强的矩阵单元也只能等待数据；如果同一份数据能在片上重复使用很多次，算子才可能接近计算峰值。这就是“内存墙”的核心：晶体管可以堆出越来越多的乘加器，但封装引脚、DRAM 访问、片上布线和跨芯片链路没有以同样速度变得便宜。

AI 系统常用五种方法绕开这堵墙：

- **提高复用**：增大批次，把 GEMV 重新变成更大的 GEMM。
- **分块与融合**：让数据在寄存器或片上 SRAM 中完成更多工作。 [FlashAttention](https://arxiv.org/abs/2205.14135) 的关键贡献正是减少 HBM 与片上 SRAM 之间的读写，而不是近似注意力计算。
- **降低精度**：FP16、BF16、FP8、FP6、FP4 让同样的带宽承载更多数值，但必须通过缩放、累加精度和量化策略控制误差。
- **扩大近计算存储**：增加缓存和 scratchpad，或者像 Cerebras、Groq 那样大规模使用片上 SRAM。
- **改变执行方式**：使用 DMA、静态排程或数据流网络，让搬运与计算重叠，并减少动态控制开销。

接下来所有架构的差异，本质上都是对这五种策略的不同组合。

## 四、NVIDIA GPU：用可编程性和软件生态吸收变化

GPU 的基本哲学是：模型和算子还会持续变化，因此硬件必须保持足够通用，再通过大量并行线程隐藏延迟。

### 计算组织

NVIDIA GPU 由许多流式多处理器（SM）组成。线程以 32 个为一组形成 warp，硬件调度器在大量驻留 warp 之间切换；当某些线程等待数据时，其他线程可以继续执行。普通 CUDA Core 负责地址计算、归一化、激活等标量或向量操作，Tensor Core 则承担密集矩阵乘加。

从 Volta 到 Hopper、Blackwell，Tensor Core 不断支持更低精度和更大的矩阵块。更重要的趋势不是矩阵指令本身变大，而是**指令发起与执行逐渐解耦**：异步矩阵指令、Tensor Memory Accelerator（TMA）、共享内存以及 Blackwell 的 Tensor Memory，让加载、矩阵乘法与逐元素操作能够形成软件流水线。NVIDIA 的 [Blackwell 调优指南](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)仍保留统一的 L1/共享内存层次和 CUDA 编程模型，同时加入第五代 NVLink。

### 内存与软件

GPU 使用多级存储：寄存器、共享内存/L1、L2、HBM。缓存提供通用性，软件可控的共享内存和异步搬运则让性能关键内核显式安排数据复用。

CUDA 是这套设计真正的护城河。cuBLAS、cuDNN、CUTLASS、TensorRT-LLM、NCCL、Triton，以及框架中的大量优化内核，让新模型通常能最快在 NVIDIA 上获得可用性能。代价是硬件必须继续兼容线程、warp 和 kernel 抽象，也必须为调度器、寄存器文件与通用控制逻辑付出面积和能耗。

### 扩展

单机或单机架内，NVLink 与 NVSwitch 提供高带宽 GPU 互连；Blackwell 的第五代 NVLink 标称每 GPU 双向 1.8 TB/s，并可在 NVL72 中形成 72 GPU 的互连域。跨机架则通常使用 InfiniBand 或以太网，通过 RDMA 和 NCCL 执行集合通信。[NVIDIA 的 NVLink 规格](https://www.nvidia.com/en-us/data-center/nvlink/)清楚地区分了 GPU 内互连带宽和整个交换域的聚合带宽，二者不能混为一谈。

**适合什么**：模型变化快、算子种类多、需要成熟工具链，或同时承担训练、推理和 HPC 的场景。

**主要代价**：通用性带来控制与存储层次开销；要接近峰值，常常需要精心设计 kernel、批处理和数据布局。

## 五、Google TPU：让编译器规划脉动阵列

TPU 的哲学更集中：神经网络的大部分时间花在矩阵运算上，那就围绕大型矩阵乘法单元（MXU）设计整台机器。

### 脉动阵列

在脉动阵列中，数据沿着规则的计算单元网格传播；每个单元接收操作数、完成乘加，再把部分数据传给相邻单元。数据进入阵列之后会被重复使用，减少反复访问大容量存储的需要。Google 的 [TPU 架构文档](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)以 128×128 阵列解释了这种工作方式。

与依赖大量动态线程的 GPU 相比，TPU 更相信静态可预测的数据流。XLA 编译器把框架生成的计算图转换为 HLO，完成算子融合、分块、内存规划和跨核心切分，再生成 TPU 程序。硬件可以减少一部分动态调度结构，但性能更加依赖编译器能否看见足够大的计算图，并为真实张量形状找到好计划。

### 存储与互连

现代 TPU 仍然使用 HBM，因此并没有消灭内存墙；它通过 MXU 内的数据复用、编译器分块与大规模并行降低影响。TPU 的系统级差异在于 ICI：芯片被组织成 3D torus，并通过可重构光路选择可用拓扑。公开的 [TPU v4 论文](https://arxiv.org/abs/2304.01433)说明，光路交换不仅为了带宽，也用于提高可用性、模块化和故障绕行能力。TPU v5p 文档则给出了单个 Pod 8,960 颗芯片、每芯片 1,200 GB/s 双向 ICI 带宽的配置。[规格详情](https://docs.cloud.google.com/tpu/docs/v5p)

**适合什么**：计算图相对规则、能够用 JAX/PyTorch XLA 表达、需要 Google Cloud 内大规模训练或推理的工作负载。

**主要代价**：编译时间、形状变化和自定义算子可能成为摩擦点；硬件主要通过 Google Cloud 提供，可移植性和部署选择弱于 GPU。

## 六、AMD Instinct：GPU 路线上的大容量与 Chiplet

AMD Instinct 与 NVIDIA 同属可编程 GPU 路线，但 CDNA 更明确地面向 AI 与 HPC，不承担图形渲染产品线的全部目标。

### Chiplet 与存储

CDNA 采用多个计算裸片与 I/O 裸片组合的封装。这样可以把先进制程用于计算，把内存控制器和互连放到更适合的工艺上，同时避开单一超大裸片的良率压力。Chiplet 不是免费午餐：跨裸片访问需要经过片内 Infinity Fabric，软件必须理解拓扑与局部性。

以 MI350 系列为例，AMD 公布的 [CDNA 4 架构](https://www.amd.com/en/technologies/cdna.html)包含 8 个计算 Chiplet、256 个 Compute Unit、1,024 个 Matrix Core、288 GB HBM3E 与最高 8 TB/s 内存带宽，并支持 MXFP4/6/8 等低精度格式。较大的 HBM 容量对单卡放入更大模型、减少张量并行规模，以及容纳长上下文 KV Cache 都很有价值。

### 执行与软件

AMD Compute Unit 使用 wavefront 执行模型，Matrix Core 负责矩阵运算，向量单元处理其余工作。ROCm 提供编译器、HIP、数学库和推理工具；RCCL 提供 all-reduce、all-gather、reduce-scatter、all-to-all 等集合通信原语。[RCCL 文档](https://rocm.docs.amd.com/projects/rccl/en/docs-6.2.1/)

ROCm 的开放程度和与主流框架的整合持续改善，但硬件支持覆盖、内核成熟度和现成优化数量仍会因模型而异。评估 AMD 时，不能只比较显存和峰值算力，必须在目标模型、框架版本和实际通信拓扑上测量端到端性能。

**适合什么**：需要大 HBM 容量、重视开放软件栈，且愿意针对确定模型完成 ROCm 验证与调优的训练或推理部署。

**主要代价**：与 CUDA 相比，部分新算子和第三方工具的成熟时间可能更晚；Chiplet 和多 GPU 拓扑也要求更精细的局部性优化。

## 七、Cerebras WSE：把“多芯片系统”做进一片晶圆

传统芯片制造会把晶圆切成许多裸片，因为裸片越大，遇到制造缺陷的概率越高。Cerebras 反过来使用整片晶圆，并依靠冗余核心和可绕过缺陷的片上网络提高可制造性。

### 分布式片上 SRAM

WSE 不是一颗拥有少数巨大核心的处理器，而是由大量细粒度计算核心、局部 SRAM 和网格互连组成。WSE-3 的公开规格是 4 万亿晶体管、90 万个 AI 核心、44 GB 片上 SRAM 和 21 PB/s 聚合片上内存带宽。[Cerebras WSE-3 规格](https://www.cerebras.ai/press-release/cerebras-announces-third-generation-wafer-scale-engine)

这些数字与 HBM GPU 不能直接逐项比较：21 PB/s 是分布在晶圆上大量本地 SRAM 的聚合带宽，只有当数据和计算被正确映射到邻近核心时才能发挥价值。它真正改变的是边界条件——许多原本需要跨 GPU、跨封装传输的数据，可以在晶圆内的低延迟网络上移动。

### 容量问题没有消失，只是被重新组织

44 GB 仍放不下大型模型的全部参数。Cerebras 的训练系统使用 MemoryX 保存参数，并通过 SwarmX 向 WSE 广播权重、归约梯度。SwarmX 是针对权重广播和梯度归约设计的树形网络，而不是通用 GPU 网络。[Cerebras 对 MemoryX/SwarmX 的说明](https://www.cerebras.ai/blog/announcing-the-cerebras-architecture-for-extreme-scale-ai)

推理时也可以把模型按层切到多台 CS 系统，或者让其他加速器负责预填充、WSE 负责低延迟解码。这体现了一个重要趋势：不是强迫同一种芯片同时优化 prefill 和 decode，而是让两阶段使用不同硬件。

**适合什么**：能够映射到细粒度数据流、追求大规模训练简化或极高解码速度，并接受专用系统交付方式的场景。

**主要代价**：编程、部署和运维与通用 GPU 集群不同；片上 SRAM 的惊人带宽不等于大模型参数天然全部在片上，系统仍需处理外部容量与多系统扩展。

## 八、AWS Trainium：云厂商定制的编译器主导型加速器

Trainium 的目标不是成为所有环境里的通用芯片，而是在 AWS 内为训练和推理提供更可控的性能与成本。

### NeuronCore 的分工

Trainium 的 NeuronCore 把不同工作交给专用引擎：Tensor Engine 负责矩阵乘法，Vector Engine 负责归一化和归约，Scalar Engine 负责逐元素函数，GPSIMD Engine 提供更通用的可编程能力。片上 SBUF 是软件管理的工作区，PSUM 用于保存并累加矩阵乘法的部分和，HBM 则提供大容量存储。

这种结构与 GPU 的硬件缓存、动态线程调度不同。Neuron 编译器和 NKI（Neuron Kernel Interface）显式安排 HBM、SBUF、PSUM 之间的数据移动；DMA 可以与计算引擎并行工作。AWS 的 [NKI 架构指南](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/architecture/trainium_inferentia2_arch.html)详细说明了这些存储层次和 DMA 约束。

### Trainium2 的系统设计

Trainium2 每芯片包含 8 个 NeuronCore-v3、96 GiB HBM、2.9 TB/s HBM 带宽和 1.28 TB/s NeuronLink-v3。专用 CC-Core 在计算同时处理 all-reduce 等集合通信。[Trainium2 官方架构](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html)

框架代码可以通过 PyTorch/XLA 或 JAX 进入编译流程；当编译器生成的通用内核不够好时，开发者可以用 NKI 控制分块、数据搬运和专用引擎。它处于“完全由编译器处理”和“像 CUDA 一样手写底层内核”之间。

**适合什么**：工作负载长期运行在 AWS，模型规模足以摊薄编译和移植成本，并且团队重视云内价格、容量与网络协同。

**主要代价**：软件和硬件与 AWS 深度绑定；新模型往往需要等待 Neuron 支持或投入 NKI 优化，生态广度仍小于 CUDA。

## 九、Groq LPU：把时间表写进程序

Groq 选择了一条更激进的路线：删掉缓存、乱序执行、动态仲裁等会产生不确定延迟的结构，让编译器提前决定每条指令和每份数据在每个周期的位置。

### 功能切片与张量流

公开的第一代架构最初名为 Tensor Streaming Processor（TSP），后来面向大模型推理称为 LPU。它不复制许多完整核心，而是把矩阵单元、向量单元、存储和交换功能做成贯穿芯片的功能切片。数据沿水平 stream 流过这些切片，指令沿另一方向到达；编译器确保二者在正确周期相遇。

Groq 的 [ISCA 2020 论文](https://groq.com/wp-content/uploads/2020/06/ISCA-TSP.pdf)公开了第一代 TSP 的关键结构：320 路向量抽象、四个 320×320 矩阵单元，以及约 220 MiB 的片上 SRAM。这里引用的是公开论文中的首代实现，不代表后续商业 LPU 的全部规格。

### 确定性带来的收益与约束

静态排程意味着硬件不需要在运行时猜测数据何时到达，也不需要依靠大量线程掩盖缓存未命中。单个请求的延迟可预测，流水线能够紧密衔接，这非常适合低 batch、追求 token 间延迟的解码。

但片上 SRAM 容量远小于大模型权重，模型必须跨许多芯片切分；这会让芯片数量首先由容量而不是算力决定。动态形状、复杂控制流和数据相关路由也更难静态安排。Groq 的优势不是“对所有负载都更便宜”，而是在确定的模型上用更多专用硬件换取稳定、极低的响应延迟。

**适合什么**：模型已经编译完成、请求强调低延迟和确定性、服务形态相对固定的在线推理。

**主要代价**：SRAM 容量导致大模型跨芯片部署；静态编译降低了面对动态工作负载时的灵活性。

## 十、互连不是配件，而是架构的一部分

模型无法放入一颗芯片后，性能取决于最慢的通信步骤。常见并行方式对网络提出不同要求：

- **数据并行**需要归约梯度，核心原语是 all-reduce 或 reduce-scatter。
- **张量并行**在每层内部切分矩阵，需要频繁、低延迟的 all-reduce/all-gather。
- **流水线并行**传递层间激活，通信量较集中，但会产生流水线气泡。
- **专家并行（MoE）**需要 all-to-all，把 token 路由到不同专家，对网络拥塞和尾延迟尤其敏感。

系统通常分成两个扩展范围：

### Scale-up：把少量芯片做成一台大机器

Scale-up 追求高带宽、低延迟和更紧密的内存访问语义。例如 NVIDIA 的 NVLink/NVSwitch、AMD 的 Infinity Fabric、TPU 的 ICI、Trainium 的 NeuronLink。这个范围适合每层都要通信的张量并行和专家并行。

### Scale-out：把许多机器组成集群

Scale-out 依赖 InfiniBand、RoCE 或数据中心网络，延迟更高、地址空间通常分离，通过 RDMA 和集合通信库显式搬运数据。它适合数据并行和较粗粒度的流水线并行。

不能只看“总带宽”宣传值。至少还要问：数字是单向还是双向、单芯片还是全系统聚合？拓扑是全互连、环、树还是 3D torus？真正的二分带宽是多少？集合通信能否和计算重叠？发生故障时能否绕行？

| 架构 | 芯片内/近芯片通信 | 规模化方式 | 调度倾向 |
| --- | --- | --- | --- |
| NVIDIA GPU | 缓存、共享内存、TMA | NVLink/NVSwitch + InfiniBand/以太网 | 硬件线程调度 + 软件流水线 |
| Google TPU | MXU、片上缓冲 | ICI 3D torus + 可重构光路 | XLA 静态规划 |
| AMD Instinct | Cache、LDS、Infinity Fabric | GPU 间 Infinity Fabric + RCCL 网络 | Wavefront 调度 + 软件优化 |
| Cerebras WSE | 分布式 SRAM、片上网格 | 晶圆内 fabric + SwarmX/MemoryX | 细粒度数据流 |
| AWS Trainium | SBUF、PSUM、DMA | NeuronLink + CC-Core + EFA | 编译器规划 + NKI |
| Groq LPU | 片上 SRAM、stream | 编译后的芯片间数据流 | 周期级静态排程 |

## 十一、六条路线，实际上是在做六种取舍

| 路线 | 最核心的赌注 | 最突出的优势 | 最需要警惕的限制 |
| --- | --- | --- | --- |
| NVIDIA GPU | 工作负载会变化，可编程性和生态最重要 | 通用、成熟、模型支持快 | 通用控制开销，优化复杂 |
| Google TPU | 大计算图可以由编译器映射到规则矩阵机器 | 规则负载效率高，Pod 扩展强 | 编译与云平台依赖 |
| AMD Instinct | GPU 模型可与 Chiplet、大 HBM 和开放栈结合 | 容量大，GPU 使用习惯相近 | 软件成熟度需按模型验证 |
| Cerebras WSE | 消除传统裸片边界能大幅降低数据移动 | 片上带宽极高，系统形态独特 | 专用部署，外部容量仍要管理 |
| AWS Trainium | 云内软硬件协同可以优化成本和规模 | AWS 集成、专用通信与编译栈 | 平台绑定，生态较年轻 |
| Groq LPU | 确定性比动态适应更适合低延迟推理 | 可预测、低 token 延迟 | SRAM 容量小，静态工作负载更合适 |

这张表不应该被当成简单排名。不同厂商公布的 FLOPS 可能使用不同精度、稀疏度、功耗边界和系统范围；SRAM 聚合带宽也不能与 HBM 带宽直接比较。真正有意义的是在同一个模型、相同精度、相同服务目标下测量端到端结果。

## 十二、怎样为实际项目选芯片

### 大规模训练

优先关注软件成熟度、集合通信效率、故障恢复、检查点速度和长期可获得的集群容量。峰值算力如果无法转化为较高的 Model FLOPs Utilization（MFU），价值有限。NVIDIA、TPU、AMD 和 Trainium 都能进入候选，但迁移成本、云平台位置和已有内核决定了实际选择。

### 高吞吐在线推理

关注每美元 token 吞吐、连续批处理效率、量化内核、显存容量和功耗。GPU 与大 HBM 加速器通常能通过 batch 摊薄权重读取；如果模型能在更少芯片中放下，还能减少张量并行通信。

### 极低延迟推理

关注 batch=1 的每 token 延迟、调度抖动和尾延迟，而不是最大 batch 下的吞吐。Groq 的静态流水线、Cerebras 的片上 SRAM，以及为低延迟精心调优的 GPU 服务栈属于不同解法。测试必须包含真实并发和上下文长度。

### 长上下文推理

KV Cache 会同时消耗容量和带宽。需要考察 HBM 容量、分页注意力、KV 量化、GQA/MQA 支持、上下文并行，以及 prefill/decode 是否能够分离部署。只看模型权重能否装下远远不够。

### 自定义模型与快速研究

如果新注意力机制、稀疏路由或自定义算子变化频繁，成熟的 GPU kernel 生态通常风险最低。编译器主导的 ASIC 也能获得高性能，但前提是编译器已经理解这些模式，或者团队愿意投入底层 DSL 优化。

## 十三、下一代 AI 芯片会趋同吗

短期内不会。相反，各条路线正在互相吸收对方的思想：

- GPU 增加更多异步 DMA、专用矩阵存储和静态描述符，减少 warp 亲自搬数据的工作。
- TPU、Trainium 等编译器主导架构提供更低层的 kernel DSL，为开发者保留“逃生通道”。
- 所有厂商都在采用更低精度、块级缩放和更强的量化支持。
- Chiplet、先进封装与机架级互连，让“芯片”边界从裸片扩展到封装、服务器乃至整机架。
- Prefill 与 decode 的需求差异推动推理分离：高算术强度阶段和带宽受限阶段可能由不同硬件完成。

最终胜出的不一定是拥有最激进单项指标的芯片，而是能把模型、编译器、内存、互连、运行时和运维工具组合成稳定系统的平台。

## 结语

AI 芯片的竞争表面上是矩阵单元的竞争，深层却是数据移动方式的竞争。

GPU 用线程和缓存拥抱不确定性；TPU 和 Trainium 用编译器提前规划；Cerebras 用晶圆级空间消除大量边界；Groq 用静态时间表换取确定性；AMD 则在 GPU 模型中押注 Chiplet、大容量 HBM 与开放软件栈。

当你再次看到一张 AI 芯片峰值算力表时，先暂时忽略最大的数字，回到最开始的四个问题：数据在哪里，如何移动，如何计算，以及如何扩展。答案往往已经决定了真实工作负载的上限。

## 参考资料

- Samuel Williams、Andrew Waterman、David Patterson：[Roofline 性能模型](https://digicoll.lib.berkeley.edu/record/136692/files/EECS-2008-134.pdf)
- Tri Dao 等：[FlashAttention](https://arxiv.org/abs/2205.14135)
- NVIDIA：[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)、[NVLink 与 NVLink Switch](https://www.nvidia.com/en-us/data-center/nvlink/)
- Google：[TPU 架构](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)、[TPU v4 论文](https://arxiv.org/abs/2304.01433)、[TPU v5p 规格](https://docs.cloud.google.com/tpu/docs/v5p)
- AMD：[CDNA 架构](https://www.amd.com/en/technologies/cdna.html)、[RCCL 文档](https://rocm.docs.amd.com/projects/rccl/en/docs-6.2.1/)
- Cerebras：[WSE-3](https://www.cerebras.ai/press-release/cerebras-announces-third-generation-wafer-scale-engine)、[MemoryX 与 SwarmX](https://www.cerebras.ai/blog/announcing-the-cerebras-architecture-for-extreme-scale-ai)
- AWS：[Trainium2 架构](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html)、[NKI 架构指南](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/nki/guides/architecture/trainium_inferentia2_arch.html)
- Groq：[Think Fast: A Tensor Streaming Processor](https://groq.com/wp-content/uploads/2020/06/ISCA-TSP.pdf)
