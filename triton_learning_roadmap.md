# Triton 学习路线图

这份路线图默认你已经完成了对 Triton 项目整体定位的第一轮了解，并且目标明确为：

**最终能够用 Triton 自己实现并理解一个 attention 算子。**

## 学习目标拆解

为了做到这一点，建议把目标拆成 4 个层次：

1. 理解 attention 的数学形式，而不是一开始就陷进复杂 kernel。
2. 掌握 Triton 的基础编程模型：program、grid、pointer、mask、load/store。
3. 掌握 attention 真正依赖的核心算子：tile 化矩阵乘、row-wise softmax、块内归约、数值稳定。
4. 阅读并拆解 Triton 官方 fused attention 实现，最后自己写一个“可运行的简化版”。

## 推荐学习顺序

### 第一阶段：补齐 Python + GPU Kernel 的最低必要知识

目标：先会“看懂”和“改动” Triton Python kernel。

建议重点掌握：

- Python 基础语法：函数、装饰器、切片、广播、上下文管理、lambda
- PyTorch 张量基础：shape、stride、contiguous、dtype、device
- GPU 基本概念：thread/block/grid 的直觉，global memory / shared memory / register 的层次
- 矩阵乘和 softmax 的数学形式

你不需要先把 CUDA 学全，但至少要知道：

- 为什么要分块（tiling）
- 为什么要 mask
- 为什么访存模式会影响性能
- 为什么 softmax 要做减最大值的数值稳定

## 第二阶段：按教程建立 Triton 直觉

推荐阅读顺序：

1. `examples/triton_learning/00_torch_attention_baseline.py`
2. `examples/triton_learning/01a_pointer_offset_mask_walkthrough.py`
3. `examples/triton_learning/01_triton_vector_add.py`
4. `examples/triton_learning/02_triton_row_softmax.py`
5. `examples/triton_learning/03_attention_math_walkthrough.py`
6. `python/tutorials/01-vector-add.py`
7. `python/tutorials/02-fused-softmax.py`
8. `python/tutorials/03-matrix-multiplication.py`
9. `python/tutorials/06-fused-attention.py`

每一阶段应该重点看什么：

### 0. `00_torch_attention_baseline.py`

这是最前面的数学热身。

你要看懂：

- `scores = q @ k^T` 在算什么
- 为什么要除 `sqrt(d)`
- `softmax(scores)` 的维度是什么
- `probs @ v` 为什么会得到输出

完成标准：

- 你能不用 Triton，只用数学解释 attention 前向

### 0.5. `01a_pointer_offset_mask_walkthrough.py`

这是从“数学/张量”过渡到“GPU 访存”的关键台阶。

你要看懂：

- pointer 只是“数组起点”的抽象
- offset 是“相对起点偏移多少”
- `pointer + offset` 为什么就代表某个元素位置
- `mask` 为什么能保护越界访问
- `load/store` 本质上是在“批量按地址读写”

完成标准：

- 你能解释 `x_ptr + offsets` 到底表示什么
- 你能解释 `tl.load(..., mask=mask)` 为什么不是只读一个数

### 1. `01-vector-add.py`

你要看懂：

- `@triton.jit` 在做什么
- `tl.program_id(axis=0)` 怎么决定当前 program 处理哪一段数据
- `tl.arange` 如何生成块内 offset
- `tl.load` / `tl.store` 为什么要带 `mask`
- host 侧 wrapper 如何组织 grid

完成标准：

- 你能自己改 `BLOCK_SIZE`
- 你能解释 `grid = lambda meta: (...)` 的含义
- 你能写出一个自己的 `add + relu`

### 2. `02-fused-softmax.py`

这是 attention 前最重要的过渡。

你要看懂：

- 为什么 softmax 是 memory-bound
- 为什么每一行最好一次性读进片上内存
- `tl.max` / `tl.sum` 这样的归约是怎么工作的
- 数值稳定写法：`x - max(x)`
- kernel 融合为什么比 PyTorch 多个算子拼接更快

完成标准：

- 你能自己写一个 row-wise softmax
- 你能解释 fused softmax 比 naive 版少了哪些内存读写

### 3. `03-matrix-multiplication.py`

这是理解 attention 中 `QK^T` 和 `PV` 的基础。

你要看懂：

- 二维 grid 如何映射到输出 tile
- A/B/C 三个矩阵的 pointer arithmetic
- `tl.dot` 的使用方式
- `BLOCK_M / BLOCK_N / BLOCK_K` 的意义
- autotune 在选什么

完成标准：

- 你能解释一个 program 为什么只负责输出矩阵的一小块
- 你能看懂 K 维循环累加

### 4. `06-fused-attention.py`

这时再读官方 fused attention，难度会显著下降。

第一遍不要追求全部细节都懂，先抓主线：

- 整体输入输出是什么
- 前向过程为什么分块
- `qk`、`p`、`acc`、`m_i`、`l_i` 分别代表什么
- 为什么需要 online softmax
- causal mask 是在哪一步加进去的

完成标准：

- 你能把 `_attn_fwd_inner` 的主循环翻译成数学步骤
- 你能解释 `m_i` / `l_i` 为什么能实现分块 softmax

## 第三阶段：围绕 attention 的专项学习

目标：从“能看”变成“能自己拆出来实现”。

建议按下面顺序做小项目：

1. 用 PyTorch 写一个最朴素的单头 attention
2. 自己写一个只做 `QK^T` 的 Triton kernel
3. 自己写一个只做 row-wise softmax 的 Triton kernel
4. 自己写一个 softmax(QK^T) @ V 的分步版本
5. 最后再尝试做 fused attention

这里最关键的知识点：

- attention 的输入布局：`[B, H, N, D]`
- 为什么通常按 sequence 维度 `N` 分块
- 为什么不能直接存整个 `QK^T`
- online softmax 的状态量如何跨 tile 维护
- causal attention 的 mask 如何映射到 tile 上

## 建议你重点掌握的源码入口

如果你是“边学源码边实践”的路线，建议优先关注这些文件：

- `python/tutorials/01-vector-add.py`
- `python/tutorials/02-fused-softmax.py`
- `python/tutorials/03-matrix-multiplication.py`
- `python/tutorials/06-fused-attention.py`
- `python/triton/language/core.py`
- `python/triton/runtime/jit.py`

理解顺序建议是：

1. 先会写 kernel
2. 再理解 Triton language API
3. 最后再深挖 runtime / compiler / MLIR / LLVM

## 不建议一开始深入的内容

如果你的目标是“先实现一个 attention 算子”，下面这些暂时不用先啃：

- Triton C++/MLIR/LLVM 全链路细节
- 后端 lowering 的所有 pass
- NVIDIA / AMD 后端差异的全部实现
- Gluon / experimental 的高级特性

这些内容应该放在你已经能稳定写出几个 Triton kernel 之后再看。

## 4 周学习计划

### 第 1 周：建立最小可运行认知

目标：

- 跑通 `00_torch_attention_baseline.py`
- 跑通 `01a_pointer_offset_mask_walkthrough.py`
- 看懂 `01_triton_vector_add.py`
- 看懂 `02_triton_row_softmax.py`

输出：

- 不再害怕 `x_ptr + offsets` / `mask=mask` 这种写法
- 自己能画出 Triton program 与数据块之间的对应关系
- 自己能解释 softmax 的数值稳定写法

### 第 2 周：掌握矩阵乘和 tile 思维

目标：

- 精读 `03-matrix-multiplication.py`
- 理解 `BLOCK_M/BLOCK_N/BLOCK_K`
- 手改参数并观察结果和性能

输出：

- 能解释 attention 里的 `QK^T` 本质上就是 matmul
- 能理解为什么 attention 必须 tile 化

### 第 3 周：attention 数学与分步实现

目标：

- 跑通 PyTorch 版 attention baseline
- 先实现非 fused 的 attention 分步流程
- 理解 causal mask 和 scale

输出：

- 自己写出 `scores = q @ k^T`
- `probs = softmax(scores)`
- `out = probs @ v`

### 第 4 周：阅读并模仿 fused attention

目标：

- 精读 `06-fused-attention.py`
- 对照自己的分步实现理解 online softmax
- 尝试写一个“简化版单头 attention kernel”

输出：

- 能解释 fused attention 的主循环
- 能做一个简化的 attention 前向实现

## 你现在最适合做的事

结合你目前的笔记状态，我建议你马上按这个顺序开始：

1. 先运行 `examples/triton_learning/00_torch_attention_baseline.py`
2. 再运行 `examples/triton_learning/01a_pointer_offset_mask_walkthrough.py`
3. 然后运行 `examples/triton_learning/01_triton_vector_add.py`
4. 再运行 `examples/triton_learning/02_triton_row_softmax.py`
5. 最后运行 `examples/triton_learning/03_attention_math_walkthrough.py`

这样你会先把 attention 数学、指针/偏移/mask 直觉、Triton 基础接起来，再去读官方 `06-fused-attention.py`，会轻松很多。

## 下一步里程碑

当你完成下面 3 件事时，就可以正式进入 attention kernel 实战：

- 你能独立解释 vector add / softmax / matmul 三个教程
- 你能写出一个 PyTorch 版单头 attention baseline
- 你能说清楚 online softmax 为什么是 fused attention 的核心

到那一步之后，最自然的下一任务就是：

**实现一个单头、前向、可选 causal mask 的简化版 Triton attention 算子。**
