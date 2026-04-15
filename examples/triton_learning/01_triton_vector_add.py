"""
脚本目的：
这是最小的 Triton kernel 入门例子，用向量加法帮助你建立 Triton 的基本编程直觉。

你会在这里看到：
1. `@triton.jit` 如何定义一个 kernel
2. 一个 program 如何处理一小块连续数据
3. `tl.arange`、`tl.load`、`tl.store`、`mask` 的基本用法
4. host 端如何用 grid 启动 Triton kernel

建议怎么学：
不要把它只当成“向量加法”。
真正要理解的是：Triton 是如何把一个大问题切成很多个小 program 去并行处理的。

阅读建议：
如果你第一次看 `x_ptr + offsets`、`mask=mask` 这些写法会发懵，
建议先看同目录下的 `01a_pointer_offset_mask_walkthrough.py`，
先把地址、偏移、屏蔽这些概念吃透，再回来看这个脚本。
"""

import torch

import triton
import triton.language as tl
from _runtime_utils import format_runtime_message, get_triton_runtime


DEVICE, HAS_TRITON_DEVICE, TRITON_ERROR = get_triton_runtime()


# 这份脚本最关键的学习目标，不是“向量加法”本身，而是理解下面这组固定模式：
#
# 1. 先用 `tl.program_id(axis=0)` 找到“我是第几个 program”
# 2. 再用 `block_start + tl.arange(...)` 生成这一整个 block 的访问下标
# 3. 用 `mask` 保护越界访问
# 4. 用 `tl.load` 把一整个 block 的数据读到寄存器里
# 5. 做计算
# 6. 用 `tl.store` 把结果写回去
#
# 这套模式会在 Triton 绝大多数基础 kernel 里不断出现。


# `kernel` 可以先理解成：一段专门给 GPU 并行执行的小程序。
# 和普通 Python 函数不同，它不是在 CPU 上一行一行跑，而是会被发到 GPU 上同时执行很多份。
#
# `@triton.jit` 的意思可以先记成：
# “把下面这个 Python 风格写出来的函数，交给 Triton 在运行时按需编译成 GPU kernel”。
# 这里的 JIT 是 Just-In-Time，也就是“即时编译”：
# 不是你写下定义时立刻编译，而是等你真正 launch 这个 kernel 时，
# Triton 才会根据这次的参数和 meta-parameters 生成/编译对应的底层 GPU 代码。
#
# 所以 `add_kernel` 更像“一个 GPU 工作模板”，而不是一个立刻执行的普通函数。
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 每个 program 负责处理一个长度为 BLOCK_SIZE 的连续片段。
    #
    # 这里的“program”可以先理解成：
    # Triton 启动 kernel 后，会并行发射很多个一模一样的小工作单元，
    # 每个小工作单元只负责全部数据中的一小块。
    #
    # `tl.program_id(axis=0)` 拿到的就是：
    # “在第 0 维 grid 上，我是第几个小工作单元？”
    #
    # 比如：
    # - pid = 0 处理第 0 块
    # - pid = 1 处理第 1 块
    # - pid = 2 处理第 2 块
    pid = tl.program_id(axis=0)

    # `block_start` 表示当前这个 program 负责的数据块，从全局向量的哪个位置开始。
    #
    # 如果：
    # - BLOCK_SIZE = 1024
    # - pid = 3
    #
    # 那么这个 program 负责的就是：
    # - [3072, 3073, ..., 4095]
    block_start = pid * BLOCK_SIZE

    # `tl.arange(0, BLOCK_SIZE)` 会生成：
    # [0, 1, 2, ..., BLOCK_SIZE-1]
    #
    # 把它加到 block_start 上，就得到这一整个 block 的全局下标。
    #
    # 所以 `offsets` 不是一个单独的整数，
    # 而是一整批要访问的位置。
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 当向量长度不是 BLOCK_SIZE 的整数倍时，
    # 最后一个 program 可能会有一部分位置越界。
    #
    # 例如：
    # - n_elements = 1000
    # - BLOCK_SIZE = 256
    # 最后一个 block 只会有一部分位置合法。
    #
    # `mask` 的作用就是：
    # - True：这个位置可以安全访问
    # - False：这个位置不要读，也不要写
    mask = offsets < n_elements

    # `x_ptr` 可以理解成输入张量 x 的“起始地址”。
    # `x_ptr + offsets` 表示：
    # 从起始地址出发，跳到 offsets 对应的那些元素位置。
    #
    # `tl.load(...)` 的作用是：
    # 从这些位置把一整批元素读出来。
    #
    # 所以这句不是“读一个值”，而是“按一批下标读一批值”。
    #
    # `mask=mask` 表示：
    # 只有 mask 为 True 的位置才真的去读，
    # False 的位置不会访问显存，从而避免越界。
    x = tl.load(x_ptr + offsets, mask=mask)

    # 对 y 做的事情完全一样：
    # 用同一组 offsets 取出 y 中对应位置的一整批元素。
    y = tl.load(y_ptr + offsets, mask=mask)

    # 这里的 `x + y` 也不是两个标量相加，
    # 而是两个 block 内向量逐元素相加。
    #
    # 你可以把它先想成：
    # result[i] = x[i] + y[i]
    #
    # 只不过这里的 i 是当前 block 内的一组位置。
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not HAS_TRITON_DEVICE:
        # 没有 Triton GPU backend 时，用 PyTorch 保持同样的功能结果。
        return x + y

    # 先准备输出张量。
    out = torch.empty_like(x)
    n_elements = out.numel()

    # `grid` 决定要启动多少个 program。
    #
    # 这里是一维 grid，所以返回的是一个单元素 tuple。
    #
    # `triton.cdiv(a, b)` 表示“向上整除”：
    # 比如 1000 个元素，每个 program 处理 256 个，
    # 那么需要 ceil(1000 / 256) = 4 个 program。
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # 这里不是普通函数调用 `add_kernel(...)`。
    # `add_kernel[grid](...)` 的意思是：
    # 1. 用 `[grid]` 指定要启动多少个 program
    # 2. 把这个 `@triton.jit` 函数当成 GPU kernel 发射出去
    # 3. 如果这是第一次遇到这组配置，Triton 往往会先即时编译，再执行
    #
    # `BLOCK_SIZE=1024` 是编译期常量（tl.constexpr），
    # Triton 会把它当成“生成这份 kernel 时就已知的值”来做专门化。
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


def main():
    torch.manual_seed(0)

    x = torch.rand(1024, device=DEVICE)
    y = torch.rand(1024, device=DEVICE)

    # 有 Triton GPU 时这里会走 Triton kernel；否则会自动回退到 PyTorch。
    result = add(x, y)
    torch_out = x + y

    print(format_runtime_message(HAS_TRITON_DEVICE, TRITON_ERROR))
    print("device:", DEVICE)
    print("这份脚本里真正值得重点理解的是：program_id -> offsets -> mask -> load/store")
    print("max abs diff:", torch.max(torch.abs(result - torch_out)).item())
    print("sample result:", result[:8])


if __name__ == "__main__":
    main()
