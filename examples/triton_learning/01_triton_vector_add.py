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
"""

import torch

import triton
import triton.language as tl
from _runtime_utils import format_runtime_message, get_triton_runtime


DEVICE, HAS_TRITON_DEVICE, TRITON_ERROR = get_triton_runtime()


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
    # 取当前 Triton program instance 在第 0 维 grid 上的编号，axis 只能为 0、1 或 2
    # program_id：Triton 启动 kernel 时，会按照一个 grid 发射很多个 program，每个 program 负责处理一块数据，tl.program_id
    #               就是拿到  "我现在是第几个 program"
    pid = tl.program_id(axis=0)

    # 我负责的起始下标志
    block_start = pid * BLOCK_SIZE

    # 偏移量
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 当元素总数不是 BLOCK_SIZE 的整数倍时，最后一个 program 可能越界，因此要加 mask。
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not HAS_TRITON_DEVICE:
        # 没有 Triton GPU backend 时，用 PyTorch 保持同样的功能结果。
        return x + y

    out = torch.empty_like(x)
    n_elements = out.numel()
    # grid 决定总共启动多少个 program。
    # 这里是一维 grid，每个 program 处理 BLOCK_SIZE 个元素。
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
    print("max abs diff:", torch.max(torch.abs(result - torch_out)).item())
    print("sample result:", result[:8])


if __name__ == "__main__":
    main()
