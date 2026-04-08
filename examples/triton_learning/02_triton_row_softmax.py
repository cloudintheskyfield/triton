"""
脚本目的：
这个脚本实现一个按行计算的 Triton softmax，它是从基础 kernel 过渡到 attention 的关键桥梁。

你会在这里看到：
1. 每个 program 处理矩阵的一整行
2. 如何在 Triton 里做行内归约，例如 `max` 和 `sum`
3. softmax 的数值稳定写法为什么要先减去最大值
4. 为什么 softmax 是理解 fused attention 前必须掌握的算子

建议怎么学：
如果你能看懂这个脚本，就已经开始接近 attention 的核心了，
因为 attention 本质上就是 `qk` 分数、row-wise softmax、再乘以 `v`。
"""

import torch

import triton
import triton.language as tl
from _runtime_utils import format_runtime_message, get_triton_runtime


DEVICE, HAS_TRITON_DEVICE, TRITON_ERROR = get_triton_runtime()


@triton.jit
def row_softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 负责一整行，因此 grid 只需要覆盖行数。
    row_id = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = input_ptr + row_id * input_row_stride
    # BLOCK_SIZE 可能大于真实列数，所以越界位置填 -inf，避免影响后续 max。
    row = tl.load(row_start + col_offsets, mask=mask, other=-float("inf"))

    # softmax 的数值稳定写法：先减去当前行的最大值。
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator

    output_row = output_ptr + row_id * output_row_stride
    tl.store(output_row + col_offsets, out, mask=mask)


def row_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, "expected a 2D tensor"

    if not HAS_TRITON_DEVICE:
        # 没有 Triton GPU backend 时，直接退回到 PyTorch softmax。
        return torch.softmax(x, dim=1)

    n_rows, n_cols = x.shape
    # Triton 常常偏好 2 的幂大小，这里把列数向上取到最近的 2 的幂。
    block_size = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    grid = (n_rows,)
    row_softmax_kernel[grid](
        out,
        x,
        x.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return out


def main():
    torch.manual_seed(0)

    x = torch.randn(4, 8, device=DEVICE)
    result = row_softmax(x)
    torch_out = torch.softmax(x, dim=1)

    print(format_runtime_message(HAS_TRITON_DEVICE, TRITON_ERROR))
    print("input shape:", tuple(x.shape))
    # 如果实现正确，这个误差应该非常小。
    print("max abs diff:", torch.max(torch.abs(result - torch_out)).item())
    print("result:")
    print(result)


if __name__ == "__main__":
    main()
