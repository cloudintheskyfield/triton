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

阅读建议：
第一次看这个脚本时，不要急着盯 `tl.max` 和 `tl.sum` 的细节。
先抓住主线：

1. 一个 program 负责一整行
2. 把这一行读进来
3. 先减最大值做数值稳定
4. 算 exp
5. 求和
6. 再除回去
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
    # 每个 program 负责一整行，因此 grid 只需要覆盖“行数”。
    #
    # 如果输入是 [n_rows, n_cols]，
    # 那么：
    # - row_id = 0 负责第 0 行
    # - row_id = 1 负责第 1 行
    # - ...
    row_id = tl.program_id(axis=0)

    # `col_offsets` 代表这一行里所有要访问的列位置。
    #
    # 例如 BLOCK_SIZE = 8 时：
    # [0, 1, 2, 3, 4, 5, 6, 7]
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 如果真实列数小于 BLOCK_SIZE，
    # 那么多出来的位置要屏蔽掉。
    mask = col_offsets < n_cols

    # `input_row_stride` 表示：
    # 从一行跳到下一行，指针要跨过多少个元素。
    #
    # 所以：
    # `input_ptr + row_id * input_row_stride`
    # 就是第 row_id 行的起始位置。
    row_start = input_ptr + row_id * input_row_stride

    # 这一步把“整行”读进来。
    #
    # `other=-inf` 很重要：
    # 对于越界位置，我们希望它们不要影响后面的 max，
    # 所以用负无穷填充最安全。
    row = tl.load(row_start + col_offsets, mask=mask, other=-float("inf"))

    # softmax 的数值稳定写法：
    # 先减去当前行最大值，再做 exp。
    #
    # 这么做不会改变 softmax 结果，
    # 但可以显著减少数值上溢的风险。
    row_minus_max = row - tl.max(row, axis=0)

    # 逐元素做指数。
    numerator = tl.exp(row_minus_max)

    # 对整行求和，得到 softmax 分母。
    denominator = tl.sum(numerator, axis=0)

    # softmax = exp(x - max) / sum(exp(x - max))
    out = numerator / denominator

    # 计算输出这一行的起始地址。
    output_row = output_ptr + row_id * output_row_stride

    # 把这一行 softmax 结果写回去。
    tl.store(output_row + col_offsets, out, mask=mask)


def row_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, "expected a 2D tensor"

    if not HAS_TRITON_DEVICE:
        # 没有 Triton GPU backend 时，直接退回到 PyTorch softmax。
        return torch.softmax(x, dim=1)

    n_rows, n_cols = x.shape

    # Triton 里经常把 block 大小设置成 2 的幂，
    # 这里做的是“向上补齐到最近的 2 的幂”。
    #
    # 例如：
    # - 8 -> 8
    # - 13 -> 16
    block_size = triton.next_power_of_2(n_cols)

    out = torch.empty_like(x)

    # 因为一个 program 负责一整行，
    # 所以这里总共要启动 n_rows 个 program。
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
    print("这份脚本最重要的主线是：load one row -> subtract max -> exp -> sum -> divide")
    # 如果实现正确，这个误差应该非常小。
    print("max abs diff:", torch.max(torch.abs(result - torch_out)).item())
    print("result:")
    print(result)


if __name__ == "__main__":
    main()
