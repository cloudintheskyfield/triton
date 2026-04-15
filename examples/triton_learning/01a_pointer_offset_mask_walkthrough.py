"""
脚本目的：
这个脚本不用 Triton kernel，也不用 GPU。
它专门把 Triton 入门里最容易卡住的 4 个概念拆开讲：

1. pointer：数组起点
2. offset：相对起点偏移多少个元素
3. mask：哪些位置可以访问，哪些位置要屏蔽
4. load/store：从“地址”读取数据，再写回结果

为什么要有这个脚本：
很多人第一次看 Triton，不是卡在 `@triton.jit`，
而是卡在下面这几行到底在“模拟什么”：

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

所以这个脚本会先用纯 Python / PyTorch 把这几行的含义讲明白，
再去看真正的 Triton kernel 就会轻松很多。

Triton kernel 本质在操作 GPU 上内容，流程为
    1. 从输入张量在的显存将数据读出来
        读到当前 program 能操作的位置； 显存为“大仓库”，tl.load是将这批要用的货从仓库拿回手边
        即 GPU 中有不同层级的存储
            （1）
    2. 在 GPU 内部做计算
    3. 将结果写回输出张量所在的显存
"""

import torch


def masked_load(x: torch.Tensor, offsets: torch.Tensor, mask: torch.Tensor, other: float = 0.0) -> torch.Tensor:
    """
        A tiny CPU version of Triton's tl.load(..., mask=..., other=...).
        模拟 tl.load(x + offset, mask=mask, other=0)
            去 offsets 位置读取数据（要读几个位置）、mask（那些位置真的能读）
            如果某个位置合法，就读出来；如果某个位置越界了，就不要读
            越界的位置用 other 默认值补上

        去读 x 张量中，下标等于 offsets 对应位置上的值
    """
    out = torch.full_like(offsets, fill_value=other, dtype=x.dtype)
    out[mask] = x[offsets[mask]]  # 只保留mask为True的位置， offsets[mask]为 tensor([8, 9])；x[offsets[mask]] 代表在这些位置上取值
    return out


def masked_store(output: torch.Tensor, offsets: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> None:
    """A tiny CPU version of Triton's tl.store(..., mask=...)."""
    output[offsets[mask]] = values[mask]


def main():
    # 假设我们手里有一个长度为 10 的一维向量。
    x = torch.arange(10, dtype=torch.float32)
    y = torch.arange(100, 110, dtype=torch.float32)

    # 下面模拟“当前 program 负责哪一块数据”。
    pid = 2
    block_size = 4
    n_elements = x.numel()

    # 在 Triton 里：
    #   block_start = pid * BLOCK_SIZE
    # 表示第 pid 个 program 处理的块从哪里开始。
    block_start = pid * block_size

    # 这一步对应：
    #   offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 它生成这一整个 program 想处理的下标。
    # 这里 8，9，10，11 对应逐元素相加
    offsets = block_start + torch.arange(block_size)

    # 这一步对应：
    #   mask = offsets < n_elements
    # 如果最后一个 block 不完整，超出真实长度的位置就要屏蔽掉。 将大于10的进行屏蔽
    mask = offsets < n_elements

    # 这一步就是对 Triton 的：tl.load 从内存中读数据
    #   x = tl.load(x_ptr + offsets, mask=mask)
    # 做一个 CPU 版本的直观模拟。
    loaded_x = masked_load(x, offsets, mask, other=0.0)
    loaded_y = masked_load(y, offsets, mask, other=0.0)

    # Triton kernel 里经常会在寄存器里先做运算（先将一批数据读到片上更快的位置），再统一写回。
    result = loaded_x + loaded_y

    # 输出张量先分配好，再把当前 block 的结果写回去。
    output = torch.full_like(x, fill_value=-1.0)
    # 模拟triton 中的 tl.store: 将数据写回内存
    #   offsets 告诉要写到那些位置
    #   result 告诉要写入那些值
    #   mask 那些位置允许写，那些位置不要写
    masked_store(output, offsets, result, mask)

    print("原始向量 x:")
    print(x)
    print("\n原始向量 y:")
    print(y)
    print("\npid:")
    print(pid)
    print("\nblock_start = pid * block_size:")
    print(block_start)
    print("\noffsets = block_start + arange(block_size):")
    print(offsets)
    print("\nmask = offsets < n_elements:")
    print(mask)
    print("\n模拟 tl.load 后得到的 x block:")
    print(loaded_x)
    print("\n模拟 tl.load 后得到的 y block:")
    print(loaded_y)
    print("\nblock 内部计算结果:")
    print(result)
    print("\n模拟 tl.store 后的输出:")
    print(output)


if __name__ == "__main__":
    main()
