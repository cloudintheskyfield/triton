"""
脚本目的：
这个脚本用纯 PyTorch 实现一个最小可读版的单头 attention，用来先理解 attention 的数学流程，
不把注意力分散到 Triton kernel 细节上。

你会在这里看到：
1. q @ k^T 如何得到 attention score
2. 为什么要除以 sqrt(d)
3. softmax 在 attention 里扮演什么角色
4. 最终如何用注意力权重对 v 做加权求和

建议怎么学：
先把这个脚本跑通，确认自己已经能从张量形状和数学公式两个角度理解 attention，
再进入 Triton 版本的 softmax 和 attention 学习。
"""

import math

import torch


def single_head_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """A tiny PyTorch baseline to make the attention math explicit."""
    # q: [Nq, D]
    # k: [Nk, D]
    # v: [Nk, D]
    # 输出: [Nq, D]
    d = q.shape[-1]
    # 每个 query 会和所有 key 做点积，得到 attention score。
    scores = q @ k.transpose(-1, -2) / math.sqrt(d)

    if causal:
        seq_len = q.shape[-2]
        # causal mask 会屏蔽“看未来”的位置。
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    # 在 key 这个维度做 softmax，得到注意力权重。
    probs = torch.softmax(scores, dim=-1)
    # 用注意力权重对 v 做加权求和。
    out = probs @ v
    return out


def main():
    torch.manual_seed(0)

    q = torch.randn(4, 8)
    k = torch.randn(4, 8)
    v = torch.randn(4, 8)

    # 先看最标准的 non-causal attention。
    out = single_head_attention(q, k, v, causal=False)
    # 再对比 causal attention 的输出差异。
    causal_out = single_head_attention(q, k, v, causal=True)

    print("q shape:", tuple(q.shape))
    print("k shape:", tuple(k.shape))
    print("v shape:", tuple(v.shape))
    print("attention output shape:", tuple(out.shape))
    print("non-causal output:")
    print(out)
    print("causal output:")
    print(causal_out)


if __name__ == "__main__":
    main()
