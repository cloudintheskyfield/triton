"""
脚本目的：
这个脚本把 attention 拆成 4 个清晰的数学步骤打印出来，
帮助你在阅读 Triton fused attention 之前先建立完整的全局图景。

你会在这里看到：
1. 原始分数 `q @ k^T`
2. 缩放后的分数 `scores / sqrt(d)`
3. softmax 后的注意力概率
4. 最终输出 `probs @ v`

建议怎么学：
运行后对照打印结果，一步一步看张量是如何变化的。
当你能把这 4 步讲清楚时，再去看 Triton attention kernel 会顺畅很多。
"""

import math

import torch


def attention_steps(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # attention 中常见的缩放因子，防止 qk 点积随着维度变大而数值过大。
    scale = 1.0 / math.sqrt(q.shape[-1])
    # scores: [Nq, Nk]
    scores = q @ k.transpose(-1, -2)
    scaled_scores = scores * scale
    # 对每个 query 对应的一整行分数做 softmax。
    probs = torch.softmax(scaled_scores, dim=-1)
    # 用 softmax 后的权重对 v 做加权和。
    out = probs @ v
    return scores, scaled_scores, probs, out


def main():
    torch.manual_seed(0)

    q = torch.randn(2, 4)
    k = torch.randn(3, 4)
    v = torch.randn(3, 4)

    scores, scaled_scores, probs, out = attention_steps(q, k, v)

    print("q shape:", tuple(q.shape))
    print("k shape:", tuple(k.shape))
    print("v shape:", tuple(v.shape))
    print("\n1) raw scores = q @ k^T")
    print(scores)
    print("\n2) scaled scores = scores / sqrt(d)")
    print(scaled_scores)
    print("\n3) attention probabilities = softmax(scaled scores)")
    print(probs)
    print("\n4) output = probs @ v")
    print(out)


if __name__ == "__main__":
    main()
