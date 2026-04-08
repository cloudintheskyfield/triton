# Triton 学习脚本

这个目录是给“从零到 attention”准备的最小学习路径。

推荐顺序：

1. `00_torch_attention_baseline.py`
2. `01_triton_vector_add.py`
3. `02_triton_row_softmax.py`
4. `03_attention_math_walkthrough.py`

## 脚本说明

### `00_torch_attention_baseline.py`

先只理解 attention 数学，不引入 Triton kernel 复杂度。

### `01_triton_vector_add.py`

建立 Triton kernel 的最小直觉：

- 一个 program 处理一段数据
- 如何用 `tl.arange` 生成 offset
- 如何用 `mask` 保护越界访存

### `02_triton_row_softmax.py`

这是 attention 前最关键的桥梁：

- 行归约
- 数值稳定 softmax
- 为什么 fused 比拆分操作更高效

### `03_attention_math_walkthrough.py`

用一个很小的例子把 `QK^T -> softmax -> PV` 整体串起来，帮助你在读官方 fused attention 前先建立数学图景。

## 运行方式

如果你已经把当前仓库构建好了，可以直接在仓库根目录运行：

```bash
python examples/triton_learning/00_torch_attention_baseline.py
python examples/triton_learning/01_triton_vector_add.py
python examples/triton_learning/02_triton_row_softmax.py
python examples/triton_learning/03_attention_math_walkthrough.py
```

如果机器没有可用 GPU：

- `00_torch_attention_baseline.py` 和 `03_attention_math_walkthrough.py` 直接运行
- `01_triton_vector_add.py` 和 `02_triton_row_softmax.py` 会自动回退到 CPU/PyTorch 实现
- 有 CUDA 或 HIP 设备时，这两个脚本会自动切回 Triton kernel 路径
