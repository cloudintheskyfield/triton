"""
Runtime helpers for the learning scripts.

These examples need a Triton-supported GPU runtime. When no CUDA/HIP driver is
active, we raise a short explanation instead of exposing an internal traceback.
"""


def get_triton_device():
    import triton

    try:
        return triton.runtime.driver.active.get_active_torch_device()
    except RuntimeError as exc:
        raise RuntimeError(
            "No active Triton GPU driver was found.\n"
            "These Triton kernel examples require a CUDA or HIP environment.\n"
            "Common reasons:\n"
            "1. The current machine has no supported NVIDIA/AMD GPU\n"
            "2. CUDA or ROCm is not installed or not visible to this environment\n"
            "3. You are on macOS without a supported Triton GPU backend\n\n"
            "You can still run the pure PyTorch examples:\n"
            "- examples/triton_learning/00_torch_attention_baseline.py\n"
            "- examples/triton_learning/03_attention_math_walkthrough.py"
        ) from exc
