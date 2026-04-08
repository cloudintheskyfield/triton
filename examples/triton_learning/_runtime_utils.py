"""
Runtime helpers for the learning scripts.

When Triton cannot find a supported GPU backend, the learning scripts can fall
back to CPU/PyTorch implementations so the user can still study the math and
API structure on machines such as macOS laptops.
"""

import torch


def get_triton_runtime():
    import triton

    try:
        device = triton.runtime.driver.active.get_active_torch_device()
        return device, True, None
    except RuntimeError as exc:
        return torch.device("cpu"), False, exc


def format_runtime_message(using_triton: bool, reason=None) -> str:
    if using_triton:
        return "Using Triton GPU backend."
    return (
        "No active Triton GPU driver was found, so this script is using the "
        "CPU/PyTorch fallback path.\n"
        f"Original Triton runtime error: {reason}"
    )
