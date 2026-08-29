"""GPU device selection and runtime helpers for the strategy pipeline."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def resolve_runtime_device(prefer_cuda: bool = True) -> torch.device:
    """Pick the best available execution device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_torch_runtime(device: torch.device, allow_tf32: bool = True) -> None:
    """Apply safe runtime optimizations for the selected device."""
    if device.type != "cuda":
        return

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32


def amp_dtype_for_device(device: torch.device) -> torch.dtype | None:
    """Return a stable mixed-precision dtype for the device."""
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, enabled: bool = True):
    """Return the right autocast context or a no-op context."""
    amp_dtype = amp_dtype_for_device(device)
    if enabled and device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def dataloader_kwargs(device: torch.device) -> dict:
    """Reasonable DataLoader settings for the current device."""
    if device.type == "cuda":
        return {
            "pin_memory": True,
            "pin_memory_device": "cuda",
        }
    return {
        "pin_memory": False,
    }


def get_device_info(prefer_cuda: bool = True) -> dict:
    """Summarize the selected runtime device."""
    device = resolve_runtime_device(prefer_cuda=prefer_cuda)
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
        "amp_dtype": str(amp_dtype_for_device(device)) if device.type == "cuda" else None,
    }
    return info


def describe_device(prefer_cuda: bool = True) -> tuple[torch.device, dict]:
    """Configure and describe the current runtime device."""
    device = resolve_runtime_device(prefer_cuda=prefer_cuda)
    configure_torch_runtime(device)
    info = get_device_info(prefer_cuda=prefer_cuda)
    if device.type == "cuda":
        print(f"  [Runtime] Using CUDA: {torch.cuda.get_device_name(device)}")
        print(f"  [Runtime] CUDA version: {info['cuda_version']} | AMP dtype: {info['amp_dtype']}")
    else:
        print("  [Runtime] CUDA unavailable, using CPU.")
    return device, info


if __name__ == "__main__":
    selected_device, device_info = describe_device()
    print(f"Device being used: {selected_device}")
    for key, value in device_info.items():
        print(f"{key}: {value}")
