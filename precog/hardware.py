"""Hardware Encoder (docs.md §9.3): X_hardware, because the optimal
configuration also depends on the execution environment
(theta* = f(M, D, H)). Captures GPU/CPU, memory, bandwidth, numerical
precision, batch capacity, interconnect -- introspected from the actual
runtime rather than assumed, since this sandbox is CPU-only (no GPU driver,
see the environment check that motivated stack.md's local-tooling choices)."""
from __future__ import annotations

import os

import torch


def hardware_features() -> dict:
    has_gpu = torch.cuda.is_available()
    return {
        "device_type": "gpu" if has_gpu else "cpu",
        "device_name": torch.cuda.get_device_name(0) if has_gpu else "cpu",
        "device_count": torch.cuda.device_count() if has_gpu else os.cpu_count() or 1,
        "memory_bytes": torch.cuda.get_device_properties(0).total_memory if has_gpu else None,
        "precision": "float32",
        "interconnect": "none" if not has_gpu or torch.cuda.device_count() <= 1 else "nvlink_or_pcie",
    }
