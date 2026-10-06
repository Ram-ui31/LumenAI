"""Device, dtype and memory helpers. The ONLY place that decides which device is used."""
from __future__ import annotations

import gc
import logging
import os
import resource
import sys
from functools import lru_cache

import torch

log = logging.getLogger("adobeps.device")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@lru_cache(maxsize=1)
def get_device() -> str:
    """Return ``"cuda"``, ``"mps"`` or ``"cpu"``. ``ADOBEPS_DEVICE`` overrides autodetection."""
    forced = os.environ.get("ADOBEPS_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    apply_memory_budget(dev)
    return dev


def apply_memory_budget(dev: str) -> None:
    """Cap the accelerator allocator at ``memory.budget_gb`` so the app is forced to live within the
    target (16 GB). With ``enforce`` an over-budget step raises out-of-memory instead of swapping."""
    try:
        from core.config import get_config

        mem = get_config().get("memory", {})
    except Exception:  # noqa: BLE001
        return
    budget, enforce = float(mem.get("budget_gb") or 0), bool(mem.get("enforce", False))
    if not budget or not enforce:
        return
    try:
        if dev == "mps":
            rec = torch.mps.recommended_max_memory() / 1e9   # macOS' working-set limit for the GPU
            frac = min(2.0, budget / rec)
            torch.mps.set_per_process_memory_fraction(frac)
            log.info("MPS memory capped at %.1f GB (fraction %.2f of %.1f GB)", budget, frac, rec)
        elif dev == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total))
            log.info("CUDA memory capped at %.1f GB of %.1f GB", budget, total)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not apply memory budget: %s", e)


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@lru_cache(maxsize=4)
def get_dtype(preferred: str = "bfloat16") -> torch.dtype:
    """Pick a compute dtype for the current device, falling back to float16 if bf16 fails.

    The choice is logged so it can be recorded in recipes and NOTES.md.
    """
    device = get_device()
    if device == "cpu":
        # bf16 matmuls on CPU are slow but work; keep preferred unless explicitly float32.
        dtype = _DTYPES.get(preferred, torch.float32)
    else:
        dtype = _DTYPES.get(preferred, torch.bfloat16)
        if dtype == torch.bfloat16:
            try:
                a = torch.ones(8, 8, device=device, dtype=dtype)
                (a @ a).sum().item()
                if device == "cuda" and not torch.cuda.is_bf16_supported():
                    raise RuntimeError("bf16 unsupported on this GPU")
            except Exception as e:  # noqa: BLE001
                log.warning("bfloat16 failed on %s (%s); falling back to float16", device, e)
                dtype = torch.float16
    log.info("Using dtype %s on %s", dtype, device)
    return dtype


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def free_memory() -> None:
    """Release cached allocator memory after deleting models."""
    gc.collect()
    dev = get_device()
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()


def device_memory_gb() -> float:
    """Memory the device driver holds (GB): tensors + allocator cache. On MPS this is unified memory."""
    dev = get_device()
    if dev == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if dev == "cuda":
        return torch.cuda.memory_reserved() / 1e9
    return 0.0


def tensor_memory_gb() -> float:
    """Memory held by live tensors only (GB): what a GPU of that size actually has to fit."""
    dev = get_device()
    if dev == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    if dev == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def process_peak_rss_gb() -> float:
    """Peak resident set size of this process (GB). macOS reports bytes, Linux KiB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r * 1024 / 1e9


class MemoryTracker:
    """Samples device memory (call :meth:`sample` often, e.g. every denoising step)."""

    def __init__(self) -> None:
        self.peak_device_gb = 0.0
        self.peak_tensor_gb = 0.0
        if get_device() == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def sample(self) -> None:
        self.peak_device_gb = max(self.peak_device_gb, device_memory_gb())
        self.peak_tensor_gb = max(self.peak_tensor_gb, tensor_memory_gb())

    def report(self) -> dict[str, float]:
        """peak_tensor_gb = live tensors (compare this with the 16 GB budget);
        peak_device_gb = what the driver held incl. cache (the Mac's footprint)."""
        self.sample()
        dev, ten = self.peak_device_gb, self.peak_tensor_gb
        if get_device() == "cuda":
            ten = max(ten, torch.cuda.max_memory_allocated() / 1e9)
            dev = max(dev, torch.cuda.max_memory_reserved() / 1e9)
        return {"peak_tensor_gb": round(ten, 2), "peak_device_gb": round(dev, 2),
                "peak_rss_gb": round(process_peak_rss_gb(), 2)}
