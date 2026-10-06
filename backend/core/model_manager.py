"""Loads models on demand and keeps at most ONE heavy model resident.

Heavy: text encoders (T5), FLUX-schnell, FLUX-Fill, VLM. Light (SAM, LaMa, ...) may stay loaded.
Load/evict events and memory are appended to ``outputs/logs/models.jsonl``.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from core.config import project_path
from core.device import device_memory_gb, free_memory, process_peak_rss_gb

log = logging.getLogger("adobeps.models")


@dataclass
class ModelSpec:
    loader: Callable[[], Any]
    heavy: bool


class ModelManager:
    """Process-wide registry. Use :func:`get_manager`."""

    def __init__(self) -> None:
        self.specs: dict[str, ModelSpec] = {}
        self.resident: dict[str, Any] = {}
        self.lock = threading.RLock()

    def register(self, name: str, loader: Callable[[], Any], heavy: bool) -> None:
        self.specs[name] = ModelSpec(loader, heavy)

    def _log(self, event: str, name: str, **kw: Any) -> None:
        rec = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "model": name,
            "device_mem_gb": round(device_memory_gb(), 2),
            "peak_rss_gb": round(process_peak_rss_gb(), 2),
            **kw,
        }
        log.info("%s %s %s", event, name, kw)
        with open(project_path("logs") / "models.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def get(self, name: str) -> Any:
        """Return the loaded model/pipeline ``name``, evicting another heavy model first if needed."""
        with self.lock:
            if name in self.resident:
                return self.resident[name]
            if name not in self.specs:
                raise KeyError(f"Unknown model '{name}'. Registered: {sorted(self.specs)}")
            spec = self.specs[name]
            if spec.heavy:
                for other in [n for n in self.resident if self.specs[n].heavy]:
                    self.evict(other)
            t0 = time.perf_counter()
            obj = spec.loader()
            self.resident[name] = obj
            self._log("load", name, seconds=round(time.perf_counter() - t0, 1), heavy=spec.heavy)
            return obj

    def evict(self, name: str) -> None:
        with self.lock:
            obj = self.resident.pop(name, None)
            if obj is None:
                return
            del obj
            free_memory()
            try:  # MLX (Mac VLM) keeps its own buffer cache
                import mlx.core as mx

                mx.clear_cache()
            except Exception:  # noqa: BLE001
                pass
            self._log("evict", name)

    def evict_heavy(self) -> None:
        for n in [n for n in self.resident if self.specs[n].heavy]:
            self.evict(n)

    def evict_all(self) -> None:
        for n in list(self.resident):
            self.evict(n)


_manager: ModelManager | None = None


def get_manager() -> ModelManager:
    """The singleton manager with all project loaders registered."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
        from core import loaders  # late import: loaders pull in diffusers

        loaders.register_all(_manager)
    return _manager
