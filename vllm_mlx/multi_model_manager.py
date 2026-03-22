# SPDX-License-Identifier: Apache-2.0
"""
Dynamic model manager for vllm-mlx (Ollama-style).

Simple on-demand model loading with automatic unloading.

Features:
- Directory scanning (--models-dir)
- Multiple --model arguments
- On-demand loading (first request loads model)
- Automatic unloading after idle timeout (keep_alive)
- No config files needed

Usage:
    # Scan directory for models
    vllm-mlx serve --models-dir ~/models --keep-alive 5m

    # Or specify models explicitly
    vllm-mlx serve --model qwen35:~/models/Qwen3.5-35B --model qwen8:~/models/Qwen3-8B

    # Request any model (auto-loads if not in memory)
    curl -d '{"model": "qwen35", "messages": [...]}'
"""

import asyncio
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .engine import BaseEngine, BatchedEngine, SimpleEngine

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """A model currently loaded in memory."""

    engine: BaseEngine
    path: str
    loaded_at: float
    last_used_at: float
    expires_at: float  # When to unload (last_used + keep_alive)
    memory_bytes: int = 0

    def touch(self, keep_alive_seconds: float):
        """Update last used time and extend expiration."""
        now = time.time()
        self.last_used_at = now
        self.expires_at = now + keep_alive_seconds


class DynamicModelManager:
    """
    Manages multiple models with Ollama-style loading.

    Models are loaded on first request and automatically unloaded
    after being idle for keep_alive seconds.
    """

    def __init__(
        self,
        keep_alive_seconds: float = 300,  # 5 minutes default
        max_loaded_models: int = 10,
        max_memory_gb: Optional[float] = None,
        use_batching: bool = False,
        force_mllm: bool = False,
    ):
        """
        Initialize the model manager.

        Args:
            keep_alive_seconds: How long to keep models loaded after last use (default: 5 min)
            max_loaded_models: Maximum models to keep loaded
            max_memory_gb: Memory budget in GB (optional)
            use_batching: Use BatchedEngine for all models
            force_mllm: Force MLLM mode for all models
        """
        self._keep_alive = keep_alive_seconds
        self._max_loaded = max_loaded_models
        self._max_memory_gb = max_memory_gb
        self._use_batching = use_batching
        self._force_mllm = force_mllm

        # Model name -> path mapping (from --model args or directory scan)
        self._registry: dict[str, str] = {}

        # Loaded models (OrderedDict for LRU)
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

        # Default model (first one registered)
        self._default_model: Optional[str] = None

    def register_model(self, name: str, path: str) -> None:
        """
        Register a model name to path mapping.

        Args:
            name: Model name (used in API requests)
            path: Model path (local or HuggingFace)
        """
        path = os.path.expanduser(path)
        self._registry[name] = path
        if self._default_model is None:
            self._default_model = name
        logger.debug(f"Registered model: {name} -> {path}")

    def scan_directory(self, directory: str) -> list[str]:
        """
        Scan a directory for models.

        Each subdirectory is treated as a model, with the directory name
        as the model name.

        Args:
            directory: Path to scan

        Returns:
            List of model names found
        """
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            logger.warning(f"Models directory not found: {directory}")
            return []

        found = []
        for entry in Path(directory).iterdir():
            if entry.is_dir():
                # Check if it looks like a model directory
                # (contains config.json, model.safetensors, etc.)
                model_files = list(entry.glob("*.safetensors")) + \
                              list(entry.glob("*.gguf")) + \
                              list(entry.glob("config.json"))
                if model_files or (entry / "config.json").exists():
                    name = entry.name
                    self.register_model(name, str(entry))
                    found.append(name)

        if found:
            logger.info(f"Found {len(found)} models in {directory}: {found}")
        return found

    def resolve_path(self, model_name: str) -> Optional[str]:
        """
        Resolve a model name to a path.

        Args:
            model_name: Name, alias, or path

        Returns:
            Model path if found, None otherwise
        """
        # 1. Check registry
        if model_name in self._registry:
            return self._registry[model_name]

        # 2. Treat as direct path (expand ~)
        expanded = os.path.expanduser(model_name)
        if os.path.isdir(expanded):
            return expanded

        # 3. Could be a HuggingFace model name (no local check needed)
        if "/" in model_name:
            return model_name

        return None

    def is_loaded(self, name: str) -> bool:
        """Check if a model is currently loaded."""
        return name in self._loaded

    def list_models(self) -> list[dict]:
        """
        List all known models.

        Returns:
            List of model info dicts
        """
        result = []
        for name, path in self._registry.items():
            loaded = name in self._loaded
            info = {
                "name": name,
                "path": path,
                "loaded": loaded,
            }
            if loaded:
                m = self._loaded[name]
                info["expires_at"] = m.expires_at
                info["memory_gb"] = round(m.memory_bytes / 1e9, 2)
            result.append(info)
        return result

    def list_loaded_models(self) -> list[dict]:
        """
        List currently loaded models (like Ollama's /api/ps).

        Returns:
            List of loaded model info dicts
        """
        result = []
        now = time.time()
        for name, m in self._loaded.items():
            result.append({
                "name": name,
                "model": name,
                "path": m.path,
                "loaded_at": m.loaded_at,
                "expires_at": m.expires_at,
                "expires_in": max(0, m.expires_at - now),
                "size": m.memory_bytes,
                "size_vram": m.memory_bytes,
            })
        return result

    async def get_engine(
        self,
        model_name: Optional[str] = None,
        keep_alive: Optional[float] = None,
    ) -> BaseEngine:
        """
        Get engine for a model, loading if necessary.

        Args:
            model_name: Model name or path (None for default)
            keep_alive: Override keep_alive for this request

        Returns:
            Engine instance

        Raises:
            ValueError: If model not found
            RuntimeError: If loading fails
        """
        # Resolve model name
        if model_name is None:
            model_name = self._default_model

        if model_name is None:
            raise ValueError("No model specified and no default model available")

        path = self.resolve_path(model_name)
        if path is None:
            raise ValueError(f"Model not found: {model_name}")

        # Use registered name if available, otherwise use the provided name
        registry_name = None
        for name, p in self._registry.items():
            if p == path:
                registry_name = name
                break

        effective_name = registry_name or model_name
        ka = keep_alive if keep_alive is not None else self._keep_alive

        # Check if already loaded
        if effective_name in self._loaded:
            m = self._loaded[effective_name]
            m.touch(ka)
            self._loaded.move_to_end(effective_name)
            logger.debug(f"Using cached model: {effective_name}")
            return m.engine

        # Need to load - check capacity
        await self._maybe_evict()

        # Load the model
        logger.info(f"Loading model: {effective_name} from {path}")
        start = time.time()

        try:
            engine = await self._load_engine(path)
            memory = self._estimate_memory(engine)

            now = time.time()
            m = LoadedModel(
                engine=engine,
                path=path,
                loaded_at=now,
                last_used_at=now,
                expires_at=now + ka,
                memory_bytes=memory,
            )

            self._loaded[effective_name] = m
            logger.info(f"Model loaded: {effective_name} ({time.time()-start:.1f}s, {memory/1e9:.1f}GB)")
            return engine

        except Exception as e:
            logger.error(f"Failed to load {effective_name}: {e}")
            raise RuntimeError(f"Failed to load model {effective_name}: {e}") from e

    async def _load_engine(self, path: str) -> BaseEngine:
        """Load an engine from a path."""
        if self._use_batching:
            engine = BatchedEngine(
                model_name=path,
                force_mllm=self._force_mllm,
            )
        else:
            engine = SimpleEngine(
                model_name=path,
                force_mllm=self._force_mllm,
            )
        await engine.start()
        return engine

    def _estimate_memory(self, engine: BaseEngine) -> int:
        """Estimate memory usage for a loaded model."""
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                return mx.get_active_memory()
        except Exception:
            pass
        return 20 * 10**9  # Default estimate: 20GB

    async def _maybe_evict(self) -> None:
        """Evict expired or excess models."""
        now = time.time()

        # 1. Remove expired models
        expired = [
            name for name, m in self._loaded.items()
            if m.expires_at <= now
        ]
        for name in expired:
            logger.info(f"Unloading expired model: {name}")
            await self._unload_model(name)

        # 2. Check model count
        while len(self._loaded) >= self._max_loaded:
            # Evict LRU (first in OrderedDict)
            lru_name = next(iter(self._loaded))
            logger.info(f"Evicting LRU model: {lru_name}")
            await self._unload_model(lru_name)

        # 3. Check memory budget
        if self._max_memory_gb:
            max_bytes = self._max_memory_gb * 10**9
            while self._loaded:
                total = sum(m.memory_bytes for m in self._loaded.values())
                if total < max_bytes * 0.9:
                    break
                lru_name = next(iter(self._loaded))
                logger.info(f"Evicting model (memory pressure): {lru_name}")
                await self._unload_model(lru_name)

    async def _unload_model(self, name: str) -> bool:
        """Unload a model."""
        if name not in self._loaded:
            return False

        m = self._loaded[name]
        try:
            await m.engine.stop()
        except Exception as e:
            logger.warning(f"Error stopping {name}: {e}")

        del self._loaded[name]
        logger.info(f"Unloaded model: {name}")
        return True

    async def unload_model(self, name: str) -> bool:
        """
        Explicitly unload a model (like Ollama's keep_alive=0).

        Args:
            name: Model name

        Returns:
            True if unloaded
        """
        return await self._unload_model(name)

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Started cleanup task")

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired models."""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            try:
                now = time.time()
                expired = [
                    name for name, m in self._loaded.items()
                    if m.expires_at <= now
                ]
                for name in expired:
                    logger.info(f"Auto-unloading expired model: {name}")
                    await self._unload_model(name)
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def stop_all(self) -> None:
        """Stop all loaded models."""
        await self.stop_cleanup_task()
        for name in list(self._loaded.keys()):
            await self._unload_model(name)

    def get_stats(self) -> dict:
        """Get manager statistics."""
        total_memory = sum(m.memory_bytes for m in self._loaded.values())
        stats = {
            "registered_models": len(self._registry),
            "loaded_models": len(self._loaded),
            "total_memory_gb": round(total_memory / 1e9, 2),
            "max_memory_gb": self._max_memory_gb,
            "keep_alive_seconds": self._keep_alive,
            "default_model": self._default_model,
        }

        # Metal memory
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                stats["metal_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
                stats["metal_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
        except Exception:
            pass

        return stats

    @property
    def default_model(self) -> Optional[str]:
        """Get the default model name."""
        return self._default_model


# Global instance
_manager: Optional[DynamicModelManager] = None


def get_model_manager() -> Optional[DynamicModelManager]:
    """Get the global model manager."""
    return _manager


def init_model_manager(
    keep_alive_seconds: float = 300,
    max_loaded_models: int = 10,
    max_memory_gb: Optional[float] = None,
    use_batching: bool = False,
    force_mllm: bool = False,
) -> DynamicModelManager:
    """Initialize the global model manager."""
    global _manager
    _manager = DynamicModelManager(
        keep_alive_seconds=keep_alive_seconds,
        max_loaded_models=max_loaded_models,
        max_memory_gb=max_memory_gb,
        use_batching=use_batching,
        force_mllm=force_mllm,
    )
    return _manager