# SPDX-License-Identifier: Apache-2.0
"""
EngineRegistry - Dynamic Engine Selection for vllm-mlx.

Eliminates hardcoded engine selection by providing:
- Request type-aware routing (simple/batch/multimodal)
- Model type detection (LLM vs MLLM)
- Automatic engine instantiation with proper configuration

Architecture:
    request_type + model_type -> EngineRegistry.select() -> engine_name
    engine_name + kwargs -> EngineRegistry.get_engine() -> engine_instance
"""

import logging
from typing import Any, Dict, Optional, Type

from .base import BaseEngine
from .simple import SimpleEngine
from .batched import BatchedEngine

logger = logging.getLogger(__name__)


class EngineRegistry:
    """
    Dynamic Engine Selection Registry.

    Provides intelligent routing between engine implementations
    based on request characteristics and model capabilities.

    Engine Types:
    - "simple": SimpleEngine - Maximum throughput for single-user
    - "batched": BatchedEngine - Continuous batching for concurrent users
    - "multimodal": BatchedEngine (MLLM mode) - Vision+text generation

    Selection Logic:
    1. MLLM models (is_vl=True) -> "multimodal" engine
    2. Batch requests (request_type="batch") -> "batched" engine
    3. Default -> "simple" engine

    Example:
        >>> registry = EngineRegistry()
        >>> engine_name = registry.select("simple", {"is_vl": False})
        >>> engine = registry.get_engine(engine_name, model_name="...")
    """

    def __init__(self):
        """
        Initialize registry with available engine types.
        """
        self._engines: Dict[str, Type[BaseEngine]] = {
            "simple": SimpleEngine,
            "batched": BatchedEngine,
            "multimodal": BatchedEngine,  # MLLM uses BatchedEngine internally
        }

        # Track registered engines for logging
        self._registered_names = list(self._engines.keys())
        logger.info(
            f"EngineRegistry initialized with engines: {self._registered_names}"
        )

    def register(self, name: str, engine_cls: Type[BaseEngine]) -> None:
        """
        Register a new engine type.

        Args:
            name: Engine type name
            engine_cls: Engine class (must inherit from BaseEngine)
        """
        if not isinstance(engine_cls, type) or not issubclass(engine_cls, BaseEngine):
            raise ValueError(
                f"Engine class must be a subclass of BaseEngine, got {engine_cls}"
            )

        self._engines[name] = engine_cls
        self._registered_names.append(name)
        logger.info(f"Registered engine: {name} -> {engine_cls.__name__}")

    def select(
        self,
        request_type: str,
        model_type: Dict[str, Any],
    ) -> str:
        """
        Select appropriate engine based on request and model type.

        Selection Priority:
        1. MLLM (Vision Language Model) -> "multimodal"
        2. Batch/concurrent requests -> "batched"
        3. Single user, simple request -> "simple"

        Args:
            request_type: Request classification
                - "simple": Single request, single user
                - "batch": Multiple concurrent requests
                - "stream": Streaming request
            model_type: Model capabilities dict
                - is_vl: bool - Vision Language Model (MLLM)
                - supports_mtp: bool - Multi-Token Prediction support
                - supports_specprefill: bool - Speculative prefill support

        Returns:
            Engine type name ("simple", "batched", "multimodal")
        """
        is_vl = model_type.get("is_vl", False)
        supports_mtp = model_type.get("supports_mtp", False)
        supports_specprefill = model_type.get("supports_specprefill", False)

        # MLLM models always use BatchedEngine (internal MLLM scheduler)
        if is_vl:
            logger.debug(
                f"[EngineRegistry] Selecting 'multimodal' for MLLM model "
                f"(request_type={request_type})"
            )
            return "multimodal"

        # MTP and SpecPrefill are SimpleEngine features
        if supports_mtp or supports_specprefill:
            logger.debug(
                f"[EngineRegistry] Selecting 'simple' for advanced features "
                f"(mtp={supports_mtp}, specprefill={supports_specprefill})"
            )
            return "simple"

        # Batch requests use BatchedEngine for continuous batching
        if request_type == "batch":
            logger.debug(
                f"[EngineRegistry] Selecting 'batched' for batch request"
            )
            return "batched"

        # Default: SimpleEngine for maximum throughput
        logger.debug(
            f"[EngineRegistry] Selecting 'simple' (default) "
            f"for request_type={request_type}"
        )
        return "simple"

    def get_engine(
        self,
        name: str,
        model_name: str,
        **kwargs,
    ) -> BaseEngine:
        """
        Instantiate engine with given name and configuration.

        Args:
            name: Engine type name from select()
            model_name: HuggingFace model name or local path
            **kwargs: Engine-specific parameters
                - trust_remote_code: bool
                - scheduler_config: SchedulerConfig (batched)
                - stream_interval: int (batched)
                - force_mllm: bool (batched)
                - mtp: bool (simple)
                - prefill_step_size: int (simple)
                - specprefill_enabled: bool (simple)
                - etc.

        Returns:
            Instantiated engine ready for start()

        Raises:
            ValueError: If engine name is unknown
        """
        engine_cls = self._engines.get(name)
        if engine_cls is None:
            raise ValueError(
                f"Unknown engine: '{name}'. "
                f"Available engines: {self._registered_names}"
            )

        # Pass model_name as first argument
        engine = engine_cls(model_name=model_name, **kwargs)

        logger.info(
            f"[EngineRegistry] Created {engine_cls.__name__} instance "
            f"for model={model_name}, engine_type={name}"
        )
        return engine

    def list_engines(self) -> Dict[str, str]:
        """
        List all registered engine types with their class names.

        Returns:
            Dict mapping engine name -> class name
        """
        return {
            name: cls.__name__
            for name, cls in self._engines.items()
        }

    def is_registered(self, name: str) -> bool:
        """
        Check if an engine type is registered.

        Args:
            name: Engine type name

        Returns:
            True if registered, False otherwise
        """
        return name in self._engines


# Singleton instance for convenience
_registry_instance: Optional[EngineRegistry] = None


def get_registry() -> EngineRegistry:
    """
    Get the global EngineRegistry instance.

    Returns:
        EngineRegistry singleton
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = EngineRegistry()
    return _registry_instance


def select_engine(
    request_type: str,
    model_type: Dict[str, Any],
) -> str:
    """
    Convenience function to select engine via global registry.

    Args:
        request_type: Request classification
        model_type: Model capabilities dict

    Returns:
        Engine type name
    """
    return get_registry().select(request_type, model_type)


def create_engine(
    engine_name: str,
    model_name: str,
    **kwargs,
) -> BaseEngine:
    """
    Convenience function to create engine via global registry.

    Args:
        engine_name: Engine type name
        model_name: Model name or path
        **kwargs: Engine configuration

    Returns:
        Engine instance
    """
    return get_registry().get_engine(engine_name, model_name, **kwargs)