# SPDX-License-Identifier: Apache-2.0
"""
UnifiedScheduler - Single scheduler for LLM and MLLM requests.

Eliminates the dual-track scheduler design by providing:
- Request type-aware scheduling (text vs multimodal)
- Automatic backend selection (BatchGenerator vs MLLMBatchGenerator)
- Unified queue management and request lifecycle

Architecture:
    UnifiedScheduler -> SchedulerBackend
    - TextSchedulerBackend: Uses mlx-lm BatchGenerator
    - MultimodalSchedulerBackend: Uses MLLMBatchGenerator
    
    Request routing:
    - has_media? -> MultimodalSchedulerBackend
    - text-only? -> TextSchedulerBackend
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Request type classification for routing."""

    TEXT = "text"
    MULTIMODAL = "multimodal"


class SchedulerBackendType(Enum):
    """Scheduler backend type."""

    TEXT = "text"
    MULTIMODAL = "multimodal"


@dataclass
class UnifiedRequest:
    """
    Unified request that can be text or multimodal.
    
    Attributes are populated based on request type:
    - TEXT: prompt, prompt_token_ids, sampling_params
    - MULTIMODAL: prompt, images, videos, sampling_params
    """

    request_id: str
    request_type: RequestType

    # Common fields
    prompt: str
    sampling_params: Any  # SamplingParams

    # Text-only fields
    prompt_token_ids: Optional[List[int]] = None
    prompt_cache: Optional[List[Any]] = None
    cached_tokens: int = 0
    remaining_tokens: Optional[List[int]] = None

    # Multimodal fields
    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None

    # Lifecycle tracking
    status: str = "waiting"  # waiting, running, finished
    arrival_time: float = field(default_factory=time.time)

    # Output tracking
    output_tokens: List[int] = field(default_factory=list)
    output_text: str = ""
    finish_reason: Optional[str] = None

    # Batch UID (assigned when scheduled)
    batch_uid: Optional[int] = None

    # Token counts
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0

    # Timing metrics
    first_token_time: Optional[float] = None

    # Cache hit tracking
    cache_hit_type: str = "miss"
    prefix_boundary: int = 0


@dataclass
class UnifiedSchedulerConfig:
    """
    Configuration for UnifiedScheduler.
    
    Combines settings from both LLM and MLLM schedulers.
    """

    # Common settings
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192

    # Text scheduler settings
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    prefill_step_size: int = 2048

    # Multimodal scheduler settings
    mllm_prefill_batch_size: int = 4
    mllm_completion_batch_size: int = 16
    mllm_prefill_step_size: int = 1024

    # Vision cache settings
    enable_vision_cache: bool = True
    vision_cache_size: int = 100

    # Prefix cache settings
    enable_prefix_cache: bool = True

    # Video settings
    default_video_fps: float = 2.0
    max_video_frames: int = 128


class SchedulerBackend:
    """
    Base class for scheduler backends.
    
    Each backend handles a specific request type (text or multimodal).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: UnifiedSchedulerConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def add_request(self, request: UnifiedRequest) -> None:
        """Add request to backend queue."""
        raise NotImplementedError

    def step(self) -> List[Any]:
        """Execute one generation step."""
        raise NotImplementedError

    def abort_request(self, request_id: str) -> bool:
        """Abort a request."""
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        raise NotImplementedError

    def has_requests(self) -> bool:
        """Check if backend has pending requests."""
        raise NotImplementedError


class TextSchedulerBackend(SchedulerBackend):
    """
    Text-only scheduler backend using mlx-lm BatchGenerator.
    
    Handles LLM requests with prefix cache support.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: UnifiedSchedulerConfig,
        prefix_cache: Optional[Any] = None,
    ):
        super().__init__(model, tokenizer, config)
        self.prefix_cache = prefix_cache

        # Request queues
        self.waiting: deque[UnifiedRequest] = deque()
        self.running: Dict[str, UnifiedRequest] = {}

        # BatchGenerator (created lazily)
        self.batch_generator: Optional[Any] = None

        # UID mappings
        self.request_id_to_uid: Dict[str, int] = {}
        self.uid_to_request_id: Dict[int, str] = {}

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _ensure_batch_generator(self) -> None:
        """Ensure BatchGenerator exists."""
        if self.batch_generator is None:
            from mlx_lm.generate import BatchGenerator
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.7, top_p=0.9)

            # Get stop tokens
            stop_tokens = set()
            if hasattr(self.tokenizer, "eos_token_id"):
                if isinstance(self.tokenizer.eos_token_id, list):
                    stop_tokens.update(self.tokenizer.eos_token_id)
                else:
                    stop_tokens.add(self.tokenizer.eos_token_id)

            self.batch_generator = BatchGenerator(
                model=self.model,
                max_tokens=256,  # Will be overridden per request
                stop_tokens=stop_tokens,
                sampler=sampler,
                prefill_batch_size=self.config.prefill_batch_size,
                completion_batch_size=self.config.completion_batch_size,
                prefill_step_size=self.config.prefill_step_size,
            )

            logger.info(
                f"[TextBackend] BatchGenerator created "
                f"(prefill_batch={self.config.prefill_batch_size}, "
                f"completion_batch={self.config.completion_batch_size})"
            )

    def add_request(self, request: UnifiedRequest) -> None:
        """Add request to text backend queue."""
        # Tokenize if needed
        if request.prompt_token_ids is None:
            request.prompt_token_ids = self.tokenizer.encode(request.prompt)
            request.num_prompt_tokens = len(request.prompt_token_ids)

        # Check prefix cache
        if self.prefix_cache is not None:
            cache, remaining = self.prefix_cache.fetch_cache(
                request.prompt_token_ids
            )
            if cache:
                request.prompt_cache = cache
                request.cached_tokens = len(request.prompt_token_ids) - len(remaining)
                request.remaining_tokens = remaining
                request.cache_hit_type = "hit"
            else:
                request.remaining_tokens = request.prompt_token_ids

        self.waiting.append(request)
        self.total_prompt_tokens += request.num_prompt_tokens

        logger.debug(
            f"[TextBackend] Added request {request.request_id[:12]} "
            f"tokens={request.num_prompt_tokens} "
            f"cache_hit={request.cache_hit_type}"
        )

    def _schedule_waiting(self) -> List[UnifiedRequest]:
        """Move requests from waiting to running."""
        self._ensure_batch_generator()

        scheduled = []

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting.popleft()

            # Determine tokens to process
            if request.remaining_tokens and len(request.remaining_tokens) == 0:
                tokens = request.prompt_token_ids[-1:]  # Last token for generation
            elif request.remaining_tokens:
                tokens = request.remaining_tokens
            else:
                tokens = request.prompt_token_ids

            # Insert into BatchGenerator
            try:
                uids = self.batch_generator.insert(
                    [tokens],
                    max_tokens=[request.sampling_params.max_tokens],
                    caches=[request.prompt_cache] if request.prompt_cache else None,
                )

                uid = uids[0]
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                request.batch_uid = uid
                request.status = "running"
                self.running[request.request_id] = request
                scheduled.append(request)

                logger.debug(
                    f"[TextBackend] Scheduled {request.request_id[:12]} "
                    f"uid={uid} tokens_to_prefill={len(tokens)}"
                )

            except Exception as e:
                logger.warning(
                    f"[TextBackend] Failed to schedule {request.request_id[:12]}: {e}"
                )
                # Put back in waiting queue
                self.waiting.appendleft(request)

        return scheduled

    def step(self) -> List[Any]:
        """Execute one generation step."""
        # Schedule waiting requests
        scheduled = self._schedule_waiting()

        outputs = []
        if self.batch_generator is not None and self.running:
            responses = self.batch_generator.next()

            if responses:
                for response in responses:
                    request_id = self.uid_to_request_id.get(response.uid)
                    if request_id is None:
                        continue

                    request = self.running.get(request_id)
                    if request is None:
                        continue

                    # Append token
                    request.output_tokens.append(response.token)
                    request.num_output_tokens = len(request.output_tokens)

                    # Record first token time
                    if request.first_token_time is None and request.num_output_tokens > 0:
                        request.first_token_time = time.time()

                    # Decode token
                    new_text = self.tokenizer.decode([response.token])

                    # Create output
                    from ..request import RequestOutput
                    output = RequestOutput(
                        request_id=request_id,
                        new_token_ids=[response.token],
                        new_text=new_text,
                        output_token_ids=list(request.output_tokens),
                        prompt_tokens=request.num_prompt_tokens,
                        completion_tokens=request.num_output_tokens,
                    )

                    # Check finish
                    if response.finish_reason is not None:
                        output.finished = True
                        output.finish_reason = response.finish_reason
                        request.finish_reason = response.finish_reason
                        request.output_text = self.tokenizer.decode(
                            request.output_tokens
                        )

                        # Cleanup
                        if request_id in self.running:
                            del self.running[request_id]
                        if request_id in self.request_id_to_uid:
                            uid = self.request_id_to_uid[request_id]
                            del self.uid_to_request_id[uid]
                            del self.request_id_to_uid[request_id]

                        self.num_requests_processed += 1
                        self.total_completion_tokens += request.num_output_tokens

                    outputs.append(output)

        return outputs

    def abort_request(self, request_id: str) -> bool:
        """Abort a request."""
        request = self.running.get(request_id)
        if request is None:
            # Check waiting queue
            for req in self.waiting:
                if req.request_id == request_id:
                    self.waiting.remove(req)
                    return True
            return False

        # Remove from batch generator
        if request_id in self.request_id_to_uid:
            uid = self.request_id_to_uid[request_id]
            if self.batch_generator is not None:
                self.batch_generator.remove([uid])
            del self.uid_to_request_id[uid]
            del self.request_id_to_uid[request_id]

        del self.running[request_id]
        request.status = "aborted"

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "processed": self.num_requests_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }

    def has_requests(self) -> bool:
        """Check if backend has pending requests."""
        return bool(self.waiting or self.running)


class MultimodalSchedulerBackend(SchedulerBackend):
    """
    Multimodal scheduler backend using MLLMBatchGenerator.
    
    Handles MLLM requests with vision cache support.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        config: UnifiedSchedulerConfig,
        vision_cache: Optional[Any] = None,
    ):
        super().__init__(model, processor, config)
        self.processor = processor
        self.vision_cache = vision_cache

        # Request queues
        self.waiting: deque[UnifiedRequest] = deque()
        self.running: Dict[str, UnifiedRequest] = {}

        # MLLMBatchGenerator (created lazily)
        self.batch_generator: Optional[Any] = None

        # UID mappings
        self.request_id_to_uid: Dict[str, int] = {}
        self.uid_to_request_id: Dict[int, str] = {}

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _ensure_batch_generator(self) -> None:
        """Ensure MLLMBatchGenerator exists."""
        if self.batch_generator is None:
            from ..mllm_batch_generator import MLLMBatchGenerator
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.7, top_p=0.9)

            # Get stop tokens from tokenizer
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            stop_tokens = set()
            if hasattr(tokenizer, "eos_token_id"):
                if isinstance(tokenizer.eos_token_id, list):
                    stop_tokens.update(tokenizer.eos_token_id)
                else:
                    stop_tokens.add(tokenizer.eos_token_id)

            self.batch_generator = MLLMBatchGenerator(
                model=self.model,
                processor=self.processor,
                max_tokens=256,
                stop_tokens=stop_tokens,
                sampler=sampler,
                prefill_batch_size=self.config.mllm_prefill_batch_size,
                completion_batch_size=self.config.mllm_completion_batch_size,
                prefill_step_size=self.config.mllm_prefill_step_size,
            )

            logger.info(
                f"[MultimodalBackend] MLLMBatchGenerator created "
                f"(prefill_batch={self.config.mllm_prefill_batch_size}, "
                f"completion_batch={self.config.mllm_completion_batch_size})"
            )

    def add_request(self, request: UnifiedRequest) -> None:
        """Add multimodal request to backend queue."""
        self.waiting.append(request)
        logger.debug(
            f"[MultimodalBackend] Added request {request.request_id[:12]} "
            f"images={len(request.images or [])} "
            f"videos={len(request.videos or [])}"
        )

    def step(self) -> List[Any]:
        """Execute one generation step."""
        self._ensure_batch_generator()

        # Schedule waiting requests
        scheduled = []
        batch_requests = []

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting.popleft()

            batch_req = {
                "uid": -1,
                "request_id": request.request_id,
                "prompt": request.prompt,
                "images": request.images,
                "videos": request.videos,
                "max_tokens": request.sampling_params.max_tokens,
                "temperature": request.sampling_params.temperature,
                "top_p": request.sampling_params.top_p,
            }
            batch_requests.append(batch_req)
            request.status = "running"
            self.running[request.request_id] = request
            scheduled.append(request)

        # Insert into MLLMBatchGenerator
        if batch_requests and self.batch_generator is not None:
            from ..mllm_batch_generator import MLLMBatchRequest
            mllm_requests = [
                MLLMBatchRequest(**br) for br in batch_requests
            ]
            uids = self.batch_generator.insert(mllm_requests)

            for uid, request in zip(uids, scheduled):
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                request.batch_uid = uid

        # Generate
        outputs = []
        if self.batch_generator is not None and self.running:
            responses = self.batch_generator.next()

            if responses:
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
                for response in responses:
                    request_id = self.uid_to_request_id.get(response.uid)
                    if request_id is None:
                        continue

                    request = self.running.get(request_id)
                    if request is None:
                        continue

                    request.output_tokens.append(response.token)
                    request.num_output_tokens = len(request.output_tokens)

                    new_text = tokenizer.decode([response.token])

                    from ..request import RequestOutput
                    output = RequestOutput(
                        request_id=request_id,
                        new_token_ids=[response.token],
                        new_text=new_text,
                        output_token_ids=list(request.output_tokens),
                        prompt_tokens=request.num_prompt_tokens,
                        completion_tokens=request.num_output_tokens,
                    )

                    if response.finish_reason is not None:
                        output.finished = True
                        output.finish_reason = response.finish_reason
                        request.output_text = tokenizer.decode(request.output_tokens)

                        # Cleanup
                        if request_id in self.running:
                            del self.running[request_id]
                        if request_id in self.request_id_to_uid:
                            uid = self.request_id_to_uid[request_id]
                            del self.uid_to_request_id[uid]
                            del self.request_id_to_uid[request_id]

                        self.num_requests_processed += 1
                        self.total_completion_tokens += request.num_output_tokens

                    outputs.append(output)

        return outputs

    def abort_request(self, request_id: str) -> bool:
        """Abort a request."""
        request = self.running.get(request_id)
        if request is None:
            for req in self.waiting:
                if req.request_id == request_id:
                    self.waiting.remove(req)
                    return True
            return False

        if request_id in self.request_id_to_uid:
            uid = self.request_id_to_uid[request_id]
            if self.batch_generator is not None:
                self.batch_generator.remove([uid])
            del self.uid_to_request_id[uid]
            del self.request_id_to_uid[request_id]

        del self.running[request_id]
        request.status = "aborted"

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "processed": self.num_requests_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }

    def has_requests(self) -> bool:
        """Check if backend has pending requests."""
        return bool(self.waiting or self.running)


class UnifiedScheduler:
    """
    Unified Scheduler for LLM and MLLM requests.

    Provides a single entry point for all request types,
    automatically routing to the appropriate backend.

    Architecture:
        UnifiedScheduler
        ├── TextSchedulerBackend (BatchGenerator)
        ├── MultimodalSchedulerBackend (MLLMBatchGenerator)
        └── Request Router (has_media -> Multimodal, else -> Text)

    Example:
        >>> scheduler = UnifiedScheduler(model, tokenizer, config)
        >>> scheduler.add_request(request)  # Auto-routes based on type
        >>> while scheduler.has_requests():
        ...     outputs = scheduler.step()
        ...     for output in outputs:
        ...         print(output.new_text)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        processor: Optional[Any] = None,
        config: Optional[UnifiedSchedulerConfig] = None,
        prefix_cache: Optional[Any] = None,
        vision_cache: Optional[Any] = None,
    ):
        """
        Initialize unified scheduler.

        Args:
            model: MLX model
            tokenizer: Tokenizer (or processor for MLLM)
            processor: Optional processor (for MLLM models)
            config: Scheduler configuration
            prefix_cache: Optional prefix cache for text requests
            vision_cache: Optional vision cache for multimodal requests
        """
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or UnifiedSchedulerConfig()
        self.prefix_cache = prefix_cache
        self.vision_cache = vision_cache

        # Create backends
        self.text_backend = TextSchedulerBackend(
            model=model,
            tokenizer=tokenizer,
            config=self.config,
            prefix_cache=prefix_cache,
        )

        self.multimodal_backend = None
        if processor is not None:
            self.multimodal_backend = MultimodalSchedulerBackend(
                model=model,
                processor=processor,
                config=self.config,
                vision_cache=vision_cache,
            )

        # Unified request tracking
        self.requests: Dict[str, UnifiedRequest] = {}

        # Async output queues
        self.output_queues: Dict[str, asyncio.Queue] = {}

        # Processing state
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None

        # Memory management
        self._step_count = 0
        self._clear_cache_interval = 32

        logger.info(
            f"[UnifiedScheduler] Initialized "
            f"text_backend=enabled "
            f"multimodal_backend={'enabled' if self.multimodal_backend else 'disabled'}"
        )

    def _classify_request(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
    ) -> RequestType:
        """
        Classify request type based on content.

        Args:
            prompt: Text prompt
            images: Optional image inputs
            videos: Optional video inputs

        Returns:
            RequestType (TEXT or MULTIMODAL)
        """
        if images or videos:
            return RequestType.MULTIMODAL
        return RequestType.TEXT

    def add_request(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        request_id: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> str:
        """
        Add a request to the scheduler.

        Automatically routes to appropriate backend based on content.

        Args:
            prompt: Text prompt
            images: Optional image inputs
            videos: Optional video inputs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            request_id: Optional custom request ID
            prompt_token_ids: Optional pre-tokenized prompt
            **kwargs: Additional parameters

        Returns:
            Request ID for tracking
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        from ..request import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        request_type = self._classify_request(prompt, images, videos)

        request = UnifiedRequest(
            request_id=request_id,
            request_type=request_type,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            images=images,
            videos=videos,
        )

        self.requests[request_id] = request

        # Route to appropriate backend
        if request_type == RequestType.MULTIMODAL and self.multimodal_backend:
            self.multimodal_backend.add_request(request)
        else:
            self.text_backend.add_request(request)

        logger.debug(
            f"[UnifiedScheduler] Added request {request_id[:12]} "
            f"type={request_type.value}"
        )

        return request_id

    def step(self) -> List[Any]:
        """
        Execute one scheduling step across all backends.

        Returns:
            List of RequestOutput objects from all backends
        """
        outputs = []

        # Run text backend
        if self.text_backend.has_requests():
            text_outputs = self.text_backend.step()
            outputs.extend(text_outputs)

        # Run multimodal backend
        if self.multimodal_backend and self.multimodal_backend.has_requests():
            mm_outputs = self.multimodal_backend.step()
            outputs.extend(mm_outputs)

        # Push to async queues
        for output in outputs:
            queue = self.output_queues.get(output.request_id)
            if queue is not None:
                try:
                    queue.put_nowait(output)
                    if output.finished:
                        queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

        # Periodic cache clear
        self._step_count += 1
        if self._step_count % self._clear_cache_interval == 0:
            mx.clear_cache()

        return outputs

    def abort_request(self, request_id: str) -> bool:
        """
        Abort a request.

        Args:
            request_id: Request ID to abort

        Returns:
            True if request was found and aborted
        """
        request = self.requests.get(request_id)
        if request is None:
            return False

        # Route abort to appropriate backend
        if request.request_type == RequestType.MULTIMODAL and self.multimodal_backend:
            success = self.multimodal_backend.abort_request(request_id)
        else:
            success = self.text_backend.abort_request(request_id)

        if success:
            request.status = "aborted"

        # Signal output queue
        if request_id in self.output_queues:
            try:
                self.output_queues[request_id].put_nowait(None)
            except asyncio.QueueFull:
                pass

        return success

    def has_requests(self) -> bool:
        """
        Check if any backend has pending requests.

        Returns:
            True if any requests pending
        """
        text_has = self.text_backend.has_requests()
        mm_has = self.multimodal_backend.has_requests() if self.multimodal_backend else False
        return text_has or mm_has

    def get_num_waiting(self) -> int:
        """Get total waiting requests across backends."""
        text_waiting = self.text_backend.get_stats().get("waiting", 0)
        mm_waiting = (
            self.multimodal_backend.get_stats().get("waiting", 0)
            if self.multimodal_backend else 0
        )
        return text_waiting + mm_waiting

    def get_num_running(self) -> int:
        """Get total running requests across backends."""
        text_running = self.text_backend.get_stats().get("running", 0)
        mm_running = (
            self.multimodal_backend.get_stats().get("running", 0)
            if self.multimodal_backend else 0
        )
        return text_running + mm_running

    # ========== Async API ==========

    async def start(self) -> None:
        """Start async processing loop."""
        if self._running:
            return

        self._running = True
        self._processing_task = asyncio.create_task(self._process_loop())
        logger.info("[UnifiedScheduler] Async processing started")

    async def stop(self) -> None:
        """Stop scheduler."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        logger.info("[UnifiedScheduler] Stopped")

    async def _process_loop(self) -> None:
        """Main async processing loop."""
        while self._running:
            try:
                if self.has_requests():
                    self.step()
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[UnifiedScheduler] Process loop error: {e}")
                await asyncio.sleep(0.1)

    async def add_request_async(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Add request with output queue for streaming.

        Args:
            prompt: Text prompt
            images: Image inputs
            videos: Video inputs
            **kwargs: Additional parameters

        Returns:
            Request ID
        """
        request_id = self.add_request(
            prompt=prompt,
            images=images,
            videos=videos,
            **kwargs,
        )

        # Create output queue
        self.output_queues[request_id] = asyncio.Queue()

        return request_id

    async def stream_outputs(
        self,
        request_id: str,
    ) -> AsyncIterator[Any]:
        """
        Stream outputs for a request.

        Args:
            request_id: Request ID to stream

        Yields:
            RequestOutput objects
        """
        output_queue = self.output_queues.get(request_id)
        if output_queue is None:
            return

        finished_normally = False
        try:
            while True:
                output = await output_queue.get()
                if output is None:
                    finished_normally = True
                    break
                yield output
                if output.finished:
                    finished_normally = True
                    break
        finally:
            if not finished_normally:
                self.abort_request(request_id)
            if request_id in self.output_queues:
                del self.output_queues[request_id]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get unified scheduler statistics.

        Returns:
            Dict with combined stats from all backends
        """
        stats = {
            "text_backend": self.text_backend.get_stats(),
            "requests": len(self.requests),
        }

        if self.multimodal_backend:
            stats["multimodal_backend"] = self.multimodal_backend.get_stats()

        # Metal memory stats
        try:
            if mx.metal.is_available():
                stats["metal_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
                stats["metal_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
                stats["metal_cache_memory_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:
            pass

        return stats

    def reset(self) -> None:
        """Reset scheduler state."""
        # Abort all requests
        for request_id in list(self.requests.keys()):
            self.abort_request(request_id)

        self.requests.clear()
        self.output_queues.clear()
        self._step_count = 0

        logger.info("[UnifiedScheduler] Reset complete")