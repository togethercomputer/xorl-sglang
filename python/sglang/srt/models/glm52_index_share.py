"""Typed per-invocation IndexShare state for the GLM-5.2 serving model."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.nsa.glm52_selector_fast import (
        Glm52ProducerSelectionPlan,
    )


@dataclass(frozen=True)
class CanonicalLogicalIndices:
    """Sparse keys expressed only as logical sequence positions.

    ``producer_layer`` is the IndexShare producer's layer id — the explicit
    identity consumers use to key producer-scoped state (e.g. selection
    plans). Tensor storage pointers are not identity: they outlive nothing.
    ``selection_plan`` optionally carries the producer-built selected-KV plan
    (physical indices, counts, deferred contract status), computed once at
    publish and sharing the payload's per-forward lifetime.
    """

    values: torch.Tensor
    producer_layer: int | None = None
    invocation: int | None = None
    selection_plan: Glm52ProducerSelectionPlan | None = None

    def __post_init__(self) -> None:
        if self.values.dtype not in (torch.int32, torch.int64):
            raise TypeError("Canonical logical indices must use an integer dtype")
        if self.values.ndim < 2:
            raise ValueError(
                "Canonical logical indices must carry query and top-k dimensions"
            )


@dataclass(frozen=True)
class Glm52IndexSharePlan:
    indexer_types: tuple[str, ...]
    producer_by_layer: tuple[int, ...]
    full_layers: tuple[int, ...]
    shared_layers: tuple[int, ...]

    @classmethod
    def from_config(cls, config) -> Glm52IndexSharePlan:
        indexer_types = tuple(getattr(config, "indexer_types", ()) or ())
        num_layers = int(config.num_hidden_layers)
        if len(indexer_types) != num_layers:
            raise ValueError(
                f"GLM-5.2 indexer_types has {len(indexer_types)} entries, expected {num_layers}"
            )
        freq = int(getattr(config, "index_topk_freq", 0))
        skip = int(getattr(config, "index_skip_topk_offset", -1))
        pattern = getattr(config, "index_topk_pattern", None)
        if freq <= 0 or skip < 0:
            raise ValueError("GLM-5.2 IndexShare frequency metadata is invalid")
        if pattern is not None:
            pattern = tuple(int(value) for value in pattern)
            if len(pattern) != num_layers or any(
                value not in (0, 1) for value in pattern
            ):
                raise ValueError(
                    "GLM-5.2 index_topk_pattern must contain one 0/1 value per layer"
                )

        producer = None
        producer_by_layer = []
        full_layers = []
        shared_layers = []
        for layer, kind in enumerate(indexer_types):
            if kind not in {"full", "shared"}:
                raise ValueError(f"Unknown indexer type {kind!r} at layer {layer}")
            expected_full = (
                bool(pattern[layer])
                if pattern is not None
                else layer < skip or (layer - skip + 1) % freq == 0
            )
            if (kind == "full") != expected_full:
                raise ValueError(
                    f"Layer {layer} indexer type disagrees with frequency metadata"
                )
            if kind == "full":
                producer = layer
                full_layers.append(layer)
            else:
                if producer is None:
                    raise ValueError(
                        "An IndexShare stage cannot begin with a shared layer"
                    )
                shared_layers.append(layer)
            producer_by_layer.append(producer)
        return cls(
            indexer_types,
            tuple(producer_by_layer),
            tuple(full_layers),
            tuple(shared_layers),
        )

    def validate_pipeline_stage(self, start_layer: int, end_layer: int) -> None:
        if not 0 <= start_layer < end_layer <= len(self.indexer_types):
            raise ValueError(
                f"Invalid GLM-5.2 pipeline layer range [{start_layer}, {end_layer})"
            )
        if self.indexer_types[start_layer] != "full":
            producer = self.producer_by_layer[start_layer]
            raise ValueError(
                "GLM-5.2 pipeline stages must begin on an IndexShare producer; "
                f"layer {start_layer} would depend on producer {producer} from another stage"
            )


@dataclass
class Glm52IndexShareContext:
    invocation: int
    plan: Glm52IndexSharePlan
    _published: dict[int, CanonicalLogicalIndices | None] = field(default_factory=dict)
    closed: bool = False

    def publish(self, layer_id: int, indices: CanonicalLogicalIndices | None) -> None:
        self._require_open()
        if self.plan.indexer_types[layer_id] != "full":
            raise RuntimeError(
                f"Shared layer {layer_id} cannot publish IndexShare state"
            )
        if layer_id in self._published:
            raise RuntimeError(f"Full layer {layer_id} published twice")
        if indices is not None and not isinstance(indices, CanonicalLogicalIndices):
            raise TypeError(
                "IndexShare may publish only typed canonical logical indices"
            )
        self._published[layer_id] = indices

    def consume(
        self, layer_id: int, *, require_indices: bool
    ) -> CanonicalLogicalIndices | None:
        self._require_open()
        if self.plan.indexer_types[layer_id] != "shared":
            raise RuntimeError(f"Full layer {layer_id} must compute IndexShare state")
        producer = self.plan.producer_by_layer[layer_id]
        if producer not in self._published:
            raise RuntimeError(
                f"Shared layer {layer_id} ran before producer {producer}"
            )
        value = self._published[producer]
        if require_indices and value is None:
            raise RuntimeError(
                f"Producer {producer} did not publish selected logical indices"
            )
        return value

    def close(self) -> None:
        self._published.clear()
        self.closed = True

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("IndexShare context is closed")


class Glm52IndexShareManager:
    def __init__(self, plan: Glm52IndexSharePlan):
        self.plan = plan
        self._active = None
        self._counter = 0

    def begin(self) -> Glm52IndexShareContext:
        if self._active is not None:
            raise RuntimeError(
                "GLM-5.2 phase one permits one live IndexShare invocation"
            )
        context = Glm52IndexShareContext(self._counter, self.plan)
        self._counter += 1
        self._active = context
        return context

    def end(self, context: Glm52IndexShareContext) -> None:
        if context is not self._active:
            raise RuntimeError("Cannot close a stale IndexShare context")
        context.close()
        self._active = None

    @contextmanager
    def invocation(self) -> Iterator[Glm52IndexShareContext]:
        context = self.begin()
        try:
            yield context
        finally:
            self.end(context)


__all__ = [
    "CanonicalLogicalIndices",
    "Glm52IndexShareContext",
    "Glm52IndexShareManager",
    "Glm52IndexSharePlan",
]
