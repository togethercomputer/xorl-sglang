"""Canonical BF16 MoE contribution folding for exact serving lanes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist

from sglang.srt.distributed.canonical_moe_kernels import (
    fused_balanced_adjacent_bf16_tree,
)

CANONICAL_MOE_FOLD_VERSION = "canonical_moe_fold_v1"
GLM52_CANONICAL_MOE_VERSION = "canonical_moe_reduce_v1"
GLM52_CANONICAL_MOE_V3_VERSION = "glm52_canonical_moe_reduce_v3"
GLM52_CANONICAL_MOE_V3B_VERSION = "glm52_canonical_moe_reduce_v3b"
GLM52_SAMPLER_LOCAL_POLICY = "glm52_routed_final_scaled_then_shared_ep_slice_bf16_v2"


class CanonicalDistribution(str, Enum):
    OWNER_SHARDED = "owner_sharded"
    CONSUMER_SHARDED = "consumer_sharded"
    REPLICATED_CANONICAL = "replicated_canonical"


class CanonicalTransport(str, Enum):
    DENSE_V1 = GLM52_CANONICAL_MOE_VERSION
    CP_SHARDED_V3 = GLM52_CANONICAL_MOE_V3_VERSION
    REPLICATED_DECODE_V3 = "glm52_canonical_moe_replicated_decode_v3"
    REPLICATED_DECODE_V3B = "glm52_canonical_moe_replicated_decode_v3b"


@dataclass(frozen=True)
class SamplerParallelPlan:
    contributor_count: int
    physical_ranks: tuple[int, ...]
    physical_to_logical: tuple[int, ...]
    launcher_tp_size: int
    global_world_size: int
    pp_rank: int = 0
    stage_layer_range: tuple[int, int] | None = None
    effective_dense_tp: int = 1
    pp_size: int = 1
    ep_size: int = 8
    attention_cp_size: int = 8
    attention_dp_size: int = 1
    production: bool = False
    version: str = GLM52_CANONICAL_MOE_VERSION

    def __post_init__(self) -> None:
        if self.contributor_count not in (2, 4, 8, 16):
            raise ValueError("Canonical MoE admits exactly 2, 4, 8, or 16 contributors")
        if (
            len(self.physical_ranks) != self.contributor_count
            or len(set(self.physical_ranks)) != self.contributor_count
        ):
            raise ValueError(
                "Physical ranks must name each stage-local contributor once"
            )
        if tuple(sorted(self.physical_to_logical)) != tuple(
            range(self.contributor_count)
        ):
            raise ValueError(
                "physical_to_logical must be a complete ordinal permutation"
            )
        if self.version != GLM52_CANONICAL_MOE_VERSION:
            raise ValueError(f"Unsupported canonical MoE version {self.version}")
        if self.pp_size not in (1, 2) or not 0 <= self.pp_rank < self.pp_size:
            raise ValueError(
                "GLM-5.2 sampler admits only PP1 or PP2 with a valid stage rank"
            )
        if self.effective_dense_tp != 1:
            raise ValueError("GLM-5.2 sampler requires effective dense TP1")
        if self.global_world_size != self.launcher_tp_size * self.pp_size:
            raise ValueError("Global world size must equal launcher TP times PP")
        if self.stage_layer_range is not None:
            start_layer, end_layer = self.stage_layer_range
            if start_layer < 0 or start_layer >= end_layer:
                raise ValueError(
                    "Stage layer range must be a non-empty half-open interval"
                )
        expected_physical_ranks = tuple(
            range(
                self.pp_rank * self.launcher_tp_size,
                (self.pp_rank + 1) * self.launcher_tp_size,
            )
        )
        if self.physical_ranks != expected_physical_ranks:
            raise ValueError(
                "Physical ranks must be the ordered ranks of this pipeline stage"
            )
        if self.production:
            if self.contributor_count not in (8, 16):
                raise ValueError(
                    "Production GLM-5.2 requires exactly 8 or 16 contributors"
                )
            if self.physical_to_logical != tuple(range(self.contributor_count)):
                raise ValueError(
                    "Production GLM-5.2 requires identity logical contributor ordinals"
                )
            if (
                self.ep_size != self.contributor_count
                or self.attention_cp_size * self.attention_dp_size
                != self.contributor_count
                or self.launcher_tp_size != self.contributor_count
            ):
                raise ValueError(
                    "Production GLM-5.2 sampler requires launcher TP and EP to "
                    "equal the contributor count, with attention DP and CP "
                    "exactly covering the logical row owners"
                )

    @classmethod
    def glm52(
        cls,
        *,
        contributors: int = 8,
        pp_size: int = 1,
        pp_rank: int = 0,
        physical_ranks: tuple[int, ...] | None = None,
        attention_dp_size: int = 1,
    ) -> SamplerParallelPlan:
        if attention_dp_size <= 0 or contributors % attention_dp_size:
            raise ValueError(
                "GLM-5.2 attention DP must be a positive divisor of the contributor count"
            )
        ranks = (
            tuple(
                range(
                    pp_rank * contributors,
                    (pp_rank + 1) * contributors,
                )
            )
            if physical_ranks is None
            else tuple(physical_ranks)
        )
        return cls(
            contributors,
            ranks,
            tuple(range(contributors)),
            launcher_tp_size=contributors,
            global_world_size=contributors * pp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            ep_size=contributors,
            attention_cp_size=contributors // attention_dp_size,
            attention_dp_size=attention_dp_size,
            production=True,
        )

    @classmethod
    def primitive(
        cls,
        contributors: int,
        *,
        physical_to_logical: tuple[int, ...] | None = None,
    ) -> SamplerParallelPlan:
        ranks = tuple(range(contributors))
        return cls(
            contributors,
            ranks,
            ranks if physical_to_logical is None else physical_to_logical,
            launcher_tp_size=contributors,
            global_world_size=contributors,
            ep_size=contributors,
            attention_cp_size=contributors,
        )

    def validate_runtime(
        self,
        *,
        group: dist.ProcessGroup,
        launcher_tp_size: int,
        effective_dense_tp: int,
        pp_size: int,
        ep_size: int,
        attention_cp_size: int,
        attention_dp_size: int = 1,
    ) -> None:
        actual = (
            dist.get_world_size(),
            launcher_tp_size,
            effective_dense_tp,
            pp_size,
            ep_size,
            attention_cp_size,
            attention_dp_size,
            dist.get_world_size(group),
        )
        expected = (
            self.global_world_size,
            self.launcher_tp_size,
            self.effective_dense_tp,
            self.pp_size,
            self.ep_size,
            self.attention_cp_size,
            self.attention_dp_size,
            self.contributor_count,
        )
        if self.production and actual != expected:
            raise RuntimeError(
                f"Resolved GLM-5.2 sampler topology must be {expected}, got {actual}"
            )
        get_group_ranks = getattr(dist, "get_process_group_ranks", None)
        if get_group_ranks is not None:
            actual_ranks = tuple(get_group_ranks(group))
            if actual_ranks != self.physical_ranks:
                raise RuntimeError(
                    f"Runtime combine-group physical rank order {actual_ranks} does not match {self.physical_ranks}"
                )

    def validate_cuda_graph_policy(self, *, disable_cuda_graph: bool) -> None:
        if self.production and self.pp_size > 1 and not disable_cuda_graph:
            raise RuntimeError(
                "GLM-5.2 canonical PP2 requires --disable-cuda-graph until "
                "pipeline-aware graph replay is certified"
            )

    @property
    def identity(self) -> str:
        payload = {
            "contributor_count": self.contributor_count,
            "physical_ranks": self.physical_ranks,
            "physical_to_logical": self.physical_to_logical,
            "launcher_tp_size": self.launcher_tp_size,
            "global_world_size": self.global_world_size,
            "pp_rank": self.pp_rank,
            "stage_layer_range": self.stage_layer_range,
            "effective_dense_tp": self.effective_dense_tp,
            "pp_size": self.pp_size,
            "ep_size": self.ep_size,
            "attention_cp_size": self.attention_cp_size,
            "attention_dp_size": self.attention_dp_size,
            "production": self.production,
            "version": self.version,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class CanonicalRowSlots:
    absolute_positions: torch.Tensor
    valid_mask: torch.Tensor
    capacity: int

    @classmethod
    def from_positions(
        cls,
        absolute_positions: torch.Tensor,
        *,
        capacity: int | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> CanonicalRowSlots:
        positions = absolute_positions.reshape(-1).to(torch.int64)
        capacity = positions.numel() if capacity is None else capacity
        if positions.numel() > capacity:
            raise ValueError(
                f"row count {positions.numel()} exceeds capacity {capacity}"
            )
        padded = positions.new_full((capacity,), -1)
        if valid_mask is None:
            source_valid = positions >= 0
        else:
            source_valid = valid_mask.reshape(-1).to(
                device=positions.device, dtype=torch.bool
            )
            if source_valid.shape != positions.shape:
                raise ValueError(
                    "Position validity must match the absolute-position shape"
                )
        valid = torch.zeros((capacity,), dtype=torch.bool, device=positions.device)
        padded[: positions.numel()] = positions
        valid[: positions.numel()] = source_valid
        return cls(padded, valid, capacity)

    def owners(self, contributors: int) -> torch.Tensor:
        return self.absolute_positions.clamp_min(0).remainder(contributors)


@dataclass(frozen=True)
class CanonicalMoEOutput:
    values: torch.Tensor
    owner_mask: torch.Tensor
    slots: CanonicalRowSlots
    contract_status: torch.Tensor

    def raise_for_status(self) -> None:
        status = int(self.contract_status.item())
        if status != 0:
            raise RuntimeError(
                f"Canonical MoE device contract failed with status {status}"
            )


class CanonicalDeferredStatusBook:
    """Per-layer device status slots checked once per model forward.

    Producers keep writing the shared workspace status scalar unchanged;
    ``record`` copies that scalar into this layer's device slot (no host
    sync), and ``check_and_clear`` performs the forward's single host
    transfer, raising the same fail-closed contract error while naming
    every failing layer.
    """

    def __init__(self, num_layers: int) -> None:
        if num_layers <= 0:
            raise ValueError("Deferred status book requires at least one layer")
        self.num_layers = num_layers
        self._statuses: torch.Tensor | None = None
        self._recorded = False

    def record(self, layer_id: int, status: torch.Tensor) -> None:
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(
                f"Layer {layer_id} is outside the {self.num_layers}-layer status book"
            )
        if self._statuses is None:
            self._statuses = torch.zeros(
                (self.num_layers,), dtype=torch.int32, device=status.device
            )
        self._statuses[layer_id].copy_(status)
        self._recorded = True

    def check_and_clear(self) -> torch.Tensor | None:
        """Run the forward's one host transfer; return per-layer statuses."""
        if not self._recorded or self._statuses is None:
            return None
        # Explicit copy: .cpu() on an already-CPU tensor aliases, and the
        # zero_() below must never erase the statuses before they are read.
        statuses = self._statuses.detach().to("cpu", copy=True)
        self._statuses.zero_()
        self._recorded = False
        failing = [
            (layer_id, status)
            for layer_id, status in enumerate(statuses.tolist())
            if status != 0
        ]
        if failing:
            detail = ", ".join(
                f"status {status} at layer {layer_id}" for layer_id, status in failing
            )
            raise RuntimeError(f"Canonical MoE device contract failed with {detail}")
        return statuses


@dataclass
class CanonicalMoEWorkspace:
    plan_hash: str
    capacity: int
    payload_shape: tuple[int, ...]
    send: torch.Tensor
    receive: torch.Tensor
    gathered: torch.Tensor
    owner_values: torch.Tensor
    result: torch.Tensor
    zero: torch.Tensor
    logical_to_group: torch.Tensor
    status: torch.Tensor
    offsets: torch.Tensor
    counts: torch.Tensor

    @classmethod
    def allocate(
        cls,
        local_partial: torch.Tensor,
        *,
        plan: SamplerParallelPlan,
        group: dist.ProcessGroup,
    ) -> CanonicalMoEWorkspace:
        contributors = plan.contributor_count
        capacity = local_partial.shape[0]
        payload_shape = tuple(local_partial.shape[1:])
        collective_shape = (contributors, capacity, *payload_shape)
        plan.validate_runtime(
            group=group,
            launcher_tp_size=plan.launcher_tp_size,
            effective_dense_tp=plan.effective_dense_tp,
            pp_size=plan.pp_size,
            ep_size=plan.ep_size,
            attention_cp_size=plan.attention_cp_size,
            attention_dp_size=plan.attention_dp_size,
        )
        return cls(
            plan_hash=plan.identity,
            capacity=capacity,
            payload_shape=payload_shape,
            send=torch.empty(
                collective_shape, dtype=local_partial.dtype, device=local_partial.device
            ),
            receive=torch.empty(
                collective_shape, dtype=local_partial.dtype, device=local_partial.device
            ),
            gathered=torch.empty(
                collective_shape, dtype=local_partial.dtype, device=local_partial.device
            ),
            owner_values=torch.empty_like(local_partial),
            result=torch.empty_like(local_partial),
            zero=torch.zeros(
                (), dtype=local_partial.dtype, device=local_partial.device
            ),
            logical_to_group=torch.tensor(
                tuple(
                    plan.physical_to_logical.index(logical)
                    for logical in range(contributors)
                ),
                dtype=torch.long,
                device=local_partial.device,
            ),
            status=torch.zeros((), dtype=torch.int32, device=local_partial.device),
            offsets=torch.arange(
                contributors, dtype=torch.int64, device=local_partial.device
            )
            * capacity,
            counts=torch.full(
                (contributors,),
                capacity,
                dtype=torch.int64,
                device=local_partial.device,
            ),
        )

    def validate(self, local_partial: torch.Tensor, plan: SamplerParallelPlan) -> None:
        if self.plan_hash != plan.identity:
            raise RuntimeError(
                "Canonical MoE workspace topology does not match the immutable plan"
            )
        if local_partial.shape != (self.capacity, *self.payload_shape):
            raise RuntimeError(
                "Canonical MoE workspace shape does not match the fixed capture shape"
            )


@dataclass
class CanonicalMoEV3Workspace:
    """Mode-shaped storage for the eager v3 serving transports."""

    plan_hash: str
    distribution: CanonicalDistribution
    capacity: int
    local_capacity: int
    payload_shape: tuple[int, ...]
    masked_input: torch.Tensor
    collective: torch.Tensor
    received: torch.Tensor | None
    result: torch.Tensor
    zero: torch.Tensor
    logical_to_group: torch.Tensor
    status: torch.Tensor

    @classmethod
    def allocate(
        cls,
        local_partial: torch.Tensor,
        *,
        plan: SamplerParallelPlan,
        group: dist.ProcessGroup,
        distribution: CanonicalDistribution,
    ) -> CanonicalMoEV3Workspace:
        if distribution not in {
            CanonicalDistribution.CONSUMER_SHARDED,
            CanonicalDistribution.REPLICATED_CANONICAL,
        }:
            raise ValueError(
                f"v3 does not admit output distribution {distribution.value}"
            )
        plan.validate_runtime(
            group=group,
            launcher_tp_size=plan.launcher_tp_size,
            effective_dense_tp=plan.effective_dense_tp,
            pp_size=plan.pp_size,
            ep_size=plan.ep_size,
            attention_cp_size=plan.attention_cp_size,
            attention_dp_size=plan.attention_dp_size,
        )
        contributors = plan.contributor_count
        capacity = local_partial.shape[0]
        payload_shape = tuple(local_partial.shape[1:])
        logical_to_group = torch.tensor(
            tuple(
                plan.physical_to_logical.index(logical)
                for logical in range(contributors)
            ),
            dtype=torch.long,
            device=local_partial.device,
        )
        if distribution is CanonicalDistribution.CONSUMER_SHARDED:
            if capacity % contributors:
                raise ValueError(
                    "v3 consumer-sharded transport requires equal padded CP-source capacity"
                )
            local_capacity = capacity // contributors
            collective = torch.empty_like(local_partial)
            received = torch.empty(
                (contributors, local_capacity, *payload_shape),
                dtype=local_partial.dtype,
                device=local_partial.device,
            )
            result = torch.empty(
                (local_capacity, *payload_shape),
                dtype=local_partial.dtype,
                device=local_partial.device,
            )
        else:
            local_capacity = capacity
            collective = torch.empty(
                (contributors, capacity, *payload_shape),
                dtype=local_partial.dtype,
                device=local_partial.device,
            )
            received = None
            result = torch.empty_like(local_partial)
        return cls(
            plan_hash=plan.identity,
            distribution=distribution,
            capacity=capacity,
            local_capacity=local_capacity,
            payload_shape=payload_shape,
            masked_input=torch.empty_like(local_partial),
            collective=collective,
            received=received,
            result=result,
            zero=torch.zeros(
                (), dtype=local_partial.dtype, device=local_partial.device
            ),
            logical_to_group=logical_to_group,
            status=torch.zeros((), dtype=torch.int32, device=local_partial.device),
        )

    def validate(
        self,
        local_partial: torch.Tensor,
        plan: SamplerParallelPlan,
        distribution: CanonicalDistribution,
    ) -> None:
        if self.plan_hash != plan.identity:
            raise RuntimeError(
                "Canonical MoE v3 workspace topology does not match the immutable plan"
            )
        if self.distribution is not distribution:
            raise RuntimeError("Canonical MoE v3 workspace distribution changed")
        if local_partial.shape != (self.capacity, *self.payload_shape):
            raise RuntimeError("Canonical MoE v3 workspace shape changed")


def _balanced_adjacent_tree(partials: torch.Tensor) -> torch.Tensor:
    if partials.dtype is not torch.bfloat16:
        raise TypeError("Canonical MoE arithmetic requires BF16 partials")
    if partials.shape[0] not in (2, 4, 8, 16):
        raise ValueError(
            "Canonical MoE arithmetic requires 2, 4, 8, or 16 contributors"
        )
    level = tuple(partials.unbind(0))
    while len(level) > 1:
        level = tuple(
            (level[index] + level[index + 1]).to(torch.bfloat16)
            for index in range(0, len(level), 2)
        )
    return level[0]


def _balanced_adjacent_tree_batched(partials: torch.Tensor) -> torch.Tensor:
    """Evaluate the canonical tree with one tensor add per tree level."""
    if partials.dtype is not torch.bfloat16:
        raise TypeError("Canonical MoE arithmetic requires BF16 partials")
    if partials.shape[0] not in (2, 4, 8, 16):
        raise ValueError(
            "Canonical MoE arithmetic requires 2, 4, 8, or 16 contributors"
        )
    level = partials
    while level.shape[0] > 1:
        level = (level[0::2] + level[1::2]).to(torch.bfloat16)
    return level[0]


def _fused_tree_or_cpu_reference(
    partials: torch.Tensor,
    logical_to_group: torch.Tensor,
    *,
    identity_order: bool,
    output: torch.Tensor,
) -> torch.Tensor:
    if partials.is_cuda:
        return fused_balanced_adjacent_bf16_tree(
            partials,
            logical_to_group,
            identity_order=identity_order,
            output=output,
        )
    logical_sources = (
        partials if identity_order else partials.index_select(0, logical_to_group)
    )
    output.copy_(_balanced_adjacent_tree_batched(logical_sources))
    return output


def canonical_moe_fold_v1(
    partials_by_physical_ordinal: torch.Tensor,
    *,
    logical_to_physical: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fold logically identified contributions with the version-1 BF16 tree.

    Raw transport may deliver contributors in a physical order. Callers either
    pass ``logical_to_physical`` or, when physical and logical ordinals are
    identical, leave it unset. The fused CUDA implementation and the eager CPU
    implementation are independent from the trainer implementation but obey
    the same versioned arithmetic contract.
    """
    partials = partials_by_physical_ordinal
    if partials.ndim < 2:
        raise ValueError(
            "Canonical MoE fold requires contributor and payload dimensions"
        )
    if partials.dtype is not torch.bfloat16:
        raise TypeError("Canonical MoE fold requires BF16 partials")
    contributors = partials.shape[0]
    if contributors not in (2, 4, 8, 16):
        raise ValueError("Canonical MoE fold requires 2, 4, 8, or 16 contributors")

    identity_order = logical_to_physical is None
    if identity_order:
        logical_to_physical = torch.empty(
            (contributors,), dtype=torch.int64, device=partials.device
        )
    elif (
        logical_to_physical.shape != (contributors,)
        or logical_to_physical.dtype is not torch.int64
        or logical_to_physical.device != partials.device
        or not logical_to_physical.is_contiguous()
    ):
        raise ValueError(
            "logical_to_physical must be a contiguous int64 tensor on the partial device"
        )
    if output is None:
        output = torch.empty_like(partials[0])
    elif (
        output.shape != partials.shape[1:]
        or output.dtype is not partials.dtype
        or output.device != partials.device
        or not output.is_contiguous()
    ):
        raise ValueError("Canonical MoE fold output metadata does not match partials")
    return _fused_tree_or_cpu_reference(
        partials,
        logical_to_physical,
        identity_order=identity_order,
        output=output,
    )


def canonical_moe_reference(
    partials: torch.Tensor, slots: CanonicalRowSlots
) -> torch.Tensor:
    folded = _balanced_adjacent_tree(partials)
    mask = slots.valid_mask.view(-1, *([1] * (folded.ndim - 1)))
    return torch.where(mask, folded, torch.zeros_like(folded))


def canonicalize_glm52_local_partial(
    local_partial: torch.Tensor,
    slots: CanonicalRowSlots,
    *,
    plan: SamplerParallelPlan,
    group: dist.ProcessGroup,
    layer_id: int,
    distribution: CanonicalDistribution = CanonicalDistribution.REPLICATED_CANONICAL,
    graph_capture: bool = False,
    workspace: CanonicalMoEWorkspace | None = None,
) -> CanonicalMoEOutput:
    if local_partial.dtype is not torch.bfloat16:
        raise TypeError("GLM-5.2 local MoE contribution must be BF16")
    if local_partial.shape[0] != slots.capacity:
        raise ValueError("Local partial rows must equal the fixed slot capacity")
    if workspace is None:
        workspace = CanonicalMoEWorkspace.allocate(
            local_partial, plan=plan, group=group
        )
    workspace.validate(local_partial, plan)

    group_rank = dist.get_rank(group)
    logical_ordinal = plan.physical_to_logical[group_rank]
    logical_to_group = tuple(
        plan.physical_to_logical.index(logical)
        for logical in range(plan.contributor_count)
    )
    owners = slots.owners(plan.contributor_count)
    payload_shape = local_partial.shape[1:]
    mask_tail = (1,) * len(payload_shape)
    for owner in range(plan.contributor_count):
        row_mask = (slots.valid_mask & (owners == owner)).view(-1, *mask_tail)
        torch.where(
            row_mask,
            local_partial,
            workspace.zero,
            out=workspace.send[logical_to_group[owner]],
        )

    dist.all_to_all_single(
        workspace.receive.view(plan.contributor_count * slots.capacity, *payload_shape),
        workspace.send.view(plan.contributor_count * slots.capacity, *payload_shape),
        group=group,
    )
    folded = _balanced_adjacent_tree(
        workspace.receive.index_select(0, workspace.logical_to_group)
    )
    owner_mask = slots.valid_mask & (owners == logical_ordinal)
    torch.where(
        owner_mask.view(-1, *([1] * len(payload_shape))),
        folded,
        workspace.zero,
        out=workspace.owner_values,
    )

    if distribution is CanonicalDistribution.OWNER_SHARDED:
        workspace.result.copy_(workspace.owner_values)
    else:
        dist.all_gather_into_tensor(
            workspace.gathered.view(
                plan.contributor_count * slots.capacity, *payload_shape
            ),
            workspace.owner_values,
            group=group,
        )
        workspace.result.zero_()
        for owner in range(plan.contributor_count):
            row_mask = (slots.valid_mask & (owners == owner)).view(-1, *mask_tail)
            torch.where(
                row_mask,
                workspace.gathered[logical_to_group[owner]],
                workspace.result,
                out=workspace.result,
            )

    invalid_position = slots.valid_mask & (slots.absolute_positions < 0)
    workspace.status.copy_(invalid_position.any().to(torch.int32))

    return CanonicalMoEOutput(workspace.result, owner_mask, slots, workspace.status)


def canonicalize_glm52_local_partial_v3(
    local_partial: torch.Tensor,
    slots: CanonicalRowSlots,
    *,
    plan: SamplerParallelPlan,
    group: dist.ProcessGroup,
    layer_id: int,
    distribution: CanonicalDistribution,
    graph_capture: bool = False,
    workspace: CanonicalMoEV3Workspace | None = None,
) -> CanonicalMoEOutput:
    """Fold identical BF16 contributors with mode-specific byte transport.

    Prefill CP sends each rank-major source bucket directly to its consuming CP
    rank. Eager decode gathers one unexpanded partial from every contributor and
    folds locally on every rank. Both modes preserve the v1 logical contributor
    order and balanced adjacent BF16 tree; dense v1 remains the independent
    oracle.
    """
    if local_partial.dtype is not torch.bfloat16:
        raise TypeError("GLM-5.2 local MoE contribution must be BF16")
    if local_partial.shape[0] != slots.capacity:
        raise ValueError("Local partial rows must equal the v3 source capacity")
    if distribution not in {
        CanonicalDistribution.CONSUMER_SHARDED,
        CanonicalDistribution.REPLICATED_CANONICAL,
    }:
        raise ValueError(f"Canonical MoE v3 does not admit {distribution.value}")
    if workspace is None:
        workspace = CanonicalMoEV3Workspace.allocate(
            local_partial,
            plan=plan,
            group=group,
            distribution=distribution,
        )
    workspace.validate(local_partial, plan, distribution)

    payload_shape = local_partial.shape[1:]
    mask_tail = (1,) * len(payload_shape)
    torch.where(
        slots.valid_mask.view(-1, *mask_tail),
        local_partial,
        workspace.zero,
        out=workspace.masked_input,
    )
    group_rank = dist.get_rank(group)
    invalid_position = slots.valid_mask & (slots.absolute_positions < 0)

    if distribution is CanonicalDistribution.CONSUMER_SHARDED:
        contributors = plan.contributor_count
        assert workspace.received is not None
        # Rank-major FULL prefill rows already form equal destination buckets.
        # The all-to-all moves only one source payload; it does not create the
        # contributor-count zero expansion used by dense v1.
        dist.all_to_all_single(
            workspace.received.view(
                contributors * workspace.local_capacity, *payload_shape
            ),
            workspace.masked_input,
            group=group,
        )
        logical_sources = workspace.received.index_select(0, workspace.logical_to_group)
        folded = _balanced_adjacent_tree(logical_sources)
        start = group_rank * workspace.local_capacity
        end = start + workspace.local_capacity
        local_valid = slots.valid_mask[start:end]
        torch.where(
            local_valid.view(-1, *mask_tail),
            folded,
            workspace.zero,
            out=workspace.result,
        )
        output_slots = CanonicalRowSlots(
            absolute_positions=slots.absolute_positions[start:end],
            valid_mask=local_valid,
            capacity=workspace.local_capacity,
        )
        owner_mask = local_valid
    else:
        # Decode rows are replicated across CP. Gather each real local partial
        # exactly once, restore logical contributor order, and fold identically
        # on every rank. No owner slots or replication gather are materialized.
        dist.all_gather_into_tensor(
            workspace.collective.view(
                plan.contributor_count * slots.capacity, *payload_shape
            ),
            workspace.masked_input,
            group=group,
        )
        logical_sources = workspace.collective.index_select(
            0, workspace.logical_to_group
        )
        folded = _balanced_adjacent_tree(logical_sources)
        torch.where(
            slots.valid_mask.view(-1, *mask_tail),
            folded,
            workspace.zero,
            out=workspace.result,
        )
        output_slots = slots
        owner_mask = slots.valid_mask

    workspace.status.copy_(invalid_position.any().to(torch.int32))
    return CanonicalMoEOutput(
        workspace.result,
        owner_mask,
        output_slots,
        workspace.status,
    )


def canonicalize_glm52_local_partial_v3b(
    local_partial: torch.Tensor,
    slots: CanonicalRowSlots,
    *,
    plan: SamplerParallelPlan,
    group: dist.ProcessGroup,
    layer_id: int,
    graph_capture: bool = False,
    workspace: CanonicalMoEV3Workspace | None = None,
) -> CanonicalMoEOutput:
    """Controlled v3 decode baseline with the fused exact-BF16 folding.

    This arm deliberately keeps v3's contributor all-gather while replacing
    the 15 separate tensor adds with one fused kernel evaluating the identical
    balanced adjacent BF16 tree (identity-order fast path included). Invalid
    rows are already zero after the input mask, so the folded tensor is
    returned directly instead of launching v3's redundant post-fold mask/copy.
    The original v3 remains unchanged as the independent performance control.
    """
    if local_partial.dtype is not torch.bfloat16:
        raise TypeError("GLM-5.2 local MoE contribution must be BF16")
    if local_partial.shape[0] != slots.capacity:
        raise ValueError("Local partial rows must equal the v3b source capacity")
    if workspace is None:
        workspace = CanonicalMoEV3Workspace.allocate(
            local_partial,
            plan=plan,
            group=group,
            distribution=CanonicalDistribution.REPLICATED_CANONICAL,
        )
    workspace.validate(
        local_partial,
        plan,
        CanonicalDistribution.REPLICATED_CANONICAL,
    )

    payload_shape = local_partial.shape[1:]
    mask_tail = (1,) * len(payload_shape)
    torch.where(
        slots.valid_mask.view(-1, *mask_tail),
        local_partial,
        workspace.zero,
        out=workspace.masked_input,
    )
    dist.all_gather_into_tensor(
        workspace.collective.view(
            plan.contributor_count * slots.capacity, *payload_shape
        ),
        workspace.masked_input,
        group=group,
    )

    identity_order = plan.physical_to_logical == tuple(range(plan.contributor_count))
    folded = canonical_moe_fold_v1(
        workspace.collective,
        logical_to_physical=(None if identity_order else workspace.logical_to_group),
        output=workspace.result,
    )

    invalid_position = slots.valid_mask & (slots.absolute_positions < 0)
    workspace.status.copy_(invalid_position.any().to(torch.int32))
    return CanonicalMoEOutput(
        folded,
        slots.valid_mask,
        slots,
        workspace.status,
    )


__all__ = [
    "CANONICAL_MOE_FOLD_VERSION",
    "GLM52_CANONICAL_MOE_V3B_VERSION",
    "GLM52_CANONICAL_MOE_V3_VERSION",
    "GLM52_CANONICAL_MOE_VERSION",
    "GLM52_SAMPLER_LOCAL_POLICY",
    "CanonicalDistribution",
    "CanonicalTransport",
    "CanonicalDeferredStatusBook",
    "CanonicalMoEOutput",
    "CanonicalRowSlots",
    "CanonicalMoEWorkspace",
    "CanonicalMoEV3Workspace",
    "SamplerParallelPlan",
    "canonical_moe_fold_v1",
    "canonical_moe_reference",
    "canonicalize_glm52_local_partial",
    "canonicalize_glm52_local_partial_v3",
    "canonicalize_glm52_local_partial_v3b",
]
