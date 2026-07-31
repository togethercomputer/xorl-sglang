"""Canonical BF16 MoE contribution folding for the GLM-5.2 serving lane."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist


GLM52_CANONICAL_MOE_VERSION = "canonical_moe_reduce_v1"
GLM52_SAMPLER_LOCAL_POLICY = "glm52_routed_final_scaled_then_shared_ep_slice_bf16_v2"


class CanonicalDistribution(str, Enum):
    OWNER_SHARDED = "owner_sharded"
    REPLICATED_CANONICAL = "replicated_canonical"


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
    production: bool = False
    version: str = GLM52_CANONICAL_MOE_VERSION

    def __post_init__(self) -> None:
        if self.contributor_count not in (2, 4, 8):
            raise ValueError("Canonical MoE admits exactly 2, 4, or 8 contributors")
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
            if self.contributor_count != 8:
                raise ValueError(
                    "Production GLM-5.2 requires exactly eight contributors"
                )
            if self.physical_to_logical != tuple(range(8)):
                raise ValueError(
                    "Production GLM-5.2 requires identity logical contributor ordinals"
                )
            if (
                self.ep_size != 8
                or self.attention_cp_size != 8
                or self.launcher_tp_size != 8
            ):
                raise ValueError(
                    "Production GLM-5.2 sampler requires launcher TP8, attention CP8, and EP8"
                )

    @classmethod
    def glm52(
        cls,
        *,
        pp_size: int = 1,
        pp_rank: int = 0,
        physical_ranks: tuple[int, ...] | None = None,
    ) -> SamplerParallelPlan:
        ranks = (
            tuple(range(pp_rank * 8, (pp_rank + 1) * 8))
            if physical_ranks is None
            else tuple(physical_ranks)
        )
        return cls(
            8,
            ranks,
            tuple(range(8)),
            launcher_tp_size=8,
            global_world_size=8 * pp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
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
    ) -> None:
        actual = (
            dist.get_world_size(),
            launcher_tp_size,
            effective_dense_tp,
            pp_size,
            ep_size,
            attention_cp_size,
            dist.get_world_size(group),
        )
        expected = (
            self.global_world_size,
            self.launcher_tp_size,
            self.effective_dense_tp,
            self.pp_size,
            self.ep_size,
            self.attention_cp_size,
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
class CanonicalMoEReceipt:
    version: str
    plan_hash: str
    layer_id: int
    distribution: CanonicalDistribution
    capacity: int
    graph_capture: bool
    local_policy: str


@dataclass(frozen=True)
class CanonicalMoEOutput:
    values: torch.Tensor
    owner_mask: torch.Tensor
    slots: CanonicalRowSlots
    receipt: CanonicalMoEReceipt
    contract_status: torch.Tensor

    def raise_for_status(self) -> None:
        status = int(self.contract_status.item())
        if status != 0:
            raise RuntimeError(
                f"Canonical MoE device contract failed with status {status}"
            )


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


def _balanced_adjacent_tree(partials: torch.Tensor) -> torch.Tensor:
    if partials.dtype is not torch.bfloat16:
        raise TypeError("Canonical MoE arithmetic requires BF16 partials")
    if partials.shape[0] not in (2, 4, 8):
        raise ValueError("Canonical MoE arithmetic requires 2, 4, or 8 contributors")
    level = tuple(partials.unbind(0))
    while len(level) > 1:
        level = tuple(
            (level[index] + level[index + 1]).to(torch.bfloat16)
            for index in range(0, len(level), 2)
        )
    return level[0]


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

    receipt = CanonicalMoEReceipt(
        version=GLM52_CANONICAL_MOE_VERSION,
        plan_hash=plan.identity,
        layer_id=layer_id,
        distribution=distribution,
        capacity=slots.capacity,
        graph_capture=graph_capture,
        local_policy=GLM52_SAMPLER_LOCAL_POLICY,
    )
    return CanonicalMoEOutput(
        workspace.result, owner_mask, slots, receipt, workspace.status
    )


__all__ = [
    "GLM52_CANONICAL_MOE_VERSION",
    "GLM52_SAMPLER_LOCAL_POLICY",
    "CanonicalDistribution",
    "CanonicalMoEOutput",
    "CanonicalMoEReceipt",
    "CanonicalRowSlots",
    "CanonicalMoEWorkspace",
    "SamplerParallelPlan",
    "canonical_moe_reference",
    "canonicalize_glm52_local_partial",
]
