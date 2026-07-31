# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_ordered_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce through a fixed reverse-rank BF16 addition chain.

    Raw all-gather communication performs no floating-point arithmetic. Every
    rank then executes the same explicit ``world_size - 1 -> 0`` addition
    sequence, so the result is independent of the NCCL all-reduce tree.
    """
    group = get_tp_group()
    if group.world_size == 1 or input_.numel() == 0:
        return input_

    input_ = input_.contiguous()
    gathered = torch.empty(
        (group.world_size * input_.shape[0], *input_.shape[1:]),
        dtype=input_.dtype,
        device=input_.device,
    )
    torch.distributed.all_gather_into_tensor(gathered, input_, group=group.device_group)
    partials = gathered.view(group.world_size, *input_.shape)
    result = partials[-1]
    for rank in range(group.world_size - 2, -1, -1):
        result = result + partials[rank]
    return result


def tensor_model_parallel_fused_allreduce_rmsnorm(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused TP all-reduce + RMSNorm.

    Policy and backend selection are owned by GroupCoordinator:
    it may dispatch to communicator-native fused APIs, custom fused kernels,
    or return None so callers can run generic fallback paths.
    """
    return get_tp_group().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
