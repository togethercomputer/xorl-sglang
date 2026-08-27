"""Overlay twin: eager NCCL communicator for RL weight-update groups.

Ported from xorl-sglang `main` (c08786bd3). XoRL eagerly initializes the
trainer-side NCCL communicator (it passes ``device_id`` to
``init_process_group``). The receiving side must do the same so every
inference rank joins the rendezvous before this endpoint reports
initialization success -- without ``device_id``, ProcessGroupNCCL is lazy
and an eager trainer blocks forever waiting for communicators the inference
ranks have not created yet. (Observed live: /init_weights_update_group
returns 200 while the xorl trainer hangs in init_process_group.)

Whole-method replacement of ``WeightUpdater.init_weights_update_group``;
main's two-phase prepare/complete receiver half is not ported.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def __apply_patch__(mod):
    WeightUpdater = mod.WeightUpdater
    NetworkAddress = mod.NetworkAddress
    init_custom_process_group = mod.init_custom_process_group

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates."""
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, "
            f"master_port={master_port}, rank_offset={rank_offset}, rank={rank}, "
            f"world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            # Eager communicator init; see module docstring.
            device_id = (
                torch.device("cuda", torch.cuda.current_device())
                if backend == "nccl"
                else None
            )
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
                device_id=device_id,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    WeightUpdater.init_weights_update_group = init_weights_update_group
