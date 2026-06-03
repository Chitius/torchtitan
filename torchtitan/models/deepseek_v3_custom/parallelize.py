# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.tools.logging import logger

from .model import DeepSeekV3CustomModel


def parallelize_deepseekv3_custom(
    model: DeepSeekV3CustomModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply activation checkpointing, torch.compile, data parallelism (FSDP),
    and optionally expert parallelism to the model.

    deepseek_v3_custom supports FSDP/HSDP/DDP + optional EP.
    TP, CP, and PP are not supported.
    """
    if parallel_dims.tp > 1:
        raise ValueError(
            f"deepseek_v3_custom does not support tensor parallelism (TP). "
            f"Got TP degree={parallel_dims.tp}."
        )
    if parallel_dims.cp > 1:
        raise ValueError(
            f"deepseek_v3_custom does not support context parallelism (CP). "
            f"Got CP degree={parallel_dims.cp}."
        )
    if parallel_dims.pp > 1:
        raise ValueError(
            f"deepseek_v3_custom does not support pipeline parallelism (PP). "
            f"Got PP degree={parallel_dims.pp}."
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    # Distribute MoE expert weights as DTensors on the sparse mesh when EP
    # is enabled. This must happen before FSDP wrapping.
    if parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
    )

    logger.info("Applied fully_shard to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model
