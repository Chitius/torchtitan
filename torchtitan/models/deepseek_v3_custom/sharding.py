# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torch.distributed.tensor import Placement, Shard

from torchtitan.models.common.moe_sharding import set_moe_sharding_config

if TYPE_CHECKING:
    from .model import DeepSeekV3CustomModel


# Routed-expert layout for GroupedExperts (w1/w2/w3) — same as deepseek_v3.
_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, Placement] = {
    "w1": Shard(1),
    "w2": Shard(2),
    "w3": Shard(1),
}


def set_deepseek_v3_custom_sharding_config(
    config: "DeepSeekV3CustomModel.Config",
    *,
    enable_ep: bool,
) -> None:
    """Populate sharding_config on MoE sub-configs for EP.

    Since deepseek_v3_custom does not use TP/SP, only MoE submodules
    receive sharding configs. Dense parts (attention, norms, dense FFN)
    remain unsharded — FSDP handles them via fully_shard.
    """
    from .model import MTPModule

    for layer_cfg in config.layers:
        moe_cfg = None
        if isinstance(layer_cfg, MTPModule.Config):
            moe_cfg = layer_cfg.transformer_block.moe
        elif hasattr(layer_cfg, "moe"):
            moe_cfg = layer_cfg.moe

        if moe_cfg is not None:
            set_moe_sharding_config(
                moe_cfg,
                enable_ep=enable_ep,
                enable_sp=False,
                expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
            )
