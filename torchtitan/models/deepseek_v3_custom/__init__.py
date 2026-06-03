# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial
from typing import Literal

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE, TransformerBlock
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_experts_config,
    make_ffn_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import (
    Attention,
    DeepSeekV3CustomModel,
    DeepSeekV3CustomTransformerBlock,
    MTPModule,
)
from .parallelize import parallelize_deepseekv3_custom
from .state_dict_adapter import DeepSeekV3CustomStateDictAdapter

__all__ = [
    "parallelize_deepseekv3_custom",
    "DeepSeekV3CustomModel",
    "deepseekv3_custom_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.006),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.006, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1": partial(nn.init.trunc_normal_, std=0.006),
        "w2": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.006, layer_id)),
        "w3": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.006, layer_id)),
    }


def _make_dsv3_attn_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float = 1.0,
    attn_backend: str,
) -> Attention.Config:
    """Build a fully-specified DeepSeek V3 MLA Attention.Config."""
    inner_attention, mask_type = get_attention_config(attn_backend)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    if q_lora_rank == 0:
        wq = Linear.Config(
            in_features=dim,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        wq_a = None
        wq_b = None
        q_norm = RMSNorm.Config(normalized_shape=1, param_init=_NORM_INIT)
    else:
        wq = None
        wq_a = Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            param_init=_LINEAR_INIT,
        )
        wq_b = Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        q_norm = RMSNorm.Config(normalized_shape=q_lora_rank, param_init=_NORM_INIT)

    return Attention.Config(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=mscale,
        wq=wq,
        wq_a=wq_a,
        wq_b=wq_b,
        q_norm=q_norm,
        wkv_a=Linear.Config(
            in_features=dim,
            out_features=kv_lora_rank + qk_rope_head_dim,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(normalized_shape=kv_lora_rank, param_init=_NORM_INIT),
        wkv_b=Linear.Config(
            in_features=kv_lora_rank,
            out_features=n_heads * (qk_nope_head_dim + v_head_dim),
            param_init=_LINEAR_INIT,
        ),
        wo=Linear.Config(
            in_features=n_heads * v_head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        inner_attention=inner_attention,
        mask_type=mask_type,
    )


def _build_dsv3_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_score_func: Literal["sigmoid", "softmax"],
    router_num_expert_groups: int | None = None,
    router_num_limited_groups: int | None = None,
    router_route_scale: float = 1.0,
    router_route_norm: bool = False,
    score_before_experts: bool = False,
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    seq_aux_loss_coeff: float | None = None,
    num_mtp_modules: int = 0,
) -> list[TransformerBlock.Config]:
    """Build the list of per-layer configs (main transformer blocks + MTP modules).

    Layers with layer_id < n_dense_layers get a dense FeedForward and no MoE.
    Layers with layer_id >= n_dense_layers get a MoE and no FeedForward.

    When num_mtp_modules > 0, additional MTPModule configs are appended.
    MTP layers reuse the architecture of the last MoE main layer.
    """
    layers = []
    for layer_id in range(n_layers):
        attn_cfg = _make_dsv3_attn_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mscale=mscale,
            attn_backend=attn_backend,
        )

        if layer_id < n_dense_layers:
            ffn_cfg = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe_cfg = None
        else:
            ffn_cfg = None
            moe_cfg = make_moe_config(
                num_experts=num_experts,
                router=make_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    gate_param_init=_depth_init(layer_id),
                    top_k=router_top_k,
                    score_func=router_score_func,
                    num_expert_groups=router_num_expert_groups,
                    num_limited_groups=router_num_limited_groups,
                    route_scale=router_route_scale,
                    route_norm=router_route_norm,
                ),
                experts=make_experts_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                    top_k=router_top_k,
                    param_init=_depth_experts_init(layer_id),
                    score_before_experts=score_before_experts,
                    comm_backend=moe_comm_backend,
                    non_blocking_capacity_factor=non_blocking_capacity_factor,
                ),
                shared_experts=make_ffn_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
                load_balance_coeff=None if seq_aux_loss_coeff is not None else 1e-3,
                seq_aux_loss_coeff=seq_aux_loss_coeff,
            )

        layers.append(
            DeepSeekV3CustomTransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                feed_forward=ffn_cfg,
                moe=moe_cfg,
            )
        )

    # Build MTP module configs, reusing the last MoE layer's architecture.
    for mtp_id in range(num_mtp_modules):
        mtp_layer_id = n_layers + mtp_id
        mtp_attn_cfg = _make_dsv3_attn_config(
            layer_id=mtp_layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mscale=mscale,
            attn_backend=attn_backend,
        )
        mtp_moe_cfg = make_moe_config(
            num_experts=num_experts,
            router=make_router_config(
                dim=dim,
                num_experts=num_experts,
                gate_param_init=_depth_init(mtp_layer_id),
                top_k=router_top_k,
                score_func=router_score_func,
                num_expert_groups=router_num_expert_groups,
                num_limited_groups=router_num_limited_groups,
                route_scale=router_route_scale,
                route_norm=router_route_norm,
            ),
            experts=make_experts_config(
                dim=dim,
                hidden_dim=moe_hidden_dim,
                num_experts=num_experts,
                top_k=router_top_k,
                param_init=_depth_experts_init(mtp_layer_id),
                score_before_experts=score_before_experts,
                comm_backend=moe_comm_backend,
                non_blocking_capacity_factor=non_blocking_capacity_factor,
            ),
            shared_experts=make_ffn_config(
                dim=dim,
                hidden_dim=moe_hidden_dim * num_shared_experts,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(mtp_layer_id),
            ),
            load_balance_coeff=None if seq_aux_loss_coeff is not None else 1e-3,
            seq_aux_loss_coeff=seq_aux_loss_coeff,
        )
        mtp_transformer_block = DeepSeekV3CustomTransformerBlock.Config(
            attention=mtp_attn_cfg,
            attention_norm=RMSNorm.Config(
                normalized_shape=dim, param_init=_NORM_INIT
            ),
            ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
            feed_forward=None,
            moe=mtp_moe_cfg,
        )
        layers.append(
            MTPModule.Config(
                transformer_block=mtp_transformer_block,
                dim=dim,
            )
        )

    return layers


def _500m(
    attn_backend: str,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    num_mtp_modules: int = 0,
    seq_aux_loss_coeff: float | None = None,
) -> DeepSeekV3CustomModel.Config:
    dim = 768
    n_layers = 10
    vocab_size = 129280
    n_heads = 12
    moe_hidden_dim = 512
    num_shared_experts = 1
    dense_hidden_dim = 4096
    rope_dim = 64
    num_experts = 24
    n_dense_layers = 2

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=768,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=rope_dim,
        v_head_dim=64,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=4,
        router_score_func="sigmoid",
        router_num_expert_groups=4,
        router_num_limited_groups=2,
        router_route_scale=2.5,
        router_route_norm=True,
        score_before_experts=False,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        num_mtp_modules=num_mtp_modules,
        seq_aux_loss_coeff=seq_aux_loss_coeff,
    )
    return DeepSeekV3CustomModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        num_mtp_modules=num_mtp_modules,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=8192,
            theta=10000.0,
            backend="complex",
            scaling="none",
        ),
        layers=layers,
    )


def _3b(
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    num_mtp_modules: int = 0,
    seq_aux_loss_coeff: float | None = None,
) -> DeepSeekV3CustomModel.Config:
    """DeepSeek-V3 3B preset"""
    dim = 1280
    n_layers = 12
    vocab_size = 129280
    n_heads = 16
    moe_hidden_dim = 896
    num_shared_experts = 2
    dense_hidden_dim = 7168
    rope_dim = 64
    num_experts = 64
    n_dense_layers = 1

    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="sigmoid",
        router_route_scale=2.5,
        router_route_norm=True,
        score_before_experts=False,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        num_mtp_modules=num_mtp_modules,
        seq_aux_loss_coeff=seq_aux_loss_coeff,
    )
    return DeepSeekV3CustomModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        num_mtp_modules=num_mtp_modules,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=1.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


deepseekv3_custom_configs = {
    "500M": _500m,
    "3B": _3b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
    num_mtp_modules: int = 0,
    seq_aux_loss_coeff: float | None = None,
) -> ModelSpec:
    config = deepseekv3_custom_configs[flavor](
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        num_mtp_modules=num_mtp_modules,
        seq_aux_loss_coeff=seq_aux_loss_coeff,
    )
    if converters is not None:
        validate_converter_order(converters)
        for c in converters:
            c.build().convert(config)
    return ModelSpec(
        name="deepseek_v3_custom",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseekv3_custom,
        pipelining_fn=None,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3CustomStateDictAdapter,
    )
