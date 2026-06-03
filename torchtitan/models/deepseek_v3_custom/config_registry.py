# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry
from .loss import MTPLoss
from torchtitan.components.loss import ChunkedCELoss

_NUM_MTP = 1
_MTP_WEIGHT = 0.3


def _make_loss():
    """When MTP is disabled, use ChunkedCELoss for lower peak memory."""
    if _NUM_MTP > 0:
        return MTPLoss.Config(num_mtp_modules=_NUM_MTP, mtp_loss_weight=_MTP_WEIGHT)
    return ChunkedCELoss.Config()


def deepseek_v3_custom_500m() -> Trainer.Config:
    """500M config with MTP=1, bfloat16, FSDP2 DP=2."""
    base_seq_len = 2048
    return Trainer.Config(
        loss=_make_loss(),
        hf_assets_path="/home/public/liuyichuan/models/custom/deepseek_v3_500m",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("500M", attn_backend="flex", num_mtp_modules=_NUM_MTP),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="fineweb_edu_50k"),
        optimizer=OptimizersContainer.Config(
            lr=8.6e-4,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_type="cosine",
            min_lr_factor=0.00814,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            global_batch_size=16,
            seq_len=base_seq_len + _NUM_MTP,
            steps=100,
            max_norm=1.0,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=2,
            data_parallel_shard_degree=1,
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=100,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
    )


def deepseek_v3_custom_500m_ep2() -> Trainer.Config:
    """500M config with EP=2 + MTP=1 + seq_aux_loss, bfloat16.
    Let dp_shard be auto-computed to account for EP splitting."""
    base_seq_len = 2048
    return Trainer.Config(
        loss=_make_loss(),
        hf_assets_path="/home/public/liuyichuan/models/custom/deepseek_v3_500m",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry(
            "500M", attn_backend="flex",
            num_mtp_modules=_NUM_MTP, seq_aux_loss_coeff=1e-4,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="fineweb_edu_50k"),
        optimizer=OptimizersContainer.Config(
            lr=8.6e-4,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_type="cosine",
            min_lr_factor=0.00814,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            global_batch_size=16,
            seq_len=base_seq_len + _NUM_MTP,
            steps=200,
            max_norm=1.0,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            expert_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
        ),
    )


def deepseek_v3_custom_3b() -> Trainer.Config:
    """3B config with MTP=1, bfloat16."""
    base_seq_len = 4096
    return Trainer.Config(
        loss=_make_loss(),
        hf_assets_path="/home/public/liuyichuan/models/custom/deepseek_v3_3b",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("3B", attn_backend="sdpa", num_mtp_modules=_NUM_MTP),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="fineweb_edu_50k"),
        optimizer=OptimizersContainer.Config(lr=9e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=base_seq_len + _NUM_MTP,
            steps=95368,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )
