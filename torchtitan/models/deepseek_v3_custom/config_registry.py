# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
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


def deepseek_v3_custom_3b() -> Trainer.Config:
    """
    Trainer configuration for deepseek_v3_custom 3B.

    Training hyperparameters are aligned with the Megatron training script
    ``train_DeepSeek_3b.sh`` and the YAML ``DeepSeek-3B.yaml``.

    Known mismatches vs. Megatron:
      - LR decay: Megatron uses multi-milestone step decay (ratios 0.8/0.9,
        coeffs 0.316/0.1). torchtitan only supports single-phase
        linear/sqrt/cosine decay. We use cosine with decay_ratio=0.8 as a
        rough approximation of the first milestone.
      - EP/TP/CP/PP: deepseek_v3_custom does not support any of these;
        only FSDP/HSDP/DDP.
    """
    return Trainer.Config(
        loss=ChunkedCELoss.Config(),
        # Official DeepSeek-V3 tokenizer (downloaded from HF Hub).
        hf_assets_path="/home/public/liuyichuan/models/hf/DeepSeek-V3-Base",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("3B", attn_backend="sdpa"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="fineweb_edu_50k"),
        # Aligned with Megatron --lr 9e-4
        optimizer=OptimizersContainer.Config(lr=9e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            # Aligned with Megatron --lr-warmup-iters 2000
            warmup_steps=2000,
            # Megatron uses step decay with two milestones (ratio 0.8 coeff 0.316,
            # ratio 0.9 coeff 0.1). torchtitan does not support multi-milestone
            # step decay; decay_ratio=0.8 approximates the first milestone.
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            # Aligned with Megatron --micro-batch-size 8
            local_batch_size=8,
            # Aligned with Megatron --seq-length 4096
            seq_len=4096,
            # Aligned with Megatron --train-iters 95368 (1T tokens)
            steps=95368,
        ),
        parallelism=ParallelismConfig(
            # Megatron script sets EP=8, but deepseek_v3_custom does not support EP.
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            # Aligned with Megatron --save-interval 1000
            interval=1000,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )
