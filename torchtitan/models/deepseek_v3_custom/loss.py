# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MTP-aware cross-entropy loss.

When num_mtp_modules > 0, the model returns a list of logits (one per
prediction). This loss computes CE for each and combines them with a
weighted sum. Per-module losses are tracked for metrics logging.
"""

from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss, cross_entropy_loss


def mtp_cross_entropy_loss(
    preds: list[torch.Tensor],
    labels: torch.Tensor,
    num_mtp_modules: int,
    mtp_loss_weight: float,
) -> torch.Tensor:
    """Multi-token cross-entropy loss.

    Args:
        preds: List of logit tensors. preds[0] is the main prediction,
            preds[1:] are MTP predictions for the 2nd, 3rd, ... future tokens.
        labels: Target token ids, shape (batch, seq_len + num_mtp_modules).
        num_mtp_modules: Number of MTP modules.
        mtp_loss_weight: Weight for the MTP loss component.
    """
    seq_len = preds[0].shape[1]
    main_loss = cross_entropy_loss(preds[0], labels[:, :seq_len])

    mtp_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
    for i, pred in enumerate(preds[1:], 1):
        end_idx = i + seq_len
        loss_i = cross_entropy_loss(pred, labels[:, i:end_idx])
        mtp_loss = mtp_loss + loss_i / num_mtp_modules

    return main_loss + mtp_loss * mtp_loss_weight


class MTPLoss(BaseLoss):
    """Cross-entropy loss with MTP support.

    When num_mtp_modules > 0, expects model output as list[Tensor].
    Otherwise falls back to standard cross-entropy.

    Per-module losses are stored in ``_last_mtp_losses`` for metrics logging.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        num_mtp_modules: int = 0
        mtp_loss_weight: float = 0.3

    def __init__(self, config: Config, *, compile_config=None):
        self.num_mtp_modules = config.num_mtp_modules
        self.mtp_loss_weight = config.mtp_loss_weight
        self.fn = cross_entropy_loss
        self._maybe_compile(compile_config)
        self._last_component_losses: dict[str, float] = {}

    def __call__(self, pred, labels, global_valid_tokens=None):
        if isinstance(pred, list):
            loss = self._compute_mtp_loss(pred, labels)
        else:
            loss = self.fn(pred, labels)
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
        return loss

    def _compute_mtp_loss(self, preds, labels):
        seq_len = preds[0].shape[1]
        main_loss = self.fn(preds[0], labels[:, :seq_len])

        mtp_loss_sum = torch.tensor(
            0.0, device=main_loss.device, dtype=main_loss.dtype
        )
        self._last_component_losses = {}
        for i, pred in enumerate(preds[1:], 1):
            end_idx = i + seq_len
            loss_i = self.fn(pred, labels[:, i:end_idx]) / self.num_mtp_modules
            mtp_loss_sum = mtp_loss_sum + loss_i
            self._last_component_losses[f"mtp_{i}_loss"] = float(
                loss_i.detach().item()
            )
        return main_loss + mtp_loss_sum * self.mtp_loss_weight
