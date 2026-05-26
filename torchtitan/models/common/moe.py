# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

from .token_dispatcher import DeepEPTokenDispatcher, LocalTokenDispatcher


class GroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        token_dispatcher: LocalTokenDispatcher.Config

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.token_dispatcher = config.token_dispatcher.build()

    def _experts_forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine."""
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            # pyrefly: ignore [missing-attribute]
            w2 = self.w2.to_local()
            # pyrefly: ignore [missing-attribute]
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        h = F.silu(
            torch._grouped_mm(
                x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets
            )
        )
        h = h * torch._grouped_mm(
            x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
        )
        return torch._grouped_mm(
            h, w2.bfloat16().transpose(-2, -1), offs=offsets
        ).type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        When parallelized, ``local_map`` (from ``sharding_config``) handles
        DTensor→local conversion on entry and local→DTensor(Partial) wrapping
        on exit. The forward body operates on plain local tensors.
        """
        routed_input, num_tokens_local, metadata = self.token_dispatcher.dispatch(
            x, top_scores, selected_experts_indices
        )
        routed_output = self._experts_forward(routed_input, num_tokens_local)
        return self.token_dispatcher.combine(routed_output, metadata, x)

    def parallelize(self, parallel_dims) -> None:
        """Parallelize expert weights, then wire EP/TP meshes on the dispatcher
        so dispatch/combine see the right meshes at runtime."""
        super().parallelize(parallel_dims)
        # TODO(@pianpwk): With spmd_types and set_current_mesh, replace wire_meshes
        # with current_mesh calls inside AllToAllTokenDispatcher and
        # DeepEPTokenDispatcher.
        self.token_dispatcher.wire_meshes(
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
        )


class TokenChoiceTopKRouter(Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Optionally supports node-limited (group-limited) routing where experts are divided into groups
    (e.g., by node), and only num_limited_groups groups are considered before selecting top_k experts.
    This reduces cross-node communication in distributed settings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        gate: Linear.Config
        num_expert_groups: int | None = None  # must be a divisor of num_experts
        num_limited_groups: int | None = None
        top_k: int = 1
        score_func: Literal["softmax", "sigmoid"] = "sigmoid"
        route_norm: bool = False
        route_scale: float = 1.0
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices [N, K] LongTensor, top_scores [N, K] FloatTensor).
        """
        n_tokens = scores.size(0)
        # Round-robin indices with exact balance
        selected_experts_indices = (
            torch.arange(
                n_tokens * self.top_k, device=scores.device, dtype=torch.int64
            ).reshape(n_tokens, self.top_k)
            % self.num_experts
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)  # [N,K]
        return selected_experts_indices, top_scores

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
            and set expert scores in non-selected groups as -inf

        Args:
            scores_for_choice: Router scores with expert_bias (if any), shape (bs*slen, num_experts)

        Returns:
            scores_for_choice: shape (bs*slen, num_experts)
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = scores_for_choice.view(
            -1, self.num_expert_groups, experts_per_group
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).view(-1, self.num_experts)

        return scores_for_choice

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
                - scores (torch.Tensor):
                    Routing scores for all experts with shape ``(bs*slen, num_experts)``.
        """
        # scores shape (bs*slen, num_experts)
        # Compute gate in float32 to help stability of expert load balancing.
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        # scored is already float32 from the autocast above.
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        # Apply node-limited routing if configured
        if self.num_expert_groups is not None:
            scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
        _, selected_experts_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        top_scores = scores.gather(dim=1, index=selected_experts_indices)

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        flat_indices = selected_experts_indices.view(-1)
        num_tokens_per_expert = torch.bincount(
            flat_indices, minlength=self.num_experts
        ).float()

        return top_scores, selected_experts_indices, num_tokens_per_expert, scores


class MoE(Module):
    """Mixture of Experts layer.

    The forward pass proceeds as:
    1. Router computes expert assignments (stays on DTensor)
    2. GroupedExperts.forward() converts DTensor to local, then handles:
       a. dispatch (TokenDispatcher) — reorder tokens by expert assignment.
          With EP, also performs all-to-all communication to send tokens
          to expert-owning ranks.
       b. expert computation (local tensors)
       c. combine (TokenDispatcher) — reverse the dispatch reordering.
          - LocalTokenDispatcher (no EP): scatter_add only.
          - AllToAll: all-to-all communication, then scatter_add.
          - DeepEP: async combine_tokens (sync deferred to step 4 when
            sp_size == 1; forced inside combine when sp_size > 1).
          - HybridEP: synchronous combine_tokens.
    3. Shared experts run on DTensor. Overlaps with DeepEP async combine
       when sp_size == 1; no overlap otherwise.
    4. Routed and shared expert outputs are summed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        experts: GroupedExperts.Config
        router: TokenChoiceTopKRouter.Config
        load_balance_coeff: float | None = 1e-3
        aux_loss_coeff: float | None = None
        seq_aux_loss_coeff: float | None = None
        shared_experts: FeedForward.Config | None = None

    def __init__(self, config: Config):
        super().__init__()

        num_experts = config.num_experts
        self.experts = config.experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = config.load_balance_coeff
        self.aux_loss_coeff = config.aux_loss_coeff
        self.seq_aux_loss_coeff = config.seq_aux_loss_coeff

        # Enforce mutual exclusivity: loss-based load balancing takes priority.
        if self.aux_loss_coeff is not None or self.seq_aux_loss_coeff is not None:
            if self.load_balance_coeff is not None:
                import warnings
                warnings.warn(
                    "MoE auxiliary loss (aux_loss_coeff / seq_aux_loss_coeff) is enabled. "
                    "Disabling auxiliary-loss-free load balancing (load_balance_coeff) to avoid conflicting signals.",
                    stacklevel=2,
                )
                self.load_balance_coeff = None

        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        # _aux_losses stores per-forward auxiliary losses for load balancing.
        # Using a Python list avoids torch.compile issues with buffer mutations
        # under activation checkpointing.
        self._aux_losses: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.

        Under TP, the MoE wrapper's ``sharding_config`` (set by
        ``set_moe_sharding_config``) handles input/output redistribution:
        input is redistributed from sp_layout to desired_input_layouts;
        output (Partial) is redistributed to sp_layout. MoE.forward()
        operates on DTensors — the DTensor→local conversion happens at
        the GroupedExperts boundary.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # top_scores and selected_experts_indices shape (bs*slen, top_k)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            scores,
        ) = self.router(x, self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        # Compute auxiliary loss for load balancing if configured.
        # Reference: Megatron-LM switch_load_balancing_loss_func
        if self.aux_loss_coeff is not None or self.seq_aux_loss_coeff is not None:
            self._compute_and_store_aux_loss(
                scores,
                selected_experts_indices,
                num_tokens_per_expert,
                bs,
                slen,
            )

        out = self.experts(x, top_scores, selected_experts_indices)

        # shared_experts runs in parallel with deepep combine communication.
        shared_out = self.shared_experts(x) if self.shared_experts is not None else None

        if (
            isinstance(self.experts.token_dispatcher, DeepEPTokenDispatcher)
            and self.experts.token_dispatcher.sp_size == 1
        ):
            # Sync the combine operation before using routed_output.
            # This inserts a CUDA stream wait, ensuring combine is complete before
            # the subsequent addition or reshape operations read routed output.
            from torchtitan.distributed.deepep.deepep import sync_combine

            sync_combine()

        if shared_out is not None:
            out = out + shared_out
        return out.reshape(bs, slen, dim)

    def _compute_and_store_aux_loss(
        self,
        scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        bs: int,
        slen: int,
    ) -> None:
        """Compute micro-batch and/or sequence-level auxiliary losses.

        Megatron's compute_routing_scores_for_aux_loss uses the original
        scores without expert_bias. We mirror that here.
        """
        if self.expert_bias is not None:
            _, selected_experts_indices_aux = torch.topk(
                scores, k=self.router.top_k, dim=-1, sorted=False
            )
            flat_indices_aux = selected_experts_indices_aux.view(-1)
            num_tokens_per_expert_aux = torch.bincount(
                flat_indices_aux, minlength=self.router.num_experts
            ).float()
        else:
            selected_experts_indices_aux = selected_experts_indices
            num_tokens_per_expert_aux = num_tokens_per_expert

        total_num_tokens = bs * slen
        num_experts = self.router.num_experts
        topk = self.router.top_k

        if self.aux_loss_coeff is not None:
            # micro-batch level aux loss
            aggregated_probs = scores.sum(dim=0)  # (num_experts,)
            aux_loss = (
                torch.sum(aggregated_probs * num_tokens_per_expert_aux)
                * num_experts
                * self.aux_loss_coeff
                / (topk * total_num_tokens * total_num_tokens)
            )
            self._aux_losses.append(aux_loss)

        if self.seq_aux_loss_coeff is not None:
            # sequence-level aux loss (DeepSeek-V2/V3 style)
            scores_reshaped = scores.view(bs, slen, num_experts)
            # Build per-sequence tokens-per-expert from selected indices
            routing_map = torch.nn.functional.one_hot(
                selected_experts_indices_aux, num_classes=num_experts
            ).sum(dim=1).to(scores.dtype)
            routing_map = routing_map.view(bs, slen, num_experts)
            tokens_per_expert_per_seq = routing_map.sum(dim=1)  # (bs, num_experts)
            probs_per_seq = scores_reshaped.sum(dim=1)  # (bs, num_experts)
            seq_aux_loss = (
                torch.sum(probs_per_seq * tokens_per_expert_per_seq, dim=1)
                * num_experts
                * self.seq_aux_loss_coeff
                / (topk * slen * slen)
            ).sum() / bs
            self._aux_losses.append(seq_aux_loss)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        assert isinstance(buffer_device, torch.device)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            self._aux_losses.clear()
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )


def collect_moe_aux_loss(model: nn.Module, clear: bool = True) -> torch.Tensor | None:
    """Collect and optionally clear accumulated MOE auxiliary losses from a model.

    Args:
        model: The model to collect aux losses from.
        clear: If True, clear the aux_loss lists after collection.

    Returns:
        Total auxiliary loss tensor, or None if no MoE layers have accumulated aux loss.
    """
    all_aux_losses: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, MoE) and hasattr(module, "_aux_losses"):
            all_aux_losses.extend(module._aux_losses)
    if not all_aux_losses:
        return None
    # Use Python sum to avoid creating an unnecessary torch.stack node.
    total_aux_loss = sum(all_aux_losses)
    if clear:
        for module in model.modules():
            if isinstance(module, MoE) and hasattr(module, "_aux_losses"):
                module._aux_losses.clear()
    return total_aux_loss
