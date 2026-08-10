# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""PyTorch autograd adapter for the BF16 MoK functional API."""

import torch
import torch.distributed as dist

from . import functional


class _BF16AutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        config: functional.MoKConfig,
        workspace: functional.MoKWorkspace,
        num_local_experts: int,
        top_experts: torch.Tensor,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        shared_gate_weights: torch.Tensor,
        shared_up_weights: torch.Tensor,
        shared_down_weights: torch.Tensor,
        routed_gate_weights: torch.Tensor,
        routed_up_weights: torch.Tensor,
        routed_down_weights: torch.Tensor,
    ) -> torch.Tensor:
        schedule = functional.build_schedule(
            workspace,
            config,
            top_experts,
            num_local_experts=num_local_experts,
        )
        output, forward_context = functional.forward(
            config,
            workspace,
            schedule,
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        assert isinstance(forward_context.x_routed, torch.Tensor)
        assert isinstance(forward_context.gate_routed, torch.Tensor)
        assert isinstance(forward_context.up_routed, torch.Tensor)
        assert isinstance(forward_context.hidden_routed, torch.Tensor)
        ctx.config = config
        ctx.workspace = workspace
        ctx.schedule = schedule
        ctx.save_for_backward(
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
            forward_context.x_routed,
            forward_context.gate_shared,
            forward_context.gate_routed,
            forward_context.up_shared,
            forward_context.up_routed,
            forward_context.hidden_shared,
            forward_context.hidden_routed,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
            x_routed,
            gate_shared,
            gate_routed,
            up_shared,
            up_routed,
            hidden_shared,
            hidden_routed,
        ) = ctx.saved_tensors
        forward_context = functional.MoKForwardContext(
            x_routed=x_routed,
            gate_shared=gate_shared,
            gate_routed=gate_routed,
            up_shared=up_shared,
            up_routed=up_routed,
            hidden_shared=hidden_shared,
            hidden_routed=hidden_routed,
        )
        (
            d_x,
            d_router_weights,
            d_routed_gate_weights,
            d_routed_up_weights,
            d_routed_down_weights,
            d_shared_gate_weights,
            d_shared_up_weights,
            d_shared_down_weights,
        ) = functional.backward(
            ctx.config,
            ctx.workspace,
            ctx.schedule,
            forward_context,
            grad_output.contiguous(),
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        return (
            None,
            None,
            None,
            None,
            d_x,
            d_router_weights,
            d_shared_gate_weights,
            d_shared_up_weights,
            d_shared_down_weights,
            d_routed_gate_weights,
            d_routed_up_weights,
            d_routed_down_weights,
        )


def bf16(
    config: functional.MoKConfig,
    group: dist.ProcessGroup,
    x: torch.Tensor,
    top_experts: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor,
    routed_up_weights: torch.Tensor,
    routed_down_weights: torch.Tensor,
) -> torch.Tensor:
    """Run BF16 MoK with gradients supplied by its explicit functional backward.

    The workspace is cached by MoK for calls with the same process group, device,
    token count, hidden size, top-k, and schedule capacity. Each invocation still
    owns its schedule and saved forward activations, so ordinary PyTorch autograd
    can execute the corresponding backward later. Calls sharing a workspace must
    execute serially on one CUDA stream. Each forward supports one first-order
    backward; concurrent, re-entrant, retained-graph replay, and higher-order
    autograd are not supported.
    """
    workspace = functional.get_workspace(
        config,
        group,
        device=x.device,
        num_local_tokens=x.shape[0],
        hidden_size=x.shape[1],
        topk=top_experts.shape[1],
    )
    return _BF16AutogradFunction.apply(
        config,
        workspace,
        routed_gate_weights.shape[0],
        top_experts.contiguous(),
        x.contiguous(),
        router_weights.float().contiguous(),
        shared_gate_weights.contiguous(),
        shared_up_weights.contiguous(),
        shared_down_weights.contiguous(),
        routed_gate_weights.contiguous(),
        routed_up_weights.contiguous(),
        routed_down_weights.contiguous(),
    )


__all__ = ["bf16"]
