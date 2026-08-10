# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import hashlib
import importlib.metadata
from pathlib import Path

import mok
import torch
import torch.distributed as dist
from mok import autograd, functional

from .utils import BF16_TOLERANCE, check_correctness, generate_inputs, run_reference_bf16

RESULT_NAMES = (
    "output",
    "d_x",
    "d_router_weights",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
)


def _make_autograd_inputs(
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    (
        x,
        top_experts,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        grad_output,
    ) = inputs
    differentiable = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (
            x,
            router_weights,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
    )
    return top_experts, grad_output, differentiable


def _reference_in_autograd_order(
    reference: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return (
        reference[0],
        reference[1],
        reference[2],
        reference[6],
        reference[7],
        reference[8],
        reference[3],
        reference[4],
        reference[5],
    )


def test_public_provenance_matches_adapter_source() -> None:
    source = Path(autograd.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == mok.AUTOGRAD_ADAPTER_SOURCE_SHA256
    assert mok.AUTOGRAD_ADAPTER_API_REVISION == 1
    assert (
        mok.AUTOGRAD_ADAPTER_PROTECTED_BASE == "3e1cf43ab93ad040afed52a45ab03cb490ffe4be"
    )
    assert mok.__version__ == importlib.metadata.version("mixture-of-kittens")
    public_version, separator, local_version = mok.__version__.partition("+")
    assert public_version == "0.1.0"
    assert separator == "+"
    assert local_version == f"graphcore.autograd{mok.AUTOGRAD_ADAPTER_API_REVISION}"
    if mok.AUTOGRAD_ADAPTER_BUILD_INFO is not None:
        assert {
            "build_image_digest",
            "cuda_version",
            "mok_arch",
            "python_abi",
            "source_commit",
            "torch_version",
        } <= mok.AUTOGRAD_ADAPTER_BUILD_INFO.keys()
        assert len(mok.AUTOGRAD_ADAPTER_BUILD_INFO["source_commit"]) == 40


def test_bf16_autograd_matches_manual_functional_and_reference(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    inputs = generate_inputs(rank, device, world_size, 1, 1, 512, 256, 256)
    reference = _reference_in_autograd_order(run_reference_bf16(*inputs))
    (
        x,
        top_experts,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        grad_output,
    ) = inputs
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=512,
        schedule_capacity_multiplier=1.5,
        all_gather_top_experts_chunk_bytes=16,
    )
    workspace = functional.get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=512,
        hidden_size=256,
        topk=1,
    )
    schedule = functional.build_schedule(
        workspace,
        config,
        top_experts,
        num_local_experts=1,
    )
    manual_output, forward_context = functional.forward(
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
    manual_gradients = functional.backward(
        config,
        workspace,
        schedule,
        forward_context,
        grad_output,
        x,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
    )
    manual = _reference_in_autograd_order((manual_output, *manual_gradients))

    top_experts, grad_output, differentiable = _make_autograd_inputs(inputs)
    autograd_output = autograd.bf16(
        config,
        dist.group.WORLD,
        differentiable[0],
        top_experts,
        differentiable[1],
        *differentiable[2:],
    )
    actual = (
        autograd_output,
        *torch.autograd.grad(autograd_output, differentiable, grad_output),
    )

    for name, manual_result, reference_result, actual_result in zip(
        RESULT_NAMES,
        manual,
        reference,
        actual,
        strict=True,
    ):
        check_correctness(
            f"autograd/manual/{name}",
            manual_result,
            actual_result,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )
        check_correctness(
            f"autograd/reference/{name}",
            reference_result,
            actual_result,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )


def test_bf16_autograd_preserves_two_live_contexts_on_one_cached_workspace(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    first_inputs = generate_inputs(rank, device, world_size, 1, 1, 512, 256, 256)
    second_inputs = generate_inputs(
        rank + world_size,
        device,
        world_size,
        1,
        1,
        512,
        256,
        256,
    )
    first_reference = _reference_in_autograd_order(run_reference_bf16(*first_inputs))
    second_reference = _reference_in_autograd_order(run_reference_bf16(*second_inputs))
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=512,
        schedule_capacity_multiplier=1.5,
        all_gather_top_experts_chunk_bytes=16,
    )
    workspace = functional.get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=512,
        hidden_size=256,
        topk=1,
    )
    first_top_experts, first_grad_output, first_differentiable = _make_autograd_inputs(
        first_inputs
    )
    second_top_experts, second_grad_output, second_differentiable = _make_autograd_inputs(
        second_inputs
    )

    first_output = autograd.bf16(
        config,
        dist.group.WORLD,
        first_differentiable[0],
        first_top_experts,
        first_differentiable[1],
        *first_differentiable[2:],
    )
    second_output = autograd.bf16(
        config,
        dist.group.WORLD,
        second_differentiable[0],
        second_top_experts,
        second_differentiable[1],
        *second_differentiable[2:],
    )
    assert (
        functional.get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=512,
            hidden_size=256,
            topk=1,
        )
        is workspace
    )

    second_actual = (
        second_output,
        *torch.autograd.grad(
            second_output,
            second_differentiable,
            second_grad_output,
        ),
    )
    first_actual = (
        first_output,
        *torch.autograd.grad(
            first_output,
            first_differentiable,
            first_grad_output,
        ),
    )

    for layer, reference, actual in (
        ("second", second_reference, second_actual),
        ("first", first_reference, first_actual),
    ):
        for name, reference_result, actual_result in zip(
            RESULT_NAMES,
            reference,
            actual,
            strict=True,
        ):
            check_correctness(
                f"autograd/two-live/{layer}/{name}",
                reference_result,
                actual_result,
                BF16_TOLERANCE,
                print_stats=rank == 0,
            )
