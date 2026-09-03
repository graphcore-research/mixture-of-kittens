# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Immutable identity of Graphcore's additive PyTorch autograd adapter."""

try:
    from ._build_info import BUILD_INFO as AUTOGRAD_ADAPTER_BUILD_INFO
except ModuleNotFoundError as error:
    if error.name != f"{__package__}._build_info":
        raise
    AUTOGRAD_ADAPTER_BUILD_INFO = None


AUTOGRAD_ADAPTER_API_REVISION = 1
AUTOGRAD_ADAPTER_PROTECTED_BASE = "22fc95ae6e331a738c4a58a227a8b03cac586e12"
AUTOGRAD_ADAPTER_SOURCE_SHA256 = (
    "a6cb4beba7574e4d1c9b3ee567edaee08c15623298c68edfe20392f491044732"
)
