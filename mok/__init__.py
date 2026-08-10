# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch  # noqa: F401

# Keep custom-op registration ahead of fake registrations and high-level APIs.
# isort: off
from . import ops as ops
from . import _fake_impls as _fake_impls
from . import functional as functional
from . import autograd as autograd

# isort: on
from ._provenance import (
    AUTOGRAD_ADAPTER_API_REVISION as AUTOGRAD_ADAPTER_API_REVISION,
)
from ._provenance import (
    AUTOGRAD_ADAPTER_BUILD_INFO as AUTOGRAD_ADAPTER_BUILD_INFO,
)
from ._provenance import (
    AUTOGRAD_ADAPTER_PROTECTED_BASE as AUTOGRAD_ADAPTER_PROTECTED_BASE,
)
from ._provenance import (
    AUTOGRAD_ADAPTER_SOURCE_SHA256 as AUTOGRAD_ADAPTER_SOURCE_SHA256,
)

try:
    __version__ = version("mixture-of-kittens")
except PackageNotFoundError:
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as file:
        __version__ = tomllib.load(file)["project"]["version"]

__all__ = [
    "AUTOGRAD_ADAPTER_API_REVISION",
    "AUTOGRAD_ADAPTER_BUILD_INFO",
    "AUTOGRAD_ADAPTER_PROTECTED_BASE",
    "AUTOGRAD_ADAPTER_SOURCE_SHA256",
    "__version__",
    "autograd",
    "functional",
    "ops",
]
