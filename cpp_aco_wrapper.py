#!/usr/bin/env python3
"""
C++ ACO Wrapper for AlphaAnt

Loads the AlphaAnt C++ ACO module and provides Python interfaces for
BPP, MKP, and OP problems.
"""

from functools import lru_cache
import os
from pathlib import Path
import sys

import numpy as np
import pybind11
import torch
from torch.utils.cpp_extension import load


# AlphaAnt paths
ALPHAANT_ROOT = Path("~/Research/AlphaAnt").expanduser()
SRC_PATH = ALPHAANT_ROOT / "src" / "aco.cpp"
BUILD_DIR = ALPHAANT_ROOT / ".cache" / "alphaant_tsp_aco_cpp"


@lru_cache(maxsize=1)
def load_alphaant_cpp_module():
    """Load the AlphaAnt C++ ACO module."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    python_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")

    return load(
        name="alphaant_tsp_aco_cpp",
        sources=[str(SRC_PATH)],
        extra_include_paths=[pybind11.get_include()],
        build_directory=str(BUILD_DIR),
        verbose=False,
        extra_cflags=["-O3", "-std=c++17", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
    )


def to_numpy_matrix(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy matrix."""
    return tensor.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()


def to_numpy_vector(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy vector."""
    return tensor.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()


def trace_paths_to_tensor(trace, device: torch.device) -> torch.Tensor:
    """Convert trace to torch tensor."""
    return torch.as_tensor(np.asarray(trace.paths_numpy(), dtype=np.int64), device=device, dtype=torch.long)


def fresh_seed(seed: int | None) -> int:
    """Generate a fresh seed."""
    if seed is not None:
        return int(seed)
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])


# Export the C++ classes
def get_aco_bpp():
    """Get the ACO_BPP class from the C++ module."""
    return load_alphaant_cpp_module().ACO_BPP


def get_aco_mkp():
    """Get the ACO_MKP class from the C++ module."""
    return load_alphaant_cpp_module().ACO_MKP


def get_aco_op():
    """Get the ACO_OP class from the C++ module."""
    return load_alphaant_cpp_module().ACO_OP


if __name__ == "__main__":
    # Test loading the module
    print("Loading AlphaAnt C++ module...")
    module = load_alphaant_cpp_module()
    print(f"Module loaded successfully!")
    print(f"Available classes: {dir(module)}")
