"""
Sets the experiment runner.
"""
from __future__ import annotations

from typing import Annotated, Literal
from dataclasses import dataclass

from loguru import logger
from omegaconf import DictConfig

import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .sift import sift_runner


@dataclass
class ExperimentConfigs:
    """
    The experimental configurations.
    """

    seed : int = 42             # Seed for reproducibility
    is_cuda: bool = False       # Whether experiment requires CUDA
    algorithm: Annotated[str, Literal["sift", "ransac"]] = "sift"
    dataset_name: Annotated[str, Literal["fans"]] = "fans"

    # SIFT params
    flann_index_kdtree: int = 1
    trees: int = 5              # KD-tree for correspondence matching

def run_experiment(cfgs: DictConfig) -> float | None:
    """
    The basic experiment runner or launcher. This is a generic method.
    Optionally the method returns a float which can be used for sweeping,
    for hyperparameter.
    
    Args:
        cfgs: (DictConfig) Hydra Configurations for the experiment.
    
    Returns:
        (float | None) : The output which tests fitness of the hyperparams.
    """

    # Set seed for experiment
    set_reproducibility(cfgs.exp.seed)

    # Whether cuda is required
    if cfgs.exp.is_cuda:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for CUDA experiments but not installed"
            )
            
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cpu":
            raise RuntimeError(
                f"CUDA unavailable, require CUDA for experiment"
            )

    # Check the algorithm and launch the experiment
    if cfgs.exp.algorithm == "sift":
        sift_runner(cfgs)
    
    else:
        raise ValueError(
            f"Unknown algorithm for experiment {cfgs.exp.algorithm}"
        )


def set_reproducibility(seed: int) -> None:
    """
    Set the reproducibility for the experiment.
    
    Args:
        seed (int) : The experiment seed to monitor
    """

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.random.seed(seed)
