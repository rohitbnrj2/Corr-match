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
    import torch # type: ignore
except ImportError:
    torch = None

try:
    from .corr_match.sift import sift_runner
except ImportError:
    sift_runner = None
    logger.error("Sift module not found")

try:
    from .data_gen.sim import sim_runner

except ImportError:
    sim_runner = None
    logger.error("Simulator module not found")


@dataclass
class ExperimentConfigs:
    """
    The experimental configurations.
    """

    seed : int = 42             # Seed for reproducibility
    is_cuda: bool = False       # Whether experiment requires CUDA
    algorithm: Annotated[str, Literal["sift", "sim", "inpaint"]] = "sift"


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
    algorithm_runners = {
        "sift": sift_runner,
        "sim": sim_runner,
        "inpaint": None  # Add when inpaint_runner is implemented
    }
    
    run_obj = algorithm_runners.get(cfgs.exp.algorithm)

    if run_obj is None:
        raise ValueError(
            f"{cfgs.exp.algorithm.capitalize()} module not found"
        )

    run_obj(cfgs)


def set_reproducibility(seed: int) -> None:
    """
    Set the reproducibility for the experiment.
    
    Args:
        seed (int) : The experiment seed to monitor
    """

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
