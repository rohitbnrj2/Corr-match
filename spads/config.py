"""
Base Configuration for the package.
"""
from __future__ import annotations

from typing import Annotated, Literal
from dataclasses import dataclass, field

from spads.experiment import ExperimentConfigs
from spads.corr_match.sift import SiftConfigs
from spads.data_gen.sim import SimConfigs

@dataclass
class LoggerConfig:
    """
    The configuration for the loguru logger.
    """

    log_level: Annotated[str, Literal['DEBUG', 'INFO', 'CRITICAL', 'ERROR']] = 'DEBUG'
    backtrace: bool = False
    rotation: str = "1 MB"      # Set only if storing log files
    retention: str = "10 days"  # Set only if storing log files
    format: str = "<green>{time}</green> | <level> {message} </level>"
    colorize: bool = True
    sink_file: Annotated[str, Literal["logs/corr.log"]] | None = None  # File name to store logs 


@dataclass
class BaseConfig:
    """
    Basic experimental configuration
    """

    exp_name: str = "simulator_spads"   # The experiment name
    exp_dir: str = "./outputs"   # The directory under which the experiment results are stored


@dataclass
class Config:
    """
    The Configuration for setting up the experiment.
    """

    log: LoggerConfig = field(default_factory=lambda: LoggerConfig())
    base: BaseConfig = field(default_factory=lambda: BaseConfig())
    exp: ExperimentConfigs = field(default_factory=lambda: ExperimentConfigs())
    # Applicable when performing correspondence matching with sift
    sift: SiftConfigs = field(default_factory=lambda: SiftConfigs())
    # SPAD simulator configurations for RGB -> Binary SPAD images
    sim: SimConfigs = field(default_factory=lambda: SimConfigs())
