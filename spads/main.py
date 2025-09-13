"""
The entrypoint to find correspondences between frames.
"""
from __future__ import annotations

from loguru import logger
import hydra

from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore

from .config import Config
from .experiment import run_experiment

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(
    version_base=None,
    config_path=None,  # No config path - use only structured configs
    config_name="config",
)
@logger.catch
def main(cfg: DictConfig) -> None:
    """
    Main function
    """    

    # Setup Logger
    logger.remove()

    # Console logging
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=cfg.log.log_level,
        colorize=cfg.log.colorize,
    )

    # File logging (if enabled)
    if cfg.log.sink_file:
        logger.add(
            sink=cfg.log.sink_file,
            level=cfg.log.log_level,
            colorize=cfg.log.colorize,  # No colors in file
            rotation=cfg.log.rotation,
            retention=cfg.log.retention,
            backtrace=cfg.log.backtrace,
        )
        logger.info(f"File logging enabled: {cfg.log.sink_file}")

    logger.info("Logger initialized...")

    # Run the experiment
    score: float | None = run_experiment(cfg)
    logger.info(f"Experiment completed")


if __name__ == "__main__":
    main()
