import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Any, Dict, List, Tuple  # noqa: E402

import hydra  # noqa: E402
from lightning import LightningDataModule, LightningModule, Trainer  # noqa: E402
from lightning.pytorch.loggers import Logger  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def predict(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generates predictions for a given input dataset.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    # If ckpt_path is provided and is a .pt file, we handle it manually
    # by overriding the baseline weight path.
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path and ckpt_path.endswith(".pt"):
        log.info(f"Using vanilla .pt weight: {ckpt_path}")
        # Override the baseline weights in the config before instantiation
        cfg.model.model_weights.baseline = ckpt_path
        # We don't pass it to trainer.predict because it's not a Lightning checkpoint
        ckpt_path = None
    elif ckpt_path:
        log.info(f"Using Lightning checkpoint: {ckpt_path}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting inference!")
    trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    log.info("Inference completed.")

    return {}, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for inference.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    extras(cfg)

    predict(cfg)


if __name__ == "__main__":
    main()
