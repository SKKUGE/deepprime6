import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Any, Dict, List, Tuple  # noqa: E402

import hydra  # noqa: E402
from lightning import LightningDataModule, LightningModule, Trainer  # noqa: E402
from lightning.pytorch.loggers import Logger  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. install project as a package: `pip install -e .`
# 2. or manually set PYTHONPATH
# ------------------------------------------------------------------------------------ #

from src.utils import (  # noqa: E40t2
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

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
        # We don't pass it to trainer.test because it's not a Lightning checkpoint
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

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)



    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
