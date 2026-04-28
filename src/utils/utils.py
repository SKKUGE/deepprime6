import gc
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # initialize defaults to prevent UnboundLocalError when exceptions occur
        metric_dict: Dict[str, Any] = {}
        object_dict: Dict[str, Any] = {}

        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")
            log.exception(ex)

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            # raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish(
                        # exit_code=0
                    )  # Asynchronously flush all data to wandb and finish the run to speed up the pipeline

            # Explicitly release large objects before garbage collection so that
            # GPU memory and dataloader workers are actually freed between sweep trials.
            #
            # Lightning creates circular references (trainer ↔ model, trainer ↔ datamodule)
            # that prevent Python's reference-counting from freeing them.  We must break
            # these cycles *before* deleting the objects, otherwise DataLoader worker
            # sub-processes (especially with persistent_workers=True) stay alive across
            # trials and eventually exhaust file-descriptor / shared-memory limits
            # (observed crash at ~44/400 trials).
            if object_dict:
                model = object_dict.pop("model", None)
                trainer = object_dict.pop("trainer", None)
                datamodule = object_dict.pop("datamodule", None)
                callbacks = object_dict.pop("callbacks", None)
                loggers = object_dict.pop("logger", None)
                object_dict.clear()

                # 1. Break circular references so refcount-based cleanup works.
                if model is not None and hasattr(model, "trainer"):
                    model.trainer = None
                if datamodule is not None and hasattr(datamodule, "trainer"):
                    datamodule.trainer = None

                # 2. Move model off GPU before deleting.
                if model is not None:
                    try:
                        model.cpu()
                    except Exception as e:
                        log.warning(f"Failed to move model to CPU during cleanup: {e}")

                # 3. Shut down persistent DataLoader workers that the Trainer may
                #    still hold (their sub-processes keep file descriptors open).
                if trainer is not None:
                    for attr in (
                        "train_dataloader",
                        "val_dataloaders",
                        "test_dataloaders",
                    ):
                        try:
                            dl_or_list = getattr(trainer, attr, None)
                            if dl_or_list is None:
                                continue
                            # val_dataloaders / test_dataloaders may be a list.
                            dls = (
                                dl_or_list
                                if isinstance(dl_or_list, (list, tuple))
                                else [dl_or_list]
                            )
                            for dl in dls:
                                it = getattr(dl, "_iterator", None)
                                if it is not None and hasattr(it, "_shutdown_workers"):
                                    it._shutdown_workers()
                        except Exception:
                            pass

                # 4. Delete in dependency order (dependents first, then trainer).
                if model is not None:
                    del model
                if datamodule is not None:
                    del datamodule
                if callbacks is not None:
                    del callbacks
                if loggers is not None:
                    del loggers
                if trainer is not None:
                    del trainer

            gc.collect()

            # Close any matplotlib figures that may have been left open
            # during the trial (e.g. by callbacks or logging hooks).
            try:
                import matplotlib.pyplot as plt

                plt.close("all")
            except ImportError:
                pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
