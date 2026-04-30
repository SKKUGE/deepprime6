import glob
import logging
import os
from typing import Any, Dict, Tuple

import pandas as pd
import torch

try:
    import wandb  # noqa: F401
    _WANDB_AVAILABLE = True
except (ImportError, AttributeError):
    _WANDB_AVAILABLE = False
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef

from src.models.components.loss.weighted_mse_loss import LABEL_IDX

from .components.blocks.parallel import ParallelDeepPrimeModels

log = logging.getLogger(__name__)


class PE6DeepPrimeModule(LightningModule):
    """A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,  # Optimizer
        scheduler: torch.optim.lr_scheduler,  # Learning rate scheduler
        feature_extractor: torch.nn.Module,  # DeepPrime models
        dataprops: Dict[str, Any],  # PE6 class list
        model_weights: Dict[str, Any],  # Path to the pre-defined model weights
        compile: bool,
        criterion: Dict[str, Any],  # Loss function
        prediction_save_path: str = "test_predictions.csv",
    ) -> None:
        # Inference
        """Initialize a `GeneInteractionLitModule`.

        Args:
            net (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        models: list = []
        if (
            "baseline" in self.hparams.model_weights
            and self.hparams.model_weights.baseline is not None
        ):
            print(f"Loading models from {self.hparams.model_weights.baseline}")
            m_files = sorted(glob.glob(self.hparams.model_weights.baseline))
            if len(m_files) == 0:
                raise FileNotFoundError(f"No baseline model weights found matching pattern: {self.hparams.model_weights.baseline}. Please check the path or set model.model_weights.baseline=null to train from scratch.")
            
            for m_file in m_files:
                model = feature_extractor()
                genet_sd = torch.load(m_file, weights_only=True)
                remapped_sd = self._remap_genet_keys(genet_sd, model)
                model.load_state_dict(remapped_sd, strict=False)
                models.append(model)
        else:
            num_ensemble = self.hparams.model_weights.get("num_ensemble", 20)
            if "num_ensemble" not in self.hparams.model_weights:
                log.warning(
                    "model_weights.num_ensemble not found in config, defaulting to %d. "
                    "Please add num_ensemble to your experiment config.",
                    num_ensemble,
                )
            print(f"Initializing {num_ensemble} models from scratch")
            for _ in range(num_ensemble):
                model = feature_extractor()
                models.append(model)
        self.feature_extractor: nn.Module = ParallelDeepPrimeModels(models)

        for param in self.feature_extractor.parameters():
            param.requires_grad = self.hparams.model_weights.requires_grad  # Transfer learning

        # loss function
        self.criterion = criterion()

        # metric objects for calculating and averaging accuracy across batches
        self.train_pearson = PearsonCorrCoef(num_outputs=len(self.hparams.dataprops.PE_class_list))
        self.val_pearson = PearsonCorrCoef(num_outputs=len(self.hparams.dataprops.PE_class_list))
        self.test_pearson = PearsonCorrCoef(num_outputs=len(self.hparams.dataprops.PE_class_list))
        self.train_spearman = SpearmanCorrCoef(
            num_outputs=len(self.hparams.dataprops.PE_class_list)
        )
        self.val_spearman = SpearmanCorrCoef(num_outputs=len(self.hparams.dataprops.PE_class_list))
        self.test_spearman = SpearmanCorrCoef(
            num_outputs=len(self.hparams.dataprops.PE_class_list)
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation general performance
        self.val_pearson_best = MaxMetric()
        self.val_spearman_best = MaxMetric()

        # For drawing scatterplots for test
        self.test_preds = []
        self.test_targets = []
        self.test_annot = []

        # For inference (prediction) results
        self.predict_preds = []
        self.predict_targets = []
        self.predict_annot = []

    @staticmethod
    def _remap_genet_keys(genet_sd, model):
        """Remap genet state_dict keys to match our model's layer indices.

        Genet weights (.pt) typically have:
            d: d.0(Linear) -> d.1(ReLU) -> d.2(Dropout) -> d.3(Linear) -> d.4(ReLU) -> d.5(Dropout) -> d.6(Linear)
            head: head.0(Dropout) -> head.1(Linear)

        Our GeneInteractionModelOriginalImplementation adds BatchNorm1d layers:
            ours: d.0(Linear) -> d.1(BN) -> d.2(ReLU) -> d.3(Dropout) -> d.4(Linear) -> d.5(BN) -> d.6(ReLU) -> d.7(Dropout) -> d.8(Linear) -> d.9(BN)
            ours: head.0(BN) -> head.1(Dropout) -> head.2(Linear)
        """
        model_name = model.__class__.__name__
        our_keys = set(model.state_dict().keys())
        
        # Determine if the current model HAS BNs in 'd' module to decide on remapping
        # This makes it robust even if class names change
        has_bn_in_d = any("d.1.weight" in k or "d.1.running_mean" in k for k in our_keys)
        has_bn_in_head = any("head.0.weight" in k or "head.0.running_mean" in k for k in our_keys)

        if not has_bn_in_d and not has_bn_in_head:
            # Matches Vanilla / Original DeepPrime structure
            remapped = {k: v for k, v in genet_sd.items() if k in our_keys}
            skipped = [k for k in genet_sd.keys() if k not in our_keys]
            if skipped:
                log.info(f"  [remap] No BNs detected in model ({model_name}). Identity mapping used. Skipped {len(skipped)} keys from Genet SD.")
            return remapped

        # Remapping for models WITH BNs (like our current OriginalImplementation)
        KEY_MAP = {
            "d.3.weight": "d.4.weight",
            "d.6.weight": "d.8.weight",
            "head.1.weight": "head.2.weight",
            "head.1.bias": "head.2.bias",
        }
        
        remapped = {}
        loaded, skipped = [], []
        for k, v in genet_sd.items():
            new_key = KEY_MAP.get(k, k)
            if new_key in our_keys:
                remapped[new_key] = v
                loaded.append(f"{k} -> {new_key}" if k != new_key else k)
            else:
                skipped.append(k)
        
        if skipped:
            log.info(f"  [remap] BNs detected in model ({model_name}). Applied remapping. Skipped {len(skipped)} keys.")
        return remapped

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Override to handle remapping from legacy checkpoints.
        
        Legacy checkpoints (.ckpt) were often trained with BNs. 
        When evaluating with Vanilla (no BNs), we need to remap the Linear layers.
        """
        our_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # Check if the checkpoint HAS BNs but the model DOES NOT
        has_bn_in_ckpt = any(".d.1.weight" in k for k in ckpt_keys)
        has_bn_in_model = any(".d.1.weight" in k for k in our_keys)
        
        if has_bn_in_ckpt and not has_bn_in_model:
            log.info("  [load_state_dict] Legacy checkpoint (with BNs) detected. Remapping to BN-less model...")
            remapped_sd = {}
            # Generic remapping for ensemble models
            # feature_extractor.models.X.d.4.weight -> feature_extractor.models.X.d.3.weight
            # feature_extractor.models.X.d.8.weight -> feature_extractor.models.X.d.6.weight
            # feature_extractor.models.X.head.2.weight -> feature_extractor.models.X.head.1.weight
            for k, v in state_dict.items():
                new_key = k
                if ".d.4." in k:
                    new_key = k.replace(".d.4.", ".d.3.")
                elif ".d.8." in k:
                    new_key = k.replace(".d.8.", ".d.6.")
                elif ".head.2." in k:
                    new_key = k.replace(".head.2.", ".head.1.")
                
                if new_key in our_keys:
                    remapped_sd[new_key] = v
                else:
                    # Skip BN keys and other mismatched ones
                    pass
            return super().load_state_dict(remapped_sd, strict=False)
            
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        Args:
            x (torch.Tensor): Input tensor containing genetic features, biofeatures, and pe_type.

        Returns:
            torch.Tensor: Output tensor of regression results.
        """
        # x == (g, b)
        # Feature extraction using GeneInteractionModule
        genetic_features, biofeatures = x

        # Forward pass through the DeepPrime feature extractor: ensemble of models
        output = self.feature_extractor(genetic_features, biofeatures)
        output = torch.mean(torch.stack(output, dim=0), dim=0)

        return output

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_pearson.reset()
        self.val_spearman.reset()
        self.val_pearson_best.reset()
        self.val_spearman_best.reset()

    def model_step(
        self, batch: Tuple[Tuple[Tuple[Tensor, Tensor, Tensor], Tensor], pd.DataFrame]
    ) -> Tuple[Tensor, Tensor, Tensor, pd.DataFrame]:
        """Perform a single model step on a batch of data.

        This method takes a batch of data containing the input tensor of images and target labels,
        and performs a forward pass through the model to obtain predictions. It then calculates
        the loss between the predictions and the target labels using the specified criterion.

        Args:
            batch (Tuple[Tuple[Tuple[Tensor, Tensor, Tensor], Tensor], pd.DataFrame]):
                A batch of data containing the input tensor of images and target labels.

        Returns:
            Tuple[Tensor, Tensor, Tensor, pd.DataFrame]:
                A tuple containing the following elements in order:
                - A tensor of losses.
                - A tensor of predictions.
                - A tensor of target labels.
                - A pandas DataFrame containing additional annotations.

        Raises:
            None

        Examples:
            >>> data = (input_tensor, target_tensor)
            >>> annot = pd.DataFrame(...)
            >>> batch = (data, annot)
            >>> loss, preds, targets, annotations = model.model_step(batch)
        """
        data, annot = batch
        x, y = data
        preds = self.forward(x)
        loss = self.criterion(preds, y)

        return loss, preds, y[:, :LABEL_IDX], annot

    def training_step(
        self, batch: Tuple[Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, _ = self.model_step(batch)

        preds = preds.flatten()
        targets = targets.flatten()

        self.train_loss.update(loss)
        self.train_pearson.update(preds, targets)
        self.train_spearman.update(preds, targets)

        # Log metrics step-wise for smooth training curves in WandB
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        # Use intermediate calculation to log the batch metrics roughly
        # Usually Pearson isn't perfect per-batch, but good enough for a training trend curve
        self.log(
            "train/pearson_step", self.train_pearson, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            "train/spearman_step", self.train_spearman, on_step=True, on_epoch=False, prog_bar=True
        )

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        # log metrics
        self.log(
            "train/loss",
            self.train_loss.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/pearson",
            self.train_pearson.compute().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/spearman",
            self.train_spearman.compute().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Reset metrics
        self.train_loss.reset()
        self.train_pearson.reset()
        self.train_spearman.reset()

    def validation_step(
        self, batch: Tuple[Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch (Tuple[Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]]): A batch of data (a tuple) containing the input tensor of images and target labels.
            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets, _ = self.model_step(batch)

        preds = preds.flatten()
        targets = targets.flatten()

        # update loss and metrics
        self.val_loss.update(loss)
        self.val_pearson.update(preds, targets)
        self.val_spearman.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        # log metrics
        self.log(
            "val/loss",
            self.val_loss.compute().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/pearson",
            self.val_pearson.compute().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/spearman",
            self.val_spearman.compute().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        pearson = self.val_pearson.compute().mean()  # get current val pearson
        self.val_pearson_best(pearson)  # update best so far val pearson
        # log `val_pearson_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/pearson_best",
            self.val_pearson_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        spearman = self.val_spearman.compute().mean()  # get current val spearman
        self.val_spearman_best(spearman)  # update best so far val spearman
        # log `val_spearman_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/spearman_best",
            self.val_spearman_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        # Reset metrics
        self.val_loss.reset()
        self.val_pearson.reset()
        self.val_spearman.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, annot = self.model_step(batch)

        preds = preds.flatten()
        targets = targets.flatten()

        # Update loss and metrics
        self.test_loss.update(loss)
        self.test_pearson.update(preds, targets)
        self.test_spearman.update(preds, targets)

        # Store predictions and targets for scatterplot
        self.test_preds.extend(preds.tolist())
        self.test_targets.extend(targets.tolist())

        # Each string is fragmented into a list of characters
        # So, we need to flatten the list of lists
        annot = ["".join(frag_id) for frag_id in annot]
        self.test_annot.extend(annot)

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single predict step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        _, preds, targets, annot = self.model_step(batch)

        preds = preds.flatten()
        targets = targets.flatten()

        # Store predictions and targets for inference
        self.predict_preds.extend(preds.tolist())
        self.predict_targets.extend(targets.tolist())

        # Each string is fragmented into a list of characters
        annot = ["".join(frag_id) for frag_id in annot]
        self.predict_annot.extend(annot)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        # log metrics
        self.log(
            "test/loss",
            self.test_loss.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/pearson",
            self.test_pearson.compute().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/spearman",
            self.test_spearman.compute().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # logging test prediction and target values
        result_table = pd.DataFrame(
            list(zip(self.test_targets, self.test_preds, self.test_annot)),
            columns=["Target", "Prediction", "ID"],
        )
        # result_table = result_table.explode(["Target", "Prediction", "ID"])   # Not used anymore

        result_table["PE_type"] = [
            self.hparams.dataprops.PE_class_list[i % len(self.hparams.dataprops.PE_class_list)]
            for i in range(len(result_table))
        ]

        result_table = result_table.reset_index(drop=False)

        if _WANDB_AVAILABLE and self.trainer.logger is not None:
            if self.trainer.logger.__class__.__name__ == "WandbLogger":
                log.info("Test results table calculation completed. (Export to WandB disabled per configuration)")

    def on_predict_epoch_end(self) -> None:
        """Lightning hook that is called when a predict epoch ends."""
        # logging test prediction and target values
        result_table = pd.DataFrame(
            list(zip(self.predict_targets, self.predict_preds, self.predict_annot)),
            columns=["Target", "Prediction", "ID"],
        )

        result_table["PE_type"] = [
            self.hparams.dataprops.PE_class_list[i % len(self.hparams.dataprops.PE_class_list)]
            for i in range(len(result_table))
        ]

        result_table = result_table.reset_index(drop=False)

        # Save to CSV for inference results
        save_path = self.hparams.prediction_save_path
        if save_path:
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            result_table.to_csv(save_path, index=False)
            log.info(f"Predictions saved to: {save_path}")

        # Reset accumulated data
        self.predict_preds.clear()
        self.predict_targets.clear()
        self.predict_annot.clear()

        # Reset metrics
        self.test_loss.reset()
        self.test_pearson.reset()
        self.test_spearman.reset()

        # Clear accumulated test data to prevent memory leaks during sweeps
        self.test_preds.clear()
        self.test_targets.clear()
        self.test_annot.clear()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.feature_extractor = torch.compile(self.feature_extractor)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = PE6DeepPrimeModule(None, None, None, None, None, False, None)
