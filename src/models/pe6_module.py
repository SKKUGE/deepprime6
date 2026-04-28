import glob
import logging
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import wandb
import wandb.plot
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
        projection: Dict[str, Any],  # embedding (projection) dimension
        dataprops: Dict[str, Any],  # PE6 class list
        merger: Dict[str, Any],  # Merge the features from the ensemble of models
        net: torch.nn.Module,  # DeepPrime6 model
        model_weights: Dict[str, Any],  # Path to the pre-defined model weights
        compile: bool,
        criterion: Dict[str, Any],  # Loss function
    ) -> None:
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
        for m_file in sorted(glob.glob(self.hparams.model_weights.baseline)):
            model = feature_extractor()
            model.load_state_dict(
                torch.load(m_file, weights_only=True), strict=False
            )  # Header removed for transfer learning
            models.append(model)
        self.feature_extractor: nn.Module = ParallelDeepPrimeModels(models)

        for param in self.feature_extractor.parameters():
            # Freeze the feature extractor
            param.requires_grad = self.hparams.model_weights.requires_grad  # Transfer learning

        self.feature_merger = self.hparams.merger()

        n_input_features: int = self._calculate_flattened_dimensions()
        self.net = net(
            input_features=n_input_features,
            output_size=len(self.hparams.dataprops.PE_class_list),
            num_classes=len(self.hparams.dataprops.PE_class_list),
        )

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

    def _calculate_flattened_dimensions(self):
        f1: int = (
            self.hparams.merger.keywords["adaptive_avgpool_output_size"]
            if self.hparams.merger.keywords["final_avgpool"]
            else self.hparams.feature_extractor.keywords["head_in_features"]
        )
        f2: int = self.hparams.merger.keywords["layer_hidden_in_channels"]
        f3: int = 2 ** ((self.hparams.merger.keywords["num_residual_blocks"] - 1) // 2)

        return f1 * f2 * f3

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
        deep_prime_features = self.feature_extractor(genetic_features, biofeatures)

        features = self.feature_merger(torch.stack(deep_prime_features, dim=1))
        features = torch.squeeze(features)

        # Forward pass through the net
        result = self.net(features)

        return result

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

        preds = preds.squeeze()
        targets = targets.squeeze()

        self.train_loss.update(loss)
        self.train_pearson.update(preds, targets)
        self.train_spearman.update(preds, targets)

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

        preds = preds.squeeze()
        targets = targets.squeeze()

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

        preds = preds.squeeze()
        targets = targets.squeeze()

        # Update loss and metrics
        self.test_loss.update(loss)
        self.test_pearson.update(preds, targets)
        self.test_spearman.update(preds, targets)

        # Store predictions and targets for scatterplot
        self.test_preds.extend(preds.tolist())
        self.test_targets.extend(targets.tolist())

        # Each string is fragmented intp a list of characters
        # So, we need to flatten the list of lists
        annot = ["".join(frag_id) for frag_id in annot]
        self.test_annot.extend(annot)

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

        # Group by pegRNA index and calculate the equality of the target and prediction grouped by pegRNA index (For orthogonal validation test/accuracy)
        result_table = result_table.reset_index(drop=False)

        if self.trainer.logger is not None and hasattr(self.trainer.logger, "experiment"):
            try:
                self.trainer.logger.experiment.log(
                    {
                        "test/result_table": wandb.Table(dataframe=result_table),
                    }
                )
            except Exception as e:
                log.warning(f"Failed to log test result table: {e}")

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
            self.net = torch.compile(self.net)

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
    _ = PE6DeepPrimeModule(None, None, None, None)
