import pathlib
from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .components import preprocess_data
from .components.data_model.collate_func import custom_collate_fn
from ..utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PE6DataModule(LightningDataModule):
    """`LightningDataModule` for the PE6 dataset.

    The purpose of this datamodule is to provide a standardized way to load PE6 data to the transfer learning models, DeepPrime.
    DeepPrime has implements its original data processing pipeline, which is not compatible with the PE6 data.
    Thus, this datamodule will provide a way to load the PE6 data in a way that is compatible with the DeepPrime models.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        datafilter: Optional[Dict[str, Any]] = None,
        datasplit: Optional[Dict[str, Any]] = None,
        dataloader: Optional[Dict[str, Any]] = None,
        dataset: Optional[Dict[str, Any]] = None,
        skip_preprocessing: bool = False,
        csv_path: Optional[str] = None,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()
        if datafilter is None:
            datafilter = {}
        if datasplit is None:
            datasplit = {}
        if dataloader is None:
            dataloader = {}
        if dataset is None:
            dataset = {}

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.data_dir: str = data_dir
        # Use a unique name for processed data based on the input data_dir to avoid stale data
        data_path_obj = pathlib.Path(data_dir)
        if data_path_obj.is_file():
            self.processed_data_path: str = str(
                data_path_obj.parent / f"{data_path_obj.stem}_processed.parquet"
            )
        else:
            self.processed_data_path: str = str(
                data_path_obj.parent / f"{data_path_obj.name}_processed.parquet"
            )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of in-scope PE6 classes (PEmaxdRNaseH, PE6a-c).
        """
        return 4

    @property
    def example_proeprty(self) -> int:
        """Get property on-demand .

        :return: 0
        """
        return 0

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        if not self.hparams.skip_preprocessing:
            if not pathlib.Path(self.processed_data_path).exists():
                output_filename = pathlib.Path(self.processed_data_path).name
                preprocess_data(data_dir=self.data_dir, output_file=output_filename)
            else:
                log.info(
                    f"Skipping preprocessing as {self.processed_data_path} already exists. To force preprocessing, delete the file."
                )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        import pandas as pd
        from sklearn.model_selection import GroupShuffleSplit

        from src.data.components.data_model.pe6_dprime_dataset import (
            PE6DeepPrimeDataset,
        )
        from src.data.components.pe6_preprocess_data import preprocess_data as preprocess_func

        # Handle predict stage specifically if csv_path is provided
        if stage == "predict" and self.hparams.csv_path:
            log.info(f"Loading prediction data from {self.hparams.csv_path}")
            if self.hparams.csv_path.endswith(".parquet"):
                processed_df = pd.read_parquet(self.hparams.csv_path)
            else:
                df = pd.read_csv(self.hparams.csv_path)
                # We need to preprocess if it's raw CSV
                processed_df = preprocess_func(data=df)
            
            # Load normalization stats if possible
            norm_mean = None
            norm_std = None
            if (
                "norm_mean_path" in self.hparams.dataset
                and "norm_std_path" in self.hparams.dataset
            ):
                try:
                    norm_mean = pd.read_csv(
                        self.hparams.dataset.norm_mean_path, index_col=0, header=None
                    ).squeeze("columns")
                    norm_std = pd.read_csv(
                        self.hparams.dataset.norm_std_path, index_col=0, header=None
                    ).squeeze("columns")
                except Exception as e:
                    log.warning(f"Failed to load normalization files: {e}")

            self.data_predict = PE6DeepPrimeDataset(
                processed_df,
                self.hparams.datafilter,
                read_count_filter=0, # No filter for prediction
                norm_mean=norm_mean,
                norm_std=norm_std,
            )
            return


        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # When skip_preprocessing is True, read directly from data_dir
            # (the user provides already-preprocessed data).
            data_path = (
                self.data_dir
                if self.hparams.skip_preprocessing
                else self.processed_data_path
            )
            data: pd.DataFrame = pd.read_parquet(data_path)

            # Stratified data split - first split the data into training and test sets
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.hparams.datasplit.test_size,
                random_state=42,
            )
            for train_val_index, test_index in gss.split(
                X=data,
                y=data[self.hparams.datasplit.stratification_column],
                groups=data[self.hparams.datasplit.group_column],
            ):
                X_train_val, X_test = (
                    data.iloc[train_val_index],
                    data.iloc[test_index],
                )

            # Further split the training data into training and validation sets
            gss_val = GroupShuffleSplit(
                n_splits=1,
                test_size=self.hparams.datasplit.val_size,
                random_state=42,
            )  # 0.25 of 0.8 is 0.2 of the original data
            for (
                train_index,
                val_index,
            ) in gss_val.split(  # TODO: 5-fold cross-validation https://gist.github.com/ashleve/ac511f08c0d29e74566900fd3efbb3ec
                X=X_train_val,
                y=X_train_val[self.hparams.datasplit.stratification_column],
                groups=X_train_val[self.hparams.datasplit.group_column],
            ):
                X_train, X_val = (
                    X_train_val.iloc[train_index],
                    X_train_val.iloc[val_index],
                )

            # Load mean/std for normalization if provided
            norm_mean = None
            norm_std = None
            if (
                "norm_mean_path" in self.hparams.dataset
                and "norm_std_path" in self.hparams.dataset
            ):
                try:
                    norm_mean = pd.read_csv(
                        self.hparams.dataset.norm_mean_path, index_col=0, header=None
                    ).squeeze("columns")
                    norm_std = pd.read_csv(
                        self.hparams.dataset.norm_std_path, index_col=0, header=None
                    ).squeeze("columns")
                except Exception as e:
                    print(f"Warning: Failed to load normalization files: {e}")

            self.data_train = PE6DeepPrimeDataset(
                X_train,
                self.hparams.datafilter,
                read_count_filter=self.hparams.dataset.train_read_count_filter,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )

            # If norm_mean and norm_std were not provided via config,
            # retrieve them from the training set to prevent data leakage in val/test.
            if norm_mean is None or norm_std is None:
                norm_mean = self.data_train.norm_mean
                norm_std = self.data_train.norm_std

            self.data_val = PE6DeepPrimeDataset(
                X_val,
                self.hparams.datafilter,
                read_count_filter=self.hparams.dataset.val_read_count_filter,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )
            self.data_test = PE6DeepPrimeDataset(
                X_test,
                self.hparams.datafilter,
                read_count_filter=self.hparams.dataset.test_read_count_filter,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )

            if self.trainer.logger:
                self.trainer.logger.log_hyperparams(
                    {
                        "train/size": len(self.data_train),
                        "val/size": len(self.data_val),
                        "test/size": len(self.data_test),
                    }
                )


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            shuffle=True,
            persistent_workers=self.hparams.dataloader.persistent_workers,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.dataloader.persistent_workers,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.dataloader.persistent_workers,
            collate_fn=custom_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        dataset = getattr(self, "data_predict", None)
        if dataset is None:
            log.info("No prediction data specifically loaded. Using test data for prediction fallback.")
            dataset = self.data_test

        if dataset is None:
            raise RuntimeError("Neither prediction data nor test data is available for predict_dataloader. "
                             "Make sure to call setup() properly.")

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        Releases dataset references so that persistent dataloader workers and the
        memory they hold can be reclaimed between Hydra sweep trials.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = PE6DataModule()
