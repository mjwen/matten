from pathlib import Path
from typing import Any, Dict, Optional, Union

from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader

from eigenn.utils import to_path


class BaseDataModule(LightningDataModule):
    """
    Base datamodule.

    Args:
        trainset_filename: path to the training set file.
        valset_filename: path to the validation set file.
        testset_filename: path to the validation set file.
        state_dict_filename: path to save the state dict of the data module.
        restore_state_dict_filename: If not `None`, the model is running in
            restore mode and the initial state dict is read from this file. If `None`,
            the model in running in regular mode and this is ignored.
            Note the difference between this and `state_dict_filename`.
            `state_dict_filename` only specifies the output state dict, does not care
            about how the initial state dict is obtained: it could be restored from
            `restore_state_dict_file` or computed from the dataset.
            pretrained model used in finetune?
        loader_kwargs: extra arguments passed to DataLoader, e.g. shuffle, batch_size...
    """

    def __init__(
        self,
        trainset_filename: Union[str, Path],
        valset_filename: Union[str, Path],
        testset_filename: Union[str, Path],
        *,
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        loader_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.trainset_filename = trainset_filename
        self.valset_filename = valset_filename
        self.testset_filename = testset_filename
        self.state_dict_filename = state_dict_filename
        self.restore_state_dict_filename = restore_state_dict_filename
        self.loader_kwargs = loader_kwargs

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self):
        """
        Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Load data.

        Set variables: self.train_data, self.val_data, self.test_data.
        """

        # init_state_dict = self.get_init_state_dict()
        #
        # self.train_data = ...
        # self.val_data = ...
        # self.test_data = ...
        #
        # # save dataset state dict
        # self.train_data.save_state_dict_file(self.state_dict_filename)

        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, **self.loader_kwargs)

    def get_to_model_info(self) -> Dict[str, Any]:
        """
        Pack necessary dataset info as as dict, and this dict will be passed as
        arguments to the model. Such info might include feature size, number of classes
        for reaction.
        """
        raise NotImplementedError

    def get_init_state_dict(self):
        """
        Determine the value of dataset state dict based on:
        - whether this is in finetune model based on pretrained_model_state_dict_filename
        - restore_state_dict_filename
        """

        # restore training
        if self.restore_state_dict_filename:
            init_state_dict = to_path(self.restore_state_dict_filename)

            if not init_state_dict.exists():
                raise FileNotFoundError(
                    "Cannot restore datamodule. Dataset state dict file does not "
                    f"exist: {init_state_dict}"
                )

        # regular training mode
        else:
            init_state_dict = None

        return init_state_dict
