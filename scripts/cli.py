"""
Command line parser class.

See pytorch_lightning docs for info on how to use this:
https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html
"""

import pl_bolts
import torch
from pytorch_lightning.utilities.cli import LightningCLI, instantiate_class


class EigennCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):

        # argument link
        # TODO, this does not work now due to a lightning bug
        # parser.link_arguments("trainer.max_epochs", "lr_scheduler.max_epochs")

        # line_to should be argument of the model
        parser.add_optimizer_args(
            (torch.optim.Adam, torch.optim.SGD),
            link_to="model.optimizer_hparams",
        )

        # line_to should be argument of the model
        parser.add_lr_scheduler_args(
            (
                pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ),
            link_to="model.lr_scheduler_hparams",
        )

    # NOTE Reimplement instantiate_classes to call datamodule setup() and pass necessary
    # dataset info to model as `hparams_dataset`
    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""

        ###################
        # original
        # self.config_init = self.parser.instantiate_classes(self.config)
        # self.datamodule = self._get(self.config_init, "data")
        # self.model = self._get(self.config_init, "model")
        # self._add_configure_optimizers_method_to_model(self.subcommand)
        # self.trainer = self.instantiate_trainer()

        ##################
        # modified

        # pop data config
        data_config = self.config.pop("data")

        # instantiate datamodule
        self.datamodule, to_model_info = self.instantiate_datamodule(data_config)

        # add model dataset_hparams from dataset
        # Note, `dataset_hparams` is an argument of the lightning model
        self.config["model"]["dataset_hparams"] = to_model_info

        # instantiate others
        self.config_init = self.parser.instantiate_classes(self.config)

        # add data and datamodule back
        self.config["data"] = data_config
        self.config_init["data"] = self.datamodule

        self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()

    @staticmethod
    def instantiate_datamodule(data_config):

        args = tuple()  # no positional args
        datamodule = instantiate_class(args, data_config)

        # setup datamodule and get to model into
        datamodule.setup()
        to_model_info = datamodule.get_to_model_info()

        return datamodule, to_model_info
