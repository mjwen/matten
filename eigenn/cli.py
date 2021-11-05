"""
Command line parser class.

See pytorch_lightning docs for info on how to use this:
https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html
"""

import os
from typing import Any, Dict, Optional

import pl_bolts
import torch
from loguru import logger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.cli import (
    SaveConfigCallback as LightningSaveConfigCallback,
)
from pytorch_lightning.utilities.cli import instantiate_class

from eigenn.log import set_logger
from eigenn.utils import to_path
from eigenn.utils_wandb import (
    get_wandb_checkpoint_and_identifier_latest,
    get_wandb_logger,
    save_files_to_wandb,
)


class EigennCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):

        parser.add_argument(
            "--restore",
            type=bool,
            default=False,
            help="Path to checkpoint to restore the training. If `True`, will try to "
            "automatically find the checkpoint in wandb logs. Will not try to restore "
            "if `None` or `False`.",
        )
        parser.add_argument(
            "--skip_test", type=bool, default=False, help="Whether to skip the test?"
        )
        parser.add_argument(
            "--log_level",
            type=str,
            default="INFO",
            help="Log level, e.g. INFO, DEBUG...",
        )

        # link_to should be argument of the model
        parser.add_optimizer_args(
            (torch.optim.Adam, torch.optim.SGD),
            link_to="model.optimizer_hparams",
        )

        # link_to should be argument of the model
        parser.add_lr_scheduler_args(
            (
                pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ),
            link_to="model.lr_scheduler_hparams",
        )

        # argument link
        # TODO, this does not work now due to a lightning bug
        # parser.link_arguments("trainer.max_epochs", "lr_scheduler.init_args.max_epochs")

        # link trainer and data configs to model to let wandb log them
        parser.link_arguments("trainer", "model.trainer_hparams")
        parser.link_arguments("data", "model.data_hparams")

    def before_instantiate_classes(self) -> None:
        level = self.config["log_level"].upper()
        set_logger(level=level)

    # Reimplement instantiate_classes to call datamodule setup() and pass necessary
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

        # restore info
        checkpoint, wandb_id, _ = self._get_restore_info(self.config)
        if checkpoint:
            self.config["trainer"]["resume_from_checkpoint"] = checkpoint
        if wandb_id:
            logger_config = self.config["trainer"].get("logger", [])
            # TODO, this assumes only one wandb logger used
            if isinstance(logger_config, dict):
                logger_config["init_args"]["id"] = wandb_id
            else:
                raise RuntimeError("Currently only support a single wandb logger.")

        # pop data config
        data_config = self.config.pop("data")

        # instantiate datamodule
        self.datamodule, to_model_info = self._instantiate_datamodule(data_config)

        # add to_model_info to config of model `dataset_hparams` (required by the model)
        self.config["model"]["dataset_hparams"] = to_model_info

        # instantiate others
        self.config_init = self.parser.instantiate_classes(self.config)

        # add data datamodule back
        self.config_init["data"] = self.datamodule

        # add data config back to let lightning cli log it
        self.config["data"] = data_config

        # remove linked config to model, added in `add_arguments_to_parser`
        self.config["model"].pop("lr_scheduler_hparams")
        self.config["model"].pop("optimizer_hparams")
        self.config["model"].pop("trainer_hparams")
        self.config["model"].pop("data_hparams")

        self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()

    @staticmethod
    def _instantiate_datamodule(data_config):
        """
        Setup datamoldule to get info needed to instantiate module.
        """

        args = tuple()  # no positional args
        datamodule = instantiate_class(args, data_config)

        # setup datamodule and get to model into
        datamodule.prepare_data()
        datamodule.setup()
        to_model_info = datamodule.get_to_model_info()

        return datamodule, to_model_info

    @staticmethod
    def _get_restore_info(config: Dict[str, Any]):
        """
        Get info to restore the model from the latest run.

        Args:
            config: experiment config

        Returns:
            checkpoint_path: path to checkpoint to restore. `None` if not found.
            dataset_state_dict_path: path to dataset state dict.
            wandb_id: unique wandb identifier to restore. `None` if not found.
        """

        restore = config["restore"]

        # automatically determine restore info
        if restore is True:
            # TODO, get save_dir from config, this is hard-coded
            checkpoint, wandb_id = get_wandb_checkpoint_and_identifier_latest(
                save_dir="wandb_logs"
            )

            if checkpoint is None:
                logger.warning(
                    "Trying to automatically restore model from checkpoint, but cannot "
                    "find latest checkpoint file. Proceed without restoring checkpoint."
                )

            if wandb_id is None:
                logger.warning(
                    "Trying to automatically restore training with the same wandb "
                    "identifier, but cannot find the identifier of latest run. A new "
                    "wandb identifier will be assigned."
                )

        # given restore checkpoint path, simply use it
        elif isinstance(restore, str):
            checkpoint = to_path(restore)
            wandb_id = None
            if not checkpoint.exists():
                raise ValueError(f"Restore checkpoint does not exist: {restore}")

        else:
            checkpoint = None
            wandb_id = None

        # TODO, we need to consider dataset state dict
        dataset_state_dict = None
        # if dataset_state_dict is None:
        #     logger.warning(
        #         "Trying to automatically restore dataset state dict, but cannot find latest "
        #         "dataset state dict file. Dataset statistics (e.g. feature mean and "
        #         "standard deviation) from the trainset."
        #     )

        if any((checkpoint, wandb_id, dataset_state_dict)):
            logger.info(
                f"Restoring training with checkpoint: {checkpoint}, wandb identifier: "
                f"{wandb_id}, and datasaet state dict: {dataset_state_dict}."
            )

        return checkpoint, wandb_id, dataset_state_dict


class SaveConfigCallback(LightningSaveConfigCallback):
    """
    Saves a LightningCLI config to the log_dir when training starts.

    Here, we add the functionalty to save the config to wandb if a wandb logger exists.
    """

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:

        super().setup(trainer, pl_module, stage)

        # save to wandb
        log_dir = trainer.log_dir
        config_path = os.path.join(log_dir, self.config_filename)

        wandb_logger = get_wandb_logger(trainer.logger)
        if wandb_logger is not None:
            save_files_to_wandb(wandb_logger.experiment, [config_path])

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:

        # TODO save running meta, e.g. git commit
        # save other files to wandb
        files_to_save = ["submit.sh", "train.py", "job_config.yaml"]
        wandb_logger = get_wandb_logger(trainer.logger)
        if wandb_logger is not None:
            save_files_to_wandb(wandb_logger.experiment, files_to_save)
