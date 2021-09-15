"""
Command line parser class.

See pytorch_lightning docs for info on how to use this:
https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html
"""

import pl_bolts
import torch
from pytorch_lightning.utilities.cli import LightningCLI


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
