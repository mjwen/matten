"""
Training script. Please do:

python train.py --help
python train.py # use the default config file specified below
python train.py --config configs/minimal.yaml

"""
import sys

import yaml
from loguru import logger

from eigenn.cli import EigennCLI, SaveConfigCallback
from eigenn.data.datamodule import BaseDataModule
from eigenn.model_factory.atomic_tensor_model import AtomicTensorModel
from eigenn.model_factory.nequip_energy_model import EnergyModel
from eigenn.model_factory.segnn_model import SEGNNModel
from eigenn.utils import to_path

CWD = to_path(__file__).parent


def main():

    logger.info("Start parsing experiment config and instantiating model!")

    # create cli
    cli = EigennCLI(
        # subclass_mode_model does not work well with `link_to` defined in cli
        # model_class=BaseModel,
        # subclass_mode_model=True,
        model_class=EnergyModel,
        # model_class=AtomicTensorModel,
        # model_class=SEGNNModel,
        datamodule_class=BaseDataModule,
        subclass_mode_data=True,
        save_config_callback=SaveConfigCallback,
        save_config_filename="cli_config.yaml",
        save_config_overwrite=True,
        description="Eigenn training command line tool",
        parser_kwargs={
            "default_config_files": [CWD.joinpath("configs", "minimal.yaml").as_posix()]
        },
        run=False,
    )

    # print config to stderr
    print(file=sys.stderr, flush=True)  # flush buffer to avoid them entering config
    print("=" * 80, file=sys.stderr)
    print(
        "Configurations (also saved as cli_config.yaml):", end="\n\n", file=sys.stderr
    )
    # print(cli.parser.dump(cli.config, skip_none=False), file=sys.stderr)
    # the below line also prints out __default_config__
    yaml.dump(cli.config, stream=sys.stderr, sort_keys=True)
    print("=" * 80, end="\n\n\n", file=sys.stderr, flush=True)

    # TODO, we may want to jit the cli.model here

    # fit the model
    logger.info("Start training!")
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test the model
    if not cli.config["skip_test"]:
        logger.info("Start testing!")
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)

    # print path to best checkpoint
    logger.info(
        f"Best checkpoint path: {cli.trainer.checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    main()
