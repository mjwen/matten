"""
Training script for atomic property.

This is 99% percent the same as train.py, except that the AtomicTensorModel and
minimal_atomic.yaml are used. So, always make them in sync.


Please do:

python train_atomic.py --help
python train_atomic.py # use the default config file specified below
python train_atomic.py --config configs/minimal_atomic.yaml
"""
import sys

import yaml
from loguru import logger

from matten.cli import SaveConfigCallback, mattenCLI
from matten.data.datamodule import BaseDataModule
from matten.model_factory.atomic_tensor_model import AtomicTensorModel
from matten.utils import to_path

CWD = to_path(__file__).parent


def main():

    logger.info("Start parsing experiment config and instantiating model!")

    # create cli
    cli = mattenCLI(
        # subclass_mode_model does not work well with `link_to` defined in cli
        # model_class=BaseModel,
        # subclass_mode_model=True,
        model_class=AtomicTensorModel,
        datamodule_class=BaseDataModule,
        subclass_mode_data=True,
        save_config_callback=SaveConfigCallback,
        save_config_filename="cli_config.yaml",
        save_config_overwrite=True,
        description="matten training command line tool",
        parser_kwargs={
            "default_config_files": [
                CWD.joinpath("configs", "minimal_atomic.yaml").as_posix()
            ]
        },
        run=False,
    )

    # print the model
    print(file=sys.stderr, flush=True)  # flush buffer to avoid them entering config
    print("=" * 80, file=sys.stderr)
    print("Model:", end="\n\n", file=sys.stderr)
    print(cli.model)
    print("=" * 80, end="\n\n\n", file=sys.stderr, flush=True)

    # print config
    print(file=sys.stderr, flush=True)  # flush buffer to avoid them entering config
    print("=" * 80, file=sys.stderr)
    print("Configuration (also saved as cli_config.yaml):", end="\n\n", file=sys.stderr)
    # print(cli.parser.dump(cli.config, skip_none=False), file=sys.stderr)
    # the below line also prints out __default_config__
    yaml.dump(cli.config, stream=sys.stderr, sort_keys=True)
    print("=" * 80, end="\n\n\n", file=sys.stderr, flush=True)

    # TODO, we may want to jit the cli.model here

    # fit
    logger.info("Start training!")
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test
    if not cli.config["skip_test"]:
        logger.info("Start testing!")
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)

    # print path of best checkpoint
    logger.info(
        f"Best checkpoint path: {cli.trainer.checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    main()
