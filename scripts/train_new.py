import sys
from pathlib import Path

import yaml
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.cli import instantiate_class as lit_instantiate_class

from matten.dataset.structure_scalar_tensor import TensorDataModule
from matten.model_factory.task import TensorRegressionTask
from matten.model_factory.tfn_scalar_tensor import ScalarTensorModel


def instantiate_class(d: dict | list):
    args = tuple()  # no positional args
    if isinstance(d, dict):
        return lit_instantiate_class(args, d)
    elif isinstance(d, list):
        return [lit_instantiate_class(args, x) for x in d]
    else:
        raise ValueError(f"Cannot instantiate class from {d}")


def get_args(path: Path):
    """Get the arguments from the config file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_model_info(config, model):
    # print the model
    print(file=sys.stderr, flush=True)
    print("=" * 80, file=sys.stderr)
    print("Model:", end="\n\n", file=sys.stderr)
    print(model)
    print("=" * 80, end="\n\n\n", file=sys.stderr, flush=True)

    # print config
    print(file=sys.stderr, flush=True)
    print("=" * 80, file=sys.stderr)
    print("Configuration (also saved as cli_config.yaml):", end="\n\n", file=sys.stderr)

    # print(cli.parser.dump(cli.config, skip_none=False), file=sys.stderr)
    # the below line also prints out __default_config__

    yaml.dump(config, stream=sys.stderr, sort_keys=True)
    print("=" * 80, end="\n\n\n", file=sys.stderr, flush=True)


def main(config: dict, project_name="matten_project"):
    dm = TensorDataModule(**config["data"])
    dm.prepare_data()
    dm.setup()

    model = ScalarTensorModel(
        tasks=TensorRegressionTask(name="elastic_tensor_full"),
        backbone_hparams=config["model"],
        dataset_hparams=dm.get_to_model_info(),
        optimizer_hparams=config["optimizer"],
        lr_scheduler_hparams=config["lr_scheduler"],
    )

    save_model_info(config, model)

    try:
        callbacks = instantiate_class(config["trainer"].pop("callbacks"))
        lit_logger = instantiate_class(config["trainer"].pop("logger"))
    except KeyError:
        callbacks = None
        lit_logger = None

    trainer = Trainer(
        callbacks=callbacks,
        logger=lit_logger,
        **config["trainer"],
    )

    logger.info("Start training!")
    trainer.fit(model, datamodule=dm)

    # test
    logger.info("Start testing!")
    trainer.test(ckpt_path="best", datamodule=dm)

    # print path of best checkpoint
    logger.info(f"Best checkpoint path: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    config_file = Path(__file__).parent / "configs" / "tfn_tensor_new.yaml"
    config = get_args(config_file)

    main(config)
