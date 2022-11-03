"""
Training script. Please do:

python train.py --help
python train.py # use the default config file specified below
python train.py --config configs/minimal.yaml

"""
import sys

import yaml
from loguru import logger

from matten.cli import SaveConfigCallback, mattenCLI
from matten.data.datamodule import BaseDataModule
from matten.model_factory.nequip_energy_model import EnergyModel
from matten.model_factory.segnn_classification import SEGNNClassification
from matten.model_factory.segnn_model import SEGNNModel
from matten.model_factory.segnn_model_paper import SEGNNModel as SEGNNModelPaper
from matten.model_factory.segnn_model_paper_tensor import SEGNNModel as SEGNNModelTensor
from matten.model_factory.tfn_hessian import TFNModel as TFNHessian
from matten.model_factory.tfn_scalar import TFNModel
from matten.model_factory.tfn_scalar_tensor import ScalarTensorModel
from matten.model_factory.tfn_scalar_tensor_global_feats import (
    ScalarTensorGlobalFeatsModel,
)
from matten.utils import to_path

CWD = to_path(__file__).parent


def main():

    logger.info("Start parsing experiment config and instantiating model!")

    # create cli
    cli = mattenCLI(
        # subclass_mode_model does not work well with `link_to` defined in cli
        # model_class=BaseModel,
        # subclass_mode_model=True,
        #
        ##
        # model_class=EnergyModel,
        # parser_kwargs={
        #     "default_config_files": [CWD.joinpath("configs", "minimal.yaml").as_posix()]
        # },
        ##
        # model_class=TFNHessian,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "tfn_hessian.yaml").as_posix()
        #     ]
        # },
        ##
        # model_class=SEGNNModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "minimal_segnn.yaml").as_posix()
        #     ]
        # },
        ##
        # model_class=SEGNNModelPaper,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "minimal_segnn_paper.yaml").as_posix()
        #     ]
        # },
        ##
        # model_class=SEGNNModelTensor,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "segnn_paper_tensor.yaml").as_posix()
        #     ]
        # },
        ##
        ##  To do classification, use the below two lines
        #
        # model_class=SEGNNClassification,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "minimal_classification.yaml").as_posix()
        #     ]
        # },
        #
        #
        ##
        # model_class=TFNModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "tfn_scalar.yaml").as_posix()
        #     ]
        # },
        ##
        # model_class=TFNModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "tfn_scalar_atom_feats.yaml").as_posix()
        #     ]
        # },
        ##
        # model_class=ScalarTensorGlobalFeatsModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "tfn_scalar_via_tensor.yaml").as_posix()
        #     ]
        # },
        ##
        # model_class=ScalarTensorGlobalFeatsModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "tfn_tensor.yaml").as_posix()
        #     ]
        # },
        ##
        # model_class=ScalarTensorGlobalFeatsModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "tfn_tensor_scalar.yaml").as_posix()
        #     ]
        # },
        ##
        # model_class=ScalarTensorGlobalFeatsModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath(
        #             "configs", "tfn_scalar_via_tensor_global_feats.yaml"
        #         ).as_posix()
        #     ]
        # },
        ##
        # model_class=ScalarTensorGlobalFeatsModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "tfn_tensor_global_feats.yaml").as_posix()
        #     ]
        # },
        ##
        # model_class=ScalarTensorGlobalFeatsModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "tfn_tensor_atom_feats.yaml").as_posix()
        #     ]
        # },
        ##
        # model_class=ScalarTensorGlobalFeatsModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "tfn_tensor_atom_global_feats.yaml").as_posix()
        #     ]
        # },
        ##
        # model_class=ScalarTensorGlobalFeatsModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath(
        #             "configs", "tfn_tensor_scalar_global_feats.yaml"
        #         ).as_posix()
        #     ]
        # },
        #
        #
        ###
        # special properties
        ###
        ##
        # model_class=ScalarTensorModel,
        # parser_kwargs={
        #     "default_config_files": [
        #         CWD.joinpath("configs", "tfn_minLC_scalar_atom_feats.yaml").as_posix()
        #     ]
        # },
        ##
        model_class=ScalarTensorModel,
        parser_kwargs={
            "default_config_files": [
                CWD.joinpath("configs", "tfn_minLC_vector_atom_feats.yaml").as_posix()
            ]
        },
        #
        #
        datamodule_class=BaseDataModule,
        subclass_mode_data=True,
        save_config_callback=SaveConfigCallback,
        save_config_filename="cli_config.yaml",
        save_config_overwrite=True,
        description="matten training command line tool",
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
