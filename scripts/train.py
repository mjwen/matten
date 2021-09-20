"""
Training script. Please do:

python train.py --help
python train.py # use the default config file specified below
python train.py --config configs/minimal.yaml

"""

from eigenn.data.datamodule import BaseDataModule
from eigenn.model_factory.nequip_energy_model import EnergyModel
from eigenn.utils import to_path
from scripts.cli import EigennCLI

CWD = to_path(__file__).parent


def main():

    # create cli
    cli = EigennCLI(
        # subclass_mode_model does not work well with `link_to` defined in cli
        # model_class=BaseModel,
        # subclass_mode_model=True,
        model_class=EnergyModel,
        datamodule_class=BaseDataModule,
        subclass_mode_data=True,
        parser_kwargs={
            "default_config_files": [CWD.joinpath("configs", "minimal.yaml").as_posix()]
        },
        run=False,
    )

    # TODO, we may want to jit the cli.model here

    # fit the model
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
