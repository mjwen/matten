import shutil
from pathlib import Path

import torch
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.core.structure import Structure
from torch_geometric.loader import DataLoader

from matten.dataset.structure_scalar_tensor import TensorDatasetPrediction
from matten.log import set_logger
from matten.model_factory.tfn_scalar_tensor import ScalarTensorModel
from matten.utils import CartesianTensorWrapper, yaml_load


def get_pretrained_model_dir(identifier: str = "20230601"):
    return Path(__file__).parent.parent / "pretrained" / identifier


def get_pretrained_model(
    identifier: str = "20230601", checkpoint: str = "model_final.ckpt"
):
    """Get the pretrained model and its training config."""

    directory = get_pretrained_model_dir(identifier)
    model = ScalarTensorModel.load_from_checkpoint(
        checkpoint_path=directory.joinpath(checkpoint).as_posix(),
        map_location="cpu",
    )
    return model


def get_pretrained_config(
    identifier: str = "20230601", config_filename: str = "config_final.yaml"
):
    directory = get_pretrained_model_dir(identifier)
    config = yaml_load(directory / config_filename)

    return config


def get_data_loader(structures: list[Structure], config: dict, batch_size: int = 200):
    # config contains info for dataset and data loader, we only use the dataset part,
    # and adjust some parameters
    config = config["data"].copy()
    for k in [
        "loader_kwargs",
        "root",
        "trainset_filename",
        "valset_filename",
        "testset_filename",
        "compute_dataset_statistics",
    ]:
        config.pop(k)

    r_cut = config.pop("r_cut")
    config["dataset_statistics_fn"] = None

    dataset = TensorDatasetPrediction(
        # The filename is not input filename, but name for processed data.
        # The input is from `structures`.
        # The filename extension does not matter, it will be replaced by `_data.pt`
        filename="./data_for_prediction.txt",
        r_cut=r_cut,
        structures=structures,
        **config,
    )

    # remove the saved data, which is not needed for prediction
    # (can be good for training)
    shutil.rmtree(dataset.processed_dirname)

    return DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)


def load_dataset(
    filename,
    root,
    # global_featurizer=None,
    atom_featurizer=None,
    reuse=False,
    batch_size=200,
):
    ARGS = CONFIG["data"]["init_args"]

    ARGS["trainset_filename"] = filename
    ARGS["valset_filename"] = filename
    ARGS["testset_filename"] = filename
    # ARGS['global_featurizer'] = global_featurizer
    ARGS["atom_featurizer"] = atom_featurizer

    ARGS["root"] = root
    ARGS["reuse"] = reuse
    ARGS["loader_kwargs"]["shuffle"] = False
    ARGS["loader_kwargs"]["batch_size"] = batch_size

    # should reuse the dataset statistics
    ARGS["compute_dataset_statistics"] = False

    na = "normalize_atom_features"
    ng = "normalize_global_features"
    nt = "normalize_tensor_target"
    ns = "normalize_scalar_targets"
    if (
        (na in ARGS and ARGS[na])
        or (ng in ARGS and ARGS[ng])
        or (nt in ARGS and ARGS[nt])
        or (ns in ARGS and ARGS[ns])
    ):
        shutil.copy(f"./{MODEL}/dataset_statistics.pt", "dataset_statistics.pt")

    dm = TensorDataModule(**ARGS)
    dm.prepare_data()
    dm.setup()

    loader = dm.train_dataloader()

    return loader


# function to evaluate on some metrics


def evaluate(
    model,
    loader,
    space="irreps",
    target_name="elastic_tensor_full",
):
    """
    Args:
        target_name
        space: irreps | cartesian. output space.
    """

    converter = CartesianTensorWrapper(
        CONFIG["data"]["init_args"]["tensor_target_formula"]
    )

    predictions = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            preds, labels = model(batch, mode=None, task_name=target_name)
            p = preds[target_name]
            t = labels[target_name]

            if (
                space == "cartesian"
                and CONFIG["data"]["init_args"]["tensor_target_format"] == "irreps"
            ):
                p = converter.to_cartesian(p)
                t = converter.to_cartesian(t)
            if (
                space == "irreps"
                and CONFIG["data"]["init_args"]["tensor_target_format"] == "cartesian"
            ):
                p = converter.from_cartesian(p)
                t = converter.from_cartesian(t)

            predictions.extend(p)
            targets.extend(t)

    return torch.stack(predictions), torch.stack(targets)


def predict(
    structures: Structure | list[Structure],
    model_path: Path = None,
    config_path: Path = None,
    logger_level: str = "ERROR",
) -> ElasticTensor | list[ElasticTensor]:
    """
    Predict the property of a structure or a list of structures.

    Args:
        structures: a structure or a list of structures to predict.
        model_path: path to the model checkpoint.
        config_path: path to the config file used to train the model.

    Returns:
        Predicted elastic tensor(s).
    """
    set_logger("ERROR")


if __name__ == "__main__":
    set_logger("ERROR")

    config = get_pretrained_config()

    struct = Structure(
        lattice=[
            [3.348898, 0.0, 1.933487],
            [1.116299, 3.157372, 1.933487],
            [0.0, 0.0, 3.866975],
        ],
        species=["Si", "Si"],
        coords=[[0.25, 0.25, 0.25], [0, 0, 0]],
    )
    structures = [struct, struct]

    get_data_loader(structures, config, batch_size=2)
