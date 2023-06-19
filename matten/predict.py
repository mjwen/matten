import shutil
from pathlib import Path

import torch
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.core.structure import Structure
from torch_geometric.loader import DataLoader

from matten.dataset.structure_scalar_tensor import TensorDatasetPrediction
from matten.model_factory.tfn_scalar_tensor import ScalarTensorModel
from matten.utils import CartesianTensorWrapper, yaml_load


def get_pretrained_model_info(identifier: str = "20230601"):
    """Get the pretrained model and its training config."""

    directory = Path(__file__).parent.parent.parent / "pretrained" / identifier

    model = ScalarTensorModel.load_from_checkpoint(
        checkpoint_path=directory.joinpath("model_final.ckpt").as_posix(),
        map_location="cpu",
    )

    config = yaml_load(directory / "config_final.yaml")

    return model, config


def get_data_loader(structure: list[Structure], config: dict, batch_size: int = 200):
    dataset = TensorDatasetPrediction(
        filename=None, r_cut=config["rcut"], structures=structure
    )
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
