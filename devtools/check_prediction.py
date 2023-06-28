"""
Check the prediction script get the correct results as in fitting.

Using the same dataset as in fitting, we check the prediction script recovers
the MAE.
"""
import pandas as pd
import torch
from pymatgen.core import Structure

from matten.predict import predict
from matten.utils import CartesianTensorWrapper


def get_data():
    filename = (
        "/Users/mjwen.admin/Packages/matten_analysis/matten_analysis"
        "/dataset/elastic_tensor/20230322/crystal_elasticity_filtered_test.json"
    )
    df = pd.read_json(filename, orient="split")

    structures = df["structure"].apply(lambda x: Structure.from_dict(x)).tolist()

    targets = df["elastic_tensor_full"].tolist()
    targets = [torch.tensor(t) for t in targets]

    return structures, targets


def get_spherical_tensor(t):
    """Convert a Cartesian tensor to a spherical tensor."""
    converter = CartesianTensorWrapper("ijkl=jikl=klij")
    spherical = converter.from_cartesian(t)
    return spherical


def mae(predictions: list[torch.Tensor], targets: list[torch.Tensor]):
    predictions = torch.stack(predictions)
    targets = torch.stack(targets)
    return torch.mean(torch.abs(predictions - targets))


if __name__ == "__main__":
    structures, targets = get_data()
    predictions = predict(structures)

    targets = [get_spherical_tensor(t) for t in targets]

    predictions = [torch.tensor(t) for t in predictions]
    predictions = [get_spherical_tensor(t) for t in predictions]

    print(mae(predictions, targets))
