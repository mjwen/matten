"""
Test the hessian model to ensure:
- the diagonal blocks are symmetric
- two off-diagonal blocks (i,j) and (j,i) are transpose of each other
"""
from pathlib import Path

import torch
from e3nn.io import CartesianTensor

from eigenn.dataset.hessian import DataLoader, HessianDataset
from eigenn.model_factory.tfn_hessian import create_model

TESTFILE_DIR = Path(__file__).parents[1]


def get_model():
    hparams = {
        "species_embedding_dim": 16,
        # "species_embedding_irreps_out": "16x0e",
        "conv_layer_irreps": "32x0o + 32x0e + 16x1o + 16x1e",
        "irreps_edge_sh": "0e + 1o + 2e",
        "num_radial_basis": 8,
        "radial_basis_start": 0.0,
        "radial_basis_end": 3.0,
        # "radial_basis_r_cut": 4,
        "num_layers": 3,
        "reduce": "sum",
        "invariant_layers": 2,
        "invariant_neurons": 64,
        "average_num_neighbors": None,
        "nonlinearity_type": "gate",
        "conv_to_output_hidden_irreps_out": "0e + 1e + 1e + 1e + 2e",
        "normalization": "batch",
    }

    dataset_hyarmas = {"allowed_species": [6, 1, 8]}

    model = create_model(hparams, dataset_hyarmas)

    return model


def load_dataset(filename, root):
    dataset = HessianDataset(filename=filename, root=root, reuse=False)
    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False)
    return loader


def test_hessian_model():
    """
    - the diagonal blocks are symmetric
    - two off-diagonal blocks (i,j) and (j,i) are transpose of each other
    """
    filename = TESTFILE_DIR.joinpath("test_files", "hessian_two.xyz")

    loader = load_dataset(filename, root="/tmp")
    model = get_model()

    with torch.no_grad():
        for batch in loader:
            graphs = batch.tensor_property_to_dict()
            out = model(graphs)

            diag = out["hessian_diag"]
            off_diag = out["hessian_off_diag"]

            layout = batch.y["hessian_off_diag_layout"]

            break

    # diag
    convert = CartesianTensor("ij=ji")
    for x in diag:
        x = convert.to_cartesian(x)
        assert torch.allclose(x, x.T)

    # off diag
    convert = CartesianTensor("ij=ij")

    layout = layout.numpy().tolist()
    for k, x in enumerate(off_diag):
        x = convert.to_cartesian(x)

        # find index of transpose in off_diag
        i, j = layout[k]
        kk = layout.index([j, i])
        y = off_diag[kk]
        y = convert.to_cartesian(y)

        assert torch.allclose(x, y.T)
