"""
Test the hessian model to ensure:
- the diagonal blocks are symmetric
- two off-diagonal blocks (i,j) and (j,i) are transpose of each other
- test - basically using the formula: $f(Qx) = Qf(x)$, where $f$ is the model
  and $x$ are the coords in input molecules.
"""
import tempfile
from pathlib import Path

import pytest
import pytorch_lightning
import torch
from ase.io.extxyz import read_extxyz, write_extxyz
from e3nn import o3
from e3nn.io import CartesianTensor

from eigenn.dataset.hessian import DataLoader, HessianDataset, get_dataset_statistics
from eigenn.model_factory.tfn_hessian import create_model

TESTFILE_DIR = Path(__file__).parents[1]


def get_model():
    hparams = {
        "species_embedding_dim": 16,
        # "species_embedding_irreps_out": "16x0e",
        "conv_layer_irreps": "32x0o + 32x0e + 16x1o + 16x1e + 8x2o + 8x2e",
        "irreps_edge_sh": "0e + 1o + 2e",
        "num_radial_basis": 8,
        "radial_basis_start": 0.5,
        "radial_basis_end": 2.0,
        # "radial_basis_r_cut": 4,
        "num_layers": 2,
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


def load_dataset(filename, root, normalize_target: bool):
    dataset = HessianDataset(
        filename=filename,
        root=root,
        reuse=False,
        compute_dataset_statistics=get_dataset_statistics,
        normalize_target=normalize_target,
    )
    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False)
    return loader


@pytest.mark.parametrize("normalize_target", [False, True])
def test_hessian_symmetric(normalize_target):
    """
    - the diagonal blocks are symmetric
    - two off-diagonal blocks (i,j) and (j,i) are transpose of each other
    """
    pytorch_lightning.seed_everything(35)

    filename = TESTFILE_DIR.joinpath("test_files", "hessian_two.xyz")

    loader = load_dataset(filename, root="/tmp", normalize_target=normalize_target)
    model = get_model()

    with torch.no_grad():
        for batch in loader:
            graphs = batch.tensor_property_to_dict()
            out = model(graphs)

            diag = out["hessian_ii_block"]
            off_diag = out["hessian_ij_block"]

            layout = batch.y["hessian_off_diag_layout"]

            break

    # diag
    convert = CartesianTensor("ij=ji")
    for x in diag:
        x = convert.to_cartesian(x)
        assert torch.allclose(x, x.T, rtol=1e-2)

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
        assert torch.allclose(x, y.T, rtol=1e-2)


@pytest.mark.parametrize("normalize_target", [False, True])
def test_hesssian_equivariance(normalize_target):
    pytorch_lightning.seed_everything(35)

    model = get_model()

    filename = TESTFILE_DIR.joinpath("test_files", "hessian_one.xyz")
    loader = load_dataset(filename, root="/tmp", normalize_target=normalize_target)

    # filename = "ani1_CHO_0-1000_hessian_one.xyz"
    # root = "/Users/mjwen/Documents/Dataset/xiaowei_hessian"
    #
    # loader = load_dataset(filename=filename, root=root)

    #
    # generate a temp file with rotated atoms, and get it data loader
    #
    # original mol
    with open(filename, "r") as f:
        atoms = read_extxyz(f)
        atoms = next(atoms)

    # transformation matrix
    torch.manual_seed(35)
    Q = o3.rand_matrix()

    # update positions
    new_positions = atoms.positions @ Q.numpy().T
    atoms.positions = new_positions

    # write it out
    filename2 = "hessian_one-rotated.xyz"
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(Path(tmpdir).joinpath(filename2), "w") as f:
            write_extxyz(f, atoms)
        loader_rotated = load_dataset(
            filename=filename2, root=tmpdir, normalize_target=normalize_target
        )

    def get_result(model, loader):
        model.eval()

        with torch.no_grad():
            for batch in loader:
                graphs = batch.tensor_property_to_dict()
                out = model(graphs)

                # return out['hessian_ii_block']
                return out["hessian_ij_block"]

    pred = get_result(model, loader)
    pred_rotated = get_result(model, loader_rotated)

    convert = CartesianTensor("ij=ij")
    pred_cart = convert.to_cartesian(pred)
    pred_rotated_cart = convert.to_cartesian(pred_rotated)

    for h, h_rotated in zip(pred_cart, pred_rotated_cart):
        h = Q @ h @ Q.T
        assert torch.allclose(h, h_rotated, rtol=1e-1)
