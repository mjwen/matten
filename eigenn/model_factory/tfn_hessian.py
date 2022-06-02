"""
Species embedding using torch.nn.Embedding. As a results, NOTE_ATTRS are learnable and
it is the same as NODE_FEATURES in the first layer. NOTE, they are the same only at the
first layer. In the model, NODE_FEATURES will be updated, but NODE_ATTRS are not.

The original NequIP uses ONE-hot embedding for NODE_ATTRS, and then use a linear layer
to map it to NODE_FEATURES.

For large number of species, we'd better use the SpeciesEmbedding one to minimize the
number of params.
"""


from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from e3nn.io import CartesianTensor
from torch import Tensor

from eigenn.dataset.hessian import (
    HessianTargetTransform,
    combine_on_off_diagonal_blocks,
)
from eigenn.model.model import ModelForPyGData
from eigenn.model.task import CanonicalRegressionTask
from eigenn.model_factory.utils import create_sequential_module
from eigenn.nn._nequip import RadialBasisEdgeEncoding, SphericalHarmonicEdgeAttrs
from eigenn.nn.embedding import EdgeLengthEmbedding, SpeciesEmbedding
from eigenn.nn.nodewise import NodewiseLinear
from eigenn.nn.readout import IrrepsToIrrepsHessian
from eigenn.nn.tfn import PointConv, PointConvWithActivation


class TFNModel(ModelForPyGData):
    def init_backbone(
        self,
        backbone_hparams: Dict[str, Any],
        dataset_hparams: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        backbone = create_model(backbone_hparams, dataset_hparams)

        # TODO
        # converter to convert irreps tensor to cartesian tensor
        # we abuse it to set them here and use in decode.
        # It seems the .to_cartesian() method is  causing memory leaking problem
        # (need further conformation): each time it needs to create a new copy of
        # reduced tensor product to obtain the transformation matrix
        self.cart_diag = CartesianTensor("ij=ji")
        self.cart_off_diag = CartesianTensor("ij=ij")
        self.cart_diag_rtp = self.cart_diag.reduced_tensor_products()
        self.cart_off_diag_rtp = self.cart_off_diag.reduced_tensor_products()

        return backbone

    def decode(self, model_input) -> Dict[str, Tensor]:
        out = self.backbone(model_input)

        preds = {
            "hessian_diag": out["hessian_ii_block"],
            "hessian_off_diag": out["hessian_ij_block"],
        }

        # transform from normalized space to original space for cartesian tensor loss
        diag = self.tasks["hessian_eigval"].transform_pred_loss(
            preds["hessian_diag"], mode="diag"
        )
        off_diag = self.tasks["hessian_eigval"].transform_pred_loss(
            preds["hessian_off_diag"], mode="off_diag"
        )

        ptr = model_input["ptr"]
        natoms = [j - i for i, j in zip(ptr[:-1], ptr[1:])]

        # convert irreps to cartesian tensor
        cart_diag = self.cart_diag.to_cartesian(
            diag, self.cart_diag_rtp.to(diag.device)
        )
        cart_off_diag = self.cart_off_diag.to_cartesian(
            off_diag, self.cart_off_diag_rtp.to(off_diag.device)
        )
        eigval = get_eigval_pred(cart_diag, cart_off_diag, natoms)

        # the eigen value is obtained in the original cartesian space
        preds["hessian_eigval"] = eigval

        return preds

    def preprocess_batch(self, batch):
        """
        Overwrite the default one to get hessian_natoms and hessian_off_diag_layout to
        label, so that we can use it to properly compute loss
        """
        graphs = batch
        graphs = graphs.to(self.device)  # lightning cannot move graphs to gpu

        # task labels :w
        labels = {
            name: graphs.y[name]
            for name in [
                "hessian_diag",
                "hessian_off_diag",
                "hessian_natoms",
                "hessian_eigval",
            ]
        }

        # convert graphs to a dict to use NequIP stuff
        graphs = graphs.tensor_property_to_dict()

        return graphs, labels

    def compute_loss(
        self, preds: Dict[str, Tensor], labels: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Tensor]:

        name_diag = "hessian_diag"
        loss_diag = get_loss(
            self.loss_fns[name_diag],
            preds[name_diag],
            labels[name_diag],
            labels["hessian_natoms"],
            mode=name_diag,
        )

        name_off = "hessian_off_diag"
        loss_off = get_loss(
            self.loss_fns[name_off],
            preds[name_off],
            labels[name_off],
            labels["hessian_natoms"],
            mode=name_off,
        )

        # eigen value loss
        name_eigval = "hessian_eigval"
        loss_eigval = get_loss_eigval(
            self.loss_fns[name_eigval],
            preds[name_eigval],
            labels[name_eigval],
            labels["hessian_natoms"],
        )

        loss_individual = {
            name_diag: loss_diag,
            name_off: loss_off,
            name_eigval: loss_eigval,
        }

        loss_total = (
            self.tasks[name_diag].loss_weight * loss_diag
            + self.tasks[name_off].loss_weight * loss_off
            + self.tasks[name_eigval].loss_weight * loss_eigval
        )

        return loss_individual, loss_total

    def transform_prediction(self, preds: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # TODO # Here, we abuse the normalizer in tasks.
        #  But in general, we should probably create the normalizer for its own.

        normalizer = self.tasks["hessian_diag"].normalizer

        task_name_diag = "hessian_diag"
        h_diag = normalizer.inverse(preds[task_name_diag], mode="diag")

        task_name_off_diag = "hessian_off_diag"
        h_off_diag = normalizer.inverse(preds[task_name_off_diag], mode="off_diag")

        return {task_name_diag: h_diag, task_name_off_diag: h_off_diag}

    # NOTE, do not need this because we can set normalize_target = False of a dataset at
    # predicting time
    # def transform_target(self, target: Dict[str, Tensor]) -> Dict[str, Tensor]:
    #     return self.transform_prediction(target)

    def transform_target(self, target: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.transform_prediction(target)


def get_loss(loss_fn, p, t, natoms, mode):
    """
    Args:
        mode: `hessian_diag` | `hessian_off_diag`
    """

    if mode == "hessian_diag":
        if isinstance(loss_fn, (torch.nn.MSELoss, TargetScaledMSE)):
            # for each molecule, there are n diagonal components;
            # repeat each n times
            scale = torch.sqrt(torch.repeat_interleave(natoms, natoms, dim=0))
        elif isinstance(loss_fn, torch.nn.L1Loss):
            scale = torch.repeat_interleave(natoms, natoms, dim=0)
        else:
            raise ValueError

    elif mode == "hessian_off_diag":
        if isinstance(loss_fn, (torch.nn.MSELoss, TargetScaledMSE)):
            # for each molecule, there are n**2 - n off-diagonal blocks
            scale = torch.sqrt(
                torch.repeat_interleave(natoms, natoms * natoms - natoms, dim=0)
            )
        elif isinstance(loss_fn, torch.nn.L1Loss):
            scale = torch.repeat_interleave(natoms, natoms * natoms - natoms, dim=0)
        else:
            raise ValueError
    else:
        raise ValueError("Not supported loss type")

    # make scale have the same shape of pred and target, except for the highest
    # dim. e.g. p has shape (108, 6) and scale has shape (108,), which will
    # make scale of shape(108, 1)
    extra_shape = [1] * (len(p.shape) - 1)
    scale = scale.reshape(-1, *extra_shape)

    p = p / scale
    t = t / scale
    loss = loss_fn(p, t)

    return loss


def get_eigval_pred(p_diag, p_off_diag, natoms: List[int]):
    """
    Compute model prediction of eigval.

    Args:
        p_diag: shape[sum(Ni), 3, 3], where Ni is the number of atoms in mol i
        p_off_diag: shape[sum(Ni*Ni-Ni), 3, 3]

    Returns:
        eigval of a set of mols
    """

    p_diag_by_mol = torch.split(p_diag, natoms)
    p_off_diag_by_mol = torch.split(p_off_diag, [n**2 - n for n in natoms])

    eigval = []
    for diag, off_diag in zip(p_diag_by_mol, p_off_diag_by_mol):
        H = combine_on_off_diagonal_blocks(diag, off_diag)  # [N*3, N*3]

        v, _ = torch.linalg.eigh(H)
        eigval.append(v)

    # predicted eigen values
    eigval = torch.cat(eigval)

    return eigval


def get_loss_eigval(loss_fn, pred, target, natoms):
    """
    Compute a loss over eigenvalues
    """

    if isinstance(loss_fn, (torch.nn.MSELoss, TargetScaledMSE)):
        # for each molecule, there are 3N eigenvalues
        # NOTE repeat each 3N times, not N times
        scale = torch.sqrt(torch.repeat_interleave(natoms, 3 * natoms, dim=0))
    elif isinstance(loss_fn, torch.nn.L1Loss):
        scale = torch.repeat_interleave(natoms, 3 * natoms, dim=0)
    else:
        raise ValueError

    # scale the prediction and target
    eigval = pred / scale
    target = target / scale
    loss = loss_fn(eigval, target)

    return loss


class HessianRegressionTask(CanonicalRegressionTask):
    """
    Only inverse transform prediction and target in metric.

    Note, in HessianTargetTransform, the target are forward transformed.

    Args:
        name: name of the task. Values with this key in model prediction dict and
            target dict will be used for loss and metrics computation.
        mode: {'diag', 'off_diag'}
    """

    def __init__(
        self,
        name: str,
        mode: str,
        loss_weight: float = 1.0,
        dataset_statistics_path: Union[str, Path] = "dataset_statistics.pt",
        normalizer_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(name, loss_weight=loss_weight)

        self.mode = mode
        if normalizer_kwargs is None:
            normalizer_kwargs = {}
        self.normalizer = HessianTargetTransform(
            dataset_statistics_path, **normalizer_kwargs
        )

    def transform_target_loss(self, t: Tensor) -> Tensor:
        return t

    def transform_pred_loss(self, t: Tensor) -> Tensor:
        return t

    def transform_target_metric(self, t: Tensor) -> Tensor:
        return self.normalizer.inverse(t, mode=self.mode)

    def transform_pred_metric(self, t: Tensor) -> Tensor:
        return self.normalizer.inverse(t, mode=self.mode)


class ScaledHessianRegressionTask(HessianRegressionTask):
    def __init__(
        self,
        name: str,
        mode: str,
        loss_weight: float = 1.0,
        dataset_statistics_path: Union[str, Path] = "dataset_statistics.pt",
        eps: float = 0.01,
        normalizer_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(
            name, mode, loss_weight, dataset_statistics_path, normalizer_kwargs
        )
        self.eps = eps

    def init_loss(self):
        return TargetScaledMSE(eps=self.eps)


class HessianEigvalTask(CanonicalRegressionTask):
    """
    Metric and tasks for eigen values.

    Note, the target values are the eigen values of the 3N x 3N hessian matrix.
    We need to transform the prediction back to 3N x 3N and then obtain the eigen
    values. This is done in model,decode()
    """

    def __init__(
        self,
        name: str,
        loss_weight: float = 1.0,
        dataset_statistics_path: Union[str, Path] = "dataset_statistics.pt",
        normalizer_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(name, loss_weight=loss_weight)

        if normalizer_kwargs is None:
            normalizer_kwargs = {}
        self.normalizer = HessianTargetTransform(
            dataset_statistics_path, **normalizer_kwargs
        )

    def transform_target_loss(self, t: Tensor) -> Tensor:
        return t

    def transform_pred_loss(self, t: Tensor, mode: str) -> Tensor:
        return self.normalizer.inverse(t, mode=mode)

    def transform_target_metric(self, t: Tensor) -> Tensor:
        return t

    # we will use self.transform_pred_loss to transform prediction in decoder,
    # so we do not need an additional transformation here
    def transform_pred_metric(self, t: Tensor) -> Tensor:
        return t


class TargetScaledMSE(torch.nn.Module):
    """
    Mean squared loss and scale each term by target value.

    Purpose: to deal with many zero hessian entries.

    Args:
        esp: Smallest scale value to be multiplied.
    """

    def __init__(self, eps: float = 0.01):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        scale = torch.abs(target)
        scale[scale < self.eps] = self.eps

        loss = scale * (pred - target) ** 2
        loss = loss.mean()

        return loss


def create_model(hparams: Dict[str, Any], dataset_hparams):
    """
    The actual function to create the model.
    """

    # ===== input embedding layers =====
    layers = {
        "one_hot": (
            SpeciesEmbedding,
            {
                "embedding_dim": hparams["species_embedding_dim"],
                "allowed_species": dataset_hparams["allowed_species"],
            },
        ),
        "spharm_edges": (
            SphericalHarmonicEdgeAttrs,
            {"irreps_edge_sh": hparams["irreps_edge_sh"]},
        ),
        # "radial_basis": (
        #     RadialBasisEdgeEncoding,
        #     {
        #         "basis_kwargs": {
        #             "num_basis": hparams["num_radial_basis"],
        #             "r_max": hparams["radial_basis_r_cut"],
        #         },
        #         "cutoff_kwargs": {"r_max": hparams["radial_basis_r_cut"]},
        #     },
        # ),
        "radial_basis": (
            EdgeLengthEmbedding,
            {
                "num_basis": hparams["num_radial_basis"],
                "start": hparams["radial_basis_start"],
                "end": hparams["radial_basis_end"],
            },
        ),
        # This embed features is not necessary any more when we change OneHotEmbedding
        # to SpeciesEmbedding.
        # SpeciesEmbedding and OneHotEmbedding+AtowiseLinear have the same effects:
        # we just need to set embedding_dim (e.g. 16) of SpeciesEmbedding to be
        # corresponding to  `irreps_out` (e.g. 16x0e) of AtomwiseLinear.
        # To be less error prone, we use SpeciesEmbedding.
        #
        # NOTE, there is some subtle difference:
        # - In OneHotEmbedding+AtowiseLinear, NODE_ATTRS is set to the one-hot
        # encoding, which is fixed throughout the model, while NODE_FEATURES is the
        # embedding, which ls learnable.
        # - In SpeciesEmbedding, both NODE_ATTRS and NODE_FEATURES are set to the
        # learnable embedding.
        # - The only use of NODE_ATTRS  in nequip is in InteractionBlock (which is in
        # ConvNetLayer), where it is used for the self-interaction layer.
        # So, if self-interaction is enabled, these two modes will give different
        # results.
        # - A side note, the self-interaction layer in nequip is different from the one
        # used in TFN paper, where self-interaction is on NODE_FEATURES only. So,
        # if we use SpeciesEmbedding, we agree with the original TFN paper.
        #
        # We can simply generalize both to see which one works better.
        #
        ##
        # -- Embed features --
        # "chemical_embedding": (AtomwiseLinear, {}),  # a linear layer on node_feats
    }

    # ===== convnet layers =====
    # insertion preserves order

    num_neigh = hparams["average_num_neighbors"]
    if isinstance(num_neigh, str) and num_neigh.lower() == "auto":
        num_neigh = dataset_hparams["average_num_neighbors"]

    for i in range(hparams["num_layers"]):
        layers[f"layer{i}_convnet"] = (
            #
            # ConvNetLayer,
            # {
            #     "feature_irreps_hidden": hparams["conv_layer_irreps"],
            #     "nonlinearity_type": hparams["nonlinearity_type"],
            #     "resnet": hparams["resnet"],
            #     "convolution_kwargs": {
            #         "invariant_layers": hparams["invariant_layers"],
            #         "invariant_neurons": hparams["invariant_neurons"],
            #         "avg_num_neighbors": num_neigh,
            #         "use_sc": hparams["use_sc"],
            #     },
            # },
            #
            # MessagePassing,
            # {
            #     "conv_layer_irreps": hparams["conv_layer_irreps"],
            #     "activation_type": hparams["nonlinearity_type"],
            #     "use_resnet": hparams["resnet"],
            #     "conv": PointConv,
            #     "conv_kwargs": {
            #         "fc_num_hidden_layers": hparams["invariant_layers"],
            #         "fc_hidden_size": hparams["invariant_neurons"],
            #         "avg_num_neighbors": num_neigh,
            #         "use_self_connection": hparams["use_sc"],
            #     },
            #     # # transformer conv
            #     # "conv": TransformerConv,
            #     # "conv_kwargs": {
            #     #     "irreps_query_and_key": hparams["conv_layer_irreps"],
            #     #     "r_max": hparams["radial_basis_r_cut"],
            #     #     "fc_num_hidden_layers": hparams["invariant_layers"],
            #     #     "fc_hidden_size": hparams["invariant_neurons"],
            #     #     "avg_num_neighbors": num_neigh,
            #     #     "use_self_connection": hparams["use_sc"],
            #     # },
            # },
            #
            PointConvWithActivation,
            {
                "conv_layer_irreps": hparams["conv_layer_irreps"],
                "activation_type": hparams["nonlinearity_type"],
                "fc_num_hidden_layers": hparams["invariant_layers"],
                "fc_hidden_size": hparams["invariant_neurons"],
                "avg_num_neighbors": num_neigh,
                "normalization": hparams["normalization"],
            },
        )

    layers["conv_layer_last"] = (
        PointConv,
        {
            "conv_layer_irreps": hparams["conv_layer_irreps"],
            "fc_num_hidden_layers": hparams["invariant_layers"],
            "fc_hidden_size": hparams["invariant_neurons"],
            "avg_num_neighbors": num_neigh,
        },
    )

    # one more linear layer before assembling output
    layers["conv_to_output_hidden"] = (
        NodewiseLinear,
        {"irreps_out": hparams["conv_to_output_hidden_irreps_out"]},
    )

    # output layer
    # outfield hard-coded IrrepsToIrrepsHessian, they are hessian_ii_block and
    # hessian_ij_block
    layers["output"] = (IrrepsToIrrepsHessian, {"out_field": None})

    # create the sequential model
    model = create_sequential_module(
        modules=OrderedDict(layers), use_kwargs_irreps_in=True
    )

    return model


if __name__ == "__main__":
    from eigenn.log import set_logger

    set_logger("DEBUG")

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
        "conv_to_output_hidden_irreps_out": "16x0e + 8x1e + 4x2e",
        "normalization": "batch",
    }

    dataset_hyarmas = {"allowed_species": [6, 1, 8]}
    create_model(hparams, dataset_hyarmas)
