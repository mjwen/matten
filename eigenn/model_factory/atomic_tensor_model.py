"""
A model outputs atomic tensors (i.e. output a tensor on each atomic site), e.g.
symmetric 2nd order NMR tensor.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from e3nn.io import CartesianTensor
from torch import Tensor

from eigenn.model.model import ModelForPyGData
from eigenn.model_factory.utils import create_sequential_module
from eigenn.nn._nequip import RadialBasisEdgeEncoding, SphericalHarmonicEdgeAttrs
from eigenn.nn.embedding import SpeciesEmbedding
from eigenn.nn.nodewise import NodewiseLinear, NodewiseSelect
from eigenn.nn.point_conv import PointConvMessagePassing
from eigenn.nn.readout import IrrepsToCartesianTensor

OUT_FIELD_SELECTED = "tensor_output"


# TODO, we can weigh different irreps (e.g., 0e, 1o, and 2e) differently, and that can
#  be set in the Task (parameters from config passed from hparams)
class AtomicTensorModel(ModelForPyGData):
    """
    A model outputs tensors, e.g. symmetric 2nd order stress tensor.
    """

    def init_backbone(
        self,
        backbone_hparams: Dict[str, Any],
        dataset_hparams: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        backbone = create_model(backbone_hparams, dataset_hparams)
        return backbone

    def decode(self, model_input) -> Dict[str, Tensor]:
        out = self.backbone(model_input)

        # current we only support one task, so 0 is the name
        task_name = list(self.tasks.keys())[0]

        preds = {task_name: out[OUT_FIELD_SELECTED]}

        return preds


def create_model(hparams: Dict[str, Any], dataset_hparams):
    """
    The actual function to create the model.
    """

    # ===== embedding layers =====
    layers = {
        # -- Encode --
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
        "radial_basis": (
            RadialBasisEdgeEncoding,
            {
                "basis_kwargs": {
                    "num_basis": hparams["num_radial_basis"],
                    "r_max": hparams["radial_basis_r_cut"],
                },
                "cutoff_kwargs": {"r_max": hparams["radial_basis_r_cut"]},
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

    for i in range(hparams["num_layers"]):
        layers[f"layer{i}_convnet"] = (
            # MessagePassing,
            # {
            #     "conv_layer_irreps": hparams["conv_layer_irreps"],
            #     "activation_type": hparams["nonlinearity_type"],
            #     "use_resnet": hparams["resnet"],
            #     "conv": PointConv,
            #     # "conv": SEGNNConv,
            #     "conv_kwargs": {
            #         "fc_num_hidden_layers": hparams["invariant_layers"],
            #         "fc_hidden_size": hparams["invariant_neurons"],
            #         "avg_num_neighbors": hparams["avg_num_neighbors"],
            #         "use_self_connection": hparams["use_sc"],
            #     },
            # },
            #
            PointConvMessagePassing,
            {
                "conv_layer_irreps": hparams["conv_layer_irreps"],
                "activation_type": hparams["nonlinearity_type"],
                "use_resnet": hparams["resnet"],
                "message_kwargs": {
                    "fc_num_hidden_layers": hparams["invariant_layers"],
                    "fc_hidden_size": hparams["invariant_neurons"],
                },
                "update_kwargs": {
                    "avg_num_neighbors": hparams["avg_num_neighbors"],
                    "use_self_connection": hparams["use_sc"],
                },
            },
        )

    #
    # determining output formula and irreps
    #

    # output cartesian tensor
    if "output_formula" in hparams:

        formula = hparams["output_formula"]  # e.g. ij=ji for a general 2D tensor

        # get irreps for the formula (CartesisanTensor is a subclass of Irreps)
        output_irreps = CartesianTensor(formula)

    # output irreps tensor
    else:
        formula = None

        if "output_irreps" in hparams:
            output_irreps = hparams["output_irreps"]
        else:
            # default to scalar
            output_irreps = "1x0e"

    # data keys
    OUT_FIELD_ATOM = "atomic_output"
    NODE_MASKS = "node_masks"

    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the
            #  last layer of convnet
            # -- output block --
            "conv_to_output_hidden": (
                NodewiseLinear,
                {"irreps_out": hparams["conv_to_output_hidden_irreps_out"]},
            ),
            "output_hidden_to_tensor": (
                NodewiseLinear,
                {"irreps_out": output_irreps, "out_field": OUT_FIELD_ATOM},
            ),
        }
    )

    # select atomic tensor for prediction
    layers["output_irreps_tensor"] = (
        NodewiseSelect,
        {
            "field": OUT_FIELD_ATOM,
            "out_field": OUT_FIELD_SELECTED,
            "mask_field": NODE_MASKS,
        },
    )

    if formula is not None:
        # convert irreps tensor to cartesian tensor
        layers["output_cartesian_tensor"] = (
            IrrepsToCartesianTensor,
            {"formula": formula, "field": OUT_FIELD_SELECTED},
        )

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
        "irreps_edge_sh": "0e + 1o",
        "num_radial_basis": 8,
        "radial_basis_r_cut": 4,
        "num_layers": 3,
        "invariant_layers": 2,
        "invariant_neurons": 64,
        "avg_num_neighbors": None,
        "use_sc": True,
        "nonlinearity_type": "gate",
        "resnet": True,
        "conv_to_output_hidden_irreps_out": "16x0e",
    }

    dataset_hyarmas = {"allowed_species": [6, 1, 8]}
    create_model(hparams, dataset_hyarmas)
