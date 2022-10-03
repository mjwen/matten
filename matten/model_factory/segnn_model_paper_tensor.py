"""
Strict reimplementation of SEGNN model.

For tensor prediction.
"""
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from e3nn.io import CartesianTensor
from torch import Tensor

from matten.model.model import ModelForPyGData
from matten.model_factory.utils import create_sequential_module
from matten.nn._nequip import RadialBasisEdgeEncoding, SphericalHarmonicEdgeAttrs
from matten.nn.embedding import NodeAttrsFromEdgeAttrs, SpeciesEmbedding
from matten.nn.nodewise import NodewiseLinear, NodewiseReduce
from matten.nn.readout import IrrepsToCartesianTensor
from matten.nn.segnn_paper import EmbeddingLayer, SEGNNMessagePassing
from matten.nn.utils import DetectAnomaly

OUT_FIELD_NAME = "my_model_output"


class SEGNNModel(ModelForPyGData):
    """
    A model to predict the energy of an atomic configuration.
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
        out = out[OUT_FIELD_NAME].reshape(-1)

        # current we only support one task, so 0 is the name
        task_name = list(self.tasks.keys())[0]

        preds = {task_name: out}

        return preds


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
        # "anomaly_1": (DetectAnomaly, {"name": "anomaly_1"}),
        "spharm_edges": (
            SphericalHarmonicEdgeAttrs,
            {"irreps_edge_sh": hparams["irreps_edge_sh"]},
        ),
        # "anomaly_2": (DetectAnomaly, {"name": "anomaly_2"}),
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
        # "anomaly_3": (DetectAnomaly, {"name": "anomaly_3"}),
        "node_attrs_layer": (NodeAttrsFromEdgeAttrs, {}),
    }

    # ===== node feats embedding layers =====
    n_embed_layers = 2
    for i in range(n_embed_layers):
        layers[f"node_feats_embedding_layer{i}"] = (
            EmbeddingLayer,
            {
                "irreps_out": {"node_features": hparams["conv_layer_irreps"]},
                "normalization": hparams["normalization"],
            },
        )

        # layers[f"anomaly_embedding_{i}"] = (
        #     DetectAnomaly,
        #     {"name": f"anomaly_embedding_{i}"},
        # )

    # ===== message passing layers =====
    for i in range(hparams["num_layers"]):
        layers[f"message_passing_layer{i}"] = (
            # SEGNNConv,
            # {
            #     "conv_layer_irreps": hparams["conv_layer_irreps"],
            #     "activation_type": hparams["nonlinearity_type"],
            #     "fc_num_hidden_layers": hparams["invariant_layers"],
            #     "fc_hidden_size": hparams["invariant_neurons"],
            #     "use_self_connection": hparams["use_sc"],
            #     "avg_num_neighbors": num_neigh,
            # },
            SEGNNMessagePassing,
            {
                "conv_layer_irreps": hparams["conv_layer_irreps"],
                "activation_type": hparams["nonlinearity_type"],
                "use_resnet": hparams["resnet"],
                "normalization": hparams["normalization"],
            },
        )

        # layers[f"anomaly_mp_{i}"] = (DetectAnomaly, {"name": f"anomaly_mp_{i}"})

    # ===== prediction layers =====
    #
    # determining output formula and irreps
    #

    output_format = hparams["output_format"]

    if output_format == "cartesian":
        # e.g. ij=ji for a symmetric 2D tensor
        formula = hparams["output_formula"]
        output_irreps = CartesianTensor(formula)

    elif output_format == "irreps":
        # e.g. '0e+2e' for a symmetric 2D tensor
        output_irreps = hparams["output_formula"]

    else:
        supported = ["cartesian", "irreps"]
        raise ValueError(
            f"Expect `output_format` to be one of {supported}; got {output_format}"
        )

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
                {"irreps_out": output_irreps, "out_field": OUT_FIELD_NAME},
            ),
        }
    )

    # convert irreps tensor to cartesian tensor if necessary
    if output_format == "cartesian":
        layers["output_cartesian_tensor"] = (
            IrrepsToCartesianTensor,
            {"formula": formula, "field": OUT_FIELD_NAME},
        )

    # pooling
    layers["output_pooling"] = (
        NodewiseReduce,
        {
            "field": OUT_FIELD_NAME,
            "out_field": OUT_FIELD_NAME,
            "reduce": hparams["reduce"],
        },
    )

    # layers[f"anomaly_pred"] = (DetectAnomaly, {"name": f"anomaly_pred"})

    model = create_sequential_module(modules=OrderedDict(layers))

    return model


if __name__ == "__main__":
    from matten.log import set_logger

    set_logger("DEBUG")

    hparams = {
        "species_embedding_dim": 16,
        # "species_embedding_irreps_out": "16x0e",
        "conv_layer_irreps": "32x0o + 32x0e + 16x1o + 16x1e",
        "irreps_edge_sh": "0e + 1o",
        "num_radial_basis": 8,
        "radial_basis_r_cut": 4,
        "num_layers": 3,
        "reduce": "sum",
        "nonlinearity_type": "gate",
        "resnet": True,
        "conv_to_output_hidden_irreps_out": "16x0e",
        "task_name": "my_task",
        "normalization": "batch",
        "output_format": "irreps",
        "output_formula": "2x0e+2x2e+1x4e",
    }

    dataset_hyarmas = {"allowed_species": [6, 1, 8]}
    create_model(hparams, dataset_hyarmas)
