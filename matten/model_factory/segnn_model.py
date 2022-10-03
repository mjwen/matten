"""
mjwen implementation of segnn style model
"""
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from matten.model.model import ModelForPyGData
from matten.model_factory.utils import create_sequential_module
from matten.nn._nequip import RadialBasisEdgeEncoding, SphericalHarmonicEdgeAttrs
from matten.nn.embedding import NodeAttrsFromEdgeAttrs, SpeciesEmbedding
from matten.nn.segnn_conv import EmbeddingLayer, PredictionHead, SEGNNMessagePassing

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

    num_neigh = hparams["average_num_neighbors"]
    if isinstance(num_neigh, str) and num_neigh.lower() == "auto":
        num_neigh = dataset_hparams["average_num_neighbors"]

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
                "message_kwargs": {
                    "fc_num_hidden_layers": hparams["invariant_layers"],
                    "fc_hidden_size": hparams["invariant_neurons"],
                    "normalization": hparams["normalization"],
                },
                "update_kwargs": {
                    "use_self_connection": hparams["use_sc"],
                    "avg_num_neighbors": num_neigh,
                    "normalization": hparams["normalization"],
                },
            },
        )

    # ===== prediction layers =====
    layers["mean_scalar_prediction"] = (
        PredictionHead,
        {"out_field": OUT_FIELD_NAME, "reduce": hparams["reduce"]},
    )

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
        "invariant_layers": 2,
        "invariant_neurons": 64,
        "average_num_neighbors": None,
        "use_sc": True,
        "nonlinearity_type": "gate",
        "resnet": True,
        "conv_to_output_hidden_irreps_out": "16x0e",
        "task_name": "my_task",
        "normalization": "batch",
    }

    dataset_hyarmas = {"allowed_species": [6, 1, 8]}
    create_model(hparams, dataset_hyarmas)
