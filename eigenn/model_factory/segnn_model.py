import sys
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Union

import torch
from nequip.nn.embedding import RadialBasisEdgeEncoding, SphericalHarmonicEdgeAttrs
from torch import Tensor

from eigenn.model.model import ModelForPyGData
from eigenn.model.task import CanonicalRegressionTask, Task
from eigenn.model_factory.utils import create_sequential_module
from eigenn.nn.embedding import SpeciesEmbedding
from eigenn.nn.segnn_conv import (
    EmbeddingLayer,
    MeanPredictionHead,
    SEGNNConv,
    SEGNNMessagePassing,
)

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

    def init_tasks(
        self,
        task_hparams: Dict[str, Any],
        dataset_hparams: Optional[Dict[str, Any]] = None,
    ) -> Union[Task, Sequence[Task]]:
        task = CanonicalRegressionTask(name=task_hparams["task_name"])

        return task

    def decode(self, model_input) -> Dict[str, Tensor]:
        out = self.backbone(model_input)
        out = out[OUT_FIELD_NAME].reshape(-1)

        task_name = self.hparams.task_hparams["task_name"]
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
    }

    # ===== node feats embedding layers =====
    n_embed_layers = 2
    for i in range(n_embed_layers):
        layers[f"node_feats_embedding_layer{i}"] = (
            EmbeddingLayer,
            {"irreps_out": {"node_features": hparams["conv_layer_irreps"]}},
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
            #     "avg_num_neighbors": hparams["avg_num_neighbors"],
            # },
            SEGNNMessagePassing,
            {
                "conv_layer_irreps": hparams["conv_layer_irreps"],
                "activation_type": hparams["nonlinearity_type"],
                "use_resnet": hparams["resnet"],
                "message_kwargs": {
                    "fc_num_hidden_layers": hparams["invariant_layers"],
                    "fc_hidden_size": hparams["invariant_neurons"],
                },
                "update_kwargs": {
                    "use_self_connection": hparams["use_sc"],
                    "avg_num_neighbors": hparams["avg_num_neighbors"],
                },
            },
        )

    # ===== prediction layers =====
    layers["mean_scalar_prediction"] = (
        MeanPredictionHead,
        {"out_field": OUT_FIELD_NAME},
    )

    model = create_sequential_module(modules=OrderedDict(layers))

    # print the model
    print(model, file=sys.stderr)

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
        "reduce": "sum",
        "invariant_layers": 2,
        "invariant_neurons": 64,
        "avg_num_neighbors": None,
        "use_sc": True,
        "nonlinearity_type": "gate",
        "resnet": True,
        "conv_to_output_hidden_irreps_out": "16x0e",
        "task_name": "my_task",
    }

    dataset_hyarmas = {"allowed_species": [6, 1, 8]}
    create_model(hparams, dataset_hyarmas)
