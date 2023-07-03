"""
TFN model for predicting graph-level scalar property.

Species embedding using torch.nn.Embedding. As a results, NOTE_ATTRS are learnable and
it is the same as NODE_FEATURES in the first layer. NOTE, they are the same only at the
first layer. In the model, NODE_FEATURES will be updated, but NODE_ATTRS are not.

The original NequIP uses ONE-hot embedding for NODE_ATTRS, and then use a linear layer
to map it to NODE_FEATURES.

For large number of species, we'd better use the SpeciesEmbedding one to minimize the
number of params.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from matten.model.model import ModelForPyGData
from matten.model_factory.utils import create_sequential_module
from matten.nn._nequip import SphericalHarmonicEdgeAttrs
from matten.nn.conv import PointConv, PointConvWithActivation
from matten.nn.embedding import EdgeLengthEmbedding, SpeciesEmbedding
from matten.nn.nodewise import NodewiseLinear, NodewiseReduce

OUT_FIELD_NAME = "my_model_output"


class TFNModel(ModelForPyGData):
    def init_backbone(
        self,
        backbone_hparams: Dict[str, Any],
        dataset_hparams: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.nn.Module, dict]:
        backbone = create_model(backbone_hparams, dataset_hparams)
        return backbone, None

    def decode(self, model_input) -> Dict[str, Tensor]:

        out = self.backbone(model_input)
        out = out[OUT_FIELD_NAME]

        # current we only support one task, so 0 is the name
        task_name = list(self.tasks.keys())[0]

        preds = {task_name: out}

        return preds

    def transform_prediction(self, preds: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Transform the normalized prediction back.
        """
        # TODO, assumes we only use one training task
        task_name = list(self.tasks.keys())[0]

        normalizer = self.tasks[task_name].normalizer

        if normalizer is not None:
            out = normalizer.inverse(preds[task_name], task_name)
        else:
            out = preds[task_name]

        return {task_name: out}

    def transform_target(self, target: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.transform_prediction(target)


def create_model(hparams: Dict[str, Any], dataset_hparams):
    """
    The actual function to create the model.
    """
    use_atom_feats = hparams.get("use_atom_feats", False)
    atom_feats_dim = dataset_hparams.get("atom_feats_size", None)

    # ===== input embedding layers =====
    layers = {
        "one_hot": (
            SpeciesEmbedding,
            {
                "embedding_dim": hparams["species_embedding_dim"],
                "allowed_species": dataset_hparams["allowed_species"],
                "use_atom_feats": use_atom_feats,
                "atom_feats_dim": atom_feats_dim,
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
        # To be less error-prone, we use SpeciesEmbedding.
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

    # conv without applying activation
    layers["conv_layer_last"] = (
        PointConv,
        {
            "conv_layer_irreps": hparams["conv_layer_irreps"],
            "fc_num_hidden_layers": hparams["invariant_layers"],
            "fc_hidden_size": hparams["invariant_neurons"],
            "avg_num_neighbors": num_neigh,
        },
    )

    # ===== prediction layers =====
    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the
            #  last layer of convnet
            # -- output block --
            "conv_to_output_hidden": (
                NodewiseLinear,
                {"irreps_out": hparams["conv_to_output_hidden_irreps_out"]},
            ),
            "output_hidden_to_output": (
                NodewiseLinear,
                {"irreps_out": "1x0e", "out_field": OUT_FIELD_NAME},
            ),
        }
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

    # model = create_sequential_module(
    #    modules=OrderedDict(layers), use_kwargs_irreps_in=True
    # )

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
        "radial_basis_start": 0.0,
        "radial_basis_end": 5.0,
        "num_layers": 3,
        "reduce": "sum",
        "invariant_layers": 2,
        "invariant_neurons": 64,
        "average_num_neighbors": None,
        "nonlinearity_type": "gate",
        "conv_to_output_hidden_irreps_out": "16x0e",
        "normalization": "batch",
    }

    dataset_hparams = {"allowed_species": [6, 1, 8]}
    create_model(hparams, dataset_hparams)