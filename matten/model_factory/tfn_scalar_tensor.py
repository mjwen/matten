"""
Scalar and Tensor target, without global features.

More general than tfn_scalar_tensor_global_feats.py in terms of the output format,
can be any scalar, vector, and tensor, which needs to be specified by the output format.

Also, this won't work for the scalar via tensor case as in
tfn_scalar_tensor_global_feats.py.

Note, this only works for a single target.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from e3nn.io import CartesianTensor
from e3nn.o3 import Irreps, Linear
from torch import Tensor

from matten.core.utils import ToCartesian
from matten.model.model import ModelForPyGData
from matten.model_factory.utils import create_sequential_module
from matten.nn._nequip import SphericalHarmonicEdgeAttrs
from matten.nn.embedding import EdgeLengthEmbedding, SpeciesEmbedding
from matten.nn.nodewise import NodewiseLinear, NodewiseReduce
from matten.nn.tfn import PointConv, PointConvWithActivation

OUT_FIELD_NAME = "my_model_output"


class ScalarTensorModel(ModelForPyGData):
    def init_backbone(
        self,
        backbone_hparams: Dict[str, Any],
        dataset_hparams: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.nn.Module, dict]:

        backbone = create_model(backbone_hparams, dataset_hparams)

        formula = (backbone_hparams["output_formula"]).lower()

        # Last linear layer to covert irreps size
        if formula == "scalar":
            # special treatment of scalars
            irreps_out = Irreps("0e")
        else:
            irreps_out = CartesianTensor(formula=formula)
        irreps_in = backbone_hparams["conv_to_output_hidden_irreps_out"]
        extra_layers_dict = {
            "out_layer": Linear(irreps_in=irreps_in, irreps_out=irreps_out)
        }

        if backbone_hparams["output_format"] == "cartesian":
            if formula == "scalar":
                self.to_cartesian = None
            else:
                self.to_cartesian = ToCartesian(formula)
        else:
            self.to_cartesian = None

        return backbone, extra_layers_dict

    def decode(self, model_input) -> Dict[str, Tensor]:

        out = self.backbone(model_input)
        out = out[OUT_FIELD_NAME]

        # convert backbone to final irreps shape
        out = self.extra_layers_dict["out_layer"](out)

        if self.to_cartesian is not None:
            out = self.to_cartesian(out)

        names = list(self.tasks.keys())
        assert len(names) == 1, f"only works for 1 target, get{len(names)}"

        task_name = names[0]
        preds = {task_name: out}

        return preds

    def transform_prediction(
        self, preds: Dict[str, Tensor], task_name: str = "elastic_tensor_full"
    ) -> Dict[str, Tensor]:
        """
        Transform the normalized prediction back.
        """

        normalizer = self.tasks[task_name].normalizer

        if normalizer is not None:
            out = normalizer.inverse(preds[task_name])
        else:
            out = preds[task_name]

        return {task_name: out}

    def transform_target(
        self, target: Dict[str, Tensor], task_name: str = "elastic_tensor_full"
    ) -> Dict[str, Tensor]:
        return self.transform_prediction(target, task_name)


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
    #
    # determining output irreps
    #
    # output_irreps = CartesianTensor(hparams["output_formula"])

    layers.update(
        {
            #  last layer of convnet
            # -- output block --
            "conv_to_output_hidden": (
                NodewiseLinear,
                {
                    "irreps_out": hparams["conv_to_output_hidden_irreps_out"],
                    "out_field": OUT_FIELD_NAME,
                },
            )
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
        "radial_basis_end": 4.0,
        "num_layers": 3,
        "reduce": "sum",
        "invariant_layers": 2,
        "invariant_neurons": 64,
        "average_num_neighbors": None,
        "nonlinearity_type": "gate",
        "conv_to_output_hidden_irreps_out": "16x0e",
        "normalization": "batch",
        "output_format": "irreps",
        "output_formula": "2x0e+2x2e+4e",
    }

    dataset_hyarmas = {"allowed_species": [6, 1, 8]}
    create_model(hparams, dataset_hyarmas)
