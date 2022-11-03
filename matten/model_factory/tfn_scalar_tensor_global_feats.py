"""
Scalar and/or tensor target with global features, and harded coded for elastic tensor.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from e3nn.o3 import Irrep, Irreps
from torch import Tensor

from matten.core.elastic import ElasticTensor
from matten.core.utils import ToCartesian
from matten.model.model import ModelForPyGData
from matten.model_factory.utils import create_sequential_module
from matten.nn._nequip import SphericalHarmonicEdgeAttrs
from matten.nn.embedding import EdgeLengthEmbedding, SpeciesEmbedding
from matten.nn.nodewise import NodewiseLinear, NodewiseReduce
from matten.nn.tfn import PointConv, PointConvWithActivation
from matten.nn.utils import ScalarMLP

OUT_FIELD_NAME = "my_model_output"
TENSOR_TASK_NAME = "elastic_tensor_full"


class ScalarTensorGlobalFeatsModel(ModelForPyGData):
    def init_backbone(
        self,
        backbone_hparams: Dict[str, Any],
        dataset_hparams: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.nn.Module, dict]:
        backbone = create_model(backbone_hparams, dataset_hparams)

        # convert irreps tensor to cartesian tensor if necessary
        self.convert_out = ToCartesian(backbone_hparams["output_formula"])
        self.output_format = backbone_hparams["output_format"]

        # linear layers for combining global feats and 0e
        irreps = Irreps(backbone_hparams["conv_to_output_hidden_irreps_out"])
        zero_e = Irrep("0e")
        zero_e_size = None
        for mul, ir in irreps:
            if ir == zero_e:
                zero_e_size = mul
                break
        assert zero_e_size is not None

        extra_layers_dict = {}
        self.global_feats_size = dataset_hparams["global_feats_size"]
        if self.global_feats_size is not None:

            extra_layers_dict["linear_global_feats"] = ScalarMLP(
                in_size=self.global_feats_size,
                hidden_sizes=backbone_hparams["linear_global_feats_hidden_sizes"],
                batch_norm=True,
                out_size=zero_e_size,
            )

            self.global_feats_mix_mode = backbone_hparams["global_feats_mix_mode"]
            if self.global_feats_mix_mode == "concat":
                in_size = 2 * zero_e_size
            elif self.global_feats_mix_mode == "add":
                in_size = zero_e_size
            else:
                raise ValueError
        else:
            in_size = zero_e_size
        # TODO hardcoded, 2x0e+2x2e+4e for elastic; here, we hardcode to transform 2e

        extra_layers_dict["linear_0e"] = ScalarMLP(
            in_size=in_size, hidden_sizes=[in_size], batch_norm=True, out_size=2
        )

        return backbone, extra_layers_dict

    def decode(self, model_input) -> Dict[str, Tensor]:

        out = self.backbone(model_input)
        out = out[OUT_FIELD_NAME]

        if self.global_feats_size is not None:
            mode = self.global_feats_mix_mode
        else:
            mode = "no_global_feats"  # not mix
        out = self.mix_global_feats(model_input, out, mode=mode)

        if self.output_format == "cartesian":
            out_tensor = out_tensor_for_scalar = self.convert_out(out)
        elif self.output_format == "irreps":
            out_tensor = out
            out_tensor_for_scalar = self.convert_out(out)
        else:
            raise ValueError

        if TENSOR_TASK_NAME in self.tasks:
            tensor_preds = {TENSOR_TASK_NAME: out_tensor}
        else:
            tensor_preds = dict()

        scalar_preds = self._decode_scalars(out_tensor_for_scalar)

        preds = {**tensor_preds, **scalar_preds}

        return preds

    def mix_global_feats(
        self,
        model_input,
        output,
        irreps="2x0e+2x2e+4e",  # TODO this needs to be input
        mode: str = "concat",
    ) -> torch.Tensor:
        """
        Add global feats to 0e scalars and then do linear transformations before
        adding them back.

        mode:
            How to combine the global features and 0e features.
            concat: concatenate global feats to 0e and then linearly map them back
                to required size of 0e.
            add: add global feats to 0e feats (this requires they be of the same
                length).
            no_global_feats: not mix at all, simply resize 0e.
        """

        # NOTE hard coded to elastic tensor
        # split 0e and other
        # ax0e + 2x2e + 4e
        values_0e = output[:, :-19]
        values_high_order = output[:, -19:]

        if mode in ["concat", "add"]:
            # resize global feats
            global_feats = model_input["global_feats"]
            global_feats = self.extra_layers_dict["linear_global_feats"](global_feats)
            if mode == "concat":
                combined = torch.hstack((global_feats, values_0e))
            elif mode == "add":
                combined = global_feats + values_0e
            else:
                raise ValueError(f"not supported mode {mode}")
        elif mode == "no_global_feats":
            # do not use global feats, just resize
            combined = values_0e
        else:
            raise ValueError(f"not supported mode {mode}")

        # scale to the same size of original 0e
        combined = self.extra_layers_dict["linear_0e"](combined)

        # add scaled feats back to high order irreps
        out = torch.hstack((combined, values_high_order))

        return out

    def _decode_scalars(self, out) -> Dict[str, Tensor]:
        if TENSOR_TASK_NAME in self.tasks:
            out = self.transform_prediction({TENSOR_TASK_NAME: out})
            out = out[TENSOR_TASK_NAME]

            # TODO this assume tensor task will always be the first
            scalar_task_names = list(self.tasks.keys())[1:]
        else:
            scalar_task_names = list(self.tasks.keys())

        preds = {}
        for name in scalar_task_names:

            # convert tensor to derived properties
            derived_prop = []

            # deal with batch
            for t in out:
                et = ElasticTensor(t)
                derived_prop.append(getattr(et, name))

            # reshape to make it 2D, target is 2D
            derived_prop = torch.stack(derived_prop).reshape(-1, 1)
            preds[name] = derived_prop

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
