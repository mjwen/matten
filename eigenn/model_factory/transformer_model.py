import sys
from collections import OrderedDict

from nequip.data import AtomicDataDict
from nequip.nn import AtomwiseLinear, AtomwiseReduce
from nequip.nn.embedding import RadialBasisEdgeEncoding, SphericalHarmonicEdgeAttrs

from eigenn.model.model import ModelForPyGData
from eigenn.model.task import CanonicalRegressionTask
from eigenn.model_factory.utils import create_sequential_module
from eigenn.nn.transformer_conv import SE3Transformer
from eigenn.nn.embedding import SpeciesEmbedding


class TransformerEnergyModel(ModelForPyGData):
    """
    A model to predict the energy of an atomic configuration.
    """

    def init_backbone(self, hparams, dataset_hparams):
        backbone = create_energy_model(hparams, dataset_hparams)
        return backbone

    def init_tasks(self, hparams, dataset_hparams):
        task = CanonicalRegressionTask(name=hparams["task_name"])

        return task

    def decode(self, model_input):
        out = self.backbone(model_input)
        out = out["total_energy"].reshape(-1)

        task_name = self.hparams.task_hparams["task_name"]
        preds = {task_name: out}

        return preds


def create_energy_model(hparams, dataset_hparams):

    # ===== embedding layers =====
    layers = {
        # -- Encode --
        "one_hot": (
            SpeciesEmbedding,
            {
                "embedding_dim": hparams["species_embedding_dim"],
                "allowed_species": dataset_hparams["allowed_species"],
                # `node_features` determines output irreps. It must be used together with
                # set_features=False, which disables overriding of the given
                # node_features. Otherwise, node_features irreps will be set to
                # node_attrs irreps, which is determined by the `allowed_species`.
                #
                # Well, the OneHOtAtomEncoding has to use set_features = True, because
                # otherwise, node_features will not be include in the output data for
                # latter use.
                # "set_features": False,
                # "irreps_in": {"node_features": hparams["species_embedding_irreps_out"]},
                # TODO fix this in SpeciesEmbedding, then we may not need to use
                #  torch.nn.Embedding. (MW answer: well then we need to use a linear
                #  layer to map it to species_embedding_irreps_out. We can just use
                #  torch.nn.Embedding )
                "set_features": True,
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
        # This embed features is not necessary any more when we change OneHotEmbedding to
        # SpeciesEmbedding.
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
            SE3Transformer,
            {
                # "feature_irreps_hidden": hparams["feature_irreps_hidden"],
                # "nonlinearity_type": hparams["nonlinearity_type"],
                # "resnet": hparams["resnet"],
                "irreps_query_key": hparams["feature_irreps_hidden"],
                "fc_num_hidden_layers": hparams["invariant_layers"],
                "fc_hidden_size": hparams["invariant_neurons"],
                "avg_num_neighbors": hparams["avg_num_neighbors"],
                "r_max": hparams["radial_basis_r_cut"],
                "use_self_connection": hparams["use_sc"],
            },
        )

    # .update also maintains insertion order
    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # -- output block --
            "conv_to_output_hidden": (
                AtomwiseLinear,
                {"irreps_out": hparams["conv_to_output_hidden_irreps_out"]},
            ),
            "output_hidden_to_scalar": (
                AtomwiseLinear,
                dict(irreps_out="1x0e", out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY),
            ),
        }
    )

    reduce = hparams["reduce"]
    layers[f"total_energy_{reduce}"] = (
        AtomwiseReduce,
        dict(
            reduce=reduce,
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    )

    model = create_sequential_module(
        modules=OrderedDict(layers), use_kwargs_irreps_in=True
    )

    # print the model
    print(model, file=sys.stderr)

    return model


if __name__ == "__main__":
    from eigenn.log import set_logger

    set_logger("DEBUG")

    hparams = {
        "species_embedding_dim": 16,
        # "species_embedding_irreps_out": "16x0e",
        "feature_irreps_hidden": "32x0o + 32x0e + 16x1o + 16x1e",
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
    }

    dataset_hyarmas = {"allowed_species": [6, 1, 8]}
    create_energy_model(hparams, dataset_hyarmas)
