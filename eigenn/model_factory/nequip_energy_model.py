import sys
from collections import OrderedDict

from nequip.data import AtomicDataDict
from nequip.nn import AtomwiseLinear, AtomwiseReduce, ConvNetLayer
from nequip.nn.embedding import RadialBasisEdgeEncoding, SphericalHarmonicEdgeAttrs

from eigenn.model.model import ModelForPyGData
from eigenn.model.task import CanonicalRegressionTask
from eigenn.model_factory.utils import create_sequential_module
from eigenn.nn.embedding import SpeciesEmbedding


class EnergyModel(ModelForPyGData):
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

    num_layers = hparams.pop("num_layers", 3)

    # ===== embedding layers =====
    layers = {
        # -- Encode --
        "one_hot": (
            SpeciesEmbedding,
            {
                "embedding_dim": hparams["species_embedding_dim"],
                "allowed_species": dataset_hparams["allowed_species"],
                # node_features determines output irreps. It must be used together with
                # set_features=False, which disables overriding of the given
                # node_features. Otherwise, node_features irreps will be set to
                # node_attrs irreps, which is determined by the `allowed_species`.
                #
                # Well, the OneHOtAtomEncoding has to sue set_features = True, because
                # otherwise, node_features will not be include in the output data for
                # latter use.
                # "set_features": False,
                # "irreps_in": {"node_features": hparams["species_embedding_irreps_out"]},
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
                "basis_kwargs": {"r_max": hparams["r_max"]},
                "cutoff_kwargs": {"r_max": hparams["r_max"]},
            },
        ),
        # -- Embed features --
        "chemical_embedding": (AtomwiseLinear, {}),
    }

    # ===== convnet layers =====
    # insertion preserves order
    for i in range(num_layers):
        layers[f"layer{i}_convnet"] = (
            ConvNetLayer,
            {
                "feature_irreps_hidden": hparams["feature_irreps_hidden"],
            },
        )

    # .update also maintains insertion order
    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # -- output block --
            "conv_to_output_hidden": (AtomwiseLinear, {}),
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
        "species_embedding_irreps_out": "16x0e",
        "feature_irreps_hidden": "32x0o + 32x0e + 16x1o + 16x1e",
        "irreps_edge_sh": "0e + 1o",
        "num_layers": 3,
        "r_max": 4,
        "reduce": "sum",
    }

    dataset_hyarmas = {"allowed_species": [6, 1, 8]}
    create_energy_model(hparams, dataset_hyarmas)
