from collections import OrderedDict
from typing import Any, Dict

from nequip.data import AtomicDataDict
from nequip.nn import AtomwiseLinear, AtomwiseReduce, ConvNetLayer
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)

from eigenn.model.model import ModelForPyGData
from eigenn.model.task import CanonicalRegressionTask
from eigenn.model_factory.utils import create_sequential_module


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
        n = out["total_energy"].reshape(-1)
        preds = {"n": n}

        return preds


def create_energy_model(hparams, dataset_hparams):

    num_layers = hparams.pop("num_layers", 3)

    # ===== embedding layers =====
    layers = {
        # -- Encode --
        "one_hot": (
            OneHotAtomEncoding,
            {
                # "num_species": dataset_hparams["num_species"],
                "allowed_species": dataset_hparams["allowed_species"],
                "irreps_in": {"node_features": hparams["species_embedding_irreps_out"]},
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

    layers["total_energy_sum"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    )

    model = create_sequential_module(
        modules=OrderedDict(layers), use_kwargs_irreps_in=True
    )

    return model


if __name__ == "__main__":
    from eigenn.log import set_logger

    params = {
        "species_embedding_irreps_out": "16x0e",
        "feature_irreps_hidden": "32x0o + 32x0e + 16x1o + 16x1e",
        "irreps_edge_sh": "0e + 1o",
        "num_layers": 3,
        "r_max": 4,
    }
    create_energy_model(params)