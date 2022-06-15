"""
TFN model for predicting graph level tensor property of the same shape.

Species embedding using torch.nn.Embedding. As a results, NOTE_ATTRS are learnable and
it is the same as NODE_FEATURES in the first layer. NOTE, they are the same only at the
first layer. In the model, NODE_FEATURES will be updated, but NODE_ATTRS are not.

The original NequIP uses ONE-hot embedding for NODE_ATTRS, and then use a linear layer
to map it to NODE_FEATURES.

For large number of species, we'd better use the SpeciesEmbedding one to minimize the
number of params.
"""


from typing import Any, Dict, Optional

import torch
from torch import Tensor

from eigenn.core.elastic import ElasticTensor
from eigenn.model.model import ModelForPyGData
from eigenn.model_factory.tfn_tensor import create_model

OUT_FIELD_NAME = "my_model_output"


class TFNModel(ModelForPyGData):
    def init_backbone(
        self,
        backbone_hparams: Dict[str, Any],
        dataset_hparams: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        backbone = create_model(backbone_hparams, dataset_hparams)
        return backbone

    def decode(self, model_input) -> Dict[str, Tensor]:

        out = self.backbone(model_input)
        out = out[OUT_FIELD_NAME]

        tensor_preds = self._decode_tensors(out)
        scalar_preds = self._decode_scalars(out)
        preds = {**tensor_preds, **scalar_preds}

        return preds

    def _decode_tensors(self, out) -> Dict[str, Tensor]:
        # TODO this assume tensor task will always be the first
        task_name = list(self.tasks.keys())[0]
        preds = {task_name: out}

        return preds

    def _decode_scalars(self, out) -> Dict[str, Tensor]:

        # TODO this assume tensor task will always be the first
        scalar_task_names = list(self.tasks.keys())[1:]

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

    def transform_prediction(self, preds: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Transform the normalized prediction back.
        """

        task_name = "elastic_tensor_full"

        normalizer = self.tasks[task_name].normalizer

        if normalizer is not None:
            out = normalizer.inverse(preds[task_name])
        else:
            out = preds[task_name]

        return {task_name: out}

    def transform_target(self, target: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.transform_prediction(target)


if __name__ == "__main__":
    from eigenn.log import set_logger

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
