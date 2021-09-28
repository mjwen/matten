from typing import List

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from eigenn.nn.irreps import DataKey, ModuleIrreps


@compile_mode("script")
class SpeciesEmbedding(nn.Module, ModuleIrreps):
    """
    Embed atomic species (number) to node attrs and node features with fixed-size lookup
    table using torch.nn.Embedding.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        allowed_species: List[int] = None,
        num_species: int = None,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        if allowed_species is not None and num_species is not None:
            raise ValueError("allowed_species and num_species cannot both be provided.")

        if allowed_species is not None:
            num_species = len(allowed_species)
            allowed_species = torch.as_tensor(allowed_species)
            self.register_buffer("_min_Z", allowed_species.min())
            self.register_buffer("_max_Z", allowed_species.max())
            Z_to_index = torch.full(
                (1 + self._max_Z - self._min_Z,), -1, dtype=torch.long
            )
            Z_to_index[allowed_species - self._min_Z] = torch.arange(num_species)
            self.register_buffer("_Z_to_index", Z_to_index)

        self.set_features = set_features

        # Output irreps are num_species even (invariant) scalars
        irreps_out = {DataKey.NODE_ATTRS: Irreps([(embedding_dim, (0, 1))])}
        if self.set_features:
            irreps_out[DataKey.NODE_FEATURES] = irreps_out[DataKey.NODE_ATTRS]
        self.init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

        # learnable embedding layer
        self.embedding = nn.Embedding(num_species, embedding_dim)

    @torch.jit.export
    def index_for_atomic_numbers(self, atomic_nums: torch.Tensor):
        if atomic_nums.min() < self._min_Z or atomic_nums.max() > self._max_Z:
            raise RuntimeError("Invalid atomic numbers for this OneHotEncoding")

        out = self._Z_to_index[atomic_nums - self._min_Z]
        assert out.min() >= 0, "Invalid atomic numbers for this OneHotEncoding"
        return out

    def forward(self, data: DataKey.Type):

        # sanity check
        if DataKey.SPECIES_INDEX in data:
            type_numbers = data[DataKey.SPECIES_INDEX]
        elif DataKey.ATOMIC_NUMBERS in data:
            type_numbers = self.index_for_atomic_numbers(data[DataKey.ATOMIC_NUMBERS])
            data[DataKey.SPECIES_INDEX] = type_numbers
        else:
            raise ValueError(
                "Nothing in this `data` to encode. Need either species index or "
                "atomic numbers"
            )

        embed = self.embedding(type_numbers)

        data[DataKey.NODE_ATTRS] = embed
        if self.set_features:
            data[DataKey.NODE_FEATURES] = embed

        return data
