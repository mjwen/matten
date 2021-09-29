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

    # TODO set these as output field to make them explict
    This adds:
        - DataKey.NODE_ATTR
        - DataKey.NODE_FEATURES (if set_features is True)

    Args:
        embedding_dim: output dim of the species embedding
        num_species: number of uniques species for embedding. If this is provided,
            the `data` for forward should contain DataKey.SPECIES_INDEX, which are
            the index of the atom species (from 0 to num_species-1).
        allowed_species: allowed atomic number of the species. This serves the same
            purpose as `num_species`, and is exclusive with it. The difference is that
            allowed_species allows non-consecutive integers as input and it will be
            mapped to consecutive species_index internally. If this is used,
            the `data` for forward should contain DAtaKey.ATOMIC_NUMBERS.
        set_features: whether to set the embedding to DataKey.NODE_FEATURES
        irreps_in:
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        num_species: int = None,
        allowed_species: List[int] = None,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.set_features = set_features

        if allowed_species is not None and num_species is not None:
            raise ValueError("allowed_species and num_species cannot both be provided.")

        if allowed_species is not None:
            self.atomic_number_to_index = _AtomicNumberToIndex(allowed_species)
            num_species = self.atomic_number_to_index.num_species

        # Output irreps are num_species even (invariant) scalars
        irreps_out = {DataKey.NODE_ATTRS: Irreps([(embedding_dim, (0, 1))])}
        if self.set_features:
            irreps_out[DataKey.NODE_FEATURES] = irreps_out[DataKey.NODE_ATTRS]
        self.init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

        # learnable embedding layer
        self.embedding = nn.Embedding(num_species, embedding_dim)

    def forward(self, data: DataKey.Type) -> DataKey.Type:

        if DataKey.SPECIES_INDEX in data:
            type_numbers = data[DataKey.SPECIES_INDEX]
        elif DataKey.ATOMIC_NUMBERS in data:
            type_numbers = self.atomic_number_to_index(data[DataKey.ATOMIC_NUMBERS])
            data[DataKey.SPECIES_INDEX] = type_numbers
        else:
            raise ValueError(
                "Nothing in `data` to encode. Need either species_index or "
                "atomic_numbers"
            )

        embed = self.embedding(type_numbers)

        data[DataKey.NODE_ATTRS] = embed
        if self.set_features:
            data[DataKey.NODE_FEATURES] = embed

        return data


@compile_mode("script")
class _AtomicNumberToIndex(nn.Module):
    """
    Map non-consecutive atomic numbers to consecutive atomic index.

    For example, suppose we have C and O with atomic numbers 6 and 8, we can map them
    to atomic index 0, and 1.
    """

    def __init__(self, allowed_atomic_numbers: List[int]):
        super().__init__()

        allowed = torch.as_tensor(sorted(allowed_atomic_numbers), dtype=torch.long)
        num_species = len(allowed)

        self.register_buffer("_min_Z", allowed.min())
        self.register_buffer("_max_Z", allowed.max())
        self.register_buffer("_num_species", torch.as_tensor(num_species))

        # initialize all map to -1
        Z_to_index = torch.full((1 + self._max_Z - self._min_Z,), -1, dtype=torch.long)

        Z_to_index[allowed - self._min_Z] = torch.arange(num_species).to(torch.long)
        self.register_buffer("_Z_to_index", Z_to_index)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atomic_numbers: 1D tensor of atomic numbers

        Returns:
            1D tensor of atomic index
        """
        if atomic_numbers.min() < self._min_Z or atomic_numbers.max() > self._max_Z:
            raise RuntimeError(
                "Invalid atomic numbers. Expect atomic numbers to be in the range "
                f"[{self._min_Z}, {self._max_Z}], but got min {atomic_numbers.min()} "
                f"and max {atomic_numbers.max()}"
            )

        index = self._Z_to_index[atomic_numbers - self._min_Z]

        if index.min() < 0:
            num = None
            for num, val in enumerate(index):
                if val == -1:
                    num = num + self._min_Z
                    break
            raise RuntimeError(f"Invalid atomic numbers. {num}")

        return index

    @property
    def num_species(self):
        return self._num_species