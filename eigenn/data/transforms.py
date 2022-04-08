"""
Utilities to normalize a set of tensors.

This is for data standardization. Unlike scalars, where we can treat each component
separately and obtain statistics from them, tensors need to be treated at least on
the irreps level.
"""
from typing import Union

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from torchtyping import TensorType


class Normalize(nn.Module):
    """
    Base class for tensor standardization.
    """

    def __init__(self, irreps: Union[str, Irreps]):
        super().__init__()

        self.irreps = Irreps(irreps)

    def forward(self, data: TensorType["batch", "D"]) -> TensorType["batch", "D"]:
        """
        Transform the data.

        Args:
            data: tensors to normalize. `batch`: batch dim of the tensors; `D`
                dimension of the tensors (should be compatible with self.irreps).
        """
        raise NotImplementedError

    def inverse(self, data: TensorType["batch", "D"]) -> TensorType["batch", "D"]:
        """
        Inverse transform the data.

        Args:
            data: tensors to normalize. `batch`: batch dim of the tensors; `D`
                dimension of the tensors (should be compatible with self.irreps).
        """
        raise NotImplementedError


# Based on e3nn BatchNorm
class MeanNormNormalize(Normalize):
    """
    Normalize tensors like e3nn BatchNorm:
    - scalars are normalized by subtracting mean and then dividing by norms
    - higher order irreps are normalized by dividing norms

    Note, each irrep is treated separated. For example, for irreps "3x1e", the norm
    is computed separated for each 1e.

    Args:
        irreps:
        mean: means used for normalization, if None, computed from data
        norm: norm used for normalization, if None, computed from data
        normalization: {'component', 'norm'}
        reduce: {'mean', 'max'}
        eps: epsilon to avoid diving be zero error
    """

    def __init__(
        self,
        irreps: Union[str, Irreps],
        mean: TensorType = None,
        norm: TensorType = None,
        normalization: str = "component",
        reduce: str = "mean",
        eps: float = 1e-5,
    ):
        super().__init__(irreps)

        self.normalization = normalization
        self.reduce = reduce
        self.eps = eps

        self.register_buffer("mean", mean)
        self.register_buffer("norm", norm)

    def forward(self, data: TensorType["batch", "D"]) -> TensorType["batch", "D"]:
        if self.mean is None or self.norm is None:
            self.mean, self.norm = self._compute_mean_and_norm(data)

        return (data - self.mean) / self.norm

    def inverse(self, data: TensorType["batch", "D"]) -> TensorType["batch", "D"]:
        if self.mean is None or self.norm is None:
            raise RuntimeError(
                "Cannot perform inverse transform; either mean or norm is `None`."
            )
        return data * self.norm + self.mean

    def _compute_mean_and_norm(self, data: TensorType["batch", "D"]):

        all_mean = []
        all_norm = []

        ix = 0

        for (mul, ir) in self.irreps:  # mul: multiplicity, ir: an irrep
            d = ir.dim
            field = data[:, ix : ix + mul * d]  # [batch, mul * repr]
            ix += mul * d

            field = field.reshape(-1, mul, d)  # [batch, mul, repr]

            if ir.is_scalar():
                # compute mean of scalars (higher order tensors does not use mean)
                field_mean = field.mean(dim=0).reshape(mul)  # [mul]

                # subtract mean for computing stdev as norm below
                field = field - field_mean.reshape(-1, mul, 1)
            else:
                # set mean to zero for high order tensors
                field_mean = torch.zeros(mul)

            # expand to the repr dimension, shape [mul*repr]
            field_mean = torch.repeat_interleave(field_mean, repeats=d, dim=0)
            all_mean.append(field_mean)

            #
            # compute the rescaling factor (norm of each feature vector)
            #
            # 1. rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                field_norm = field.pow(2).sum(dim=-1)  # [batch, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(dim=-1)  # [batch, mul]
            else:
                raise ValueError(f"Invalid normalization option {self.normalization}")

            # 2. reduction method
            if self.reduce == "mean":
                field_norm = field_norm.mean(dim=0)  # [mul]
            elif self.reduce == "max":
                field_norm = field.max(dim=0)  # [mul]
            else:
                raise ValueError("Invalid reduce option {}".format(self.reduce))

            # Then apply the rescaling
            # divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(0.5)  # [mul]

            # expand to the repr dimension, shape [mul*repr]
            field_norm = torch.repeat_interleave(field_norm, repeats=d, dim=0)
            all_norm.append(field_norm)

        dim = data.shape[-1]
        assert (
            ix == dim
        ), f"`ix` should have reached data.size(-1) ({dim}), but it ended at {ix}"

        all_mean = torch.cat(all_mean)  # [dim]
        all_norm = torch.cat(all_norm)  # [dim]

        assert len(all_mean) == dim, (
            f"Expect len(all_mean) and data.shape[-1] to be equal; got {len(all_mean)} "
            f"and {dim}."
        )
        assert len(all_norm) == dim, (
            f"Expect len(all_norm) and data.shape[-1] to be equal; got {len(all_norm)} "
            f"and {dim}."
        )

        return all_mean, all_norm
