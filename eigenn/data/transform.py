"""
Utilities to normalize a set of tensors.

This is for data standardization. Unlike scalars, where we can treat each component
separately and obtain statistics from them, tensors need to be treated at least on
the irreps level.
"""
from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from sklearn.preprocessing import StandardScaler
from torchtyping import TensorType


class Normalize(nn.Module):
    """
    Base class for tensor standardization.
    """

    def __init__(self, irreps: Union[str, Irreps]):
        super().__init__()

        self.irreps = Irreps(irreps)

    def forward(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821

        """
        Transform the data.

        Args:
            data: tensors to normalize. `batch`: batch dim of the tensors; `D`
                dimension of the tensors (should be compatible with self.irreps).
        """
        raise NotImplementedError

    def inverse(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821
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
        irreps: irreps of the tensor to normalize.
        mean: means used for normalization. If None, need to call
            `self.compute_statistics()` first to generate it.
        norm: norm used for normalization, If None, need to call
            `self.compute_statistics()` first to generate it.
        normalization: {'component', 'norm'}
        reduce: {'mean', 'max'}
        eps: epsilon to avoid diving be zero error
        scale: scale factor to multiply by norm. Because the data to be normalized
            will divide the norm, a value smaller than 1 will result in wider data
            distribution after normalization and a value larger than 1 will result in
            tighter data distribution.
    """

    def __init__(
        self,
        irreps: Union[str, Irreps],
        mean: TensorType = None,
        norm: TensorType = None,
        normalization: str = "component",
        reduce: str = "mean",
        eps: float = 1e-5,
        scale: float = 1.0,
    ):
        super().__init__(irreps)

        self.normalization = normalization
        self.reduce = reduce
        self.eps = eps
        self.scale = scale

        # Cannot register None as buffer for mean and norm, which means this module
        # does not need them. As a result, we cannot load them via state dict.
        if mean is None or norm is None:
            self.mean_norm_initialized = False
        else:
            self.mean_norm_initialized = True

        if mean is None:
            mean = torch.zeros(self.irreps.dim)
        if norm is None:
            norm = torch.zeros(self.irreps.dim)

        self.register_buffer("mean", mean)
        self.register_buffer("norm", norm)

    def forward(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821
        if not self.mean_norm_initialized:
            raise RuntimeError("mean and norm not initialized.")

        return (data - self.mean) / (self.norm * self.scale)

    def inverse(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821
        if not self.mean_norm_initialized:
            raise RuntimeError("mean and norm not initialized.")

        return data * (self.norm * self.scale) + self.mean

    # mean and norm
    def load_state_dict(
        self, state_dict: "OrderedDict[str, Tensor]", strict: bool = True
    ):
        super().load_state_dict(state_dict, strict)
        self.mean_norm_initialized = True

    def compute_statistics(self, data: TensorType["batch", "D"]):  # noqa: F821
        """
        Compute the mean and norm statistics.
        """

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
                field_norm = field_norm.max(dim=0)  # [mul]
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

        # Warning, do not delete this line
        self.load_state_dict({"mean": all_mean, "norm": all_norm})

        return all_mean, all_norm


class ScalarNormalize(nn.Module):
    """
    Normalize scalar quantities of shape [num_samples, num_features], each feature
    is normalized individually.

    Args:
        num_features: feature dim for the data to be normalized.
        scale: scale factor to multiply by norm. Because the data to be normalized
            will divide the norm, a value smaller than 1 will result in wider data
            distribution after normalization and a value larger than 1 will result in
            tighter data distribution.

    """

    def __init__(
        self,
        num_features: int,
        mean: TensorType = None,
        norm: TensorType = None,
        scale: float = 1.0,
    ):
        super().__init__()

        self.scale = scale

        # Cannot register None as buffer for mean and norm, which means this module
        # does not need them. As a result, we cannot load them via state dict.
        if mean is None or norm is None:
            self.mean_norm_initialized = False
        else:
            self.mean_norm_initialized = True

        if mean is None:
            mean = torch.zeros(num_features)
        if norm is None:
            norm = torch.zeros(num_features)

        self.register_buffer("mean", mean)
        self.register_buffer("norm", norm)

    def forward(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821
        if not self.mean_norm_initialized:
            raise RuntimeError("mean and norm not initialized.")

        return (data - self.mean) / (self.norm * self.scale)

    def inverse(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821
        if not self.mean_norm_initialized:
            raise RuntimeError("mean and norm not initialized.")

        return data * (self.norm * self.scale) + self.mean

    # mean and norm
    def load_state_dict(
        self, state_dict: "OrderedDict[str, Tensor]", strict: bool = True
    ):
        super().load_state_dict(state_dict, strict)
        self.mean_norm_initialized = True

    def compute_statistics(self, data: TensorType["batch", "D"]):  # noqa: F821
        """
        Compute the mean and norm statistics.
        """

        assert data.ndim == 2, "Can only deal with tensor [N_samples, N_features]"

        dtype = data.dtype
        data = data.numpy()

        scaler = StandardScaler()
        scaler.fit(data)
        mean = scaler.mean_
        std = scaler.scale_

        mean = torch.as_tensor(mean, dtype=dtype)
        std = torch.as_tensor(std, dtype=dtype)

        # Warning, do not delete this line,
        self.load_state_dict({"mean": mean, "norm": std})

        return mean, std
