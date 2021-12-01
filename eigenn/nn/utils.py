from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as fn
from e3nn.nn import FullyConnectedNet, Gate, NormActivation
from e3nn.o3 import Irreps, TensorProduct
from nequip.nn.nonlinearities import ShiftedSoftPlus
from nequip.utils.tp_utils import tp_path_exists
from torch import Tensor

from eigenn.nn.irreps import DataKey, ModuleIrreps
from eigenn.utils import detect_nan_and_inf

ACTIVATION = {
    # for even irreps
    "e": {
        "ssp": ShiftedSoftPlus,
        "silu": fn.silu,
    },
    # for odd irreps
    "o": {
        "abs": torch.abs,
        "tanh": torch.tanh,
    },
}


class ActivationLayer(torch.nn.Module):
    def __init__(
        self,
        tp_irreps_in1: Irreps,
        tp_irreps_in2: Irreps,
        tp_irreps_out: Irreps,
        *,
        activation_type: str = "gate",
        activation_scalars: Dict[str, str] = None,
        activation_gates: Dict[str, str] = None,
    ):
        """
        Nonlinear equivariant activation function layer.

        This is intended to be applied after a tensor product convolution layer.

        Args:
            tp_irreps_in1: first irreps for the tensor product layer
            tp_irreps_in2: second irreps for the tensor product layer
            tp_irreps_out: intended output irreps for the tensor product layer.
                Note, typically this is not the actual irreps out we will use for the
                tensor product. The actual one is typically determined here, i.e.
                the `irreps_in` attribute of this class.
            activation_type: `gate` or `norm`
            activation_scalars: activation function for scalar irreps (i.e. l=0).
                Should be something like {'e':act_e, 'o':act_o}, where `act_e` is the
                name of the activation function ('ssp' or 'silu') for even irreps;
                `act_o` is the name of the activation function ('abs' or 'tanh') for
                odd irreps.
            activation_gates: activation function for tensor irreps (i.e. l>0) when
                using the `Gate` activation. Ignored for `NormActivation`.
                Should be something like {'e':act_e, 'o':act_o}, where `act_e` is the
                name of the activation function ('ssp' or 'silu') for even irreps;
                `act_o` is the name of the activation function ('abs' or 'tanh') for
                odd irreps.
        """
        super().__init__()

        # set defaults

        # activation function for even (i.e. 1) and odd (i.e. -1) scalars
        if activation_scalars is None:
            activation_scalars = {
                1: ACTIVATION["e"]["ssp"],
                # odd scalars requires either an even or odd activation,
                # not an arbitrary one like relu
                -1: ACTIVATION["o"]["tanh"],
            }
        else:
            # change key from e or v to 1 or -1
            key_mapping = {"e": 1, "o": -1}
            activation_scalars = {
                key_mapping[k]: ACTIVATION[k][v] for k, v in activation_scalars.items()
            }

        # activation function for even (i.e. 1) and odd (i.e. -1) high-order tensors
        if activation_gates is None:
            activation_gates = {1: ACTIVATION["e"]["ssp"], -1: ACTIVATION["o"]["abs"]}
        else:
            # change key from e or v to 1 or -1
            key_mapping = {"e": 1, "o": -1}
            activation_gates = {
                key_mapping[k]: ACTIVATION[k][v] for k, v in activation_gates.items()
            }

        # in and out irreps of activation

        ir, _, _ = Irreps(tp_irreps_out).sort()
        tp_irreps_out = ir.simplify()

        irreps_scalars = Irreps(
            [
                (mul, ir)
                for mul, ir in tp_irreps_out
                if ir.l == 0 and tp_path_exists(tp_irreps_in1, tp_irreps_in2, ir)
            ]
        )
        irreps_gated = Irreps(
            [
                (mul, ir)
                for mul, ir in tp_irreps_out
                if ir.l > 0 and tp_path_exists(tp_irreps_in1, tp_irreps_in2, ir)
            ]
        )

        if activation_type == "gate":
            # Setting all ir to 0e if there is path exist, since 0e gates will not
            # change the parity of the output, and we want to keep its parity.
            # If there is no 0e, we use 0o to change the party of the high-order irreps.
            ir = "0e" if tp_path_exists(tp_irreps_in1, tp_irreps_in2, "0e") else "0o"
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])

            self.activation = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[activation_scalars[ir.p] for _, ir in irreps_scalars],
                irreps_gates=irreps_gates,
                act_gates=[activation_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

        elif activation_type == "norm":

            self.activation = NormActivation(
                irreps_in=(irreps_scalars + irreps_gated).simplify(),
                # norm is an even scalar, so activation_scalars[1]
                scalar_nonlinearity=activation_scalars[1],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )

        else:
            supported = ("gate", "norm")
            raise ValueError(
                f"Support `activation_type` includes {supported}, got {activation_type}"
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x)

    @property
    def irreps_in(self):
        # irreps_in for Gates and NormActivation and simplified, here we redo it just
        # for reminder purpose
        return self.activation.irreps_in.simplify()

    @property
    def irreps_out(self):
        return self.activation.irreps_out


class UVUTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        *,
        internal_and_share_weights: bool = False,
        mlp_input_size: int = None,
        mlp_hidden_size: int = 8,
        mlp_num_hidden_layers: int = 1,
        mlp_activation: Callable = ACTIVATION["e"]["ssp"],
    ):
        """
        UVU tensor product as in NeuqIP.

        Args:
            irreps_in1: irreps of first input, with available keys in `DataKey`
            irreps_in2: input of second input, with available keys in `DataKey`
            irreps_out: output irreps, with available keys in `DataKey`
            internal_and_share_weights: whether to create weights for the tensor
                product, if `True` all `mlp_*` params are ignored and if `False`,
                they should be provided to create an MLP to transform some data to be
                used as the weight of the tensor product.
            mlp_input_size: size of the input data used as the weight for the tensor
                product transformation via an MLP
            mlp_hidden_size: hidden layer size for the MLP
            mlp_num_hidden_layers: number of hidden layers for the radial MLP, excluding
                input and output layers
            mlp_activation: activation function for the MLP.
        """

        super().__init__()

        # uvu instructions for tensor product
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_in1):
            for j, (_, ir_in2) in enumerate(irreps_in2):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # sort irreps_mid so we can simplify them later
        irreps_mid = Irreps(irreps_mid)
        self.irreps_mid, permutation, _ = irreps_mid.sort()

        assert irreps_mid.dim > 0, (
            f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} produces no "
            f"instructions in irreps_out={irreps_out}"
        )

        # sort instructions accordingly
        instructions = [
            (i_1, i_2, permutation[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        self.tp = TensorProduct(
            irreps_in1,
            irreps_in2,
            self.irreps_mid,
            instructions,
            internal_weights=internal_and_share_weights,
            shared_weights=internal_and_share_weights,
        )

        if not internal_and_share_weights:
            # radial network on scalar edge embedding (e.g. edge distance)
            layer_sizes = (
                [mlp_input_size]
                + mlp_num_hidden_layers * [mlp_hidden_size]
                + [self.tp.weight_numel]
            )
            self.weight_nn = FullyConnectedNet(layer_sizes, act=mlp_activation)
        else:
            self.weight_nn = None

    def forward(
        self, data1: Tensor, data2: Tensor, data_weight: Optional[Tensor] = None
    ) -> Tensor:

        if self.weight_nn is not None:
            assert data_weight is not None, "data for weight not provided"
            weight = self.weight_nn(data_weight)
        else:
            weight = None
        x = self.tp(data1, data2, weight)

        return x

    @property
    def irreps_out(self):
        """
        Output irreps of the layer.

        This is different from the input `irreps_out`, since we we use the UVU tensor
        product with given instructions.
        """

        # the simplify is possible because irreps_mid is sorted
        return self.irreps_mid.simplify()


class ScalarMLP(torch.nn.Module):
    """
    Multilayer perceptron for scalars.

    By default, activation is applied to each hidden layer. For hidden layers:
    Linear -> BN (default to False) -> Activation

    Optionally, one can add an output layer by setting `out_size`. For output layer:
    Linear with the option to use bias of not, activation is not applied

    Args:
        in_size: input feature size
        hidden_sizes: sizes for hidden layers
        batch_norm: whether to add 1D batch norm
        activation: activation function for hidden layers
        out_size: size of output layer
        out_bias: bias for output layer, this use set to False internally if
            out_batch_norm is used.
    """

    def __init__(
        self,
        in_size: int,
        hidden_sizes: List[int],
        *,
        batch_norm: bool = False,
        activation: Callable = ACTIVATION["e"]["ssp"],
        out_size: Optional[int] = None,
        out_bias: bool = True,
    ):
        super().__init__()
        self.num_hidden_layers = len(hidden_sizes)
        self.has_out_layer = out_size is not None

        layers = []

        # hidden layers
        if batch_norm:
            bias = False
        else:
            bias = True

        for size in hidden_sizes:
            layers.append(torch.nn.Linear(in_size, size, bias=bias))

            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(size))

            if activation is not None:
                layers.append(activation)

            in_size = size

        # output layer
        if out_size is not None:
            layers.append(torch.nn.Linear(in_size, out_size, bias=out_bias))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    def __repr__(self):
        s = f"Scalar MLP, num hidden layers: {self.num_hidden_layers}"
        if self.has_out_layer:
            s += "; with output layer"
        return s


def scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """
    Special case of torch_scatter.scatter with dim=0
    """
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)


class DetectAnomaly(ModuleIrreps, torch.nn.Module):
    def __init__(self, irreps_in: Dict[str, Irreps], name: str):
        super().__init__()
        self.init_irreps(irreps_in=irreps_in)

        self.name = name

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        for k, v in data.items():
            if v is None:
                continue

            try:
                detect_nan_and_inf(v)
            except ValueError:
                raise ValueError(f"Anomaly detected for {k} of {self.name}")

        return data
