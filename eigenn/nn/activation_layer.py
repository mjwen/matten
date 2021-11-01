from typing import Dict

import torch
from e3nn.nn import Gate, NormActivation
from e3nn.o3 import Irreps
from nequip.utils.tp_utils import tp_path_exists
from torch import Tensor

from eigenn.nn.utils import ACTIVATION


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
                # odd scalars requires either an even or ordd activation,
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
