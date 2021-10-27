from typing import Dict, Optional

import torch
from e3nn.nn import Gate, NormActivation
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from nequip.utils.tp_utils import tp_path_exists

from eigenn.nn.irreps import DataKey, ModuleIrreps
from eigenn.nn.tfn_conv import TFNConv
from eigenn.nn.utils import ACTIVATION


@compile_mode("script")
class EquivariantLayer(ModuleIrreps, torch.nn.Module):

    REQUIRED_KEYS_IRREPS_IN = [
        DataKey.NODE_FEATURES,
        DataKey.NODE_ATTRS,
        DataKey.EDGE_EMBEDDING,
        DataKey.EDGE_ATTRS,
    ]

    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        conv_layer_irreps: Irreps,
        conv=TFNConv,
        conv_kwargs: Optional[Dict] = None,
        activation_type: str = "gate",
        activation_scalars: Dict[str, str] = None,
        activation_gates: Dict[str, str] = None,
        use_resnet: bool = True,
    ):
        """
        TFN layer, basically TFN conv with nonlinear activation functions.

        input -> conv layer -> nonlinear activation -> resnet (optional)

        Args:
            irreps_in:
            conv_layer_irreps: irreps for the node features in the conv layer
            conv:
            conv_kwargs: dict of kwargs passed to `conv`
            activation_type: `gate` or `norm`
            activation_scalars: activation function for scalar irreps (i.e. l=0).
                Should be something like {'e':act_e, 'o':act_o}, where `act_e` is the
                name of the activation function ('ssp' or 'silu') for even irreps;
                `act_o` is the name of the activation function ('abs' or 'tanh') for
                odd irreps.
            activation_gates: activation function for tensor irreps (i.e. l>0).
                Should be something like {'e':act_e, 'o':act_o}, where `act_e` is the
                name of the activation function ('ssp' or 'silu') for even irreps;
                `act_o` is the name of the activation function ('abs' or 'tanh') for
                odd irreps.
            use_resnet:
        """
        super().__init__()

        #
        # set defaults
        #
        conv_kwargs = {} if conv_kwargs is None else conv_kwargs

        # activation function for even (i.e. 1) and odd (i.e. -1) irreps
        if activation_scalars is None:
            activation_scalars = {
                1: ACTIVATION["e"]["ssp"],
                -1: ACTIVATION["o"]["tanh"],
            }
        else:
            # change key from e or v to 1 or -1
            key_mapping = {"e": 1, "o": -1}
            activation_scalars = {
                key_mapping[k]: ACTIVATION[k][v] for k, v in activation_scalars.items()
            }
        if activation_gates is None:
            activation_gates = {1: ACTIVATION["e"]["ssp"], -1: ACTIVATION["o"]["abs"]}
        else:
            # change key from e or v to 1 or -1
            key_mapping = {"e": 1, "o": -1}
            activation_gates = {
                key_mapping[k]: ACTIVATION[k][v] for k, v in activation_gates.items()
            }

        #
        # irreps_out will be set later when we know them
        #
        self.init_irreps(irreps_in)

        edge_attrs_irreps = self.irreps_in[DataKey.EDGE_ATTRS]
        node_feats_irreps_in = self.irreps_in[DataKey.NODE_FEATURES]

        #
        # Nonlinear activation
        #
        # Chose activation function and determine their irreps
        # The flow of the irreps are as follows:
        # (node_feats_irreps_in) --> conv_layer
        # -->(conv_irreps_out)--> activation
        # -->(irreps_after_act)
        #

        ir, _, _ = Irreps(conv_layer_irreps).sort()
        conv_layer_irreps = ir.simplify()

        irreps_scalars = Irreps(
            [
                (mul, ir)
                for mul, ir in conv_layer_irreps
                if ir.l == 0
                and tp_path_exists(node_feats_irreps_in, edge_attrs_irreps, ir)
            ]
        )
        irreps_gated = Irreps(
            [
                (mul, ir)
                for mul, ir in conv_layer_irreps
                if ir.l > 0
                and tp_path_exists(node_feats_irreps_in, edge_attrs_irreps, ir)
            ]
        )

        if activation_type == "gate":
            ir = (
                "0e"
                if tp_path_exists(node_feats_irreps_in, edge_attrs_irreps, "0e")
                else "0o"
            )
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])

            equivariant_nonlinear = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[activation_scalars[ir.p] for _, ir in irreps_scalars],
                irreps_gates=irreps_gates,
                act_gates=[activation_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

            # conv layer is applied before activation, so conv_irreps_out is the same
            # as equivariant_nonlinear.irreps_in
            conv_irreps_out = equivariant_nonlinear.irreps_in.simplify()

        elif activation_type == "norm":
            conv_irreps_out = (irreps_scalars + irreps_gated).simplify()

            equivariant_nonlinear = NormActivation(
                irreps_in=conv_irreps_out,
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

        self.equivariant_nonlinear = equivariant_nonlinear
        irreps_after_act = self.equivariant_nonlinear.irreps_out

        #
        # Conv layer
        #
        # This is applied before activation, but need to initialize after
        # activation because when using `Gate`, conv_irreps_out needs to be set based on
        # Gate.irreps_in
        # Remove user set irreps_in and irreps_out (if any) to use what determined here
        conv_kwargs.pop("irreps_in", None)
        conv_kwargs.pop("irreps_out", None)
        self.conv = conv(
            irreps_in=self.irreps_in,
            irreps_out={DataKey.NODE_FEATURES: conv_irreps_out},
            **conv_kwargs,
        )

        #
        # Resnet
        #
        # TODO: partial resnet?
        if use_resnet and irreps_after_act == node_feats_irreps_in:
            self.use_resnet = True
        else:
            self.use_resnet = False

        #
        # Set irreps_out
        #
        # The output features are whatever we got updated from the conv outputs,
        # but the irreps of node features should be from the activation
        self.irreps_out.update(self.conv.irreps_out)
        self.irreps_out[DataKey.NODE_FEATURES] = irreps_after_act

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        # shallow copy to not modify the input
        data = data.copy()

        # save old features for resnet
        old_x = data[DataKey.NODE_FEATURES]

        # run convolution
        data = self.conv(data)

        # nonlinear activation
        x = data[DataKey.NODE_FEATURES]
        x = self.equivariant_nonlinear(x)

        # resnet
        if self.use_resnet:
            data[DataKey.NODE_FEATURES] = old_x + x
        else:
            data[DataKey.NODE_FEATURES] = x

        return data
