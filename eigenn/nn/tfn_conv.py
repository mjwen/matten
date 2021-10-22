from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as fn
from e3nn.nn import FullyConnectedNet, Gate, NormActivation
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear, TensorProduct
from e3nn.util.jit import compile_mode
from nequip.nn.nonlinearities import ShiftedSoftPlus
from nequip.utils.tp_utils import tp_path_exists
from torch_scatter import scatter

from eigenn.nn.irreps import DataKey, ModuleIrreps

ACTIVATION = {
    # for even irreps
    "e": {
        "ssp": ShiftedSoftPlus,
        "silu": torch.nn.functional.silu,
    },
    # for odd irreps
    "o": {
        "abs": torch.abs,
        "tanh": torch.tanh,
    },
}


@compile_mode("script")
class TFNConv(nn.Module, ModuleIrreps):
    """
    TFN convolution.
    """

    REQUIRED_KEYS_IRREPS_IN = [
        DataKey.NODE_FEATURES,
        DataKey.NODE_ATTRS,
        DataKey.EDGE_EMBEDDING,
        DataKey.EDGE_ATTRS,
        DataKey.EDGE_INDEX,
    ]

    REQUIRED_TYPE_IRREPS_IN = {DataKey.EDGE_EMBEDDING: "0e"}

    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        irreps_out: Dict[str, Irreps],
        fc_num_hidden_layers: int = 1,
        fc_hidden_size: int = 8,
        activation_scalars: Dict[str, str] = None,
        use_self_connection: bool = True,
        avg_num_neighbors: int = None,
    ):
        """
        The Tensor Field Network equivariant convolution layer.

        This is based on `Convolution` in `e3nn.nn.models.v2106.point_convolution.py`
        and `InteractionBlock` in `nequip.nn._interaction_block.py`

        Args:
            irreps_in: input irreps, with available keys in `DataKey`
            irreps_out: output irreps, with available keys in `DataKey`
            fc_num_hidden_layers: number of hidden layers for the radial MLP, excluding
                input and output layers
            fc_hidden_size: hidden layer size for the radial MLP
            use_self_connection: whether to use self interaction, e.g. Eq.10 in the
                SE3-Transformer paper.
            activation_scalars: activation function for scalar irreps (i.e. l=0).
                Should be something like {'e':act_e, 'o':act_o}, where `act_e` is the
                name of the activation function ('ssp' or 'silu') for even irreps;
                `act_o` is the name of the activation function ('abs' or 'tanh') for
                odd irreps.
        """

        super().__init__()

        self.avg_num_neighbors = avg_num_neighbors

        # init irreps
        self.init_irreps(irreps_in, irreps_out)

        node_feats_irreps_in = self.irreps_in[DataKey.NODE_FEATURES]
        node_feats_irreps_out = self.irreps_out[DataKey.NODE_FEATURES]
        node_attrs_irreps_in = self.irreps_in[DataKey.NODE_ATTRS]
        edge_attrs_irreps = self.irreps_in[DataKey.EDGE_ATTRS]

        #
        # Convolution:
        #
        # linear_1 on node feats  -->
        # tensor product on node feats and edge attrs -->
        # linear_2 on node feats --> output

        # first linear on node feats
        self.linear_1 = Linear(
            irreps_in=node_feats_irreps_in,
            irreps_out=node_feats_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )

        # Construct `uvu` tensor product instructions for node feats and edge attrs
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(node_feats_irreps_in):
            for j, (_, ir_edge) in enumerate(edge_attrs_irreps):
                for ir_out in ir_in * ir_edge:
                    if ir_out in node_feats_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # sort irreps_mid so we can simplify them when providing to self.linear_2 below
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, permutation, _ = irreps_mid.sort()

        # sort instructions accordingly
        instructions = [
            (i_1, i_2, permutation[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        assert irreps_mid.dim > 0, (
            f"node_feats_irreps_in={node_feats_irreps_in} times "
            f"edge_attrs_irreps={edge_attrs_irreps} produces nothing in "
            f"node_feats_irreps_out={node_feats_irreps_out}"
        )

        # tensor product
        self.tp = TensorProduct(
            node_feats_irreps_in,
            edge_attrs_irreps,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )

        # radial network on scalar edge embedding (e.g. edge distance)
        layer_sizes = (
            [self.irreps_in[DataKey.EDGE_EMBEDDING].num_irreps]
            + fc_num_hidden_layers * [fc_hidden_size]
            + [self.tp.weight_numel]
        )
        if activation_scalars is None:
            act = ACTIVATION["e"]["ssp"]
        else:
            act = ACTIVATION["e"][activation_scalars["e"]]
        self.radial_nn = FullyConnectedNet(layer_sizes, act=act)

        # second linear on node feats
        #
        # In the tensor product using `uvu` instruction above, its output (i.e.
        # irreps_mid) can be uncoallesed (for example, irreps_mid =1x0e+2x0e+2x1e).
        # Here, we simplify it to combine irreps of the same the together (for example,
        # irreps_mid.simplify()='3x0e+2x1e').
        # The normalization in Linear is different for unsimplified and simplified
        # irreps.
        #
        # Note, the data corresponds to irreps_mid before and after simplification will
        # be the same, i.e. their order does not change, since irreps_mid is sorted.
        self.linear_2 = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=node_feats_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        # # inspired by https://arxiv.org/pdf/2002.10444.pdf
        # self.alpha = FullyConnectedTensorProduct(irreps_mid, node_attrs_irreps_in, "0e")
        # with torch.no_grad():
        #     self.alpha.weight.zero_()
        # assert (
        #     self.alpha.output_mask[0] == 1.0
        # ), f"irreps_mid={irreps_mid} and irreps_node_attr={self.irreps_node_attr} are not able to generate scalars"

        #
        # self connection
        #
        if use_self_connection:
            self.self_connection = FullyConnectedTensorProduct(
                node_feats_irreps_in, node_attrs_irreps_in, node_feats_irreps_out
            )
        else:
            self.self_connection = None

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        # shallow copy to avoid modifying the input
        data = data.copy()

        node_feats_in = data[DataKey.NODE_FEATURES]
        node_attrs = data[DataKey.NODE_ATTRS]
        edge_embedding = data[DataKey.EDGE_EMBEDDING]
        edge_attrs = data[DataKey.EDGE_ATTRS]

        edge_src = data[DataKey.EDGE_INDEX][0]
        edge_dst = data[DataKey.EDGE_INDEX][1]

        node_feats = self.linear_1(node_feats_in)

        weight = self.radial_nn(edge_embedding)
        edge_feats = self.tp(node_feats[edge_src], edge_attrs, weight)
        node_feats = scatter(edge_feats, edge_dst, dim=0, dim_size=len(node_feats))

        if self.avg_num_neighbors is not None:
            node_feats = node_feats / self.avg_num_neighbors ** 0.5

        node_feats = self.linear_2(node_feats)

        # alpha = self.alpha(node_features, node_attr)
        #
        # m = self.sc.output_mask
        # alpha = (1 - m) + alpha * m
        #
        # return node_self_connection + alpha * node_conv_out

        if self.self_connection is not None:
            node_feats = node_feats + self.self_connection(node_feats_in, node_attrs)

        data[DataKey.NODE_FEATURES] = node_feats

        return data


# TODO, the part to apply nonlinearity can be write as a separate class for reuse
@compile_mode("script")
class TFNLayer(ModuleIrreps, nn.Module):

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
