from typing import Dict, Optional

import torch
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear
from e3nn.util.jit import compile_mode

from matten.data.irreps import DataKey, ModuleIrreps
from matten.nn.utils import ActivationLayer, UVUTensorProduct, scatter_add


class PointConvMessage(ModuleIrreps, torch.nn.Module):

    REQUIRED_KEYS_IRREPS_IN = [
        DataKey.NODE_FEATURES,
        DataKey.EDGE_EMBEDDING,
        DataKey.EDGE_ATTRS,
        DataKey.EDGE_INDEX,
    ]

    REQUIRED_TYPE_IRREPS_IN = {DataKey.EDGE_EMBEDDING: "0e"}

    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        irreps_out: Dict[str, Irreps],
        *,
        fc_num_hidden_layers: int = 1,
        fc_hidden_size: int = 8,
    ):
        """
        Point convolution message function.

        Basically tensor product between node features and edge attributes, based on
        https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/v2106/points_convolution.py

        This generates DataKey.EDGE_MESSAGE.

        Args:
            irreps_in: input irreps, with available keys in `DataKey`
            irreps_out: output irreps, with available keys in `DataKey`
            fc_num_hidden_layers: number of hidden layers for the radial MLP, excluding
                input and output layers
            fc_hidden_size: hidden layer size for the radial MLP
        """

        super().__init__()

        # init irreps
        self.init_irreps(irreps_in, irreps_out)

        node_feats_irreps_in = self.irreps_in[DataKey.NODE_FEATURES]
        node_feats_irreps_out = self.irreps_out[DataKey.NODE_FEATURES]
        edge_attrs_irreps = self.irreps_in[DataKey.EDGE_ATTRS]

        # first linear on node feats
        self.linear_1 = Linear(
            irreps_in=node_feats_irreps_in,
            irreps_out=node_feats_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )

        # tensor product for message function
        self.tp = UVUTensorProduct(
            node_feats_irreps_in,
            edge_attrs_irreps,
            node_feats_irreps_out,
            mlp_input_size=self.irreps_in[DataKey.EDGE_EMBEDDING].dim,
            mlp_hidden_size=fc_hidden_size,
            mlp_num_hidden_layers=fc_num_hidden_layers,
        )

        # keep track of irreps out
        self.irreps_out[DataKey.EDGE_MESSAGE] = self.tp.irreps_out

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        edge_src, edge_dst = data[DataKey.EDGE_INDEX]

        node_feats = self.linear_1(data[DataKey.NODE_FEATURES])

        msg_per_edge = self.tp(
            node_feats[edge_src], data[DataKey.EDGE_ATTRS], data[DataKey.EDGE_EMBEDDING]
        )

        data[DataKey.EDGE_MESSAGE] = msg_per_edge

        return data


class PointConvUpdate(ModuleIrreps, torch.nn.Module):
    REQUIRED_KEYS_IRREPS_IN = [
        DataKey.NODE_FEATURES,
        DataKey.NODE_ATTRS,
        DataKey.EDGE_MESSAGE,
        DataKey.EDGE_INDEX,
    ]

    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        irreps_out: Dict[str, Irreps],
        *,
        use_self_connection: bool = True,
        avg_num_neighbors: int = None,
    ):
        """
        Point convolution update function.

        This updates DataKey.NOTE_FEATURES.

        Args:
            irreps_in:
            irreps_out:
            use_self_connection:
            avg_num_neighbors:
        """
        super().__init__()

        self.avg_num_neighbors = avg_num_neighbors

        # init irreps
        self.init_irreps(irreps_in, irreps_out)

        node_feats_irreps_in = self.irreps_in[DataKey.NODE_FEATURES]
        node_feats_irreps_out = self.irreps_out[DataKey.NODE_FEATURES]
        node_attrs_irreps = self.irreps_in[DataKey.NODE_ATTRS]
        edge_message_irreps_in = self.irreps_in[DataKey.EDGE_MESSAGE]

        # second linear
        self.linear_2 = Linear(
            irreps_in=edge_message_irreps_in,
            irreps_out=node_feats_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        # # inspired by https://arxiv.org/pdf/2002.10444.pdf
        # self.alpha = FullyConnectedTensorProduct(
        #     self.tp.irreps_out, node_attrs_irreps, "0e"
        # )
        # with torch.no_grad():
        #     self.alpha.weight.zero_()
        # assert self.alpha.output_mask[0] == 1.0, (
        #     f"Unable to generate scalar (0e) from "
        #     f"self.tp.irreps_out={self.tp.irreps_out} and "
        #     f"node_attrs_irreps ={node_attrs_irreps} "
        # )

        #
        # self connection
        #
        # We can use node_attrs_irreps here (which is not related to spherical harmonics
        # embedding of anything because it is just the embedding of species info, they
        # are scalars. So it is possible. For higher order
        # tensors, they cannot be used in the tensor product to product equivariant
        # layer. For example, you cannot tensor product two node feats and expect
        # them to be equivariant.

        if use_self_connection:
            self.self_connection = FullyConnectedTensorProduct(
                node_feats_irreps_in, node_attrs_irreps, node_feats_irreps_out
            )
        else:
            self.self_connection = None

    def forward(self, data: DataKey.Type) -> DataKey.Type:

        node_feats = data[DataKey.NODE_FEATURES]

        edge_src, edge_dst = data[DataKey.EDGE_INDEX]

        # aggregate message
        msg = scatter_add(
            data[DataKey.EDGE_MESSAGE], edge_dst, dim_size=len(node_feats)
        )

        if self.avg_num_neighbors is not None:
            msg = msg / self.avg_num_neighbors**0.5

        msg = self.linear_2(msg)

        #
        # update step
        #
        if self.self_connection is not None:
            node_feats = msg + self.self_connection(
                node_feats, data[DataKey.NODE_ATTRS]
            )
        else:
            node_feats = msg

        data[DataKey.NODE_FEATURES] = node_feats

        return data


@compile_mode("script")
class PointConvMessagePassing(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        conv_layer_irreps: Irreps,
        message=PointConvMessage,
        update=PointConvUpdate,
        message_kwargs: Optional[Dict] = None,
        update_kwargs: Optional[Dict] = None,
        activation_type: str = "gate",
        activation_scalars: Dict[str, str] = None,
        activation_gates: Dict[str, str] = None,
        use_resnet: bool = True,
    ):
        """
        TFN layer, basically TFN message_fn with nonlinear activation functions.

        input -> message_fn layer -> nonlinear activation -> resnet (optional)

        Args:
            irreps_in:
            conv_layer_irreps: irreps for the node features in the message and update
                functions
            message: the class implements the message function, should be a subclass
                of `BaseMessage`
            update: the class implements the update function, should be a subclass
                of `BaseUpdate`
            message_kwargs: dict of kwargs passed to `message`
            update_kwargs: dict of kwargs passed to `update`
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
        message_kwargs = {} if message_kwargs is None else message_kwargs
        update_kwargs = {} if update_kwargs is None else update_kwargs

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
        # -->(irreps_before_act)--> activation
        # -->(irreps_after_act)
        #
        # We initialize ActivationLayer before the message_fn layer because the activation
        # requires special treatment of its input irreps. This special treatment also
        # determines the output irreps of the message_fn layer (i.e. irreps_before_act)

        self.equivariant_nonlinear = ActivationLayer(
            tp_irreps_in1=node_feats_irreps_in,
            tp_irreps_in2=edge_attrs_irreps,
            tp_irreps_out=conv_layer_irreps,
            activation_type=activation_type,
            activation_scalars=activation_scalars,
            activation_gates=activation_gates,
        )
        irreps_before_act = self.equivariant_nonlinear.irreps_in
        irreps_after_act = self.equivariant_nonlinear.irreps_out

        #
        # message layer
        #
        # This is applied before activation, but need to initialize after
        # activation because when using `Gate`, conv_irreps_out needs to be set based on
        # Gate.irreps_in
        # Remove user set irreps_in and irreps_out (if any) to use what determined here
        message_kwargs.pop("irreps_in", None)
        message_kwargs.pop("irreps_out", None)
        self.message_fn = message(
            irreps_in=self.irreps_in,
            irreps_out={DataKey.NODE_FEATURES: irreps_before_act},  # can be whatever
            **message_kwargs,
        )

        #
        # update layer
        #
        update_fn_irreps_in = self.irreps_in.copy()
        update_fn_irreps_in[DataKey.EDGE_MESSAGE] = self.message_fn.irreps_out[
            DataKey.EDGE_MESSAGE
        ]

        update_kwargs.pop("irreps_in", None)
        update_kwargs.pop("irreps_out", None)
        self.update_fn = update(
            irreps_in=update_fn_irreps_in,
            irreps_out={DataKey.NODE_FEATURES: irreps_before_act},
            **update_kwargs,
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
        # The output features are whatever we got updated from the update_fn outputs,
        # but the irreps of node features should be from the activation
        self.irreps_out.update(self.update_fn.irreps_out)
        self.irreps_out[DataKey.NODE_FEATURES] = irreps_after_act

    def forward(self, data: DataKey.Type) -> DataKey.Type:

        # save old features for resnet
        old_x = data[DataKey.NODE_FEATURES]

        # run message passing
        data = self.message_fn(data)
        data = self.update_fn(data)

        # nonlinear activation
        x = data[DataKey.NODE_FEATURES]
        x = self.equivariant_nonlinear(x)

        # resnet
        if self.use_resnet:
            data[DataKey.NODE_FEATURES] = old_x + x
        else:
            data[DataKey.NODE_FEATURES] = x

        return data
