from typing import Callable, Dict, Optional

import torch
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear
from torch_scatter import scatter

from eigenn.nn.irreps import DataKey, ModuleIrreps
from eigenn.nn.utils import ACTIVATION, ActivationLayer, ScalarMLP, UVUTensorProduct


class SEGNNConv(ModuleIrreps, torch.nn.Module):
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
        conv_layer_irreps: Irreps,
        *,
        fc_num_hidden_layers: int = 1,
        fc_hidden_size: int = 8,
        activation_type: str = "gate",
        activation_scalars: Dict[str, str] = None,
        activation_gates: Dict[str, str] = None,
        use_self_connection: bool = True,
        avg_num_neighbors: int = None,
    ):
        """

        SEGNN message passing.

        Message function: tensor product between node feats of neighbors and edge
        embedding, with params derived from fully connected layer on edge distance.

        Update function: tensor product between aggregated message from neighbors nodes
        and node feats of atom i, with params derived from node attrs (e.g. initial species
        embedding).

        Alternative update function: tensor product between aggregated message from
        neighbors and steerable node attrs (e.g. sum of edge embedding of neighbors plus
        optionally steerable node feats, e.g. node force, velocity)
        With params derived from node attrs (e.g. initial species embedding).

        We should use alternative update function. The reasoning is that:
        in the message function, all we care is neighboring atoms j, then in the tensor
        product, we do tensor product between its node features (trainable) and the SH
        edge embedding (not trainable).
        To follow this idea, when do update, we care about node i, we should do tensor
        product between the aggregated message from all j (trainable) (in fact we can
        combine i and the aggregate j features. No matter how we do it, the underlying
        principle is that this is trainable) and some feature that is not trainable.
        Apparently the alternative update function satisfy this.

        But why we want the second component in the tensor product be not trainable?
        Well, this serves as a kernel, although the kernel is trainable, we should not
        parameterize the kernel on features, but can on attrs, or we can initializes the
        weights to follow some distribution, not making it dependent on feats or attrs.

        Args:
            irreps_in: input irreps, with available keys in `DataKey`
            conv_layer_irreps: irreps for the node features in the conv layer
            fc_num_hidden_layers: number of hidden layers for the radial MLP, excluding
                input and output layers
            fc_hidden_size: hidden layer size for the radial MLP
            use_self_connection: whether to use self interaction, e.g. Eq.10 in the
                SE3-Transformer paper.
        """

        super().__init__()

        self.avg_num_neighbors = avg_num_neighbors

        # init irreps
        # irreps_out will be set later when we know them
        self.init_irreps(irreps_in)

        node_feats_irreps_in = self.irreps_in[DataKey.NODE_FEATURES]
        node_attrs_irreps = self.irreps_in[DataKey.NODE_ATTRS]
        edge_attrs_irreps = self.irreps_in[DataKey.EDGE_ATTRS]
        conv_layer_irreps = Irreps(conv_layer_irreps)

        #
        # message step
        #

        # nonlinear activation function for message
        # (this is applied after the tensor product, but we need to init it first to
        # determine its irreps in, which is the irreps_out for tensor product)
        self.message_activation = ActivationLayer(
            node_feats_irreps_in,
            edge_attrs_irreps,
            conv_layer_irreps,
            activation_type=activation_type,
            activation_scalars=activation_scalars,
            activation_gates=activation_gates,
        )
        irreps_before_message_act = self.message_activation.irreps_in
        irreps_after_message_act = self.message_activation.irreps_out

        # TODO, linear_1 is not necessary to make the model work, used to increase
        #  model capacity
        # first linear on node feats
        self.linear_1 = Linear(
            irreps_in=node_feats_irreps_in,
            irreps_out=node_feats_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )

        # tensor product for message function
        self.message_tp = UVUTensorProduct(
            node_feats_irreps_in,
            edge_attrs_irreps,
            irreps_before_message_act,
            mlp_input_size=self.irreps_in[DataKey.EDGE_EMBEDDING].dim,
            mlp_hidden_size=fc_hidden_size,
            mlp_num_hidden_layers=fc_num_hidden_layers,
        )

        # second linear on node feats
        # This layer is necessary to convert the irreps_out of the tensor product to
        # the irreps_in of message_activation.
        # Although self.message_activation.irreps_in is used as the input irreps_out
        # for message_tp, the actual irreps_out of message_tp may not be equal to it
        self.linear_2 = Linear(
            irreps_in=self.message_tp.irreps_out,
            irreps_out=irreps_before_message_act,
            internal_weights=True,
            shared_weights=True,
        )

        #
        # update step
        #
        self.update_activation = ActivationLayer(
            irreps_after_message_act,
            node_attrs_irreps,
            conv_layer_irreps,
            activation_type=activation_type,
            activation_scalars=activation_scalars,
            activation_gates=activation_gates,
        )
        irreps_before_update_act = self.update_activation.irreps_in
        irreps_after_update_act = self.update_activation.irreps_out

        # Note, here the weights does not depend on node attrs and such, so we share
        # them. Of course, wen can make them dependent of scalar node attrs as done
        # above.
        self.update_tp = UVUTensorProduct(
            irreps_after_message_act,
            node_attrs_irreps,
            irreps_before_update_act,
            internal_and_share_weights=True,
        )

        self.linear_3 = Linear(
            irreps_in=self.update_tp.irreps_out,
            irreps_out=irreps_before_update_act,
            internal_weights=True,
            shared_weights=True,
        )

        #
        # self connection
        #
        if use_self_connection:
            self.self_connection = FullyConnectedTensorProduct(
                node_feats_irreps_in, node_attrs_irreps, irreps_before_update_act
            )
        else:
            self.self_connection = None

        #
        # Set irreps_out
        #
        self.irreps_out[DataKey.NODE_FEATURES] = irreps_after_update_act

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        # shallow copy to avoid modifying the input
        data = data.copy()

        node_feats_in = data[DataKey.NODE_FEATURES]
        node_attrs = data[DataKey.NODE_ATTRS]
        edge_embedding = data[DataKey.EDGE_EMBEDDING]
        edge_attrs = data[DataKey.EDGE_ATTRS]
        edge_src, edge_dst = data[DataKey.EDGE_INDEX]

        #
        # message step
        #
        node_feats = self.linear_1(node_feats_in)
        msg_per_edge = self.message_tp(node_feats[edge_src], edge_attrs, edge_embedding)
        msg_per_edge = self.linear_2(msg_per_edge)
        msg_per_edge = self.message_activation(msg_per_edge)

        # aggregate message
        msg = scatter(msg_per_edge, edge_dst, dim=0, dim_size=len(node_feats))

        if self.avg_num_neighbors is not None:
            msg = msg / self.avg_num_neighbors ** 0.5

        #
        # update step
        #
        node_feats = self.update_tp(msg, node_attrs)
        node_feats = self.linear_3(node_feats)

        if self.self_connection is not None:
            node_feats = node_feats + self.self_connection(node_feats_in, node_attrs)

        node_feats = self.update_activation(node_feats)

        data[DataKey.NODE_FEATURES] = node_feats

        return data


class MeanPredictionHead(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        field: str = DataKey.NODE_FEATURES,
        out_field: Optional[str] = None,
        activation: Callable = ACTIVATION["e"]["ssp"],
        hidden_size: int = None,
    ):
        """
        Prediction head for scalar property.

        -> linear1 to select 0e
        -> activation
        -> linear2
        -> reduce (mean)
        after
        -> linear
        -> activation
        -> linear

        The hidden

        Args:
            irreps_in:
            field:
            out_field:
            activation:
            hidden_size: hidden layer size, if `None`, use the multiplicity of the
                0e irreps
        """
        super().__init__()
        self.init_irreps(irreps_in, required_keys_irreps_in=[field])

        self.field = field
        self.out_field = field if out_field is None else out_field

        field_irreps = self.irreps_in[self.field]

        if hidden_size is None:
            # get multiplicity of 0e irreps
            for mul, ir in field_irreps:
                if ir == (0, 1):
                    hidden_size = mul
                    break
            assert (
                hidden_size is not None
            ), f"`irreps_in[{field}] = {field_irreps}`, not contain `0e`"

        linear1 = Linear(field_irreps, Irreps(f"{hidden_size}x0e"))
        activation = activation
        linear2 = torch.nn.Linear(hidden_size, hidden_size)

        self.mlp1 = torch.nn.Sequential(linear1, activation, linear2)
        self.mlp2 = ScalarMLP(
            in_size=hidden_size,
            hidden_sizes=[hidden_size],
            activation=activation,
            out_size=1,
        )

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        x = data[self.field]

        x = self.mlp1(x)
        x = scatter(x, data[DataKey.BATCH], dim=0, reduce="mean")
        data[self.out_field] = self.mlp2(x)

        return data


class EmbeddingLayer(ModuleIrreps, torch.nn.Module):
    REQUIRED_KEYS_IRREPS_IN = [
        DataKey.NODE_FEATURES,
        DataKey.NODE_ATTRS,
    ]

    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        irreps_out: Dict[str, Irreps],
    ):
        """
        Node embedding layers.

        Tensor product between node feats and node attrs.

        Args:
            irreps_in:
            irreps_out: expected irreps out, actual irreps out determined within layer
        """

        super().__init__()

        self.init_irreps(irreps_in, irreps_out)

        node_feats_irreps_in = self.irreps_in[DataKey.NODE_FEATURES]
        node_attrs_irreps = self.irreps_in[DataKey.NODE_ATTRS]
        node_feats_irreps_out = self.irreps_out[DataKey.NODE_FEATURES]

        self.activation = ActivationLayer(
            node_feats_irreps_in, node_attrs_irreps, node_feats_irreps_out
        )

        # tensor product between node feats and node attrs
        self.tp = FullyConnectedTensorProduct(
            node_feats_irreps_in, node_attrs_irreps, self.activation.irreps_in
        )

        # store actual irreps out
        self.irreps_out[DataKey.NODE_FEATURES] = self.activation.irreps_out

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        node_feats = data[DataKey.NODE_FEATURES]
        node_attrs = data[DataKey.NODE_ATTRS]

        node_feats = self.tp(node_feats, node_attrs)
        node_feats = self.activation(node_feats)

        data[DataKey.NODE_FEATURES] = node_feats

        return data