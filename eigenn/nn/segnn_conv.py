"""
Equivariant message passing layer based on SEGNN, which uses tensor product for both
message passing and update function.

https://arxiv.org/abs/2110.02905
"""

from typing import Dict

from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear, TensorProduct
from torch import Tensor
from torch_scatter import scatter

from eigenn.nn.irreps import DataKey, ModuleIrreps
from eigenn.nn.utils import ACTIVATION, get_uvu_instructions


class SEGNNConv(ModuleIrreps, MessagePassing):
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

        SEGNN conv layer.

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
        node_attrs_irreps = self.irreps_in[DataKey.NODE_ATTRS]
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

        # tensor product for message function
        instructions, irreps_mid = get_uvu_instructions(
            node_feats_irreps_in, edge_attrs_irreps, node_feats_irreps_out
        )
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

        #
        # tensor product for update function
        #
        instructions, _ = get_uvu_instructions(
            node_feats_irreps_out, node_attrs_irreps, node_feats_irreps_out
        )
        # TODO, here the weights does not depend node feats and such, so we share them
        # of course, wen can make them dependent of scalar node attrs and then use a
        # radial network to get the weights as done above
        # Also, this can be thought as the self-connection step
        self.tp_update = TensorProduct(
            node_feats_irreps_out,
            node_attrs_irreps,
            node_feats_irreps_out,
            instructions,
            internal_weights=True,
            shared_weights=True,
        )
        # TODO we can explore to add another linear_layer3 after the tp_update, should
        # think about what this means and whether it is necessary. Or maybe move
        # linear_2 after self.tp_update.

        #
        # self connection
        #
        if use_self_connection:
            self.self_connection = FullyConnectedTensorProduct(
                node_feats_irreps_in, node_attrs_irreps, node_feats_irreps_out
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
        edge_src, edge_dst = data[DataKey.EDGE_INDEX]

        node_feats = self.linear_1(node_feats_in)

        #
        # message passing step
        #
        weight = self.radial_nn(edge_embedding)
        msg_per_edge = self.tp(node_feats[edge_src], edge_attrs, weight)

        # aggregate message
        msg = scatter(msg_per_edge, edge_dst, dim=0, dim_size=len(node_feats))

        if self.avg_num_neighbors is not None:
            msg = msg / self.avg_num_neighbors ** 0.5

        msg = self.linear_2(msg)

        #
        # update step
        #
        node_attrs = self._get_node_attrs(data)
        node_feats = self.tp_update(msg, node_attrs)

        if self.self_connection is not None:
            node_feats = node_feats + self.self_connection(node_feats_in, node_attrs)

        data[DataKey.NODE_FEATURES] = node_feats

        return data

    # TODO Since attrs are fixed, this should be done only once for each structure and
    # should be moved to other places, maybe a module to do it.
    @staticmethod
    def _get_node_attrs(data: DataKey.Type, reduce="mean") -> Tensor:
        """
        Get the attrs of each atom, which is mean of attrs of the neighboring atoms.
        """
        # node_attrs = data[DataKey.NODE_ATTRS]
        # edge_index = data[DataKey.EDGE_INDEX]

        # # get the neighboring atoms' attrs
        # edge_attrs = scatter(
        #     node_attrs[edge_index[1]], edge_index[0], dim=0, reduce=reduce
        # )

        # return edge_attrs

        return data[DataKey.NODE_ATTRS]
