from typing import Dict

import torch
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear, TensorProduct
from torch_scatter import scatter

from eigenn.nn.irreps import DataKey, ModuleIrreps
from eigenn.nn.utils import UVUTensorProduct


class TFNConv(ModuleIrreps, torch.nn.Module):
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
        *,
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
        self.tp = UVUTensorProduct(
            node_feats_irreps_in,
            edge_attrs_irreps,
            node_feats_irreps_out,
            mlp_input_size=self.irreps_in[DataKey.EDGE_EMBEDDING].dim,
            mlp_hidden_size=fc_hidden_size,
            mlp_num_hidden_layers=fc_num_hidden_layers,
        )

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
            irreps_in=self.tp.irreps_out.simplify(),
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
        # We can use node_attrs_irreps here (which is not related spherical harmonics
        # embedding of anything because in Nequip, node_attrs are simply embedding of
        # species info, they are scalers. So it is possible. For higher order
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
        msg_per_edge = self.tp(node_feats[edge_src], edge_attrs, edge_embedding)

        # aggregate message
        msg = scatter(msg_per_edge, edge_dst, dim=0, dim_size=len(node_feats))

        if self.avg_num_neighbors is not None:
            msg = msg / self.avg_num_neighbors ** 0.5

        msg = self.linear_2(msg)

        #
        # update step
        #
        if self.self_connection is not None:
            node_feats = msg + self.self_connection(node_feats_in, node_attrs)
        else:
            node_feats = msg

        data[DataKey.NODE_FEATURES] = node_feats

        return data
