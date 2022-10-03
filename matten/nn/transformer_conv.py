from typing import Callable, Dict

import torch
from e3nn.math import soft_unit_step
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear
from torch_scatter import scatter

from matten.data.irreps import DataKey, ModuleIrreps
from matten.nn.utils import UVUTensorProduct


class TransformerConv(ModuleIrreps, torch.nn.Module):

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
        irreps_query_and_key: Irreps,
        r_max: float,
        *,
        fc_num_hidden_layers: int = 1,
        fc_hidden_size: int = 8,
        activation_scalars: Dict[str, Callable] = None,
        use_self_connection: bool = True,
        avg_num_neighbors: int = None,
    ):
        """
        The SE3-Transformer equivariant layer: https://arxiv.org/abs/2006.10503

        Args:
            irreps_in: input irreps, with available keys in `DataKey`
            irreps_out: output irreps, with available keys in `DataKey`, this is
                typically the node features. If None, will set it to the irreps of node
                features in irreps_in.
            irreps_query_and_key: irreps for the query and key
            r_max: cutoff distance, should be the same as used for building neighlist
            fc_num_hidden_layers: number of hidden layers for the radial MLP, excluding
                input and output layers
            fc_hidden_size: hidden layer size for the radial MLP
            activation_scalars: activation function for scalars (i.e. irreps l=0).
                Should be something like {'e': act1 'o':act2}, where act1 is for even
                parity irreps (i.e. 0e) and act2 is for odd parity irreps (i.e. 0o);
                both should be callable.
            use_self_connection: whether to use self interaction, e.g. Eq.10 in the
                SE3-Transformer paper.
        """
        super().__init__()

        self.avg_num_neighbors = avg_num_neighbors
        self.r_max = r_max

        # init irreps
        self.init_irreps(irreps_in, irreps_out)

        node_feats_irreps_in = self.irreps_in[DataKey.NODE_FEATURES]
        node_feats_irreps_out = self.irreps_out[DataKey.NODE_FEATURES]
        node_attrs_irreps = self.irreps_in[DataKey.NODE_ATTRS]
        edge_attrs_irreps = self.irreps_in[DataKey.EDGE_ATTRS]

        #
        # Convolution:
        #
        # - query: Linear layer
        # - key: uvu tensor product and then linear layer
        # - value: uvu tensor product and then linear layer

        #
        # query
        #
        self.h_q = Linear(node_feats_irreps_in, irreps_query_and_key)

        #
        # key and value
        #
        # key and value use the same type of tensor product, we can share this
        # because we do not use internal weights
        self.tp_k = UVUTensorProduct(
            node_feats_irreps_in,
            edge_attrs_irreps,
            node_feats_irreps_out,
            mlp_input_size=self.irreps_in[DataKey.EDGE_EMBEDDING].dim,
            mlp_hidden_size=fc_hidden_size,
            mlp_num_hidden_layers=fc_num_hidden_layers,
        )

        self.tp_v = UVUTensorProduct(
            node_feats_irreps_in,
            edge_attrs_irreps,
            node_feats_irreps_out,
            mlp_input_size=self.irreps_in[DataKey.EDGE_EMBEDDING].dim,
            mlp_hidden_size=fc_hidden_size,
            mlp_num_hidden_layers=fc_num_hidden_layers,
        )

        # In the tensor product using `uvu` instruction above, its output (i.e.
        # irreps_mid) can be uncoallesed (for example, irreps_mid =1x0e+2x0e+2x1e).
        # Here, we simplify it to combine irreps of the same the together (for example,
        # irreps_mid.simplify()='3x0e+2x1e').
        # The normalization in `Linear` is different for unsimplified and simplified
        # irreps.
        self.linear_k = Linear(
            self.tp_k.irreps_out.simplify(),
            irreps_query_and_key,
            internal_weights=True,
            shared_weights=True,
        )
        self.linear_v = Linear(
            self.tp_v.irreps_out.simplify(),
            node_feats_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        # TODO, replace this with TensorProduct `uuu` (do not use any parameter)
        self.dot = FullyConnectedTensorProduct(
            irreps_query_and_key, irreps_query_and_key, "0e"
        )

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

        q = self.h_q(node_feats_in)
        q = q[edge_dst]

        # TODO think carefully whether linear_k is needed, same for linear_v below
        #  well, it can be used to increase model capacity
        k = self.tp_k(node_feats_in[edge_src], edge_attrs, edge_embedding)
        k = self.linear_k(k)

        v = self.tp_v(node_feats_in[edge_src], edge_attrs, edge_embedding)
        v = self.linear_v(v)

        # use edge weight cutoff to make it smooth
        pos_src = data[DataKey.POSITIONS][edge_src]
        pos_dst = data[DataKey.POSITIONS][edge_src]
        edge_length = torch.norm(pos_dst - pos_src, dim=1)
        edge_weight_cutoff = soft_unit_step(
            2 * self.r_max * (1 - edge_length / self.r_max)
        )

        exp = edge_weight_cutoff[:, None] * self.dot(q, k).exp()
        z = scatter(exp, edge_dst, dim=0, dim_size=len(node_feats_in))
        z[z == 0] = 1  # deal with atoms all of their neighbors are at r_max
        alpha = exp / z[edge_dst]

        # sqrt is mysterious
        node_feats = scatter(
            alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(node_feats_in)
        )

        if self.self_connection is not None:
            node_feats = node_feats + self.self_connection(node_feats_in, node_attrs)

        data[DataKey.NODE_FEATURES] = node_feats

        return data
