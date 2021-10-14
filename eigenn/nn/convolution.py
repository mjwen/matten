from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as fn
from e3nn.math import soft_unit_step
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, Irrep, Irreps, Linear, TensorProduct
from e3nn.util.jit import compile_mode
from nequip.nn.nonlinearities import ShiftedSoftPlus
from torch_scatter import scatter

from eigenn.nn.irreps import DataKey, ModuleIrreps, _check_irreps_type


@compile_mode("script")
class TFNConv(nn.Module, ModuleIrreps):

    REQUIRED_KEYS_IRREPS_IN = [
        DataKey.NODE_FEATURES,
        DataKey.NODE_ATTRS,
        DataKey.EDGE_EMBEDDING,
        DataKey.EDGE_ATTRS,
    ]

    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        irreps_out: Dict[str, Irreps],
        fc_num_hidden_layers: int = 1,
        fc_hidden_size: int = 8,
        activation_scalars: Dict[str, Callable] = None,
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
            activation_scalars: activation function for scalars (i.e. irreps l=0).
                Should be something like {'e': act1 'o':act2}, where act1 is for even
                parity irreps (i.e. 0e) and act2 is for odd parity irreps (i.e. 0o);
                both should be callable.

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
            node_feats_irreps_in,
            node_feats_irreps_in,
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

        # second linear on node feats
        #
        # In the tensor product using `uvu` instruction above, its output (i.e.
        # irreps_mid) can be uncoallesed (for example, irreps_mid =1x0e+2x0e+2x1e).
        # Here, we simplify it to combine irreps of the same the together (for example,
        # irreps_mid.simplify()='3x0e+2x1e').
        # The normalization in Linear is different for unsimplified and simplified irreps.
        self.linear_2 = Linear(
            irreps_mid.simplify(),
            node_feats_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        # radial network on edge embedding (e.g. edge distance)

        # TODO silu may also be used
        if activation_scalars is None:
            activation_scalars = {"e": ShiftedSoftPlus, "o": fn.tanh}
        self.radial_nn = FullyConnectedNet(
            [self.irreps_in[DataKey.EDGE_EMBEDDING].num_irreps]
            + fc_num_hidden_layers * [fc_hidden_size]
            + [self.tp.weight_numel],
            act=activation_scalars["e"],
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

        node_feats_in = data[DataKey.NODE_FEATURES]
        node_attrs = data[DataKey.NODE_ATTRS]
        edge_embedding = data[DataKey.EDGE_EMBEDDING]
        edge_attrs = data[DataKey.EDGE_ATTRS]

        edge_src = data[DataKey.EDGE_INDEX][0]
        edge_dst = data[DataKey.EDGE_INDEX][1]

        node_feats = self.linear_1(node_feats_in)

        weight = self.radial_nn(edge_embedding)
        edge_feats = self.tp(node_feats[edge_src], edge_attrs, weight)
        node_feats = scatter(edge_feats, edge_dst, dim=0, dim_size=node_feats.shape[0])

        if self.avg_num_neighbors is not None:
            node_feats = node_feats / self.avg_num_neighbors ** 0.5

        node_feats = self.lin2(node_feats)

        # alpha = self.alpha(node_features, node_attr)
        #
        # m = self.sc.output_mask
        # alpha = (1 - m) + alpha * m
        #
        # return node_self_connection + alpha * node_conv_out

        if self.self_connection is not None:
            sc = +self.self_connection(node_feats_in, node_attrs)
            node_feats = sc + node_feats

        data[DataKey.NODE_FEATURES] = node_feats

        return data

    @staticmethod
    def fix_irreps_in(irreps_in: Dict[str, Irreps]) -> Dict[str, Irreps]:

        irreps_in = super().fix_irreps_in(irreps_in)

        # ensure EDGE_EMBEDDING (e.g. embedding of radial distance) to be invariant
        # scalars (i.e. 0e) in order to use a dense network
        ok = _check_irreps_type(irreps_in[DataKey.EDGE_EMBEDDING], [Irrep("0e")])
        if not ok:
            raise ValueError(
                f"Expect edge embedding irreps only contain `0e`; "
                f"got {irreps_in[DataKey.EDGE_EMBEDDING]}"
            )

        return irreps_in


@compile_mode("script")
class SE3Transformer(nn.Module, ModuleIrreps):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        irreps_query_key: Irreps,
        irreps_out: Dict[str, Irreps] = None,
        *,
        fc_num_hidden_layers: int = 1,
        fc_hidden_size: int = 8,
        r_max: float = 5.0,
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
            irreps_query_key: irreps for the query and key
            fc_num_hidden_layers: number of hidden layers for the radial MLP, excluding
                input and output layers
            fc_hidden_size: hidden layer size for the radial MLP
            r_max: cutoff distance
            use_self_connection: whether to use self interaction, e.g. Eq.10 in the
                SE3-Transformer paper.
            activation_scalars: activation function for scalars (i.e. irreps l=0).
                Should be something like {'e': act1 'o':act2}, where act1 is for even
                parity irreps (i.e. 0e) and act2 is for odd parity irreps (i.e. 0o);
                both should be callable.
        """

        super().__init__()

        self.avg_num_neighbors = avg_num_neighbors
        self.r_max = r_max

        #
        # set up irreps
        #
        required_irreps_in = [
            DataKey.NODE_FEATURES,
            DataKey.NODE_ATTRS,
            DataKey.EDGE_EMBEDDING,
            DataKey.EDGE_ATTRS,
        ]

        # ensure EDGE_EMBEDDING (e.g. embedding of radial distance) to be invariant
        # scalars (i.e. 0e) in order to use a dense network
        ok = _check_irreps_type(irreps_in[DataKey.EDGE_EMBEDDING], [Irrep("0e")])
        if not ok:
            raise ValueError(
                f"Expect edge embedding irreps only contain `0e`; "
                f"got {irreps_in[DataKey.EDGE_EMBEDDING]}"
            )

        if irreps_out is None:
            irreps_out = {DataKey.NODE_FEATURES: irreps_in[DataKey.NODE_FEATURES]}
        self.init_irreps(
            irreps_in, irreps_out, required_keys_irreps_in=required_irreps_in
        )

        node_feats_irreps_in = self.irreps_in[DataKey.NODE_FEATURES]
        node_feats_irreps_out = self.irreps_out[DataKey.NODE_FEATURES]
        node_attrs_irreps_in = self.irreps_in[DataKey.NODE_ATTRS]
        edge_attrs_irreps = self.irreps_in[DataKey.EDGE_ATTRS]

        #
        # Convolution:
        #
        # - query: Linear layer
        # - key: uvu tensor product and then linear layer
        # - value: uvu tensor product and then linear layer

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

        #
        # query
        self.h_q = Linear(node_feats_irreps_in, irreps_query_key)

        #
        # key
        #
        self.tp_k = TensorProduct(
            node_feats_irreps_in,
            edge_attrs_irreps,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )

        # radial network on edge embedding (e.g. edge distance)
        # TODO silu may also be used
        if activation_scalars is None:
            activation_scalars = {"e": ShiftedSoftPlus, "o": fn.tanh}

        self.radial_nn_k = FullyConnectedNet(
            [self.irreps_in[DataKey.EDGE_EMBEDDING].num_irreps]
            + fc_num_hidden_layers * [fc_hidden_size]
            + [self.tp_k.weight_numel],
            act=activation_scalars["e"],
        )

        # In the tensor product using `uvu` instruction above, its output (i.e.
        # irreps_mid) can be uncoallesed (for example, irreps_mid =1x0e+2x0e+2x1e).
        # Here, we simplify it to combine irreps of the same the together (for example,
        # irreps_mid.simplify()='3x0e+2x1e').
        # The normalization in Linear is different for unsimplified and simplified irreps.
        self.linear_k = Linear(
            irreps_mid.simplify(),
            irreps_query_key,
            internal_weights=True,
            shared_weights=True,
        )

        #
        # value
        #
        self.tp_v = TensorProduct(
            node_feats_irreps_in,
            edge_attrs_irreps,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )

        self.radial_nn_v = FullyConnectedNet(
            [self.irreps_in[DataKey.EDGE_EMBEDDING].num_irreps]
            + fc_num_hidden_layers * [fc_hidden_size]
            + [self.tp_v.weight_numel],
            act=activation_scalars["e"],
        )

        self.linear_v = Linear(
            irreps_mid.simplify(),
            node_feats_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        # TODO, replace this with TensorProduct `uuu` (do not use any parameter)
        #  and  let irreps_query and irreps_key be the same
        #  Well, not really, in fact should check h_q.irreps_out and tp_k.irreps_out be
        #  the same
        self.dot = FullyConnectedTensorProduct(irreps_query_key, irreps_query_key, "0e")

        if use_self_connection:
            self.self_connection = FullyConnectedTensorProduct(
                node_feats_irreps_in, node_attrs_irreps_in, node_feats_irreps_out
            )
        else:
            self.self_connection = None

    # def forward(self, data: DataKey.Type):
    #     edge_src, edge_dst = radius_graph(pos, max_radius)
    #     edge_vec = pos[edge_src] - pos[edge_dst]
    #     edge_length = edge_vec.norm(dim=1)
    #
    #     edge_length_embedded = soft_one_hot_linspace(
    #         edge_length,
    #         start=0.0,
    #         end=max_radius,
    #         number=number_of_basis,
    #         basis="smooth_finite",
    #         cutoff=True,
    #     )
    #     edge_length_embedded = edge_length_embedded.mul(number_of_basis ** 0.5)
    #     edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / max_radius))
    #
    #     edge_sh = o3.spherical_harmonics(
    #         irreps_sh, edge_vec, True, normalization="component"
    #     )
    #
    #     q = h_q(f)
    #     k = tp_k(f[edge_src], edge_sh, fc_k(edge_length_embedded))
    #     v = tp_v(f[edge_src], edge_sh, fc_v(edge_length_embedded))
    #
    #     exp = edge_weight_cutoff[:, None] * dot(q[edge_dst], k).exp()
    #     z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
    #     z[z == 0] = 1
    #     alpha = exp / z[edge_dst]
    #
    #     return scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f))
    #

    def forward(self, data: DataKey.Type):

        node_feats_in = data[DataKey.NODE_FEATURES]
        node_attrs = data[DataKey.NODE_ATTRS]
        edge_embedding = data[DataKey.EDGE_EMBEDDING]
        edge_attrs = data[DataKey.EDGE_ATTRS]

        edge_src = data[DataKey.EDGE_INDEX][0]
        edge_dst = data[DataKey.EDGE_INDEX][1]

        q = self.h_q(node_feats_in)
        q = q[edge_dst]

        weight_k = self.radial_nn_k(edge_embedding)
        k = self.tp_k(node_feats_in[edge_src], edge_attrs, weight_k)
        # TODO think carefully whether linear_k is needed, same for linear_v below
        k = self.linear_k(k)

        weight_v = self.radial_nn_v(edge_embedding)
        v = self.tp_v(node_feats_in[edge_src], edge_attrs, weight_v)
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
        z[z == 0] = 1
        alpha = exp / z[edge_dst]

        # sqrt is mysterious
        node_feats = scatter(
            alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(node_feats_in)
        )

        if self.self_connection is not None:
            sc = +self.self_connection(node_feats_in, node_attrs)
            node_feats = sc + node_feats

        data[DataKey.NODE_FEATURES] = node_feats

        return data
