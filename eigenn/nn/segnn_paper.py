from typing import Callable, Dict, Optional

import torch
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear
from torch import Tensor
from torch_scatter import scatter

from eigenn.data.irreps import DataKey, ModuleIrreps
from eigenn.nn.utils import ACTIVATION, ActivationLayer, NormalizationLayer, ScalarMLP


class SEGNNMessagePassing(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        conv_layer_irreps: Irreps,
        activation_type: str = "gate",
        activation_scalars: Dict[str, str] = None,
        activation_gates: Dict[str, str] = None,
        normalization: str = None,
        use_resnet: bool = True,
    ):
        """
        SEGNN message passing.

        Args:
            irreps_in:
            conv_layer_irreps:
            activation_type:
            activation_scalars:
            activation_gates:
            normalization: batch, instance, or None
            use_resnet:
        """
        super().__init__()

        # irreps_out will be set later when we know them

        self.init_irreps(irreps_in)

        node_feats_irreps_in = self.irreps_in[DataKey.NODE_FEATURES]
        node_attrs_irreps = self.irreps_in[DataKey.NODE_ATTRS]
        edge_attrs_irreps = self.irreps_in[DataKey.EDGE_ATTRS]
        edge_embedding_irreps = self.irreps_in[DataKey.EDGE_EMBEDDING]
        conv_layer_irreps = Irreps(conv_layer_irreps)

        ##
        # message
        ##

        # Initial message contains the node feats of node i and j and the embedding
        # of the edge length.
        # Note, simplify() will not sort, but only combines the same irreps adjacent
        # to each other.
        message_irreps_in = (
            node_feats_irreps_in + node_feats_irreps_in + edge_embedding_irreps
        ).simplify()

        self.message_layer1 = TensorProductWithActivation(
            message_irreps_in,
            edge_attrs_irreps,
            conv_layer_irreps,
            activation_type=activation_type,
            activation_scalars=activation_scalars,
            activation_gates=activation_gates,
        )
        self.message_layer2 = TensorProductWithActivation(
            self.message_layer1.irreps_out,
            edge_attrs_irreps,
            conv_layer_irreps,
            activation_type=activation_type,
            activation_scalars=activation_scalars,
            activation_gates=activation_gates,
        )

        self.message_norm = NormalizationLayer(
            self.message_layer2.irreps_out, method=normalization
        )

        ##
        # update
        ##

        # Note, the cat order should be the same below in the forward function
        update_irreps_in = (
            node_feats_irreps_in + self.message_layer2.irreps_out
        ).simplify()

        self.update_layer1 = TensorProductWithActivation(
            update_irreps_in,
            node_attrs_irreps,
            conv_layer_irreps,
            activation_type=activation_type,
            activation_scalars=activation_scalars,
            activation_gates=activation_gates,
        )

        self.update_layer2 = FullyConnectedTensorProduct(
            self.update_layer1.irreps_out,
            node_attrs_irreps,
            conv_layer_irreps,
        )

        self.update_norm = NormalizationLayer(conv_layer_irreps, method=normalization)

        # residual connection
        if use_resnet and node_feats_irreps_in == conv_layer_irreps:
            self.use_resnet = True
        else:
            self.use_resnet = False

        # set output irreps
        self.irreps_out[DataKey.NODE_FEATURES] = conv_layer_irreps

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        node_feats = data[DataKey.NODE_FEATURES]
        node_attrs = data[DataKey.NODE_ATTRS]
        batch = data[DataKey.BATCH]

        edge_embedding = data[DataKey.EDGE_EMBEDDING]
        edge_attrs = data[DataKey.EDGE_ATTRS]
        edge_src, edge_dst = data[DataKey.EDGE_INDEX]

        x_i = node_feats[edge_dst]
        x_j = node_feats[edge_src]
        x = torch.cat((x_i, x_j, edge_embedding), dim=-1)

        x = self.message_layer1(x, edge_attrs)
        x = self.message_layer2(x, edge_attrs)
        x = self.message_norm(x, edge_dst)

        msg = scatter(x, edge_dst, dim=0, dim_size=len(node_feats), reduce="sum")
        x = torch.cat((node_feats, msg), dim=-1)

        x = self.update_layer1(x, node_attrs)
        x = self.update_layer2(x, node_attrs)

        if self.use_resnet:
            x = x + node_feats

        x = self.update_norm(x, batch)

        data[DataKey.NODE_FEATURES] = x

        return data


class TensorProductWithActivation(torch.nn.Module):
    """
    A tensor product layer followed by an activation function.

    Args:
        irreps_out: this is the intended irreps_out, but not the actual one. The
            actual one is determined by the activation function.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        activation_type,
        activation_scalars,
        activation_gates,
    ):
        super().__init__()

        self.act = ActivationLayer(
            irreps_in1,
            irreps_in2,
            irreps_out,
            activation_type=activation_type,
            activation_scalars=activation_scalars,
            activation_gates=activation_gates,
        )

        self.tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=self.act.irreps_in,
        )

        self._irreps_in = irreps_in1
        self._irreps_out = self.act.irreps_out

    def forward(self, data1: Tensor, data2: Tensor) -> Tensor:
        x = self.tp(data1, data2)
        x = self.act(x)
        return x

    @property
    def irreps_in(self):
        return self._irreps_in

    @property
    def irreps_out(self):
        return self._irreps_out


class PredictionHead(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        field: str = DataKey.NODE_FEATURES,
        out_field: Optional[str] = None,
        activation: Callable = ACTIVATION["e"]["ssp"],
        hidden_size: int = None,
        reduce: str = "mean",
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
            reduce: how to aggregate atomic value to final value, `mean` or `sum`
        """
        super().__init__()
        self.init_irreps(irreps_in, required_keys_irreps_in=[field])

        self.field = field
        self.out_field = field if out_field is None else out_field
        self.reduce = reduce

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
        batch = data[DataKey.BATCH]

        x = self.mlp1(x)

        # pooling
        x = scatter(x, batch, dim=0, reduce=self.reduce)

        x = self.mlp2(x)
        data[self.out_field] = x

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
        normalization: str = None,
    ):
        """
        Node embedding layer.

        - tensor product between node feats and node attrs.
        - activation
        - normalization (optional)

        Args:
            irreps_in:
            irreps_out: expected irreps out, actual irreps out determined within layer
            normalization: applied normalization, should be `batch`, `instance` or
                `none`.
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

        out = self.activation.irreps_out
        self.normalization = NormalizationLayer(out, method=normalization)

        # store actual irreps out
        self.irreps_out[DataKey.NODE_FEATURES] = out

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        node_feats = data[DataKey.NODE_FEATURES]
        node_attrs = data[DataKey.NODE_ATTRS]
        batch = data[DataKey.BATCH]

        node_feats = self.tp(node_feats, node_attrs)
        node_feats = self.activation(node_feats)
        node_feats = self.normalization(node_feats, batch)

        data[DataKey.NODE_FEATURES] = node_feats

        return data
