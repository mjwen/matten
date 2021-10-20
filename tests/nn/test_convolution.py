from e3nn.o3 import Irreps

from eigenn.nn.tfn_conv import TFNConv
from eigenn.nn.transformer_conv import SE3Transformer
from eigenn.nn.irreps import DataKey


def test_TFNConv():

    conv = TFNConv(
        irreps_in={
            DataKey.NODE_FEATURES: Irreps("4x1e+2x0e"),
            DataKey.NODE_ATTRS: Irreps("4x1e+2x0e"),
            DataKey.EDGE_ATTRS: Irreps("4x1e+2x0e"),
            DataKey.EDGE_EMBEDDING: Irreps("4x0e"),
        },
        irreps_out={
            DataKey.NODE_FEATURES: Irreps("8x1e+4x0e"),
        },
    )


def test_Transformer():

    conv = SE3Transformer(
        irreps_in={
            DataKey.NODE_FEATURES: Irreps("4x1e+2x0e"),
            DataKey.NODE_ATTRS: Irreps("4x1e+2x0e"),
            DataKey.EDGE_ATTRS: Irreps("4x1e+2x0e"),
            DataKey.EDGE_EMBEDDING: Irreps("4x0e"),
        },
        irreps_out={
            DataKey.NODE_FEATURES: Irreps("8x1e+4x0e"),
        },
        irreps_query_key=Irreps("8x1e+4x0e"),
    )
