import pandas as pd
import pytest
from pymatgen.core.structure import Structure

from matten.data.featurizer import GlobalFeaturizer


@pytest.fixture
def MgO():
    return Structure(
        lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
        species=["Mg", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture
def CaO():
    return Structure(
        lattice=[[0, 2.1, 2.1], [2.1, 0, 2.1], [2.1, 2.1, 0]],
        species=["Ca", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )


def test_featurizer(MgO, CaO):
    structures = [MgO, CaO]
    df = pd.DataFrame({"structure": structures})
    featurizer = GlobalFeaturizer()
    df = featurizer(df)

    # columns expected to exist
    expected_columns = ["structure"] + featurizer.feature_names

    for ec in expected_columns:
        assert ec in df.columns, ec
