"""
This module implements node and global featurizers.
"""
from __future__ import annotations

import abc
import copy
from typing import Callable

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty, OxidationStates
from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.structure import DensityFeatures
from matminer.utils.data import MagpieData
from pymatgen.core import Structure


class BaseFeaturizer:
    """Base featurizer for a dataframe."""

    def __init__(self, featurizers: list[Callable] = None):
        self.featurizers = featurizers

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Featurize a dataframe.

        Args:
            df: A dataframe to be featurized. Each row of the dataframe corresponds to
            a structure. The columns can contain `structure`, `composition`, ect.

        Returns:
            A featurized dataframe.
        """

    @property
    def feature_names(self) -> list[str]:
        """
        Return the names of the features.

        Returns:
            A list of feature names.
        """
        return [f.__name__ for f in self.featurizers]


class GlobalFeaturizer(BaseFeaturizer):
    # a dict of {required column: featurizer to use this column}
    DEFAULT_FEATURIZER_CLASS = {
        "composition": ElementProperty,
        "composition_oxid": OxidationStates,
        "structure": DensityFeatures,
    }

    # features from the featurizers to use
    DEFAULT_FEATURE_NAMES = [
        "MagpieData minimum Number",
        "MagpieData minimum MendeleevNumber",
        "density",
        "vpa",
    ]

    def __init__(
        self, featurizers: list[Callable] = None, feature_names: list[str] = None
    ):
        super().__init__(featurizers)
        if self.featurizers is None:
            self.featurizers = []
            self.featurizer_column = []
            for name, C in self.DEFAULT_FEATURIZER_CLASS.items():
                if C == ElementProperty:
                    f = ElementProperty.from_preset(preset_name="magpie")
                else:
                    f = C()
                self.featurizers.append(f)
                self.featurizer_column.append(name)

        if feature_names is None:
            self._feature_names = self.DEFAULT_FEATURE_NAMES
        else:
            self._feature_names = feature_names

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = copy.copy(df_in)

        # add composition info to dataframe
        if "composition" not in df.columns:
            df["composition"] = df["structure"].apply(lambda s: s.composition)
        if "composition_oxid" not in df.columns:
            df = CompositionToOxidComposition().featurize_dataframe(
                df, "composition", ignore_errors=True
            )

        for name, c in zip(self.featurizer_column, self.featurizers):
            df = c.featurize_dataframe(df, name, ignore_errors=True)

        # only return requested features
        df_out = copy.copy(df_in)
        for col_name in self.feature_names:
            df_out[col_name] = df[col_name]

        return df_out


class MagpieAtomFeaturizer(BaseFeaturizer):
    DEFAULT_FEATURE_NAMES = [
        "AtomicVolume",
        "AtomicWeight",
        "Column",
        "CovalentRadius",
    ]

    def __init__(self, feature_names: list[str] = None):
        super().__init__()

        if feature_names is None:
            self._feature_names = self.DEFAULT_FEATURE_NAMES
        else:
            self._feature_names = feature_names

        self.featurizer = MagpieData()

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        def featurize_a_structure(struct: Structure) -> np.ndarray:
            """
            Returns a 2D array of shape (M, N), where M is the number of atoms in the
            structure and N is the number of features.
            """
            features = []
            for atom in struct:
                feat = [
                    self.featurizer.all_elemental_props[prop_name][atom.specie.symbol]
                    for prop_name in self.feature_names
                ]
                features.append(feat)

            features = np.asarray(features)

            return features

        df_in["atom_feats"] = df_in["structure"].apply(featurize_a_structure)

        return df_in
