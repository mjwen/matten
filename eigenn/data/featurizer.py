"""
This module implements node and global featurizers.
"""
from __future__ import annotations

import abc
import copy
from typing import Callable

import pandas as pd
from matminer.featurizers.composition import ElementProperty, OxidationStates
from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.structure import DensityFeatures


class BaseFeaturizer:
    """Base featurizer for a dataframe."""

    def __init__(self, featurizers: list[Callable] = None):
        self.featurizers = featurizers

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Featurize a dataframe.

        Args:
            df: A dataframe to be featurized.

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
            df = CompositionToOxidComposition().featurize_dataframe(df, "composition")

        for name, c in zip(self.featurizer_column, self.featurizers):
            df = c.featurize_dataframe(df, name)

        # only return requested features
        df_out = copy.copy(df_in)
        for col_name in self.feature_names:
            df_out[col_name] = df[col_name]

        return df_out
