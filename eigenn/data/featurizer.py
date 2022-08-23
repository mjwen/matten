"""
This module implements node and global featurizers.
"""
from __future__ import annotations

import abc
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
    def featurizer_names(self) -> list[str]:
        """
        Return the names of the featurizers.

        Returns:
            A list of featurizer names.
        """
        return [f.__name__ for f in self.featurizers]


class GlobalFeaturizer(BaseFeaturizer):
    # a dict of {required column: featurizer to use this column}
    DEFAULT_FEATURIZER_CLASS = {
        "composition": ElementProperty,
        "composition_oxid": OxidationStates,
        "structure": DensityFeatures,
    }

    def __init__(self, featurizers: list[Callable] = None):
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

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # add composition info to dataframe
        if "composition" not in df.columns:
            df["composition"] = df["structure"].apply(lambda s: s.composition)
        if "composition_oxid" not in df.columns:
            df = CompositionToOxidComposition().featurize_dataframe(df, "composition")

        for name, c in zip(self.featurizer_column, self.featurizers):
            df = c.featurize_dataframe(df, name)

        return df
