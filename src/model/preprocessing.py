from typing import Self

import numpy as np
import pandas as pd

from .base import FloatArray, IntArray


class Preprocessor:
    """
    Urutan pipeline: clean -> construct -> impute -> encode -> scale -> (select/PCA).
    """

    def __init__(self, target: str = "is_canceled", id_col: str = "id") -> None:
        self.target = target
        self.id_col = id_col

    def fit(self, df: pd.DataFrame) -> Self:
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> tuple[FloatArray, IntArray | None]:
        """Return (X, y); y bernilai None jika kolom target tidak ada."""
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> tuple[FloatArray, IntArray | None]:
        return self.fit(df).transform(df)

    @property
    def feature_names(self) -> list[str]:
        raise NotImplementedError
