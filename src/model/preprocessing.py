from typing import Self

import numpy as np
import pandas as pd

from .base import FloatArray, IntArray

MEAL_ORDER = ["Room Only", "Breakfast", "Half Board", "Full Board"]
CAP_COLUMNS = ["nightly_rate_idr", "days_before_arrival"]


class Preprocessor:
    """Bersihkan, impute, encoding ordinal/one-hot. Statistik diambil dari data fit saja."""

    def __init__(self, target: str = "is_canceled", id_col: str = "id") -> None:
        self.target = target
        self.id_col = id_col

    def fit(self, df: pd.DataFrame) -> Self:
        data = self._clean(df.drop(columns=[self.id_col], errors="ignore").drop_duplicates())
        X = data.drop(columns=[self.target], errors="ignore")
        self.num = X.select_dtypes("number").columns.tolist()
        self.cat = [c for c in X.columns if c not in self.num]
        self.med = X[self.num].median()
        self.mode = X[self.cat].mode().iloc[0] if self.cat else pd.Series(dtype=object)

        self.bounds = {}
        for c in CAP_COLUMNS:
            if c in self.num:
                q1, q3 = X[c].dropna().quantile([0.25, 0.75])
                iqr = q3 - q1
                self.bounds[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        filled = self._fill(X)
        self.mu = filled[self.num].mean()
        self.sd = filled[self.num].std().replace(0, 1)

        self.levels = {}
        for c in self.cat:
            if c == "meal_package":
                self.levels[c] = MEAL_ORDER
            else:
                self.levels[c] = sorted(X[c].dropna().astype(str).unique())
        return self

    def transform(self, df: pd.DataFrame) -> tuple[FloatArray, IntArray | None]:
        """Return (X, y) dengan kategori berupa kode integer; y None jika tidak ada target."""
        X = self._filled(df)
        out = X[self.num].copy()
        for c in self.cat:
            out[c] = pd.Categorical(X[c], categories=self.levels[c]).codes
        return out.to_numpy(dtype=float), self._target(df)

    def transform_dense(self, df: pd.DataFrame) -> tuple[FloatArray, IntArray | None]:
        """Return (X, y) dengan numerik distandardisasi dan kategorik one-hot."""
        X = self._filled(df)
        parts = [(X[self.num] - self.mu) / self.sd]
        for c in self.cat:
            codes = pd.Categorical(X[c], categories=self.levels[c])
            parts.append(pd.get_dummies(codes, prefix=c, dtype=float))
        return pd.concat(parts, axis=1).to_numpy(dtype=float), self._target(df)

    def fit_transform(self, df: pd.DataFrame) -> tuple[FloatArray, IntArray | None]:
        return self.fit(df).transform(df)

    @property
    def feature_names(self) -> list[str]:
        return self.num + self.cat

    @property
    def categorical_idx(self) -> list[int]:
        return [self.feature_names.index(c) for c in self.cat]

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        if "num_children" in X:
            X.loc[X["num_children"] >= 9, "num_children"] = np.nan
        return X

    def _fill(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.num] = X[self.num].fillna(self.med)
        X[self.cat] = X[self.cat].fillna(self.mode)
        for c, (lower, upper) in self.bounds.items():
            X[c] = X[c].clip(lower, upper)
        return X

    def _filled(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._fill(self._clean(df.drop(columns=[self.id_col], errors="ignore")))

    def _target(self, df: pd.DataFrame) -> IntArray | None:
        if self.target not in df:
            return None
        return df[self.target].to_numpy()
