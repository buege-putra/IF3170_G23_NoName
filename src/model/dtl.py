from typing import Self

from .base import Arrays, BaseModel, FloatArray, IntArray, Params


class C45DecisionTree(BaseModel):
    """C4.5: gain ratio, numeric threshold, categorical split, missing value, dan pruning.
    NaN pada X untuk menandai missing value. Index kolom categorical di-pass lewat `categorical`.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_gain: float = 0.0,
        categorical: list[int] | None = None,
        ccp_alpha: float = 0.0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.categorical = categorical or []
        self.ccp_alpha = ccp_alpha

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        raise NotImplementedError

    def predict(self, X: FloatArray) -> IntArray:
        raise NotImplementedError

    def predict_proba(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def prune(self, X_val: FloatArray, y_val: IntArray) -> Self:
        raise NotImplementedError

    def get_state(self) -> tuple[Params, Arrays]:
        raise NotImplementedError

    def set_state(self, params: Params, arrays: Arrays) -> None:
        raise NotImplementedError
