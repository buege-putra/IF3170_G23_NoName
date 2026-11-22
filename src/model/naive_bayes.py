from typing import Self

from .base import Arrays, BaseModel, FloatArray, IntArray, Params


class NaiveBayes(BaseModel):
    """Gaussian untuk kolom numeric, sisanya categorical (Laplace)."""

    def __init__(
        self,
        categorical: list[int] | None = None,
        var_smoothing: float = 1e-9,
        laplace: float = 1.0,
    ) -> None:
        self.categorical = categorical or []
        self.var_smoothing = var_smoothing
        self.laplace = laplace

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        raise NotImplementedError

    def predict(self, X: FloatArray) -> IntArray:
        raise NotImplementedError

    def predict_proba(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def get_state(self) -> tuple[Params, Arrays]:
        raise NotImplementedError

    def set_state(self, params: Params, arrays: Arrays) -> None:
        raise NotImplementedError
