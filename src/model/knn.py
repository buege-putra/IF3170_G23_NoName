from typing import Literal, Self

from .base import Arrays, BaseModel, FloatArray, IntArray, Params


class KNN(BaseModel):
    def __init__(
        self,
        k: int = 5,
        metric: Literal["euclidean", "manhattan", "minkowski"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        self.k = k
        self.metric = metric
        self.weights = weights

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
