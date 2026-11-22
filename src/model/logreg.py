from typing import Literal, Self

from .base import Arrays, BaseModel, FloatArray, IntArray, Params

Penalty = Literal["l1", "l2", "elasticnet"] | None


class LogisticRegression(BaseModel):
    """Binary logistic regression yang dilatih dengan gradient descent.
    l1_ratio untuk elasticnet. L1 pakai proximal step (soft-threshold).
    """

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 1000,
        penalty: Penalty = "l2",
        alpha: float = 1e-2,
        l1_ratio: float = 0.5,
        tol: float = 1e-6,
        batch_size: int | None = None,
        seed: int = 67,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.batch_size = batch_size
        self.seed = seed
        self.history_: list[float] = []

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        raise NotImplementedError

    def predict(self, X: FloatArray, threshold: float = 0.5) -> IntArray:
        raise NotImplementedError

    def predict_proba(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def get_state(self) -> tuple[Params, Arrays]:
        raise NotImplementedError

    def set_state(self, params: Params, arrays: Arrays) -> None:
        raise NotImplementedError
