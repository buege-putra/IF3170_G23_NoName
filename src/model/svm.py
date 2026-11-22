from collections.abc import Callable
from typing import Literal, Self

from .base import Arrays, BaseModel, FloatArray, IntArray, Params

Kernel = Literal["linear", "poly", "rbf"]


def linear_kernel(A: FloatArray, B: FloatArray) -> FloatArray:
    raise NotImplementedError


def polynomial_kernel(
    A: FloatArray,
    B: FloatArray,
    degree: int = 3,
    gamma: float = 1.0,
    coef0: float = 1.0,
) -> FloatArray:
    raise NotImplementedError


def gaussian_kernel(A: FloatArray, B: FloatArray, gamma: float = 1.0) -> FloatArray:
    raise NotImplementedError


KERNELS: dict[str, Callable[..., FloatArray]] = {
    "linear": linear_kernel,
    "poly": polynomial_kernel,
    "rbf": gaussian_kernel,
}


class SVM(BaseModel):
    """Soft-margin binary SVM, di-train dengan SMO pada bentuk dual."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: Kernel = "rbf",
        degree: int = 3,
        gamma: float = 1.0,
        coef0: float = 1.0,
        tol: float = 1e-3,
        max_passes: int = 5,
        max_iter: int = 1000,
        seed: int = 67,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        raise NotImplementedError

    def decision_function(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def predict(self, X: FloatArray) -> IntArray:
        raise NotImplementedError

    def get_state(self) -> tuple[Params, Arrays]:
        raise NotImplementedError

    def set_state(self, params: Params, arrays: Arrays) -> None:
        raise NotImplementedError
