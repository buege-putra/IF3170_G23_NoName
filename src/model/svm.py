from collections.abc import Callable
from typing import Literal, Self

import numpy as np

from .base import Arrays, BaseModel, FloatArray, IntArray, Params

Kernel = Literal["linear", "poly", "rbf"]


# Dari slides Support Vector Machine modul 20
# Halaman 44 dinyatakan bahwa
# Linear Kernel: K(x_i, x_j) = x_i^Tx_j
# Polinomial K(x_i, x_j) = (gamma dot x_i^Tx_j + r)^p, gamma > 0
# RBF K(x_i, x_j) = exp(-gamma dot ||X_i - X_j||^2)
def linear_kernel(A: FloatArray, B: FloatArray) -> FloatArray:
    return A @ B.T


def polynomial_kernel(
    A: FloatArray,
    B: FloatArray,
    degree: int = 3,
    gamma: float = 1.0,
    coef0: float = 1.0,
) -> FloatArray:
    return (gamma * (A @ B.T) + coef0) ** degree


def gaussian_kernel(A: FloatArray, B: FloatArray, gamma: float = 1.0) -> FloatArray:
    squared_distance = (A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2 * (A @ B.T)
    return np.exp(-gamma * np.maximum(squared_distance, 0.0))


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
        max_samples: int | None = 3000,
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
        self.max_samples = max_samples
        self.seed = seed

    def _kernel(self, A: FloatArray, B: FloatArray) -> FloatArray:
        if self.kernel == "linear":
            return linear_kernel(A, B)
        if self.kernel == "poly":
            return polynomial_kernel(A, B, self.degree, self.gamma, self.coef0)
        return gaussian_kernel(A, B, self.gamma)

    def _subsample(self, X: FloatArray, y: IntArray, rng: np.random.Generator) -> tuple[FloatArray, IntArray]:
        if self.max_samples is None or len(X) <= self.max_samples:
            return X, y

        keep = []
        for label in np.unique(y):
            rows = np.flatnonzero(y == label)
            count = round(self.max_samples * len(rows) / len(y))
            keep.append(rng.choice(rows, size=count, replace=False))

        keep = np.sort(np.concatenate(keep))
        return X[keep], y[keep]

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        rng = np.random.default_rng(self.seed)
        X, y = self._subsample(np.asarray(X, dtype=float), np.asarray(y), rng)
        d = np.where(y == 1, 1.0, -1.0)
        n = len(X)

        K = self._kernel(X, X)
        alpha = np.zeros(n)
        b = 0.0
        passes = 0
        iterations = 0

        while passes < self.max_passes and iterations < self.max_iter:
            changed = 0

            for i in range(n):
                E_i = (alpha * d) @ K[:, i] + b - d[i]

                if (d[i] * E_i < -self.tol and alpha[i] < self.C) or (d[i] * E_i > self.tol and alpha[i] > 0):
                    j = rng.integers(n - 1)
                    j = j + 1 if j >= i else j
                    E_j = (alpha * d) @ K[:, j] + b - d[j]

                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                    if d[i] != d[j]:
                        L, H = max(0.0, alpha[j] - alpha[i]), min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L, H = max(0.0, alpha[i] + alpha[j] - self.C), min(self.C, alpha[i] + alpha[j])
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alpha[j] = np.clip(alpha[j] - d[j] * (E_i - E_j) / eta, L, H)
                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    alpha[i] += d[i] * d[j] * (alpha_j_old - alpha[j])

                    b1 = b - E_i - d[i] * (alpha[i] - alpha_i_old) * K[i, i] - d[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - d[i] * (alpha[i] - alpha_i_old) * K[i, j] - d[j] * (alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    changed += 1

            passes = passes + 1 if changed == 0 else 0
            iterations += 1

        support = alpha > 1e-8
        self.support_vectors_ = X[support]
        self.alpha_ = alpha[support]
        self.d_ = d[support]

        # b = d_sv - sum(alpha_i d_i K(x_i, x_sv)), averaged over support vectors with 0 < alpha < C
        free = support & (alpha < self.C - 1e-8)
        if free.any():
            free_K = K[np.ix_(support, free)]
            self.b_ = float(np.mean(d[free] - (self.alpha_ * self.d_) @ free_K))
        else:
            self.b_ = float(b)

        self.n_iter_ = iterations
        return self

    def decision_function(self, X: FloatArray) -> FloatArray:
        K = self._kernel(np.asarray(X, dtype=float), self.support_vectors_)
        return K @ (self.alpha_ * self.d_) + self.b_

    def predict(self, X: FloatArray) -> IntArray:
        return (self.decision_function(X) >= 0).astype(int)

    def get_state(self) -> tuple[Params, Arrays]:
        params = {
            "C": self.C,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self.gamma,
            "coef0": self.coef0,
            "tol": self.tol,
            "max_passes": self.max_passes,
            "max_iter": self.max_iter,
            "max_samples": self.max_samples,
            "seed": self.seed,
            "b": self.b_,
        }
        arrays = {"support_vectors": self.support_vectors_, "alpha": self.alpha_, "d": self.d_}
        return params, arrays

    def set_state(self, params: Params, arrays: Arrays) -> None:
        self.__init__(**{key: value for key, value in params.items() if key != "b"})
        self.b_ = params["b"]
        self.support_vectors_ = arrays["support_vectors"]
        self.alpha_ = arrays["alpha"]
        self.d_ = arrays["d"]
