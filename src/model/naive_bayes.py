from typing import Self

import numpy as np

from .base import Arrays, BaseModel, FloatArray, IntArray, Params


class NaiveBayes(BaseModel):
    """Gaussian untuk kolom numeric, Laplace untuk kolom categorical.

    var_smoothing relatif terhadap varians maksimum, jadi Gaussian cocok untuk
    fitur kontinu yang sudah ditransformasi. threshold None berarti argmax.
    """

    def __init__(
        self,
        categorical: list[int] | None = None,
        var_smoothing: float = 1e-9,
        laplace: float = 1.0,
        threshold: float | None = None,
    ) -> None:
        self.categorical = categorical or []
        self.var_smoothing = var_smoothing
        self.laplace = laplace
        self.threshold = threshold

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, yi = np.unique(y, return_inverse=True)
        n_class = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.num_idx = [j for j in range(X.shape[1]) if j not in self.categorical]

        self.prior = np.bincount(yi, minlength=n_class) / len(yi)
        Xn = X[:, self.num_idx]
        self.mu = np.array([Xn[yi == k].mean(0) for k in range(n_class)])
        self.var = np.array([Xn[yi == k].var(0) for k in range(n_class)])
        if Xn.shape[1]:
            scale = Xn.var(0).max()
            self.var += self.var_smoothing * (scale if scale else 1e-9)

        self.cat_logp = []
        self.cat_sizes = []
        for j in self.categorical:
            col = X[:, j].astype(int)
            K = int(col.max()) + 1
            logp = np.zeros((n_class, K + 1))
            for k in range(n_class):
                known = (yi == k) & (col >= 0) & (col < K)
                count = np.bincount(col[known], minlength=K)
                denom = count.sum() + self.laplace * K
                logp[k, :K] = np.log((count + self.laplace) / denom)
                logp[k, K] = np.log(self.laplace / denom)
            self.cat_logp.append(logp)
            self.cat_sizes.append(K)
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        joint = self._log_joint(X)
        p = np.exp(joint - joint.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: FloatArray) -> IntArray:
        if self.threshold is None:
            return self.classes_[self._log_joint(X).argmax(axis=1)]
        p = self.predict_proba(X)[:, 1]
        return self.classes_[np.where(p >= self.threshold, 1, 0)]

    def get_state(self) -> tuple[Params, Arrays]:
        params: Params = {
            "categorical": self.categorical,
            "var_smoothing": self.var_smoothing,
            "laplace": self.laplace,
            "n_features": self.n_features_,
            "cat_sizes": self.cat_sizes,
            "threshold": self.threshold,
        }
        arrays: Arrays = {
            "classes": self.classes_,
            "prior": self.prior,
            "mu": self.mu,
            "var": self.var,
        }
        for i, logp in enumerate(self.cat_logp):
            arrays[f"cat_{i}"] = logp
        return params, arrays

    def set_state(self, params: Params, arrays: Arrays) -> None:
        self.categorical = params["categorical"]
        self.var_smoothing = params["var_smoothing"]
        self.laplace = params["laplace"]
        self.n_features_ = params["n_features"]
        self.cat_sizes = params["cat_sizes"]
        self.threshold = params.get("threshold")
        self.classes_ = arrays["classes"]
        self.prior = arrays["prior"]
        self.mu = arrays["mu"]
        self.var = arrays["var"]
        self.num_idx = [j for j in range(self.n_features_) if j not in self.categorical]
        self.cat_logp = [arrays[f"cat_{i}"] for i in range(len(self.categorical))]

    def _log_joint(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        out = np.tile(np.log(self.prior), (len(X), 1))
        for k in range(len(self.classes_)):
            d = X[:, self.num_idx] - self.mu[k]
            out[:, k] += (
                -0.5 * (np.log(2 * np.pi * self.var[k]) + d**2 / self.var[k])
            ).sum(axis=1)

        for j, logp, K in zip(self.categorical, self.cat_logp, self.cat_sizes):
            code = X[:, j].astype(int)
            code = np.where((code < 0) | (code >= K), K, code)
            out += logp[:, code].T
        return out
