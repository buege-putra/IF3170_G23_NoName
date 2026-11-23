from typing import Literal, Self

import numpy as np

from .base import Arrays, BaseModel, FloatArray, IntArray, Params

Metric = Literal["euclidean", "manhattan", "minkowski"]
Weights = Literal["uniform", "distance"]

# batas jumlah elemen matriks jarak sementara (query x train x fitur) per batch
MAX_CHUNK_ELEMENTS = 20_000_000
# mencegah pembagian nol pada bobot 1/distance saat jarak 0
DISTANCE_EPS = 1e-9


class KNN(BaseModel):
    """K-Nearest Neighbors.

    Parameter:
        k: jumlah tetangga terdekat, dibatasi jumlah data training.
        metric: "euclidean", "manhattan", atau "minkowski".
        weights: "uniform" (semua tetangga sama) atau "distance" (bobot 1/jarak).
        p: orde jarak jika pakai Minkowski.

    Usage:
        model = KNN(k=5, metric="minkowski", weights="distance", p=3.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        model.save("model.npz")
        model = KNN.load("model.npz")
    """

    def __init__(
        self,
        k: int = 5,
        metric: Metric = "euclidean",
        weights: Weights = "uniform",
        p: float = 3.0,
    ) -> None:
        self.k = k
        self.metric = metric
        self.weights = weights
        self.p = p

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        self.X_ = np.asarray(X, dtype=np.float64)
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=np.float64)
        proba = np.zeros((len(X), len(self.classes_)))
        for start in range(0, len(X), self._chunk_size()):
            rows = slice(start, start + self._chunk_size())
            proba[rows] = self._vote(X[rows])
        return proba

    def predict(self, X: FloatArray) -> IntArray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_state(self) -> tuple[Params, Arrays]:
        params = {
            "k": self.k,
            "metric": self.metric,
            "weights": self.weights,
            "p": self.p,
        }
        return params, {"X": self.X_, "y": self.y_, "classes": self.classes_}

    def set_state(self, params: Params, arrays: Arrays) -> None:
        self.k = params["k"]
        self.metric = params["metric"]
        self.weights = params["weights"]
        self.p = params["p"]
        self.X_ = arrays["X"]
        self.y_ = arrays["y"]
        self.classes_ = arrays["classes"]

    def _chunk_size(self) -> int:
        """Jumlah query per batch agar matriks jarak sementara tetap kecil."""
        return max(1, MAX_CHUNK_ELEMENTS // (len(self.X_) * max(1, self.X_.shape[1])))

    def _vote(self, queries: FloatArray) -> FloatArray:
        distances = self._distances(queries)
        k = min(self.k, distances.shape[1])
        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        nearest_dist = np.take_along_axis(distances, nearest, axis=1)
        weights = self._neighbor_weights(nearest_dist)
        votes = np.zeros((len(queries), len(self.classes_)))
        labels = self.y_[nearest]
        for c in range(len(self.classes_)):
            votes[:, c] = np.sum(weights * (labels == c), axis=1)
        return votes / votes.sum(axis=1, keepdims=True)

    def _neighbor_weights(self, distances: FloatArray) -> FloatArray:
        if self.weights == "uniform":
            return np.ones_like(distances)
        return 1.0 / (distances + DISTANCE_EPS)

    def _distances(self, queries: FloatArray) -> FloatArray:
        if self.metric == "euclidean":
            return self._euclidean(queries)
        order = 1.0 if self.metric == "manhattan" else self.p
        diff = np.abs(queries[:, None, :] - self.X_[None, :, :])
        return np.sum(diff**order, axis=2) ** (1.0 / order)

    def _euclidean(self, queries: FloatArray) -> FloatArray:
        squared = (
            np.sum(queries**2, axis=1)[:, None]
            - 2.0 * queries @ self.X_.T
            + np.sum(self.X_**2, axis=1)[None, :]
        )
        return np.sqrt(np.maximum(squared, 0.0))
