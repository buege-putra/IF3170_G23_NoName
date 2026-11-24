from collections.abc import Sequence
from typing import Literal, Self

import numpy as np

from .base import Arrays, BaseModel, FloatArray, IntArray, Params
from .metrics import log_loss

Penalty = Literal["l1", "l2", "elasticnet"] | None

_VALID_PENALTIES = (None, "l1", "l2", "elasticnet")

# batas argumen exp agar sigmoid tidak overflow
_LOGIT_CLIP = 500


def _sigmoid(logits: FloatArray) -> FloatArray:
    """Fungsi logistik 1 / (1 + e^-z), clipping z dengan [-_LOGIT_CLIP, _LOGIT_CLIP]."""
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)))


def _soft_threshold(weights: FloatArray, threshold: float) -> FloatArray:
    """Proximal operator L1, geser tiap bobot ke nol sejauh threshold lalu potong di nol."""
    return np.sign(weights) * np.maximum(np.abs(weights) - threshold, 0.0)


class LogisticRegression(BaseModel):
    """Logistic regression biner, p = sigmoid(w^T x + bias), dengan estimasi maximum likelihood.

    Parameter:
        lr: learning rate.
        epochs: jumlah maksimum pass atas seluruh data.
        penalty: None, "l1", "l2", atau "elasticnet".
        alpha: kekuatan regularisasi, berlaku pada log-loss rata-rata.
        l1_ratio: porsi L1 pada elasticnet, sisanya L2.
        tol: training berhenti jika perubahan objective antar epoch lebih kecil dari ini.
        batch_size: ukuran mini-batch, 1 untuk stochastic, None untuk full-batch.
        seed: seed urutan acak data tiap epoch.
        init_bias: nilai awal bias.
        init_weights: nilai awal bobot, None berarti nol.
        track_params: jika True, simpan (bias, bobot) tiap epoch di trajectory_.

    Usage:
        model = LogisticRegression(penalty="elasticnet", alpha=1e-3, l1_ratio=0.5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test, threshold=0.4)
        y_proba = model.predict_proba(X_test)
        model.save("model")
        model = LogisticRegression.load("model")

    Notes:
        - Default batch_size=1 adalah stochastic gradient ascent pada log conditional likelihood.
        - Penalti L2 masuk ke gradien, penalti L1 lewat proximal step (soft-thresholding,
          Parikh & Boyd 2014, Proximal Algorithms, sec. 6.5.2).
        - trajectory_ berukuran (epoch + 1, 1 + n_features), baris pertama adalah titik awal.
          Tidak ikut tersimpan oleh save.
    """

    w_: FloatArray
    bias_: float
    trajectory_: FloatArray

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 100,
        penalty: Penalty = None,
        alpha: float = 1e-2,
        l1_ratio: float = 0.5,
        tol: float = 1e-6,
        batch_size: int | None = 1,
        seed: int = 67,
        init_bias: float = 0.0,
        init_weights: Sequence[float] | None = None,
        track_params: bool = False,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.batch_size = batch_size
        self.seed = seed
        self.init_bias = init_bias
        self.init_weights = None if init_weights is None else list(init_weights)
        self.track_params = track_params
        self.history_: list[float] = []

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        """Latih model pada X dan y bernilai 0 atau 1."""
        if self.penalty not in _VALID_PENALTIES:
            raise ValueError(f"penalty tidak dikenal: {self.penalty}")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        batch_size = (
            n_samples if self.batch_size is None else min(self.batch_size, n_samples)
        )
        l1_strength, l2_strength = self._penalty_strengths()
        rng = np.random.default_rng(self.seed)

        self.w_ = self._initial_weights(n_features)
        self.bias_ = float(self.init_bias)
        self.history_ = []
        trajectory = [self._parameters()]
        previous_objective = np.inf

        for _ in range(self.epochs):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch = order[start : start + batch_size]
                self._step(X[batch], y[batch], l1_strength, l2_strength)

            objective = self._objective(X, y, l1_strength, l2_strength)
            self.history_.append(objective)
            trajectory.append(self._parameters())
            if abs(previous_objective - objective) < self.tol:
                break
            previous_objective = objective

        self.trajectory_ = (
            np.array(trajectory) if self.track_params else np.empty((0, n_features + 1))
        )
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Return P(y=1) dengan shape (n_samples,)."""
        return _sigmoid(np.asarray(X, dtype=np.float64) @ self.w_ + self.bias_)

    def predict(self, X: FloatArray, threshold: float = 0.5) -> IntArray:
        """Return 1 jika P(y=1) >= threshold, selain itu 0."""
        return (self.predict_proba(X) >= threshold).astype(np.int_)

    def get_state(self) -> tuple[Params, Arrays]:
        """Return (params, arrays) untuk save. Bobot disimpan sebagai array."""
        params: Params = {
            "lr": self.lr,
            "epochs": self.epochs,
            "penalty": self.penalty,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "tol": self.tol,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "init_bias": self.init_bias,
            "init_weights": self.init_weights,
            "track_params": self.track_params,
            "bias": float(self.bias_),
            "history": self.history_,
        }
        return params, {"w": self.w_}

    def set_state(self, params: Params, arrays: Arrays) -> None:
        """Rebuild model dari hasil get_state."""
        hyperparameters = {
            key: value
            for key, value in params.items()
            if key not in ("bias", "history")
        }
        self.__init__(**hyperparameters)
        self.bias_ = params["bias"]
        self.history_ = params["history"]
        self.w_ = arrays["w"]
        self.trajectory_ = np.empty((0, len(self.w_) + 1))

    def _initial_weights(self, n_features: int) -> FloatArray:
        """Return bobot awal, nol jika init_weights tidak diberikan."""
        if self.init_weights is None:
            return np.zeros(n_features)
        if len(self.init_weights) != n_features:
            raise ValueError(
                f"init_weights berisi {len(self.init_weights)} nilai, fitur ada {n_features}"
            )
        return np.array(self.init_weights, dtype=np.float64)

    def _parameters(self) -> FloatArray:
        """Return (bias, bobot) saat ini sebagai satu vektor."""
        return np.concatenate([[self.bias_], self.w_])

    def _penalty_strengths(self) -> tuple[float, float]:
        """Return (kekuatan L1, kekuatan L2) sesuai penalty."""
        if self.penalty == "l1":
            return self.alpha, 0.0
        if self.penalty == "l2":
            return 0.0, self.alpha
        if self.penalty == "elasticnet":
            return self.alpha * self.l1_ratio, self.alpha * (1 - self.l1_ratio)
        return 0.0, 0.0

    def _step(
        self,
        X_batch: FloatArray,
        y_batch: FloatArray,
        l1_strength: float,
        l2_strength: float,
    ) -> None:
        """1 update gradient pada 1 batch, diikuti proximal step jika ada L1."""
        error = _sigmoid(X_batch @ self.w_ + self.bias_) - y_batch
        weight_gradient = X_batch.T @ error / len(y_batch) + l2_strength * self.w_
        self.w_ -= self.lr * weight_gradient
        self.bias_ -= self.lr * error.mean()
        if l1_strength > 0:
            self.w_ = _soft_threshold(self.w_, self.lr * l1_strength)

    def _objective(
        self, X: FloatArray, y: FloatArray, l1_strength: float, l2_strength: float
    ) -> float:
        """Log-loss rata-rata ditambah penalti L1 dan L2 pada bobot."""
        penalty = l1_strength * np.abs(self.w_).sum()
        penalty += 0.5 * l2_strength * np.square(self.w_).sum()
        return log_loss(y, self.predict_proba(X)) + float(penalty)
