from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ..model.base import FloatArray
from ..model.logreg import LogisticRegression, Penalty

_GRID_SIZE = 140
_GRID_MARGIN = 0.8
_CONTOUR_LEVELS = 25
_LABELED_LEVEL_STEP = 3


class _Variant(NamedTuple):
    name: str
    penalty: Penalty
    l1_strength: float
    l2_strength: float


@dataclass
class _Panel:
    axes: Axes
    variant: _Variant
    path: FloatArray
    line: Line2D
    point: Line2D


def _log_loss_surface(
    feature: FloatArray, labels: FloatArray, bias: FloatArray, weight: FloatArray
) -> FloatArray:
    """Log-loss rata-rata..

    bias dan weight bisa berupa array yang di-broadcast, bentuk stabil dari log-loss
    log(1 + e^z) - y * z dipakai agar tidak overflow.
    """
    logits = np.asarray(bias)[..., None] + np.asarray(weight)[..., None] * feature
    return np.mean(np.logaddexp(0, logits) - labels * logits, axis=-1)


def _variants(regularization: float, l1_ratio: float) -> list[_Variant]:
    return [
        _Variant("Tanpa regularisasi", None, 0.0, 0.0),
        _Variant(f"L1 (λ={regularization})", "l1", regularization, 0.0),
        _Variant(f"L2 (λ={regularization})", "l2", 0.0, regularization),
        _Variant(
            f"Elastic Net (λ={regularization}, α={l1_ratio})",
            "elasticnet",
            regularization * l1_ratio,
            regularization * (1 - l1_ratio),
        ),
    ]


def _train_path(
    feature: FloatArray,
    labels: FloatArray,
    variant: _Variant,
    regularization: float,
    l1_ratio: float,
    lr: float,
    n_steps: int,
    init: tuple[float, float],
) -> FloatArray:
    """Latih model 2 parameter secara full-batch, return lintasan (bias, bobot) per langkah."""
    model = LogisticRegression(
        lr=lr,
        epochs=n_steps,
        penalty=variant.penalty,
        alpha=regularization,
        l1_ratio=l1_ratio,
        tol=0.0,
        batch_size=None,
        init_bias=init[0],
        init_weights=[init[1]],
        track_params=True,
    )
    model.fit(feature[:, None], labels)
    return model.trajectory_


def save_loss_contour_video(
    feature: FloatArray,
    labels: FloatArray,
    path: str | Path,
    feature_name: str = "x",
    regularization: float = 0.1,
    l1_ratio: float = 0.5,
    lr: float = 0.5,
    n_steps: int = 120,
    init: tuple[float, float] = (2.0, -2.5),
    hold_frames: int = 25,
    fps: int = 12,
    dpi: int = 110,
) -> dict[str, tuple[float, float]]:
    """Simpan video kontur log-loss dan lintasan (bias, bobot) untuk 4 varian regularisasi.

    Parameter:
        feature: satu fitur terstandardisasi, shape (n_samples,).
        labels: label 0 atau 1, shape (n_samples,).
        path: lokasi file mp4.
        feature_name: nama fitur untuk label sumbu.
        regularization: kekuatan regularisasi (λ). Sengaja dibesarkan agar efeknya terlihat.
        l1_ratio: porsi L1 pada Elastic Net (α).
        lr: learning rate.
        n_steps: jumlah langkah full-batch gradient descent.
        init: titik awal (bias, bobot), sama untuk semua varian.
        hold_frames: jumlah frame terakhir yang ditahan.
        fps: frame per detik.
        dpi: resolusi video.

    Usage:
        final_params = save_loss_contour_video(x_train, y_train, "output/video.mp4", "log_lead_time")

    Notes:
        - Kontur adalah J(bias, bobot) = log-loss + penalti, dengan penalti hanya pada bobot.
        - Perlu ffmpeg yang bisa dipanggil oleh matplotlib.
        - Return dict nama varian ke (bias, bobot) di langkah terakhir.
    """
    feature = np.asarray(feature, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    variants = _variants(regularization, l1_ratio)
    paths = [
        _train_path(
            feature, labels, variant, regularization, l1_ratio, lr, n_steps, init
        )
        for variant in variants
    ]

    all_points = np.vstack(paths)
    bias_grid = np.linspace(
        all_points[:, 0].min() - _GRID_MARGIN,
        all_points[:, 0].max() + _GRID_MARGIN,
        _GRID_SIZE,
    )
    weight_grid = np.linspace(
        all_points[:, 1].min() - _GRID_MARGIN,
        all_points[:, 1].max() + _GRID_MARGIN,
        _GRID_SIZE,
    )
    bias_mesh, weight_mesh = np.meshgrid(bias_grid, weight_grid)
    base_loss = _log_loss_surface(feature, labels, bias_mesh, weight_mesh)

    figure, all_axes = plt.subplots(2, 2, figsize=(12, 10.5))
    panels = [
        _draw_panel(
            axes,
            variant,
            variant_path,
            base_loss,
            (bias_grid, weight_grid),
            init,
            feature_name,
        )
        for axes, variant, variant_path in zip(all_axes.ravel(), variants, paths)
    ]

    def update(step: int) -> list[Line2D]:
        artists: list[Line2D] = []
        for panel in panels:
            bias, weight = panel.path[step]
            panel.line.set_data(panel.path[: step + 1, 0], panel.path[: step + 1, 1])
            panel.point.set_data([bias], [weight])
            loss = float(_log_loss_surface(feature, labels, bias, weight))
            objective = (
                loss
                + panel.variant.l1_strength * abs(weight)
                + 0.5 * panel.variant.l2_strength * weight**2
            )
            panel.axes.set_title(
                f"{panel.variant.name}\n"
                f"θ=({bias:.3f}, {weight:.3f})  log-loss={loss:.4f}  J={objective:.4f}",
                fontsize=10,
            )
            artists += [panel.line, panel.point]
        figure.suptitle(
            f"Kontur log-loss & lintasan (θ₀, θ₁), iterasi {step}/{n_steps}",
            fontsize=14,
        )
        return artists

    update(0)
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
    frames = list(range(n_steps + 1)) + [n_steps] * hold_frames
    movie = animation.FuncAnimation(figure, update, frames=frames, blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=3000), dpi=dpi)
    plt.close(figure)

    return {
        variant.name: (float(variant_path[-1, 0]), float(variant_path[-1, 1]))
        for variant, variant_path in zip(variants, paths)
    }


def _draw_panel(
    axes: Axes,
    variant: _Variant,
    path: FloatArray,
    base_loss: FloatArray,
    grids: tuple[FloatArray, FloatArray],
    init: tuple[float, float],
    feature_name: str,
) -> _Panel:
    """Gambar kontur J, minimum, dan titik awal, return panel dengan artist lintasan kosong."""
    bias_grid, weight_grid = grids
    _, weight_mesh = np.meshgrid(bias_grid, weight_grid)
    objective = (
        base_loss
        + variant.l1_strength * np.abs(weight_mesh)
        + 0.5 * variant.l2_strength * weight_mesh**2
    )
    levels = np.geomspace(objective.min() * 1.002, objective.max(), _CONTOUR_LEVELS)
    axes.contourf(
        bias_grid, weight_grid, objective, levels=levels, cmap="viridis", alpha=0.85
    )
    labeled = axes.contour(
        bias_grid,
        weight_grid,
        objective,
        levels=levels[::_LABELED_LEVEL_STEP],
        colors="white",
        linewidths=0.6,
    )
    axes.clabel(labeled, fmt="%.2f", fontsize=7)

    weight_index, bias_index = np.unravel_index(np.argmin(objective), objective.shape)
    axes.plot(
        bias_grid[bias_index],
        weight_grid[weight_index],
        marker="*",
        markersize=16,
        color="gold",
        markeredgecolor="black",
        label="minimum objektif",
    )
    axes.plot(
        *init,
        "o",
        color="white",
        markeredgecolor="black",
        markersize=7,
        label="titik awal",
    )
    (line,) = axes.plot([], [], "-", color="red", linewidth=2, label="lintasan GD")
    (point,) = axes.plot(
        [], [], "o", color="red", markeredgecolor="black", markersize=8
    )
    axes.set_xlabel("θ₀ (bias)")
    axes.set_ylabel(f"θ₁ (bobot {feature_name})")
    axes.legend(loc="upper right", fontsize=8)
    return _Panel(axes, variant, path, line, point)
