"""Figures for the diffusion classifier. 16:9, PNG, legends outside the data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be set first)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from diffcls import constants as C  # noqa: E402
from diffcls.classify.report import MethodReport  # noqa: E402

FIGSIZE = (16, 9)
DPI = 160
TITLE_SIZE = 20
LABEL_SIZE = 15
TICK_SIZE = 13
METHOD_COLORS = {"ddpm": "#1f77b4", "flow_matching": "#d62728"}
METHOD_NAMES = {"ddpm": "DDPM (noise prediction)", "flow_matching": "Flow matching (velocity prediction)"}
BASELINE_COLORS = ("#2ca02c", "#9467bd", "#7f7f7f")
JOINT_COLORS = ("#ff7f0e", "#17becf", "#8c564b")


def save_figure(fig, path: Path) -> Path:
    """Writes fig as a PNG and closes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


class FigureWriter:
    """Writes the comparison figures from the method reports."""

    def __init__(self, output_dir: Path, baselines: dict[str, float] | None = None) -> None:
        self.output_dir = output_dir
        self.baselines = baselines or {}

    def accuracy_vs_cost(self, reports: list[MethodReport]) -> Path:
        """Held-out accuracy against model calls per image (levels x classes), raw and normalized levels."""
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for report in reports:
            color = METHOD_COLORS[report.method]
            for result, style, kind in ((report.raw, "o--", "plain average"),
                                        (report.normalized, "o-", "normalized levels")):
                counts = sorted(int(c) for c in result.accuracy_vs_level_count)
                calls = [c * C.CIFAR10_NUM_CLASSES for c in counts]
                accuracy = [100 * result.accuracy_vs_level_count[c] for c in counts]
                ax.plot(calls, accuracy, style, lw=2.5, ms=7, color=color, alpha=1.0 if "norm" in kind else 0.55,
                        label=f"{METHOD_NAMES[report.method]}, {kind}")
                ax.annotate(f"{accuracy[-1]:.1f}%", (calls[-1], accuracy[-1]), textcoords="offset points",
                            xytext=(8, -4), fontsize=TICK_SIZE, color=color)
        for (name, value), color in zip(self.baselines.items(), BASELINE_COLORS):
            ax.axhline(100 * value, ls=":", lw=2, color=color, label=f"{name} (trained classifier): {100 * value:.1f}%")
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        held_out = reports[0].held_out_images
        self._style(ax, f"Diffusion classifier accuracy on CIFAR-10 test ({held_out:,} held-out images)",
                    "DiT calls per image (noise levels × 10 classes, log scale)", "Accuracy (%)")
        return self._save(fig, "accuracy_vs_cost.png")

    def accuracy_per_level(self, reports: list[MethodReport]) -> Path:
        """Accuracy when one noise level is used, against the log signal-to-noise ratio."""
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for report in reports:
            color = METHOD_COLORS[report.method]
            ax.plot(report.log_snr, 100 * np.array(report.accuracy_per_level), lw=2.5, color=color,
                    label=f"{METHOD_NAMES[report.method]}, one level")
            lo, hi = report.normalized.window_log_snr
            ax.axvspan(min(lo, hi), max(lo, hi), color=color, alpha=0.12,
                       label=f"{METHOD_NAMES[report.method]}, best window of normalized levels")
        ax.axhline(100 / C.CIFAR10_NUM_CLASSES, ls=":", lw=2, color="gray", label="Chance (10%)")
        ax.invert_xaxis()  # left = clean, right = noisy, as in the sampling direction reversed
        ax.set_ylim(0, 100)
        self._style(ax, "Which noise levels carry the class? Accuracy from a single noise level",
                    "log signal-to-noise ratio (left: little noise, right: much noise)", "Accuracy (%)",
                    legend_loc="upper right")
        return self._save(fig, "accuracy_per_noise_level.png")

    def confusion(self, reports: list[MethodReport]) -> Path:
        """Row-normalized confusion matrices, one panel per method."""
        fig, axes = plt.subplots(1, len(reports), figsize=FIGSIZE, squeeze=False)
        for ax, report in zip(axes[0], reports):
            matrix = np.array(report.confusion, dtype=float)
            matrix = 100 * matrix / matrix.sum(axis=1, keepdims=True)
            ax.imshow(matrix, cmap="viridis", vmin=0, vmax=100)
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=9,
                            color="black" if matrix[i, j] > 60 else "white")
            ax.set_xticks(range(len(C.CIFAR10_CLASSES)), C.CIFAR10_CLASSES, rotation=45, ha="right",
                          fontsize=TICK_SIZE - 2)
            ax.set_yticks(range(len(C.CIFAR10_CLASSES)), C.CIFAR10_CLASSES, fontsize=TICK_SIZE - 2)
            ax.set_xlabel("Predicted class", fontsize=LABEL_SIZE)
            ax.set_ylabel("True class", fontsize=LABEL_SIZE)
            ax.set_title(f"{METHOD_NAMES[report.method]}\n{100 * report.normalized.accuracy_all_levels:.1f}% with "
                         f"all {report.levels} levels, normalized", fontsize=LABEL_SIZE)
        fig.suptitle(f"Confusion matrices (% of each true class, {reports[0].held_out_images:,} held-out images)",
                     fontsize=TITLE_SIZE)
        fig.tight_layout()
        return self._save(fig, "confusion.png")

    @staticmethod
    def _style(ax, title: str, xlabel: str, ylabel: str, legend_loc: str = "lower right") -> None:
        """Titles, grid and a legend inside the axes; legend_loc must be a corner without data."""
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=TICK_SIZE, loc=legend_loc)

    def _save(self, fig, name: str) -> Path:
        return save_figure(fig, self.output_dir / name)


class JointFigureWriter:
    """Figures for the joint image and label runs, from their histories, evaluations and snapshots."""

    def __init__(self, output_dir: Path, run_labels: list[str]) -> None:
        self.output_dir = output_dir
        self.colors = dict(zip(run_labels, JOINT_COLORS))  # one color per run in every figure

    def classification_cost(self, joint_runs: dict[str, list[dict]], reports: list[MethodReport],
                            baselines: dict[str, float]) -> Path:
        """Accuracy against DiT calls per image: the joint runs and the diffusion classifiers on one axis.

        Each diffusion classifier shows its better scoring over all levels (plain average or normalized).
        """
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for report in reports:
            normalized_wins = report.normalized.accuracy_all_levels >= report.raw.accuracy_all_levels
            result = report.normalized if normalized_wins else report.raw
            counts = sorted(int(c) for c in result.accuracy_vs_level_count)
            ax.plot([c * C.CIFAR10_NUM_CLASSES for c in counts],
                    [100 * result.accuracy_vs_level_count[c] for c in counts], "o-", lw=2.5, ms=7,
                    color=METHOD_COLORS[report.method],
                    label=f"Diffusion classifier, {METHOD_NAMES[report.method]}, "
                          f"{'normalized levels' if normalized_wins else 'plain average'}")
        for name, rows in joint_runs.items():
            ax.plot([r["dit_calls"] for r in rows], [100 * r["accuracy"] for r in rows], "s-", lw=2.5, ms=7,
                    color=self.colors[name], label=f"Joint model, {name}")
        for (name, value), color in zip(baselines.items(), BASELINE_COLORS):
            ax.axhline(100 * value, ls=":", lw=2, color=color, label=f"{name} (trained classifier): {100 * value:.1f}%")
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        FigureWriter._style(ax, "Classification accuracy on CIFAR-10 test against cost",
                            "DiT calls per image (log scale)", "Accuracy (%)")
        return save_figure(fig, self.output_dir / "joint_accuracy_vs_cost.png")

    def training_accuracy(self, histories: dict[str, dict]) -> Path:
        """Quick test accuracy of the EMA weights during training."""
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for name, history in histories.items():
            ax.plot(history["eval_epochs"], [100 * a for a in history["eval_accuracy"]], "o-", lw=2.5, ms=5,
                    color=self.colors[name], label=name)
        ax.set_ylim(0, 100)
        FigureWriter._style(ax, f"Joint model accuracy during training ({C.EVAL_IMAGES:,} test images, "
                                f"{C.EVAL_CLASSIFY_STEPS} label steps, EMA weights)", "Epoch", "Accuracy (%)")
        return save_figure(fig, self.output_dir / "joint_training_accuracy.png")

    def sample_grids(self, grids: dict[str, torch.Tensor]) -> Path:
        """One panel per run: SAMPLES_PER_CLASS samples per class (rows), from the same noise."""
        height = FIGSIZE[0] / len(grids) + 1.2  # square panels plus the titles; a 16:9 canvas leaves a gap
        fig, axes = plt.subplots(1, len(grids), figsize=(FIGSIZE[0], height), squeeze=False, layout="constrained")
        for ax, (name, grid) in zip(axes[0], grids.items()):
            ax.imshow(self._tile(grid))
            ax.set_title(name, fontsize=LABEL_SIZE)
            ax.set_yticks([(i + 0.5) * (grid.shape[-1] + 2) for i in range(C.CIFAR10_NUM_CLASSES)],
                          C.CIFAR10_CLASSES, fontsize=TICK_SIZE - 2)
            ax.set_xticks([])
        fig.suptitle("Joint model samples, guidance 2, 16 Euler steps (same noise in every panel)",
                     fontsize=TITLE_SIZE)
        return save_figure(fig, self.output_dir / "joint_samples.png")

    @staticmethod
    def _tile(grid: torch.Tensor) -> np.ndarray:
        """(classes * per_class, 3, H, W) in [-1, 1] to one image with a 2-pixel white gap."""
        images = ((grid.float().clamp(-1, 1) + 1) / 2).numpy().transpose(0, 2, 3, 1)
        per_class, size = len(images) // C.CIFAR10_NUM_CLASSES, images.shape[1]
        step = size + 2
        canvas = np.ones((C.CIFAR10_NUM_CLASSES * step, per_class * step, 3))
        for index, image in enumerate(images):
            row, col = divmod(index, per_class)
            canvas[row * step + 1:row * step + 1 + size, col * step + 1:col * step + 1 + size] = image
        return canvas
