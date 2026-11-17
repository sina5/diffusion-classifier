"""Turns an error table into the numbers that the README and the figures use.

Every choice that is fitted to data (the level window, the level scales, the level weights) is
fitted on the first selection_split test images. The "held_out" numbers use only the other images,
so those choices cannot inflate them.
"""

from dataclasses import asdict, dataclass

import torch

from diffcls import constants as C
from diffcls.classify.analysis import LevelNormalizer, LevelWeightFitter, LevelWindowSelector, TableClassifier
from diffcls.classify.error_table import ErrorTable


@dataclass(frozen=True)
class ScoringResult:
    """Held-out accuracy of one way to combine the noise levels."""

    accuracy_all_levels: float
    accuracy_vs_level_count: dict[int, float]
    window_levels: tuple[int, int]  # chosen on the selection images, end exclusive
    window_log_snr: tuple[float, float]
    accuracy_window: float


@dataclass(frozen=True)
class MethodReport:
    """Accuracies of one method's diffusion classifier on the CIFAR-10 test set."""

    method: str
    images: int
    levels: int
    accuracy_all_levels: float  # raw mean over all levels, on all test images (no fitted choice)
    accuracy_per_level: list[float]
    log_snr: list[float]
    held_out_images: int
    raw: ScoringResult
    normalized: ScoringResult
    held_out_accuracy_learned_weights: float
    per_class_accuracy: dict[str, float]  # held-out, normalized scores over all levels
    confusion: list[list[int]]  # held-out, normalized scores over all levels

    def to_dict(self) -> dict:
        """JSON-friendly form."""
        return asdict(self)


class ReportBuilder:
    """Computes a MethodReport for one error table."""

    def __init__(self, counts: tuple[int, ...] = C.LEVEL_COUNTS, selection_split: int = C.SELECTION_SPLIT,
                 class_names: tuple[str, ...] = C.CIFAR10_CLASSES) -> None:
        self.counts = counts
        self.selection_split = selection_split
        self.class_names = class_names

    def build(self, table: ErrorTable) -> MethodReport:
        """All numbers for one table."""
        selection = table.subset(slice(0, self.selection_split))
        held_out = table.subset(slice(self.selection_split, None))
        normalizer = LevelNormalizer().fit(selection)
        normalized_held_out = normalizer.apply(held_out)
        confusion = TableClassifier(normalized_held_out).confusion()
        return MethodReport(
            method=table.method.value, images=len(table.labels), levels=table.errors.shape[2],
            accuracy_all_levels=TableClassifier(table).accuracy(),
            accuracy_per_level=TableClassifier(table).accuracy_per_level().tolist(),
            log_snr=table.log_snr.tolist(),
            held_out_images=len(held_out.labels),
            raw=self._scoring(selection, held_out),
            normalized=self._scoring(normalizer.apply(selection), normalized_held_out),
            held_out_accuracy_learned_weights=LevelWeightFitter().fit(normalizer.apply(selection)).accuracy(
                normalized_held_out),
            per_class_accuracy={name: float(confusion[i, i] / confusion[i].sum())
                                for i, name in enumerate(self.class_names)},
            confusion=confusion.tolist(),
        )

    def _scoring(self, selection: ErrorTable, held_out: ErrorTable) -> ScoringResult:
        classifier, total = TableClassifier(held_out), held_out.errors.shape[2]
        window = LevelWindowSelector().select(selection)
        return ScoringResult(
            accuracy_all_levels=classifier.accuracy(),
            accuracy_vs_level_count=classifier.accuracy_vs_level_count(tuple(c for c in self.counts if c <= total)),
            window_levels=(int(window[0]), int(window[-1]) + 1),
            window_log_snr=(float(held_out.log_snr[window[0]]), float(held_out.log_snr[window[-1]])),
            accuracy_window=classifier.accuracy(window),
        )

    @staticmethod
    def level_indices(result: ScoringResult) -> torch.Tensor:
        """The chosen window as level indices."""
        return torch.arange(*result.window_levels)
