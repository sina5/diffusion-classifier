"""Accuracy from an error table: the predicted class is the one with the lowest average error."""

import torch
import torch.nn.functional as F

from diffcls.classify.error_table import ErrorTable
from diffcls.errors import ConfigError


def evenly_spaced(total: int, count: int) -> torch.Tensor:
    """count level indices spread evenly over 0..total-1 (the bin centers of count equal bins)."""
    if not 1 <= count <= total:
        raise ConfigError("count must be between 1 and the number of levels")
    return ((torch.arange(count, dtype=torch.float64) + 0.5) * total / count).floor().long()


class TableClassifier:
    """Predicts classes from an error table, using the levels it is given."""

    def __init__(self, table: ErrorTable) -> None:
        self.table = table

    def scores(self, level_indices: torch.Tensor | None = None) -> torch.Tensor:
        """(N, C) mean error over the chosen levels (all levels if None). Lower means more likely."""
        errors = self.table.errors if level_indices is None else self.table.errors[:, :, level_indices]
        return errors.mean(dim=2)

    def predict(self, level_indices: torch.Tensor | None = None) -> torch.Tensor:
        """(N,) predicted classes."""
        return self.scores(level_indices).argmin(dim=1)

    def accuracy(self, level_indices: torch.Tensor | None = None) -> float:
        """Share of images whose predicted class is the true class."""
        return float((self.predict(level_indices) == self.table.labels).float().mean())

    def accuracy_vs_level_count(self, counts: tuple[int, ...]) -> dict[int, float]:
        """Accuracy with count evenly spaced levels, for each count (the cost is count x classes calls)."""
        total = self.table.errors.shape[2]
        return {count: self.accuracy(evenly_spaced(total, count)) for count in counts}

    def accuracy_per_level(self) -> torch.Tensor:
        """(K,) accuracy when only one level is used."""
        predictions = self.table.errors.argmin(dim=1)  # (N, K)
        return (predictions == self.table.labels[:, None]).float().mean(dim=0)

    def confusion(self, level_indices: torch.Tensor | None = None) -> torch.Tensor:
        """(C, C) counts: row = true class, column = predicted class."""
        classes = self.table.errors.shape[1]
        index = self.table.labels * classes + self.predict(level_indices)
        return torch.bincount(index, minlength=classes * classes).view(classes, classes)


class LevelWindowSelector:
    """Chooses a contiguous range of levels on one set of images, to report on another.

    Choosing the levels on the same images that measure the accuracy would overstate it.
    """

    def __init__(self, min_width: int = 1) -> None:
        if min_width < 1:
            raise ConfigError("min_width must be at least 1")
        self.min_width = min_width

    def select(self, table: ErrorTable) -> torch.Tensor:
        """Indices lo..hi-1 of the window with the highest accuracy on this table (the narrowest on a tie)."""
        total = table.errors.shape[2]
        cumulative = torch.cat([torch.zeros_like(table.errors[:, :, :1]), table.errors.cumsum(dim=2)], dim=2)
        best, best_window = -1.0, (0, total)
        for lo in range(total):
            for hi in range(lo + self.min_width, total + 1):
                window = cumulative[:, :, hi] - cumulative[:, :, lo]
                accuracy = float((window.argmin(dim=1) == table.labels).float().mean())
                narrower = accuracy == best and hi - lo < best_window[1] - best_window[0]
                if accuracy > best or narrower:  # on a tie, fewer levels cost fewer model calls
                    best, best_window = accuracy, (lo, hi)
        return torch.arange(*best_window)


class LevelNormalizer:
    """Puts every noise level on the same scale before the levels are averaged.

    The raw error of a level grows with the size of its training target. For velocity prediction,
    the noisiest levels have the largest errors, so a plain mean lets them decide the class even
    though they carry little class information. This divides each level by its typical spread
    between classes, measured on a separate set of images (fit), after removing the image's mean
    over the classes at that level.
    """

    def __init__(self) -> None:
        self.scales: torch.Tensor | None = None

    def fit(self, table: ErrorTable) -> "LevelNormalizer":
        """Measure the per-level spread on these images."""
        self.scales = self._centered(table.errors).std(dim=1).mean(dim=0)
        return self

    def apply(self, table: ErrorTable) -> ErrorTable:
        """The table with normalized errors. Raises ConfigError if fit() was not called."""
        if self.scales is None:
            raise ConfigError("call fit() before apply()")
        return ErrorTable(table.method, self._centered(table.errors) / self.scales, table.labels, table.levels,
                          table.log_snr)

    @staticmethod
    def _centered(errors: torch.Tensor) -> torch.Tensor:
        return errors - errors.mean(dim=1, keepdim=True)


class LevelWeightFitter:
    """Learns one weight per noise level (shared by all classes) with multinomial logistic regression.

    The class logits are -sum_k w_k * error[n, c, k]. This is the most a per-level weighting can do,
    so it shows how far a simpler weighting is from the best one.
    """

    def __init__(self, max_iter: int = 200) -> None:
        self.max_iter = max_iter
        self.weights: torch.Tensor | None = None

    def fit(self, table: ErrorTable) -> "LevelWeightFitter":
        """Fit the weights on these images."""
        weights = torch.zeros(table.errors.shape[2], requires_grad=True)
        optimizer = torch.optim.LBFGS([weights], max_iter=self.max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(-(table.errors * weights).sum(dim=2), table.labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.weights = weights.detach()
        return self

    def accuracy(self, table: ErrorTable) -> float:
        """Accuracy of the weighted score on these images. Raises ConfigError if fit() was not called."""
        if self.weights is None:
            raise ConfigError("call fit() before accuracy()")
        predicted = (table.errors * self.weights).sum(dim=2).argmin(dim=1)
        return float((predicted == table.labels).float().mean())
