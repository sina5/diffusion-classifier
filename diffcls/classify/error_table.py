"""The table of denoising errors: one value per test image, candidate class and noise level.

For each image and level, one noise draw is shared by all candidate classes, so the only
difference between the classes is the label that the DiT gets (Li et al., 2023). The table is
computed once on the GPU; every accuracy in this project comes from it without another pass.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from diffcls import constants as C
from diffcls.classify.processes import MethodKind, NoiseProcess
from diffcls.data import to_model_range
from diffcls.errors import CheckpointError, ConfigError

NOISE_SEED_STRIDE = 1_000_003  # seed * stride + image index: different seeds never share a stream


@dataclass
class ErrorTable:
    """errors[n, c, k]: mean squared error of the DiT output for image n, given class c, at level k."""

    method: MethodKind
    errors: torch.Tensor  # (N, C, K) float32
    labels: torch.Tensor  # (N,) true classes
    levels: torch.Tensor  # (K,) in (0, 1)
    log_snr: torch.Tensor  # (K,)

    def __post_init__(self) -> None:
        n, _, k = self.errors.shape
        if len(self.labels) != n or len(self.levels) != k or len(self.log_snr) != k:
            raise ConfigError("errors, labels, levels and log_snr have inconsistent sizes")

    def subset(self, rows: torch.Tensor | slice) -> "ErrorTable":
        """The same table for a subset of the images."""
        return ErrorTable(self.method, self.errors[rows], self.labels[rows], self.levels, self.log_snr)

    def save(self, path: Path) -> None:
        """Write the table as a .pt file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"method": self.method.value, "errors": self.errors, "labels": self.labels,
                    "levels": self.levels, "log_snr": self.log_snr}, path)

    @classmethod
    def load(cls, path: Path) -> "ErrorTable":
        """Read a table that save() wrote."""
        if not path.exists():
            raise CheckpointError(f"no error table at {path.name}; run scripts/compute_errors.py first")
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(MethodKind(state["method"]), state["errors"], state["labels"], state["levels"], state["log_snr"])


class ErrorTableBuilder:
    """Runs the DiT on every (image, class, level) and records the denoising error."""

    def __init__(self, model: nn.Module, process: NoiseProcess, num_classes: int = C.CIFAR10_NUM_CLASSES,
                 levels: int = C.ERROR_LEVELS, batch_size: int = C.ERROR_BATCH, seed: int = C.ERROR_SEED,
                 bf16: bool = True, verbose: bool = True) -> None:
        self.model = model.eval()
        self.process = process
        self.num_classes = num_classes
        self.levels = NoiseProcess.levels(levels)
        self.batch_size = batch_size
        self.seed = seed
        self.bf16 = bf16
        self.verbose = verbose

    @torch.inference_mode()
    def build(self, images: torch.Tensor, labels: torch.Tensor) -> ErrorTable:
        """images: uint8 (N, 3, H, W). The noise depends only on the seed and the image index, not the batch."""
        device = next(self.model.parameters()).device
        rows, start = [], time.perf_counter()
        for first in range(0, len(images), self.batch_size):
            x0 = to_model_range(images[first:first + self.batch_size].to(device))
            noise = self._noise(range(first, first + len(x0)), x0.shape[1:], device)
            rows.append(self._batch_errors(x0, noise).cpu())
            if self.verbose and (first // self.batch_size) % 10 == 0:
                print(f"[errors] {first + len(x0):6d}/{len(images)} images, {time.perf_counter() - start:.0f}s",
                      flush=True)
        log_snr = torch.tensor([self.process.log_snr(float(level)) for level in self.levels])
        return ErrorTable(self.process.kind, torch.cat(rows), labels.clone(), self.levels.float(), log_snr.float())

    def _noise(self, indices: range, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """(K, B, *shape) noise. Each image has its own generator, seeded by its index."""
        per_image = []
        for index in indices:
            generator = torch.Generator(device=device).manual_seed(self.seed * NOISE_SEED_STRIDE + index)
            per_image.append(torch.randn((len(self.levels), *shape), generator=generator, device=device))
        return torch.stack(per_image, dim=1)

    def _batch_errors(self, x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """(B, C, K) errors for one batch of clean images; noise is (K, B, 3, H, W)."""
        batch, classes = len(x0), self.num_classes
        y = torch.arange(classes, device=x0.device).repeat(batch)  # image-major: b0c0, b0c1, ...
        out = torch.empty(batch, classes, len(self.levels))
        for k, level in enumerate(self.levels.tolist()):
            x_t, target = self.process.noisy(x0, level, noise[k])
            t = torch.full((batch * classes,), self.process.time_input(level), device=x0.device)
            with torch.autocast(x0.device.type, dtype=torch.bfloat16, enabled=self.bf16 and x0.is_cuda):
                prediction = self.model(x_t.repeat_interleave(classes, dim=0), t, y).float()
            error = (prediction - target.repeat_interleave(classes, dim=0)).pow(2).flatten(1).mean(dim=1)
            out[:, :, k] = error.view(batch, classes).cpu()
        return out
