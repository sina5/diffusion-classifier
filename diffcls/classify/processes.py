"""The noise process of each trained method: how to noise an image, and what the DiT predicts.

A diffusion classifier needs three things from a method at each noise level: the noisy image,
the training target, and the time value that the DiT gets. The error between the DiT output and
the target is what the classifier compares across classes.
"""

from abc import ABC, abstractmethod
from enum import Enum

import torch

from diffcls import constants as C
from diffcls.errors import ConfigError


class MethodKind(Enum):
    """Training methods with a checkpoint that this project can score."""

    DDPM = "ddpm"
    FLOW_MATCHING = "flow_matching"


def _broadcast(values: torch.Tensor) -> torch.Tensor:
    return values.view(-1, 1, 1, 1)


class NoiseProcess(ABC):
    """Maps a level in (0, 1) (0 = clean, 1 = pure noise) to a noisy image, a target and a time input."""

    kind: MethodKind

    @staticmethod
    def levels(count: int) -> torch.Tensor:
        """count evenly spaced levels at the centers of count equal bins of (0, 1)."""
        if count < 1:
            raise ConfigError("count must be at least 1")
        return (torch.arange(count, dtype=torch.float64) + 0.5) / count

    @abstractmethod
    def noisy(self, x0: torch.Tensor, level: float, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_t, target) for clean images x0 at one level."""

    @abstractmethod
    def time_input(self, level: float) -> float:
        """The time value that the DiT was trained with at this level."""

    @abstractmethod
    def log_snr(self, level: float) -> float:
        """log(signal power / noise power) at this level, to compare the methods on one axis."""


class DdpmProcess(NoiseProcess):
    """DDPM with a linear beta schedule; the DiT predicts the noise. Levels map to timesteps 0..T-1."""

    kind = MethodKind.DDPM

    def __init__(self, timesteps: int = C.TIMESTEPS, beta_start: float = C.BETA_START,
                 beta_end: float = C.BETA_END) -> None:
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        self.timesteps = timesteps

    def timestep(self, level: float) -> int:
        """The discrete timestep for a level."""
        return min(self.timesteps - 1, int(level * self.timesteps))

    def noisy(self, x0: torch.Tensor, level: float, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = float(self.alpha_bar[self.timestep(level)])
        return a ** 0.5 * x0 + (1.0 - a) ** 0.5 * noise, noise

    def time_input(self, level: float) -> float:
        return float(self.timestep(level))

    def log_snr(self, level: float) -> float:
        a = self.alpha_bar[self.timestep(level)]
        return float(torch.log(a / (1.0 - a)))


class FlowMatchingProcess(NoiseProcess):
    """Linear path x_t = (1 - t) x0 + t noise; the DiT predicts the velocity noise - x0."""

    kind = MethodKind.FLOW_MATCHING

    def __init__(self, time_scale: float = C.TIME_SCALE) -> None:
        self.time_scale = time_scale

    def noisy(self, x0: torch.Tensor, level: float, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (1.0 - level) * x0 + level * noise, noise - x0

    def time_input(self, level: float) -> float:
        return level * self.time_scale

    def log_snr(self, level: float) -> float:
        return float(2.0 * torch.log(torch.tensor((1.0 - level) / level, dtype=torch.float64)))


def build_process(kind: MethodKind) -> NoiseProcess:
    """The noise process for a method."""
    return {MethodKind.DDPM: DdpmProcess, MethodKind.FLOW_MATCHING: FlowMatchingProcess}[kind]()
