"""Flow matching on the image and the label at the same time, and the samplers that use it.

Both parts follow the straight path z_t = (1 - t) z_0 + t noise (t = 1 is pure noise), and the
network predicts the velocity noise - z_0 of each part. The training mode of each example fixes
the times (see constants.JOINT_MODE_SHARES). Sampling then sets the times for the task:

- classify: the image stays clean (t_x = 0) and the label goes from noise to clean;
- generate: the label stays clean (t_y = 0) and the image goes from noise to clean, with
  classifier-free guidance against the unconditional prediction (t_y = 1, label = noise).
"""

from enum import Enum

import torch
import torch.nn.functional as F

from diffcls import constants as C
from diffcls.errors import ConfigError
from diffcls.joint.model import JointDiT

PIXEL_MIN, PIXEL_MAX = -1.0, 1.0


class TrainingMode(Enum):
    """Which parts of an example stay clean during training."""

    GENERATE = "generate"
    CLASSIFY = "classify"
    UNCONDITIONAL = "unconditional"
    INDEPENDENT = "independent"


class LabelCodec:
    """Class index <-> label vector (+1 at the class, -1 elsewhere)."""

    def __init__(self, num_classes: int = C.CIFAR10_NUM_CLASSES) -> None:
        self.num_classes = num_classes

    def encode(self, labels: torch.Tensor) -> torch.Tensor:
        """(B,) indices to (B, classes) vectors."""
        return F.one_hot(labels, self.num_classes).float() * 2 - 1

    @staticmethod
    def decode(vectors: torch.Tensor) -> torch.Tensor:
        """(B, classes) vectors to (B,) indices: the largest entry."""
        return vectors.argmax(dim=1)


def _broadcast(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return t.view(-1, *([1] * (like.dim() - 1)))


class JointFlow:
    """Loss and samplers of the joint model."""

    def __init__(self, shares: dict[str, float] = C.JOINT_MODE_SHARES,
                 label_weight: float = C.JOINT_LABEL_LOSS_WEIGHT, guidance: float = C.GUIDANCE_SCALE,
                 time_scale: float = C.TIME_SCALE, codec: LabelCodec | None = None) -> None:
        if set(shares) != {m.value for m in TrainingMode} or abs(sum(shares.values()) - 1.0) > 1e-6:
            raise ConfigError("shares must give every training mode a share, and the shares must sum to 1")
        self.modes = list(TrainingMode)
        self.shares = torch.tensor([shares[m.value] for m in self.modes])
        self.label_weight = label_weight
        self.guidance = guidance
        self.time_scale = time_scale
        self.codec = codec or LabelCodec()

    # --- training ------------------------------------------------------------------

    def sample_times(self, count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """(t_x, t_y) per example, from the training mode that each example draws."""
        modes = torch.multinomial(self.shares, count, replacement=True).to(device)
        t_x = torch.sigmoid(C.FLOW_LOGIT_MEAN + C.FLOW_LOGIT_STD * torch.randn(count, device=device))
        t_y = torch.rand(count, device=device)
        index = {m: i for i, m in enumerate(self.modes)}
        t_y = torch.where(modes == index[TrainingMode.GENERATE], torch.zeros_like(t_y), t_y)
        t_x = torch.where(modes == index[TrainingMode.CLASSIFY], torch.zeros_like(t_x), t_x)
        t_y = torch.where(modes == index[TrainingMode.UNCONDITIONAL], torch.ones_like(t_y), t_y)
        return t_x, t_y

    def loss(self, model: JointDiT, x0: torch.Tensor,
             labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Image MSE + label_weight * label MSE, both per element. Also returns the two parts."""
        y0 = self.codec.encode(labels)
        t_x, t_y = self.sample_times(len(x0), x0.device)
        noise_x, noise_y = torch.randn_like(x0), torch.randn_like(y0)
        x_t = (1 - _broadcast(t_x, x0)) * x0 + _broadcast(t_x, x0) * noise_x
        y_t = (1 - _broadcast(t_y, y0)) * y0 + _broadcast(t_y, y0) * noise_y
        v_x, v_y = model(x_t, y_t, t_x * self.time_scale, t_y * self.time_scale)
        image_loss = F.mse_loss(v_x.float(), noise_x - x0)
        label_loss = F.mse_loss(v_y.float(), noise_y - y0)
        return image_loss + self.label_weight * label_loss, {"image": image_loss.detach(), "label": label_loss.detach()}

    # --- sampling --------------------------------------------------------------------

    @torch.inference_mode()
    def classify(self, model: JointDiT, images: torch.Tensor, steps: int,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Predicted classes for clean images in [-1, 1]: Euler steps on the label from noise to t = 0."""
        if steps < 1:
            raise ConfigError("steps must be at least 1")
        y = torch.randn(len(images), self.codec.num_classes, device=images.device, generator=generator)
        zero = torch.zeros(len(images), device=images.device)
        grid = torch.linspace(1.0, 0.0, steps + 1).tolist()
        for t, s in zip(grid[:-1], grid[1:]):
            _, v_y = model(images, y, zero, torch.full_like(zero, t * self.time_scale))
            y = y - (t - s) * v_y.float()
        return self.codec.decode(y)

    @torch.inference_mode()
    def generate(self, model: JointDiT, noise: torch.Tensor, labels: torch.Tensor, steps: int) -> torch.Tensor:
        """Images in [-1, 1] for the labels: Euler steps on the image with a clean label and guidance."""
        if steps < 1:
            raise ConfigError("steps must be at least 1")
        y_clean = self.codec.encode(labels)
        y_noise = torch.randn_like(y_clean)  # the unconditional input: a label at t = 1 is pure noise
        zero, one = torch.zeros(len(noise), device=noise.device), torch.ones(len(noise), device=noise.device)
        x, grid = noise, torch.linspace(1.0, 0.0, steps + 1).tolist()
        for t, s in zip(grid[:-1], grid[1:]):
            t_x = torch.full_like(zero, t * self.time_scale)
            v = self._guided_velocity(model, x, y_clean, y_noise, t_x, zero, one * self.time_scale)
            x = x - (t - s) * v
        return x.clamp(PIXEL_MIN, PIXEL_MAX)

    def _guided_velocity(self, model: JointDiT, x: torch.Tensor, y_clean: torch.Tensor, y_noise: torch.Tensor,
                         t_x: torch.Tensor, t_clean: torch.Tensor, t_noise: torch.Tensor) -> torch.Tensor:
        """v_uncond + w (v_cond - v_uncond), from one batched forward pass."""
        if self.guidance == 1.0:
            return model(x, y_clean, t_x, t_clean)[0].float()
        v, _ = model(torch.cat([x, x]), torch.cat([y_clean, y_noise]), torch.cat([t_x, t_x]),
                     torch.cat([t_clean, t_noise]))
        cond, uncond = v.float().chunk(2)
        return uncond + self.guidance * (cond - uncond)
