"""Training loop of the joint model: AdamW, warmup, bf16, EMA, checkpoints and a quick classification check.

The optimizer, schedule, EMA and batch size match tiny-diffusion-transformer, so the image
quality can be compared with its flow-matching run.
"""

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from diffcls import constants as C
from diffcls.data import to_model_range
from diffcls.errors import CheckpointError, ConfigError, TrainingError
from diffcls.joint.flow import JointFlow
from diffcls.joint.model import JointDiT
from diffcls.models.dit import DiTConfig

CHECKPOINT_FILE = "checkpoint.pt"
HISTORY_FILE = "history.json"
SNAPSHOT_DIR = "snapshots"


class GpuImageFeeder:
    """Holds the training images on the device and yields shuffled, flipped batches in [-1, 1]."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int, device: torch.device) -> None:
        if len(images) < batch_size:
            raise ConfigError("need at least one full batch of images")
        self.images = images.to(device)
        self.labels = labels.to(device)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.images) // self.batch_size

    def __iter__(self):
        order = torch.randperm(len(self.images), device=self.images.device)
        for i in range(len(self)):
            pick = order[i * self.batch_size:(i + 1) * self.batch_size]
            x = to_model_range(self.images[pick])
            flip = torch.rand(len(pick), device=x.device) < 0.5
            yield torch.where(flip[:, None, None, None], x.flip(3), x), self.labels[pick]


class ExponentialMovingAverage:
    """A slowly updated copy of the weights. Evaluation uses this copy."""

    def __init__(self, model: nn.Module, decay: float = C.EMA_DECAY) -> None:
        if not 0.0 <= decay < 1.0:
            raise ConfigError("decay must be in [0, 1)")
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """ema = decay * ema + (1 - decay) * current."""
        for ema_param, param in zip(self.model.parameters(), model.parameters(), strict=True):
            ema_param.lerp_(param.detach(), 1.0 - self.decay)


@dataclass(frozen=True)
class JointSettings:
    """Run length and evaluation cadence."""

    epochs: int = C.EPOCHS
    lr: float = C.LR
    eval_every: int = C.EVAL_EVERY
    eval_images: int = C.EVAL_IMAGES
    eval_steps: int = C.EVAL_CLASSIFY_STEPS

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.eval_every < 1 or self.eval_images < 1 or self.lr <= 0:
            raise ConfigError("epochs, eval_every, eval_images and lr must be positive")


@dataclass
class JointHistory:
    """Per-epoch losses and time; quick classification accuracy at each evaluation."""

    epochs: list[int] = field(default_factory=list)
    image_loss: list[float] = field(default_factory=list)
    label_loss: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    eval_epochs: list[int] = field(default_factory=list)
    eval_accuracy: list[float] = field(default_factory=list)
    params: int = 0
    peak_vram_mib: float = 0.0

    def save(self, path: Path) -> None:
        """Write as JSON."""
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "JointHistory":
        """Read what save() wrote."""
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class JointTrainer:
    """Trains a JointDiT with JointFlow; resumes from the last checkpoint in output_dir."""

    def __init__(self, settings: JointSettings, feeder: GpuImageFeeder, test_images: torch.Tensor,
                 test_labels: torch.Tensor, output_dir: Path, device: torch.device,
                 config: DiTConfig | None = None, flow: JointFlow | None = None) -> None:
        self.settings = settings
        self.feeder = feeder
        self.test_images = test_images[:settings.eval_images]
        self.test_labels = test_labels[:settings.eval_images]
        self.output_dir = output_dir
        self.device = device
        self.config = config or DiTConfig()
        self.flow = flow or JointFlow()
        self.model = JointDiT(self.config).to(device)
        self.ema = ExponentialMovingAverage(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=settings.lr, weight_decay=C.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(1.0, (step + 1) / C.WARMUP_STEPS))
        self.history = JointHistory(params=sum(p.numel() for p in self.model.parameters()))

    @property
    def checkpoint_path(self) -> Path:
        """Where the resumable state lives."""
        return self.output_dir / CHECKPOINT_FILE

    def fit(self) -> JointHistory:
        """Train to settings.epochs. Raises TrainingError on a loss that is not finite."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        start = self._resume()
        for epoch in range(start + 1, self.settings.epochs + 1):
            began = time.perf_counter()
            image_loss, label_loss = self._run_epoch()
            self._record(epoch, image_loss, label_loss, time.perf_counter() - began)
            if epoch % self.settings.eval_every == 0 or epoch == self.settings.epochs:
                self._evaluate_and_save(epoch)
        return self.history

    def _run_epoch(self) -> tuple[float, float]:
        self.model.train()
        totals = torch.zeros(2, device=self.device)
        for x0, labels in self.feeder:
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                loss, parts = self.flow.loss(self.model, x0, labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), C.GRAD_CLIP_NORM)
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update(self.model)
            totals += torch.stack([parts["image"], parts["label"]])
        image_loss, label_loss = (totals / max(1, len(self.feeder))).tolist()
        if not (math.isfinite(image_loss) and math.isfinite(label_loss)):
            raise TrainingError(f"loss is not finite (image {image_loss}, label {label_loss}); stopping")
        return image_loss, label_loss

    def _record(self, epoch: int, image_loss: float, label_loss: float, seconds: float) -> None:
        self.history.epochs.append(epoch)
        self.history.image_loss.append(image_loss)
        self.history.label_loss.append(label_loss)
        self.history.seconds.append(seconds)
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / C.BYTES_PER_MIB
            self.history.peak_vram_mib = max(peak, self.history.peak_vram_mib)
        print(f"[joint] epoch {epoch:4d} image={image_loss:.4f} label={label_loss:.4f} ({seconds:.1f}s)", flush=True)

    def quick_accuracy(self) -> float:
        """EMA classification accuracy on the first eval_images test images."""
        generator = torch.Generator(device=self.device).manual_seed(C.DEFAULT_SEED)
        correct = 0
        for first in range(0, len(self.test_images), C.FID_BATCH):
            images = to_model_range(self.test_images[first:first + C.FID_BATCH].to(self.device))
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                predicted = self.flow.classify(self.ema.model, images, self.settings.eval_steps, generator)
            correct += int((predicted.cpu() == self.test_labels[first:first + C.FID_BATCH]).sum())
        return correct / len(self.test_images)

    def sample_grid(self) -> torch.Tensor:
        """SAMPLES_PER_CLASS images per class from fixed noise, with the EMA weights; float16 on the CPU."""
        generator = torch.Generator(device=self.device).manual_seed(C.SAMPLE_SEED)
        labels = torch.arange(self.config.num_classes, device=self.device).repeat_interleave(C.SAMPLES_PER_CLASS)
        size = self.config.image_size
        noise = torch.randn(len(labels), self.config.channels, size, size, device=self.device, generator=generator)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return self.flow.generate(self.ema.model, noise, labels, C.GENERATE_STEPS[0]).float().cpu().half()

    def _evaluate_and_save(self, epoch: int) -> None:
        accuracy = self.quick_accuracy()
        self.history.eval_epochs.append(epoch)
        self.history.eval_accuracy.append(accuracy)
        print(f"[joint] epoch {epoch:4d} quick accuracy ({self.settings.eval_steps} steps, "
              f"{len(self.test_images)} test images) = {accuracy:.4f}", flush=True)
        (self.output_dir / SNAPSHOT_DIR).mkdir(exist_ok=True)
        torch.save(self.sample_grid(), self.output_dir / SNAPSHOT_DIR / f"epoch_{epoch:04d}.pt")
        self.history.save(self.output_dir / HISTORY_FILE)
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "ema": self.ema.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                    "rng": torch.get_rng_state(), "config": asdict(self.config)}, self.checkpoint_path)

    def _resume(self) -> int:
        if not self.checkpoint_path.exists():
            return 0
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if DiTConfig(**state["config"]) != self.config:
            raise CheckpointError(f"{self.checkpoint_path.name} was trained with a different config")
        self.model.load_state_dict(state["model"])
        self.ema.model.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"].cpu())  # map_location moves it to the device
        self.history = JointHistory.load(self.output_dir / HISTORY_FILE)
        print(f"resumed from epoch {state['epoch']}", flush=True)
        return int(state["epoch"])


def load_joint_ema(output_dir: Path, device: torch.device) -> JointDiT:
    """The EMA weights of a finished (or stopped) joint run, in eval mode."""
    path = output_dir / CHECKPOINT_FILE
    if not path.exists():
        raise CheckpointError(f"no {CHECKPOINT_FILE} in {output_dir.name}/; run scripts/train_joint.py first")
    state = torch.load(path, map_location=device, weights_only=False)
    model = JointDiT(DiTConfig(**state["config"])).to(device)
    model.load_state_dict(state["ema"])
    return model.eval().requires_grad_(False)
