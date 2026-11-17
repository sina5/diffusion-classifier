from pathlib import Path

import pytest
import torch

from diffcls.errors import ConfigError, TrainingError
from diffcls.joint.flow import JointFlow, LabelCodec, TrainingMode
from diffcls.joint.model import JointDiT
from diffcls.joint.trainer import GpuImageFeeder, JointSettings, JointTrainer, load_joint_ema
from diffcls.models.dit import DiTConfig

CPU = torch.device("cpu")
TINY = DiTConfig(image_size=8, patch_size=2, dim=16, depth=1, heads=2, num_classes=3, time_freq_dim=8)
EVEN = {m.value: 0.25 for m in TrainingMode}


def _flow(shares: dict[str, float] = EVEN, guidance: float = 2.0) -> JointFlow:
    return JointFlow(shares=shares, guidance=guidance, codec=LabelCodec(3))


def _only(mode: TrainingMode) -> dict[str, float]:
    return {m.value: float(m is mode) for m in TrainingMode}


# --- model -------------------------------------------------------------------------


def test_model_returns_an_image_and_a_label_and_starts_at_zero() -> None:
    x, y = torch.randn(2, 3, 8, 8), torch.randn(2, 3)
    v_x, v_y = JointDiT(TINY)(x, y, torch.zeros(2), torch.ones(2))
    assert v_x.shape == x.shape and v_y.shape == y.shape
    assert torch.count_nonzero(v_x) == 0 and torch.count_nonzero(v_y) == 0  # zero-initialized outputs


def test_label_changes_the_image_prediction_after_training_starts() -> None:
    model, x = JointDiT(TINY), torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    a, _ = model(x, torch.tensor([[1.0, -1, -1]]), torch.zeros(1), torch.zeros(1))
    b, _ = model(x, torch.tensor([[-1.0, 1, -1]]), torch.zeros(1), torch.zeros(1))
    assert not torch.allclose(a, b)  # the label token reaches the image through attention


# --- codec and flow ----------------------------------------------------------------


def test_codec_round_trips() -> None:
    codec, labels = LabelCodec(3), torch.tensor([2, 0, 1])
    vectors = codec.encode(labels)
    assert set(vectors.unique().tolist()) == {-1.0, 1.0} and torch.equal(codec.decode(vectors), labels)


@pytest.mark.parametrize(("mode", "check"), [
    (TrainingMode.GENERATE, lambda t_x, t_y: torch.all(t_y == 0) and torch.all(t_x > 0)),
    (TrainingMode.CLASSIFY, lambda t_x, t_y: torch.all(t_x == 0) and torch.all(t_y >= 0)),
    (TrainingMode.UNCONDITIONAL, lambda t_x, t_y: torch.all(t_y == 1)),
    (TrainingMode.INDEPENDENT, lambda t_x, t_y: torch.all((t_x > 0) & (t_x < 1) & (t_y < 1))),
])
def test_each_mode_sets_the_expected_times(mode: TrainingMode, check) -> None:
    t_x, t_y = _flow(_only(mode)).sample_times(500, CPU)
    assert check(t_x, t_y)


def test_mode_shares_are_respected() -> None:
    shares = {"generate": 0.5, "classify": 0.5, "unconditional": 0.0, "independent": 0.0}
    t_x, _ = _flow(shares).sample_times(20_000, CPU)
    assert (t_x == 0).float().mean().item() == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("shares", [{"generate": 1.0}, {**EVEN, "generate": 0.5}])
def test_flow_rejects_bad_shares(shares: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        _flow(shares)


class TrueVelocity(torch.nn.Module):
    """Predicts the exact velocity toward known clean images and labels (straight paths)."""

    def __init__(self, x0: torch.Tensor, y0: torch.Tensor) -> None:
        super().__init__()
        self.x0, self.y0 = x0, y0

    def forward(self, x, y, t_x, t_y):
        tx, ty = (t_x / 1000).clamp_min(1e-6).view(-1, 1, 1, 1), (t_y / 1000).clamp_min(1e-6).view(-1, 1)
        n = len(x) // len(self.x0)
        return (x - self.x0.repeat(n, 1, 1, 1)) / tx, (y - self.y0.repeat(n, 1)) / ty


def test_classify_with_the_true_velocity_recovers_the_label() -> None:
    codec, labels = LabelCodec(3), torch.tensor([2, 0, 1, 1])
    model = TrueVelocity(torch.zeros(4, 3, 8, 8), codec.encode(labels))
    assert torch.equal(_flow().classify(model, torch.zeros(4, 3, 8, 8), steps=1), labels)


def test_generate_with_the_true_velocity_recovers_the_image() -> None:
    x0 = torch.rand(2, 3, 8, 8) * 2 - 1
    model = TrueVelocity(x0, LabelCodec(3).encode(torch.tensor([0, 1])))
    images = _flow(guidance=2.0).generate(model, torch.randn(2, 3, 8, 8), torch.tensor([0, 1]), steps=3)
    assert torch.allclose(images, x0, atol=1e-4)  # cond and uncond agree here, so guidance changes nothing


@pytest.mark.parametrize("method", ["classify", "generate"])
def test_samplers_reject_zero_steps(method: str) -> None:
    flow, x = _flow(), torch.zeros(1, 3, 8, 8)
    with pytest.raises(ConfigError):
        flow.classify(JointDiT(TINY), x, 0) if method == "classify" else flow.generate(JointDiT(TINY), x,
                                                                                      torch.zeros(1).long(), 0)


def test_loss_has_two_finite_parts_and_a_gradient() -> None:
    model = JointDiT(TINY)
    loss, parts = _flow().loss(model, torch.rand(4, 3, 8, 8) * 2 - 1, torch.tensor([0, 1, 2, 0]))
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"image", "label"} and model.label_out.weight.grad is not None


# --- trainer -----------------------------------------------------------------------


def _trainer(tmp_path: Path, epochs: int = 2, device: torch.device = CPU) -> JointTrainer:
    images, labels = torch.randint(0, 256, (16, 3, 8, 8), dtype=torch.uint8), torch.arange(16) % 3
    return JointTrainer(JointSettings(epochs=epochs, eval_every=1, eval_images=4, eval_steps=2),
                        GpuImageFeeder(images, labels, 8, device), images, labels, tmp_path, device, TINY,
                        _flow())


DEVICES = [CPU] + ([torch.device("cuda")] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: d.type)
def test_trainer_trains_saves_and_resumes(tmp_path: Path, device: torch.device) -> None:
    history = _trainer(tmp_path, epochs=1, device=device).fit()
    resumed = _trainer(tmp_path, epochs=2, device=device).fit()  # on CUDA, the saved RNG state must go back to the CPU
    assert history.epochs == [1] and resumed.epochs == [1, 2] and len(resumed.eval_accuracy) == 2
    assert load_joint_ema(tmp_path, device).config == TINY


def test_trainer_stops_on_a_nan_loss(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, epochs=1)
    with torch.no_grad():
        trainer.model.image_out.bias.fill_(float("nan"))
    with pytest.raises(TrainingError):
        trainer.fit()


def test_feeder_needs_a_full_batch() -> None:
    with pytest.raises(ConfigError):
        GpuImageFeeder(torch.zeros(3, 3, 8, 8, dtype=torch.uint8), torch.zeros(3).long(), 4, CPU)
