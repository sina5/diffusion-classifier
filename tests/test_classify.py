from pathlib import Path

import pytest
import torch

from diffcls.checkpoints import CheckpointStore
from diffcls.classify.analysis import (
    LevelNormalizer, LevelWeightFitter, LevelWindowSelector, TableClassifier, evenly_spaced,
)
from diffcls.classify.error_table import ErrorTable, ErrorTableBuilder
from diffcls.classify.processes import DdpmProcess, FlowMatchingProcess, MethodKind, NoiseProcess
from diffcls.errors import CheckpointError, ConfigError
from diffcls.models.dit import DiT, DiTConfig

CPU = torch.device("cpu")
TINY = DiTConfig(image_size=8, patch_size=2, dim=16, depth=1, heads=2, num_classes=3, time_freq_dim=8)


def _table(errors: torch.Tensor, labels: list[int]) -> ErrorTable:
    k = errors.shape[2]
    return ErrorTable(MethodKind.DDPM, errors.float(), torch.tensor(labels), torch.zeros(k), torch.zeros(k))


class OracleDenoiser(torch.nn.Module):
    """Knows the clean image of each class, so it predicts the exact target for the true class only."""

    def __init__(self, class_images: torch.Tensor, process: NoiseProcess, level: float) -> None:
        super().__init__()
        self.class_images = class_images
        self.process = process
        self.level = level
        self.anchor = torch.nn.Parameter(torch.zeros(1))  # gives the builder a device

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x0 = self.class_images[y]
        if isinstance(self.process, FlowMatchingProcess):
            return (x - x0) / self.level  # velocity noise - x0 on the straight path
        a = float(self.process.alpha_bar[self.process.timestep(self.level)])
        return (x - a ** 0.5 * x0) / (1 - a) ** 0.5  # the noise


# --- noise processes ---------------------------------------------------------------


def test_levels_are_bin_centers() -> None:
    assert torch.allclose(NoiseProcess.levels(4), torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=torch.float64))


def test_levels_reject_zero() -> None:
    with pytest.raises(ConfigError):
        NoiseProcess.levels(0)


@pytest.mark.parametrize("process", [DdpmProcess(), FlowMatchingProcess()])
def test_log_snr_falls_as_the_level_rises(process: NoiseProcess) -> None:
    values = [process.log_snr(float(level)) for level in NoiseProcess.levels(10)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_ddpm_time_input_is_the_discrete_timestep() -> None:
    assert DdpmProcess().time_input(0.9995) == 999.0 and DdpmProcess().time_input(0.0005) == 0.0


def test_flow_matching_target_is_the_velocity() -> None:
    x0, noise = torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)
    x_t, target = FlowMatchingProcess().noisy(x0, 0.25, noise)
    assert torch.allclose(x_t, 0.75 * x0) and torch.allclose(target, -x0)


# --- error table -------------------------------------------------------------------


@pytest.mark.parametrize("process", [DdpmProcess(), FlowMatchingProcess()])
def test_builder_with_an_oracle_classifies_every_image(process: NoiseProcess) -> None:
    class_images = torch.randint(0, 256, (3, 3, 8, 8), dtype=torch.uint8)  # one clean image per class
    labels = torch.tensor([2, 0, 1, 2])
    level = float(NoiseProcess.levels(1)[0])
    oracle = OracleDenoiser(class_images.float() / 127.5 - 1, process, level)
    table = ErrorTableBuilder(oracle, process, num_classes=3, levels=1, batch_size=3, verbose=False).build(
        class_images[labels], labels)
    assert table.errors.shape == (4, 3, 1) and TableClassifier(table).accuracy() == 1.0


def test_builder_noise_does_not_depend_on_the_batch_size() -> None:
    model, images = DiT(TINY), torch.randint(0, 256, (6, 3, 8, 8), dtype=torch.uint8)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    tables = [ErrorTableBuilder(model, DdpmProcess(), 3, levels=2, batch_size=size, verbose=False)
              .build(images, torch.zeros(6, dtype=torch.long)) for size in (6, 4)]
    assert torch.allclose(tables[0].errors, tables[1].errors, rtol=1e-5)


def test_table_round_trips(tmp_path: Path) -> None:
    table = _table(torch.rand(2, 3, 4), [0, 1])
    table.save(tmp_path / "t.pt")
    loaded = ErrorTable.load(tmp_path / "t.pt")
    assert torch.equal(loaded.errors, table.errors) and loaded.method is MethodKind.DDPM


def test_table_rejects_inconsistent_sizes() -> None:
    with pytest.raises(ConfigError):
        ErrorTable(MethodKind.DDPM, torch.rand(2, 3, 4), torch.zeros(3), torch.zeros(4), torch.zeros(4))


def test_missing_table_has_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        ErrorTable.load(tmp_path / "none.pt")


# --- analysis ----------------------------------------------------------------------


def test_prediction_is_the_lowest_mean_error() -> None:
    errors = torch.tensor([[[1.0, 1.0], [0.5, 0.5], [2.0, 2.0]], [[0.1, 0.9], [0.6, 0.6], [0.7, 0.7]]])
    classifier = TableClassifier(_table(errors, [1, 0]))
    assert classifier.predict().tolist() == [1, 0] and classifier.accuracy() == 1.0


def test_accuracy_per_level_uses_one_level_at_a_time() -> None:
    errors = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])  # level 0 says class 0, level 1 says class 1
    assert TableClassifier(_table(errors, [0])).accuracy_per_level().tolist() == [1.0, 0.0]


def test_evenly_spaced_indices() -> None:
    assert evenly_spaced(100, 2).tolist() == [25, 75] and evenly_spaced(100, 100).tolist() == list(range(100))


@pytest.mark.parametrize("count", [0, 5])
def test_evenly_spaced_rejects_bad_counts(count: int) -> None:
    with pytest.raises(ConfigError):
        evenly_spaced(4, count)


def test_confusion_counts_every_image_once() -> None:
    confusion = TableClassifier(_table(torch.rand(20, 3, 2), [0, 1, 2, 0] * 5)).confusion()
    assert int(confusion.sum()) == 20 and confusion.sum(dim=1).tolist() == [10, 5, 5]


def test_window_selector_finds_the_informative_levels() -> None:
    errors = torch.rand(30, 3, 6)
    labels = torch.arange(30) % 3
    errors[torch.arange(30), labels, 2:4] = -1.0  # only levels 2 and 3 point to the true class
    assert LevelWindowSelector().select(_table(errors, labels.tolist())).tolist() == [2]


# --- checkpoints -------------------------------------------------------------------


def test_store_imports_the_ema_and_loads_it(tmp_path: Path) -> None:
    model = DiT(TINY)
    (tmp_path / "runs").mkdir()
    state = {"epoch": 7, "config": TINY.__dict__, "ema": model.state_dict(), "model": {}}
    torch.save(state, tmp_path / "runs/checkpoint.pt")
    store = CheckpointStore(tmp_path / "checkpoints")
    store.import_run(MethodKind.DDPM, tmp_path / "runs")
    loaded = store.load(MethodKind.DDPM, CPU)
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), model.state_dict().values()))


def test_store_rejects_a_run_of_another_method(tmp_path: Path) -> None:
    (tmp_path / "flow_matching").mkdir()
    torch.save({"epoch": 1, "config": TINY.__dict__, "ema": {}, "method": "meanflow"},
               tmp_path / "flow_matching/checkpoint.pt")
    with pytest.raises(CheckpointError):
        CheckpointStore(tmp_path / "c").import_run(MethodKind.FLOW_MATCHING, tmp_path)


def test_store_reports_a_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        CheckpointStore(tmp_path).load(MethodKind.DDPM, CPU)


def test_normalizer_stops_a_loud_level_from_deciding() -> None:
    """Normalizing gives each level the same scale: the loud random level no longer decides alone."""
    torch.manual_seed(0)
    labels = torch.arange(30) % 3
    errors = torch.rand(30, 3, 2) * 0.01
    errors[torch.arange(30), labels, 0] -= 0.1  # level 0: small errors, points to the true class
    errors[:, :, 1] = torch.rand(30, 3) * 100  # level 1: huge, random errors
    table = _table(errors, labels.tolist())
    normalized = LevelNormalizer().fit(table).apply(table)
    assert TableClassifier(normalized).accuracy() > TableClassifier(table).accuracy() + 0.2


def test_normalizer_needs_fit() -> None:
    with pytest.raises(ConfigError):
        LevelNormalizer().apply(_table(torch.rand(2, 3, 2), [0, 1]))


def test_weight_fitter_learns_to_ignore_a_misleading_level() -> None:
    labels = torch.arange(60) % 3
    errors = torch.rand(60, 3, 2)
    errors[torch.arange(60), labels, 0] -= 1.0  # level 0 is informative
    errors[torch.arange(60), (labels + 1) % 3, 1] -= 1.0  # level 1 points to a wrong class
    table = _table(errors, labels.tolist())
    assert LevelWeightFitter().fit(table).accuracy(table) == 1.0


def test_weight_fitter_needs_fit() -> None:
    with pytest.raises(ConfigError):
        LevelWeightFitter().accuracy(_table(torch.rand(2, 3, 2), [0, 1]))
