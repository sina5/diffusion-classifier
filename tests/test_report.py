import importlib.util
from pathlib import Path

import pytest
import torch

from diffcls import constants as C
from diffcls.classify.error_table import ErrorTable
from diffcls.classify.processes import MethodKind
from diffcls.classify.report import ReportBuilder
from diffcls.errors import ConfigError
from diffcls.plotting import FigureWriter, JointFigureWriter

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze.py"


def _analyze_module():
    spec = importlib.util.spec_from_file_location("analyze", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def table() -> ErrorTable:
    torch.manual_seed(0)
    labels = torch.arange(40) % C.CIFAR10_NUM_CLASSES
    errors = torch.rand(40, C.CIFAR10_NUM_CLASSES, 6)
    errors[torch.arange(40), labels, 3] -= 2.0  # level 3 points to the true class
    return ErrorTable(MethodKind.DDPM, errors, labels, torch.linspace(0.1, 0.9, 6), torch.linspace(5, -5, 6))


def test_report_covers_every_image_and_chooses_the_informative_level(table: ErrorTable) -> None:
    report = ReportBuilder(counts=(1, 2, 6, 50), selection_split=20).build(table)
    assert report.raw.window_levels == (3, 4) and report.raw.accuracy_window == 1.0
    assert report.normalized.accuracy_window == 1.0 and report.held_out_accuracy_learned_weights == 1.0
    assert set(report.raw.accuracy_vs_level_count) == {1, 2, 6} and sum(map(sum, report.confusion)) == 20
    assert report.held_out_images == 20 and len(report.per_class_accuracy) == C.CIFAR10_NUM_CLASSES


def test_figures_are_written(table: ErrorTable, tmp_path: Path) -> None:
    report = ReportBuilder(counts=(1, 6), selection_split=20).build(table)
    writer = FigureWriter(tmp_path, {"ViT": 0.93})
    paths = [writer.accuracy_vs_cost([report]), writer.accuracy_per_level([report]), writer.confusion([report])]
    assert all(p.exists() and p.suffix == ".png" for p in paths)


def test_baselines_parse_names_with_equals_signs() -> None:
    assert _analyze_module().parse_baselines(["a=b=0.5"]) == {"a=b": 0.5}


@pytest.mark.parametrize("item", ["no-value", "x=1.5"])
def test_baselines_reject_bad_items(item: str) -> None:
    with pytest.raises(ConfigError):
        _analyze_module().parse_baselines([item])


JOINT_SCRIPT = SCRIPT.parent / "joint_figures.py"


def _joint_module():
    spec = importlib.util.spec_from_file_location("joint_figures", JOINT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_figures_are_written(table: ErrorTable, tmp_path: Path) -> None:
    report = ReportBuilder(counts=(1, 6), selection_split=20).build(table)
    rows = [{"dit_calls": 1, "accuracy": 0.8}, {"dit_calls": 4, "accuracy": 0.7}]
    history = {"eval_epochs": [10, 20], "eval_accuracy": [0.2, 0.5]}
    writer = JointFigureWriter(tmp_path, ["a", "b"])
    paths = [writer.classification_cost({"a": rows, "b": rows}, [report], {"ViT": 0.93}),
             writer.training_accuracy({"b": history}),
             writer.sample_grids({"a": torch.zeros(C.CIFAR10_NUM_CLASSES * 2, 3, 8, 8)})]
    assert all(p.exists() and p.suffix == ".png" for p in paths)


def test_a_resumed_run_hides_its_shorter_start() -> None:
    short, long = {"eval_accuracy": [0.1, 0.2]}, {"eval_accuracy": [0.1, 0.2, 0.3]}
    other = {"eval_accuracy": [0.1, 0.4]}
    kept = _joint_module().without_continued({"short": short, "long": long, "other": other})
    assert set(kept) == {"long", "other"}
