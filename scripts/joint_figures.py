"""Draw the joint-model figures (runs/figures/joint_*.png) from the finished joint runs.

Reads runs/<run>/eval.json, history.json and the last snapshot of each run, and the error
tables of the diffusion classifiers for comparison.

Example:
    uv run python scripts/joint_figures.py --run "200 epochs=joint" --run "400 epochs=joint_e400"
"""

import argparse
import json

import torch

from diffcls import constants as C
from diffcls.classify.error_table import ErrorTable
from diffcls.classify.processes import MethodKind
from diffcls.classify.report import ReportBuilder
from diffcls.errors import ConfigError
from diffcls.plotting import JointFigureWriter

DEFAULT_RUNS = ["200 epochs=joint"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", help='"label=run folder in runs/"; give it once per run')
    return parser.parse_args()


def load_runs(items: list[str]) -> dict[str, dict]:
    """Label to the run's evaluation, history and last sample grid."""
    runs = {}
    for item in items:
        label, sep, name = item.rpartition("=")
        run_dir = C.RUNS_DIR / name
        if not sep or not (run_dir / "eval.json").exists():
            raise ConfigError(f"need label=run with runs/<run>/eval.json, got {item!r}")
        snapshots = sorted((run_dir / "snapshots").glob("epoch_*.pt"))
        runs[label] = {"eval": json.loads((run_dir / "eval.json").read_text(encoding="utf-8")),
                       "history": json.loads((run_dir / "history.json").read_text(encoding="utf-8")),
                       "grid": torch.load(snapshots[-1])}
    return runs


def without_continued(histories: dict[str, dict]) -> dict[str, dict]:
    """Drops a run whose evaluations are the start of another run's (a run that was resumed for longer)."""
    def is_prefix(short: dict, long: dict) -> bool:
        n = len(short["eval_accuracy"])
        return n < len(long["eval_accuracy"]) and long["eval_accuracy"][:n] == short["eval_accuracy"]
    return {k: h for k, h in histories.items() if not any(is_prefix(h, o) for o in histories.values())}


def main() -> None:
    args = parse_args()
    runs = load_runs(args.run or DEFAULT_RUNS)
    results = json.loads((C.RUNS_DIR / "results.json").read_text(encoding="utf-8"))
    reports = [ReportBuilder().build(ErrorTable.load(C.RUNS_DIR / k.value / "errors.pt")) for k in MethodKind]
    writer = JointFigureWriter(C.RUNS_DIR / "figures", list(runs))
    paths = [writer.classification_cost({k: v["eval"]["classification"] for k, v in runs.items()}, reports,
                                        results["baselines"]),
             writer.training_accuracy(without_continued({k: v["history"] for k, v in runs.items()})),
             writer.sample_grids({k: v["grid"] for k, v in runs.items()})]
    for path in paths:
        print(f"wrote {path.relative_to(C.REPO_ROOT)}")


if __name__ == "__main__":
    main()
