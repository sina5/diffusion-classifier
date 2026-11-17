"""Compute accuracies from the error tables and draw the figures.

Reads runs/<method>/errors.pt for each method that has one, and writes runs/results.json and
the figures in runs/figures/. Baselines are trained classifiers to compare with; give their
measured test accuracy.

Example:
    uv run python scripts/analyze.py --baseline "ViT from scratch=0.9302" --baseline "ResNet-18=0.9261"
"""

import argparse
import json

from diffcls import constants as C
from diffcls.classify.error_table import ErrorTable
from diffcls.classify.processes import MethodKind
from diffcls.classify.report import ReportBuilder
from diffcls.errors import ConfigError
from diffcls.plotting import FigureWriter

TABLE_FILE = "errors.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", action="append", default=[], help='"name=accuracy", accuracy in [0, 1]')
    return parser.parse_args()


def parse_baselines(items: list[str]) -> dict[str, float]:
    """'name=0.93' strings to a dict."""
    baselines = {}
    for item in items:
        name, sep, value = item.rpartition("=")
        if not sep or not 0.0 <= float(value) <= 1.0:
            raise ConfigError(f"baseline must look like name=0.93, got {item!r}")
        baselines[name] = float(value)
    return baselines


def main() -> None:
    args = parse_args()
    tables = [ErrorTable.load(C.RUNS_DIR / k.value / TABLE_FILE) for k in MethodKind
              if (C.RUNS_DIR / k.value / TABLE_FILE).exists()]
    if not tables:
        raise ConfigError("no error tables in runs/; run scripts/compute_errors.py first")
    reports = [ReportBuilder().build(table) for table in tables]
    output = {"baselines": parse_baselines(args.baseline), "methods": [r.to_dict() for r in reports]}
    (C.RUNS_DIR / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    writer = FigureWriter(C.RUNS_DIR / "figures", output["baselines"])
    for path in (writer.accuracy_vs_cost(reports), writer.accuracy_per_level(reports), writer.confusion(reports)):
        print(f"wrote {path.relative_to(C.REPO_ROOT)}")
    for r in reports:
        print(f"{r.method}: all test images, plain average {r.accuracy_all_levels:.4f}; held-out "
              f"{r.held_out_images}: plain {r.raw.accuracy_all_levels:.4f}, plain window {r.raw.accuracy_window:.4f}, "
              f"normalized {r.normalized.accuracy_all_levels:.4f}, normalized window {r.normalized.accuracy_window:.4f}, "
              f"learned weights {r.held_out_accuracy_learned_weights:.4f}")

if __name__ == "__main__":
    main()
