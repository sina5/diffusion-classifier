"""Score every CIFAR-10 test image with every class at evenly spaced noise levels.

Writes runs/<method>/errors.pt (the error table). With --precision-check, it also scores a few
images in fp32 and reports how often the bf16 and fp32 predictions agree.

Examples:
    uv run python scripts/compute_errors.py --method ddpm
    uv run python scripts/compute_errors.py --method flow_matching
    uv run python scripts/compute_errors.py --method ddpm --images 200 --levels 10   # quick check
"""

import argparse
import json
import time

import torch

from diffcls import constants as C
from diffcls.checkpoints import CheckpointStore
from diffcls.classify.analysis import TableClassifier
from diffcls.classify.error_table import ErrorTableBuilder
from diffcls.classify.processes import MethodKind, build_process
from diffcls.data import Cifar10Split

TABLE_FILE = "errors.pt"
PRECISION_FILE = "precision_check.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", type=MethodKind, choices=list(MethodKind), required=True)
    parser.add_argument("--images", type=int, default=None, help="score only the first N test images")
    parser.add_argument("--levels", type=int, default=C.ERROR_LEVELS)
    parser.add_argument("--precision-check", action="store_true", help="compare bf16 with fp32 on a few images")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False  # the fp32 reference must be real fp32
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model = CheckpointStore().load(args.method, device)
    images, labels = Cifar10Split(train=False).load()
    if args.images is not None:
        images, labels = images[:args.images], labels[:args.images]
    output_dir = C.RUNS_DIR / args.method.value
    if args.precision_check:
        check_precision(model, args.method, images, labels, args.levels, output_dir)
        return
    start = time.perf_counter()
    table = ErrorTableBuilder(model, build_process(args.method), levels=args.levels).build(images, labels)
    table.save(output_dir / TABLE_FILE)
    print(f"accuracy (all {args.levels} levels) {TableClassifier(table).accuracy():.4f}, "
          f"{time.perf_counter() - start:.0f}s, wrote runs/{args.method.value}/{TABLE_FILE}")


def check_precision(model, kind: MethodKind, images: torch.Tensor, labels: torch.Tensor, levels: int,
                    output_dir) -> None:
    """Score the same images in bf16 and in fp32, and write how much the results differ."""
    images, labels = images[:C.PRECISION_CHECK_IMAGES], labels[:C.PRECISION_CHECK_IMAGES]
    tables = {name: ErrorTableBuilder(model, build_process(kind), levels=levels, bf16=bf16, verbose=False)
              .build(images, labels) for name, bf16 in (("bf16", True), ("fp32", False))}
    predictions = {name: TableClassifier(t).predict() for name, t in tables.items()}
    relative = ((tables["bf16"].errors - tables["fp32"].errors).abs() / tables["fp32"].errors).median()
    row = {"images": len(images), "levels": levels,
           "prediction_agreement": float((predictions["bf16"] == predictions["fp32"]).float().mean()),
           "accuracy_bf16": TableClassifier(tables["bf16"]).accuracy(),
           "accuracy_fp32": TableClassifier(tables["fp32"]).accuracy(),
           "median_relative_error_difference": float(relative)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PRECISION_FILE).write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(row)


if __name__ == "__main__":
    main()
