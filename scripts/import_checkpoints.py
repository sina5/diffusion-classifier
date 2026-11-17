"""Copy the EMA weights of the trained DiTs from tiny-diffusion-transformer to checkpoints/.

Example:
    uv run python scripts/import_checkpoints.py
    uv run python scripts/import_checkpoints.py --source ../tiny-diffusion-transformer/runs
"""

import argparse
from pathlib import Path

from diffcls import constants as C
from diffcls.checkpoints import CheckpointStore
from diffcls.classify.processes import MethodKind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=C.SOURCE_RUNS_DIR, help="runs/ folder of the trained models")
    parser.add_argument("--method", type=MethodKind, choices=list(MethodKind), nargs="+", default=list(MethodKind))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = CheckpointStore()
    for kind in args.method:
        path = store.import_run(kind, args.source)
        print(f"wrote {path.relative_to(C.REPO_ROOT)}")


if __name__ == "__main__":
    main()
