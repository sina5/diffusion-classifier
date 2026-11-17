"""Train the joint image + label model on CIFAR-10 (writes runs/joint/, resumes if stopped).

Examples:
    uv run python scripts/train_joint.py
    uv run python scripts/train_joint.py --smoke     # 2 epochs on 512 images, to runs/joint_smoke/
    uv run python scripts/train_joint.py --run-name joint_gen60 --mode-shares 0.6 0.2 0.1 0.1
"""

import argparse
import json

import torch

from diffcls import constants as C
from diffcls.data import Cifar10Split
from diffcls.errors import ConfigError
from diffcls.joint.flow import JointFlow, TrainingMode
from diffcls.joint.trainer import GpuImageFeeder, JointSettings, JointTrainer

SETTINGS_FILE = "settings.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--run-name", default="joint", help="output folder in runs/")
    parser.add_argument("--mode-shares", type=float, nargs=4, metavar=("GENERATE", "CLASSIFY", "UNCOND", "INDEPENDENT"),
                        help="training mode shares (default: constants.JOINT_MODE_SHARES)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def record_shares(run_dir, shares: dict[str, float]) -> None:
    """Writes the mode shares to run_dir/settings.json; a resumed run must use the same shares."""
    path = run_dir / SETTINGS_FILE
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["mode_shares"] != shares:
        raise ConfigError(f"{run_dir.name}/ was trained with other mode shares; use a new --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode_shares": shares}, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(C.DEFAULT_SEED)
    device = torch.device(args.device)
    train_x, train_y = Cifar10Split(train=True).load()
    test_x, test_y = Cifar10Split(train=False).load()
    if args.smoke:
        train_x, train_y = train_x[:C.SMOKE_SUBSET], train_y[:C.SMOKE_SUBSET]
        settings = JointSettings(epochs=C.SMOKE_EPOCHS, eval_every=1, eval_images=200)
    else:
        settings = JointSettings(epochs=args.epochs)
    shares = (dict(zip((m.value for m in TrainingMode), args.mode_shares)) if args.mode_shares
              else C.JOINT_MODE_SHARES)
    run_dir = C.RUNS_DIR / ("joint_smoke" if args.smoke else args.run_name)
    record_shares(run_dir, shares)
    trainer = JointTrainer(settings, GpuImageFeeder(train_x, train_y, C.BATCH_SIZE, device), test_x, test_y,
                           run_dir, device,
                           flow=JointFlow(shares=shares))
    history = trainer.fit()
    print(f"done: {sum(history.seconds) / 60:.1f} min, peak VRAM {history.peak_vram_mib:.0f} MiB, "
          f"{history.params:,} parameters")


if __name__ == "__main__":
    main()
