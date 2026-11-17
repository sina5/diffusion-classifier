"""Import the trained DiTs from tiny-diffusion-transformer and load them for scoring.

The import keeps only the EMA weights and the config, because sampling and scoring use the EMA.
"""

from pathlib import Path

import torch

from diffcls import constants as C
from diffcls.classify.processes import MethodKind
from diffcls.errors import CheckpointError
from diffcls.models.dit import DiT, DiTConfig

SOURCE_FILE = "checkpoint.pt"


class CheckpointStore:
    """checkpoints/<method>.pt files: {"method", "config", "ema", "source_epoch"}."""

    def __init__(self, folder: Path = C.CHECKPOINT_DIR) -> None:
        self.folder = folder

    def path(self, kind: MethodKind) -> Path:
        """Where the checkpoint of a method lives."""
        return self.folder / f"{kind.value}.pt"

    def import_run(self, kind: MethodKind, source_runs: Path = C.SOURCE_RUNS_DIR) -> Path:
        """Copy the EMA weights of a finished run. DDPM is at the top of runs/, others in runs/<method>/."""
        source = source_runs / SOURCE_FILE if kind is MethodKind.DDPM else source_runs / kind.value / SOURCE_FILE
        if not source.exists():
            raise CheckpointError(f"no trained {kind.value} run found ({source.name} in {source.parent.name}/)")
        state = torch.load(source, map_location="cpu", weights_only=False)
        if state.get("method", MethodKind.DDPM.value) != kind.value:
            raise CheckpointError(f"{source.parent.name}/{source.name} holds a {state['method']} run")
        self.folder.mkdir(parents=True, exist_ok=True)
        target = self.path(kind)
        torch.save({"method": kind.value, "config": state["config"], "ema": state["ema"],
                    "source_epoch": state["epoch"]}, target)
        return target

    def load(self, kind: MethodKind, device: torch.device) -> DiT:
        """The EMA DiT of a method, in eval mode."""
        path = self.path(kind)
        if not path.exists():
            raise CheckpointError(f"no {path.name}; run scripts/import_checkpoints.py first")
        state = torch.load(path, map_location=device, weights_only=True)
        model = DiT(DiTConfig(**state["config"])).to(device)
        model.load_state_dict(state["ema"])
        return model.eval().requires_grad_(False)
