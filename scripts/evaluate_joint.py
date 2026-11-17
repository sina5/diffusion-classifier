"""Evaluate the trained joint model: classification accuracy and FID-10k of generated images.

Classification uses all 10,000 test images at each number of label steps. FID uses 10,000
class-balanced samples against the 50,000 training images (the reference statistics are cached
in runs/). A guided image step costs two DiT calls. Writes runs/joint/eval.json.

Examples:
    uv run python scripts/evaluate_joint.py
    uv run python scripts/evaluate_joint.py --classify-steps 1 4 --generate-steps 16
    uv run python scripts/evaluate_joint.py --classify-steps --generate-steps 64 --guidance 1.5 --output eval_g1.5.json
"""

import argparse
import json
import time

import torch

from diffcls import constants as C
from diffcls.data import Cifar10Split, to_model_range
from diffcls.evaluation.fid import FidEvaluator, InceptionFeatures
from diffcls.joint.flow import JointFlow
from diffcls.joint.trainer import load_joint_ema

EVAL_FILE = "eval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classify-steps", type=int, nargs="*", default=list(C.CLASSIFY_STEPS))
    parser.add_argument("--generate-steps", type=int, nargs="*", default=list(C.GENERATE_STEPS))
    parser.add_argument("--samples", type=int, default=C.FID_SAMPLES)
    parser.add_argument("--guidance", type=float, default=C.GUIDANCE_SCALE, help="classifier-free guidance scale")
    parser.add_argument("--run-name", default="joint", help="run folder in runs/")
    parser.add_argument("--output", default=EVAL_FILE, help="file name in the run folder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class JointEvaluator:
    """Runs the classification and generation evaluations of one joint model."""

    def __init__(self, model, flow: JointFlow, device: torch.device) -> None:
        self.model = model
        self.flow = flow
        self.device = device

    def classification(self, images: torch.Tensor, labels: torch.Tensor, steps: int) -> dict:
        """Accuracy over all images with a fixed label-noise seed."""
        generator = torch.Generator(device=self.device).manual_seed(C.DEFAULT_SEED)
        start, correct = time.perf_counter(), 0
        for first in range(0, len(images), C.FID_BATCH):
            batch = to_model_range(images[first:first + C.FID_BATCH].to(self.device))
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                predicted = self.flow.classify(self.model, batch, steps, generator)
            correct += int((predicted.cpu() == labels[first:first + C.FID_BATCH]).sum())
        return {"steps": steps, "dit_calls": steps, "accuracy": correct / len(images),
                "images": len(images), "seconds": time.perf_counter() - start}

    def generation(self, fid: FidEvaluator, reference, samples: int, steps: int) -> dict:
        """FID of class-balanced samples (samples // classes per class)."""
        classes = self.model.config.num_classes
        labels = torch.arange(classes).repeat_interleave(samples // classes)
        generator = torch.Generator(device=self.device).manual_seed(C.FID_SEED)
        size, start, images = self.model.config.image_size, time.perf_counter(), []
        for batch_labels in labels.split(C.FID_BATCH):
            noise = torch.randn(len(batch_labels), 3, size, size, device=self.device, generator=generator)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                images.append(self.flow.generate(self.model, noise, batch_labels.to(self.device), steps).float().cpu())
        calls = steps * (1 if self.flow.guidance == 1.0 else 2)
        return {"steps": steps, "dit_calls": calls, "fid": fid.score(torch.cat(images), reference),
                "samples": len(labels), "seconds": time.perf_counter() - start}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dir = C.RUNS_DIR / args.run_name
    evaluator = JointEvaluator(load_joint_ema(run_dir, device), JointFlow(guidance=args.guidance), device)
    test_x, test_y = Cifar10Split(train=False).load()
    result = {"guidance": args.guidance, "classification": [], "generation": []}
    for steps in args.classify_steps:
        result["classification"].append(evaluator.classification(test_x, test_y, steps))
        print("classify", result["classification"][-1], flush=True)
    if args.generate_steps:
        fid = FidEvaluator(InceptionFeatures(device).load(), C.RUNS_DIR / C.FID_REFERENCE_FILE)
        reference = fid.reference(Cifar10Split(train=True).load()[0])
        for steps in args.generate_steps:
            result["generation"].append(evaluator.generation(fid, reference, args.samples, steps))
            print("generate", result["generation"][-1], flush=True)
    (run_dir / args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
