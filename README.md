# diffusion-classifier

This project uses class-conditional diffusion models as image classifiers on CIFAR-10.

A class-conditional diffusion model learns to remove noise from an image when it knows the class. If the model gets the wrong class, it removes the noise less well. A diffusion classifier uses this: it adds noise to a test image, asks the model to remove the noise once for each of the 10 classes, and predicts the class with the lowest error (Li et al., 2023, "Your Diffusion Model is Secretly a Zero-Shot Classifier"). The model was never trained to classify.

The models are the trained Diffusion Transformers (DiT, 9.8 M parameters) from the [tiny-diffusion-transformer](https://github.com/sina5/tiny-diffusion-transformer) project: one trained with DDPM (noise prediction) and one trained with flow matching (velocity prediction).

## Method

- **Error table** (`diffcls/classify/error_table.py`): the script scores every test image at 100 evenly spaced noise levels. At each level, it draws one noise sample and uses that same noise for all 10 classes, so the only difference between the classes is the label. The error is the mean squared error between the model output and its training target. The table has one value for each image, class and level (10,000 × 10 × 100).
- **Prediction** (`diffcls/classify/analysis.py`): the predicted class has the lowest mean error over the chosen levels. All accuracies come from the table, so a different choice of levels needs no new GPU pass.
- **Cost**: each level costs 10 model calls for each image (one for each class).
- **Noise levels on one axis**: DDPM and flow matching use different noise schedules. The figures use the log signal-to-noise ratio, so the two methods can be compared.
- **Level selection**: a range of levels is chosen on the first 5,000 test images and reported on the other 5,000, so the choice does not inflate the result.

## Joint image and label model

The second part trains one network that models the image and the label together (`diffcls/joint/`), in the style of UniDiffuser (Bao et al., 2023).

- **Model** (`diffcls/joint/model.py`): the same DiT as above, with one extra token for the label. The label is a vector with +1 for the class and −1 for the other classes. The image and the label each have their own noise level, and both levels condition every block.
- **Training** (`diffcls/joint/flow.py`): flow matching on the image and on the label at the same time. Each training example draws one of four modes. In 40% of the examples, the label stays clean, so the model learns to generate an image for a label. In 30%, the image stays clean, so the model learns to find the label of an image. In 10%, the label is pure noise, so the model learns to generate without a label (for classifier-free guidance). In 20%, both are noisy at independent levels.
- **Classification**: the image stays clean, and the model removes the noise from the label in a few Euler steps. Each step is one DiT call. The predicted class is the largest entry of the final label vector.
- **Generation**: the label stays clean, and the model removes the noise from the image, with classifier-free guidance of scale 2. Each guided step is two DiT calls.
- **Optimizer**: AdamW with learning rate 3e-4, a 1,000-step warmup, batch size 128, 200 epochs, bfloat16 and an EMA with decay 0.9999. These settings are the same as in tiny-diffusion-transformer, so the image quality can be compared with its flow-matching model. Training stops with an error if the loss is not finite.

## Results

All numbers are on the CIFAR-10 test set. The diffusion-classifier numbers use the 5,000 held-out test images. The joint-model numbers use all 10,000. Any choice that is fitted to data is fitted on the first 5,000 test images.

### Diffusion classifier (1,000 DiT calls per image: 100 levels × 10 classes)

| Scoring | DDPM DiT | Flow-matching DiT |
|---|---|---|
| Plain average of all levels | 82.1% | 69.2% |
| Best window of levels | 82.1% | 84.1% |
| Levels normalized by their spread between classes | 71.6% | 86.1% |
| Best window of normalized levels | 80.4% | **86.7%** |
| 100 learned level weights | **82.4%** | 86.6% |

- Only the difference in error between the classes affects the vote. For flow matching, this difference grows by about 15 times from the least noise to the most noise. A plain average therefore lets the noisiest levels decide, so flow matching scores 69.2%. For DDPM, the difference is largest at little and medium noise and almost zero at high noise, so a plain average already gives the most weight to the better levels. Normalization gives the weak, noisy levels the same weight, and DDPM falls to 71.6%. When the levels are normalized, flow matching is the better classifier. At each single level from log SNR 4.5 to the noisiest level, flow matching is also more accurate. At less noise, DDPM is more accurate.
- A single level gives at most 58% (flow matching, log SNR 1.8) and 55% (DDPM, log SNR 2.5). With less noise, the accuracy falls: the flow-matching level with the least noise (log SNR 10.6) is at chance (10.5%). The DDPM schedule stops at log SNR 7.0, where the accuracy is 35.6%.
- bfloat16 and float32 give the same prediction for 99.0% of 500 test images.
- Cat, dog and bird are the weakest classes for both models.

### Joint model

Three runs, all evaluated on the 10,000 test images. The 400-epoch run continues the 200-epoch run with the same settings. The "60% generate" run uses the mode shares 0.6 generate, 0.2 classify, 0.1 unconditional and 0.1 independent.

| Label steps (DiT calls) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 200 epochs | 85.9% | 84.6% | 83.3% | 82.8% | 82.4% | 82.3% |
| 400 epochs | **87.2%** | 87.1% | 86.6% | 86.3% | 86.1% | 86.0% |
| 200 epochs, 60% generate | 85.0% | 83.0% | 81.5% | 80.8% | 80.3% | 80.1% |

| Image steps (DiT calls), FID-10k | 16 (32) | 64 (128) | 128 (256) |
|---|---|---|---|
| 200 epochs | 51.8 | 50.9 | 53.2 |
| 400 epochs | 41.5 | 39.3 | **39.0** |
| 200 epochs, 60% generate | 40.9 | **38.6** | 39.6 |
| Flow-matching DiT from tiny-diffusion-transformer (200 epochs) | 20.2 | 14.7 | 14.0 |

| Guidance scale at 64 image steps (200 epochs) | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|
| FID-10k | 57.1 | 53.3 | 50.9 | 50.9 |

- With one DiT call, the 400-epoch joint model gets 87.2%. The best diffusion classifier gets 86.7% with 1,000 calls (on the 5,000 held-out images). In all runs, more label steps make the accuracy lower.
- Training for 400 epochs improves both tasks: accuracy from 85.9% to 87.2%, and FID at 64 steps from 50.9 to 39.3. The quick accuracy (2,000 test images, 4 label steps) increases slowly after epoch 250, from 84.6% to 86.6%.
- More generation examples (60% instead of 40%) give the same FID improvement as 200 more epochs, at the cost of 0.9 points of accuracy.
- All joint models make much worse images than the flow-matching DiT. A different guidance scale does not correct this.
- For comparison, trained classifiers get 93.0% (ViT from scratch) and 92.6% (ResNet-18).
- Each 200 epochs took about 88 minutes on one GPU, with a peak memory of 3.6 GiB.

Figures: `scripts/joint_figures.py --run "200 epochs=joint" --run "400 epochs=joint_e400" --run "200 epochs, 60% generate=joint_gen60"` writes `runs/figures/joint_*.png`.

## Setup

You need [uv](https://docs.astral.sh/uv/) and an NVIDIA driver that supports CUDA 12.8. The trained checkpoints come from a local clone of tiny-diffusion-transformer next to this folder.

```bash
uv sync --group dev
uv run pytest
uv run python scripts/import_checkpoints.py      # copies the EMA weights to checkpoints/
```

## Run

```bash
uv run python scripts/compute_errors.py --method ddpm
uv run python scripts/compute_errors.py --method flow_matching
uv run python scripts/compute_errors.py --method ddpm --precision-check   # bf16 vs fp32 on 500 images

# Accuracies and figures from the error tables (runs/results.json, runs/figures/)
uv run python scripts/analyze.py --baseline "ViT from scratch=0.9302" --baseline "ResNet-18=0.9261"

# Joint model: train (runs/joint/, resumes if stopped), then evaluate (runs/joint/eval.json)
uv run python scripts/train_joint.py --smoke
uv run python scripts/train_joint.py
uv run python scripts/evaluate_joint.py
uv run python scripts/evaluate_joint.py --classify-steps --generate-steps 64 --guidance 1.5 --output eval_g1.5.json

# Longer training: copy runs/joint/ to runs/joint_e400/, then resume it to 400 epochs
uv run python scripts/train_joint.py --run-name joint_e400 --epochs 400
uv run python scripts/evaluate_joint.py --run-name joint_e400

# More generation examples
uv run python scripts/train_joint.py --run-name joint_gen60 --mode-shares 0.6 0.2 0.1 0.1
uv run python scripts/evaluate_joint.py --run-name joint_gen60
```

The baselines are the test accuracies of two classifiers trained on CIFAR-10: the "patch 4, 2x epochs" ViT from the vit-cifar10-from-scratch project and the ResNet-18 judge from tiny-diffusion-transformer.

Git ignores `data/`, `checkpoints/` and `runs/`.

## License

The code is licensed under the [Apache License 2.0](LICENSE).
