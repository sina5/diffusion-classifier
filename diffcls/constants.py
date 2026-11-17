"""Central constants for this repository.

Scripts and classes read their defaults from here so that one value has one home.
"""

from pathlib import Path

# --- Paths --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
SOURCE_RUNS_DIR = REPO_ROOT.parent / "tiny-diffusion-transformer" / "runs"  # where the trained DiTs come from

# --- General ------------------------------------------------------------------
DEFAULT_SEED = 0

# --- CIFAR-10 -----------------------------------------------------------------
CIFAR10_IMAGE_SIZE = 32
CIFAR10_NUM_CLASSES = 10
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
UINT8_MAX = 255

# --- DiT (must match the trained checkpoints of tiny-diffusion-transformer) ------
PATCH_SIZE = 2
DIM = 256
DEPTH = 8
HEADS = 4
MLP_RATIO = 4.0
TIME_FREQ_DIM = 256
MAX_PERIOD = 10_000.0

# --- Noise processes -------------------------------------------------------------
TIMESTEPS = 1000  # DDPM: discrete timesteps 0..999
BETA_START = 1e-4
BETA_END = 0.02
TIME_SCALE = 1000.0  # flow matching: t in [0, 1] goes to the DiT as t * 1000

# --- Diffusion classifier ------------------------------------------------------
# Each test image is scored at evenly spaced noise levels, with one noise draw per level.
# The same noise and level are used for every candidate class (Li et al., 2023).
ERROR_LEVELS = 100
ERROR_BATCH = 100  # images per forward pass; the pass holds ERROR_BATCH * classes inputs
ERROR_SEED = 2024
LEVEL_COUNTS = (1, 2, 5, 10, 20, 50, 100)  # evenly spaced subsets for accuracy vs cost
PRECISION_CHECK_IMAGES = 500  # images scored in both bf16 and fp32 to check agreement
SELECTION_SPLIT = 5000  # test images used to choose noise levels; the rest report the result

# --- Joint image + label model (phase 2, UniDiffuser-style; Bao et al., 2023) ---------
# The label is a vector (+1 for the class, -1 elsewhere) that is noised like the image. Each
# training example picks a mode that fixes which part stays clean:
#   generate: clean label, noisy image  -> learns p(image | label)
#   classify: clean image, noisy label  -> learns p(label | image)
#   unconditional: pure-noise label     -> learns p(image), for classifier-free guidance
#   independent: both noisy at independent times -> learns the joint p(image, label)
JOINT_MODE_SHARES = {"generate": 0.4, "classify": 0.3, "unconditional": 0.1, "independent": 0.2}
JOINT_LABEL_LOSS_WEIGHT = 1.0  # weight of the label MSE (per element) against the image MSE (per element)
FLOW_LOGIT_MEAN = 0.0  # image times: logit-normal (Esser et al., 2024), as in tiny-diffusion-transformer
FLOW_LOGIT_STD = 1.0
EPOCHS = 200
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 1000
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.9999
GUIDANCE_SCALE = 2.0
EVAL_EVERY = 10  # epochs between quick evaluations and checkpoints
EVAL_IMAGES = 2000  # test images in the quick classification check
EVAL_CLASSIFY_STEPS = 4
CLASSIFY_STEPS = (1, 2, 4, 8, 16, 32)  # label ODE steps in the final evaluation
GENERATE_STEPS = (16, 64, 128)  # image ODE steps for FID
SAMPLE_SEED = 1234
SAMPLES_PER_CLASS = 8
SMOKE_EPOCHS = 2
SMOKE_SUBSET = 512
BYTES_PER_MIB = 1024 * 1024

# --- FID -------------------------------------------------------------------------
FID_SAMPLES = 10_000
FID_BATCH = 500
FID_SEED = 4321
FID_DIMS = 2048
FID_REFERENCE_FILE = "fid_reference_cifar10_train.npz"
