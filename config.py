import torch

# Dataset and causal split.
DATA_MONTH_START = 1
DATA_MONTH_END = 12
TRAIN_CUTOFF = "20181031"
VAL_START = "20181108"
VAL_CUTOFF = "20181115"
TEST_START = "20181122"
SPLIT_GAP_DAYS = 7
MAX_DISKS = 0

# Temporal Progressive Sampling.
L = 7
TEST_LEAD_TIME = 7
SEQ_LEN = 8
MAX_NEIGHBORS = 5

# Thirty vendor-normalized SMART attributes and four causal feature groups.
SMART_IDS = [5, 9, 12, 170, 171, 172, 173, 174, 175, 177,
             180, 181, 182, 183, 184, 187, 188, 190, 192, 194,
             195, 196, 197, 198, 199, 206, 232, 233, 241, 242]
RAW_FEAT_DIM = len(SMART_IDS)
INPUT_FEAT_DIM = RAW_FEAT_DIM * 5
MODEL_DIM = 48
FEAT_DIM = MODEL_DIM
DERIVED_CLIP = 10.0
BASELINE_DAYS = 30
SLOPE_DAYS = 4

# Model.
TRANSFORMER_LAYERS = 3
NUM_HEADS = 3
DROPOUT = 0.1
USE_NEIGHBORHOOD = True

# Training.
BATCH_SIZE = 64
LEARNING_RATE = 9e-5
EPOCHS = 20
WARMUP_EPOCHS = 2
POS_WEIGHT = 2.0
NEGATIVE_RATIO = 3
PATIENCE = 4
HARD_NEGATIVE_MINING = True
HARD_NEGATIVE_RATIO = 1
HARD_NEGATIVE_EPOCHS = 1
SEED = 42
TRAIN_SHARDS = 10
VAL_SHARDS = 3
TEST_SHARDS = 5
MAX_TEST_SAMPLES = -1
USE_AMP = False
MEMORY_LIMIT_GB = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Versioned artifacts prevent accidental reuse of old r_ or 30-dimensional shards.
DATASET_VERSION = "n_enhanced_v1"
SAVE_DIR = "/mnt/newdisk/qhmiao/saved_models/n_enhanced_v1"
