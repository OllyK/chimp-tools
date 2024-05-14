"""Data to be shared across files
"""

# Logging format
LOGGING_FMT = "%(asctime)s - %(levelname)s - %(message)s"
LOGGING_DATE_FMT = "%d-%b-%y %H:%M:%S"

TQDM_BAR_FORMAT = "{l_bar}{bar: 30}{r_bar}{bar: -30b}"  # tqdm progress bar format
# Settings yaml file locations
CLF_TRAIN_SETTINGS_FN = "classifier_train_settings.yaml"
DET_TRAIN_SETTINGS_FN = "detector_train_settings.yaml"
PREDICTION_DEVICE = "cpu"

BIG_CUDA_THRESHOLD = 8  # GPU Memory (GB), above this value batch size is increased
BIG_CUDA_TRAIN_BATCH = 24  # Size of training batch on big GPU
BIG_CUDA_PRED_BATCH = 24  # Size of prediction batch on big GPU
SMALL_CUDA_BATCH = 2  # Size of batch on small GPU
MPS_CPU_BATCH = 6 # Size of batch on MPS device or CPU

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_WORKERS = 8  # Number of parallel workers for training/validation dataloaders
PIN_CUDA_MEMORY = True  # Whether to pin CUDA memory for faster data transfer

DEFAULT_MIN_LR = 0.00075 # Learning rate to return if LR finder fails
LR_DIVISOR = 3 # Divide the automatically calculated learning rate (min gradient) by this magic number
