import os
import torch


# Environment parameters
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_FOLDER = PROJECT_ROOT + '/results'
STATS_FOLDER = RESULT_FOLDER + '/stats'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Parameters for data loading
DATA_FOLDER = PROJECT_ROOT + '/data'
CIFAR10_NUM_TRN_SAMPLES = 50000
NUM_WORKERS = 4

# Network parameters
INPUT_SHAPE = (3, 32, 32) # Channels, height and width
NUM_CLASSES = 10

# Experiment configuration parameters
CONFIG_FAMILY_HEBB = 'hebb' # Key to identify configurations based on hebbian learning
CONFIG_FAMILY_GDES = 'gdes' # Key to identify configurations based on gradient descent learning
DEFAULT_CONFIG = 'gdes/config_base'

