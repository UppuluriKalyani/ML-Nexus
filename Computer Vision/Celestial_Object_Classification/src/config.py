
import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model parameters
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 25
RANDOM_STATE = 42
TEST_SIZE = 0.2
