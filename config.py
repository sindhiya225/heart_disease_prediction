# Configuration settings
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "heart_disease.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "heart_processed.csv")
SPLIT_DATA_DIR = os.path.join(DATA_DIR, "split_data")

# Model parameters - INPUT_SIZE will be updated dynamically based on data
INPUT_SIZE = 13  # Default, will be updated when data is loaded
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2  # Binary classification: heart disease or not
NUM_CLIENTS = 3

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
FED_ROUNDS = 10

# Streamlit config
STREAMLIT_PORT = 8501

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
os.makedirs(SPLIT_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "metrics"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)