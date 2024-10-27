import cv2
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from .config import *

def load_data():
    """Load and preprocess image data."""
    data = []
    labels = []
    
    # Load galaxy images
    galaxy_paths = glob.glob(os.path.join(DATA_DIR, 'galaxy', '*'))
    for path in galaxy_paths:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(0)  # 0 for galaxy
    
    # Load star images
    star_paths = glob.glob(os.path.join(DATA_DIR, 'star', '*'))
    for path in star_paths:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(1)  # 1 for star
    
    data = np.array(data) / 255.0  # Normalize
    labels = np.array(labels)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, 
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    
    return x_train, x_test, y_train, y_test