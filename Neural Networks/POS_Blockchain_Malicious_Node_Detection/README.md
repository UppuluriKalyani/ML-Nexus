# Blockchain Node Classification using LSTM

This project implements a Long Short-Term Memory (LSTM) neural network for classifying blockchain nodes based on various features.

## Project Structure

```
blockchain_node_classification/
├── data/               # Data directory
├── results/            # Results directory
│   ├── figures/        # Generated plots
│   └── metrics/        # Performance metrics
├── src/                # Source code
└── requirements.txt    # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Place your dataset (Blockchain101.csv) in the `data/` directory.

## Usage

Run the main script:
```bash
python src/main.py
```

## Results

The following results are generated in the `results/` directory:

1. Figures:
   - Training history plots
   - ROC curves
   - Precision-Recall curves
   - Confusion matrix

2. Metrics:
   - Classification report (precision, recall, F1-score)
   - Confusion matrix

## Model Architecture

The LSTM model consists of:
- LSTM layer with 64 units
- Dense output layer with softmax activation
- Adam optimizer with learning rate 0.001
- Categorical crossentropy loss function

Training is performed using 5-fold cross-validation.