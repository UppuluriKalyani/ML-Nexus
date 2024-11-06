# Cotton Disease Prediction with Transfer Learning

This project utilizes deep learning and transfer learning techniques to classify cotton leaf images as either healthy or diseased. By leveraging pre-trained models, it achieves effective disease detection, which can aid in early intervention and improve crop management.

## Project Overview

This cotton disease prediction model uses multiple transfer learning approaches to identify diseases in cotton leaves accurately. The model is based on various deep learning architectures, including ResNet152V2, fine-tuned specifically for this classification task. The notebooks provide detailed steps for data preprocessing, model training, and evaluation.

## Features

- **Disease Classification**: Identifies whether a cotton leaf is healthy or affected by disease.
- **Transfer Learning**: Utilizes pre-trained models such as ResNet152V2 for enhanced accuracy.
- **Comprehensive Analysis**: Multiple architectures are tested and compared for the best results.

## Setup and Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook

### Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/cotton-disease-prediction.git
   cd cotton-disease-prediction
   ```

2. **Install Dependencies**:
   Install the required packages using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open Jupyter Notebooks**:
   Start Jupyter Notebook to explore and run each of the transfer learning approaches provided.
   ```bash
   jupyter notebook
   ```

## Project Structure

```plaintext
/cotton-disease-prediction
│
├── Transfer_Learning_ResNet152V2.ipynb  # Notebook for ResNet152V2 model training
├── Transfer Learning Resnet 50.ipynb       # Notebook for second model approach
├── Transfer Learning Inception V3.ipynb       # Notebook for third model approach
└── requirements.txt                     # List of dependencies
```

## Model Details

The project investigates different transfer learning architectures to classify cotton leaf images, including:
- **ResNet152V2**: Fine-tuned for cotton disease prediction.
- **Other Architectures**: Two additional models are explored for comparison.
- **Evaluation**: Each model is evaluated on accuracy and performance to select the optimal model.
