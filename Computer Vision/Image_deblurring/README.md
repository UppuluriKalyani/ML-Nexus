# DeblurCNN: Image Deblurring with Convolutional Neural Networks

This project implements a deep Convolutional Neural Network (CNN) for image deblurring using PyTorch.

## Features

- Deep 15-layer CNN architecture optimized for image deblurring
- GPU acceleration with CUDA support
- Efficient data handling for large datasets
- Comprehensive training pipeline
- Visualization tools for performance monitoring
- Model persistence for easy saving and loading
- Separate evaluation framework
- Scalable design for various dataset sizes
- Dedicated testing script for new images

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- tqdm

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/DeblurCNN.git
   cd DeblurCNN
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model:

1. Prepare your dataset in the following structure:

   ```
   DBlur/
   └── CelebA/
       ├── train/
       │   ├── blur/
       │   └── sharp/
       ├── validation/
       │   ├── blur/
       │   └── sharp/
       └── test/
           ├── blur/
           └── sharp/
   ```

2. Run the training script:
   ```
   python train.py
   ```

The script will train the model and save it as `deblur_cnn_model.pth`.

### Testing

To test the model on a single image:

1. Ensure you have a trained model (`deblur_cnn_model.pth`) in the project directory.

2. Update the `test_image_path` in `test.py` to point to your test image.

3. Run the test script:
   ```
   python test.py
   ```

The script will process the image and display the results.

## Model Architecture

The DeblurCNN consists of 15 convolutional layers with batch normalization and ReLU activations. The architecture is designed to effectively handle the complex task of image deblurring.

## Results

After training, the model's performance can be visualized using the generated loss curves (`loss_curves.png`) and sample deblurred images (`deblur_result_*.png`).
