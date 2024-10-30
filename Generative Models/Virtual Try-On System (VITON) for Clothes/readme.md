VITON (Virtual Try-On Network) is a deep learning-based virtual try-on system that allows users to see how clothes would look on them in a 2D image. Without using 3D models, the system overlays an image of clothing onto a user's photo, generating a realistic composite that maintains the proportions, texture, and draping of the clothing as it would appear in real life.

## Features

- **User Upload**: Upload a photo of yourself in a neutral pose.
- **Clothing Selection**: Choose a clothing image with a transparent background.
- **Realistic Composite**: The system will generate an image of the user wearing the selected clothing.
- **Two-Step Process**: The system first generates a coarse composite image, then refines it for improved realism.

## Requirements

### Dependencies

Ensure you have the following dependencies installed:

- Python 3.8 or later
- PyTorch (1.9.0 or later)
- Torchvision (0.10.0 or later)
- Pillow
- NumPy

Install dependencies using `pip`:

```bash
pip install torch torchvision pillow numpy
```

### Dataset

- You can use the [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) dataset or your own custom images.
- Ensure that user images are preprocessed to a standard size of **256x192** pixels and that clothing images have transparent backgrounds.

## Installation

Download or prepare your dataset, placing the user images and clothing images in the appropriate folders.

## Running the Project

1. **Preprocess the Data**:
   Ensure that the user and clothing images are of the correct size (256x192 pixels) and clothing images have transparent backgrounds.

2. **Train the Model** (Optional):
   If you are training the model from scratch, you can run the training script:
   ```bash
   python train.py
   ```

3. **Use Pre-Trained Model**:
   If you are using a pre-trained model, download the model weights and place them in the appropriate directory (`/models`).

4. **Run the Virtual Try-On System**:
   To generate virtual try-on outputs, run the following command:
   ```bash
   python main.py --user_image /path/to/user/image.jpg --clothing_image /path/to/clothing/image.png
   ```

5. **View Results**:
   The generated image will be saved in the `outputs/` directory.

## Directory Structure

```bash
.
├── data/
│   ├── user_images/          # User photos for virtual try-on
│   ├── clothing_images/      # Clothing images with transparent backgrounds
├── models/
│   ├── viton_model.pth       # Pre-trained model weights
├── outputs/                  # Directory where results are saved
├── README.md                 # Project documentation
```

## Sample Input

1. **User Image**:
   - A person standing in front of a neutral background.
   - Example size: 256x192 pixels.

2. **Clothing Image**:
   - A front-facing clothing image with a transparent background.
   - Example size: 256x192 pixels.

## Output

The system will generate an image of the user wearing the selected clothing, with realistic alignment and texture rendering.

## Troubleshooting

### AttributeError: 'torch' has no attribute 'version'
- Ensure that PyTorch and torchvision are properly installed.
- Check your installation by running:
  ```python
  import torch
  import torchvision
  print(torch.__version__)
  print(torchvision.__version__)
  ```

### Other Issues
- Verify that your input images are the correct size and format.
- Ensure you are using Python 3.8 or higher and that all required libraries are installed.
