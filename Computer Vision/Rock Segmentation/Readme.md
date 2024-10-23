# Lunar Rock Segmentation with UNet and VGG16

This project performs segmentation on artificial lunar rocky landscapes. The objective is to detect and segment rocks from rendered lunar images using a modified UNet architecture with a VGG16 encoder backbone.

## Dataset

The dataset used is the **Artificial Lunar Rocky Landscape Dataset** available on Kaggle:

- **Download Link**: [Artificial Lunar Rocky Landscape Dataset](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset)
- **Dataset Size**: 5.02 GB
- **License**: CC-BY-NC-SA-4.0

## Data Generator

The project utilizes a custom data generator function to efficiently load and preprocess the dataset. The generator reads images from the source directory (`/content/images/render/`) and the corresponding ground truth segmentation masks from the mask directory (`/content/images/ground/`). The key features of the generator include:

- **Image Resizing**: Both the input images and the masks are resized to a uniform size of 512x512 pixels.
- **Normalization**: The pixel values of the images and masks are normalized to the range [0, 1] to ensure consistency and speed up the training process.
- **Batching**: The generator loads the images in batches, allowing for memory-efficient training on large datasets.
- **Augmentation**: Data augmentation techniques such as random flips, rotations, and shifts can be applied to increase dataset variability and prevent overfitting.

## Model Architecture

This project employs a UNet-based architecture with a VGG16 encoder. Key components:

- **VGG16 Encoder**: Pretrained VGG16 model from ImageNet, with layers from the first 4 blocks frozen, while the last block (block5) remains trainable.
- **Decoder**: Uses transpose convolution layers for upsampling, combined with intermediate outputs from the encoder.
- **Skip Connections**: Added between the VGG16 layers and the upsampling decoder layers to retain spatial information.

### Key Model Layers

- **Conv2DTranspose**: Used for upsampling feature maps.
- **LeakyReLU**: As the activation function to introduce non-linearity.
- **BatchNormalization**: To stabilize and accelerate training.
- **Concatenate**: For connecting the corresponding encoder and decoder layers.

## ModelCheckpoint

To ensure the best version of the model is saved during training, we use a `ModelCheckpoint` callback. This saves the model whenever there is an improvement in validation performance, specifically in terms of the Dice Coefficient or Intersection over Union (IoU) metrics.

### Key Features of Model Checkpointing:
- **Best Model Saving**: The model weights are saved only when there is an improvement in the validation IoU or Dice Coefficient, ensuring that the best-performing model is preserved.
- **Filepath Structure**: Models are saved with a structured filename that includes the epoch number and validation score for easy reference.
- **Monitoring**: The callback monitors the validation IoU or Dice Coefficient to decide when to save the model.
