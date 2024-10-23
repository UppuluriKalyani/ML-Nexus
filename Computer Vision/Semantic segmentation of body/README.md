# VITON UNet-based Image Segmentation

This project implements an image segmentation model using a U-Net architecture. The dataset used is VITON, which consists of images and corresponding masks for virtual try-on purposes.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [U-Net Architecture](#u-net-architecture)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Saving Predictions](#saving-predictions)

## Overview
The project leverages a **U-Net** model for segmenting clothing in the VITON dataset, which contains images and segmentation masks. The U-Net model is capable of accurately predicting the clothing masks, which can later be used for virtual try-on systems.

Key components:
- **Image preprocessing** using Albumentations library.
- **U-Net** model for segmentation.
- **Training loop** with optional checkpointing.
- **Dice score** and accuracy to measure the model's performance.

## Dataset
The VITON dataset consists of images and corresponding masks. The dataset is used to train the U-Net model to generate segmentation masks for clothing in an image.


