# Neural Style Transfer using GANs

## Overview
This project demonstrates **Neural Style Transfer** using **Generative Adversarial Networks (GANs)**, where we apply the artistic style of one image onto the content of another image. 

The implementation utilizes a pre-trained **VGG-19** model to extract image features and transfer the style of a reference image onto a content image.

## Project Structure

- **style_transfer_gan.ipynb**: The Jupyter Notebook containing the implementation of style transfer.
- **README.md**: This documentation file.
- **content_image.jpg**: The image whose content will be retained.
- **style_image.jpg**: The image whose artistic style will be transferred.

## Prerequisites

Before running the notebook, ensure you have the following installed:

- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- matplotlib

You can install the necessary packages using the following command:

```bash
pip install torch torchvision pillow matplotlib

How to Run
Clone the repository or download the files.

bash
Copy code
git clone https://github.com/your-username/neural-style-transfer.git
cd neural-style-transfer
Replace path_to_content_image.jpg and path_to_style_image.jpg in the notebook with your images.

Run the Jupyter Notebook style_transfer_gan.ipynb.

After running, you will see the stylized image where the content of the content_image is preserved, and the style of the style_image is applied.

Example
The content image:

The style image:

The final stylized image:

References
PyTorch Documentation
VGG-19 Pre-trained Model
