# Blur Detection Using Laplacian

## Description
This project focuses on detecting blur in images using the Laplacian operator, a widely used technique in image processing. Blur detection is crucial in various applications such as photography, computer vision, and image-based quality control. Ensuring that images are sharp is especially important in fields like medical imaging, industrial inspection, and surveillance, where image quality significantly affects the accuracy of downstream processes.

The Laplacian method is efficient, requiring minimal computation while effectively detecting blur in images. This project implements a simple yet robust approach to quantify image blur using the variance of the Laplacian operator, making it an ideal choice for real-time applications or as part of a pre-processing pipeline for image-based tasks.

## Task Overview
- **Image Preprocessing**: Convert input images to grayscale as the first step in processing.

- **Laplacian Operator**: Apply the Laplacian operator to compute the second-order derivatives of the image. This helps in detecting areas of rapid intensity change, which are crucial for sharpness.

- **Variance of Laplacian**: Calculate the variance of the Laplacian of the image to quantify the blur. A low variance indicates a blurry image, while a high variance suggests the image is sharp.

- **Thresholding**: Set a threshold for the variance of the Laplacian to classify images as "blurry" or "sharp."

- **Visualization**: Display the original image along with the calculated variance to provide a visual understanding of the blur level.

## Results
![Screenshot 2024-10-12 121351](https://github.com/user-attachments/assets/08b3e04f-fa82-4bf3-b4fe-25648eab00fd)
![Screenshot 2024-10-12 121408](https://github.com/user-attachments/assets/97ec07de-dcbc-4545-a68a-c80f376d25f4)



## Features
- Efficient blur detection using the Laplacian operator, making it suitable for real-time applications.
- Easy-to-implement algorithm with minimal dependencies.
- Provides a numerical measure (variance) for image sharpness.
- Useful as a pre-processing step in computer vision tasks or for quality control in photography and industrial applications.

## Requirements
- Python 3.x
- OpenCV (cv2)
- numpy
- matplotlib

## Conclusion
This project provides a simple, efficient solution for detecting blur in images using the Laplacian operator. By calculating the variance of the Laplacian, this method quantifies the sharpness of an image and can be easily integrated into larger systems for image quality assessment. Its fast performance makes it suitable for real-time applications in photography, industrial inspection, and computer vision.
