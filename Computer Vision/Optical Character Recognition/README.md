# Optical Character Recognition (OCR) with OpenCV and Tesseract

This project implements an Optical Character Recognition (OCR) system using OpenCV for image preprocessing and Tesseract for text recognition. Users can upload images, which are then processed to extract text and display the results.

## Code Breakdown:

1. **Image Upload**:
   - Users can upload images directly through Google Colab, enabling easy testing with various files.

2. **Preprocessing Function (`preprocess_image`)**:
   - **Grayscale Conversion**: Converts images to grayscale to simplify processing.
   - **Gaussian Blur**: Reduces noise, improving text detection accuracy.
   - **Adaptive Thresholding**: Binarizes images to enhance text visibility.
   - **Deskewing**: Corrects skewed text orientation for better recognition.

3. **Text Extraction**:
   - Utilizes Tesseract to extract text from preprocessed images via `pytesseract.image_to_string()`.

4. **Text Refinement**:
   - Cleans up extracted text by removing unwanted characters and spaces for improved readability.

5. **Visualization**:
   - Displays processed images and extracted text using Matplotlib (`plt.imshow()`).

6. **Output**:
   - Prints refined text in the console and saves it to a text file.

## Possible Enhancements:
- **Image Variety**: Test with a diverse range of images to evaluate robustness.
- **Custom Configurations**: Adjust Tesseract settings to improve accuracy for specific use cases.
- **Error Handling**: Implement additional error checks for unsupported formats or empty results.
- **Batch Processing**: Enable processing of multiple images in one go.

## Example Use Case:
This OCR system can be used for digitizing printed documents, extracting text from photos (e.g., signs or book pages), or automating data entry by converting image forms into editable text.

## Requirements:
- OpenCV
- Tesseract OCR
- Matplotlib
- Pytesseract

### How to Run:
1. Upload your image file.
2. Run the cell to preprocess the image and extract text.
3. View the results and extracted text in the output.


