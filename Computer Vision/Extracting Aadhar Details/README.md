# Aadhaar Information Extraction Project

This project focuses on extracting relevant information from Aadhaar card images using Optical Character Recognition (OCR). Two approaches have been implemented:

1. **Tesseract OCR with Pre-processing**: In this approach, the image is converted to greyscale and passed to Tesseract OCR. The text extracted is processed using regular expressions to find the useful information.
2. **EasyOCR**: This approach utilizes the `EasyOCR` library, which supports multi-language OCR. Both **Hindi** and **English** are used to extract the text from Aadhaar cards. The extracted text is then processed using regular expressions to extract key Aadhaar information.

---

## Project Theory and Overview

The purpose of the project is to automate the extraction of critical Aadhaar information such as:

- **First Name**
- **Middle Name**
- **Last Name**
- **Gender**
- **Date of Birth (DOB)**
- **Aadhaar Number**

### Approach 1: Tesseract OCR with Pre-processing

Tesseract OCR is a popular open-source OCR engine but has certain limitations when working with complex documents like Aadhaar cards. To improve accuracy, the following pre-processing techniques are used:

- **Image Pre-processing**: The image is converted to greyscale to enhance text visibility.
- **Text Extraction**: Tesseract extracts the text from the greyscale image.
- **Post-processing**: Regular expressions (`re`) are used to search for relevant patterns in the text, such as names, gender, DOB, and Aadhaar numbers.

#### Limitations of Tesseract Approach:
- **Low Accuracy**: Due to the complex fonts and mixed-language content on Aadhaar cards, Tesseract often struggles with accurate extraction, especially for Hindi text.
- **Reliance on Pre-processing**: Image quality and pre-processing techniques significantly affect Tesseract's output.
  
### Approach 2: EasyOCR with Multi-language Support (English and Hindi)

To overcome the limitations of Tesseract, the second approach uses **EasyOCR**, a more advanced OCR library that supports multiple languages, including both **English** and **Hindi**. This enables better extraction from Aadhaar cards, which typically contain text in both languages.

- **Text Extraction**: EasyOCR reads the Aadhaar card image and extracts text from both Hindi and English regions.
- **Regular Expressions for Information Extraction**: Once the text is extracted, regular expressions are used to identify and extract specific pieces of information, including:
  - First, Middle, and Last Names
  - Gender (Male/Female in both Hindi and English)
  - Date of Birth (DOB)
  - Aadhaar Number in `XXXX XXXX XXXX` format

#### Advantages of EasyOCR Approach:
- **Higher Accuracy**: The combination of Hindi and English support allows for better recognition of names and other details from Aadhaar cards.
- **No Need for Extensive Pre-processing**: EasyOCR works well even with the original image without the need for intense pre-processing steps.

---

## How It Works

1. **Input**: The user provides an image of an Aadhaar card.
2. **Processing**:
   - The image is processed by either Tesseract OCR (with greyscale conversion) or EasyOCR.
   - The extracted text is then scanned using regular expressions to find important details.
3. **Output**: The extracted information is presented in a structured format, including:
   - Name (First, Middle, Last)
   - Gender
   - Date of Birth (DOB)
   - Aadhaar Number

---

## Installation

### 1. Clone the Repository:
`git clone https://UppuluriKalyani/ML-Nexus`


`cd <project-directory>`

### 2. Install Requirements
`pip install -r requirements.txt`
