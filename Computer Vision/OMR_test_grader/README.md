# OMR Test Grader
## Introduction
OMR (Optical Mark Recognition) is a technique used to capture human-marked data from documents like multiple-choice exams, surveys, and ballots. An OMR scanner identifies marks made by users (typically filled bubbles or boxes) on a printed form, and then processes that data. This project aims to build an OMR scanner and grader that can read and grade multiple-choice test forms including handling of multiple marked bubbles , Non-filled bubbles, Visualization of correct and incorrect answers and displaying the percentage of correct answers at the bottom of the OMR sheet. An essential feature includes reading the Answer Key from a CSV file , simulating real world competitive examinations which store answers for every question in a seperate file.

## Key Features
 - Thresholding: The threshold for determining whether a bubble is filled is based on the ratio of black pixels to total pixels in the bubble area.
 - Multiple Marks: If multiple bubbles are marked for a single correct-type mcq question, it is considered wrong.
 - Grading: Correct answers are highlighted in green, and incorrect ones in red.
-  Result Calculation: The total score and percentage are displayed on the OMR sheet, making it easier to visualize answers on the spot and evaluation becomes faster.

## Methodology

1. Image Preprocessing:
Image Acquisition: Capture or scan the OMR sheet (scanned or photographed).
Grayscale Conversion: Convert the image to grayscale to simplify further processing.
Thresholding/Binarization: Apply thresholding to convert the grayscale image into a binary image (black and white). This helps in clearly distinguishing the filled marks from the background.
Noise Reduction: Use filters (e.g., Gaussian or median filters) to remove noise from the image.

3. OMR Sheet Alignment:
 - Perspective Transformation: Correct any skew or rotation in the scanned form using perspective transformation.
 - Contour Detection: Find the contours of the OMR form, ensuring to process the area that contains the marks (typically a grid of answer choices).
 - 
3. Bubble/Mark Detection:
 - Grid Division: Divide the detected region of interest (ROI) into sections corresponding to the answer options (i.e., the grid where bubbles are located).
 - Mark Identification: For each section, check whether a bubble is filled. This can be done by:
--Counting the number of black pixels in each bubble region.
--Setting a threshold to decide if a bubble is marked (e.g., more than 60% of the area is filled).
--Handling Multiple Marks: If the user marks more than one option for a question, it can be marked as incorrect based on predefined rules.

4. Answer Key Comparison:
Once the marked bubbles are detected, compare the selected answers with the correct answer key stored in the system. The given code reads the answer key from a CSV file using the pandas library. Each correct answer receives a point, and wrong or multiple answers get no points.

5. Grading and Output:
 - Score Calculation: Calculate the total score by summing up the correct answers.
 - Result: Output the results to the user, which include the number of correct answers, incorrect answers, and the total score.
 - Visualization: Highlight correct and incorrect answers on the image by drawing circles around the marks.
