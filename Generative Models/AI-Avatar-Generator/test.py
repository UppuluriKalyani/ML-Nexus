import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Display the Image
def load_image(image_path):
    """
    Load an image from file and convert it to RGB format.
    
    Parameters:
    image_path (str): Path to the image file.
    
    Returns:
    np.array: Loaded image in RGB format.
    """
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Apply Smoothing and Sharpening Filters to Enhance Anime Effect
def enhance_image(image):
    """
    Enhance the image by applying bilateral filter, edge enhancement, and contrast adjustment
    to give a clean anime-like effect.
    
    Parameters:
    image (np.array): Input image in RGB format.
    
    Returns:
    np.array: Enhanced image with cartoon/anime effect.
    """
    # Apply bilateral filter for smoothing while retaining edges
    smoothed_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to grayscale and apply adaptive threshold for edge detection
    gray_image = cv2.cvtColor(smoothed_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)

    # Combine the smoothed image with the detected edges
    cartoon_image = cv2.bitwise_and(smoothed_image, smoothed_image, mask=edges)

    # Enhance contrast and brightness for a more vibrant anime-like effect
    cartoon_image = cv2.convertScaleAbs(cartoon_image, alpha=1.3, beta=30)

    return cartoon_image

# Step 3: Detect and Smooth Faces for Cleaner Appearance
def smooth_faces(image):
    """
    Detect faces and apply smoothing to enhance the face region while keeping the rest of the image sharp.
    
    Parameters:
    image (np.array): Input image in RGB format.
    
    Returns:
    np.array: Image with smoothed faces.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Apply Gaussian blur on detected faces for smoother appearance
    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]
        face_region = cv2.GaussianBlur(face_region, (15, 15), 30)  # Apply stronger blur for smoothing
        image[y:y+h, x:x+w] = face_region  # Replace with smoothed face

    return image

# Step 4: Display Original and Enhanced Images Side by Side
def display_images(original, enhanced):
    """
    Display the original and enhanced anime/cartoon images side by side.
    
    Parameters:
    original (np.array): Original image.
    enhanced (np.array): Enhanced anime/cartoon image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display Original Image
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display Enhanced Image
    axes[1].imshow(enhanced)
    axes[1].set_title('Smoothed Anime-like Image')
    axes[1].axis('off')
    
    plt.show()

# Step 5: Generate the Enhanced Anime-Like Avatar with Face Smoothing
def generate_avatar(image_path):
    """
    Generate a simple avatar with an anime-like effect using image enhancement techniques.
    
    Parameters:
    image_path (str): Path to the input image file.
    """
    # Load the original image
    original_image = load_image(image_path)
    
    # Apply enhancements to create the cartoon/anime effect
    enhanced_image = enhance_image(original_image)
    
    # Smoothen the face areas for a cleaner anime effect
    smoothed_image = smooth_faces(enhanced_image)
    
    # Display the original and cartoon images
    display_images(original_image, smoothed_image)

# Usage (replace with your image path)
generate_avatar(r"C:\Users\RAMESWAR BISOYI\Documents\DEV\Open Source\GSSOC\ML-Nexus\Generative Models\AI-Avatar-Generator\elon intro.jpg")
