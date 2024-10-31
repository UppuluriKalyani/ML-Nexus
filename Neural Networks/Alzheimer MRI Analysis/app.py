import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import pandas as pd

# Define the possible classification categories for Alzheimer's stages
class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

def load_model():
    """
    Load and configure the ResNet50 model for Alzheimer's detection.
    
    Returns:
        torch.nn.Module: Configured ResNet50 model with loaded weights
    """
    # Initialize ResNet50 model without pretrained weights
    model = models.resnet50(pretrained=False)
    
    # Modify the final fully connected layer to match our number of classes
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    
    # Load the trained weights for Alzheimer's detection
    model.load_state_dict(torch.load("alzheimer_model.pth", map_location=torch.device("cpu")))
    
    # Set model to evaluation mode
    model.eval()
    return model

def predict(image, model):
    """
    Process an image and predict the Alzheimer's stage.
    
    Args:
        image (PIL.Image): Input MRI scan image
        model (torch.nn.Module): Trained neural network model
        
    Returns:
        str: Predicted Alzheimer's stage
    """
    # Define image transformations for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to model's expected size
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(           # Normalize using ImageNet statistics
            [0.485, 0.456, 0.406],      # Mean values
            [0.229, 0.224, 0.225]       # Standard deviation values
        )
    ])
    
    # Process the image and make prediction
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():                  # Disable gradient calculation
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return class_names[predicted.item()]

def get_info_and_treatment(label):
    """
    Retrieve information and treatment recommendations for each Alzheimer's stage.
    
    Args:
        label (str): Predicted Alzheimer's stage
        
    Returns:
        tuple: (Information about the stage, Recommended treatments)
    """
    info = {
        "Non Demented": (
            "This stage shows no signs of dementia. However, regular check-ups are recommended.", 
            "No treatment necessary, but a healthy lifestyle can help maintain cognitive health."
        ),
        "Very Mild Demented": (
            "Minor memory issues that may be attributed to aging but might also be early signs of Alzheimer's.", 
            "Monitoring and cognitive exercises may help. Regular check-ups are recommended."
        ),
        "Mild Demented": (
            "Clear signs of cognitive decline, noticeable to family and friends, affecting daily life mildly.", 
            "Medications like cholinesterase inhibitors may help. Cognitive therapy is also recommended."
        ),
        "Moderate Demented": (
            "Significant memory loss, confusion, and assistance required with daily activities.", 
            "Combination of medications and supportive care, including memory aids and possibly full-time care."
        )
    }
    return info.get(label, ("Information not available", "Treatment options not available"))

# Set up the Streamlit interface
st.markdown("<h1 style='text-align: center;'>🧠 AI Powered Alzheimer App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload MRI scans of alzheimer's disease to receive classifications and recommendations.</p>", unsafe_allow_html=True)

# Create file uploader for MRI scans
uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)

if uploaded_files:
    # Load the model once for all predictions
    model = load_model()
    results = []
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Make prediction on the image
        image = Image.open(uploaded_file).convert("RGB")
        label = predict(image, model)
        info, treatment = get_info_and_treatment(label)
        
        # Store results for each image
        results.append({
            "File Name": uploaded_file.name,
            "Prediction": label,
            "Information": info,
            "Treatment": treatment
        })
    
    # Convert results to DataFrame for display and download
    df = pd.DataFrame(results)
        
    # Style the download button with custom CSS
    st.markdown("""
        <style>
            .css-1v3fvcr > div {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .stDownloadButton {
                text-align: center;
                display: inline-block;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Add download button for results
    st.download_button(
        "Download as CSV",
        df.to_csv(index=False).encode('utf-8'),
        "alzheimer_analysis_results.csv",
        "text/csv",
        key='download-csv'
    )
    
    # Style the results table with custom CSS
    st.markdown("""
    <style>
        .dataframe {
            width: 100%;
            font-size: 14px;
            text-align: center;
            margin-top: 20px;
        }
        .dataframe th {
            padding: 10px;
            text-align: center !important;
            font-weight: bold;
        }
        .dataframe td {
            padding: 8px;
            border-bottom: 1px solid #e1e4e8;
            text-align: center !important;
        }
        thead th {
            text-align: center !important;
        }
        .centered-caption {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display results table
    html_table = df.to_html(escape=False, index=False, classes='dataframe')
    st.markdown(html_table, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p class='centered-caption'>&copy; Rakhesh Krishna. All rights reserved.</p>", unsafe_allow_html=True)