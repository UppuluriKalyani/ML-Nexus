import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import torch
import time
import os

# Define COCO dataset classes
# These classes represent the objects that the YOLO model can detect
# The index position corresponds to the class ID returned by the model
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Configure Streamlit page settings
st.set_page_config(initial_sidebar_state='expanded')

# Load and apply custom CSS styling
with open('main.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def detect_objects(image, model, classes, conf):
    """
    Perform object detection on an input image using the YOLO model.
    
    Args:
        image: Input image array
        model: Loaded YOLO model instance
        classes: List of class indices to detect
        conf: Confidence threshold for detections
    
    Returns:
        tuple: (annotated_image, detected_class_indices)
            - annotated_image: Image with detection boxes drawn
            - detected_class_indices: List of class indices for detected objects
    """
    results = model(image, conf=conf, classes=classes)
    annotated_image = results[0].plot()
    return annotated_image, results[0].boxes.cls.tolist()

def is_cuda_available():
    """
    Check if CUDA (GPU acceleration) is available.
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()

def main():
    """
    Main application function that sets up the Streamlit interface and handles
    the object detection pipeline for both webcam and uploaded video inputs.
    """
    # Initialize session state variables for metrics tracking
    if 'frame_rate' not in st.session_state:
        st.session_state.frame_rate = 0
    if 'tracked_objects' not in st.session_state:
        st.session_state.tracked_objects = 0
    if 'detected_classes' not in st.session_state:
        st.session_state.detected_classes = 0

    # Set up main page title and divider
    st.markdown("<h1 style='text-align: center;'>Real Time Object Counter", unsafe_allow_html=True)
    st.markdown("---")
    
    # Configure sidebar settings and styling
    st.sidebar.title("⚙️ Settings")
    st.markdown("""
                <style>
                .stButton > button {
                    width: 100%;
                }
                </style>""", 
                unsafe_allow_html=True)
    
    # Check and display CUDA availability
    cuda_available = is_cuda_available()
    
    # Create settings container in sidebar
    st.sidebar.markdown('<div class="settings-container">', unsafe_allow_html=True)
    
    # Add user configuration options
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.1)
    st.sidebar.markdown("---")
    enable_gpu = st.sidebar.checkbox("🤖 Enable GPU", value=False, disabled=not cuda_available)    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Add input source selection options
    use_webcam = st.sidebar.button("Use Webcam")
    selected_classes = st.sidebar.multiselect("Select Classes", CLASSES, default=['person', 'car'])
    
    # Display CUDA availability warning if needed
    if not cuda_available:
        st.sidebar.warning("CUDA is not available. GPU acceleration is disabled.")
        st.sidebar.info("To enable GPU acceleration, make sure you have a CUDA-capable GPU and PyTorch is installed with CUDA support.")
    
    # Add video upload option
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    # Add credits
    st.sidebar.markdown('''
    Created with ❤️ by [Rakhesh Krishna](https://github.com/rakheshkrishna2005/)
    ''')
    
    # Convert selected class names to their corresponding indices
    class_indices = [CLASSES.index(cls) for cls in selected_classes]

    # Initialize YOLO model with appropriate device (GPU/CPU)
    model = YOLO('yolov8n.pt')
    if enable_gpu and cuda_available:
        model.to('cuda')
        st.sidebar.success("GPU enabled successfully!")
    else:
        model.to('cpu')
        st.sidebar.info("Using CPU for processing.")
    
    # Create video display container
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # Create metrics display layout
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    tracked_objects_metric = kpi_col1.empty()
    frame_rate_metric = kpi_col2.empty()
    classes_metric = kpi_col3.empty()

    # Initialize metrics with default values
    tracked_objects_metric.metric("Tracked Objects", "0")
    frame_rate_metric.metric("Frame Rate", "0.00 FPS")
    classes_metric.metric("Classes", "0")

    # Create placeholder for object count table
    object_count_placeholder = st.empty()

    # Add CSS styling for the detection results table
    st.markdown("""
    <style>
    .detected-object-table {
        width: 100%;
        border-collapse: collapse;
        text-align: center;
    }
    .detected-object-table th, .detected-object-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .detected-object-table th {
        background-color: var(--background-color);
    }
    .detected-object-table tr:nth-child(even) {
        background-color: var(--background-color);
    }
    </style>
    """, unsafe_allow_html=True)

    # Store processed frames for potential future use
    processed_frames = []

    # Handle webcam input
    if use_webcam:
        cap = cv2.VideoCapture(0)
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            # Process and display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, detected_classes = detect_objects(frame, model, class_indices, conf_threshold)
            video_placeholder.image(annotated_frame, channels="RGB")
            processed_frames.append(annotated_frame)

            # Update performance metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update metrics every second
                frame_rate = frame_count / elapsed_time
                unique_classes = len(np.unique(detected_classes))
                tracked_objects_metric.metric("Tracked Objects", str(len(detected_classes)))
                frame_rate_metric.metric("Frame Rate", f"{frame_rate:.2f} FPS")
                classes_metric.metric("Classes", str(unique_classes))
                frame_count = 0
                start_time = time.time()

            # Update and display object count table
            unique_classes, counts = np.unique(detected_classes, return_counts=True)
            object_data = [{"Class": CLASSES[int(cls)], "Count": count} for cls, count in zip(unique_classes, counts)]
            object_count_placeholder.markdown(
                "<table class='detected-object-table'>" +
                "<tr><th>Class</th><th>Count</th></tr>" +
                "".join([f"<tr><td>{item['Class']}</td><td>{item['Count']}</td></tr>" for item in object_data]) +
                "</table>",
                unsafe_allow_html=True
            )

            if not use_webcam:
                break

        cap.release()

    # Handle uploaded video input
    elif uploaded_video is not None:
        # Create temporary file for video processing
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        vf = cv2.VideoCapture(tfile.name)
        frame_count = 0
        start_time = time.time()

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            # Process and display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, detected_classes = detect_objects(frame, model, class_indices, conf_threshold)
            video_placeholder.image(annotated_frame, channels="RGB")
            processed_frames.append(annotated_frame)

            # Update performance metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 0.5:  # Update metrics every half second
                frame_rate = frame_count / elapsed_time
                unique_classes = len(np.unique(detected_classes))
                tracked_objects_metric.metric("Tracked Objects", str(len(detected_classes)))
                frame_rate_metric.metric("Frame Rate", f"{frame_rate:.2f} FPS")
                classes_metric.metric("Classes", str(unique_classes))
                frame_count = 0
                start_time = time.time()

            # Update and display object count table
            unique_classes, counts = np.unique(detected_classes, return_counts=True)
            object_data = [{"Class": CLASSES[int(cls)], "Count": count} for cls, count in zip(unique_classes, counts)]
            object_count_placeholder.markdown(
                "<table class='detected-object-table'>" +
                "<tr><th>Class</th><th>Count</th></tr>" +
                "".join([f"<tr><td>{item['Class']}</td><td>{item['Count']}</td></tr>" for item in object_data]) +
                "</table>",
                unsafe_allow_html=True
            )

        vf.release()

    # Update final metrics display
    st.markdown(f"""
    <script>
        document.getElementById('tracked-objects').innerText = "{st.session_state.tracked_objects}";
        document.getElementById('frame-rate').innerText = "{st.session_state.frame_rate:.2f} FPS";
        document.getElementById('detected-classes').innerText = "{st.session_state.detected_classes}";
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()