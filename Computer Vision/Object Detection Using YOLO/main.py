import cv2
import numpy as np

# Load YOLO model
#Yolo weights and cfg are to be downloaded from the official website of YOLO Algorithm
net = cv2.dnn.readNet("C:\\Users\\billa\\OneDrive\\Desktop\\Programs\\ML_DL\\yolov3.weights", 
                      "C:\\Users\\billa\\OneDrive\\Desktop\\Programs\\ML_DL\\yolov3.cfg")

# Load the input image
img = cv2.imread("C:\\Users\\billa\\OneDrive\\Desktop\\Programs\\ML_DL\\YO_img.jpg")

# Prepare the image for the network (resize, normalize)
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Get the layer names from the YOLO model
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform the forward pass to get output from the output layers
layer_outputs = net.forward(output_layers)

# Load COCO class labels
with open("C:\\Users\\billa\\OneDrive\\Desktop\\Programs\\ML_DL\\coco (1).names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get image height and width
height, width, channels = img.shape

# Lists to hold detected class IDs, confidence scores, and bounding boxes
class_ids = []
confidences = []
boxes = []

# Process each output layer
for output in layer_outputs:
    for detection in output:
        # Extract the scores, class ID, and confidence of the prediction
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filter out weak predictions by ensuring confidence is greater than a threshold
        if confidence > 0.5:
            # Get bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the top-left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Add the bounding box coordinates, confidence score, and class ID to the lists
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression to remove overlapping bounding boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw the bounding boxes and labels on the image
if len(indices) > 0:
    for i in indices.flatten():  # Flatten the indices list
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Draw a bounding box rectangle and put the label text
        color = (0, 255, 0)  # Green color for the box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Show the final output image
import matplotlib.pyplot as plt

# Convert the image from BGR (OpenCV format) to RGB (for displaying correctly in matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(img_rgb)
plt.axis('off')  # Hide axis
plt.show()
