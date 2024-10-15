import cv2
import mediapipe as mp
import numpy as np
import json
import socket

# Set up the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 12345)  # Port for Unity to listen to

# Function to read and return 3D landmark positions
def read_landmark_positions_3d(results):
    if results.pose_world_landmarks is None:
        return None
    else:
        # Extract 3D landmark positions
        landmarks = [results.pose_world_landmarks.landmark[lm] for lm in mp.solutions.pose.PoseLandmark]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])

# Function to draw landmarks on the image
def draw_landmarks_on_image(frame, results):
    if results.pose_landmarks is not None:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

# Real-time 3D pose estimation function
def real_time_pose_estimation():
    # Initialize webcam or video
    cap = cv2.VideoCapture(0)  # Use 0 for webcam

    # Initialize Mediapipe Pose model
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (required by Mediapipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to obtain pose landmarks
        results = pose_detector.process(frame_rgb)

        # Draw landmarks on the frame (optional for visualization)
        draw_landmarks_on_image(frame, results)

        # Extract 3D landmarks
        landmark_positions_3d = read_landmark_positions_3d(results)
        if landmark_positions_3d is not None:
            # Send landmark positions to Unity via UDP
            data = json.dumps(landmark_positions_3d.tolist())  # Convert to JSON format
            sock.sendto(data.encode('utf-8'), server_address)  # Send data to Unity
            print(f'3D Landmarks: {landmark_positions_3d}')  # Optional: Print landmarks to console

        # Display the frame with landmarks drawn
        cv2.imshow('Real-Time 3D Pose Estimation', frame)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_pose_estimation()
