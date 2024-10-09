### Solution Description:

The proposed solution integrates **Emotion Detection** and **Pose Detection** using real-time analysis of CCTV footage to identify violent behavior in public or private spaces. This system can be crucial for early intervention in cases of violence, enhancing public safety and protecting vulnerable individuals. The approach leverages **OpenCV** for video processing, **MediaPipe** for pose detection, and **DeepFace** for emotion recognition, forming a robust pipeline for detecting potentially harmful behavior.

**Key Objectives:**
1. **Emotion Detection**: Identifying distressing emotions (anger, fear, sadness) in individuals to detect potential violence.
2. **Pose Detection**: Detecting aggressive postures or body movements like hitting, kicking, or pushing.
3. **Alert System**: Generating automatic alerts for authorities when suspicious behavior is detected.

### Libraries Used:
1. **OpenCV**: For capturing and processing CCTV video streams.
2. **MediaPipe**: To track human poses and movements in real-time, useful for detecting violent gestures.
3. **DeepFace**: To perform real-time emotion recognition, especially focusing on detecting emotions like anger, fear, or distress.

### Workflow:
1. **Video Input**: The system will continuously capture frames from CCTV feeds using OpenCV.
2. **Pose Detection (MediaPipe)**: Each frame will be passed through the MediaPipe pose estimation model to identify human poses and track key body points (e.g., joints, arms, legs).
    - Aggressive poses or abnormal body movements (like raised fists, fast motion, or body slams) will be flagged.
3. **Emotion Detection (DeepFace)**: Concurrently, DeepFace will analyze the facial expressions of individuals in each frame to detect emotions.
    - Focused emotions: anger, fear, and sadness (high-risk indicators of violence).
4. **Action Classification**: Based on pose and emotion data, the system classifies if the detected actions and emotions are likely related to violence.
5. **Alert Generation**: If suspicious activity or emotions are detected, the system triggers an alert, sending notifications or signals to security teams for further inspection.

### Alternatives Considered:
1. **Manual Surveillance**: Traditional security methods rely on human operators monitoring CCTV feeds. However, this is inefficient as humans may miss subtle signs of violence, and it's impractical to monitor all footage 24/7.
2. **Single-Modality Detection**: A system using only emotion detection or only pose detection might miss key indicators. For example, an angry face might not necessarily indicate violence unless paired with aggressive body movements.
3. **Machine Learning-Based Action Recognition**: Using pre-trained action detection models (like those from PyTorch or TensorFlow). However, this would require substantial labeled datasets and may miss emotional cues, making the solution less versatile.

### Approach to be Followed:
1. **Data Collection**: Capture CCTV footage, ensuring diverse environments and actions to improve model training.
2. **Model Training**: 
   - Use a combination of MediaPipe’s built-in pose detection model and DeepFace's emotion classifier.
   - Fine-tune models for detecting aggressive behavior using a custom dataset, if necessary.
3. **Real-Time Monitoring**: Implement real-time video feed analysis, optimizing for computational efficiency, and reducing latency.
4. **Testing & Validation**: Test the system in controlled environments with simulated violent actions to ensure accurate detection.
5. **Deployment**: Deploy the solution in real-world environments, ensuring that privacy and data security standards are upheld.

Here’s a Python program that integrates **Emotion Detection** and **Pose Detection** using OpenCV, MediaPipe, and DeepFace. The program captures video from a CCTV camera or webcam and detects poses and emotions in real-time.

### Required Libraries:
- `opencv-python`: For handling video streams.
- `mediapipe`: For pose detection.
- `deepface`: For emotion detection.

You can install the required libraries using pip:
```bash
pip install opencv-python mediapipe deepface
```

### Key Functions:
1. **`detect_emotion(frame)`**: This function uses DeepFace to analyze the facial expression in the given frame and returns the dominant emotion (e.g., anger, fear).
2. **`detect_pose(frame)`**: This function uses MediaPipe to detect human body poses in the frame and draws the key points on the person. It returns the pose landmarks for further analysis.
3. **`main()`**: The main loop of the program captures video frames, performs emotion and pose detection, and displays the results in real time. Press `q` to exit.

### Instructions to Run:
1. Run the script, and it will start capturing the video from your webcam or a provided CCTV feed URL.
2. The system will analyze the emotions and body poses in each frame.
3. It will overlay the detected dominant emotion on the frame, along with body pose key points.

### Possible Enhancements:
- **Violence Detection**: You can expand the pose detection part to recognize aggressive actions like punches or kicks by classifying specific poses using key points.
- **Alert System**: Integrate a notification or alert system when a violent pose or dangerous emotion (anger) is detected.

This setup creates the basis for a system that can monitor CCTV footage for both emotional cues and violent actions, providing early warnings for potential incidents.


output ![alt text](<sample outputs.png>)