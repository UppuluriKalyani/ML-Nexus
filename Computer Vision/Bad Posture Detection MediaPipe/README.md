
This GitHub repository contains a posture detection program that utilizes [YOLOv5](https://github.com/ultralytics/yolov5),And mediaPipe an advanced object detection algorithm, to detect and predict lateral sitting postures. The program is designed to analyze the user's sitting posture in real-time and provide feedback on whether the posture is good or bad based on predefined criteria. The goal of this project is to promote healthy sitting habits and prevent potential health issues associated with poor posture.

Key Features:

* YOLOv5: The program leverages the power of YOLOv5, which is an object detection algorithm, to
  accurately detect the user's sitting posture from a webcam.
* Real-time Posture Detection: The program provides real-time feedback on the user's sitting posture, making it suitable
  for use in applications such as office ergonomics, fitness, and health monitoring.
* Good vs. Bad Posture Classification: The program uses a pre-trained model to classify the detected posture as good or
  bad, enabling users to improve their posture and prevent potential health issues associated with poor sitting habits.
* Open-source: The program is released under an open-source license, allowing users to access the source code, modify
  it, and contribute to the project.

### Built With

![Python]

# Getting Started

### Prerequisites

* Python 3.9.x



### Run the program

`python application.py <optional: model_file.pt>` **OR** `python3 application.py <optional: model_file.pt>`

The default model is loaded if no model file is specified.



*Fig. 1: YOLOv5s network architecture (based on Liu et al.). The CBS module consists of a Convolutional layer, a Batch Normalization layer, and a Sigmoid Linear Unit (SiLU) activation function. The C3 module consists of three CBS modules and one bottleneck block. The SPPF module consists of two CBS modules and three Max Pooling layers.*

## Model Results
The validation set contains 80 images (40 sitting_good, 40 sitting_bad). The results are as follows:
|Class|Images|Instances|Precision|Recall|mAP50|mAP50-95|
|--|--|--|--|--|--|--|
|all| 80 | 80 | 0.87 | 0.939 | 0.931 | 0.734 |
|sitting_good| 40 |  40| 0.884 | 0.954 | 0.908 |0.744  |
|sitting_bad| 80 | 40 | 0.855 | 0.925 | 0.953 | 0.724 |

F1, Precision, Recall, and Precision-Recall plots:

<p align="middle">
<img src="https://raw.githubusercontent.com/itakurah/SittingPostureDetection/main/data/images/F1_curve.png" width=40% height=40%>
<img src="https://raw.githubusercontent.com/itakurah/SittingPostureDetection/main/data/images/P_curve.png" width=40% height=40%>
<img src="https://raw.githubusercontent.com/itakurah/SittingPostureDetection/main/data/images/R_curve.png" width=40% height=40%>
<img src="https://raw.githubusercontent.com/itakurah/SittingPostureDetection/main/data/images/PR_curve.png" width=40% height=40%>
</p>

