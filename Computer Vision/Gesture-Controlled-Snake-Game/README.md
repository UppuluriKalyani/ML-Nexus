## Gesture Controlled Snake Game
This program can be used to play a Snake Game (or like it) by detecting hand gestures to control movement of the snake on the screen. It has been implemented in Python using OpenCV library. Although, the demo shown here refers to playing a Snake Game, it can be used to play any game or control any application using hand gestures only.
This program shall be used to play a Snake Game by detecting hand gestures to control movement of the snake on the screen. It has been implemented in Python using OpenCV library. Although, the demo shown here refers to playing a Snake Game, it can be used to play any game or control any application using hand gestures only.
<p align="center">
  <img src="http://img.youtube.com/vi/PE_rgc2K0sg/0.jpg?raw=true" alt="Gesture Controlled Snake Game Using OpenCV and Python"/>
</p>
Watch on Youtube

[Gesture Controlled Snake Game Using OpenCV and Python](http://www.youtube.com/watch?v=PE_rgc2K0sg)

## Getting Started
 ### Prerequisites
 The program depends on the following libraries-
 
    numpy==1.15.2
	imutils==0.5.1
	PyAutoGUI==0.9.38
	opencv_python==3.4.3.18
	pygame==1.9.4

Install the libraries using `pip install -r requirements.txt`

### Installing

 1. Clone the repository in your local computer.
 2. Use `python <filename.py>` to run specific files, which are described below.

## Built With

 - [OpenCV](https://opencv.org/) - The Open Computer Vision Library
 - [PyAutoGUI](https://pypi.org/project/PyAutoGUI/) - Cross platform GUI Automation Python Module

## Contributing
Please feel free to contribute to the project and Pull Requests are open for everyone willing to improve upon the project. Feel free to provide any suggestions.

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/mohitwildbeast/Gesture-Controlled-Snake-Game/blob/master/LICENSE) file for details.
## Acknowledgements

 - The inspiration for the project came through PyImageSearch blog's article- [OpenCV Tracking Object in Images](https://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/) blog post.
 - PyAutoGUI helped a lot for keyboard automation tasks.
## File Contents
The entire project has been made using a bottom up approach, and the project consists of the following files which are described below-
 
 ## [Object Detection](https://github.com/mohitwildbeast/Gesture-Controlled-Snake-Game/blob/master/object-detection.py)

> This script detects a object of specified object colour from the webcam video feed. Using OpenCV library for vision tasks and HSV color space for detecting object of given specific color.  
>See the demo on Youtube - [ Object Detection and Motion Tracking in OpenCV](https://www.youtube.com/watch?v=mtGBuMlusXQ)



 ## [Object Tracking and Direction Detection](https://github.com/mohitwildbeast/Gesture-Controlled-Snake-Game/blob/master/object-tracking-direction-detection.py)

> This script can detect objects specified by the HSV color and also sense
direction of their movement.  

> See the demo on Youtube - [Object Tracking and Direction Detection using OpenCV](https://www.youtube.com/watch?v=zapq9QT9uwc)

 ## [Game Control Using Object Tracking (Multithreaded Implementation)](https://github.com/mohitwildbeast/Gesture-Controlled-Snake-Game/blob/master/game-control-using-object-tracking-multithreaded.py)
> This script can detect objects specified by the HSV color and also sense the
direction of their movement.Using this script a Snake Game which has been loaded in the repo, can be played. Implemented using OpenCV. Uses seperate thread for reading frames through OpenCV.  

>See the demo on Youtube - [Gesture Controlled Snake Game Playing with OpenCV and Computer Vision](https://www.youtube.com/watch?v=PE_rgc2K0sg)

## Snake Game
>The Snake Game present in this video, has been taken from my previous repository [SnakeFun](https://github.com/mohitwildbeast/SnakeFun). The files from the game corrrespond to SnakeFun.py and settingsSnakeFun.py  

>Run the game using the code ```python SnakeFun.py```


