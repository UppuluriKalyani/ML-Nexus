'''This script can detect objects specified by the HSV color and also sense the
direction of their movement.Using this script a Snake Game which has been loaed
in the repo, can be played. Implemented using OpenCV.'''

#Import necessary modules
import cv2
import imutils
import numpy as np
from collections import deque
import time
import pyautogui

#Define HSV colour range for green colour objects
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

#Used in deque structure to store no. of given buffer points
buffer = 20

#Used so that pyautogui doesn't click the center of the screen at every frame
flag = 0

#Points deque structure storing 'buffer' no. of object coordinates
pts = deque(maxlen = buffer)
#Counts the minimum no. of frames to be detected where direction change occurs
counter = 0
#Change in direction is stored in dX, dY
(dX, dY) = (0, 0)
#Variable to store direction string
direction = ''
#Last pressed variable to detect which key was pressed by pyautogui
last_pressed = ''

#Start video capture
video_capture = cv2.VideoCapture(0)

#Sleep for 2 seconds to let camera initialize properly.
time.sleep(2)

#Use pyautogui function to detect width and height of the screen
width,height = pyautogui.size()


#Loop until OpenCV window is not closed
while True:

    '''game_window = pyautogui.locateOnScreen(r'images\SnakeGameWelcomeScreen.png')
    game_window_center = pyautogui.center(game_window)
    pyautogui.click(game_window_center)'''


    #Store the readed frame in frame, ret defines return value
    ret, frame = video_capture.read()
    #Flip the frame to avoid mirroring effect
    frame = cv2.flip(frame,1)
    #Resize the given frame to a 600*600 window
    frame = imutils.resize(frame, width = 600)
    #Blur the frame using Gaussian Filter of kernel size 11, to remove excessivve noise
    blurred_frame = cv2.GaussianBlur(frame, (11,11), 0)
    #Convert the frame to HSV, as HSV allow better segmentation.
    hsv_converted_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    #Create a mask for the frame, showing green values
    mask = cv2.inRange(hsv_converted_frame, greenLower, greenUpper)
    #Erode the masked output to delete small white dots present in the masked image
    mask = cv2.erode(mask, None, iterations = 2)
    #Dilate the resultant image to restore our target
    mask = cv2.dilate(mask, None, iterations = 2)

    #Find all contours in the masked image
    _,cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Define center of the ball to be detected as None
    center = None

    #If any object is detected, then only proceed
    if(len(cnts) > 0):
        #Find the contour with maximum area
        c = max(cnts, key = cv2.contourArea)
        #Find the center of the circle, and its radius of the largest detected contour.
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        #Calculate the centroid of the ball, as we need to draw a circle around it.
        M = cv2.moments(c)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        #Proceed only if a ball of considerable size is detected
        if radius > 10:
            #Draw circles around the object as well as its centre
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
            cv2.circle(frame, center, 5, (0,255,255), -1)
            #Append the detected object in the frame to pts deque structure
            pts.appendleft(center)

    #Using numpy arange function for better performance. Loop till all detected points
    for i in np.arange(1, len(pts)):
        #If no points are detected, move on.
        if(pts[i-1] == None or pts[i] == None):
            continue

        #If atleast 10 frames have direction change, proceed
        if counter >= 10 and i == 1 and pts[-10] is not None:
            #Calculate the distance between the current frame and 10th frame before
            dX = pts[-10][0] - pts[i][0]
            dY = pts[-10][1] - pts[i][1]
            (dirX, dirY) = ('', '')

            #If distance is greater than 50 pixels, considerable direction change has occured.
            if np.abs(dX) > 50:
                dirX = 'West' if np.sign(dX) == 1 else 'East'

            if np.abs(dY) > 50:
                dirY = 'North' if np.sign(dY) == 1 else 'South'

            #Set direction variable to the detected direction
            direction = dirX if dirX != '' else dirY

        #Draw a trailing red line to depict motion of the object.
        thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    #If deteced direction is East, press right button
    if direction == 'East':
        if last_pressed != 'right':
            pyautogui.press('right')
            last_pressed = 'right'
            print("Right Pressed")
            #pyautogui.PAUSE = 2
    #If deteced direction is West, press Left button
    elif direction == 'West':
        if last_pressed != 'left':
            pyautogui.press('left')
            last_pressed = 'left'
            print("Left Pressed")
            #pyautogui.PAUSE = 2
    #if detected direction is North, press Up key
    elif direction == 'North':
        if last_pressed != 'up':
            last_pressed = 'up'
            pyautogui.press('up')
            print("Up Pressed")
            #pyautogui.PAUSE = 2
    #If detected direction is South, press down key
    elif direction == 'South':
        if last_pressed != 'down':
            pyautogui.press('down')
            last_pressed = 'down'
            print("Down Pressed")
            #pyautogui.PAUSE = 2


    #Write the detected direction on the frame.
    cv2.putText(frame, direction, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    #Show the output frame.
    cv2.imshow('Game Control Window', frame)
    key = cv2.waitKey(1) & 0xFF
    #Update counter as the direction change has been detected.
    counter += 1

    #If pyautogui has not clicked on center, click it once to focus on game window.
    if (flag == 0):
        #Click on the centre of the screen, game window should be placed here.
        pyautogui.click(int(width/2), int(height/2))
        flag = 1

    #If q is pressed, close the window
    if(key == ord('q')):
        break
#After all the processing, release webcam and destroy all windows
video_capture.release()
cv2.destroyAllWindows()
