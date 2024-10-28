import mediapipe as mp
import cv2
import math
import pyautogui
import time

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def press_gas(hand_landmarks):
    flag = False

    index_finger_tip = hand_landmarks.landmark[8]
    index_finger_tip_x = index_finger_tip.x * w
    index_finger_tip_y = index_finger_tip.y * h

    index_finger_pip = hand_landmarks.landmark[6]
    index_finger_pip_x = index_finger_pip.x * w
    index_finger_pip_y = index_finger_pip.y * h

    middle_finger_tip = hand_landmarks.landmark[12]
    middle_finger_tip_x = middle_finger_tip.x * w
    middle_finger_tip_y = middle_finger_tip.y * h

    middle_finger_pip = hand_landmarks.landmark[10]
    middle_finger_pip_x = middle_finger_pip.x * w
    middle_finger_pip_y = middle_finger_pip.y * h

    index_middle = calculate_distance(index_finger_tip_x, index_finger_tip_y, middle_finger_tip_x, middle_finger_tip_y)

    if index_finger_pip_y > index_finger_tip_y and middle_finger_pip_y > middle_finger_tip_y and index_middle <= 40:
        flag = True

    return flag

def press_enter(hand_landmarks):
    flag = False

    thumb_finger_tip = hand_landmarks.landmark[4]
    thumb_finger_tip_x = thumb_finger_tip.x * w
    thumb_finger_tip_y = thumb_finger_tip.y * h

    index_finger_tip = hand_landmarks.landmark[8]
    index_finger_tip_x = index_finger_tip.x * w
    index_finger_tip_y = index_finger_tip.y * h

    index_thumb = calculate_distance(index_finger_tip_x, index_finger_tip_y, thumb_finger_tip_x, thumb_finger_tip_y)

    if index_thumb <= 20:
        flag = True

    return flag

def straight_index(hand_landmarks):
    flag11 = False
    flag12 = False
    flag2 = False
    flag3 = False
    flag4 = False
    array_finger = []
    for i in range(21):
        arr = []
        finger = hand_landmarks.landmark[i]
        x = finger.x * w
        y = finger.y * h
        arr.append(x)
        arr.append(y)
        array_finger.append(arr)
    if array_finger[8][1] < array_finger[7][1] and array_finger[7][1] < array_finger[6][1]:
        flag11 = True
    if array_finger[8][1] > array_finger[7][1] and array_finger[7][1] > array_finger[6][1]:
        flag12 = True
    if array_finger[12][1] > array_finger[11][1] and array_finger[11][1] > array_finger[10][1]:
        flag2 = True
    if array_finger[16][1] > array_finger[15][1] and array_finger[15][1] > array_finger[14][1]:
        flag3 = True
    if array_finger[20][1] > array_finger[19][1] and array_finger[19][1] > array_finger[18][1]:
        flag4 = True
    return (flag11 or flag12) and flag2 and flag3 and flag4


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75) as hands:

    while cap.isOpened():
        success, image = cap.read()
        h, w, c = image.shape
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        cv2.putText(image, "Player - Vishal Pattar", (160,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,215,0), 4)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

                index_middle = press_gas(hand_landmarks)
                index_thumb = press_enter(hand_landmarks)

                index_finger = straight_index(hand_landmarks)
                index_finger_tip = hand_landmarks.landmark[8]
                index_finger_tip_x = index_finger_tip.x * 1920
                index_finger_tip_y = index_finger_tip.y * 1080

                if handedness.classification[0].label == "Left":
                    if index_middle:
                        pyautogui.keyDown('left')
                        pyautogui.keyUp('right')
                        cv2.putText(image, "Brake", (0,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
                    else:
                        pyautogui.keyUp('left')

                if handedness.classification[0].label == "Right":
                    if index_thumb:
                        pyautogui.keyDown('enter')
                        pyautogui.keyUp('enter')
                        time.sleep(0.05)
                    if index_middle:
                        pyautogui.keyDown('right')
                        pyautogui.keyUp('left')
                        cv2.putText(image, "Gas", (500,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
                    else:
                        pyautogui.keyUp('right')

        cv2.imshow('Hill Climbing Game', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
