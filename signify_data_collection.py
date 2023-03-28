import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import time
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("Coordinates")

# define all required actions below

gestures = np.array(['OK'])
noSequences = 50
sequenceLength = 30

for action in gestures:
    for sequence in range(noSequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


def drawLandmarks(image, results):
    # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    #                           mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
    #                           mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    #                           )
    # # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extractResultData(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5) as holistic:
    for action in gestures:
        for sequence in range(noSequences):
            for frame_num in range(sequenceLength):

                isTrue, img = cap.read()
                imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                results = holistic.process(imgRGB)
                drawLandmarks(img, results)
                # print(results)

                if frame_num == 0:
                    cv.putText(img, 'STARTING COLLECTION', (120, 200),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv.LINE_AA)
                    cv.putText(img, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                    cv.imshow('Live', img)
                    cv.waitKey(2000)
                else:
                    cv.putText(img, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

                    cv.imshow('Live', img)

                keypoints = extractResultData(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv.destroyAllWindows()
