import cv2 as cv
import numpy as np
import mediapipe as mp
import joblib
import pyttsx3 as psx
import os
from gtts import gTTS
import time


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


def drawLandmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
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


engine = psx.init()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv.VideoCapture(0)
signify = joblib.load('signify_3-1-temp.h5')
gestures = np.array(['Thankyou'])

# language = 'en'

sequence = []
sentence = []
predictions = []
thres = 0.9

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5) as holistic:
    while True:

        success, img = cap.read()
        img = cv.flip(img, 1)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = holistic.process(imgRGB)
        # print(results)
        drawLandmarks(img, results)

        data = extractResultData(results)
        sequence.append(data)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = signify.predict(np.expand_dims(sequence, axis=0))[0]
            print(gestures[np.argmax(res)])

            if res[np.argmax(res)] > thres:
                if len(sentence) > 0:
                    if gestures[np.argmax(res)] != sentence[-1]:
                        sentence.append(gestures[np.argmax(res)])
                else:
                    sentence.append(gestures[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

        cv.rectangle(img, (0, 0), (640, 40), (245, 117, 16), -1)
        cv.putText(img, ' '.join(sentence), (3, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow('Live', img)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    engine.say(sentence)
    engine.runAndWait()

    myEngine = gTTS(text = sentence,lang=language,slow=False)
    myEngine.save('Sentence.mp3')
    os.system('mpg321 sentence.mp3')

np.save('0',results_data)
print(results_data)
