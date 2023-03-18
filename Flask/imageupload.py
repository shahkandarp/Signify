import cv2 as cv
import numpy as np
import mediapipe as mp
import pickle


# firebase = pyrebase.initialize_app(config)

# storage = firebase.storage()

from flask import *
from tensorflow import keras



app = Flask(__name__)
signify = pickle.load(open(os.path.abspath('signify.pkl'), 'rb'))



def extractResultData(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


@app.route('/', methods=['POST'])
def basic():
  gestures = np.array(['Hello', 'Danger','A'])
  sequence = []
  sentence = []
  thres = 0.9
  
  # obj = {
  #   "res":"Success",
  #   "url":links
  # }

  # return jsonify(obj)

  json = request.get_json()
  print(json)
  mp_holistic = mp.solutions.holistic
  cap = cv.VideoCapture(json['url'])  
  with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5) as holistic:
    while True:
        
        success, img = cap.read()
        if success == False:
            break
        img = cv.flip(img, 1)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = holistic.process(imgRGB)
        data = extractResultData(results)
        sequence.insert(0,data)
        sequence = sequence[:30]

        if len(sequence) == 30:
            res = signify.predict(np.expand_dims(sequence, axis=0))[0]
            print(gestures[np.argmax(res)])
            del sequence[:30]

            if res[np.argmax(res)] > thres:
                if len(sentence) > 0:
                    if gestures[np.argmax(res)] != sentence[-1]:
                        sentence.append(gestures[np.argmax(res)])
                else:
                    sentence.append(gestures[np.argmax(res)])



    cap.release()
    cv.destroyAllWindows()

    obj = {
      'res':'Success',
      'data':sentence
    }
    return jsonify(obj)

@app.route('/abc',methods=['POST'])
def abc():
    print('Hello')
    json = request.get_json()
    print(json)
    data = {
        "res":"Success"
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
