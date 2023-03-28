from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os
import joblib

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

gestures = np.array(['Thankyou'])

label_map = {label: num for num, label in enumerate(gestures)}
# print(label_map)

DATA_PATH = os.path.join("Coordinates")
no_sequences = 30
sequence_length = 30

sequences, labels = [], []
for action in gestures:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

print(np.array(sequences).shape)
# print(np.array(labels).shape)

X = np.array(sequences)
# print(X.shape)

y = to_categorical(labels).astype(int)
# print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(y_test.shape)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(gestures.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=1200, callbacks=[tb_callback])

# model.summary()

# print(X_test)

res = model.predict(X_test)

# print(res)

print(gestures[np.argmax(res[0])])
print(gestures[np.argmax(y_test[0])])


filename = 'signify_3-1-temp.h5'

joblib.dump(model,filename)

