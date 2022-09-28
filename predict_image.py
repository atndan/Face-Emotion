import numpy as np
import argparse
import cv2
from keras.models import load_model
import os

model = load_model('model.h5')


# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral",7: "Contempt"}

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    allfaces = []
    rects = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x, w, y, h))
    return rects, allfaces, img


img = cv2.imread(path_image)
rects, faces, image = face_detector(img)
i = 0
for face in faces:
    roi = face.astype("float") / 255.0
    roi = np.expand_dims(cv2.resize(roi, (48, 48)), -1)
    roi = np.expand_dims(roi, axis=0)

    # make a prediction on the ROI, then lookup the class
    preds = model.predict(roi)[0]
    label = emotion_dict[preds.argmax()]

    # Overlay our detected emotion on our pic
    label_position = (rects[i][0] + int((rects[i][1] / 2)), abs(rects[i][2] -5))
    i = + 1
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Emotion Detector", image)
cv2.waitKey(0)

cv2.destroyAllWindows()