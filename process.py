import cv2
import imutils
import numpy as np
import time

import pickle
with open("model/mnist_classifier_sgd.pkl", "rb") as f:
    clf = pickle.loads(f.read())

#=====================================================================
image = cv2.imread("sample/image.jpeg")
image = imutils.resize(image, height=600)
orig = image.copy()

x1,y1,x2,y2 = 154, 260, 309, 288

crop = image[y1:y2, x1:x2]

gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#finding contours
cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

segments = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 10:
        segments.append(thresh[y:y+h, x:x+w])
#=====================================================================

final_number = ""
for i, seg in enumerate(segments):
    seg = cv2.resize(seg, (28,28))
    num = clf.predict([seg.flatten()])[0]
    final_number += num



cv2.imshow("crop", crop)
print("[predicted]:", final_number)
cv2.waitKey(0)