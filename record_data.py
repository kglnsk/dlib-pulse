#!/usr/bin/env python

import time
import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2
from scipy.fft import fft, ifft

T = 1.0 / 24.0

N = 128

roi_landmarks = [21,70,80,22]

cap = cv2.VideoCapture(0)


predictor_path = 'detector/shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

data = [0]
arr = []
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dets = detector(frame, 0)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for k, d in enumerate(dets):
        face_shapes = predictor(frame, d)
        landmarks = np.matrix([[p.x, p.y] for p in face_shapes.parts()])
        for num in range(face_shapes.num_parts):
            cv2.circle(frame, (face_shapes.parts()[num].x, face_shapes.parts()[num].y), 3, (0,255,0), -1)
    roi_mat = [[pos.x,pos.y] for pos in [face_shapes.parts()[num_point] for num_point in roi_landmarks]]
    mask = cv2.fillPoly(np.zeros(gray_img.shape), [np.array(roi_mat)], 1)
    roi_color = np.sum(gray_img*mask)/np.sum(mask>0)
    data.append(roi_color)
    arr.append(roi_color)
    #print(data)
    if len(data) > N:
         xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
         data = data[-N:]
         yf = fft(data)
         f_pulse = np.argmax(yf)


         
    

    cv2.imshow('frame', frame)
    #cv2.imshow(gray_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        np.savetxt("dataset_a.csv",np.asarray(arr))
        break


cap.release()
cv2.destroyAllWindows()