import os
import sys
import cv2
import time
from cnn_dlib import CNNFaceDetector
from hog_dlib import HoGFaceDetector
from face_cvlib import FaceDetector

video = cv2.VideoCapture(0)
# weight = 'mmod_human_face_detector.dat'
count = 0
timestep0 = time.time()
timestep1 = timestep0
while True:
    timestep1 = time.time()
    ret, frame = video.read()
    face_detector = FaceDetector(frame = frame)
    # face_detector = CNNFaceDetector(frame = frame, weight = weight)
    # face_detector = HoGFaceDetector(frame = frame)
    
    count += 1

    detected_frame = face_detector.detection()
    cv2.imshow('frame', detected_frame)
    if timestep1 >= timestep0 + 1:
        print('FPS: ', count)
        timestep0 = time.time()
        count = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
