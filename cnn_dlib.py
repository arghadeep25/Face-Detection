import cv2
import numpy as np
import time
import dlib


class CNNFaceDetector:
    """
        Face detection with CNN using dlib
    """
    def __init__(self, frame, weight):
        self.frame = frame
        self.weight = weight

    def detection(self):
        region_of_interest = dlib.cnn_face_detection_model_v1(self.weight)
        faces = region_of_interest(self.frame, 1)

        for face in faces:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y

            cv2.rectangle(self.frame, (x,y), (x+w, y+h), (0,0,255), 2)
        return self.frame
