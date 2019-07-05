import cv2
import numpy as np

class FaceDetectorHaar:
    """
    """
    def __init__(self, frame):
        self.frame = frame

    def detection(self):
        if self.frame is None:
            return None

        model = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('Face Detect', cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty('Face Detect',0) >= 0:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces = model.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = self.frame[y:y+h, x:x+w]

            return self.frame
