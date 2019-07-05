import cv2
import dlib


class HoGFaceDetector:
    """
        Face detection with CNN using dlib
    """
    def __init__(self, frame):
        self.frame = frame

    def detection(self):
        region_of_interest = dlib.get_frontal_face_detector()
        faces = region_of_interest(self.frame, 1)

        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            cv2.rectangle(self.frame,
                          (x, y),
                          (x+w, y+h),
                          (0, 0, 255),
                          2)
        return self.frame
