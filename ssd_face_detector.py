import os
import cv2
import cvlib
import numpy as np

class FaceDetectorCaffe:
    """
        Face detection with CNN using dlib
    """
    def __init__(self, frame, threshold = 0.5):
        self.frame = frame
        self.threshold = threshold

    def detection(self):
        if self.frame is None:
            return None

        prototxt = 'model/model.prototxt'
        model = 'model/res10_300x300_ssd_iter_140000.caffemodel'

        net = cv2.dnn.readNetFromCaffe(prototxt=prototxt, caffeModel=model)
        (height, width) = self.frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(self.frame,
                                    (300, 300)),
                                    1.0,
                                    (300,300),
                                    (104.0,177.0,123.0))
        net.setInput(blob)

        detections = net.forward()

        faces = []
        confidences = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0,0,i,2]

            if confidence < self.threshold:
                continue

            box = detections[0,0,i,3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype('int')

            faces.append([startX, startY, endX, endY])
            confidences.append(confidence)

        return faces, confidences

class FaceDetector:
    """
    """
    def __init__(self, frame):
        self.frame = frame

    def detection(self):
        detector = FaceDetectorCaffe(self.frame)
        face, confidence = detector.detection()
        for idx, f in enumerate(face):

            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(self.frame, (startX,startY), (endX,endY), (0,255,0), 2)

            text = "{:.2f}%".format(confidence[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write confidence percentage on top of face rectangle
            cv2.putText(self.frame, text, (startX,Y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0), 2)

        return self.frame


    # def detection(self):
    #     # region_of_interest = dlib.get_frontal_face_detector()
    #     face, confidence = cvlib.detect_face(self.frame)
    #
    #     # for face, confidence in zip(faces, confidences):
    #     #     (startX, startY) = face[0], face[1]
    #     #     (endX, endY) = face[2], face[3]
    #     #
    #     #     cv2.rectangle(self.frame, (startX,startY), (endX, endY), (0,0,255), 2)
    #     for idx, f in enumerate(face):
    #
    #         (startX, startY) = f[0], f[1]
    #         (endX, endY) = f[2], f[3]
    #
    #         # draw rectangle over face
    #         cv2.rectangle(self.frame, (startX,startY), (endX,endY), (0,255,0), 2)
    #
    #         text = "{:.2f}%".format(confidence[idx] * 100)
    #
    #         Y = startY - 10 if startY - 10 > 10 else startY + 10
    #
    #         # write confidence percentage on top of face rectangle
    #         cv2.putText(self.frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
    #                     (0,255,0), 2)
    #     return self.frame
