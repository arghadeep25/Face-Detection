import cv2
import numpy as np


class FaceDetectorCaffe:
    """
        Face detection with trained SSD caffe model
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.prototxt = 'model/model.prototxt'
        self.model = 'model/res10_300x300_ssd_iter_140000.caffemodel'
        self.net = cv2.dnn.readNetFromCaffe(prototxt=self.prototxt,
                                            caffeModel=self.model)

    def detection(self, frame):
        if frame is None:
            return None

        (height, width) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame,
                                     (300, 300)),
                                     1.0,
                                     (300, 300),
                                     (104.0, 177.0, 123.0))
        self.net.setInput(blob)

        detections = self.net.forward()

        faces = []
        confidences = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < self.threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([width,
                                                      height,
                                                      width,
                                                      height])
            (startX, startY, endX, endY) = box.astype('int')

            faces.append([startX, startY, endX, endY])
            confidences.append(confidence)

        return faces, confidences


class FaceDetector:
    """
    """
    def __init__(self):
        self.detector = FaceDetectorCaffe()

    def detection(self, frame):
        face, confidence = self.detector.detection(frame)
        for idx, f in enumerate(face):

            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(frame,
                          (startX, startY),
                          (endX, endY),
                          (0, 255, 0),
                          2)

            text = "{:.2f}%".format(confidence[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write confidence percentage on top of face rectangle
            cv2.putText(frame, text, (startX, Y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        return frame
