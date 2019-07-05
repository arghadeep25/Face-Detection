import cv2
import time
import argparse
from detectors.cnn_dlib import CNNFaceDetector
from detectors.hog_dlib import HoGFaceDetector
from detectors.haarcascade import FaceDetectorHaar
from detectors.ssd_face_detector import FaceDetector

def cmd_line_parser():
    parser = argparse.ArgumentParser(description='Face Detection')
    parser.add_argument('-m', '--method',
                        help='Algorithms',
                        choices=['haar',
                                  'hog',
                                  'cnn',
                                  'ssd'],
                        type = str,
                        default = None
                        )
    parser.add_argument('-i', '--input',
                        help='Input',
                        type=str,
                        default = 'webcam')
    return parser.parse_args()

def main():
    args = cmd_line_parser()

    if args.input is 'webcam':
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(args.input)

    count = 0
    timestep0 = time.time()
    timestep1 = timestep0
    while True:
        timestep1 = time.time()
        ret, frame = video.read()
        if args.method == 'haar':
            face_detector = FaceDetectorHaar(frame=frame)
        elif args.method == 'hog':
            face_detector = HoGFaceDetector(frame=frame)
        elif args.method == 'cnn':
            face_detector = CNNFaceDetector(frame=frame)
        elif args.method == 'ssd':
            face_detector = FaceDetector(frame=frame)
        else:
            print('Face detection scheme not selected...')
            return

        count += 1

        detected_frame = face_detector.detection()
        cv2.imshow('frame', detected_frame)
        if timestep1 >= timestep0 + 1:
            print('FPS: ', count)
            timestep0 = time.time()
            count = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
