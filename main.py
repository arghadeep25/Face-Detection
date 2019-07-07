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
    parser.add_argument('-t', '--type',
                        help = 'Input Type',
                        choices = ['image',
                                    'video'],
                        type = str,
                        default = 'video')
    return parser.parse_args()

def main():
    args = cmd_line_parser()

    # Face detection in video
    if args.type == 'video':
        if args.input is 'webcam':
            video = cv2.VideoCapture(0)
            width = int(video.get(3))
            height = int(video.get(4))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter('output.avi',codec, 20.0, (width, height))
        else:
            video = cv2.VideoCapture(args.input)
            width = int(video.get(3))
            height = int(video.get(4))
            codec = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter('output_cnn.avi',codec, 25.0, (width, height))

        count = 0
        timestep0 = time.time()
        timestep1 = timestep0
        while(video.isOpened()):
            timestep1 = time.time()
            ret, frame = video.read()
            if ret == True:
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
                cv2.imshow('Face Detection - Video', detected_frame)
                writer.write(detected_frame)
                if timestep1 >= timestep0 + 1:
                    print('FPS: ', count)
                    timestep0 = time.time()
                    count = 0
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break
        video.release()
        writer.release()
        cv2.destroyAllWindows()

    # Face detection in image
    elif args.type == 'image':
        frame = cv2.imread(args.input)
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
        detected_frame = face_detector.detection()
        cv2.imwrite("img_115_hog.jpg", detected_frame)
        cv2.imshow('Face Detection - Image', detected_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print('Please select a proper input type. For help use -h or --help')

if __name__ == '__main__':
    main()
