# Face Detection
Face detection is one the hot topics in the field of image processing. The objective is to find whether there is a face in an image or not. Various kind of approaches are taken so far to solve this problem. Some of the famous detection algorithms were discussed here.  
## Models
### Haar Cascade   
<img src="https://github.com/arghadeep25/Face-Detection/blob/master/dataset/images/img_115.jpg" width="300">  <img src="https://github.com/arghadeep25/Face-Detection/blob/master/results/images/img_115_haar.jpg" width="300">  
Source: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
### CNN (Convolutional Neura Network)  
<img src="https://github.com/arghadeep25/Face-Detection/blob/master/dataset/images/img_115.jpg" width="300"> <img src="https://github.com/arghadeep25/Face-Detection/blob/master/results/images/img_115_cnn.jpg" width="300">  
Source: https://arxiv.org/pdf/1502.00046.pdf  
### HoG (Histogram of Oriented Gradients)  
<img src="https://github.com/arghadeep25/Face-Detection/blob/master/dataset/images/img_115.jpg" width="300"> <img src="https://github.com/arghadeep25/Face-Detection/blob/master/results/images/img_115_hog.jpg" width="300">  
Source: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf  
### SSD (Single Shot Detector)
<img src="https://github.com/arghadeep25/Face-Detection/blob/master/dataset/images/img_115.jpg" width="300"> <img src="https://github.com/arghadeep25/Face-Detection/blob/master/results/images/img_115_ssd.jpg" width="300">  
Source: https://www.cs.unc.edu/~wliu/papers/ssd.pdf  

## Dependency
- Python 3.5  
- Dlib (http://dlib.net/)  
- Numpy (https://www.numpy.org/)  
- OpenCV (https://opencv.org/)  

### Installation

    pip3 install -r requirements.txt

## Usage  
### User Guide
For help, you can use
    
    python3 main.py -h   
 
For applying different models on images, you can use the following methods:

    python3 main.py -m ssd -t image -i dataset/images/[IMAGE_NAME]
    python3 main.py -m hog -t image -i dataset/images/[IMAGE_NAME]
    python3 main.py -m haar -t image -i dataset/images/[IMAGE_NAME]
    python3 main.py -m cnn -t image -i dataset/images/[IMAGE_NAME]

For applying different models on videos, you can use the following methods:

    python3 main.py -m ssd -t video -i dataset/video/[video_NAME]
    python3 main.py -m hog -t video -i dataset/video/[video_NAME]
    python3 main.py -m haar -t video -i dataset/video/[video_NAME]
    python3 main.py -m cnn -t video -i dataset/video/[video_NAME]

For using webcam:

    python3 main.py -m ssd -t video
    python3 main.py -m hog -t video
    python3 main.py -m haar -t video
    python3 main.py -m cnn -t video


