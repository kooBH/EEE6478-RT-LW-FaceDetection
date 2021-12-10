import cv2
import sys
sys.path.append("./yolo5face")
sys.path.append("./EXTD_Pytorch")
from pytictoc import TicToc
import numpy as np

import torch

from detect_face import YOLO5face_detector
from opencv_haar import opencv_haar
#from module_DSFD import DSFD
from ULFGFD_module import ULFGFD
from module_EXTD import module_EXTD

t = TicToc()
path = "../sample/kbh3.webm"
root_pretrained = "../pretrained/"
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

vid = cv2.VideoCapture(path)

width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) )  # float `width`
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
fps = vid.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

duration = frame_count/fps
vid.release()

print("duration : " + str(duration))
print("fps : " + str(fps))
print("frame_count : " + str(frame_count))
print("width : " + str(width))
print("height : " + str(height))

n_iter = 10

vid = cv2.VideoCapture(path)
ret,frame = vid.read()
# print(frame.shape)
frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
# print(frame.shape)

data = np.expand_dims(frame, 0)
print(data.shape)

while(vid.isOpened()):
    ret,frame = vid.read()
    if ret :
        frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
        frame = np.expand_dims(frame, 0)
        data = np.concatenate((data,frame), axis=0)
    else :
        break
vid.release()

print(data.shape)


for d in ["cpu","cuda:0"] : 
    print("device : " + d)

    # EXTD 
    detector = module_EXTD(d,64)
    t.tic()
    s = t.tocvalue()
    for j in range(n_iter) : 
        for i in range(data.shape[0]) : 
            frame = detector.detect(data[i,...])
    e = t.tocvalue()
    time = (e-s)/n_iter
    print("EXTD64 : " + str(time) +  " sec ")

    # YOLOv5m
    name_model = "yolov5m-face.pt"
    detector = YOLO5face_detector(root_pretrained+name_model,d)
    t.tic()
    s = t.tocvalue()
    for j in range(n_iter) : 
        for i in range(data.shape[0]) : 
            frame = detector.detect(data[i,...])
    e = t.tocvalue()
    time = (e-s)/n_iter
    print(" YOLOv5m : " + str(time) +  " sec ")

    # YOLOv5s
    name_model = "yolov5s-face.pt"
    detector = YOLO5face_detector(root_pretrained+name_model,d)
    t.tic()
    s = t.tocvalue()
    for j in range(n_iter) : 
        for i in range(data.shape[0]) : 
            frame = detector.detect(data[i,...])
    e = t.tocvalue()
    time = (e-s)/n_iter
    print(" YOLOv5s : " + str(time) +  " sec ")

    # YOLOv5m
    name_model = "yolov5n-face.pt"
    detector = YOLO5face_detector(root_pretrained+name_model,d)
    t.tic()
    s = t.tocvalue()
    for j in range(n_iter) : 
        for i in range(data.shape[0]) : 
            frame = detector.detect(data[i,...])
    e = t.tocvalue()
    time = (e-s)/n_iter
    print(" YOLOv5m : " + str(time) +  " sec ")
    
    # YOLOv5n-0.5
    name_model = "yolov5n-0.5.pt"
    detector = YOLO5face_detector(root_pretrained+name_model,d)
    t.tic()
    s = t.tocvalue()
    for j in range(n_iter) : 
        for i in range(data.shape[0]) : 
            frame = detector.detect(data[i,...])
    e = t.tocvalue()
    time = (e-s)/n_iter
    print(" YOLOv5n-0.5 : " + str(time) +  " sec ")

    if d == "cpu" : 
        # Haar
        haar = opencv_haar()
        t.tic()
        s = t.tocvalue()
        for j in range(n_iter) : 
            for i in range(data.shape[0]) : 
                frame = haar.detect(data[i,...])
        e = t.tocvalue()
        time = (e-s)/n_iter
        print(" haar : " + str(time) +  " sec ")

