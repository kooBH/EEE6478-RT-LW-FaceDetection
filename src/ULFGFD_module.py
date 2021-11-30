"""
This code is used to batch detect images in a folder.
"""
import argparse
import os
import sys

import cv2

sys.path.append("./ULFGFD/")

from ULFGFD.vision.ssd.config.fd_config import define_img_size
from ULFGFD.vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from ULFGFD.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

class ULFGFD():
    def __init__(self,model='RFB',device='cuda:0',candidate_size = 200, threshold=0.6):
        self.device = device
        self.candidate_size = candidate_size
        self.threshold = threshold
        self.model = model

        label_path = "ULFGFD/models/voc-model-labels.txt"

        class_names = [name.strip() for name in open(label_path).readlines()]
        if model == 'slim':
            #model_path = "ULFGFD/models/pretrained/version-slim-320.pth"
            model_path = "ULFGFD/models/pretrained/version-slim-640.pth"
            self.net = create_mb_tiny_fd(len(class_names), is_test=True, device=device)
            self.predictor = create_mb_tiny_fd_predictor(self.net, candidate_size=candidate_size, device=device)
        elif model == 'RFB':
            #model_path = "ULFGFD/models/pretrained/version-RFB-320.pth"
            model_path = "ULFGFD/models/pretrained/version-RFB-640.pth"
            self.net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
            self.predictor = create_Mb_Tiny_RFB_fd_predictor(self.net, candidate_size=candidate_size, device=device)
        else:
            print("The net type is wrong!")
            sys.exit(1)
        self.net.load(model_path)


    def detect(self,image):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.predictor.predict(image, self.candidate_size / 2, self.threshold)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        return image 