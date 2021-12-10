from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.append("./EXTD_Pytorch")

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from EXTD_Pytorch.data.config import cfg
from EXTD_Pytorch.EXTD_64 import build_extd
from EXTD_Pytorch.utils.augmentations import to_chw_bgr
from EXTD_Pytorch.layers.functions.detection import Detect

import tqdm

import warnings
warnings.filterwarnings("ignore")

class module_EXTD :
    def __init__(self,device,in_channel=32):
        self.net = build_extd('test', cfg.NUM_CLASSES,in_channel=in_channel)
        path = "./EXTD_Pytorch/weights/EXTD_"+str(in_channel)+".pth"
        #print("path : " + path)
        self.net.load_state_dict(torch.load(path),strict=False)
        
        self.net.eval()
        self.net = self.net.to(device)

        self.device = device



    def detect(self, img, thresh=0.6):
        x = to_chw_bgr(img)
        x = x.astype('float32')
        x -= cfg.img_mean
        x = x[[2, 1, 0], :, :]

        x = (torch.from_numpy(x).unsqueeze(0))

        x=x.to(self.device)

        with torch.no_grad():
            y = self.net(x)

            detections = y.data
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                img.shape[1], img.shape[0]])

            #img = cv2.imread(img, cv2.IMREAD_COLOR)

            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= thresh:
                    score = detections[0, i, j, 0]
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    left_up, right_bottom = (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3]))
                    j += 1
                    cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
                    conf = "{:.3f}".format(score)
                    point = (int(left_up[0]), int(left_up[1] - 5))
                    # cv2.putText(img, conf, point, cv2.FONT_HERSHEY_COMPLEX,
                    #            0.6, (0, 255, 0), 1)

        return img

if __name__ == "__main__":
    m = module_EXTD('cuda:0')