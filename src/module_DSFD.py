import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__))+'/DSFD')
#print(sys.path)

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms

from face_ssd import build_ssd
# def build_ssd(phase, size=640, num_classes=2) : 


class DSFD :
    def __init__(self):
        self.net = build_ssd('test')
        self.net.load_state_dict(torch.load('../pretrained/WIDERFace_DSFD_RES152.pth'))
        self.net.cuda()
        self.net.eval()

    def detect(self,image):
        x = image

        width = x.shape[1]
        height = x.shape[0]
        x = x.astype(np.float32)
        # ? 
        x -= np.array([104, 117, 123],dtype=np.float32)

        x = torch.from_numpy(x).permute(2, 0, 1)
        x = x.unsqueeze(0)
        #x = Variable(x.cuda(), volatile=True)

        #net.priorbox = PriorBoxLayer(width,height)
        y = self.net(x.cuda())
        detections = y.data
        scale = torch.Tensor([width, height, width, height])

        boxes=[]
        scores = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.01:
                score = detections[0,i,j,0]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                boxes.append([pt[0],pt[1],pt[2],pt[3]])
                scores.append(score)
                j += 1
                if j >= detections.size(2):
                    break

        det_conf = np.array(scores)
        boxes = np.array(boxes)

        if boxes.shape[0] == 0:
            return np.array([[0,0,0,0,0.001]])

        det_xmin = boxes[:,0] / shrink
        det_ymin = boxes[:,1] / shrink
        det_xmax = boxes[:,2] / shrink
        det_ymax = boxes[:,3] / shrink
        det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

        keep_index = np.where(det[:, 4] >= 0)[0]
        det = det[keep_index, :]
        return det